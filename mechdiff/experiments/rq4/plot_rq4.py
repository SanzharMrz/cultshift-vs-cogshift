#!/usr/bin/env python3
import os, re, glob, json
import matplotlib.pyplot as plt
import seaborn as sns


def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_mask_from_fname(path):
    """Return list of head indices from filename; 'ALL' -> 'ALL'; 'NONE' -> []."""
    name = os.path.basename(path)
    m = re.search(r"head_mask_L\d+_[A-Za-z0-9]+_(.+)\.json$", name)
    if not m:
        return []
    token = m.group(1)
    up = token.upper()
    if up == "ALL":
        return "ALL"
    if up == "NONE":
        return []
    heads = []
    for t in token.split("-"):
        t = t.strip()
        if t == "":
            continue
        try:
            heads.append(int(t))
        except Exception:
            pass
    return sorted(heads)


def _collect_headmask_runs(art_dir, layer=26, hook="attn_out"):
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No head_mask files found at {pat}")
    runs = []
    kr_all = None
    for p in files:
        mask = _parse_mask_from_fname(p)
        d = _read_json(p)
        kr = d.get("KL_raw_mean")
        if mask == "ALL":
            kr_all = kr
            break
    if kr_all is None or kr_all == 0:
        # Fallback: use the maximum KL_raw among all head-mask runs as denominator
        kr_all = 0.0
        for p in files:
            d = _read_json(p)
            kr = d.get("KL_raw_mean")
            if isinstance(kr, (int, float)) and kr is not None:
                kr_all = max(kr_all, float(kr))
        if kr_all == 0.0:
            raise RuntimeError("No usable denominator: ALL missing and no head_mask file has KL_raw_mean > 0")
    for p in files:
        mask = _parse_mask_from_fname(p)
        if mask == "ALL":
            continue
        d = _read_json(p)
        kr = d.get("KL_raw_mean")
        if kr is None:
            continue
        k = len(mask) if isinstance(mask, list) else -1
        cov = 100.0 * (kr / kr_all)
        runs.append({
            "file": os.path.basename(p),
            "mask": mask,
            "k": k,
            "KL_raw": kr,
            "coverage_pct": cov,
        })
    return runs


def _find_denominator(art_dir, layer=26, hook="attn_out"):
    """Return (denom, label) where denom = Δ_full if available else KL_raw(full)."""
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    full_path = None
    for p in files:
        if os.path.basename(p).endswith("_ALL.json"):
            full_path = p
            break
    if full_path is None:
        # fallback: max KL_raw across files
        kr_max = 0.0
        for p in files:
            d = _read_json(p)
            kr = d.get("KL_raw_mean")
            if isinstance(kr, (int, float)):
                kr_max = max(kr_max, float(kr))
        return (kr_max, "KL_raw(full_fallback=max)")
    d = _read_json(full_path)
    kr = float(d.get("KL_raw_mean", 0.0) or 0.0)
    km = float(d.get("KL_mapped_mean", 0.0) or 0.0)
    delta_full = kr - km
    if abs(delta_full) >= 1e-12:
        return (delta_full, "Δ_full")
    return (kr, "KL_raw(full)")


def plot_best_coverage_by_k(art_dir="mechdiff/artifacts/rq4", layer=26, hook="attn_out",
                            savepath=None, title=None):
    """
    Line plot: BEST coverage% vs k (how many heads you need to match FULL raw effect).
    """
    runs = _collect_headmask_runs(art_dir, layer, hook)
    by_k = {}
    for r in runs:
        if r["k"] < 0:
            continue
        by_k.setdefault(r["k"], []).append(r)
    ks = sorted(k for k in by_k.keys() if k >= 0)
    best_cov = [max(by_k[k], key=lambda x: x["coverage_pct"])["coverage_pct"] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, best_cov, marker="o")
    plt.axhline(100.0, linestyle="--")
    plt.xlabel("k heads")
    plt.ylabel("coverage of FULL raw effect (%)")
    ttl = title or f"L{layer}/{hook}: best coverage by k"
    plt.title(ttl)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()


def plot_single_head_coverage(art_dir="mechdiff/artifacts/rq4", layer=26, hook="attn_out",
                              top_n=10, savepath=None, title=None):
    """
    Bar chart: coverage% for single heads (k=1), sorted descending.
    """
    runs = [r for r in _collect_headmask_runs(art_dir, layer, hook) if r["k"] == 1]
    runs.sort(key=lambda x: x["coverage_pct"], reverse=True)
    tops = runs[:top_n]
    labels = []
    vals = []
    for r in tops:
        h = r["mask"][0] if isinstance(r["mask"], list) and r["mask"] else r["mask"]
        labels.append(str(h))
        vals.append(r["coverage_pct"])
    plt.figure(figsize=(max(6, 0.6 * len(labels) + 2), 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("single head id")
    plt.ylabel("coverage of FULL raw effect (%)")
    ttl = title or f"L{layer}/{hook}: single-head coverage (top {top_n})"
    plt.title(ttl)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()


def plot_single_head_details(art_dir="mechdiff/artifacts/rq4", layer=26, hook="attn_out",
                             savepath=None, title=None, csv_out=None):
    """
    Horizontal bar chart and optional CSV showing (layer, head, Δ, coverage%).
    Uses raw coverage denominator (KL_raw(ALL) or fallback max KL_raw).
    """
    runs = [r for r in _collect_headmask_runs(art_dir, layer, hook) if r["k"] == 1]
    # Derive head id, keep Δ via KL_raw difference is not stored here; use KL_raw as proxy and coverage already computed
    # We also load each JSON to get reduction if present
    enriched = []
    for r in runs:
        fname = r["file"]
        d = _read_json(os.path.join(art_dir, fname))
        delta = d.get("reduction")
        head_id = r["mask"][0] if isinstance(r["mask"], list) and r["mask"] else None
        enriched.append({
            "layer": layer,
            "head": head_id,
            "delta": float(delta) if isinstance(delta, (int, float)) else float("nan"),
            "coverage": r["coverage_pct"],
        })
    # Sort by coverage desc
    enriched.sort(key=lambda x: (x["coverage"] if x["coverage"] == x["coverage"] else -1e9), reverse=True)
    # Prepare plot
    labels = [str(e["head"]) for e in enriched]
    covs = [e["coverage"] for e in enriched]
    deltas = [e["delta"] for e in enriched]
    plt.figure(figsize=(max(6, 0.5 * len(labels) + 2), 0.4 * max(6, len(labels))))
    y = list(range(len(labels)))
    plt.barh(y, covs, color="#4e79a7")
    for yi, c, d in zip(y, covs, deltas):
        plt.text(c + 1, yi, f"cov={c:.1f}%  Δ={d:.3f}", va="center", fontsize=8)
    plt.yticks(y, labels)
    plt.xlabel("coverage of FULL raw effect (%)")
    ttl = title or f"L{layer}/{hook}: single-head coverage and Δ"
    plt.title(ttl)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()
    # Optional CSV
    if csv_out:
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        with open(csv_out, "w", encoding="utf-8") as f:
            f.write("layer,head,delta,coverage_pct\n")
            for e in enriched:
                f.write(f"{e['layer']},{e['head']},{e['delta']:.6f},{e['coverage']:.2f}\n")


def plot_single_head_coverage_by_head(art_dir="mechdiff/artifacts/rq4", layer=26, hook="attn_out",
                                      savepath=None, title=None):
    """Bar chart: coverage% vs head id (sorted by head id), using summarizer-compatible denominator."""
    denom, _ = _find_denominator(art_dir, layer, hook)
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    items = []
    for p in files:
        mask = _parse_mask_from_fname(p)
        if not (isinstance(mask, list) and len(mask) == 1):
            continue
        d = _read_json(p)
        delta = d.get("reduction")
        if not isinstance(delta, (int, float)):
            continue
        head_id = mask[0]
        cov = (float(delta) / denom * 100.0) if denom else float('nan')
        items.append((head_id, cov))
    items.sort(key=lambda x: x[0])
    heads = [str(h) for h, _ in items]
    covs = [c for _, c in items]
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(max(12, 0.7 * len(heads) + 2), 6))
    ax = sns.barplot(x=list(range(len(covs))), y=covs, palette=sns.color_palette("Blues", n_colors=max(3, len(covs))))
    ax.set_xticks(list(range(len(heads))))
    ax.set_xticklabels(heads)
    ax.set_xlabel("head id")
    ax.set_ylabel("coverage of FULL raw effect (%)")
    ttl = title or f"L{layer}/{hook}: coverage by head (k=1)"
    ax.set_title(ttl)
    ax.axhline(100.0, ls="--", lw=1.2, color="#888", alpha=0.8)
    try:
        ymax = max([c for c in covs if c == c] + [100.0]) * 1.1
        ax.set_ylim(0, ymax)
    except Exception:
        pass
    sns.despine()
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()


def plot_single_head_delta_by_head(art_dir="mechdiff/artifacts/rq4", layer=26, hook="attn_out",
                                   savepath=None, title=None):
    """Bar chart: Δ (reduction) vs head id (sorted by head id)."""
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    items = []
    for p in files:
        mask = _parse_mask_from_fname(p)
        if not (isinstance(mask, list) and len(mask) == 1):
            continue
        d = _read_json(p)
        delta = d.get("reduction")
        if not isinstance(delta, (int, float)):
            continue
        head_id = mask[0]
        items.append((head_id, float(delta)))
    items.sort(key=lambda x: x[0])
    heads = [str(h) for h, _ in items]
    deltas = [v for _, v in items]
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(max(14, 0.8 * len(heads) + 3), 7))
    # Use distinct colors per bar
    colors = sns.color_palette("viridis", n_colors=max(3, len(deltas)))
    ax = plt.gca()
    ax.bar(list(range(len(deltas))), deltas, color=colors, label="Δ (KL_raw - KL_masked)")
    ax.set_xticks(list(range(len(heads))))
    ax.set_xticklabels(heads)
    ax.set_xlabel("head id")
    ax.set_ylabel("Δ (KL_raw - KL_masked)")
    ttl = title or f"L{layer}/{hook}: Δ by head (k=1)"
    ax.set_title(ttl)
    ax.axhline(0.0, ls=":", lw=1.0, color="#666", alpha=0.8)
    try:
        ymin = min([0.0] + deltas) * 1.1
        ymax = max([0.0] + deltas) * 1.1
        ax.set_ylim(ymin, ymax)
    except Exception:
        pass
    sns.despine()
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    outdir = "mechdiff/artifacts/rq4/fig"
    plot_single_head_coverage_by_head(savepath=os.path.join(outdir, "single_head_coverage_by_head.png"))
    plot_single_head_delta_by_head(savepath=os.path.join(outdir, "single_head_delta_by_head.png"))


