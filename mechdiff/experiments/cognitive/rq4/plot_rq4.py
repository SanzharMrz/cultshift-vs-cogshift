#!/usr/bin/env python3
import os, re, glob, json, shutil
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_mask_from_fname(path):
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


def _find_full_kl_raw(art_dir, layer=30, hook="attn_out"):
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    full_path = None
    for p in files:
        if os.path.basename(p).endswith("_ALL.json"):
            full_path = p
            break
    if full_path is not None:
        d = _read_json(full_path)
        kr = float(d.get("KL_raw_mean", 0.0) or 0.0)
        return kr
    # fallback: max KL_raw across files
    kr_max = 0.0
    for p in files:
        d = _read_json(p)
        kr = d.get("KL_raw_mean")
        if isinstance(kr, (int, float)):
            kr_max = max(kr_max, float(kr))
    return kr_max


def _collect_single_head_rows(art_dir, layer=30, hook="attn_out"):
    denom = _find_full_kl_raw(art_dir, layer, hook)
    if denom <= 0:
        raise RuntimeError("No usable FULL raw denominator (ALL missing and no KL_raw>0)")
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    rows = []
    for p in files:
        mask = _parse_mask_from_fname(p)
        # only single-head files
        if not (isinstance(mask, list) and len(mask) == 1):
            continue
        d = _read_json(p)
        km = d.get("KL_mapped_mean")
        if not isinstance(km, (int, float)):
            continue
        cov = 100.0 * float(km) / denom
        rows.append({
            "file": os.path.basename(p),
            "head": mask[0],
            "KL_sel": float(km),
            "coverage_pct": cov,
            "reduction": d.get("reduction"),
        })
    return rows, denom


def plot_best_coverage_by_k(art_dir="mechdiff/artifacts/cognitive/rq4", layer=30, hook="attn_out",
                            savepath=None, title=None):
    denom = _find_full_kl_raw(art_dir, layer, hook)
    pat = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(pat))
    runs = []
    for p in files:
        mask = _parse_mask_from_fname(p)
        if mask in ("ALL", []) or not isinstance(mask, list):
            continue
        d = _read_json(p)
        km = d.get("KL_mapped_mean")
        if not isinstance(km, (int, float)):
            continue
        cov = 100.0 * float(km) / max(1e-12, denom)
        runs.append({"k": len(mask), "coverage_pct": cov})
    by_k = {}
    for r in runs:
        by_k.setdefault(r["k"], []).append(r)
    ks = sorted(k for k in by_k.keys() if k >= 1)
    best_cov = [max(by_k[k], key=lambda x: x["coverage_pct"]) ["coverage_pct"] for k in ks]
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


def plot_single_head_coverage(art_dir="mechdiff/artifacts/cognitive/rq4", layer=30, hook="attn_out",
                              top_n=10, savepath=None, title=None):
    rows, _ = _collect_single_head_rows(art_dir, layer, hook)
    rows.sort(key=lambda r: r["coverage_pct"], reverse=True)
    tops = rows[:top_n]
    labels = [str(r["head"]) for r in tops]
    vals = [r["coverage_pct"] for r in tops]
    plt.figure(figsize=(max(6, 0.6 * len(labels) + 2), 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("single head id")
    plt.ylabel("coverage of FULL raw effect (%)")
    ttl = title or f"L{layer}/{hook}: single-head coverage (top {len(labels)})"
    plt.title(ttl)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()


def plot_single_head_coverage_by_head(art_dir="mechdiff/artifacts/cognitive/rq4", layer=30, hook="attn_out",
                                      savepath=None, title=None):
    rows, _ = _collect_single_head_rows(art_dir, layer, hook)
    rows.sort(key=lambda r: r["head"])  # by head id
    heads = [str(r["head"]) for r in rows]
    covs = [r["coverage_pct"] for r in rows]
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(max(12, 0.7 * len(heads) + 2), 6))
    ax = plt.gca()
    ax.bar(list(range(len(covs))), covs, color="#4e79a7")
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


def plot_single_head_delta_by_head(art_dir="mechdiff/artifacts/cognitive/rq4", layer=30, hook="attn_out",
                                   savepath=None, title=None):
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
        items.append((mask[0], float(delta)))
    items.sort(key=lambda x: x[0])
    heads = [str(h) for h, _ in items]
    deltas = [v for _, v in items]
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(max(14, 0.8 * len(heads) + 3), 7))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_dir", default="mechdiff/artifacts/cognitive/rq4")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--hook", default="attn_out")
    ap.add_argument("--mode", default="all",
                    choices=[
                        "all",
                        "coverage_by_k",
                        "single_top",
                        "by_head",
                        "delta_by_head",
                        "single_details",
                    ])
    ap.add_argument("--out", default=None)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--csv_out", default=None)
    args = ap.parse_args()

    if args.mode == "coverage_by_k":
        plot_best_coverage_by_k(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                                 savepath=args.out)
        return
    if args.mode == "single_top":
        plot_single_head_coverage(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                                  top_n=args.top_n, savepath=args.out)
        return
    if args.mode == "by_head":
        plot_single_head_coverage_by_head(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                                          savepath=args.out)
        return
    if args.mode == "delta_by_head":
        plot_single_head_delta_by_head(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                                       savepath=args.out)
        return
    if args.mode == "single_details":
        rows, _ = _collect_single_head_rows(args.art_dir, args.layer, args.hook)
        labels = [str(r["head"]) for r in rows]
        covs = [r["coverage_pct"] for r in rows]
        plt.figure(figsize=(max(6, 0.6 * len(labels) + 2), 4))
        plt.bar(range(len(covs)), covs)
        plt.xticks(range(len(labels)), labels)
        plt.xlabel("single head id")
        plt.ylabel("coverage of FULL raw effect (%)")
        plt.title(f"L{args.layer}/{args.hook}: single-head details (coverage)")
        plt.tight_layout()
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            plt.savefig(args.out, dpi=200)
        else:
            plt.show()
        if args.csv_out:
            os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
            with open(args.csv_out, "w", encoding="utf-8") as f:
                f.write("head,KL_sel,coverage_pct\n")
                for r in rows:
                    f.write(f"{r['head']},{r['KL_sel']:.6f},{r['coverage_pct']:.2f}\n")
        return

    # mode == all: write everything into fig subfolder under art_dir
    fig_dir = os.path.join(args.art_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    plot_best_coverage_by_k(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                             savepath=os.path.join(fig_dir, "coverage_by_k.png"))
    plot_single_head_coverage(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                              top_n=args.top_n, savepath=os.path.join(fig_dir, "single_head_top.png"))
    cov_by_head_path = os.path.join(fig_dir, "single_head_by_head.png")
    plot_single_head_coverage_by_head(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                                      savepath=cov_by_head_path)
    compat_cov_path = os.path.join(fig_dir, "single_head_coverage_by_head.png")
    try:
        shutil.copyfile(cov_by_head_path, compat_cov_path)
    except Exception:
        pass
    plot_single_head_delta_by_head(art_dir=args.art_dir, layer=args.layer, hook=args.hook,
                                   savepath=os.path.join(fig_dir, "single_head_delta_by_head.png"))
    # Also dump full single-head table
    rows, _ = _collect_single_head_rows(args.art_dir, args.layer, args.hook)
    with open(os.path.join(fig_dir, "single_head_details.csv"), "w", encoding="utf-8") as f:
        f.write("head,KL_sel,coverage_pct\n")
        for r in sorted(rows, key=lambda x: x["head"]):
            f.write(f"{r['head']},{r['KL_sel']:.6f},{r['coverage_pct']:.2f}\n")


if __name__ == "__main__":
    main()

