#!/usr/bin/env python3
"""
Plots for Cognitive RQ3 (alpha sweep, layer/hook comparisons).

Inputs:
- Primary CSV: mechdiff/artifacts/cognitive/rq3/rq3_combined.csv
- Fallback JSON scan: mechdiff/artifacts/cognitive/rq3/mapped_patch_L*_*alpha*.json

Outputs:
- Figures under mechdiff/artifacts/cognitive/rq3/fig/
"""

import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# --------- config & styling (cognitive paths)
IN_DIR  = "mechdiff/artifacts/cognitive/rq3"
_DEFAULT_OUT = "mechdiff/artifacts/cognitive/rq3/fig"
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default=os.environ.get("COG_RQ3_OUT_DIR", _DEFAULT_OUT))
_ARGS = parser.parse_args()
OUT_DIR = _ARGS.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

if _HAS_SNS:
    sns.set_theme(style="whitegrid", context="talk")
    LINE_PALETTE = sns.color_palette("colorblind")
    BAR_PALETTE  = sns.color_palette("crest")
    HM_CMAP      = "viridis"
else:
    plt.style.use("seaborn-v0_8-whitegrid")
    LINE_PALETTE = plt.cm.tab10.colors  # type: ignore
    BAR_PALETTE  = plt.cm.Blues(np.linspace(0.4, 0.9, 6))  # type: ignore
    HM_CMAP      = "viridis"

# --------- load results (CSV first, then fallback JSON scan)
csv_path_primary = os.path.join(IN_DIR, "rq3_combined.csv")
if os.path.exists(csv_path_primary):
    df = pd.read_csv(csv_path_primary)
    required = {"layer","hook","alpha","KL_raw_mean","KL_mapped_mean"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Missing required columns in {csv_path_primary}")
    if "delta" not in df.columns:
        df["delta"] = df["KL_raw_mean"] - df["KL_mapped_mean"]
    if "drop_pct" not in df.columns:
        kr = df["KL_raw_mean"].replace(0, np.nan)
        df["drop_pct"] = 100.0 * (df["KL_raw_mean"] - df["KL_mapped_mean"]) / kr
    df["layer"] = df["layer"].astype(int)
    df["hook"] = df["hook"].astype(str)
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
else:
    rows = []
    file_pats = [
        os.path.join(IN_DIR, "mapped_patch_L*_alpha*.json"),
        os.path.join(IN_DIR, "mapped_patch_L*_*alpha*.json"),
    ]
    files = []
    for pat in file_pats:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    fname_re = re.compile(r".*mapped_patch_L(?P<layer>\d+)_(?P<hook>[A-Za-z0-9_]+)_alpha(?P<alpha>[0-9.]+).*\.json$")
    def safe_float(x, default=np.nan):
        try:
            return float(x)
        except Exception:
            return default
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        layer = d.get("layer")
        hook  = d.get("hook", "unknown")
        alpha = d.get("alpha", None)
        m = fname_re.match(fp)
        if m is not None:
            layer = int(m.group("layer"))
            hook  = m.group("hook")
            alpha = safe_float(m.group("alpha"), alpha)
        KR = d.get("KL_raw_mean")
        KM = d.get("KL_mapped_mean")
        if KR is None or KM is None:
            continue
        delta = KR - KM
        drop_pct = 100.0 * delta / max(1e-9, KR)
        rows.append({
            "file": os.path.basename(fp),
            "path": fp,
            "layer": int(layer) if layer is not None else -1,
            "hook": str(hook),
            "alpha": safe_float(alpha),
            "KL_raw_mean": float(KR),
            "KL_mapped_mean": float(KM),
            "delta": float(delta),
            "drop_pct": float(drop_pct),
        })
    if not rows:
        raise SystemExit("No cognitive RQ3 results found. Ensure rq3_combined.csv or mapped_patch_L*_*.json files exist in artifacts/cognitive/rq3.")
    df = pd.DataFrame(rows)

df = df.sort_values(["layer","hook","alpha"]).reset_index(drop=True)

# Optional: write a per-plot CSV of what we actually plotted
csv_path = os.path.join(OUT_DIR, "rq3_alpha_sweep_summary_plotted.csv")
df.to_csv(csv_path, index=False)
print(f"Wrote: {csv_path}")

def savefig_tight(path):
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()

# 1) For each (layer,hook): alpha vs drop% (line+markers)
for (L, H), g in df.groupby(["layer","hook"], sort=True):
    gg = g.dropna(subset=["alpha"]).sort_values("alpha")
    if gg.empty:
        continue
    fig = plt.figure(figsize=(7.0, 4.5))
    if _HAS_SNS:
        ax = sns.lineplot(
            data=gg,
            x="alpha",
            y="drop_pct",
            marker="o",
            linewidth=2.2,
            markersize=8,
            color=LINE_PALETTE[0],
        )
    else:
        ax = plt.gca()
        ax.plot(gg["alpha"].values, gg["drop_pct"].values, marker="o", linewidth=2.2)
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("α (scale)")
    ax.set_ylabel("ΔKL drop (%)")
    ax.set_title(f"RQ3 (cognitive): ΔKL vs α — L{L} / {H}")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=False, nbins=6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    out = os.path.join(OUT_DIR, f"rq3_drop_vs_alpha_L{L}_{H}.png")
    savefig_tight(out)

# 2) For each layer: best drop per hook (bar)
for L, g in df.groupby("layer", sort=True):
    best = g.loc[g.groupby("hook")["drop_pct"].idxmax()].sort_values("drop_pct", ascending=False)
    fig = plt.figure(figsize=(7.0, 4.5))
    if _HAS_SNS:
        ax = sns.barplot(data=best, x="hook", y="drop_pct", palette=BAR_PALETTE)
    else:
        ax = plt.gca()
        x = np.arange(len(best))
        ax.bar(x, best["drop_pct"].values, color=BAR_PALETTE if isinstance(BAR_PALETTE, list) else None)
        ax.set_xticks(x)
        ax.set_xticklabels(best["hook"].tolist(), rotation=20, ha="right")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width()/2, height),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 4), textcoords="offset points")
    ax.set_ylabel("Best ΔKL drop (%)")
    ax.set_xlabel("hook")
    ax.set_title(f"RQ3 (cognitive): Best ΔKL per hook — L{L}")
    out = os.path.join(OUT_DIR, f"rq3_best_drop_per_hook_L{L}.png")
    savefig_tight(out)

# 3) Heatmap per layer: hooks × alpha → drop%
for L, g in df.groupby("layer", sort=True):
    gg = g.dropna(subset=["alpha"]).copy()
    if gg.empty:
        continue
    piv = gg.pivot_table(index="hook", columns="alpha", values="drop_pct", aggfunc="max")
    piv = piv.reindex(sorted(piv.index), axis=0)
    piv = piv.reindex(sorted(piv.columns), axis=1)
    fig = plt.figure(figsize=(7.6, 4.8))
    ax = plt.gca()
    if _HAS_SNS:
        hm = sns.heatmap(piv, cmap=HM_CMAP, annot=True, fmt=".1f", cbar_kws={"label": "ΔKL drop (%)"}, linewidths=0.4, linecolor="white")
        ax = hm
    else:
        im = ax.imshow(piv.values, aspect="auto", cmap=HM_CMAP)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("ΔKL drop (%)")
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels([f"{a:g}" for a in piv.columns])
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels(piv.index.tolist())
    ax.set_xlabel("alpha")
    ax.set_ylabel("hook")
    ax.set_title(f"RQ3 (cognitive): ΔKL drop (%) heatmap — L{L}")
    out = os.path.join(OUT_DIR, f"rq3_heatmap_L{L}.png")
    savefig_tight(out)

# 4) Overall best-per-(layer,hook) table to TXT alongside figures
best_rows = []
for (L, H), g in df.groupby(["layer","hook"], sort=True):
    idx = g["drop_pct"].idxmax()
    r = df.loc[idx]
    best_rows.append({
        "layer": L, "hook": H, "alpha": r["alpha"],
        "KL_raw_mean": r["KL_raw_mean"], "KL_mapped_mean": r["KL_mapped_mean"],
        "delta": r["delta"], "drop_pct": r["drop_pct"], "file": r.get("file", "")
    })
best_df = pd.DataFrame(best_rows).sort_values(["layer","drop_pct"], ascending=[True, False])
best_txt = os.path.join(OUT_DIR, "rq3_best_per_layer_hook.txt")
wfile = min(42, max(24, best_df["file"].map(len).max() if not best_df.empty else 24))
whook = max(8, best_df["hook"].map(len).max() if not best_df.empty else 8)
lines = []
lines.append("Best per (layer, hook)\n")
lines.append(f"{'layer':>5}  {'hook':<{whook}}  {'alpha':>5}  {'KL_raw':>8}  {'KL_mapped':>10}  {'Δ':>8}  {'drop%':>7}  {'file':<{wfile}}")
lines.append("-" * (wfile + whook + 60))
for _, r in best_df.iterrows():
    lines.append(
        f"{int(r['layer']):>5}  "
        f"{r['hook']:<{whook}}  "
        f"{(r['alpha'] if not np.isnan(r['alpha']) else 'NA'):>5}  "
        f"{r['KL_raw_mean']:>8.3f}  {r['KL_mapped_mean']:>10.3f}  "
        f"{r['delta']:>8.3f}  {r['drop_pct']:>6.1f}  "
        f"{r['file']:<{wfile}}"
    )
with open(best_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Wrote: {best_txt}")

print(f"\nDone. Plots saved in: {OUT_DIR}")


