#!/usr/bin/env python3
# plots for cognitive RQ3 ranks (reads cognitive artifacts, writes cognitive figs)
import os, re, glob, json, math, argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Use cognitive artifact directories
ART_DIR = "mechdiff/artifacts/cognitive/rq3"
RANK_DIR = os.path.join(ART_DIR, "ranks")
_DEFAULT_OUT = os.path.join(ART_DIR, "fig", "ranks")

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default=os.environ.get("COG_RQ3_OUT_DIR", _DEFAULT_OUT))
_ARGS = parser.parse_args()
FIG_DIR = _ARGS.out_dir

# ---------- helpers ----------
def parse_one_json(path: str) -> Dict[str, Any]:
    d = json.load(open(path))
    layer = int(d.get("layer", re.search(r"L(\d+)", path).group(1)) if re.search(r"L(\d+)", path) else -1)
    hook = d.get("hook")
    if not hook:
        m = re.search(r"mapped_patch_L\d+_(\w+)_", os.path.basename(path))
        hook = m.group(1) if m else "unknown"
    alpha = d.get("alpha", None)
    if alpha is None:
        m = re.search(r"alpha([0-9.]+)", os.path.basename(path))
        alpha = float(m.group(1)) if m else np.nan
    k = d.get("k", None)
    if k is None:
        m = re.search(r"_k(\w+)_", os.path.basename(path))
        if m:
            k = m.group(1)
    kl_r = float(d.get("KL_raw_mean", np.nan))
    kl_m = float(d.get("KL_mapped_mean", np.nan))
    drop = 100.0 * (kl_r - kl_m) / kl_r if (kl_r and not math.isnan(kl_r) and not math.isnan(kl_m)) else np.nan
    return {
        "file": os.path.basename(path),
        "path": path,
        "layer": layer,
        "hook": hook,
        "alpha": float(alpha) if alpha is not None else np.nan,
        "k": k,
        "KL_raw": kl_r,
        "KL_mapped": kl_m,
        "delta": (kl_r - kl_m) if (not math.isnan(kl_r) and not math.isnan(kl_m)) else np.nan,
        "drop_pct": drop,
    }

def load_mapped_patch_df() -> pd.DataFrame:
    js = glob.glob(os.path.join(RANK_DIR, "mapped_patch_*.json"))
    rows = [parse_one_json(p) for p in js]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[~df["KL_mapped"].isna()].copy()
    hook_order = ["attn_out", "mlp_out", "resid_post", "unknown"]
    df["hook"] = pd.Categorical(df["hook"], hook_order, ordered=True)
    df = df.sort_values(["layer", "hook", "alpha"], kind="mergesort")
    return df

def find_map_pts(layer: int, hook: str) -> List[str]:
    pats = [
        os.path.join(ART_DIR, "rq2_clt_L{}_procrustes_scaled_*{}.json".format(layer, hook)),
        os.path.join(ART_DIR, "rq2_clt_L{}_procrustes_scaled_*.json".format(layer)),
    ]
    hits: List[str] = []
    for p in pats:
        hits.extend(glob.glob(p))
    pts: List[str] = []
    for j in sorted(hits):
        try:
            mp = json.load(open(j)).get("map_path")
            if mp and os.path.exists(mp):
                pts.append(mp)
        except Exception:
            pass
    seen: set[str] = set()
    uniq: List[str] = []
    for p in pts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def try_singular_spectrum_from_map(pt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mp = torch.load(pt_path, map_location="cpu")
    if "W" in mp:
        W = mp["W"].float().numpy()
        s = np.linalg.svd(W, compute_uv=False)
        return s, s
    have = mp.keys()
    needed = {"Cb_evecs", "Cb_evals", "Ct_evecs", "Ct_evals", "Q", "s"}
    if needed.issubset(have):
        Cb_U = mp["Cb_evecs"].float().numpy()
        Cb_l = mp["Cb_evals"].float().numpy()
        Ct_U = mp["Ct_evecs"].float().numpy()
        Ct_l = mp["Ct_evals"].float().numpy()
        Q = mp["Q"].float().numpy()
        s = float(mp["s"])
        floor_b = max(1e-6, float(Cb_l.max()) * 1e-6)
        floor_t = max(1e-6, float(Ct_l.max()) * 1e-6)
        Cb_invhalf = Cb_U @ np.diag(1.0 / np.sqrt(np.clip(Cb_l, floor_b, None))) @ Cb_U.T
        Ct_half = Ct_U @ np.diag(np.sqrt(np.clip(Ct_l, floor_t, None))) @ Ct_U.T
        W = Ct_half @ (s * Q) @ Cb_invhalf
        sig = np.linalg.svd(W, compute_uv=False)
        return sig, sig
    return np.array([]), np.array([])

# ---------- main ----------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_mapped_patch_df()
    if df.empty:
        print("No mapped_patch_*.json files found in:", RANK_DIR)
        return

    # Plot 1: drop% vs alpha with hue=k per (layer,hook)
    for (L, H), gdf in df.groupby(["layer", "hook"], sort=True):
        gg = gdf.dropna(subset=["alpha"]).copy()
        if gg.empty:
            continue
        plt.figure(figsize=(7.0, 4.2))
        sns.lineplot(data=gg, x="alpha", y="drop_pct", hue="k", marker="o")
        plt.xlabel("alpha (patch scale)")
        plt.ylabel("ΔKL drop (%)")
        plt.title(f"Ranks: ΔKL vs α — L{L} / {H}")
        plt.grid(True, alpha=0.3)
        out1 = os.path.join(FIG_DIR, f"rq3_ranks_drop_vs_alpha_L{L}_{H}.png")
        plt.savefig(out1, dpi=220, bbox_inches="tight")
        plt.close()
        print("Saved", out1)

    # Plot 2: heatmap over k × alpha -> drop%
    for (L, H), gdf in df.groupby(["layer", "hook"], sort=True):
        gg = gdf.dropna(subset=["alpha"]).copy()
        if gg.empty:
            continue
        gg["k_str"] = gg["k"].astype(str)
        piv = gg.pivot_table(index="k_str", columns="alpha", values="drop_pct", aggfunc="max")
        piv = piv.reindex(sorted(piv.columns), axis=1)
        plt.figure(figsize=(7.0, 4.2))
        ax = sns.heatmap(piv, annot=True, fmt=".1f", cbar=True)
        ax.set_xlabel("alpha")
        ax.set_ylabel("rank k")
        ax.set_title(f"Ranks: ΔKL drop (%) — L{L} / {H}")
        out2 = os.path.join(FIG_DIR, f"rq3_ranks_heatmap_L{L}_{H}.png")
        plt.savefig(out2, dpi=220, bbox_inches="tight")
        plt.close()
        print("Saved", out2)

    # Plot 3: scree + cumulative (use strongest cell) — outputs under fig/ranks
    strongest = df.sort_values("drop_pct", ascending=False).head(1)
    if strongest.empty:
        print("No strong cell found for scree.")
        return
    L = int(strongest.iloc[0]["layer"])
    H = str(strongest.iloc[0]["hook"])
    cand_pts = find_map_pts(L, H)
    if not cand_pts:
        print(f"No map *.pt found for layer={L} hook={H}; skipping scree.")
        return
    used = None
    for pt in reversed(cand_pts):
        try:
            svals, _ = try_singular_spectrum_from_map(pt)
            if svals.size > 0:
                used = (pt, svals)
                break
        except Exception:
            continue
    if used is None:
        print("Could not parse any map *.pt to get singular spectrum; skipping scree.")
        return
    pt_path, svals = used
    svals = np.sort(svals)[::-1]
    energy = svals ** 2
    cum = np.cumsum(energy) / max(1e-12, np.sum(energy))
    k = np.arange(1, len(svals) + 1)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(k, svals)
    ax1.set_yscale("log")
    ax1.set_xlabel("rank k")
    ax1.set_ylabel("singular value (log)")
    ax1.set_title(f"Scree — L{L} / {H}\n{os.path.basename(pt_path)}")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(k, cum)
    ax2.set_ylim(0, 1.01)
    ax2.set_xlabel("rank k")
    ax2.set_ylabel("cumulative energy")
    ax2.set_title("Cumulative variance explained")
    out3 = os.path.join(FIG_DIR, f"rq3_scree_L{L}_{H}.png")
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", out3)
    # write small txt with k@50/80/90
    def k_for(th):
        return int(np.searchsorted(cum, th) + 1)
    txt = os.path.join(FIG_DIR, f"rq3_scree_stats_L{L}_{H}.txt")
    with open(txt, "w") as f:
        f.write(f"map: {pt_path}\n")
        f.write(f"k@50% = {k_for(0.50)}\n")
        f.write(f"k@80% = {k_for(0.80)}\n")
        f.write(f"k@90% = {k_for(0.90)}\n")
    print("Saved", txt)

if __name__ == "__main__":
    main()


