"""
viz.py
Minimal plotting helpers for the comparative diffing pipeline.

Dependencies:
  pip install matplotlib numpy

All functions either return the created matplotlib Axes (for composition)
or save to `savepath` if provided.

# In your notebook/script after running RQ1–RQ3…
from viz import (
    plot_representation_divergence, bar_behavior_deltas,
    bar_clt_alignment, plot_steer_ablations_vs_k,
    bar_edge_churn, summary_figure_for_pair
)

# RQ1 plots for one pair
plot_representation_divergence(rq1_c["rep_similarity"], title="[Cultural] CKA by Layer")
bar_behavior_deltas(
    rq1_c["cultural_refusal_delta"],
    rq1_c["cultural_style_delta"],
    rq1_c["cognitive_logprob_delta"],
    title="[Cultural] Behavioral Deltas"
)

# RQ2 bars (pass a list if you ran CLT for several layers)
bar_clt_alignment(
    [{"pair":"base→cultural", **rq2_c}, {"pair":"base→cognitive", **rq2_g}],
    metric="r2", title="CLT Alignment (R²)"
)

# RQ3 steer/ablate vs k
plot_steer_ablations_vs_k(rq3_c, metric="refusal", title="[Cultural] Causal Control vs k")

# RQ4 edge churn
# bar_edge_churn(churn_result, thresh=0.05, title="[Cultural] Edge Churn")

# 2x2 summary panel for one pair
summary_figure_for_pair(rq1_c, [rq2_c], rq3_c, pair_name="Base→Cultural")


"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _maybe_save(ax, savepath: Optional[str]):
    if savepath:
        ax.figure.tight_layout()
        ax.figure.savefig(savepath, dpi=200)
    return ax


# -----------------------------
# RQ1: Baseline Divergence
# -----------------------------

def plot_representation_divergence(
    rep_similarity: List[Dict],               # [{"layer": int, "cka": float}, ...]
    title: str = "Layer-wise Representation Similarity (CKA)",
    savepath: Optional[str] = None,
):
    """
    Line plot of CKA similarity per layer for one pair (higher = more similar).
    """
    layers = [d["layer"] for d in rep_similarity]
    cka    = [d["cka"]   for d in rep_similarity]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(layers, cka, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA (Base vs Tuned)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _maybe_save(ax, savepath)


def bar_behavior_deltas(
    cultural_refusal_delta: float,
    cultural_style_delta: float,
    cognitive_logprob_delta: float,
    title: str = "Behavioral Deltas",
    savepath: Optional[str] = None,
):
    """
    Simple 3-bar chart showing deltas from RQ1 for one pair.
    - cultural_refusal_delta: tuned - base refusal rate
    - cultural_style_delta: tuned - base style marker rate
    - cognitive_logprob_delta: tuned - base avg log-prob(correct)
    """
    labels = ["Refusal Δ (cult)", "Style Δ (cult)", "Δ logP(correct) (cog)"]
    vals   = [cultural_refusal_delta, cultural_style_delta, cognitive_logprob_delta]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.axhline(0.0, lw=1)
    return _maybe_save(ax, savepath)


# -----------------------------
# RQ2: Transportability (CLT)
# -----------------------------

def bar_clt_alignment(
    clt_results: List[Dict],   # e.g., [{"pair":"base→cultural","layer":22,"r2":0.42,"cka":0.68}, ...]
    metric: str = "r2",        # "r2" or "cka"
    title: str = "Cross-Layer Coding Alignment",
    savepath: Optional[str] = None,
):
    """
    Bar chart comparing CLT alignment across pairs/layers.
    Pass a list for multiple pairs or multiple layers; the label is "pair@L{layer}".
    """
    assert metric in ("r2", "cka"), "metric must be 'r2' or 'cka'"
    labels = [f'{d.get("pair","pair")}@L{d["layer"]}' for d in clt_results]
    vals   = [d[metric] for d in clt_results]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1)  # typical for R²/CKA
    return _maybe_save(ax, savepath)


# -----------------------------
# RQ3: Causal Control & Rank
# -----------------------------

def plot_steer_ablations_vs_k(
    rq3_result: Dict,              # {"layer": L, "by_k": [{"k":1, "steer_delta_refusal":..., ...}, ...]}
    metric: str = "refusal",       # "refusal" or "style"
    title: Optional[str] = None,
    savepath: Optional[str] = None,
):
    """
    Line plot of steer/ablate effect sizes vs rank k for cultural metrics.
    """
    assert metric in ("refusal", "style")
    ks = [d["k"] for d in rq3_result["by_k"]]
    steer = [d[f"steer_delta_{metric}"] for d in rq3_result["by_k"]]
    ablate = [d[f"ablate_delta_{metric}"] for d in rq3_result["by_k"]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ks, steer, marker="o", label="Steer Δ")
    ax.plot(ks, ablate, marker="s", label="Ablate Δ")
    ax.set_xlabel("Rank k")
    ax.set_ylabel(f"Δ {metric.capitalize()} (tuned units)")
    ax.axhline(0.0, lw=1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ttl = title or f"Causal Control vs Rank k (Layer {rq3_result['layer']})"
    ax.set_title(ttl)
    return _maybe_save(ax, savepath)


# -----------------------------
# RQ4: Structure Change (Edges)
# -----------------------------

def bar_edge_churn(
    churn_result: Dict,            # {"layer": L, "churn": [{"head":i, "delta_metric": x}, ...]}
    thresh: float = 0.05,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
):
    """
    Bar chart of per-head metric change after ablation; draws a horizontal threshold line.
    """
    heads = [c["head"] for c in churn_result["churn"]]
    deltas = [c["delta_metric"] for c in churn_result["churn"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(heads, deltas)
    ax.axhline(0.0, lw=1)
    ax.axhline(thresh, ls="--", lw=1)
    ax.axhline(-thresh, ls="--", lw=1)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Δ metric")
    ttl = title or f"Edge Churn (Layer {churn_result['layer']})"
    ax.set_title(ttl)
    return _maybe_save(ax, savepath)


# -----------------------------
# Convenience: Side-by-side summary
# -----------------------------

def summary_figure_for_pair(
    rq1: Dict,
    rq2_list: List[Dict],
    rq3: Dict,
    pair_name: str,
    savepath: Optional[str] = None,
):
    """
    Small 2x2 summary:
      (1) RQ1 representation curve
      (2) RQ1 behavior deltas
      (3) RQ2 CLT alignment bars (R²)
      (4) RQ3 steer/ablate vs k (refusal)
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    # (1) rep curve
    layers = [d["layer"] for d in rq1["rep_similarity"]]
    cka    = [d["cka"]   for d in rq1["rep_similarity"]]
    axs[0,0].plot(layers, cka, marker="o")
    axs[0,0].set_title(f"{pair_name}: CKA by Layer")
    axs[0,0].set_xlabel("Layer"); axs[0,0].set_ylabel("CKA"); axs[0,0].grid(True, alpha=0.3)

    # (2) behavior deltas
    labels = ["Refusal Δ", "Style Δ", "Δ logP(correct)"]
    vals = [rq1["cultural_refusal_delta"], rq1["cultural_style_delta"], rq1["cognitive_logprob_delta"]]
    axs[0,1].bar(np.arange(len(vals)), vals)
    axs[0,1].set_xticks(np.arange(len(vals)))
    axs[0,1].set_xticklabels(labels, rotation=15, ha="right")
    axs[0,1].axhline(0.0, lw=1)
    axs[0,1].set_title(f"{pair_name}: Behavioral Deltas")

    # (3) CLT bars (R²)
    labels3 = [f'L{d["layer"]}' for d in rq2_list]
    r2s     = [d["r2"] for d in rq2_list]
    axs[1,0].bar(np.arange(len(r2s)), r2s)
    axs[1,0].set_xticks(np.arange(len(r2s))); axs[1,0].set_xticklabels(labels3)
    axs[1,0].set_ylim(0, 1); axs[1,0].set_title(f"{pair_name}: CLT R²")

    # (4) steer/ablate vs k (refusal)
    ks = [d["k"] for d in rq3["by_k"]]
    steer = [d["steer_delta_refusal"] for d in rq3["by_k"]]
    ablate = [d["ablate_delta_refusal"] for d in rq3["by_k"]]
    axs[1,1].plot(ks, steer, marker="o", label="Steer Δ")
    axs[1,1].plot(ks, ablate, marker="s", label="Ablate Δ")
    axs[1,1].axhline(0.0, lw=1); axs[1,1].legend()
    axs[1,1].set_xlabel("Rank k"); axs[1,1].set_title(f"{pair_name}: Causal Control vs k")

    fig.suptitle(pair_name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if savepath:
        fig.savefig(savepath, dpi=200)
    return axs
