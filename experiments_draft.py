"""
comparative_diffing.py
Comparative, causal model-diffing for Behavioral/Cultural vs Cognitive fine-tunes
Backbone: Llama-3.1-8B family (same tokenizer strongly recommended)

Dependencies:
  pip install torch transformers einops
Optional (if you add your own): datasets, numpy, matplotlib

This file provides:
  - Setup & fairness controls
  - RQ1 Baseline Divergence
  - RQ2 Transportability via CLT (ridge)
  - RQ3 Causal Control & Rank (steer/ablate)
  - RQ4 Structure Change (head/path patching)
  - RQ5 Generality (replication) skeleton
  
Notes You’ll Likely Want To Tweak

MC Scoring: Replace the naive “logprob of a digit label” with your dataset’s exact formatting (e.g., score the completion for each option and compare).

Head Ablation (RQ4): The stub shows where to place a hook; for precise head ablation, hook inside the attention module before head merge (e.g., modify module.self_attn.o_proj input per head). Keep this optional if time is tight.

Subspace Construction (RQ3): You can also build 𝑈𝑘 from difference-in-means neighborhoods or with PCA on aligned-minus-neutral activations—whatever is quickest.

CLT Data Size: 5–10k token positions is often enough; you can subsample your prompts.

Fairness: If tokenizers match (recommended), the shared-vocab code still works and costs nothing.
  
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional
import math, re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Config & Bundles
# -----------------------------

@dataclass
class ModelConfig:
    model_id: str
    dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"

@dataclass
class RunConfig:
    layer_indices: List[int]           # e.g., [20, 22, 24]
    ridge_lambda: float = 1e-2
    max_seq_len: int = 2048
    seed: int = 123

@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tok: AutoTokenizer
    id: str

def load_bundle(cfg: ModelConfig) -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=cfg.dtype, device_map=cfg.device_map
    )
    return ModelBundle(model=model, tok=tok, id=cfg.model_id)

# -----------------------------
# Fairness Controls (Tokenizer)
# -----------------------------

def shared_vocab_sets(tok_a: AutoTokenizer, tok_b: AutoTokenizer) -> Tuple[set, set, set, float]:
    """Return (shared_tokens, only_a, only_b, pct_increase_b_over_a)."""
    va = tok_a.get_vocab(); vb = tok_b.get_vocab()
    sa = set(va.keys()); sb = set(vb.keys())
    shared = sa & sb
    pct_inc = (len(sb) - len(sa)) / max(1, len(sa)) * 100.0
    return shared, (sa - sb), (sb - sa), pct_inc

def uses_only_shared(text: str, tok: AutoTokenizer, shared: set) -> bool:
    inv = {i:s for s,i in tok.get_vocab().items()}
    ids = tok(text, add_special_tokens=False).input_ids
    return all(inv[i] in shared for i in ids)

def filter_shared_prompts(prompts: List[str], tok_a, tok_b, shared: set) -> List[str]:
    return [p for p in prompts if uses_only_shared(p, tok_a, shared) and uses_only_shared(p, tok_b, shared)]

def allowed_id_set(tok: AutoTokenizer, shared: set) -> set:
    vb = tok.get_vocab()
    return {vb[s] for s in shared}

def mask_logits_to_ids(logits: torch.Tensor, allowed_ids: set) -> torch.Tensor:
    """Mask logits to allowed ids (batch, vocab)."""
    out = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(sorted(list(allowed_ids)), device=logits.device, dtype=torch.long)
    out.index_copy_(1, idx, logits.index_select(1, idx))
    return out

# -----------------------------
# Utilities: Hooks & Captures
# -----------------------------

def llama_layers(model) -> nn.ModuleList:
    # Llama-like models: model.model.layers is the list of decoder blocks
    return model.model.layers

class Capture:
    """Forward hook capture of hidden states at a given layer (block output)."""
    def __init__(self, module: nn.Module):
        self.acts = []
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # For LlamaDecoderLayer, 'out' is the hidden_states tensor (B, T, D)
        if isinstance(out, tuple):
            out = out[0]
        self.acts.append(out.detach())

    def close(self):
        self.hook.remove()

def collect_last_token_states(bundle: ModelBundle, texts: List[str], layer_idx: int, max_seq_len: int = 2048) -> torch.Tensor:
    """Return (N, D) hidden states at the LAST token for each text at a specific layer."""
    model, tok = bundle.model, bundle.tok
    model.eval()
    layer = llama_layers(model)[layer_idx]
    cap = Capture(layer)
    outs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model.device)
        with torch.no_grad():
            _ = model(**enc)
        h = cap.acts[-1]  # (1, T, D)
        outs.append(h[:, -1, :].to("cpu"))
    cap.close()
    return torch.cat(outs, dim=0)  # (N, D)

# -----------------------------
# RQ1: Baseline Divergence
# -----------------------------

def refusal_rate(responses: List[str]) -> float:
    PAT = re.compile(r"\b(i (can('|)t|cannot|won('|)t|am unable)|sorry,?\s*i\b", re.I)
    hits = sum(1 for r in responses if PAT.search(r or ""))
    return hits / max(1, len(responses))

def style_marker_rate(responses: List[str], markers: List[str]) -> float:
    m = [re.compile(re.escape(s), re.I) for s in markers]
    hits = sum(1 for r in responses if any(mm.search(r or "") for mm in m))
    return hits / max(1, len(responses))

def generate_next_token(bundle: ModelBundle, prompt: str, allowed_ids: Optional[set]) -> str:
    model, tok = bundle.model, bundle.tok
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[:, -1, :]
        if allowed_ids is not None:
            logits = mask_logits_to_ids(logits, allowed_ids)
        nxt = logits.argmax(dim=-1)
    return tok.decode(nxt)

def average_logprob_correct_mc(bundle: ModelBundle, batch_inputs: List[Dict], allowed_ids: Optional[set]) -> float:
    """
    batch_inputs: list of dicts like {
        "stem": "Q ...",
        "choices": ["A", "B", "C", "D"],
        "correct_idx": 2
    }
    """
    model, tok = bundle.model, bundle.tok
    model.eval()
    logps = []
    for ex in batch_inputs:
        # TODO: define your prompt formatting for MC
        prompt = ex["stem"] + "\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(ex["choices"]))
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits[:, -1, :]
            if allowed_ids is not None:
                logits = mask_logits_to_ids(logits, allowed_ids)
            probs = logits.log_softmax(dim=-1)
        # Naive: match first token id of correct option label (you may replace with better scoring)
        correct_label = str(ex["correct_idx"])
        cid = tok(correct_label, add_special_tokens=False).input_ids[0]
        logps.append(float(probs[0, cid].to("cpu")))
    return sum(logps) / max(1, len(logps))

def cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between (N, D) matrices. Returns scalar in [0, 1].
    """
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    Kxy = (Xc.t() @ Yc)
    num = (Kxy.norm(p="fro") ** 2)
    den = (Xc.t() @ Xc).norm(p="fro") * (Yc.t() @ Yc).norm(p="fro")
    if den == 0:
        return 0.0
    return float((num / (den + 1e-8)).item())

def rq1_baseline_divergence(
    base: ModelBundle,
    tuned: ModelBundle,
    cultural_prompts_free: List[str],
    cultural_mc: List[Dict],
    cognitive_mc: List[Dict],
    shared: set,
    layers: List[int],
) -> Dict:
    """
    Returns dict with:
      - behavior deltas (cultural & cognitive)
      - layerwise representation similarity (CKA) per pair
    """
    # Allowed ids for masking
    allow_base = allowed_id_set(base.tok, shared)
    allow_tuned = allowed_id_set(tuned.tok, shared)

    # Filter free-form prompts to shared vocab
    free = [p for p in cultural_prompts_free if uses_only_shared(p, base.tok, shared) and uses_only_shared(p, tuned.tok, shared)]

    # Behavior deltas (Cultural)
    resp_base = [generate_next_token(base, p, allow_base) for p in free]
    resp_tuned = [generate_next_token(tuned, p, allow_tuned) for p in free]

    cultural_refusal_delta = refusal_rate(resp_tuned) - refusal_rate(resp_base)
    cultural_style_delta = style_marker_rate(resp_tuned, markers=["please", "thank", "sorry"]) - \
                           style_marker_rate(resp_base, markers=["please", "thank", "sorry"])

    # Behavior deltas (Cognitive): average log-prob of correct option
    cog_base = average_logprob_correct_mc(base, cognitive_mc, allow_base)
    cog_tuned = average_logprob_correct_mc(tuned, cognitive_mc, allow_tuned)
    cognitive_logprob_delta = cog_tuned - cog_base

    # Representation similarity per layer (CKA of last-token states)
    rep_sim = []
    for L in layers:
        X = collect_last_token_states(base, free, L)
        Y = collect_last_token_states(tuned, free, L)
        rep_sim.append({"layer": L, "cka": cka_linear(X, Y)})

    return {
        "cultural_refusal_delta": cultural_refusal_delta,
        "cultural_style_delta": cultural_style_delta,
        "cognitive_logprob_delta": cognitive_logprob_delta,
        "rep_similarity": rep_sim,
        "n_free": len(free),
    }

# -----------------------------
# RQ2: Transportability via CLT
# -----------------------------

def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Solve (X^T X + lam I) W = X^T Y
    X: (N, D), Y: (N, D) -> W: (D, D)
    """
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    XtX = Xc.t() @ Xc
    D = XtX.shape[0]
    reg = lam * torch.eye(D, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, Xc.t() @ Yc)  # (D, D)
    return W

def clt_transportability(
    base: ModelBundle,
    tuned: ModelBundle,
    prompts: List[str],
    layer_idx: int,
    lam: float = 1e-2,
    max_n: int = 1000,
) -> Dict:
    """
    Fit linear map W: H_base -> H_tuned at a layer and report R^2 and CKA.
    """
    texts = prompts[:max_n]
    X = collect_last_token_states(base, texts, layer_idx)  # (N, D)
    Y = collect_last_token_states(tuned, texts, layer_idx) # (N, D)

    W = fit_ridge(X, Y, lam)
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    Yhat = Xc @ W + Y.mean(0, keepdim=True)

    # R^2
    ss_res = ((Yc - (Yhat - Y.mean(0, keepdim=True)))**2).sum().item()
    ss_tot = ((Yc)**2).sum().item() + 1e-8
    r2 = 1.0 - ss_res/ss_tot

    # CKA between Yhat and Y
    cka = cka_linear(Yhat, Y)

    return {"layer": layer_idx, "r2": r2, "cka": cka, "W": W}

# -----------------------------
# RQ3: Causal Control & Rank
# -----------------------------

def mean_direction(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Unit vector of mean(A) - mean(B)."""
    d = (A.mean(0) - B.mean(0))
    return d / (d.norm() + 1e-8)

def topk_subspace(X: torch.Tensor, k: int) -> torch.Tensor:
    """Return U_k (D, k) from SVD of centered X (N, D)."""
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    # X = U S Vt; principal directions are V (columns of V)
    Vk = Vt[:k, :].t().contiguous()  # (D, k)
    return Vk

def proj_mat(Uk: torch.Tensor) -> torch.Tensor:
    """Projection matrix P = Uk Uk^T (D, D)."""
    return Uk @ Uk.t()

def steer_hook_factory(P: torch.Tensor, alpha: float):
    """Return a forward hook that adds alpha * P h at the layer output."""
    P = P.to(torch.float32)
    def hook(module, inp, out):
        if isinstance(out, tuple): out = out[0]
        return out + alpha * (out @ P.to(out.device))
    return hook

def ablate_hook_factory(P: torch.Tensor):
    """Return a forward hook that projects out P h at the layer output."""
    P = P.to(torch.float32)
    def hook(module, inp, out):
        if isinstance(out, tuple): out = out[0]
        return out - (out @ P.to(out.device))
    return hook

def rq3_causal_control_rank(
    bundle_base: ModelBundle,
    bundle_tuned: ModelBundle,
    aligned_prompts: List[str],       # prompts eliciting target behavior (for subspace)
    neutral_prompts: List[str],       # neutral controls (for direction)
    eval_prompts_cultural: List[str], # free-form eval for cultural
    eval_mc_cognitive: List[Dict],    # MC eval for cognitive
    layer_idx: int,
    ks: List[int],                     # e.g., [1,2,3,5]
    alpha: float,                      # steering strength
    shared: set,
) -> Dict:
    """
    Build subspaces, steer/ablate at layer, and report effect sizes & side-effects.
    """
    allow_base = allowed_id_set(bundle_base.tok, shared)
    allow_tuned = allowed_id_set(bundle_tuned.tok, shared)

    # Collect residuals for subspace construction on tuned model (behavioral case)
    A = collect_last_token_states(bundle_tuned, aligned_prompts, layer_idx)   # (Na, D)
    B = collect_last_token_states(bundle_tuned, neutral_prompts, layer_idx)   # (Nb, D)

    results = {"layer": layer_idx, "by_k": []}
    for k in ks:
        Uk = topk_subspace(A, k)     # (D, k)
        P  = proj_mat(Uk)            # (D, D)

        # STEER base model
        layer_b = llama_layers(bundle_base.model)[layer_idx]
        h1 = layer_b.register_forward_hook(steer_hook_factory(P, alpha))
        # Evaluate cultural metrics on free-form
        resp_b_steer = [generate_next_token(bundle_base, p, allow_base) for p in eval_prompts_cultural]
        h1.remove()

        resp_b_base = [generate_next_token(bundle_base, p, allow_base) for p in eval_prompts_cultural]

        steer_delta_refusal = refusal_rate(resp_b_steer) - refusal_rate(resp_b_base)
        steer_delta_style   = style_marker_rate(resp_b_steer, ["please","thank","sorry"]) - \
                              style_marker_rate(resp_b_base,  ["please","thank","sorry"])

        # ABLATE tuned model
        layer_t = llama_layers(bundle_tuned.model)[layer_idx]
        h2 = layer_t.register_forward_hook(ablate_hook_factory(P))
        resp_t_ablate = [generate_next_token(bundle_tuned, p, allow_tuned) for p in eval_prompts_cultural]
        h2.remove()

        resp_t_tuned = [generate_next_token(bundle_tuned, p, allow_tuned) for p in eval_prompts_cultural]

        ablate_delta_refusal = refusal_rate(resp_t_ablate) - refusal_rate(resp_t_tuned)
        ablate_delta_style   = style_marker_rate(resp_t_ablate, ["please","thank","sorry"]) - \
                               style_marker_rate(resp_t_tuned,  ["please","thank","sorry"])

        # Cognitive side check (optional): effect on math MC log-prob when steering/ablating
        # (Run small evals with hooks similar to above if desired.)

        results["by_k"].append({
            "k": k,
            "steer_delta_refusal": steer_delta_refusal,
            "steer_delta_style":   steer_delta_style,
            "ablate_delta_refusal": ablate_delta_refusal,
            "ablate_delta_style":   ablate_delta_style,
        })

    return results

# -----------------------------
# RQ4: Structure Change (Edges)
# -----------------------------

def ablate_single_head_hook(layer_module, head_idx: int):
    """
    Returns a hook that zeros one attention head output at a LlamaDecoderLayer.
    NOTE: This is architecture-specific; you may need to adapt if the module splits heads differently.
    """
    def hook(module, inp, out):
        # out: (B, T, D). We need to access attention proj before merge to D for exact head ablation.
        # Simplified surrogate: subtract per-head component via a learned basis is nontrivial.
        # TODO: Replace with a hook inside the attention submodule (module.self_attn) at the right tensor.
        return out
    return hook

def rq4_structure_change_edge_churn(
    bundle: ModelBundle,
    eval_prompts: List[str],
    layer_idx: int,
    head_indices: List[int],
    metric_fn: Callable[[List[str]], float],
    shared: set,
) -> Dict:
    """
    Ablate heads one-by-one; compute how many exceed threshold change in metric_fn (edge-churn rate).
    metric_fn: converts generated responses -> scalar metric (e.g., refusal rate)
    """
    allow = allowed_id_set(bundle.tok, shared)
    base_resp = [generate_next_token(bundle, p, allow) for p in eval_prompts]
    base_metric = metric_fn(base_resp)

    churn = []
    for h in head_indices:
        layer = llama_layers(bundle.model)[layer_idx]
        hk = layer.register_forward_hook(ablate_single_head_hook(layer, h))
        resp = [generate_next_token(bundle, p, allow) for p in eval_prompts]
        hk.remove()
        m = metric_fn(resp)
        churn.append({"head": h, "delta_metric": m - base_metric})
    return {"layer": layer_idx, "churn": churn}

# -----------------------------
# RQ5: Generality (Replication)
# -----------------------------

def replicate_pair_pipeline(
    base_cfg: ModelConfig,
    tuned_cfg: ModelConfig,
    run_cfg: RunConfig,
    cultural_prompts_free: List[str],
    cultural_mc: List[Dict],
    cognitive_mc: List[Dict],
) -> Dict:
    """One-call convenience to repeat RQ1–RQ3 for another pair."""
    base = load_bundle(base_cfg); tuned = load_bundle(tuned_cfg)
    shared, _, _, _ = shared_vocab_sets(base.tok, tuned.tok)

    # RQ1
    rq1 = rq1_baseline_divergence(
        base, tuned, cultural_prompts_free, cultural_mc, cognitive_mc, shared, run_cfg.layer_indices
    )

    # Choose top layer by lowest CKA (highest divergence)
    top_layer = sorted(rq1["rep_similarity"], key=lambda d: d["cka"])[0]["layer"]

    # RQ2
    rq2 = clt_transportability(
        base, tuned, cultural_prompts_free, top_layer, lam=run_cfg.ridge_lambda
    )

    # RQ3 (example: k sweep)
    rq3 = rq3_causal_control_rank(
        base, tuned,
        aligned_prompts=cultural_prompts_free,    # TODO: or a subset that elicits target behavior
        neutral_prompts=cultural_prompts_free[:10],
        eval_prompts_cultural=cultural_prompts_free[:30],
        eval_mc_cognitive=cognitive_mc[:20],
        layer_idx=top_layer,
        ks=[1,2,3,5],
        alpha=1.0,
        shared=shared
    )

    return {"rq1": rq1, "rq2": rq2, "rq3": rq3}

# -----------------------------
# Entry Point (Example)
# -----------------------------

def main():
    # TODO: Fill these with your chosen models
    base_cfg = ModelConfig(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
    cultural_cfg = ModelConfig(model_id="__CULTURAL_TUNE_MODEL_ID__")  # e.g., safety/role-play tune on 3.1-8B
    cognitive_cfg = ModelConfig(model_id="__COGNITIVE_TUNE_MODEL_ID__") # e.g., OpenMath2-Llama3.1-8B

    run_cfg = RunConfig(layer_indices=[20,22,24], ridge_lambda=1e-2)

    # TODO: Load or define prompts
    cultural_prompts_free = [
        # 30–50 free-form prompts eliciting politeness/refusal/cultural norms
        "Please explain why you cannot provide personal medical advice.",
        "How should I address an elder formally in a professional email?",
    ]
    cultural_mc = [
        # List of MC dicts: {"stem": "...", "choices": ["...","..."], "correct_idx": 0}
    ]
    cognitive_mc = [
        # 20–50 math items (short MC for quick scoring)
    ]

    # Run baseline vs cultural
    base = load_bundle(base_cfg)
    cult = load_bundle(cultural_cfg)
    shared, _, _, pct_inc = shared_vocab_sets(base.tok, cult.tok)
    print(f"[Cultural Pair] Shared tokens: {len(shared)} | Vocab delta: {pct_inc:.2f}%")

    rq1_c = rq1_baseline_divergence(base, cult, cultural_prompts_free, cultural_mc, cognitive_mc, shared, run_cfg.layer_indices)
    topL_c = sorted(rq1_c["rep_similarity"], key=lambda d: d["cka"])[0]["layer"]
    rq2_c = clt_transportability(base, cult, cultural_prompts_free, topL_c, run_cfg.ridge_lambda)
    rq3_c = rq3_causal_control_rank(base, cult, cultural_prompts_free, cultural_prompts_free[:10], cultural_prompts_free[:30], cognitive_mc[:20], topL_c, [1,2,3], 1.0, shared)

    # Run baseline vs cognitive
    cog = load_bundle(cognitive_cfg)
    shared2, _, _, pct_inc2 = shared_vocab_sets(base.tok, cog.tok)
    print(f"[Cognitive Pair] Shared tokens: {len(shared2)} | Vocab delta: {pct_inc2:.2f}%")

    rq1_g = rq1_baseline_divergence(base, cog, cultural_prompts_free, cultural_mc, cognitive_mc, shared2, run_cfg.layer_indices)
    topL_g = sorted(rq1_g["rep_similarity"], key=lambda d: d["cka"])[0]["layer"]
    rq2_g = clt_transportability(base, cog, cultural_prompts_free, topL_g, run_cfg.ridge_lambda)
    rq3_g = rq3_causal_control_rank(base, cog, cultural_prompts_free, cultural_prompts_free[:10], cultural_prompts_free[:30], cognitive_mc[:20], topL_g, [1,2,3], 1.0, shared2)

    print("Done. Package results and plot as needed.")

if __name__ == "__main__":
    main()
