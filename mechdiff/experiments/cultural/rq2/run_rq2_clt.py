import argparse
import importlib.util
import json
import os
from typing import Tuple
from datetime import datetime
import math
import logging
from datetime import datetime

import torch
from typing import List
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
from transformers import AutoModelForCausalLM

from mechdiff.config import DEFAULTS
from mechdiff.utils.tokenizers import load_tokenizers, shared_vocab_maps
from mechdiff.utils.activations import collect_last_token_resids
from mechdiff.utils.cka import linear_cka


def load_pair(pair_path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def standardize(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    return (X - mu) / sd, mu, sd


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    XtX = X.T @ X
    d = XtX.shape[0]
    return torch.linalg.solve(XtX + lam * torch.eye(d, device=X.device), X.T @ Y)


# ---------- Helpers for Procrustes-scaled pipeline ----------
logging.basicConfig(level=logging.INFO, format='[CLT] %(message)s')


def _to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32, copy=True)


def cov_shrinkage(X: torch.Tensor, gamma: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    assert X.dim() == 2
    N, d = X.shape
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(N - 1, 1)
    tr = torch.trace(C)
    C_sh = (1.0 - gamma) * C + gamma * (tr / d) * torch.eye(d, device=C.device, dtype=C.dtype)
    return C_sh, mu


def zca_factors(C: torch.Tensor, floor_ratio: float = 1e-6):
    evals, evecs = torch.linalg.eigh(C)
    ev_max = torch.max(evals)
    floor = ev_max * floor_ratio
    evs_f = torch.clamp(evals, min=floor)
    n_floored = int((evals < floor).sum().item())
    C_invh = (evecs @ torch.diag(torch.rsqrt(evs_f)) @ evecs.T).contiguous()
    C_h = (evecs @ torch.diag(torch.sqrt(evs_f)) @ evecs.T).contiguous()
    cond = (ev_max / torch.min(evs_f)).item()
    return C_invh, C_h, {
        "ev_min": float(evals.min().item()),
        "ev_max": float(ev_max.item()),
        "floor": float(floor.item()),
        "floored": n_floored,
        "cond": cond,
    }


def r2_whitened(Yw_true: torch.Tensor, Yw_hat: torch.Tensor) -> float:
    ss_res = torch.sum((Yw_true - Yw_hat) ** 2).item()
    ss_tot = torch.sum((Yw_true - Yw_true.mean(dim=0, keepdim=True)) ** 2).item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def cosine_stats(A: torch.Tensor, B: torch.Tensor):
    a = torch.nn.functional.normalize(A, dim=1)
    b = torch.nn.functional.normalize(B, dim=1)
    cos = torch.sum(a * b, dim=1).cpu()
    return {
        "mean": float(cos.mean().item()),
        "median": float(cos.median().item()),
        "p10": float(torch.quantile(cos, 0.10).item()),
        "p90": float(torch.quantile(cos, 0.90).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--layer", type=int, default=24)
    # n_samples ignored in new pipeline; kept for backward-compat
    ap.add_argument("--n_samples", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam_grid", default="1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1.0")
    ap.add_argument("--pca", type=int, default=0, help="Dimensionality for PCA (0=off)")
    ap.add_argument("--procrustes", action="store_true", help="Use orthogonal Procrustes instead of ridge (deprecated; use --solver procrustes)")
    ap.add_argument("--solver", default="ridge", help="ridge|procrustes|procrustes_scaled")
    ap.add_argument("--shrink", type=float, default=0.0, help="Shrinkage gamma for covariance (procrustes_scaled)")
    ap.add_argument("--alpha", default="none", help="none|auto: rescale mapped norms on val")
    ap.add_argument("--hook", default="resid_post", help="Hook site: resid_pre|resid_post|attn_out|mlp_out")
    ap.add_argument("--logit_subspace", type=str, default="0", help="'q' for tuned Vq, or 'joint:q' for joint unembedding")
    ap.add_argument("--k1_decision", action="store_true", help="Use K=1 last content token (decision token) per prompt")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for activation collection")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    # Ensure pad token exists for batch padding (set to EOS to avoid adding new ids)
    for _tok in (tok_b, tok_t):
        try:
            if getattr(_tok, "pad_token_id", None) is None and getattr(_tok, "eos_token", None) is not None:
                _tok.pad_token = _tok.eos_token
        except Exception:
            pass

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else None
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id, torch_dtype=dtype).to(device)

    # New: read prepared prompts and positions from data/
    rq2_dir = os.path.join("mechdiff", "data", "rq2")
    tr_prompts_path = os.path.join(rq2_dir, "train_prompts.jsonl")
    va_prompts_path = os.path.join(rq2_dir, "val_prompts.jsonl")
    tr_pos_path = os.path.join(rq2_dir, "train_positions.json")
    va_pos_path = os.path.join(rq2_dir, "val_positions.json")
    if not (os.path.exists(tr_prompts_path) and os.path.exists(va_prompts_path) and os.path.exists(tr_pos_path) and os.path.exists(va_pos_path)):
        print("Missing prepared prompts/positions in mechdiff/artifacts/rq2; please run scripts/prep_prompts_rq2.py")
        return
    def read_jsonl(p):
        lst = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    t = obj.get("text") or obj.get("prompt") or ""
                except Exception:
                    t = line.strip()
                if t:
                    lst.append(t)
        return lst
    tr_prompts = read_jsonl(tr_prompts_path)
    va_prompts = read_jsonl(va_prompts_path)
    with open(tr_pos_path, "r", encoding="utf-8") as f:
        tr_positions = json.load(f)
    with open(va_pos_path, "r", encoding="utf-8") as f:
        va_positions = json.load(f)

    L = args.layer

    def apply_chat(tok, text: str) -> str:
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return text

    def content_index_to_input_index(tok, text: str, content_idx: int) -> int:
        enc = tok(text, add_special_tokens=True, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if hasattr(ids, "tolist"):
            ids = ids if isinstance(ids, list) else ids.tolist()
        sids = set(getattr(tok, "all_special_ids", []) or [])
        cnt = -1
        for i, tid in enumerate(ids):
            if tid not in sids:
                cnt += 1
                if cnt == content_idx:
                    return i
        return len(ids) - 1

    def last_content_content_index(tok, text: str) -> int:
        enc = tok(text, add_special_tokens=True, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if hasattr(ids, "tolist"):
            ids = ids if isinstance(ids, list) else ids.tolist()
        sids = set(getattr(tok, "all_special_ids", []) or [])
        for i in range(len(ids)-1, -1, -1):
            if ids[i] not in sids:
                # translate to content index by counting content tokens up to i
                cnt = -1
                for j, tid in enumerate(ids):
                    if tid not in sids:
                        cnt += 1
                    if j == i:
                        return cnt
        return 0

    def collect_for_split(prompts: List[str], pos_map: dict, model, tok, split_name: str) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        fallbacks = 0
        total = 0
        model.eval()
        bs = max(1, int(args.batch_size))
        from mechdiff.utils.hooks import cache_layer_with_hook
        for i in tqdm(range(0, len(prompts), bs), desc=f"Collect {split_name} (L={L})", unit="batch"):
            batch_prompts = prompts[i:i+bs]
            chats = [apply_chat(tok, p) for p in batch_prompts]
            if args.k1_decision:
                pos_lists = [[last_content_content_index(tok, ch)] for ch in chats]
            else:
                pos_lists = [pos_map.get(p) or [] for p in batch_prompts]
            # filter out empties
            keep_idx = [j for j, pl in enumerate(pos_lists) if pl]
            if not keep_idx:
                continue
            chats_kept = [chats[j] for j in keep_idx]
            pos_lists_kept = [pos_lists[j] for j in keep_idx]
            # batch tokenize with padding on device
            with torch.no_grad():
                enc = tok(chats_kept, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with cache_layer_with_hook(model, L, args.hook) as c:
                    _ = model(**enc)
                acts = c.acts[-1]  # (B, T, d)
            # per-sample index mapping and gather
            for b, ch in enumerate(chats_kept):
                idxs_b = [content_index_to_input_index(tok, ch, ci) for ci in pos_lists_kept[b]]
                if not idxs_b:
                    continue
                total += len(idxs_b)
                # fallback counting: last index equals last token and last token is special
                if max(idxs_b) == acts.size(1) - 1:
                    ids = tok(ch, add_special_tokens=True).input_ids
                    last_id = int(ids[-1] if isinstance(ids, list) else ids[-1])
                    sids = set(getattr(tok, "all_special_ids", []) or [])
                    if last_id in sids:
                        fallbacks += 1
                h = acts[b:b+1, idxs_b, :].to(dtype=torch.float32)  # (1,K,d)
                feats.append(h.reshape(-1, acts.size(-1)))
        fb_rate = (fallbacks / max(1, total))
        print(f"[CLT] {split_name} fallback_rate={fb_rate:.4f} ({fallbacks}/{total})")
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

    Hb_tr = collect_for_split(tr_prompts, tr_positions, base, tok_b, "train")
    Ht_tr = collect_for_split(tr_prompts, tr_positions, tuned, tok_t, "train")
    Hb_val = collect_for_split(va_prompts, va_positions, base, tok_b, "val")
    Ht_val = collect_for_split(va_prompts, va_positions, tuned, tok_t, "val")
    if Hb_tr.numel() == 0 or Ht_tr.numel() == 0 or Hb_val.numel() == 0 or Ht_val.numel() == 0:
        print("Empty features; check prompts and positions.")
        return

    # Standardize on train only
    Hb_tr_z, mu_b, sd_b = standardize(Hb_tr.float())
    Ht_tr_z, mu_t, sd_t = standardize(Ht_tr.float())
    Hb_val_z = (Hb_val.float() - mu_b) / sd_b
    Ht_val_z = (Ht_val.float() - mu_t) / sd_t

    # Baseline raw CKA on val (before mapping)
    try:
        cka_raw = linear_cka(Hb_val, Ht_val)
        print(f"[CLT] Raw CKA (val, no mapping): {cka_raw:.4f}")
    except Exception:
        pass

    # Finite-value checks
    def assert_finite(name: str, T: torch.Tensor):
        if not torch.isfinite(T).all():
            bad = (~torch.isfinite(T)).nonzero(as_tuple=False)[:10].tolist()
            raise ValueError(f"{name} has non-finite values; first 10 indices: {bad}")

    assert_finite("Hb_tr_z", Hb_tr_z)
    assert_finite("Ht_tr_z", Ht_tr_z)
    assert_finite("Hb_val_z", Hb_val_z)
    assert_finite("Ht_val_z", Ht_val_z)

    # Variance filter to drop degenerate/near-constant dims (consistent across base/tuned)
    eps = 1e-6
    keep_mask = (sd_b.squeeze(0) > eps) & (sd_t.squeeze(0) > eps)
    if keep_mask.ndim == 0:
        keep_mask = keep_mask.view(1)
    Hb_tr_z = Hb_tr_z[:, keep_mask]
    Ht_tr_z = Ht_tr_z[:, keep_mask]
    Hb_val_z = Hb_val_z[:, keep_mask]
    Ht_val_z = Ht_val_z[:, keep_mask]
    mu_t = mu_t[:, keep_mask]
    sd_t = sd_t[:, keep_mask]
    Ht_val = Ht_val[:, keep_mask]
    print(f"[CLT] Kept {int(keep_mask.sum().item())}/{keep_mask.numel()} dims after variance filter.")

    # Optional logit-subspace projection (on tuned side) or PCA projection
    # Only treat logit_subspace as active when it's not "0"
    if (args.logit_subspace != "0") and args.pca:
        raise ValueError("Use either --logit_subspace or --pca, not both.")

    # Logit-subspace: compute SVD on tuned unembedding and project both sides into that q-dim subspace
    # Logit subspace selection
    if args.logit_subspace != "0" and hasattr(tuned, "lm_head"):
        joint = False
        if isinstance(args.logit_subspace, str) and args.logit_subspace.startswith("joint:"):
            q = int(args.logit_subspace.split(":", 1)[1])
            joint = True
        else:
            q = int(args.logit_subspace)
        W_U = tuned.lm_head.weight.detach().float()  # (V, d) on current device
        if joint and hasattr(base, "lm_head"):
            W_B = base.lm_head.weight.detach().float()
            # Build joint covariance in dxd space: C = Wb^T Wb + Wt^T Wt
            C = W_B.T @ W_B + W_U.T @ W_U
            Uc, Sc, Vc = torch.linalg.svd(C, full_matrices=False)
            Vq = Uc[:, :q]  # (d, q)
        else:
            # Right singular vectors of W_U: columns of V in SVD(W_U) = U S V^T
            _, _, Vt = torch.linalg.svd(W_U, full_matrices=False)
            Vq = Vt[:q, :].T  # (d, q)
        Vq = Vq.to(Hb_tr_z.device, dtype=Hb_tr_z.dtype)
        Hb_tr_z = Hb_tr_z @ Vq
        Ht_tr_z = Ht_tr_z @ Vq
        Hb_val_z = Hb_val_z @ Vq
        Ht_val_z = Ht_val_z @ Vq
        src = "joint" if joint else "tuned"
        print(f"[CLT] Logit-subspace({src}) → q={q}: X {tuple(Hb_tr_z.shape)}, Y {tuple(Ht_tr_z.shape)}")

    # Optional PCA projection (train-derived) applied to both train and val per side
    def pca_project_train_val(X_tr: torch.Tensor, X_val: torch.Tensor, q: int):
        # X assumed zero-mean (z-scored). Returns projected (N,q) and basis (d,q)
        U, S, V = torch.pca_lowrank(X_tr, q=q)
        Vq = V[:, :q]
        return (X_tr @ Vq, X_val @ Vq, Vq)

    Vb = None
    Vt = None
    if args.pca and args.pca < Hb_tr_z.shape[1] and not args.logit_subspace:
        q = args.pca
        Hb_tr_z, Hb_val_z, Vb = pca_project_train_val(Hb_tr_z, Hb_val_z, q)
        Ht_tr_z, Ht_val_z, Vt = pca_project_train_val(Ht_tr_z, Ht_val_z, q)
        print(f"[CLT] PCA → q={q}: X {tuple(Hb_tr_z.shape)}, Y {tuple(Ht_tr_z.shape)}")

    # Solvers
    lams = [float(x) for x in args.lam_grid.split(",") if x]
    def fit_ridge_svd(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        shrink = S / (S * S + lam)
        return Vh.transpose(-2, -1) @ (shrink[:, None] * (U.transpose(-2, -1) @ Y))

    solver = args.solver
    if args.procrustes:
        solver = "procrustes"
    if solver == "ridge":
        print(f"Grid search over lambdas: {lams}; N_tr={Hb_tr_z.shape[0]}, N_val={Hb_val_z.shape[0]}, d={Hb_tr_z.shape[1]}")
        best = {"lam": None, "r2": -1.0}
        for lam in lams:
            W_try = fit_ridge_svd(Hb_tr_z, Ht_tr_z, lam)
            pred = Hb_val_z @ W_try
            assert_finite("pred_val", pred)
            ss_res = ((pred - Ht_val_z) ** 2).sum().item()
            ss_tot = ((Ht_val_z - Ht_val_z.mean(0, keepdim=True)) ** 2).sum().item()
            r2 = 1.0 - (ss_res / max(1e-8, ss_tot))
            if r2 > best["r2"]:
                best = {"lam": lam, "r2": r2}
        lam_best = best["lam"] if best["lam"] is not None else 1e-2
        if best["lam"] is None:
            print(f"[WARN] No finite R²; falling back to lam={lam_best}")
        W = fit_ridge_svd(Hb_tr_z, Ht_tr_z, lam_best)
    elif solver == "procrustes":
        # Orthogonal Procrustes on train
        M = Hb_tr_z.T @ Ht_tr_z
        U_p, _, Vh_p = torch.linalg.svd(M, full_matrices=False)
        W = U_p @ Vh_p
    elif solver == "procrustes_scaled":
        # A) K=1 sanity
        if args.k1_decision:
            kpos_train = 1.0
            kpos_val = 1.0
        else:
            kpos_train = sum(len(v) for v in tr_positions.values()) / max(1, len(tr_positions))
            kpos_val = sum(len(v) for v in va_positions.values()) / max(1, len(va_positions))
        logging.info(f"K sanity: train_k≈{kpos_train:.2f} val_k≈{kpos_val:.2f}")
        assert kpos_train <= 1.05 and kpos_val <= 1.05, "K=1 expected; collector still returning multiple positions."
        # B) Train-only shrinkage covariances
        X_tr = _to_f32(Hb_tr); Y_tr = _to_f32(Ht_tr)
        X_va = _to_f32(Hb_val); Y_va = _to_f32(Ht_val)
        gamma = max(0.0, min(1.0, float(args.shrink)))
        Cb, mu_b = cov_shrinkage(X_tr, gamma)
        Ct, mu_t = cov_shrinkage(Y_tr, gamma)
        # C) Center by train means
        Xc_tr = X_tr - mu_b; Xc_va = X_va - mu_b
        Yc_tr = Y_tr - mu_t; Yc_va = Y_va - mu_t
        # D) ZCA with floor
        Cb_invh, Cb_h, stats_b = zca_factors(Cb, floor_ratio=1e-6)
        Ct_invh, Ct_h, stats_t = zca_factors(Ct, floor_ratio=1e-6)
        logging.info(f"Cb ev[min,max]=[{stats_b['ev_min']:.3e},{stats_b['ev_max']:.3e}] floor={stats_b['floor']:.3e} floored={stats_b['floored']} cond={stats_b['cond']:.2e}")
        logging.info(f"Ct ev[min,max]=[{stats_t['ev_min']:.3e},{stats_t['ev_max']:.3e}] floor={stats_t['floor']:.3e} floored={stats_t['floored']} cond={stats_t['cond']:.2e}")
        # E) Whiten
        Xw_tr = Xc_tr @ Cb_invh; Yw_tr = Yc_tr @ Ct_invh
        Xw_va = Xc_va @ Cb_invh; Yw_va = Yc_va @ Ct_invh
        for name, T in [("Xw_tr", Xw_tr), ("Yw_tr", Yw_tr), ("Xw_va", Xw_va), ("Yw_va", Yw_va)]:
            assert torch.isfinite(T).all(), f"{name} contains NaN/Inf"
        # F) Procrustes in whitened space (SVD-based; robust and widely available)
        M = (Xw_tr.T @ Yw_tr).to(torch.float32)
        U_p, S_p, Vh_p = torch.linalg.svd(M, full_matrices=False)
        Q = U_p @ Vh_p
        orth_err = torch.linalg.norm(Q.T @ Q - torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)).item()
        logging.info(f"Procrustes orth_error ||Q^T Q - I||_F = {orth_err:.2e}")
        den = torch.sum(Xw_tr * Xw_tr).item()
        s = (torch.sum(S_p).item() / max(den, 1e-12))
        assert s > 0, "Non-positive scale from Procrustes"
        logging.info(f"Procrustes scale s = {s:.4f}")
        # G) R2 in whitened space
        Yw_hat_va = s * (Xw_va @ Q)
        val_r2_w = r2_whitened(Yw_va, Yw_hat_va)
        train_r2_w = r2_whitened(Yw_tr, s * (Xw_tr @ Q))
        logging.info(f"R2_whitened: train={train_r2_w:.4f} val={val_r2_w:.4f}")
        # H) Color back
        mapped_val = ((Xc_va @ Cb_invh) @ Q) * s
        mapped_val = mapped_val @ Ct_h + mu_t
        # I) Alpha rescale
        if args.alpha.lower() == "auto":
            nt = torch.linalg.norm(Y_va, dim=1) + 1e-12
            nm = torch.linalg.norm(mapped_val, dim=1) + 1e-12
            alpha = torch.median(nt / nm).item()
            mapped_val = mapped_val * alpha
            logging.info(f"alpha(auto)={alpha:.4f}")
        else:
            alpha = 1.0
        lam_best = None
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Train R² and shuffled baseline
    if solver in ("ridge", "procrustes"):
        W_best = W
        pred_tr = Hb_tr_z @ W_best
        ss_res_tr = ((pred_tr - Ht_tr_z) ** 2).sum().item()
        ss_tot_tr = ((Ht_tr_z - Ht_tr_z.mean(0, keepdim=True)) ** 2).sum().item()
        r2_train = 1.0 - (ss_res_tr / max(1e-8, ss_tot_tr))
        # shuffled baseline
        perm = torch.randperm(Ht_val_z.shape[0])
        pred_shuf = Hb_val_z @ W_best
        ss_res_sh = ((pred_shuf - Ht_val_z[perm]) ** 2).sum().item()
        ss_tot_sh = ((Ht_val_z[perm] - Ht_val_z[perm].mean(0, keepdim=True)) ** 2).sum().item()
        r2_shuffled = 1.0 - (ss_res_sh / max(1e-8, ss_tot_sh))
    else:
        # procrustes_scaled: use whitened-space diagnostics already computed
        W_best = None
        r2_train = train_r2_w
        perm = torch.randperm(Yw_va.shape[0])
        r2_shuffled = r2_whitened(Yw_va[perm], Yw_hat_va)

    # Final model and CKA(mapped, tuned)
    if solver in ("ridge", "procrustes"):
        mapped_val = Hb_val_z @ W
        # Optional alpha rescale on val to match RMS norms
        if args.alpha.lower() == "auto":
            def rms(T: torch.Tensor) -> torch.Tensor:
                return torch.sqrt((T*T).mean(dim=1) + 1e-8)
            A = mapped_val
            B = Ht_val_z
            alpha_scale = (rms(B) / (rms(A) + 1e-8)).median().item()
            mapped_val = mapped_val * alpha_scale
            logging.info(f"alpha(auto)={alpha_scale:.4f}")
    # procrustes_scaled sets mapped_val earlier (colored back into raw space)
    # CKA comparison: if PCA used, operate in reduced z-space; else de-standardize
    if solver == "procrustes_scaled":
        cka_mapped = linear_cka(mapped_val, Y_va)
    else:
        if (args.logit_subspace != "0") or (args.pca and Vt is not None and args.logit_subspace == "0"):
            cka_mapped = linear_cka(mapped_val, Ht_val_z)
        else:
            mapped_val_ds = mapped_val * sd_t + mu_t
            Ht_val_ds = Ht_val
            cka_mapped = linear_cka(mapped_val_ds, Ht_val_ds)

    out_dir = os.path.join("mechdiff", "artifacts", "rq2")
    os.makedirs(out_dir, exist_ok=True)
    # K metadata: 1 for decision-token mode; otherwise average positions per prompt in train
    import math
    if args.k1_decision:
        k_avg = 1
    else:
        k_avg = int(round(sum(len(v) for v in tr_positions.values()) / max(1, len(tr_positions))))
    # Compute val R² from final W
    if solver in ("ridge", "procrustes"):
        pred_val = Hb_val_z @ W
        ss_res_val = ((pred_val - Ht_val_z) ** 2).sum().item()
        ss_tot_val = ((Ht_val_z - Ht_val_z.mean(0, keepdim=True)) ** 2).sum().item()
        val_r2_final = 1.0 - (ss_res_val / max(1e-8, ss_tot_val))
    else:
        val_r2_final = val_r2_w
    # Cosine statistics in the final comparison space
    try:
        if solver == "procrustes_scaled":
            A = mapped_val; B = Y_va
        else:
            if (args.logit_subspace != "0") or (args.pca and Vt is not None and args.logit_subspace == "0"):
                A = mapped_val; B = Ht_val_z
            else:
                A = mapped_val_ds; B = Ht_val_ds
        cos = torch.nn.functional.cosine_similarity(A, B, dim=1).cpu().numpy().tolist()
        cos_sorted = sorted(cos)
        cos_stats = {
            "mean": float(sum(cos) / max(1, len(cos))),
            "median": float(cos_sorted[len(cos_sorted)//2] if cos_sorted else 0.0),
            "p10": float(cos_sorted[int(0.10*(len(cos_sorted)-1))] if cos_sorted else 0.0),
            "p90": float(cos_sorted[int(0.90*(len(cos_sorted)-1))] if cos_sorted else 0.0),
        }
    except Exception:
        cos_stats = None

    out = {
        "layer": L,
        "n_train": int(Hb_tr.shape[0]),
        "n_val": int(Hb_val.shape[0]),
        "k_positions": k_avg,
        "hook": args.hook,
        "val_r2": round(val_r2_final, 4),
        "train_r2": round(r2_train, 4),
        "r2_shuffled": round(r2_shuffled, 4),
        "lam": None if solver == "procrustes" else lam_best,
        "solver": solver,
        "pca_q": args.pca,
        "logit_subspace": args.logit_subspace,
        "cka_mapped_vs_tuned_val": round(float(cka_mapped), 4),
        "cos_stats": cos_stats,
    }
    suffix_parts = []
    # Include hook in filename for easier disambiguation across runs
    if args.hook:
        suffix_parts.append(str(args.hook))
    if args.pca and Vt is not None and args.logit_subspace == "0":
        suffix_parts.append(f"pca{args.pca}")
    if args.logit_subspace != "0":
        suffix_parts.append(f"logit{args.logit_subspace}")
    suffix_parts.append(solver)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rq2_clt_L{L}_" + "_".join(suffix_parts) + f"_{ts}.json"

    # Save mapping and scalers for mapped-patch validation
    map_art = {
        "layer": L,
        "solver": solver,
        "lam": None if solver == "procrustes" else lam_best,
        "hook": args.hook,
        "pca_q": args.pca,
        "logit_subspace": args.logit_subspace,
        "mu_b": mu_b.cpu(),
        "sd_b": sd_b.cpu(),
        "mu_t": mu_t.cpu(),
        "sd_t": sd_t.cpu(),
        "W": W_best.cpu() if 'W_best' in locals() and W_best is not None else None,
        "Vb": Vb.cpu() if 'Vb' in locals() and Vb is not None else None,
        "Vt": Vt.cpu() if 'Vt' in locals() and Vt is not None else None,
        "Vq": Vq.cpu() if 'Vq' in locals() and Vq is not None else None,
    }
    if solver == "procrustes_scaled":
        map_art.update({
            "Cb_invh": Cb_invh.cpu(),
            "Ct_h": Ct_h.cpu(),
            "Q": Q.cpu(),
            "s": float(s),
            "alpha": float(alpha),
        })
    maps_dir = os.path.join(out_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    map_fname = os.path.splitext(fname)[0] + ".pt"
    torch.save(map_art, os.path.join(maps_dir, map_fname))
    out["map_path"] = os.path.join(maps_dir, map_fname)
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {os.path.join(out_dir, fname)}")

    if out["val_r2"] > 0.98:
        print("[WARN] val_r2 > 0.98 — possible leakage. First 5 train/val prompts:")
        print("TRAIN:")
        for p in tr_prompts[:5]:
            print(p)
        print("VAL:")
        for p in va_prompts[:5]:
            print(p)


if __name__ == "__main__":
    main()


