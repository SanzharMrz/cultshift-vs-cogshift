import argparse
import importlib.util
import json
import os
from typing import Tuple
from datetime import datetime
import logging

import torch
from typing import List
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
from transformers import AutoModelForCausalLM

from mechdiff.utils.tokenizers import load_tokenizers
from mechdiff.utils.activations import collect_last_token_resids
from mechdiff.utils.cka import linear_cka


def load_pair(path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def standardize(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    return (X - mu) / sd, mu, sd


logging.basicConfig(level=logging.INFO, format='[CLT] %(message)s')


def _to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32, copy=True)


def cov_shrinkage(X: torch.Tensor, gamma: float = 0.05):
    N, d = X.shape
    mu = X.mean(0, keepdim=True)
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
    C_invh = (evecs @ torch.diag(torch.rsqrt(evs_f)) @ evecs.T).contiguous()
    C_h = (evecs @ torch.diag(torch.sqrt(evs_f)) @ evecs.T).contiguous()
    return C_invh, C_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam_grid", default="1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1.0")
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--solver", default="procrustes_scaled")
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--alpha", default="auto")
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--k1_decision", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
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

    # read prepared prompts and positions from cognitive path
    rq2_dir = os.path.join("mechdiff", "data", "cognitive", "rq2")
    tr_prompts_path = os.path.join(rq2_dir, "train_prompts.jsonl")
    va_prompts_path = os.path.join(rq2_dir, "val_prompts.jsonl")
    tr_pos_path = os.path.join(rq2_dir, "train_positions.json")
    va_pos_path = os.path.join(rq2_dir, "val_positions.json")
    for pth in (tr_prompts_path, va_prompts_path, tr_pos_path, va_pos_path):
        if not os.path.exists(pth):
            print("Missing:", pth)
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
    tr_positions = json.load(open(tr_pos_path, "r", encoding="utf-8"))
    va_positions = json.load(open(va_pos_path, "r", encoding="utf-8"))

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
                cnt = -1
                for j, tid in enumerate(ids):
                    if tid not in sids:
                        cnt += 1
                    if j == i:
                        return cnt
        return 0

    def collect_for_split(prompts: List[str], pos_map: dict, model, tok, split_name: str) -> torch.Tensor:
        feats: List[torch.Tensor] = []
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
            keep_idx = [j for j, pl in enumerate(pos_lists) if pl]
            if not keep_idx:
                continue
            chats_kept = [chats[j] for j in keep_idx]
            pos_lists_kept = [pos_lists[j] for j in keep_idx]
            with torch.no_grad():
                enc = tok(chats_kept, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with cache_layer_with_hook(model, L, args.hook) as c:
                    _ = model(**enc)
                acts = c.acts[-1]
            for b, ch in enumerate(chats_kept):
                idxs_b = [content_index_to_input_index(tok, ch, ci) for ci in pos_lists_kept[b]]
                if not idxs_b:
                    continue
                h = acts[b:b+1, idxs_b, :].to(dtype=torch.float32)
                feats.append(h.reshape(-1, acts.size(-1)))
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

    Hb_tr = collect_for_split(tr_prompts, tr_positions, base, tok_b, "train")
    Ht_tr = collect_for_split(tr_prompts, tr_positions, tuned, tok_t, "train")
    Hb_val = collect_for_split(va_prompts, va_positions, base, tok_b, "val")
    Ht_val = collect_for_split(va_prompts, va_positions, tuned, tok_t, "val")
    if Hb_tr.numel() == 0 or Ht_tr.numel() == 0 or Hb_val.numel() == 0 or Ht_val.numel() == 0:
        print("Empty features; check prompts and positions.")
        return

    Hb_tr_z, mu_b, sd_b = standardize(Hb_tr.float())
    Ht_tr_z, mu_t, sd_t = standardize(Ht_tr.float())
    Hb_val_z = (Hb_val.float() - mu_b) / sd_b
    Ht_val_z = (Ht_val.float() - mu_t) / sd_t

    try:
        cka_raw = linear_cka(Hb_val, Ht_val)
        print(f"[CLT] Raw CKA (val): {cka_raw:.4f}")
    except Exception:
        pass

    # Variance filter
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

    # Solver: procrustes_scaled only for cognitive default
    X_tr = _to_f32(Hb_tr)
    Y_tr = _to_f32(Ht_tr)
    X_va = _to_f32(Hb_val)
    Y_va = _to_f32(Ht_val)
    Cb, muB = cov_shrinkage(X_tr, float(args.shrink))
    Ct, muT = cov_shrinkage(Y_tr, float(args.shrink))
    Cb_invh, Cb_h = zca_factors(Cb)
    Ct_invh, Ct_h = zca_factors(Ct)
    Xw_tr = (X_tr - muB) @ Cb_invh
    Yw_tr = (Y_tr - muT) @ Ct_invh
    Xw_va = (X_va - muB) @ Cb_invh
    Yw_va = (Y_va - muT) @ Ct_invh
    M = (Xw_tr.T @ Yw_tr).to(torch.float32)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k = int(args.pca or 0)
    if k > 0 and k < U.shape[1]:
        U_k = U[:, :k]
        Vh_k = Vh[:k, :]
        S_k = S[:k]
        Q = U_k @ Vh_k
        s = (torch.sum(S_k).item() / max(torch.sum(Xw_tr * Xw_tr).item(), 1e-12))
    else:
        Q = U @ Vh
        s = (torch.sum(S).item() / max(torch.sum(Xw_tr * Xw_tr).item(), 1e-12))
    Yw_hat_va = s * (Xw_va @ Q)

    def r2_w(y_true, y_hat):
        ss_res = torch.sum((y_true - y_hat) ** 2).item()
        ss_tot = torch.sum((y_true - y_true.mean(0, keepdim=True)) ** 2).item()
        return 1.0 - ss_res / max(1e-12, ss_tot)

    r2_train = r2_w(Yw_tr, s * (Xw_tr @ Q))
    r2_val = r2_w(Yw_va, Yw_hat_va)

    mapped_val = ((X_va - muB) @ Cb_invh @ Q) * s
    mapped_val = mapped_val @ Ct_h + muT
    if str(args.alpha).lower() == "auto":
        nt = torch.linalg.norm(Y_va, dim=1) + 1e-12
        nm = torch.linalg.norm(mapped_val, dim=1) + 1e-12
        alpha = torch.median(nt / nm).item()
        mapped_val = mapped_val * alpha
        logging.info(f"alpha(auto)={alpha:.4f}")
    cka_mapped = linear_cka(mapped_val, Y_va)

    out_dir = os.path.join("mechdiff", "artifacts", "cognitive", "rq2")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rq2_clt_L{L}_{args.hook}_procrustes_scaled_{ts}.json"

    out = {
        "layer": L,
        "hook": args.hook,
        "k_positions": 1 if args.k1_decision else None,
        "solver": "procrustes_scaled",
        "val_r2": float(r2_val),
        "train_r2": float(r2_train),
        "cka_mapped_vs_tuned_val": float(cka_mapped),
        "cos_stats": None,
        "pca_q": int(args.pca or 0),
    }
    # Save bundle
    bundle = {
        "layer": L,
        "solver": "procrustes_scaled",
        "hook": args.hook,
        "mu_b": muB.cpu(),
        "sd_b": torch.std(X_tr, 0).cpu(),
        "mu_t": muT.cpu(),
        "sd_t": torch.std(Y_tr, 0).cpu(),
        "Cb_invh": Cb_invh.cpu(),
        "Ct_h": Ct_h.cpu(),
        "Q": Q.cpu(),
        "s": float(s),
        "alpha": float(alpha) if str(args.alpha).lower()=="auto" else 1.0,
    }
    maps_dir = os.path.join(out_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    map_fname = os.path.splitext(fname)[0] + ".pt"
    torch.save(bundle, os.path.join(maps_dir, map_fname))
    out["map_path"] = os.path.join(maps_dir, map_fname)

    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {os.path.join(out_dir, fname)}")


if __name__ == "__main__":
    main()

from mechdiff.experiments.cultural.rq2.run_rq2_clt import *  # noqa: F401,F403
