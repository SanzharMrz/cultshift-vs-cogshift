#!/usr/bin/env python3
import argparse, json, os, importlib.util
from typing import Optional
import torch
from transformers import AutoModelForCausalLM
from mechdiff.utils.tokenizers import load_tokenizers


def load_pair(pair_path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def apply_chat(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def last_content_index(tok, chat):
    enc = tok(chat, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, "tolist"):
        ids = ids if isinstance(ids, list) else ids.tolist()
    sids = set(getattr(tok, "all_special_ids", []) or [])
    for i in range(len(ids)-1, -1, -1):
        if ids[i] not in sids:
            return i
    return len(ids)-1


def parse_head_mask(mask_str: str, n_heads: int, device) -> torch.Tensor:
    ms = (mask_str or "").strip()
    up = ms.upper()
    if up == "ALL":
        keep = [True] * n_heads
    elif up in ("", "NONE"):
        keep = [False] * n_heads
    else:
        try:
            idxs = [int(x) for x in ms.split(",") if x.strip()]
        except Exception:
            idxs = []
        keep = [h in idxs for h in range(n_heads)]
    return torch.tensor(keep, dtype=torch.bool, device=device)


def capture_pre_o_proj_inputs(model, layer: int, chat: str, tok, device: str) -> torch.Tensor:
    attn = model.model.layers[layer].self_attn
    captured: Optional[torch.Tensor] = None

    def pre_hook(module, inputs):
        nonlocal captured
        x, = inputs  # [B, S, H*d_head]
        captured = x.detach().to(torch.float32).cpu()
        return None

    h = attn.o_proj.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        enc = tok(chat, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)
    h.remove()
    assert captured is not None
    return captured  # (B, S, D)


def run_one(pair_path: str, layer: int, alpha: float, head_mask_str: str, split: str, device: str):
    pair = load_pair(pair_path)
    base_id, tuned_id = pair["base_id"], pair["tuned_id"]
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    base = AutoModelForCausalLM.from_pretrained(base_id).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id).to(device)

    rq2_dir = os.path.join("mechdiff", "data", "rq2")
    prompts_path = os.path.join(rq2_dir, f"{split}_prompts.jsonl")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(l).get("text") for l in f if l.strip()]

    KL_raw = []
    KL_masked = []

    def logits_at(model, tok, chat, j_tgt: int, head_mask_bool: torch.Tensor, src_vec: torch.Tensor):
        attn = model.model.layers[layer].self_attn
        cfg = getattr(model, "config", None)
        n_heads = getattr(cfg, "num_attention_heads", None)
        assert n_heads is not None

        def pre_hook(module, inputs):
            x, = inputs  # [B,S,D]
            B,S,D = x.shape
            d_head = D // n_heads
            x_h = x.view(B, S, n_heads, d_head)
            # Source vector for target token only: [B, D] -> [B, H, d_head]
            src = src_vec.to(x.dtype).to(x.device)
            src_h = src.view(B, n_heads, d_head)
            hm = head_mask_bool.view(1,1,n_heads,1)
            # Blend only at the target token index
            x_h[:, j_tgt, :, :] = torch.where(hm, (1.0 - alpha) * x_h[:, j_tgt, :, :] + alpha * src_h[:, :, :], x_h[:, j_tgt, :, :])
            return (x_h.view(B, S, D),)

        h = attn.o_proj.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            enc = tok(chat, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
        h.remove()
        return out.logits[:, j_tgt, :]

    from torch.nn.functional import log_softmax
    debug = os.getenv("RQ4_DEBUG", "0") == "1"
    deltas = []
    for p in prompts:
        chat_b = apply_chat(tok_b, p)
        chat_t = apply_chat(tok_t, p)
        j_t = last_content_index(tok_t, chat_t)
        j_b = last_content_index(tok_b, chat_b)
        # Original tuned logits
        with torch.no_grad():
            enc = tok_t(chat_t, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logit_orig = tuned(**enc).logits[:, j_t, :]
        # Capture base pre-o_proj inputs
        pre_base = capture_pre_o_proj_inputs(base, layer, chat_b, tok_b, device)
        # Select base vector at its target token
        src_vec = pre_base[:, j_b, :]  # [B, D]
        # Raw full-head replacement (ALL)
        hm_all = parse_head_mask("ALL", getattr(tuned.config, "num_attention_heads"), device)
        logit_full = logits_at(tuned, tok_t, chat_t, j_t, hm_all, src_vec)
        # Head-masked replacement (selected heads)
        hm_sel = parse_head_mask(head_mask_str, getattr(tuned.config, "num_attention_heads"), device)
        logit_masked = logits_at(tuned, tok_t, chat_t, j_t, hm_sel, src_vec)
        p0 = log_softmax(logit_orig, dim=-1)
        p_full = log_softmax(logit_full, dim=-1)
        p_masked = log_softmax(logit_masked, dim=-1)
        KL_raw.append((p_full.exp() * (p_full - p0)).sum(-1).item())
        KL_masked.append((p_masked.exp() * (p_masked - p0)).sum(-1).item())
        if debug and len(deltas) < 8:
            # Approximate delta magnitude via logits difference norm as a proxy (cheap)
            deltas.append(float((p_full - p0).norm().item()))

    out_dir = os.path.join("mechdiff", "artifacts", "rq4")
    os.makedirs(out_dir, exist_ok=True)
    res = {
        "layer": layer,
        "hook": "attn_out",
        "split": split,
        "k_positions": 1,
        "n": len(KL_raw),
        "KL_raw_mean": float(sum(KL_raw)/max(1,len(KL_raw))) if KL_raw else None,
        "KL_mapped_mean": float(sum(KL_masked)/max(1,len(KL_masked))) if KL_masked else None,
        "reduction": None,
        "alpha": float(alpha),
        "head_mask": head_mask_str,
        "mode": "raw_pre_o_proj",
    }
    if KL_raw and KL_masked:
        res["reduction"] = float(res["KL_raw_mean"] - res["KL_mapped_mean"])
    if deltas:
        res["debug_delta_norm_mean_first8"] = sum(deltas)/len(deltas)
    fname = f"head_mask_L{layer}_attn_out_{head_mask_str.replace(',', '-')}.json"
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved", os.path.join(out_dir, fname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hook", default="attn_out")
    ap.add_argument("--k1_decision", action="store_true")
    ap.add_argument("--split", default="val", choices=["train","val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--head_mask", default="ALL")
    args = ap.parse_args()

    assert args.hook == "attn_out", "head_mask_patch supports attn_out only"
    run_one(args.pair, args.layer, float(args.alpha), args.head_mask, args.split, args.device)


if __name__ == "__main__":
    main()


