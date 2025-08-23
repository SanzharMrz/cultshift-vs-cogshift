#!/usr/bin/env python3
"""
Alpha sweep at patch time (no retraining).
Loads a saved CLT map bundle (.pt), overrides alpha scaling, runs mapped vs raw patch,
and writes per-alpha JSONs with KL stats.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

# Ensure repo root on sys.path for `mechdiff` imports when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM


def load_pair(pair_path: str):
    import importlib.util
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


def last_content_index(tok, chat: str) -> int:
    enc = tok(chat, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, "tolist"):
        ids = ids if isinstance(ids, list) else ids.tolist()
    sids = set(getattr(tok, "all_special_ids", []) or [])
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] not in sids:
            return i
    return len(ids) - 1


def capture_at_hook(model, tok, layer: int, hook: str, chat: str, device: str) -> torch.Tensor:
    from mechdiff.utils.hooks import cache_layer_with_hook
    with torch.no_grad():
        enc = tok(chat, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with cache_layer_with_hook(model, layer, hook) as c:
            _ = model(**enc)
        acts = c.acts[-1]  # (1, T, d)
    return acts


def inject_and_logits(model, tok, layer: int, hook: str, chat: str, j: int, vec: torch.Tensor, device: str) -> torch.Tensor:
    # Register a hook at the same module as capture and replace position j
    if hook == "resid_post":
        block = model.model.layers[layer]
        def injector(module, inputs, output):
            out = output.clone()
            out[:, j, :] = vec.to(out.dtype).to(out.device)
            return out
        h = block.register_forward_hook(lambda m, i, o: injector(m, i, o))
    elif hook in ("attn_out", "mlp_out"):
        block = model.model.layers[layer]
        target = getattr(block, "self_attn" if hook == "attn_out" else "mlp")
        def injector(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            out = out.clone()
            out[:, j, :] = vec.to(out.dtype).to(out.device)
            # Match expected return signature
            if hook == "attn_out":
                return (out, None)
            return out
        h = target.register_forward_hook(lambda m, i, o: injector(m, i, o))
    else:
        raise ValueError(f"Unsupported hook {hook}")
    with torch.no_grad():
        enc = tok(chat, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
    h.remove()
    return out.logits[:, j, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cultural.py")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--hook", default="attn_out")
    ap.add_argument("--k1_decision", action="store_true")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,1.0")
    ap.add_argument("--map", default="", help="Path to .pt map bundle; if empty, pick latest JSON's map_path for layer")
    ap.add_argument("--split", default="val", choices=["train","val"], help="Which split to evaluate (no retrain)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id, tuned_id = pair["base_id"], pair["tuned_id"]
    from mechdiff.utils.tokenizers import load_tokenizers
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    base = AutoModelForCausalLM.from_pretrained(base_id).to(args.device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id).to(args.device)

    # Pick map bundle
    map_pt = args.map
    if not map_pt:
        js = sorted(glob.glob(f"mechdiff/artifacts/rq2/rq2_clt_L{args.layer}_procrustes_scaled_*.json"))
        if not js:
            raise SystemExit("No CLT JSONs found")
        d = json.load(open(js[-1]))
        map_pt = d.get("map_path", "")
        if not map_pt:
            raise SystemExit("map_path missing in JSON")
    bundle = torch.load(map_pt, map_location="cpu")

    mu_b, sd_b = bundle.get("mu_b"), bundle.get("sd_b")
    mu_t, sd_t = bundle.get("mu_t"), bundle.get("sd_t")
    solver = bundle.get("solver", "procrustes_scaled")
    W = bundle.get("W")
    Q, s = bundle.get("Q"), bundle.get("s", 1.0)
    Cb_invh, Ct_h = bundle.get("Cb_invh"), bundle.get("Ct_h")

    # Data
    rq2_dir = os.path.join("mechdiff", "data", "rq2")
    pfile = "val_prompts.jsonl" if args.split=="val" else "train_prompts.jsonl"
    prompts_path = os.path.join(rq2_dir, pfile)
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text") or obj.get("prompt")
            except Exception:
                t = line.strip()
            if t:
                prompts.append(t)

    # Evaluate per alpha
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    for alpha in alphas:
        KL_raw, KL_mapped = [], []
        for p in prompts[:150]:
            chat_b = apply_chat(tok_b, p)
            chat_t = apply_chat(tok_t, p)
            j = last_content_index(tok_t, chat_t) if args.k1_decision else -1
            # original logits at j
            with torch.no_grad():
                enc = tok_t(chat_t, return_tensors="pt").to(args.device)
                logit_orig = tuned(**enc).logits[:, j, :]
            # capture base act at hook
            acts_b = capture_at_hook(base, tok_b, args.layer, args.hook, chat_b, args.device)
            h_b = acts_b[:, j, :].to(torch.float32).cpu()
            # raw baseline (simple stats alignment)
            h_raw = ((h_b - mu_b) / (sd_b + 1e-6)) * sd_t + mu_t
            logit_raw = inject_and_logits(tuned, tok_t, args.layer, args.hook, chat_t, j, h_raw, args.device)
            p1 = log_softmax(logit_raw, dim=-1); p0 = log_softmax(logit_orig, dim=-1)
            KL_raw.append((p1.exp() * (p1 - p0)).sum(-1).item())
            # mapped with alpha override
            if solver == "ridge" and W is not None:
                zb = (h_b - mu_b) / (sd_b + 1e-6)
                zhat = zb @ W
                h_map = zhat * sd_t + mu_t
                h_map = h_map * alpha
            elif solver == "procrustes_scaled" and all(x is not None for x in [Cb_invh, Ct_h, Q]):
                xb = (h_b - mu_b) @ Cb_invh
                yhat_w = (xb @ Q) * float(s)
                h_map = yhat_w @ Ct_h + mu_t
                h_map = h_map * float(alpha)
            else:
                h_map = None
            if h_map is not None:
                logit_map = inject_and_logits(tuned, tok_t, args.layer, args.hook, chat_t, j, h_map, args.device)
                p1m = log_softmax(logit_map, dim=-1)
                KL_mapped.append((p1m.exp() * (p1m - p0)).sum(-1).item())

        # write result
        out_dir = os.path.join("mechdiff", "artifacts", "rq2")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = {
            "layer": args.layer,
            "hook": args.hook,
            "k_positions": 1 if args.k1_decision else None,
            "alpha": alpha,
            "n": len(KL_raw),
            "KL_raw_mean": float(sum(KL_raw)/max(1,len(KL_raw))) if KL_raw else None,
            "KL_mapped_mean": float(sum(KL_mapped)/max(1,len(KL_mapped))) if KL_mapped else None,
        }
        if out["KL_raw_mean"] is not None and out["KL_mapped_mean"] is not None:
            out["reduction"] = out["KL_raw_mean"] - out["KL_mapped_mean"]
        out_path = os.path.join(out_dir, f"mapped_patch_L{args.layer}_{args.hook}_alpha{alpha}_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Wrote", out_path, out)


if __name__ == "__main__":
    main()


