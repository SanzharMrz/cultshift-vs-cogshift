import argparse, json, os, importlib.util
import torch
from transformers import AutoModelForCausalLM
from mechdiff.utils.tokenizers import load_tokenizers
from mechdiff.utils.hooks import cache_layer_with_hook


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--map_file", required=True, help="Path to map bundle (.pt) or CLT JSON with map_path")
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--k1_decision", action="store_true")
    ap.add_argument("--split", default="val", choices=["train","val"], help="Evaluate on train or val prompts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    meta = None
    map_path = args.map_file
    if args.map_file.endswith(".json"):
        with open(args.map_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        map_path = meta.get("map_path", map_path)
    bundle = torch.load(map_path, map_location="cpu")

    # Use the requested target layer/hook for patching to support cross-layer tests (e.g., bogus L24→L10)
    map_layer = (bundle.get("layer") if isinstance(bundle, dict) else None) or (meta.get("layer") if meta else None)
    layer = args.layer
    hook = args.hook

    pair = load_pair(args.pair)
    base_id, tuned_id = pair["base_id"], pair["tuned_id"]
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    base = AutoModelForCausalLM.from_pretrained(base_id).to(args.device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id).to(args.device)

    # data
    rq2_dir = os.path.join("mechdiff", "data", "rq2")
    prompts_path = os.path.join(rq2_dir, f"{args.split}_prompts.jsonl")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(l).get("text") for l in f if l.strip()]

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

    def kl_nexttoken_at(model, tok, chat, j, patched_vec=None):
        # optionally patch tuned residual at decision token
        with torch.no_grad():
            enc = tok(chat, return_tensors="pt")
            enc = {k: v.to(args.device) for k, v in enc.items()}
            if patched_vec is None:
                out = model(**enc)
                return out.logits[:, j, :]
            # patch via forward hook
            def injector(module, inputs, output):
                out = output.clone()
                out[:, j, :] = patched_vec.to(out.dtype).to(out.device)
                return out
            block = model.model.layers[layer]
            h = block.register_forward_hook(lambda m, i, o: injector(m, i, o))
            out = model(**enc)
            h.remove()
            return out.logits[:, j, :]

    import math
    from torch.nn.functional import log_softmax

    KL_raw = []
    KL_mapped = []
    # retrieve mapping parts
    mu_b = bundle.get("mu_b"); sd_b = bundle.get("sd_b")
    mu_t = bundle.get("mu_t"); sd_t = bundle.get("sd_t")
    solver = bundle.get("solver", "ridge")
    W = bundle.get("W")
    Q = bundle.get("Q"); s = bundle.get("s", 1.0); alpha = bundle.get("alpha", 1.0)
    Cb_invh = bundle.get("Cb_invh"); Ct_h = bundle.get("Ct_h")

    for p in prompts[:150]:
        chat_b = apply_chat(tok_b, p)
        chat_t = apply_chat(tok_t, p)
        j = last_content_index(tok_t, chat_t)
        # Original logits at decision index
        logit_orig = kl_nexttoken_at(tuned, tok_t, chat_t, j, None)
        # Base residual at decision token
        with torch.no_grad():
            enc_b = tok_b(chat_b, return_tensors="pt").to(args.device)
            with cache_layer_with_hook(base, layer, hook) as c:
                _ = base(**enc_b)
            h_b = c.acts[-1][:, j, :].to(torch.float32).cpu()
        # Raw baseline whiten/color
        h_raw = ((h_b - mu_b) / (sd_b + 1e-6)) * sd_t + mu_t
        logit_raw = kl_nexttoken_at(tuned, tok_t, chat_t, j, h_raw.to(args.device))
        p1 = log_softmax(logit_raw, dim=-1)
        p0 = log_softmax(logit_orig, dim=-1)
        KL_raw.append((p1.exp() * (p1 - p0)).sum(-1).item())
        # Mapped: apply bundle mapping
        if solver == "ridge" and W is not None:
            zb = (h_b - mu_b) / (sd_b + 1e-6)
            zhat = zb @ W
            h_map = zhat * sd_t + mu_t
        elif solver == "procrustes_scaled" and all(x is not None for x in [Cb_invh, Ct_h, Q]):
            xb = (h_b - mu_b) @ Cb_invh
            yhat_w = (xb @ Q) * float(s)
            h_map = yhat_w @ Ct_h + mu_t
            h_map = h_map * float(alpha)
        else:
            h_map = None
        if h_map is not None:
            logit_map = kl_nexttoken_at(tuned, tok_t, chat_t, j, h_map.to(args.device))
            p1m = log_softmax(logit_map, dim=-1)
            KL_mapped.append((p1m.exp() * (p1m - p0)).sum(-1).item())

    out_dir = os.path.join("mechdiff", "artifacts", "rq2")
    os.makedirs(out_dir, exist_ok=True)
    res = {
        "layer": layer,
        "hook": hook,
        "split": args.split,
        "k_positions": 1 if args.k1_decision else None,
        "n": len(KL_raw),
        "KL_raw_mean": float(sum(KL_raw)/max(1,len(KL_raw))) if KL_raw else None,
        "KL_mapped_mean": float(sum(KL_mapped)/max(1,len(KL_mapped))) if KL_mapped else None,
        "reduction": None,
        "map_layer": map_layer,
        "solver": solver,
        "alpha": float(alpha) if isinstance(alpha, (int, float)) else alpha,
        "map_file": map_path,
    }
    if KL_raw and KL_mapped:
        res["reduction"] = float(res["KL_raw_mean"] - res["KL_mapped_mean"])
    suffix = f"_{args.split}" if args.split in ("train","val") else ""
    with open(os.path.join(out_dir, f"mapped_patch_L{layer}{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved mapped_patch results (split=", args.split, "):", res)


if __name__ == "__main__":
    main()


