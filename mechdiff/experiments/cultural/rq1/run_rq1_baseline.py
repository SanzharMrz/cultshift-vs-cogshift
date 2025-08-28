import argparse
import importlib.util
import json
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechdiff.config import DEFAULTS
from mechdiff.utils.tokenizers import load_tokenizers, shared_vocab_maps
from mechdiff.utils.fairness import filter_shared, filter_shared_ratio
from mechdiff.utils.activations import collect_last_token_resids
from mechdiff.utils.cka import linear_cka


def load_pair(pair_path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", default="", help="Comma-separated explicit layer indices; otherwise stride=")
    ap.add_argument("--stride", type=int, default=DEFAULTS["layer_stride"]) 
    ap.add_argument("--prompt_file", default=None, help="Override freeform file path")
    ap.add_argument("--min_shared_ratio", type=float, default=1.0, help="Keep prompts where both tokenizers have ≥ this fraction of tokens in the shared set")
    ap.add_argument("--bootstrap", type=int, default=0, help="If >0, number of bootstrap resamples (size=200) for CI")
    ap.add_argument("--hook", default="resid_post", help="Hook site: resid_pre|resid_post|attn_out|mlp_out")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    # Tokenizers & shared vocab
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    shared, ids_b, ids_t, allowed_b, allowed_t = shared_vocab_maps(tok_b, tok_t)
    print(f"Shared tokens: {len(shared)}")

    # Prompts: load cultural freeform file when present
    if args.prompt_file:
        freeform_path = args.prompt_file
    else:
        freeform_path = pair.get("datasets", {}).get("freeform_file")
    prompts: List[str] = []
    if freeform_path and os.path.exists(os.path.join("mechdiff", freeform_path)):
        full = os.path.join("mechdiff", freeform_path)
        with open(full, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    prompts.append(obj.get("text") or obj.get("prompt") or "")
                except Exception:
                    s = line.strip()
                    if s:
                        prompts.append(s)
    else:
        # Minimal stub: fallback to a few KK strings to enable a dry run
        prompts = [
            "Сәлем! Бұл сынақ сұрағы.",
            "Қазақстанның астанасы қай қала?",
            "Абай Құнанбайұлы кім?",
        ]
    # Filter to shared-vocab prompts
    if args.min_shared_ratio >= 1.0:
        kept = filter_shared(prompts, tok_b, tok_t, shared)
    else:
        kept = filter_shared_ratio(prompts, tok_b, tok_t, allowed_b, allowed_t, args.min_shared_ratio)
    print(f"Prompts kept after shared-vocab filter: {len(kept)}/{len(prompts)}")
    if not kept:
        print("No prompts survived filtering; exiting.")
        return

    # Load models
    device = args.device
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16 if device.startswith("cuda") else None).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id, torch_dtype=torch.bfloat16 if device.startswith("cuda") else None).to(device)

    # Layers to scan
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        n_layers = len(tuned.model.layers)
        layers = list(range(0, n_layers, args.stride))
    print(f"Scanning layers: {layers}")

    cka_by_layer = {}
    for L in layers:
        Hb = collect_last_token_resids(base, tok_b, kept, L, device=device, hook_type=args.hook)
        Ht = collect_last_token_resids(tuned, tok_t, kept, L, device=device, hook_type=args.hook)
        cka = linear_cka(Hb, Ht)
        cka_by_layer[L] = cka
        print(f"Layer {L}: CKA = {cka:.4f}")

    out_dir = os.path.join("mechdiff", "artifacts", "rq1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rq1_cka.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cka_by_layer, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    # Optional bootstrap CI
    if args.bootstrap and len(kept) >= 200:
        import random
        import math
        B = args.bootstrap
        boot = {L: [] for L in layers}
        for b in range(B):
            sample = [kept[i] for i in random.sample(range(len(kept)), k=min(200, len(kept)))]
            for L in layers:
                Hb = collect_last_token_resids(base, tok_b, sample, L, device=device, hook_type=args.hook)
                Ht = collect_last_token_resids(tuned, tok_t, sample, L, device=device, hook_type=args.hook)
                boot[L].append(linear_cka(Hb, Ht))
        def mean(xs):
            return sum(xs) / max(1, len(xs))
        def ci95(xs):
            xs = sorted(xs)
            if not xs:
                return [None, None]
            lo_i = int(0.025 * (len(xs)-1))
            hi_i = int(0.975 * (len(xs)-1))
            return [xs[lo_i], xs[hi_i]]
        cka_boot = {int(L): {"mean": mean(boot[L]), "ci95": ci95(boot[L])} for L in layers}
        with open(os.path.join(out_dir, "cka_boot.json"), "w", encoding="utf-8") as f:
            json.dump(cka_boot, f, ensure_ascii=False, indent=2)
        print(f"Saved: {os.path.join(out_dir, 'cka_boot.json')}")


if __name__ == "__main__":
    main()
