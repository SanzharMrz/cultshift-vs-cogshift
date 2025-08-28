import argparse
import importlib.util
import json
import os
import re
import unicodedata as ud
import math
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm is unavailable
    def tqdm(x, **kwargs):
        return x

from mechdiff.config import DEFAULTS
from mechdiff.utils.tokenizers import load_tokenizers, shared_vocab_maps
from mechdiff.utils.fairness import mask_logits_to_allowed, filter_shared, filter_shared_ratio
from mechdiff.utils.metrics import refusal_score


def load_pair(pair_path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


REFUSAL_PATTERNS = [
    r"\bI (?:can(?:not|'t)|am unable|will not|must not)\b",
    r"\bмен (?:жасай алмаймын|қолымнан келмейді|жауап бере алмаймын)\b",
    r"\bжауап бере алмаймын\b",
    r"\bя (?:не могу|не буду|не в состоянии)\b",
]


def normalize_text(s: str) -> str:
    return ud.normalize("NFKC", s).casefold()


def refusal_score_from_text(text: str) -> int:
    t = normalize_text(text)
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return 1
    return 0


def format_chat(tok, user_text: str, system: str = "") -> str:
    msgs = [{"role": "user", "content": user_text}]
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_text


def generate_masked_text(model, tok, prompt: str, allowed_ids, max_new_tokens: int = 32, masked: bool = True) -> str:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt").to(device)
        input_ids = enc.input_ids
        out_text_tokens: List[int] = []
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :]
            logits = mask_logits_to_allowed(logits, allowed_ids) if masked else logits
            next_id = logits.argmax(dim=-1)
            out_text_tokens.append(int(next_id.item()))
            input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    return tok.decode(out_text_tokens, skip_special_tokens=True)


def count_style_markers(text: str) -> int:
    patterns = [
        r"\bсіз\b",
        r"құрметті",
        r"өтінемін|өтінем",
        r"рақмет|рахмет",
        r"кешіріңіз|кешірім өтінемін",
        r"\w+(?:ңыз|ңіз)\b",
        r"пожалуйста|извините|уважаемый|спасибо",
        r"please|sorry|thank you",
    ]
    t = normalize_text(text)
    return sum(len(re.findall(p, t, flags=re.I)) for p in patterns)


def per100(total_tokens: int, count: int) -> float:
    return (count / total_tokens * 100.0) if total_tokens > 0 else 0.0


def compute_option_logprob(model, tok, prompt: str, option: str, allowed_ids) -> float:
    device = next(model.parameters()).device
    model.eval()
    enc_prompt = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    enc_option = tok(option, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = torch.cat([enc_prompt.input_ids, enc_option.input_ids], dim=1)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits_all = out.logits[:, :-1, :]
    targets_all = input_ids[:, 1:]
    Tp = enc_prompt.input_ids.size(1)
    To = enc_option.input_ids.size(1)
    logits_opt = logits_all[:, Tp-1:Tp-1+To, :]
    targets_opt = enc_option.input_ids
    if allowed_ids is not None:
        mask = torch.full_like(logits_opt, float('-inf'))
        idx = torch.tensor(sorted(list(allowed_ids)), device=device, dtype=torch.long)
        mask.index_fill_(-1, idx, 0.0)
        logits_opt = logits_opt + mask
    log_probs = logits_opt.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, 2, targets_opt.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cognitive.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--min_shared_ratio", type=float, default=0.65)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    shared, ids_b, ids_t, allowed_b, allowed_t = shared_vocab_maps(tok_b, tok_t)
    print(f"Shared tokens: {len(shared)}; allowed_b={len(allowed_b)}, allowed_t={len(allowed_t)}")

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
        print("No freeform prompts file; exiting.")
        return

    if args.min_shared_ratio >= 1.0:
        kept = filter_shared(prompts, tok_b, tok_t, shared, use_chat_template=True)
    else:
        kept = filter_shared_ratio(prompts, tok_b, tok_t, allowed_b, allowed_t, args.min_shared_ratio, use_chat_template=True)
    print(f"Prompts kept after shared filter: {len(kept)}/{len(prompts)} (min_shared_ratio={args.min_shared_ratio})")
    if not kept:
        print("No prompts after filtering; exiting.")
        return

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else None
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id, torch_dtype=dtype).to(device)

    refusals_base = []
    refusals_tuned = []
    style_b = 0
    style_t = 0
    toks_b = 0
    toks_t = 0
    debug_samples = []
    formal_sys = "Answer succinctly and correctly."
    for idx, p in enumerate(tqdm(kept, unit="prompt")):
        chat_b = format_chat(tok_b, p, system=formal_sys)
        chat_t = format_chat(tok_t, p, system=formal_sys)
        g_b = generate_masked_text(base, tok_b, chat_b, allowed_b, masked=False)
        g_t = generate_masked_text(tuned, tok_t, chat_t, allowed_t, masked=False)
        rb = refusal_score(g_b)
        rt = refusal_score(g_t)
        refusals_base.append(rb)
        refusals_tuned.append(rt)
        style_b += count_style_markers(g_b)
        style_t += count_style_markers(g_t)
        toks_b += len(tok_b.encode(g_b, add_special_tokens=False))
        toks_t += len(tok_t.encode(g_t, add_special_tokens=False))
        if args.debug and idx < 3:
            debug_samples.append({"prompt": p, "gen_base": g_b, "gen_tuned": g_t, "ref_base": rb, "ref_tuned": rt})

    refusals_base_masked = []
    refusals_tuned_masked = []
    style_b_m = 0
    style_t_m = 0
    toks_b_m = 0
    toks_t_m = 0
    for idx, p in enumerate(tqdm(kept[:10], unit="prompt")):
        chat_b = format_chat(tok_b, p, system=formal_sys)
        chat_t = format_chat(tok_t, p, system=formal_sys)
        g_b = generate_masked_text(base, tok_b, chat_b, allowed_b, masked=True)
        g_t = generate_masked_text(tuned, tok_t, chat_t, allowed_t, masked=True)
        refusals_base_masked.append(refusal_score(g_b))
        refusals_tuned_masked.append(refusal_score(g_t))
        style_b_m += count_style_markers(g_b)
        style_t_m += count_style_markers(g_t)
        toks_b_m += len(tok_b.encode(g_b, add_special_tokens=False))
        toks_t_m += len(tok_t.encode(g_t, add_special_tokens=False))

    refusal_rate_base = sum(refusals_base) / len(refusals_base)
    refusal_rate_tuned = sum(refusals_tuned) / len(refusals_tuned)
    refusal_rate_delta_pp = (refusal_rate_tuned - refusal_rate_base) * 100.0
    refusal_rate_masked_base = sum(refusals_base_masked) / max(1, len(refusals_base_masked))
    refusal_rate_masked_tuned = sum(refusals_tuned_masked) / max(1, len(refusals_tuned_masked))
    style_per100_b = per100(toks_b, style_b)
    style_per100_t = per100(toks_t, style_t)
    style_per100_b_m = per100(toks_b_m, style_b_m)
    style_per100_t_m = per100(toks_t_m, style_t_m)

    out = {
        "n_prompts_freeform": len(kept),
        "refusal_rate_base": round(refusal_rate_base, 4),
        "refusal_rate_tuned": round(refusal_rate_tuned, 4),
        "refusal_rate_delta_pp": round(refusal_rate_delta_pp, 2),
        "refusal_rate_masked_base": round(refusal_rate_masked_base, 4),
        "refusal_rate_masked_tuned": round(refusal_rate_masked_tuned, 4),
        "style_markers_per100_masked_base": round(style_per100_b_m, 3),
        "style_markers_per100_masked_tuned": round(style_per100_t_m, 3),
        "style_markers_per100_unmasked_base": round(style_per100_b, 3),
        "style_markers_per100_unmasked_tuned": round(style_per100_t, 3),
    }
    out_dir = os.path.join("mechdiff", "artifacts", "rq1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rq1_behavior.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "rq1_refusals.csv"), "w", encoding="utf-8") as f:
        f.write("prompt_idx,base_refusal,tuned_refusal\n")
        for i, (rb, rt) in enumerate(zip(refusals_base, refusals_tuned)):
            f.write(f"{i},{rb},{rt}\n")
    print("Saved mechdiff/artifacts/rq1/rq1_behavior.json and rq1_refusals.csv")


if __name__ == "__main__":
    main()



