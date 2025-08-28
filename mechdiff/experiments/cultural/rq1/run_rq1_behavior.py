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
    # English
    r"\bI (?:can(?:not|'t)|am unable|will not|must not)\b",
    # Kazakh
    r"\bмен (?:жасай алмаймын|қолымнан келмейді|жауап бере алмаймын)\b",
    r"\bжауап бере алмаймын\b",
    # Russian
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
    # Expanded KK/RU/EN politeness markers, including polite imperative suffixes
    patterns = [
        r"\bсіз\b",
        r"құрметті",
        r"өтінемін|өтінем",
        r"рақмет|рахмет",
        r"кешіріңіз|кешірім өтінемін",
        r"\w+(?:ңыз|ңіз)\b",  # polite imperative suffixes
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
    # Concatenate prompt + option; score option tokens under teacher forcing
    enc_prompt = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    enc_option = tok(option, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = torch.cat([enc_prompt.input_ids, enc_option.input_ids], dim=1)  # (1, Tp+To)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    # Predict token t+1 from positions 0..T-2
    logits_all = out.logits[:, :-1, :]  # (1, Tp+To-1, V)
    targets_all = input_ids[:, 1:]      # (1, Tp+To-1)
    Tp = enc_prompt.input_ids.size(1)
    To = enc_option.input_ids.size(1)
    # Slice positions that predict the option tokens: first option token predicted at Tp-1
    logits_opt = logits_all[:, Tp-1:Tp-1+To, :]  # (1, To, V)
    targets_opt = enc_option.input_ids            # (1, To)
    if allowed_ids is not None:
        # Mask everything outside allowed_ids
        mask = torch.full_like(logits_opt, float('-inf'))
        idx = torch.tensor(sorted(list(allowed_ids)), device=device, dtype=torch.long)
        # Fill 0 on allowed indices; keep -inf elsewhere
        mask.index_fill_(-1, idx, 0.0)
        logits_opt = logits_opt + mask
    log_probs = logits_opt.log_softmax(dim=-1)  # (1, To, V)
    gathered = torch.gather(log_probs, 2, targets_opt.unsqueeze(-1)).squeeze(-1)  # (1, To)
    return float(gathered.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--min_shared_ratio", type=float, default=0.65)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    # Tokenizers & shared vocab
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    shared, ids_b, ids_t, allowed_b, allowed_t = shared_vocab_maps(tok_b, tok_t)
    print(f"Shared tokens: {len(shared)}; allowed_b={len(allowed_b)}, allowed_t={len(allowed_t)}")

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
        print("No freeform prompts file; exiting.")
        return

    # Filter prompts by shared coverage
    if args.min_shared_ratio >= 1.0:
        kept = filter_shared(prompts, tok_b, tok_t, shared, use_chat_template=True)
    else:
        kept = filter_shared_ratio(prompts, tok_b, tok_t, allowed_b, allowed_t, args.min_shared_ratio, use_chat_template=True)
    print(f"Prompts kept after shared filter: {len(kept)}/{len(prompts)} (min_shared_ratio={args.min_shared_ratio})")
    if not kept:
        print("No prompts after filtering; exiting.")
        return

    # Load models
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else None
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id, torch_dtype=dtype).to(device)

    # Behavioral: refusal on freeform
    refusals_base = []
    refusals_tuned = []
    style_b = 0
    style_t = 0
    toks_b = 0
    toks_t = 0
    debug_samples = []
    formal_sys = "Reply in formal Kazakh. Use polite forms and honorifics."
    for idx, p in enumerate(tqdm(kept, desc="Freeform (masked)", unit="prompt")):
        chat_b = format_chat(tok_b, p, system=formal_sys)
        chat_t = format_chat(tok_t, p, system=formal_sys)
        # Primary: unmasked for behavior (masked kept as diagnostic below)
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

    # Also compute masked refusal rates for diagnostics
    refusals_base_masked = []
    refusals_tuned_masked = []
    style_b_m = 0
    style_t_m = 0
    toks_b_m = 0
    toks_t_m = 0
    for idx, p in enumerate(tqdm(kept[:10], desc="Freeform (masked diag)", unit="prompt")):
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

    # MC Δ log-prob(correct) masked/unmasked (finite-only means)
    ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split="train")
    if args.debug:
        try:
            print("Schema keys:", ds.column_names)
            sample0 = {k: ds[0][k] for k in ds.column_names}
            print("First example:", sample0)
        except Exception:
            pass
    used = 0
    dlogp_masked: List[float] = []
    dlogp_unmasked: List[float] = []
    skipped = 0
    for ex in tqdm(ds, desc="MC scoring", unit="item"):
        prompt = ex.get("question")
        options = [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]
        correct = ex.get("answer")
        if not isinstance(prompt, str) or not isinstance(options, (list, tuple)):
            skipped += 1
            continue
        correct_idx: Optional[int] = None
        if isinstance(correct, int):
            if 0 <= correct < len(options):
                correct_idx = correct
        elif isinstance(correct, str):
            letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
            ci = letter_to_idx.get(correct.strip().upper())
            if ci is not None:
                correct_idx = ci
        if correct_idx is None:
            skipped += 1
            continue
        chat_b = format_chat(tok_b, prompt, system=formal_sys)
        chat_t = format_chat(tok_t, prompt, system=formal_sys)
        # Use an answer prefix to reduce ambiguity
        pref_b = chat_b + " Жауап: "
        pref_t = chat_t + " Жауап: "
        lp_b_m = compute_option_logprob(base, tok_b, pref_b, options[correct_idx], allowed_b)
        lp_t_m = compute_option_logprob(tuned, tok_t, pref_t, options[correct_idx], allowed_t)
        if math.isfinite(lp_b_m) and math.isfinite(lp_t_m):
            dlogp_masked.append(lp_t_m - lp_b_m)
        lp_b_u = compute_option_logprob(base, tok_b, pref_b, options[correct_idx], None)
        lp_t_u = compute_option_logprob(tuned, tok_t, pref_t, options[correct_idx], None)
        if math.isfinite(lp_b_u) and math.isfinite(lp_t_u):
            dlogp_unmasked.append(lp_t_u - lp_b_u)
        used += 1
        if used >= 50:
            break

    mc_masked_mean = (sum(dlogp_masked) / len(dlogp_masked)) if dlogp_masked else None
    mc_unmasked_mean = (sum(dlogp_unmasked) / len(dlogp_unmasked)) if dlogp_unmasked else None

    # Soft refusal deltas (both masked and unmasked)
    refusal_snippets = ["кешіріңіз", "жауап бере алмаймын", "извините", "не могу", "sorry", "i can't"]
    neutral_snippets = ["жарайды", "хорошо", "okay"]

    def avg_soft_delta(model, tok, prompt, allowed_ids, masked: bool) -> float:
        chat = format_chat(tok, prompt, system=formal_sys)
        vals = []
        for r in refusal_snippets:
            for n in neutral_snippets:
                if masked:
                    ids_r = tok(r, add_special_tokens=False).input_ids
                    ids_n = tok(n, add_special_tokens=False).input_ids
                    if not ids_r or not ids_n:
                        continue
                    if any((i not in allowed_ids) for i in ids_r + ids_n):
                        continue
                lp_r = compute_option_logprob(model, tok, chat, r, allowed_ids if masked else None)
                lp_n = compute_option_logprob(model, tok, chat, n, allowed_ids if masked else None)
                vals.append(lp_r - lp_n)
        return float(sum(vals) / max(1, len(vals)))

    subset = kept[:min(20, len(kept))]
    soft_delta_base_unmasked = sum(avg_soft_delta(base, tok_b, p, allowed_b, masked=False) for p in subset) / max(1, len(subset))
    soft_delta_tuned_unmasked = sum(avg_soft_delta(tuned, tok_t, p, allowed_t, masked=False) for p in subset) / max(1, len(subset))
    soft_delta_base_masked = sum(avg_soft_delta(base, tok_b, p, allowed_b, masked=True) for p in subset) / max(1, len(subset))
    soft_delta_tuned_masked = sum(avg_soft_delta(tuned, tok_t, p, allowed_t, masked=True) for p in subset) / max(1, len(subset))

    out = {
        "n_prompts_freeform": len(kept),
        "n_mc_items_used": used,
        "refusal_rate_base": round(refusal_rate_base, 4),
        "refusal_rate_tuned": round(refusal_rate_tuned, 4),
        "refusal_rate_delta_pp": round(refusal_rate_delta_pp, 2),
        "refusal_rate_masked_base": round(refusal_rate_masked_base, 4),
        "refusal_rate_masked_tuned": round(refusal_rate_masked_tuned, 4),
        "refusal_rate_unmasked_base": round(refusal_rate_base, 4),
        "refusal_rate_unmasked_tuned": round(refusal_rate_tuned, 4),
        "n_mc_items_scored_masked": len(dlogp_masked),
        "n_mc_items_scored_unmasked": len(dlogp_unmasked),
        "mc_dlogp_correct_mean_masked": None if mc_masked_mean is None else round(mc_masked_mean, 4),
        "mc_dlogp_correct_mean_unmasked": None if mc_unmasked_mean is None else round(mc_unmasked_mean, 4),
        "min_shared_ratio": args.min_shared_ratio,
        "style_markers_per100_masked_base": round(style_per100_b_m, 3),
        "style_markers_per100_masked_tuned": round(style_per100_t_m, 3),
        "style_markers_per100_unmasked_base": round(style_per100_b, 3),
        "style_markers_per100_unmasked_tuned": round(style_per100_t, 3),
        "soft_refusal_delta_base_unmasked": round(soft_delta_base_unmasked, 4),
        "soft_refusal_delta_tuned_unmasked": round(soft_delta_tuned_unmasked, 4),
        "soft_refusal_delta_base_masked": round(soft_delta_base_masked, 4),
        "soft_refusal_delta_tuned_masked": round(soft_delta_tuned_masked, 4),
    }
    out_dir = os.path.join("mechdiff", "artifacts", "rq1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rq1_behavior.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if args.debug:
        print("[DEBUG] Freeform refusal samples (first 3):")
        for s in debug_samples:
            print(json.dumps(s, ensure_ascii=False))
        print(f"[DEBUG] MC used={used}, skipped={skipped}, masked_n={len(dlogp_masked)}, unmasked_n={len(dlogp_unmasked)}")
        print(f"[DEBUG] MC means: masked={out['mc_dlogp_correct_mean_masked']}, unmasked={out['mc_dlogp_correct_mean_unmasked']}")

    # Per-prompt refusal CSV
    with open(os.path.join(out_dir, "rq1_refusals.csv"), "w", encoding="utf-8") as f:
        f.write("prompt_idx,base_refusal,tuned_refusal\n")
        for i, (rb, rt) in enumerate(zip(refusals_base, refusals_tuned)):
            f.write(f"{i},{rb},{rt}\n")
    print("Saved mechdiff/artifacts/rq1/rq1_behavior.json and rq1_refusals.csv")


if __name__ == "__main__":
    main()


