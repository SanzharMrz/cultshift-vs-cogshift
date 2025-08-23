import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


def apply_chat(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def count_content_tokens(tok, text: str) -> int:
    sids = set(getattr(tok, "all_special_ids", []) or [])
    enc = tok(text, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, "tolist"):
        ids = ids if isinstance(ids, list) else ids.tolist()
    return sum(1 for i in ids if i not in sids)


def get_content_positions(tok, text: str) -> List[int]:
    enc = tok(text, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, "tolist"):
        ids = ids if isinstance(ids, list) else ids.tolist()
    sids = set(getattr(tok, "all_special_ids", []) or [])
    return [idx for idx, i in enumerate(ids) if i not in sids]


def harvest_prompts(seed: int, include: List[str], exclude: List[str]) -> List[str]:
    ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split="train")
    inc = [re.compile(p, flags=re.I) for p in include] if include else []
    exc = [re.compile(p, flags=re.I) for p in exclude] if exclude else []

    def pick_text(ex: dict) -> str:
        for key in ("question", "text", "prompt", "instruction"):
            v = ex.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def pick_category(ex: dict) -> str:
        for key in ("category", "topic", "subtopic"):
            v = ex.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    prompts: List[str] = []
    scanned = 0
    matched = 0
    for ex in ds:
        scanned += 1
        q = pick_text(ex)
        ctx = ex.get("context")
        if isinstance(ctx, str) and ctx.strip():
            stem = (ctx.strip() + "\n\n" + q).strip()
        else:
            stem = q
        if not stem:
            continue
        cat = pick_category(ex)
        cat_str = f"{cat} {stem}"
        if inc and not any(p.search(cat_str) for p in inc):
            continue
        if exc and any(p.search(cat_str) for p in exc):
            continue
        matched += 1
        prompts.append(stem)
    # dedupe exact string
    prompts = list({p: True for p in prompts}.keys())
    random.Random(seed).shuffle(prompts)
    print(f"Scanned={scanned}, Category-matched={matched}, Unique-after-match={len(prompts)}")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include", nargs="*", default=[
        r"tradit", r"custom", r"culture", r"etiquette", r"family", r"community",
        r"holiday", r"ritual", r"heritage", r"norm", r"social",
    ])
    ap.add_argument("--exclude", nargs="*", default=[r"relig", r"faith", r"islam", r"church", r"mosque", r"quran"])
    ap.add_argument("--n_total", type=int, default=0, help="If >0, limit total prompts before split")
    ap.add_argument("--n_train", type=int, default=0, help="If >0, set train count explicitly")
    ap.add_argument("--n_val", type=int, default=0, help="If >0, set val count explicitly")
    args = ap.parse_args()
    # Tokenizers for filtering and positions
    tok_base = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=False)
    tok_tuned = AutoTokenizer.from_pretrained("inceptionai/Llama-3.1-Sherkala-8B-Chat", use_fast=False)

    # Step 1: harvest unique prompts, filter by >=25 content tokens under chat-format (base tok)
    raw = harvest_prompts(seed=args.seed, include=args.include, exclude=args.exclude)
    kept: List[str] = []
    for p in raw:
        chat = apply_chat(tok_base, p)
        if count_content_tokens(tok_base, chat) >= 25:
            kept.append(p)

    # Optional limit before split
    if args.n_total and args.n_total > 0:
        kept = kept[: args.n_total]

    # grouped split by exact string with optional explicit sizes
    random.Random(0).shuffle(kept)
    if args.n_train and args.n_val:
        n_train = min(args.n_train, len(kept))
        n_val = min(args.n_val, max(0, len(kept) - n_train))
        train_prompts = kept[:n_train]
        val_prompts = kept[n_train:n_train + n_val]
    else:
        n_train = int(0.8 * len(kept))
        train_prompts = kept[:n_train]
        val_prompts = kept[n_train:]
    assert not (set(train_prompts) & set(val_prompts)), "Train/val overlap detected"

    out_dir = os.path.join("mechdiff", "data", "rq2")
    os.makedirs(out_dir, exist_ok=True)
    tr_path = os.path.join(out_dir, "train_prompts.jsonl")
    va_path = os.path.join(out_dir, "val_prompts.jsonl")
    with open(tr_path, "w", encoding="utf-8") as f:
        for p in train_prompts:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    with open(va_path, "w", encoding="utf-8") as f:
        for p in val_prompts:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")

    print(f"Saved {tr_path} ({len(train_prompts)}) and {va_path} ({len(val_prompts)})")

    # Step 2: pre-sample token positions per prompt
    def sample_positions_for_prompt(p: str) -> Tuple[List[int], int, int]:
        chat_b = apply_chat(tok_base, p)
        chat_t = apply_chat(tok_tuned, p)
        pos_b = get_content_positions(tok_base, chat_b)
        pos_t = get_content_positions(tok_tuned, chat_t)
        T = min(len(pos_b), len(pos_t))
        if T <= 1:
            return [], T, 0
        # Exclude last 4 tokens; sample from content-index space [1, max(2, T-4))
        hi = max(2, T - 4)
        choices = list(range(1, hi))
        rng = random.Random(0)
        rng.shuffle(choices)
        K_target = 16
        sel = sorted(choices[: min(K_target, len(choices))])
        return sel, T, len(sel)

    stats = {"T_min": 10**9, "T_max": 0, "K_sum": 0, "N": 0, "T_lt8": 0}

    def build_positions(prompts: List[str]) -> Dict[str, List[int]]:
        d: Dict[str, List[int]] = {}
        for p in prompts:
            pos, T, K_assigned = sample_positions_for_prompt(p)
            if T > 0 and pos:
                d[p] = pos
                stats["T_min"] = min(stats["T_min"], T)
                stats["T_max"] = max(stats["T_max"], T)
                stats["K_sum"] += K_assigned
                stats["N"] += 1
                if T < 8:
                    stats["T_lt8"] += 1
        return d

    train_pos = build_positions(train_prompts)
    val_pos = build_positions(val_prompts)

    with open(os.path.join(out_dir, "train_positions.json"), "w", encoding="utf-8") as f:
        json.dump(train_pos, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "val_positions.json"), "w", encoding="utf-8") as f:
        json.dump(val_pos, f, ensure_ascii=False, indent=2)

    avg_K = (stats["K_sum"] / max(1, stats["N"]))
    pct_Tlt8 = 100.0 * stats["T_lt8"] / max(1, stats["N"]) 
    print(f"Positions saved. N_prompts={stats['N']} avg_K={avg_K:.2f} T_min={stats['T_min']} T_max={stats['T_max']} %T<8={pct_Tlt8:.2f}")
    # write stats file
    with open(os.path.join(out_dir, "positions_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "N_prompts": stats["N"],
            "avg_K": avg_K,
            "T_min": stats["T_min"],
            "T_max": stats["T_max"],
            "%T<8": pct_Tlt8,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


