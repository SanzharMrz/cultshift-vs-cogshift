#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import re
import sys
from typing import Iterable, List, Optional, Set

from datasets import load_dataset
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm is unavailable
    def tqdm(x, **kwargs):
        return x

# Ensure repo root is on sys.path so local package imports work when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local imports
from mechdiff.utils.tokenizers import load_tokenizers, shared_vocab_maps
from mechdiff.utils.fairness import uses_only_shared


def load_pair_dict(pair_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.I) for p in patterns]


def matches_any(text: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(text) is not None for p in pats)


def pick_text(example: dict) -> Optional[str]:
    # Try common fields for question/text in MC datasets
    for key in ("question", "text", "prompt", "instruction"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def pick_category(example: dict) -> str:
    for key in ("category", "topic", "subtopic"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def curate_prompts(
    include_patterns: List[str],
    exclude_patterns: List[str],
    max_count: int,
    pair_path: str,
    out_path: str,
    split: str = "train",
    min_shared_ratio: float = 1.0,
) -> int:
    pair = load_pair_dict(pair_path)
    base_id: str = pair["base_id"]
    tuned_id: str = pair["tuned_id"]

    ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split=split)

    inc = compile_patterns(include_patterns)
    exc = compile_patterns(exclude_patterns)

    print("Loading tokenizers and computing shared vocab...")
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    shared, _, _, allowed_b, allowed_t = shared_vocab_maps(tok_b, tok_t)

    kept: List[str] = []
    scanned = 0
    matched = 0
    for ex in tqdm(ds, desc="Scan+filter (category→fairness)", unit="ex", total=len(ds) if hasattr(ds, "__len__") else None):
        scanned += 1
        cat = pick_category(ex)
        text = pick_text(ex)
        if not text:
            continue
        cat_str = f"{cat} {text}"
        if inc and not matches_any(cat_str, inc):
            continue
        if exc and matches_any(cat_str, exc):
            continue
        matched += 1
        ids_b = tok_b(text, add_special_tokens=False).input_ids
        ids_t = tok_t(text, add_special_tokens=False).input_ids
        if not ids_b or not ids_t:
            continue
        shared_b = sum(1 for i in ids_b if i in allowed_b)
        shared_t = sum(1 for j in ids_t if j in allowed_t)
        ratio_b = shared_b / max(1, len(ids_b))
        ratio_t = shared_t / max(1, len(ids_t))
        if ratio_b >= min_shared_ratio and ratio_t >= min_shared_ratio:
            kept.append(text)
            if len(kept) >= max_count:
                break
    print(f"Scanned: {scanned}; Category-matched: {matched}; Fairness-kept: {len(kept)} (min_shared_ratio={min_shared_ratio})")
    print(f"Fairness-kept: {len(kept)}")

    # Trim strictly to fairness-surviving prompts only (no backfill)
    final: List[str] = kept[:max_count]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in final:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    return len(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cultural.py")
    ap.add_argument("--out", default="mechdiff/data/prompts_freeform_kk.jsonl")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--split", default="train")
    ap.add_argument("--min_shared_ratio", type=float, default=1.0, help="Keep prompts where both tokenizers have ≥ this fraction of tokens in shared vocab")
    ap.add_argument(
        "--include",
        nargs="*",
        default=[
            r"tradit", r"custom", r"culture", r"etiquette", r"family", r"community",
            r"holiday", r"ritual", r"heritage", r"norm", r"social",
        ],
        help="Regexes to include by category/topic/text",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[r"relig", r"faith", r"islam", r"church", r"mosque", r"quran"],
        help="Regexes to exclude (religion etc.)",
    )
    args = ap.parse_args()

    n = curate_prompts(
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_count=args.max,
        pair_path=args.pair,
        out_path=args.out,
        split=args.split,
        min_shared_ratio=args.min_shared_ratio,
    )
    print(f"Wrote {n} prompts to {args.out}")


if __name__ == "__main__":
    main()


