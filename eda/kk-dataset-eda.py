#!/usr/bin/env python
"""
Quick EDA for kz-transformers/kk-socio-cultural-bench-mc.

What it does:
- Loads the dataset via datasets.load_dataset
- Prints splits, row counts, column names and detected feature types
- Shows a compact pretty-print of a few sample rows
- Detects likely category fields and prints their frequency distribution
- Detects likely question/options/answer fields and prints a formatted example

Usage (from repo root):
  venv/bin/python eda/kk-dataset-eda.py

Optional args:
  --name DATASET_NAME (default: kz-transformers/kk-socio-cultural-bench-mc)
  --samples N          (default: 3) number of sample rows to pretty-print
  --max-value-len N    (default: 180) truncate long values when printing
  --topk N             (default: 30) number of top categories to show
"""

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for kk-socio-cultural-bench-mc")
    parser.add_argument(
        "--name",
        default="kz-transformers/kk-socio-cultural-bench-mc",
        help="HF dataset name",
    )
    parser.add_argument("--samples", type=int, default=3, help="Number of sample rows to print")
    parser.add_argument(
        "--max-value-len",
        type=int,
        default=180,
        help="Truncate long values to this many characters",
    )
    parser.add_argument("--topk", type=int, default=30, help="Top-K category values to display")
    return parser.parse_args()


def safe_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except Exception as exc:
        print("ERROR: Could not import 'datasets'. Install it in your venv:", file=sys.stderr)
        print("       venv/bin/pip install datasets", file=sys.stderr)
        raise


def truncate_value(value: Any, max_len: int) -> str:
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def summarize_splits(ds) -> None:
    print("Splits and sizes:")
    for split_name, split in ds.items():
        try:
            print(f"- {split_name}: {len(split):,} rows")
        except Exception:
            print(f"- {split_name}: <unknown size>")
    print()


def summarize_schema(ds) -> None:
    # Use the first split as reference for schema
    split_name = next(iter(ds.keys()))
    d = ds[split_name]
    print(f"Columns for split '{split_name}': {d.column_names}")
    # Try to get feature types, if available
    features = getattr(d, "features", None)
    if features is not None:
        print("Feature types:")
        try:
            for key in d.column_names:
                print(f"- {key}: {features[key]}")
        except Exception:
            # Fallback generic print
            print(features)
    print()


def detect_candidate_fields(column_names: List[str]) -> Dict[str, List[str]]:
    candidates = {
        "category": [
            "category",
            "subject",
            "subcategory",
            "topic",
            "domain",
            "section",
            "theme",
            "field",
            "source",
            "origin",
            "area",
            "category_name",
        ],
        "question": ["question", "prompt", "stem"],
        "options": ["options", "choices", "variants", "answers"],
        "answer": [
            "answer",
            "label",
            "correct",
            "target",
            "answer_idx",
            "correct_option",
        ],
    }
    present: Dict[str, List[str]] = {k: [] for k in candidates}
    for kind, names in candidates.items():
        present[kind] = [n for n in names if n in column_names]
    return present


def print_sample_rows(ds, samples: int, max_value_len: int) -> None:
    split_name = next(iter(ds.keys()))
    d = ds[split_name]
    n = min(samples, len(d))
    print(f"Sample {n} row(s) from split '{split_name}':")
    for i in range(n):
        row = d[i]
        print(f"- idx={i}")
        for key in d.column_names:
            val = row.get(key)
            if isinstance(val, list):
                # Show length and first few elements compactly
                preview_items = ", ".join(truncate_value(v, 60) for v in val[:5])
                suffix = "" if len(val) <= 5 else ", ..."
                s = f"list(len={len(val)}): [" + preview_items + suffix + "]"
            elif isinstance(val, dict):
                preview_items = ", ".join(f"{k}={truncate_value(v, 40)}" for k, v in list(val.items())[:5])
                suffix = "" if len(val) <= 5 else ", ..."
                s = f"dict({preview_items}{suffix})"
            else:
                s = truncate_value(val, max_value_len)
            print(f"    {key}: {s}")
    print()


def collect_all_rows(ds) -> Iterable[Dict[str, Any]]:
    for split in ds.values():
        for row in split:
            yield row


def count_categories(ds, candidate_fields: List[str]) -> List[Tuple[str, List[Tuple[str, int]]]]:
    results: List[Tuple[str, List[Tuple[str, int]]]] = []
    all_rows = list(collect_all_rows(ds))
    for field in candidate_fields:
        counter: Counter = Counter()
        for row in all_rows:
            val = row.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                # If the field is a list of tags/categories, count each item
                for item in val:
                    if item is None:
                        continue
                    counter[str(item)] += 1
            else:
                counter[str(val)] += 1
        if counter:
            sorted_counts = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            results.append((field, sorted_counts))
    return results


def print_category_distributions(dist: List[Tuple[str, List[Tuple[str, int]]]], topk: int) -> None:
    if not dist:
        print("No candidate category fields found.")
        print()
        return
    print("Candidate category distributions:")
    for field, counts in dist:
        print(f"- Field '{field}': top {min(topk, len(counts))}")
        for value, cnt in counts[:topk]:
            print(f"    {value}: {cnt}")
    print()


def print_formatted_example(ds, question_fields: List[str], options_fields: List[str], answer_fields: List[str]) -> None:
    split_name = next(iter(ds.keys()))
    d = ds[split_name]
    if len(d) == 0:
        return
    row = d[0]
    q_key = question_fields[0] if question_fields else None
    o_key = options_fields[0] if options_fields else None
    a_key = answer_fields[0] if answer_fields else None

    print("Formatted example (first row):")
    if q_key:
        print(f"Question: {row.get(q_key)}")
    if o_key:
        opts = row.get(o_key)
        if isinstance(opts, list):
            for idx, opt in enumerate(opts):
                print(f"  {chr(ord('A') + idx)}. {opt}")
        else:
            print(f"Options: {opts}")
    if a_key:
        print(f"Answer field '{a_key}': {row.get(a_key)}")
    print()


def main() -> None:
    args = parse_args()
    load_dataset = safe_import_datasets()

    # If user set HF token via env, datasets will pick it up. Otherwise, loading may fail if the dataset is gated.
    try:
        ds = load_dataset(args.name)
    except Exception as exc:
        print("ERROR: Failed to load dataset. If access is required, run:", file=sys.stderr)
        print("  huggingface-cli login", file=sys.stderr)
        print("Or set the env var HUGGINGFACE_TOKEN and retry.", file=sys.stderr)
        raise

    print(ds)
    print()
    summarize_splits(ds)
    summarize_schema(ds)
    print_sample_rows(ds, samples=args.samples, max_value_len=args.max_value_len)

    # Detect candidate fields and compute distributions
    first_split = next(iter(ds.keys()))
    columns = ds[first_split].column_names
    present = detect_candidate_fields(columns)
    print("Detected candidate fields:")
    for kind, fields in present.items():
        print(f"- {kind}: {fields}")
    print()

    cat_dists = count_categories(ds, present.get("category", []))
    print_category_distributions(cat_dists, topk=args.topk)

    print_formatted_example(ds, present.get("question", []), present.get("options", []), present.get("answer", []))


if __name__ == "__main__":
    main()


