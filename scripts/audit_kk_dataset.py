import os
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset, DatasetDict

# Save under mechdiff/artifacts to match repo layout
ART_DIR = os.path.join(os.path.dirname(__file__), "..", "mechdiff", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


def detect_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> List[str]:
    cats = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].nunique(dropna=True) <= max_unique:
            cats.append(col)
    return cats


def merge_splits(ds: DatasetDict) -> pd.DataFrame:
    frames = []
    for split_name, d in ds.items():
        df = d.to_pandas()
        df["__split__"] = split_name
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def is_cyrillic_char(ch: str) -> bool:
    cp = ord(ch)
    # Cyrillic: U+0400–U+04FF; add U+0500–U+052F (Cyrillic Supplement) just in case
    return (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F)


def is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)


def script_presence_ratio(texts: List[str]) -> Dict[str, float]:
    counts = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        has_cyr = any(is_cyrillic_char(ch) for ch in t)
        has_lat = any(is_latin_char(ch) for ch in t)
        if has_cyr and has_lat:
            counts["mixed"] += 1
        elif has_cyr:
            counts["cyrillic"] += 1
        elif has_lat:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values()) or 1
    return {k: 100.0 * v / total for k, v in counts.items()}


def value_counts_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False)
    total = vc.sum()
    out = (
        vc.rename("count")
        .to_frame()
        .assign(share=lambda x: (100.0 * x["count"] / total).round(2))
        .reset_index(names=[col])
    )
    return out


def cross_tab(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    tab = pd.crosstab(df[col_a], df[col_b], dropna=False)
    tab["__row_total__"] = tab.sum(axis=1)
    tab.loc["__col_total__"] = tab.sum(axis=0)
    # Also percentage by grand total
    grand = tab.loc["__col_total__", "__row_total__"]
    pct = (100.0 * tab / grand).round(2)
    pct = pct.add_suffix("_%")
    return pd.concat([tab, pct], axis=1)


def find_question_and_options_cols(df: pd.DataFrame) -> Tuple[str, List[str]]:
    # Heuristics for question stem
    q_candidates = [c for c in df.columns if c.lower() in {"question", "prompt", "stem", "instruction", "text"}]
    q_col = q_candidates[0] if q_candidates else df.columns[0]

    # Options as list field
    for name in ["options", "choices", "answer_choices", "answers"]:
        if name in df.columns and df[name].apply(lambda x: isinstance(x, (list, tuple))).any():
            return q_col, [name]

    # Options spread across columns
    opt_cols = [
        c for c in df.columns
        if any(c.lower().startswith(p) for p in ["option", "choice", "ans_"]) or c in list("ABCD")
    ]
    # Keep ordering stable
    opt_cols = sorted(opt_cols)
    return q_col, opt_cols[:4] if opt_cols else []


def sample_examples(df: pd.DataFrame, category_col: str, k_cat: int = 5, k_per: int = 3) -> None:
    cats = df[category_col].value_counts().head(k_cat).index.tolist()
    q_col, opt_cols = find_question_and_options_cols(df)
    print(f"\n=== Examples by {category_col} (top {k_cat}) ===")
    for cat in cats:
        sub = df[df[category_col] == cat].head(k_per)
        print(f"\n-- {category_col} = {cat} --")
        for _, row in sub.iterrows():
            q = row.get(q_col, "")
            print(f"Q: {q}")
            if opt_cols:
                if len(opt_cols) == 1 and isinstance(row.get(opt_cols[0]), (list, tuple)):
                    opts = row.get(opt_cols[0])
                    for i, o in enumerate(opts):
                        print(f"  - {chr(65+i)}. {o}")
                else:
                    for oc in opt_cols:
                        val = row.get(oc)
                        if pd.notna(val):
                            print(f"  - {oc}: {val}")


def main():
    ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc")
    split_sizes = {k: len(v) for k, v in ds.items()}
    total_items = sum(split_sizes.values())

    df_all = merge_splits(ds)

    # Detect categorical columns
    cat_cols = detect_categorical_columns(df_all)

    # Population tables per categorical field
    for col in cat_cols:
        tbl = value_counts_table(df_all, col)
        out_path = os.path.join(ART_DIR, f"counts_{col}.csv")
        tbl.to_csv(out_path, index=False)
        print(f"\n=== Counts for {col} (top 15) ===")
        print(tbl.head(15).to_string(index=False))

    # Crosstab if both category & subcategory present
    cat_like = [c for c in cat_cols if c.lower() in {"category", "subcategory", "topic", "domain"}]
    lower_map = {c.lower(): c for c in cat_cols}
    if "category" in lower_map and "subcategory" in lower_map:
        a, b = lower_map["category"], lower_map["subcategory"]
        ct = cross_tab(df_all, a, b)
        out_path = os.path.join(ART_DIR, "crosstab_cat_subcat.csv")
        ct.to_csv(out_path)
        print("\n=== Crosstab category × subcategory (top 15 rows) ===")
        print(ct.head(15).to_string())

    # Also provide category × answer crosstab when available
    if "category" in lower_map and "answer" in df_all.columns:
        a = lower_map["category"]
        b = "answer"
        ct2 = cross_tab(df_all, a, b)
        out_path2 = os.path.join(ART_DIR, "crosstab_category_answer.csv")
        ct2.to_csv(out_path2)
        print("\n=== Crosstab category × answer (top 15 rows) ===")
        print(ct2.head(15).to_string())

    # Script stats: use question-like text
    q_col, _ = find_question_and_options_cols(df_all)
    ratios = script_presence_ratio(df_all[q_col].astype(str).tolist())

    # Sample examples for top categories (prefer 'category' if available)
    category_col = lower_map.get("category") or (cat_cols[0] if cat_cols else None)
    if category_col:
        sample_examples(df_all, category_col)

    # Report basics
    print("\n=== Dataset summary ===")
    print(f"Total items: {total_items}")
    for split, n in split_sizes.items():
        print(f"  {split}: {n}")
    print("\n=== Script presence (% of items) ===")
    for k in ["cyrillic", "latin", "mixed", "other"]:
        if k in ratios:
            print(f"  {k}: {ratios[k]:.2f}%")


if __name__ == "__main__":
    main()
