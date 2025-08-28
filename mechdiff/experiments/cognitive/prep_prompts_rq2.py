import argparse
import json
import os
import random
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

from datasets import load_dataset
from transformers import AutoTokenizer


def apply_chat(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def count_content_tokens(tok, text: str) -> int:
    enc = tok(text, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, "tolist"):
        ids = ids if isinstance(ids, list) else ids.tolist()
    sids = set(getattr(tok, "all_special_ids", []) or [])
    return sum(1 for i in ids if i not in sids)


def get_content_positions(tok, text: str) -> List[int]:
    enc = tok(text, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, "tolist"):
        ids = ids if isinstance(ids, list) else ids.tolist()
    sids = set(getattr(tok, "all_special_ids", []) or [])
    return [idx for idx, i in enumerate(ids) if i not in sids]


OP_RE = re.compile(r"[+\-*/^]")
BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _iter_gsm8k(split: str) -> Iterable[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    for r in ds:
        q = (r.get("question") or "").strip()
        s = (r.get("answer") or "").strip()
        if not q or not s:
            continue
        m = re.search(r"####\s*([\-]?\d+(?:\.\d+)?)\s*$", s)
        if not m:
            continue
        final = m.group(1)
        yield {
            "src": "gsm8k",
            "question": q,
            "solution": s,
            "final": final,
            "ops": len(OP_RE.findall(s)),
            "lines": s.count("\n") + 1,
            "len_sol": len(s),
        }


def _iter_compmath(split: str) -> Iterable[dict]:
    try:
        ds = load_dataset("hendrycks/competition_math", split=split)
    except Exception as e:
        print(f"[WARN] Could not load hendrycks/competition_math ({e}). Skipping this source.")
        return
    for i, r in enumerate(ds):
        prob = (r.get("problem") or "").strip()
        sol = (r.get("solution") or "").strip()
        if not prob or not sol:
            continue
        lvl = int(r.get("level") or 0)
        typ = str(r.get("type") or "").lower()
        boxes = BOX_RE.findall(sol)
        if not boxes:
            continue
        ans = boxes[-1].strip()
        if re.fullmatch(r"\-?\d+(?:/\d+)?", ans) is None:
            continue
        yield {
            "src": "compmath",
            "id": i,
            "question": prob,
            "solution": sol,
            "final": ans,
            "type": typ,
            "level": lvl,
            "len_sol": len(sol),
        }


def _iter_compmath_local(path_like: str) -> Iterable[dict]:
    """Yield Competition Math-like rows from a local jsonl file or directory of jsonl files.

    Expected fields per line: {"problem": str, "solution": str, "level": int|str, "type": str}
    Final answer extracted from last \\boxed{...} occurrence.
    """
    paths: List[str] = []
    p = Path(path_like)
    if p.is_dir():
        paths = sorted(glob.glob(str(p / "*.jsonl")))
    elif p.is_file():
        paths = [str(p)]
    else:
        print(f"[WARN] --cm_local path not found: {path_like}")
        return
    idx = 0
    for fp in paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    prob = (r.get("problem") or "").strip()
                    sol = (r.get("solution") or "").strip()
                    if not prob or not sol:
                        continue
                    boxes = BOX_RE.findall(sol)
                    if not boxes:
                        continue
                    ans = boxes[-1].strip()
                    if re.fullmatch(r"\-?\d+(?:/\d+)?", ans) is None:
                        continue
                    lvl = r.get("level");
                    try:
                        lvl = int(str(lvl).strip().lower().replace("level", "").strip())
                    except Exception:
                        lvl = 0
                    typ = str(r.get("type") or "").lower()
                    yield {
                        "src": "compmath_local",
                        "id": idx,
                        "question": prob,
                        "solution": sol,
                        "final": ans,
                        "type": typ,
                        "level": lvl,
                        "len_sol": len(sol),
                    }
                    idx += 1
        except Exception:
            # Skip unreadable files
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", type=str, default="math500", choices=["math500","legacy"],
                    help="Select data source: math500 (default) or legacy (GSM8K+CompMath)")
    ap.add_argument("--n_total", type=int, default=0, help="If >0, cap total prompts before split (applied after filtering)")
    ap.add_argument("--n_train", type=int, default=450)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--min_tokens", type=int, default=0, help="Min content tokens under chat template (sanity filter)")
    # Legacy (GSM8K+CompMath) knobs
    ap.add_argument("--gsm8k_min_ops", type=int, default=3)
    ap.add_argument("--gsm8k_min_lines", type=int, default=4)
    ap.add_argument("--gsm8k_min_len", type=int, default=200)
    ap.add_argument("--cm_levels", default="4,5")
    ap.add_argument("--cm_types", default="algebra,number_theory,combinatorics,geometry")
    ap.add_argument("--cm_local", type=str, default="", help="Optional path to local Competition Math jsonl or directory of jsonl files")
    # MATH-500 knobs
    ap.add_argument("--subjects", default="",
                    help="(math500) comma-separated subjects to keep")
    ap.add_argument("--math500_min_len", type=int, default=0, help="(math500) min solution length to keep")
    ap.add_argument("--out_dir", type=str, default=os.path.join("mechdiff", "data", "cognitive", "rq2"))
    args = ap.parse_args()

    # Tokenizers to mirror pair_cognitive
    tok_base = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=False)
    tok_tuned = AutoTokenizer.from_pretrained("nvidia/OpenMath2-Llama3.1-8B", use_fast=False)

    random.seed(args.seed)
    levels = {int(x) for x in str(args.cm_levels).split(",") if x}
    types = {x.strip().lower() for x in str(args.cm_types).split(",") if x}

    # 1) Harvest and filter
    def mk_prompt(q: str) -> str:
        # Use the bare question followed by the decision token anchor only.
        return (q or "").strip() + "\n\nFinal answer:"

    def keep_sufficient_tokens(q: str) -> bool:
        chat = apply_chat(tok_base, q)
        return count_content_tokens(tok_base, chat) >= int(args.min_tokens)

    rows_all: List[dict] = []
    if args.source == "math500":
        try:
            from datasets import load_dataset  # already imported
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        except Exception as e:
            raise SystemExit(f"Failed to load MATH-500: {e}")
        subj_keep = {s.strip().lower() for s in str(args.subjects).split(",") if s.strip()}
        box_re = re.compile(r"\\boxed\{([^}]*)\}")
        num_re = re.compile(r"^\s*-?\d+(?:/\d+)?(?:\.\d+)?\s*$")
        def extract_final(sol: str) -> str:
            sol = (sol or "").strip()
            m = box_re.findall(sol)
            if m:
                cand = (m[-1] or "").strip()
                return cand  # accept any boxed final, numeric or LaTeX
            # fallback: last non-empty line token (best-effort)
            for line in reversed(sol.splitlines()):
                t = line.strip().rstrip(".")
                if t:
                    return t
            return ""
        for r in ds:
            q = (r.get("problem") or "").strip()
            sol = (r.get("solution") or "").strip()
            subj = (r.get("subject") or r.get("type") or "").strip().lower()
            lvl = r.get("level") or r.get("difficulty") or ""
            if subj_keep and subj not in subj_keep:
                continue
            if len(sol) < int(args.math500_min_len):
                continue
            final = extract_final(sol)
            if not final:
                continue
            text = mk_prompt(q).strip()
            # Keep all by default; optionally enforce token length if min_tokens>0
            if int(args.min_tokens) > 0 and not keep_sufficient_tokens(text):
                continue
            rows_all.append({
                "text": text,
                "label": final,
                "meta": {"src": "math500", "subject": subj, "level": lvl, "len_sol": len(sol)},
            })
        # Use the whole curated dataset first, then split deterministically by seed
        random.Random(int(args.seed)).shuffle(rows_all)
        total = len(rows_all)
        # If user asked beyond available, fill train first then allocate remainder to val
        n_train_eff = min(int(args.n_train), total)
        rem = total - n_train_eff
        n_val_eff = min(int(args.n_val), max(0, rem))
        train_rows = rows_all[: n_train_eff]
        val_rows = rows_all[n_train_eff : n_train_eff + n_val_eff]
    else:
        # legacy path (GSM8K + CompMath)
        gsm = [r for r in _iter_gsm8k("train")
               if (r["ops"] >= args.gsm8k_min_ops and r["lines"] >= args.gsm8k_min_lines and r["len_sol"] >= args.gsm8k_min_len)]
        cm_all = [r for r in _iter_compmath("train")
                  if (r["level"] in levels) and (r["type"] in types) and (r["len_sol"] >= 200)]
        if not cm_all and args.cm_local:
            cm_all = [r for r in _iter_compmath_local(args.cm_local)
                      if (r["level"] in levels) and (r["type"] in types) and (r["len_sol"] >= 200)]
        def to_row(rec: dict) -> dict:
            t = mk_prompt(rec["question"]).strip()
            meta = {k: v for k, v in rec.items() if k not in ("question", "solution")}
            return {"text": t, "label": rec["final"], "meta": meta}
        gsm_rows = [to_row(r) for r in gsm if keep_sufficient_tokens(mk_prompt(r["question"]))]
        cm_rows = [to_row(r) for r in cm_all if keep_sufficient_tokens(mk_prompt(r["question"]))]
        def take_balanced(A: List[dict], B: List[dict], n: int) -> List[dict]:
            A = A[:]; B = B[:]
            random.shuffle(A); random.shuffle(B)
            half = n // 2
            sel_a = A[:half]; sel_b = B[:half]
            rest = A[half:] + B[half:]
            random.shuffle(rest)
            need = max(0, n - len(sel_a) - len(sel_b))
            return sel_a + sel_b + rest[:need]
        train_rows = take_balanced(gsm_rows, cm_rows, args.n_train)
        train_texts = {r["text"] for r in train_rows}
        gsm_val_pool = [r for r in gsm_rows if r["text"] not in train_texts]
        cm_val_pool = [r for r in cm_rows if r["text"] not in train_texts]
        val_rows = take_balanced(gsm_val_pool, cm_val_pool, args.n_val)

    # Optional cap after filtering (applies evenly to both sets)
    if args.n_total and args.n_total > 0:
        random.shuffle(train_rows)
        random.shuffle(val_rows)
        train_rows = train_rows[: min(args.n_total, len(train_rows))]
        val_rows = val_rows[: min(max(0, args.n_total - len(train_rows)), len(val_rows))]

    os.makedirs(args.out_dir, exist_ok=True)
    tr_path = os.path.join(args.out_dir, "train_prompts.jsonl")
    va_path = os.path.join(args.out_dir, "val_prompts.jsonl")
    with open(tr_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(va_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {tr_path} ({len(train_rows)}) and {va_path} ({len(val_rows)})")

    # 3) Pre-sample token positions per prompt (K up to 16, excluding trailing specials)
    def sample_positions_for_prompt(p: str) -> Tuple[List[int], int, int]:
        chat_b = apply_chat(tok_base, p)
        chat_t = apply_chat(tok_tuned, p)
        pos_b = get_content_positions(tok_base, chat_b)
        pos_t = get_content_positions(tok_tuned, chat_t)
        T = min(len(pos_b), len(pos_t))
        if T <= 1:
            return [], T, 0
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

    train_pos = build_positions([r["text"] for r in train_rows])
    val_pos = build_positions([r["text"] for r in val_rows])

    with open(os.path.join(args.out_dir, "train_positions.json"), "w", encoding="utf-8") as f:
        json.dump(train_pos, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "val_positions.json"), "w", encoding="utf-8") as f:
        json.dump(val_pos, f, ensure_ascii=False, indent=2)

    avg_K = (stats["K_sum"] / max(1, stats["N"]))
    pct_Tlt8 = 100.0 * stats["T_lt8"] / max(1, stats["N"]) 
    with open(os.path.join(args.out_dir, "positions_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "N_prompts": stats["N"],
            "avg_K": avg_K,
            "T_min": stats["T_min"],
            "T_max": stats["T_max"],
            "%T<8": pct_Tlt8,
        }, f, ensure_ascii=False, indent=2)
    print(f"Positions saved. N_prompts={stats['N']} avg_K={avg_K:.2f} T_min={stats['T_min']} T_max={stats['T_max']} %T<8={pct_Tlt8:.2f}")


if __name__ == "__main__":
    main()

