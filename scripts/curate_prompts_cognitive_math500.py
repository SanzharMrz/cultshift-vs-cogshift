#!/usr/bin/env python3
import argparse, json, random, re
from pathlib import Path
from datasets import load_dataset

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
NUM_RE = re.compile(r"^\s*-?\d+(?:/\d+)?(?:\.\d+)?\s*$")  # int | fraction | decimal


def extract_final(sol: str):
    sol = (sol or "").strip()
    m = BOX_RE.findall(sol)
    cand = (m[-1] if m else "").strip()
    if cand and NUM_RE.match(cand):  # prefer boxed numeric
        return cand
    # fallback: last numeric-looking token at line end
    for line in reversed(sol.splitlines()):
        t = line.strip().rstrip(".")
        if NUM_RE.match(t):
            return t
    return None


def prompt_text(q: str) -> str:
    return (
        "Solve the problem and give ONLY the final numeric answer.\n\n"
        + q.strip()
        + "\n\nFinal answer:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n_train", type=int, default=350)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument(
        "--subjects",
        default="algebra,number_theory,geometry,combinatorics",
        help="Comma-separated subject names to keep (subset of dataset labels).",
    )
    ap.add_argument(
        "--min_len",
        type=int,
        default=200,
        help="Min solution length to keep (reasoning-heavy).",
    )
    ap.add_argument("--out_dir", default="mechdiff/data/cognitive_math500")
    args = ap.parse_args()
    random.seed(args.seed)

    keep_subjects = {s.strip().lower() for s in args.subjects.split(",") if s.strip()}

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for r in ds:
        q = (r.get("problem") or "").strip()
        sol = (r.get("solution") or "").strip()
        subj = (r.get("subject") or r.get("type") or "").strip().lower()
        lvl = r.get("level") or r.get("difficulty") or ""
        if keep_subjects and subj not in keep_subjects:
            continue
        if len(sol) < args.min_len:
            continue
        final = extract_final(sol)
        if not final:
            continue
        rows.append(
            {
                "text": prompt_text(q),
                "label": final,
                "meta": {"src": "math500", "subject": subj, "level": lvl, "len_sol": len(sol)},
            }
        )

    if not rows:
        raise SystemExit("No rows after filtering. Loosen --subjects or --min_len.")

    # simple balanced sampling by subject
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["meta"]["subject"], []).append(r)
    for v in by_subj.values():
        random.shuffle(v)

    def take(n):
        bucketed = []
        subs = list(by_subj.keys())
        i = 0
        while len(bucketed) < n and any(by_subj[s] for s in subs):
            s = subs[i % len(subs)]
            if by_subj[s]:
                bucketed.append(by_subj[s].pop())
            i += 1
        return bucketed

    train = take(min(args.n_train, len(rows)))
    val = take(min(args.n_val, len(rows) - len(train)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("train_prompts.jsonl", train), ("val_prompts.jsonl", val)]:
        with open(out / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train / {len(val)} val → {out}")


if __name__ == "__main__":
    main()


