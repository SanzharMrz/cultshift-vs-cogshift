import argparse
import csv
import glob
import json
import os
from typing import List, Tuple, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(pattern: str) -> List[Tuple[int, str, str, Optional[float], Optional[float], Optional[float], Optional[float]]]:
    paths = glob.glob(pattern)
    rows = []
    for p in paths:
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        L = d.get("layer")
        h = d.get("hook")
        a = d.get("alpha")
        kr = d.get("KL_raw_mean")
        km = d.get("KL_mapped_mean")
        delta = (kr - km) if (isinstance(kr, (int, float)) and isinstance(km, (int, float))) else None
        drop = (100.0 * (kr - km) / kr) if (isinstance(kr, (int, float)) and isinstance(km, (int, float)) and kr > 0) else None
        rows.append((int(L) if L is not None else -1, str(h), str(a), kr, km, delta, drop, os.path.basename(p)))
    # Sort by layer, hook, alpha (numeric if possible)
    def parse_alpha(x):
        try:
            return float(x)
        except Exception:
            return 1e9
    rows.sort(key=lambda r: (r[0], r[1], parse_alpha(r[2])))
    return rows


def write_text_summary(rows, out_txt: str, out_best: Optional[str] = None) -> None:
    ensure_dir(os.path.dirname(out_txt))
    w_name = max(40, max((len(r[-1]) for r in rows), default=40))
    lines = []
    lines.append("RQ3 — α sweep summary (val)\n")
    lines.append(f"{'file':<{w_name}}  layer hook        alpha   KL_raw   KL_mapped     Δ     drop%")
    for L, h, a, kr, km, delta, drop, fname in rows:
        kr_s = f"{kr:7.3f}" if isinstance(kr, (int, float)) else "    n/a"
        km_s = f"{km:9.3f}" if isinstance(km, (int, float)) else "      n/a"
        de_s = f"{delta:7.3f}" if isinstance(delta, (int, float)) else "    n/a"
        dr_s = f"{drop:6.1f}" if isinstance(drop, (int, float)) else "   n/a"
        lines.append(f"{fname:<{w_name}}  {L:>5} {h:<10} {a!s:>5}   {kr_s}   {km_s}   {de_s}  {dr_s}")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if out_best:
        # Best per (layer,hook) by drop%
        best = {}
        for r in rows:
            key = (r[0], r[1])
            if not isinstance(r[6], (int, float)):
                continue
            if key not in best or r[6] > best[key][6]:
                best[key] = r
        lines_b = []
        lines_b.append("Best per (layer,hook)\n")
        lines_b.append("layer hook        alpha   KL_raw   KL_mapped     Δ     drop%   file")
        for (L, h), r in sorted(best.items()):
            _, _, a, kr, km, delta, drop, fname = r
            lines_b.append(
                f"{L:>5} {h:<10} {a!s:>5}   {kr:7.3f}   {km:9.3f}   {delta:7.3f}  {drop:6.1f}  {fname}"
            )
        with open(out_best, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_b) + "\n")


def write_csv(rows, out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["layer", "hook", "alpha", "KL_raw_mean", "KL_mapped_mean", "delta", "drop_pct", "file"])
        for L, h, a, kr, km, delta, drop, fname in rows:
            wtr.writerow([L, h, a, kr, km, delta, drop, fname])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="mechdiff/artifacts/rq3/mapped_patch_L*_*.json,mechdiff/artifacts/rq3/ranks/mapped_patch_L*_*.json",
                    help="Comma-separated glob(s) for per-run JSONs")
    ap.add_argument("--out_txt", default="mechdiff/artifacts/rq3/rq3_summary.txt")
    ap.add_argument("--out_best", default="mechdiff/artifacts/rq3/rq3_best.txt")
    ap.add_argument("--out_csv", default="mechdiff/artifacts/rq3/rq3_combined.csv")
    args = ap.parse_args()

    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
    rows_all = []
    for pat in patterns:
        rows_all.extend(load_rows(pat))
    # de-duplicate by (layer, hook, alpha, file)
    seen = set()
    rows = []
    for r in rows_all:
        key = (r[0], r[1], r[2], r[-1])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    write_text_summary(rows, args.out_txt, args.out_best)
    write_csv(rows, args.out_csv)
    print(f"Wrote {len(rows)} rows to:\n- {args.out_txt}\n- {args.out_best}\n- {args.out_csv}")


if __name__ == "__main__":
    main()


