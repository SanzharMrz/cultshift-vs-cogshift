#!/usr/bin/env python3
import os, glob, json, re
from pathlib import Path
from typing import Dict, Any, List, Tuple


def jread(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_full_mapped_ref(art_dir: str, layer: int = 26, hook: str = "attn_out") -> Tuple[str, Dict[str, Any]]:
    # Prefer files with 'full' in the name; otherwise pick max reduction
    cands = sorted(glob.glob(os.path.join(art_dir, f"mapped_patch_L{layer}_{hook}_*.json")))
    if not cands:
        return "", {}
    # First try explicit 'full'
    full_cands = [p for p in cands if "full" in os.path.basename(p)]
    if full_cands:
        path = full_cands[-1]
        return path, jread(path)
    # Else choose by maximal delta
    best = None
    best_delta = -1e9
    best_path = ""
    for p in cands:
        d = jread(p)
        kr = d.get("KL_raw_mean")
        km = d.get("KL_mapped_mean")
        if kr is None or km is None:
            continue
        delta = kr - km
        if delta > best_delta:
            best_delta = delta
            best = d
            best_path = p
    return best_path, (best or {})


def parse_k_from_head_mask_filename(path: str) -> int:
    name = os.path.basename(path)
    # Expected: head_mask_L{layer}_{hook}_{mask}.json, mask like '3' or '0-1-2' or 'ALL'
    m = re.search(r"head_mask_L\d+_\w+_(.+)\.json$", name)
    if not m:
        return -1
    mask = m.group(1)
    if mask.upper() == "ALL":
        return -1
    if mask.upper() == "NONE":
        return 0
    # count integer tokens
    items = [x for x in mask.split("-") if x.strip()]
    # Only count entries that look like integers
    cnt = 0
    for x in items:
        try:
            int(x)
            cnt += 1
        except Exception:
            continue
    return cnt


def summarize_rq4(art_dir: str = "mechdiff/artifacts/rq4", layer: int = 26, hook: str = "attn_out") -> None:
    art_dir = str(art_dir)
    Path(art_dir).mkdir(parents=True, exist_ok=True)

    # Full reference
    ref_path, ref = find_full_mapped_ref(art_dir, layer=layer, hook=hook)
    if not ref:
        print("No mapped FULL reference found in", art_dir)
        return
    kr = ref.get("KL_raw_mean", 0.0) or 0.0
    km = ref.get("KL_mapped_mean", 0.0) or 0.0
    full_delta = kr - km
    print(f"FULL mapped @ L{layer}/{hook}: raw={kr:.4f} mapped={km:.4f} Δ={full_delta:.4f}  ({ref_path})")

    # Head-masked runs
    head_files = sorted(glob.glob(os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")))
    if not head_files:
        print("No head_mask runs found.")
        return
    by_k: Dict[int, List[Tuple[str, float, float, float]]] = {}
    for p in head_files:
        d = jread(p)
        kr_h = d.get("KL_raw_mean")
        km_h = d.get("KL_mapped_mean")
        if kr_h is None or km_h is None:
            continue
        delta = kr_h - km_h
        k = parse_k_from_head_mask_filename(p)
        cov = (delta / full_delta * 100.0) if full_delta != 0 else float("nan")
        by_k.setdefault(k, []).append((p, kr_h, km_h, cov))

    # Print aggregated summary
    print("\nCoverage vs FULL by k (top entries):")
    for k in sorted(by_k.keys()):
        rows = by_k[k]
        covs = [r[3] for r in rows if r[3] == r[3]]  # drop NaN
        if not covs:
            continue
        best_idx = max(range(len(rows)), key=lambda i: (rows[i][3] if rows[i][3] == rows[i][3] else -1e9))
        best = rows[best_idx]
        mean_cov = sum(covs) / len(covs)
        print(f"k={k:>2}: n={len(rows):<3} mean_cov={mean_cov:6.1f}%  best_cov={best[3]:6.1f}%  file={os.path.basename(best[0])}")

    # Optional single-head ranking cache
    single_cache = os.path.join(art_dir, f"single_head_L{layer}_{hook}_alpha0.3.json")
    if os.path.exists(single_cache):
        try:
            per = jread(single_cache)
            top = sorted(per, key=lambda x: (x.get("drop_pct") if x.get("drop_pct") == x.get("drop_pct") else -1e9), reverse=True)[:8]
            print("\nTop single heads by drop_pct (from cache):")
            for r in top:
                print(f"  head={r['head']:>2} drop_pct={r.get('drop_pct', float('nan')):6.2f}%")
        except Exception:
            pass


def main():
    summarize_rq4()


if __name__ == "__main__":
    main()


