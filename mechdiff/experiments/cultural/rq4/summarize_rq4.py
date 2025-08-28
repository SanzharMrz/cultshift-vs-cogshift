#!/usr/bin/env python3
import os, glob, json, re, math, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

ART_DIR = "mechdiff/artifacts/rq4"
LAYER = 26
HOOK = "attn_out"


def jread(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_full_raw_ref(art_dir: str, layer: int, hook: str) -> Tuple[str, Dict[str, Any]]:
    """Prefer head_mask=ALL as the FULL raw reference; fallback to max-Δ file."""
    cands = sorted(glob.glob(os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")))
    # 1) Strict: use ALL
    for p in cands:
        if os.path.basename(p).endswith("_ALL.json"):
            d = jread(p)
            kr, km = d.get("KL_raw_mean"), d.get("KL_mapped_mean")
            if kr is not None and km is not None:
                return p, d
    # 2) Fallback: take max Δ over any mask
    best, best_path, best_delta = None, "", -1e9
    for p in cands:
        d = jread(p)
        kr, km = d.get("KL_raw_mean"), d.get("KL_mapped_mean")
        if kr is None or km is None:
            continue
        delta = kr - km
        if delta > best_delta:
            best_delta, best, best_path = delta, d, p
    return best_path, (best or {})


def parse_k_from_name(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"head_mask_L\d+_\w+_(.+)\.json$", name)
    if not m:
        return -1
    mask = m.group(1)
    if mask.upper() == "ALL":
        return -1
    if mask.upper() == "NONE":
        return 0
    cnt = 0
    for x in mask.split("-"):
        x = x.strip()
        if not x:
            continue
        try:
            int(x)
            cnt += 1
        except Exception:
            pass
    return cnt


def print_journal_section():
    print("\n### RQ4 — Where does the transport live? (Head-level localization)\n")
    print("Setup: L26 / attn_out on the cultural pair. FULL raw reference replaces all heads (alpha=1.0) at the decision token. Coverage% = 100 * Δ_mask / Δ_full.")
    print("Expectations: single late heads (~24) ≈ 100% coverage; early bundle 0–7 ≈ 35–40%; top-2 (24–26) ≈ ~100%; L24 control small/negative.")


def summarize_rq4(art_dir: str = ART_DIR, layer: int = LAYER, hook: str = HOOK, show_journal: bool = False) -> None:
    Path(art_dir).mkdir(parents=True, exist_ok=True)

    # FULL raw reference (denominator)
    ref_path, ref = find_full_raw_ref(art_dir, layer, hook)
    if not ref:
        print("No FULL raw reference found (run head_mask_patch with --head_mask ALL).")
        return
    kr_full = ref.get("KL_raw_mean", 0.0) or 0.0
    km_full = ref.get("KL_mapped_mean", 0.0) or 0.0
    delta_full = kr_full - km_full
    # Choose denominator: normally use FULL Δ; if Δ≈0 (raw head-mask runs), fall back to raw KL
    use_raw_denominator = abs(delta_full) < 1e-12
    denom = delta_full if not use_raw_denominator else kr_full
    den_label = "Δ_full" if not use_raw_denominator else "KL_raw(full)"
    print(f"FULL RAW @ L{layer}/{hook}: raw={kr_full:.6f} mapped={km_full:.6f} Δ={delta_full:.6f}   ({os.path.basename(ref_path)}); denom={den_label}")
    if use_raw_denominator:
        print("\n[INFO] FULL Δ≈0 (expected for raw head-mask ALL). Using KL_raw(full) as coverage denominator.\n")

    # Collect all head-masked runs for this site
    head_files = sorted(glob.glob(os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")))
    rows: List[Tuple[int, str, float, float]] = []
    for p in head_files:
        d = jread(p)
        kr, km = d.get("KL_raw_mean"), d.get("KL_mapped_mean")
        if kr is None or km is None:
            continue
        delta = kr - km
        coverage = float("nan") if abs(denom) < 1e-12 else (delta / denom) * 100.0
        rows.append((parse_k_from_name(p), os.path.basename(p), delta, coverage))

    # Print sorted view (by k then by coverage desc)
    rows.sort(key=lambda r: (r[0], - (r[3] if r[3] == r[3] else -1e9)))
    print("\nCoverage vs FULL RAW by k (best per k):")
    seen_k = set()
    for k, fname, delta, cov in rows:
        if k in seen_k:
            continue
        seen_k.add(k)
        cov_str = f"{cov:6.1f}%" if cov == cov else "   n/a"
        print(f"k={k:>2}  Δ={delta:7.3f}  cover={cov_str}  file={fname}")

    # Optional: list a few notable single heads
    singles = [r for r in rows if r[0] == 1]
    singles.sort(key=lambda r: (r[3] if r[3] == r[3] else -1e9), reverse=True)
    if singles:
        print("\nTop single-head masks (by coverage%):")
        for k, fname, delta, cov in singles[:8]:
            print(f"  {fname:<40}  Δ={delta:7.3f}  cover={cov:6.1f}%")

    if show_journal:
        print_journal_section()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_dir", default=ART_DIR)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--hook", default=HOOK)
    ap.add_argument("--journal", action="store_true")
    args = ap.parse_args()
    summarize_rq4(args.art_dir, args.layer, args.hook, show_journal=args.journal)


if __name__ == "__main__":
    main()


