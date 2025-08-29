#!/usr/bin/env python3
import os, glob, json, argparse, re


def jread(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_mask_from_name(path: str):
    name = os.path.basename(path)
    m = re.search(r"head_mask_L\d+_\w+_(.+)\.json$", name)
    if not m:
        return None
    token = m.group(1)
    up = token.upper()
    if up == "ALL":
        return "ALL"
    if up == "NONE":
        return []
    heads = []
    for t in token.split("-"):
        t = t.strip()
        if not t:
            continue
        try:
            heads.append(int(t))
        except Exception:
            pass
    return sorted(heads)


def summarize_cognitive_rq4(art_dir: str, layer: int, hook: str) -> None:
    patt = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(patt))
    if not files:
        print(f"No head_mask files at {patt}")
        return
    # Denominator: prefer explicit ALL file; fallback to max KL_raw across files
    all_file = None
    for p in files:
        if os.path.basename(p).endswith("_ALL.json"):
            all_file = p
            break
    if all_file is not None:
        ref = jread(all_file)
        KL_full = float(ref.get("KL_raw_mean") or 0.0)
        print(f"FULL RAW (ALL heads) @ L{layer}/{hook}: KL_raw(full)={KL_full:.6f}  ({os.path.basename(all_file)})")
    else:
        KL_full = 0.0
        for p in files:
            d = jread(p)
            kr = d.get("KL_raw_mean")
            if isinstance(kr, (int, float)):
                KL_full = max(KL_full, float(kr))
        print(f"[WARN] No _ALL file found. Using max KL_raw across files as FULL: {KL_full:.6f}")
        if KL_full == 0.0:
            print("No usable denominator (KL_full == 0).")
            return

    # Compute coverage% = 100 * KL_selected / KL_full
    rows = []
    for p in files:
        mask = parse_mask_from_name(p)
        if mask == "ALL":
            continue
        d = jread(p)
        KL_sel = d.get("KL_mapped_mean")
        if not isinstance(KL_sel, (int, float)):
            continue
        k = len(mask) if isinstance(mask, list) else 0
        cov = (float(KL_sel) / max(1e-12, KL_full)) * 100.0
        rows.append((k, os.path.basename(p), float(KL_sel), cov))

    # Top single heads
    singles = [r for r in rows if r[0] == 1]
    singles.sort(key=lambda r: r[3], reverse=True)
    if singles:
        print("\nTop single-head masks by coverage% (KL_selected / KL_full):")
        for k, fname, kls, cov in singles[:16]:
            print(f"  {fname:<38} KL_sel={kls:9.6f}  cover={cov:6.2f}%")

    # Best per k
    by_k = {}
    for k, fname, kls, cov in rows:
        if k <= 0:
            continue
        by_k.setdefault(k, []).append((fname, kls, cov))
    if by_k:
        print("\nBest coverage per k:")
        for k in sorted(by_k.keys()):
            fname, kls, cov = max(by_k[k], key=lambda x: x[2])
            print(f"  k={k:>2}  cover={cov:6.2f}%  KL_sel={kls:9.6f}  file={fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_dir", default="mechdiff/artifacts/cognitive/rq4")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--hook", default="attn_out")
    ap.add_argument("--journal", action="store_true")
    args = ap.parse_args()
    summarize_cognitive_rq4(args.art_dir, args.layer, args.hook)


if __name__ == "__main__":
    main()

