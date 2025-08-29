#!/usr/bin/env python3
import os, glob, json, argparse, shlex, subprocess, re


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


def build_topk(art_dir: str, layer: int, hook: str, pair: str, split: str, alpha: float,
               ks=(1, 2, 4, 8)) -> None:
    # Denominator from explicit ALL file
    all_path = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_ALL.json")
    if not os.path.exists(all_path):
        raise FileNotFoundError(f"Missing ALL file: {all_path}. Run head_mask_patch with --head_mask ALL first.")
    KL_full = float(jread(all_path).get("KL_raw_mean") or 0.0)
    if KL_full <= 0:
        raise RuntimeError("KL_full <= 0 from ALL; cannot compute coverage.")

    # Collect single-head results
    patt = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_*.json")
    files = sorted(glob.glob(patt))
    singles = []
    for p in files:
        mask = parse_mask_from_name(p)
        if not (isinstance(mask, list) and len(mask) == 1):
            continue
        d = jread(p)
        KL_sel = d.get("KL_mapped_mean")
        if not isinstance(KL_sel, (int, float)):
            continue
        cov = float(KL_sel) / KL_full * 100.0
        singles.append((mask[0], float(KL_sel), cov, p))
    if not singles:
        raise RuntimeError("No single-head files found to rank.")

    # Rank by KL_sel descending (equivalently coverage)
    singles.sort(key=lambda t: t[1], reverse=True)

    # Build and run top-k masks
    for k in ks:
        heads = [h for h, _, _, _ in singles[:k]]
        mask_csv = ",".join(str(h) for h in heads)
        out_expected = os.path.join(art_dir, f"head_mask_L{layer}_{hook}_{'-'.join(str(h) for h in heads)}.json")
        if os.path.exists(out_expected):
            print(f"[SKIP] exists: {out_expected}")
            continue
        cmd = (
            f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.cognitive.rq4.head_mask_patch "
            f"--pair {pair} --layer {layer} --hook {hook} --split {split} "
            f"--alpha {alpha} --head_mask {mask_csv}"
        )
        print(">>", cmd)
        rc = subprocess.call(cmd, shell=True)
        if rc != 0:
            raise RuntimeError(f"head_mask_patch failed for k={k} (rc={rc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cognitive.py")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--hook", default="attn_out")
    ap.add_argument("--split", default="val")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--art_dir", default="mechdiff/artifacts/cognitive/rq4")
    ap.add_argument("--k_list", default="1,2,4,8")
    args = ap.parse_args()

    ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    build_topk(args.art_dir, args.layer, args.hook, args.pair, args.split, float(args.alpha), ks)


if __name__ == "__main__":
    main()


