#!/usr/bin/env python3
import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime


def run(cmd: str) -> int:
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=False).returncode


def pct_drop(d: dict) -> float:
    kr = d.get("KL_raw_mean")
    km = d.get("KL_mapped_mean")
    if kr is None or km is None:
        return float("nan")
    return 100.0 * (kr - km) / max(1e-9, kr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cultural.py")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--hook", default="attn_out")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,1.0")
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    alphas = [a.strip() for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        print("No alphas provided.")
        sys.exit(1)

    results = []
    for a in alphas:
        # 1) Train/eval CLT with given alpha
        clt_cmd = [
            sys.executable,
            "-m",
            "mechdiff.experiments.rq2.run_rq2_clt",
            "--pair", args.pair,
            "--layer", str(args.layer),
            "--hook", args.hook,
            "--k1_decision",
            "--solver", "procrustes_scaled",
            "--shrink", str(args.shrink),
            "--alpha", str(a),
        ]
        if args.device:
            clt_cmd.extend(["--device", args.device])
        rc = 0 if args.dry else run(" ".join(shlex.quote(x) for x in clt_cmd))
        if rc != 0:
            print(f"CLT failed for alpha={a}")
            continue

        # 2) Load latest matching JSON for this layer and hook
        jsons = sorted(glob.glob(f"mechdiff/artifacts/rq2/rq2_clt_L{args.layer}_procrustes_scaled_*.json"))
        if not jsons:
            print("No CLT JSONs found.")
            continue
        clt_json = None
        for j in reversed(jsons):
            d = json.load(open(j))
            if d.get("hook") == args.hook:
                clt_json = j
                break
        if clt_json is None:
            clt_json = jsons[-1]
        dclt = json.load(open(clt_json))
        map_pt = dclt.get("map_path")
        if not map_pt or not os.path.exists(map_pt):
            print(f"map_path missing for alpha={a}: {map_pt}")
            continue

        # 3) Run mapped-patch
        mp_cmd = [
            sys.executable,
            "-m",
            "mechdiff.experiments.rq2.run_rq2_mapped_patch",
            "--pair", args.pair,
            "--layer", str(args.layer),
            "--hook", args.hook,
            "--k1_decision",
            "--map_file", map_pt,
        ]
        if args.device:
            mp_cmd.extend(["--device", args.device])
        rc = 0 if args.dry else run(" ".join(shlex.quote(x) for x in mp_cmd))
        if rc != 0:
            print(f"Mapped-patch failed for alpha={a}")
            continue

        # 4) Rename the mapped-patch JSON to include alpha
        src = f"mechdiff/artifacts/rq2/mapped_patch_L{args.layer}.json"
        if not os.path.exists(src):
            print(f"Expected mapped-patch output not found: {src}")
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = f"mechdiff/artifacts/rq2/mapped_patch_L{args.layer}_{args.hook}_alpha{a}_{ts}.json"
        os.replace(src, dst)
        dmp = json.load(open(dst))
        drop = pct_drop(dmp)
        results.append((a, drop, dst))
        print(f"alpha={a} => drop={drop:.1f}% -> {dst}")

    if results:
        print("\nSummary (sorted by drop desc):")
        for a, drop, path in sorted(results, key=lambda x: (x[1] if x[1]==x[1] else -1e9), reverse=True):
            print(f"alpha={a:<4}  drop={drop:6.1f}%  file={path}")
    else:
        print("No results recorded.")


if __name__ == "__main__":
    main()


