import argparse
import glob
import json
import os
import shlex
import subprocess
from datetime import datetime
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_latest_map_json(layer: int, hook: str) -> str:
    """Find the most recent CLT JSON for the given layer whose bundle hook matches.

    Filenames produced by run_rq2_clt do not include the hook; we therefore:
    - list all procrustes_scaled JSONs for the layer
    - iterate newest-first, load JSON → map_path → bundle (.pt)
    - check bundle["hook"] equals requested hook
    - return the first match
    """
    pattern = f"mechdiff/artifacts/rq2/rq2_clt_L{layer}_*procrustes_scaled*.json"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CLT map JSONs found for L={layer}. Run run_rq2_clt first.")
    for path in reversed(matches):
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            map_path = meta.get("map_path")
            if not map_path or not os.path.exists(map_path):
                continue
            bundle = torch.load(map_path, map_location="cpu")
            b_hook = bundle.get("hook") if isinstance(bundle, dict) else None
            if b_hook == hook:
                return path
        except Exception:
            continue
    raise FileNotFoundError(f"No CLT map JSON with hook={hook} found for L={layer}. Train it first.")


def run_mapped_patch(pair: str, layer: int, hook: str, map_path: str, alpha: float, device: str, k1: bool) -> str:
    cmd = [
        "python",
        "-m",
        "mechdiff.experiments.rq2.run_rq2_mapped_patch",
        "--pair",
        pair,
        "--layer",
        str(layer),
        "--hook",
        hook,
        "--map_file",
        map_path,
        "--alpha",
        str(alpha),
        "--split",
        "val",
        "--device",
        device,
    ]
    if k1:
        cmd.append("--k1_decision")
    # Note: run_rq2_mapped_patch writes a fixed filename under artifacts/rq2
    subprocess.run(" ".join(shlex.quote(x) for x in cmd), shell=True, check=True)
    out_path = os.path.join("mechdiff", "artifacts", "rq2", f"mapped_patch_L{layer}_val.json")
    if not os.path.exists(out_path):
        raise FileNotFoundError("Expected mapped_patch output not found: " + out_path)
    return out_path


def save_per_run_copy_rq3(src_path: str, hook: str, alpha: float, k_dim: int = 0) -> dict:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    kr = float(data.get("KL_raw_mean")) if data.get("KL_raw_mean") is not None else None
    km = float(data.get("KL_mapped_mean")) if data.get("KL_mapped_mean") is not None else None
    drop_pct = None
    if kr is not None and km is not None and kr > 0:
        drop_pct = 100.0 * (kr - km) / kr
    copy = {
        "layer": int(data.get("layer")),
        "hook": hook,
        "alpha": float(alpha),
        "KL_raw_mean": kr,
        "KL_mapped_mean": km,
        "reduction": (kr - km) if (kr is not None and km is not None) else None,
        "drop_pct": drop_pct,
        "source_file": os.path.basename(src_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    # Save under ranks/ if a k-dimension is provided (>0), else in the main rq3 folder
    out_dir = os.path.join("mechdiff", "artifacts", "rq3", "ranks" if (k_dim and k_dim > 0) else "")
    ensure_dir(out_dir)
    suffix_k = f"_k{k_dim}" if (k_dim and k_dim > 0) else ""
    out_name = f"mapped_patch_L{copy['layer']}_{hook}{suffix_k}_alpha{alpha}.json"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(copy, f, ensure_ascii=False, indent=2)
    return copy


def write_summary(rows: list) -> None:
    out_dir = os.path.join("mechdiff", "artifacts", "rq3")
    ensure_dir(out_dir)
    # Sort by layer, hook, alpha
    def parse_alpha(a):
        try:
            return float(a)
        except Exception:
            return 1e9
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1], parse_alpha(r[2])))
    w_name = max(40, max((len(r[6]) for r in rows_sorted), default=40))
    lines = []
    lines.append("RQ3 — α sweep summary (val)\n")
    header = f"{'file':<{w_name}}  layer hook        alpha   KL_raw   KL_mapped     Δ     drop%"
    lines.append(header)
    for L, hook, alpha, kr, km, drop, fname in rows_sorted:
        delta = (kr - km) if (kr is not None and km is not None) else float('nan')
        drop_str = f"{drop:6.1f}" if drop is not None else "   n/a"
        lines.append(
            f"{fname:<{w_name}}  {L:>5} {hook:<10} {str(alpha):>5}   "
            f"{(kr if kr is not None else float('nan')):7.3f}   "
            f"{(km if km is not None else float('nan')):9.3f}   "
            f"{delta:7.3f}  {drop_str}"
        )
    with open(os.path.join(out_dir, "rq3_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cultural.py")
    ap.add_argument("--layers", default="24,26", help="Comma-separated layers; include 10 only for controls")
    ap.add_argument("--hooks", default="resid_post,attn_out,mlp_out", help="Comma-separated hooks")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,1.0", help="Comma-separated alpha values")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k1_decision", action="store_true")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    hooks = [x.strip() for x in args.hooks.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    rows = []
    for L in layers:
        for hook in hooks:
            try:
                meta_json = find_latest_map_json(L, hook)
            except FileNotFoundError as e:
                print(str(e))
                continue
            meta = json.load(open(meta_json, "r", encoding="utf-8"))
            map_path = meta.get("map_path")
            if not map_path or not os.path.exists(map_path):
                print(f"Missing map bundle at {map_path} for L={L} hook={hook}; skipping.")
                continue
            # Determine k (rank) if provided (via PCA setting) to diversify filenames
            k_dim = 0
            try:
                k_dim = int(meta.get("pca_q") or 0)
            except Exception:
                k_dim = 0
            for a in alphas:
                print(f"[RQ3] L={L} hook={hook} alpha={a} → mapped-patch (val)")
                src = run_mapped_patch(args.pair, L, hook, map_path, a, args.device, args.k1_decision)
                rec = save_per_run_copy_rq3(src, hook, a, k_dim=k_dim)
                drop = rec.get("drop_pct")
                rows.append((rec["layer"], hook, a, rec["KL_raw_mean"], rec["KL_mapped_mean"], drop, os.path.basename(src)))

    write_summary(rows)
    print("Saved RQ3 results to mechdiff/artifacts/rq3/")


if __name__ == "__main__":
    main()


