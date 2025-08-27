#!/usr/bin/env python3
import os, glob, json, subprocess, shlex, random
import torch
from pathlib import Path

PAIR = "mechdiff/pairs/pair_cultural.py"
VAL_SPLIT = "val"
ART = Path("mechdiff/artifacts/rq4"); ART.mkdir(parents=True, exist_ok=True)

LAYER_MAIN = 26
LAYER_CTRL = 24
HOOK = "attn_out"
ALPHA = 0.3
K_LIST = [1,2,4,8]
RAND_SEEDS = [0,1,2]


def run(cmd: str) -> int:
    print(">>", cmd)
    return subprocess.call(cmd, shell=True)


def jread(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def jwrite(obj, p: Path):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def latest_map_json(layer: int, hook: str):
    # Gather candidates and filter strictly to avoid mismatches
    cands = sorted(glob.glob(f"mechdiff/artifacts/rq2/rq2_clt_L{layer}_*{hook}*procrustes_scaled*.json"))
    if not cands:
        cands = sorted(glob.glob(f"mechdiff/artifacts/rq2/rq2_clt_L{layer}_*procrustes_scaled*.json"))
    best = None
    best_key = -1e9
    for j in reversed(cands):
        try:
            d = jread(j)
        except Exception:
            continue
        # Strict filters
        if d.get("layer") != layer:
            continue
        if d.get("hook") != hook:
            continue
        if d.get("solver") != "procrustes_scaled":
            continue
        if d.get("k_positions") != 1:
            continue
        if d.get("pca_q", 0) not in (0, None):
            continue
        if str(d.get("logit_subspace", "0")) != "0":
            continue
        map_pt = d.get("map_path")
        if not map_pt or not os.path.exists(map_pt):
            continue
        # Optional: verify the .pt content matches (layer/hook/solver)
        try:
            bundle = torch.load(map_pt, map_location="cpu")
            if bundle.get("layer") != layer or bundle.get("solver") != "procrustes_scaled":
                continue
            # hook may not be saved in older bundles; skip if present and mismatched
            if bundle.get("hook") not in (None, hook):
                continue
        except Exception:
            continue
        # Scoring key: prefer higher CKA; fallback to recency if missing
        cka = d.get("cka_mapped_vs_tuned_val")
        key = float(cka) if isinstance(cka, (int, float)) else 0.0
        if key >= best_key:
            best = d
            best_key = key
    return best


def heads(model_cfg_json: str = "mechdiff/artifacts/model_cfg.json") -> int:
    try:
        d = jread(model_cfg_json)
        return int(d.get("num_attention_heads", 32))
    except Exception:
        return 32


def pct_drop(d: dict) -> float:
    kr = d.get("KL_raw_mean")
    km = d.get("KL_mapped_mean")
    if kr is None or km is None or kr == 0:
        return float("nan")
    return (kr - km) / max(1e-9, kr) * 100.0


def mask_str(idxs):
    return ",".join(str(i) for i in idxs) if idxs else "NONE"


def run_full_mapped(layer: int, hook: str, alpha: float, tag: str):
    dmap = latest_map_json(layer, hook)
    assert dmap and dmap.get("map_path"), f"No map JSON for L{layer}/{hook}"
    map_pt = dmap["map_path"]
    out_name = ART / f"mapped_patch_L{layer}_{hook}_{tag}.json"
    if out_name.exists():
        return jread(out_name)
    cmd = f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.rq2.run_rq2_mapped_patch " \
          f"--pair {PAIR} --layer {layer} --hook {hook} --k1_decision --split {VAL_SPLIT} " \
          f"--alpha {alpha} --map_file {shlex.quote(map_pt)}"
    rc = run(cmd)
    if rc != 0:
        raise RuntimeError(f"mapped-patch failed: {cmd}")
    newest = sorted(glob.glob("mechdiff/artifacts/rq2/mapped_patch_*.json"))[-1]
    os.replace(newest, out_name)
    return jread(out_name)


def topk_from_single_head(layer: int, hook: str, alpha: float, k: int):
    H = heads()
    single_path = ART / f"single_head_L{layer}_{hook}_alpha{alpha}.json"
    if single_path.exists():
        per = jread(str(single_path))
    else:
        per = []
        for h in range(H):
            # Use raw head-masked patcher per head
            tag = f"head{h}_alpha{alpha}"
            out = ART / f"head_mask_L{layer}_{hook}_head{h}_alpha{alpha}.json"
            if not out.exists():
                cmd = f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.rq4.head_mask_patch --pair {PAIR} --layer {layer} --hook {hook} --k1_decision --split {VAL_SPLIT} --alpha {alpha} --head_mask {h}"
                rc = run(cmd)
                if rc != 0:
                    raise RuntimeError("head_mask_patch (single head) failed")
                newest = sorted(glob.glob(f"mechdiff/artifacts/rq4/head_mask_L{layer}_{hook}_*.json"))[-1]
                os.replace(newest, out)
            d = jread(out)
            per.append({"head": h, "drop_pct": pct_drop(d), "KL_raw": d.get("KL_raw_mean"), "KL_mapped": d.get("KL_mapped_mean")})
        jwrite(per, single_path)
    per = sorted(per, key=lambda x: (x["drop_pct"] if x["drop_pct"] == x["drop_pct"] else -1e9), reverse=True)
    return [p["head"] for p in per[:k]], per


def main():
    H = heads()
    results = []

    # FULL reference uses mapped patch, no head masking
    res_full = run_full_mapped(LAYER_MAIN, HOOK, ALPHA, f"full_alpha{ALPHA}")
    results.append({"layer": LAYER_MAIN, "hook": HOOK, "cond": "full", "alpha": ALPHA, **res_full})

    for k in K_LIST:
        topk, per = topk_from_single_head(LAYER_MAIN, HOOK, ALPHA, k)
        # Raw head-masked patch (pre o_proj) for top-k
        hk = mask_str(topk)
        # For raw head substitution, use alpha=1.0
        cmd_top = f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.rq4.head_mask_patch --pair {PAIR} --layer {LAYER_MAIN} --hook {HOOK} --k1_decision --split {VAL_SPLIT} --alpha 1.0 --head_mask {hk}"
        rc = run(cmd_top)
        if rc != 0:
            raise RuntimeError("head_mask_patch (top-k) failed")
        j_top = sorted(glob.glob(f"mechdiff/artifacts/rq4/head_mask_L{LAYER_MAIN}_{HOOK}_*.json"))[-1]
        d_top = jread(j_top)
        results.append({"layer": LAYER_MAIN, "hook": HOOK, "cond": f"top{k}", "alpha": ALPHA, "heads": topk, **d_top})
        for seed in RAND_SEEDS:
            random.seed(seed)
            rmask = sorted(random.sample(range(H), k))
            hk_r = mask_str(rmask)
            cmd_rand = f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.rq4.head_mask_patch --pair {PAIR} --layer {LAYER_MAIN} --hook {HOOK} --k1_decision --split {VAL_SPLIT} --alpha 1.0 --head_mask {hk_r}"
            rc = run(cmd_rand)
            if rc != 0:
                raise RuntimeError("head_mask_patch (rand-k) failed")
            j_rand = sorted(glob.glob(f"mechdiff/artifacts/rq4/head_mask_L{LAYER_MAIN}_{HOOK}_*.json"))[-1]
            d_rand = jread(j_rand)
            results.append({"layer": LAYER_MAIN, "hook": HOOK, "cond": f"rand{k}_s{seed}", "alpha": ALPHA, "heads": rmask, **d_rand})

    res_ctrl = run_full_mapped(LAYER_CTRL, HOOK, ALPHA, f"ctrl_L{LAYER_CTRL}_full_alpha{ALPHA}")
    results.append({"layer": LAYER_CTRL, "hook": HOOK, "cond": "ctrl_full", "alpha": ALPHA, **res_ctrl})

    out_json = ART / "rq4_results.json"
    jwrite(results, out_json)

    full_drop = (res_full.get("KL_raw_mean", 0.0) - res_full.get("KL_mapped_mean", 0.0))
    print("\n=== RQ4 Summary (ΔKL coverage vs FULL @ L26/attn_out) ===")
    print(f"FULL @ L26/attn_out: raw={res_full.get('KL_raw_mean'):.3f} mapped={res_full.get('KL_mapped_mean'):.3f} Δ={full_drop:.3f}")
    rows = []
    for r in results:
        if r["layer"] == LAYER_MAIN and r["hook"] == HOOK and r["cond"].startswith(("top", "rand")):
            delta = r.get("KL_raw_mean", 0.0) - r.get("KL_mapped_mean", 0.0)
            coverage = 100.0 * delta / max(1e-9, full_drop)
            rows.append((r["cond"], r.get("heads", []), delta, coverage))
    def k_of(c):
        try:
            s = c.split("_")[0]
            return int(s.replace("top", "").replace("rand", ""))
        except Exception:
            return 0
    rows.sort(key=lambda x: (k_of(x[0]), x[0]))
    w = max([len(x[0]) for x in rows] + [12])
    print(f"\n{'cond':<{w}}  {'ΔKL':>8}  {'coverage%':>10}  heads")
    for cond, heads_, delta, cov in rows:
        print(f"{cond:<{w}}  {delta:8.3f}  {cov:10.1f}  {heads_}")

    d_ctrl = res_ctrl.get("KL_raw_mean", 0.0) - res_ctrl.get("KL_mapped_mean", 0.0)
    print(f"\nControl L24/attn_out (full): ΔKL={d_ctrl:.3f}  (expect ≤ 0 / not helpful)")


if __name__ == "__main__":
    main()


