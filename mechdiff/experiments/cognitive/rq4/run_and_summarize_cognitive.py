#!/usr/bin/env python3
import os, glob, json, subprocess, shlex, random, torch
from pathlib import Path

PAIR = "mechdiff/pairs/pair_cognitive.py"
VAL_SPLIT = "val"

# Use the cognitive artifact root
ART_ROOT = Path("mechdiff/artifacts/cognitive")
ART_RQ2  = ART_ROOT / "rq2"
ART_RQ4  = ART_ROOT / "rq4"; ART_RQ4.mkdir(parents=True, exist_ok=True)

# MAIN layer: strongest cognitive effect
LAYER_MAIN = 30
# CONTROL: early/weak layer (use 10 if you want a cleaner control)
LAYER_CTRL = 10

# Two hooks:
HOOK_FULL = "resid_post"   # where the full mapped patch is applied (k=full map)
HOOK_MASK = "attn_out"     # where head-masked raw substitutions happen

ALPHA_FULL = 1.0           # use bundle's scale or 1.0; we stick to 1.0 for clarity
K_LIST = [1,2,4,8]         # top-k masks
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
    # cognitive RQ2 maps, prefer highest CKA
    patt = f"{ART_RQ2}/rq2_clt_L{layer}_*{hook}*procrustes_scaled*.json"
    cands = sorted(glob.glob(patt))
    best = None; best_key = -1e9
    for j in reversed(cands):
        try:
            d = jread(j)
        except Exception:
            continue
        if d.get("layer") != layer:           continue
        if d.get("hook")  != hook:            continue
        if d.get("solver")!= "procrustes_scaled": continue
        # We want FULL rank (pca_q==0)
        if int(d.get("pca_q") or 0) != 0:     continue
        mp = d.get("map_path")
        if not mp or not os.path.exists(mp):  continue
        try:
            bundle = torch.load(mp, map_location="cpu")
            if bundle.get("layer") != layer or bundle.get("solver") != "procrustes_scaled":
                continue
        except Exception:
            continue
        cka = d.get("cka_mapped_vs_tuned_val")
        key = float(cka) if isinstance(cka, (int,float)) else 0.0
        if key >= best_key:
            best = d; best_key = key
    return best

def heads(model_cfg_json: str = "mechdiff/artifacts/model_cfg.json") -> int:
    try:
        d = jread(model_cfg_json)
        return int(d.get("num_attention_heads", 32))
    except Exception:
        return 32

def mask_str(idxs):
    return ",".join(str(i) for i in idxs) if idxs else "NONE"

def run_full_mapped(layer: int):
    # FULL reference at resid_post using full-rank map
    dmap = latest_map_json(layer, HOOK_FULL)
    assert dmap and dmap.get("map_path"), f"No map JSON for L{layer}/{HOOK_FULL}"
    map_pt = dmap["map_path"]
    out_name = ART_RQ4 / f"mapped_patch_L{layer}_{HOOK_FULL}_full.json"
    if out_name.exists():
        return jread(out_name)
    cmd = (
      f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch "
      f"--pair {PAIR} --layer {layer} --hook {HOOK_FULL} --k1_decision --split {VAL_SPLIT} "
      f"--alpha {ALPHA_FULL} --map_file {shlex.quote(map_pt)} --device cuda"
    )
    rc = run(cmd);  assert rc == 0, "mapped-patch failed"
    newest = sorted(glob.glob(f"{ART_RQ2}/mapped_patch_L{layer}_{HOOK_FULL}_*.json"))[-1]
    os.replace(newest, out_name)
    return jread(out_name)

def run_head_mask(layer: int, heads_csv: str, tag: str):
    out = ART_RQ4 / f"head_mask_L{layer}_{HOOK_MASK}_{tag}.json"
    if out.exists():
        return jread(out)
    cmd = (
      f"{shlex.quote(os.sys.executable)} -m mechdiff.experiments.cognitive.rq4.head_mask_patch "
      f"--pair {PAIR} --layer {layer} --hook {HOOK_MASK} --k1_decision "
      f"--split {VAL_SPLIT} --alpha 1.0 --head_mask {heads_csv}"
    )
    rc = run(cmd);  assert rc == 0, "head_mask_patch failed"
    pattern_cog = f"{ART_ROOT}/rq4/head_mask_L{layer}_{HOOK_MASK}_*.json"
    pattern_root = f"mechdiff/artifacts/rq4/head_mask_L{layer}_{HOOK_MASK}_*.json"
    cands = sorted(glob.glob(pattern_cog) + glob.glob(pattern_root))
    assert cands, f"No head_mask outputs found for L{layer}/{HOOK_MASK}"
    newest = cands[-1]
    os.replace(newest, out)
    return jread(out)

def pct_drop(d: dict) -> float:
    kr = d.get("KL_raw_mean"); km = d.get("KL_mapped_mean")
    if kr is None or km is None or kr == 0: return float("nan")
    return (kr - km) / max(1e-9, kr) * 100.0

def main():
    H = heads()
    # 1) FULL denominator at resid_post
    ref = run_full_mapped(LAYER_MAIN)
    full_delta = (ref.get("KL_raw_mean",0.0) - ref.get("KL_mapped_mean",0.0))
    print(f"[FULL] L{LAYER_MAIN}/{HOOK_FULL}: raw={ref.get('KL_raw_mean'):.4f} mapped={ref.get('KL_mapped_mean'):.4f} Δ={full_delta:.4f}")

    # 2) Single-head scan (attn_out)
    #    We’ll cache all single-head effects then form top-k vs random-k
    singles = []
    for h in range(H):
        d = run_head_mask(LAYER_MAIN, str(h), f"head{h}")
        delta = (d.get("KL_raw_mean",0.0) - d.get("KL_mapped_mean",0.0))
        singles.append((h, delta))
    singles.sort(key=lambda x: x[1], reverse=True)
    top_order = [h for h,_ in singles]

    # 3) Summaries
    print("\nTop-8 single heads by Δ (raw head-mask at attn_out):")
    for h,delta in singles[:8]:
        cov = 100.0*delta/max(1e-9, full_delta)
        print(f"  head={h:>2}  Δ={delta:7.3f}  cover={cov:6.1f}%")

    # 4) top-k vs random-k evaluations
    for k in [1,2,4,8]:
        hk = mask_str(top_order[:k])
        d_top = run_head_mask(LAYER_MAIN, hk, f"top{k}")
        dt = (d_top.get("KL_raw_mean",0.0) - d_top.get("KL_mapped_mean",0.0))
        cov_t = 100.0*dt/max(1e-9, full_delta)
        print(f"\n[k={k}] TOP: Δ={dt:.3f} cover={cov_t:.1f}%  heads={top_order[:k]}")
        for s in RAND_SEEDS:
            random.seed(s)
            rmask = sorted(random.sample(range(H), k))
            d_r = run_head_mask(LAYER_MAIN, mask_str(rmask), f"rand{k}_s{s}")
            dr = (d_r.get("KL_raw_mean",0.0) - d_r.get("KL_mapped_mean",0.0))
            cov_r = 100.0*dr/max(1e-9, full_delta)
            print(f"       RAND{s}: Δ={dr:.3f} cover={cov_r:.1f}%  heads={rmask}")

    # 5) Control layer FULL (should be ≈0)
    try:
        ref_ctrl = run_full_mapped(LAYER_CTRL)
        d_ctrl = (ref_ctrl.get("KL_raw_mean",0.0) - ref_ctrl.get("KL_mapped_mean",0.0))
        print(f"\n[CTRL] L{LAYER_CTRL}/{HOOK_FULL} FULL: Δ={d_ctrl:.3f}  (expect ~0)")
    except AssertionError:
        print(f"\n[CTRL] No full map found for L{LAYER_CTRL}/{HOOK_FULL}. Skipping.")

if __name__ == "__main__":
    main()
