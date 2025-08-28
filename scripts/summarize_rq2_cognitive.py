#!/usr/bin/env python3
import glob, json, math, os
from pathlib import Path

OUT = Path("mechdiff/artifacts/cognitive/rq2")
OUT.mkdir(parents=True, exist_ok=True)

def loadj(p):
    try:
        with open(p,'r',encoding='utf-8') as f: return json.load(f)
    except Exception:
        return None

def best_by_drop(rows):
    """Pick best row per (L,hook) by max drop%, ignoring NaNs."""
    by = {}
    for r in rows:
        key=(r.get("layer"), r.get("hook"))
        dp=r.get("drop")
        if dp is None or (isinstance(dp,float) and math.isnan(dp)): continue
        if key not in by or dp > by[key]["drop"]:
            by[key]=r
    return by

def fmt(x):
    return "nan" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{x:.3f}"

def main():
    # CLT
    clt = []
    for j in glob.glob("mechdiff/artifacts/cognitive/rq2/rq2_clt_L*_procrustes_scaled_*.json"):
        d = loadj(j)
        if not d or d.get("k_positions")!=1: continue
        clt.append(dict(layer=d["layer"], hook=d["hook"], val_r2=d.get("val_r2"), cka=d.get("cka_mapped_vs_tuned_val"), cos=(d.get("cos_stats") or {}).get("mean")))

    # mapped-patch grid
    mp = []
    for j in glob.glob("mechdiff/artifacts/cognitive/rq2/mapped_patch_*.json"):
        d = loadj(j)
        if not d: continue
        kr, km = d.get("KL_raw_mean"), d.get("KL_mapped_mean")
        drop = (kr-km)/kr*100 if (isinstance(kr,(int,float)) and isinstance(km,(int,float)) and kr>0) else float('nan')
        mp.append(dict(layer=d.get("layer"), hook=d.get("hook"), alpha=d.get("alpha_used"), kr=kr, km=km, drop=drop, file=os.path.basename(j)))

    best = best_by_drop(mp)
    # Build table
    rows=[]
    for c in clt:
        key=(c["layer"], c["hook"]) 
        b = best.get(key)
        rows.append((c["layer"], c["hook"], c["val_r2"], c["cka"], c["cos"], None if not b else b["kr"], None if not b else b["km"], None if not b else b["drop"], None if not b else b["alpha"]))

    rows.sort(key=lambda r:(r[0], r[1]))
    hdr = f"{'L':>3}  {'hook':<10}  {'R2':>7}  {'CKA':>6}  {'cos':>6}  {'KL_raw':>8}  {'KL_map':>8}  {'drop%':>7}  {'alpha':>5}"
    lines=[hdr]
    for L,H,R2,CKA,COS,KR,KM,DP,A in rows:
        lines.append(f"{L:>3}  {H:<10}  {fmt(R2):>7}  {fmt(CKA):>6}  {fmt(COS):>6}  {fmt(KR):>8}  {fmt(KM):>8}  {fmt(DP):>7}  {fmt(A):>5}")

    outp = OUT / "summary.txt"
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved → {outp}")

if __name__ == "__main__":
    main()


