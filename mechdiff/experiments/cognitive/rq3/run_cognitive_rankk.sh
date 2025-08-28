#!/usr/bin/env bash
set -euo pipefail

PAIR="mechdiff/pairs/pair_cognitive.py"
DEVICE="cuda"
OUT_DIR="mechdiff/artifacts/cognitive/rq3/ranks"
mkdir -p "$OUT_DIR"

# Layers and hooks to probe
LAYERS=("24" "26" "30")
HOOKS=("resid_post")          # add "mlp_out" for L26 if you want
K_LIST=(1 8 32 0)             # 0 = full (no PCA truncation)
ALPHAS_RESID=("0.8" "1.0" "1.2")
ALPHAS_MLP=("0.8" "1.0")      # narrower sweep is fine

for L in "${LAYERS[@]}"; do
  for H in "${HOOKS[@]}"; do
    # Pick alpha grid per hook
    if [[ "$H" == "resid_post" ]]; then ALPHAS=("${ALPHAS_RESID[@]}"); else ALPHAS=("${ALPHAS_MLP[@]}"); fi

    for K in "${K_LIST[@]}"; do
      echo "[CLT] Train map: L=$L hook=$H pca_q=$K → mechdiff/artifacts/cognitive/rq2"
      python -m mechdiff.experiments.cognitive.rq2.run_rq2_clt \
        --pair "$PAIR" --layer "$L" --hook "$H" \
        --k1_decision --solver procrustes_scaled --pca "$K" \
        --shrink 0.05 --device "$DEVICE"

      # Fetch the matching map bundle (pass variables via argv to avoid env issues)
      MAP_PT=$(python - "$L" "$H" "$K" <<'PY'
import json,glob,os,sys
L,H,K=sys.argv[1],sys.argv[2],sys.argv[3]
cands=sorted(glob.glob(f"mechdiff/artifacts/cognitive/rq2/rq2_clt_L{L}_*procrustes_scaled*.json"), reverse=True)
for j in cands:
    try:
        d=json.load(open(j))
    except Exception:
        continue
    if str(d.get("layer"))==L and d.get("hook")==H and int(d.get("pca_q") or 0)==int(K):
        mp=d.get("map_path"); 
        if mp and os.path.exists(mp):
            print(mp); sys.exit(0)
print("")
PY
)
      if [[ -z "${MAP_PT}" ]]; then
        echo "[WARN] no map found for L=$L H=$H K=$K; candidates:"
        ls -1t mechdiff/artifacts/cognitive/rq2/rq2_clt_L${L}_*procrustes_scaled*.json 2>/dev/null || true
        continue
      fi

      NAME_K="$K"; [[ "$K" == "0" ]] && NAME_K="full"
      for A in "${ALPHAS[@]}"; do
        echo "[RQ3] map->patch: L=$L H=$H k=$NAME_K alpha=$A → mechdiff/artifacts/cognitive/rq2"
        python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
          --pair "$PAIR" --layer "$L" --hook "$H" --k1_decision \
          --split val --map_file "$MAP_PT" --alpha "$A" --device "$DEVICE"

        SRC="mechdiff/artifacts/cognitive/rq2/mapped_patch_L${L}_${H}_val.json"
        DST="${OUT_DIR}/mapped_patch_L${L}_${H}_k${NAME_K}_alpha${A}.json"
        [[ -f "$SRC" ]] && mv "$SRC" "$DST"
      done
    done
  done
done

# Aggregate a small table
python - <<'PY'
import glob,json,os
rows=[]
for p in sorted(glob.glob("mechdiff/artifacts/cognitive/rq3/ranks/mapped_patch_*.json")):
    d=json.load(open(p))
    kr,km=d.get("KL_raw_mean"),d.get("KL_mapped_mean")
    drop=None if (kr is None or km is None or kr==0) else 100*(kr-km)/kr
    rows.append((os.path.basename(p), kr, km, drop))
w=max(40,max((len(r[0]) for r in rows), default=40))
print(f"{'file':<{w}}  {'KL_raw':>8}  {'KL_map':>8}  {'drop%':>7}")
for name,kr,km,dp in rows:
    dps="   n/a" if dp is None else f"{dp:7.1f}"
    krs="   n/a" if kr is None else f"{kr:8.3f}"
    kms="   n/a" if km is None else f"{km:8.3f}"
    print(f"{name:<{w}}  {krs}  {kms}  {dps}")
PY
