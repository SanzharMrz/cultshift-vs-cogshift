#!/usr/bin/env bash
set -euo pipefail

# Grid-evaluate mapped-patch on VAL for (layer, hook) across alphas; pick best numeric.
# Writes results directly to mechdiff/artifacts/cognitive/rq2

PAIR=${PAIR:-"mechdiff/pairs/pair_cognitive.py"}
LAYERS=${LAYERS:-"10,24,26,30"}
ALPHAS=${ALPHAS:-"0.3,0.5,0.7,1.0"}

get_map_json () {
python - "$1" "$2" <<'PY'
import sys,glob,json,os
L, hook = sys.argv[1], sys.argv[2]
cands = sorted(glob.glob(f"mechdiff/artifacts/cognitive/rq2/rq2_clt_L{L}_*procrustes_scaled*.json"))
for j in reversed(cands):
    try:
        d = json.load(open(j))
    except Exception:
        continue
    if d.get("layer") == int(L) and d.get("hook") == hook and d.get("k_positions") == 1:
        print(j); break
PY
}

run_one () {
  L=$1; H=$2; META=$(get_map_json $L $H)
  if [ -z "$META" ]; then echo "[SKIP] No CLT meta for L=$L hook=$H"; return 0; fi
  echo "[RUN ] L=$L H=$H alphas=[$ALPHAS]"
  IFS=',' read -ra AARR <<< "$ALPHAS"
  for A in "${AARR[@]}"; do
    python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
      --pair "$PAIR" --layer $L --hook $H --k1_decision --split val \
      --alpha "$A" --map_file "$META" || true
  done
}

for L in ${LAYERS//,/ }; do run_one $L resid_post; done
for L in 26 30; do run_one $L attn_out; run_one $L mlp_out; done

echo "[DONE] RQ2 cognitive mapped-patch grid."


