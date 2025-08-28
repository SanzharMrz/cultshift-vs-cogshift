#!/usr/bin/env bash
set -euo pipefail

PAIR=${PAIR:-"mechdiff/pairs/pair_cognitive.py"}
LAYERS=${LAYERS:-"10,24,26,30"}

echo "[START] RQ2 cognitive mapped-patch (VAL only). Pair=$PAIR Layers=$LAYERS"
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
    if d.get("layer")==int(L) and d.get("hook")==hook and d.get("k_positions")==1:
        print(j); break
PY
}

eval_and_save () {
  L=$1; H=$2
  META=$(get_map_json $L $H)
  if [ -z "$META" ]; then
    echo "[SKIP] No CLT meta for L=$L hook=$H"
    return 0
  fi
  echo "[RUN ] mapped-patch VAL L=$L hook=$H"
  python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
    --pair "$PAIR" --layer $L --hook $H --k1_decision --split val \
    --map_file "$META"
  echo "[OK  ] mapped-patch VAL L=$L hook=$H → mechdiff/artifacts/cognitive/rq2"
}

for L in ${LAYERS//,/ }; do eval_and_save $L resid_post; done
for L in 26 30; do eval_and_save $L attn_out; eval_and_save $L mlp_out; done

echo "[DONE] RQ2 cognitive mapped-patch."
