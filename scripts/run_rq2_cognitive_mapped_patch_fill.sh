#!/usr/bin/env bash
set -euo pipefail

PAIR=${PAIR:-"mechdiff/pairs/pair_cognitive.py"}

echo "[RUN ] L26 resid_post (VAL)"
python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
  --pair "$PAIR" --layer 26 --hook resid_post --k1_decision --split val \
  --map_file "$(ls -1t mechdiff/artifacts/cognitive/rq2/rq2_clt_L26_*resid_post*procrustes_scaled*.json | head -n1)"

echo "[RUN ] L26 attn_out (VAL)"
python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
  --pair "$PAIR" --layer 26 --hook attn_out --k1_decision --split val \
  --map_file "$(ls -1t mechdiff/artifacts/cognitive/rq2/rq2_clt_L26_*attn_out*procrustes_scaled*.json | head -n1)"

echo "[RUN ] L30 resid_post (VAL)"
python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
  --pair "$PAIR" --layer 30 --hook resid_post --k1_decision --split val \
  --map_file "$(ls -1t mechdiff/artifacts/cognitive/rq2/rq2_clt_L30_*resid_post*procrustes_scaled*.json | head -n1)"

echo "[RUN ] L30 attn_out (VAL)"
python -m mechdiff.experiments.cognitive.rq2.run_rq2_mapped_patch \
  --pair "$PAIR" --layer 30 --hook attn_out --k1_decision --split val \
  --map_file "$(ls -1t mechdiff/artifacts/cognitive/rq2/rq2_clt_L30_*attn_out*procrustes_scaled*.json | head -n1)"

echo "[DONE] Filled mapped-patch gaps; results in mechdiff/artifacts/cognitive/rq2"


