#!/usr/bin/env bash
set -euo pipefail

# RQ2 (Cognitive) — Train CLT maps (train/val inside the runner), resid_post for all,
# and attn_out/mlp_out for late layers. Writes directly to mechdiff/artifacts/cognitive/rq2.

PAIR=${PAIR:-"mechdiff/pairs/pair_cognitive.py"}
LAYERS=${LAYERS:-"10,24,26,30"}

LOG_DIR="mechdiff/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%F_%H%M%S)
LOG="$LOG_DIR/rq2_cognitive_${TS}.log"
PID_FILE="$LOG_DIR/rq2_cognitive.pid"

echo "Launching RQ2 cognitive CLT in background (logs: $LOG)" | tee -a "$LOG"

nohup bash -c '
  set -euo pipefail
  echo "[START] $(date -Is) RQ2 cognitive CLT" 
  echo "PAIR=$PAIR" 
  echo "LAYERS=$LAYERS" 
  echo "Artifacts → mechdiff/artifacts/cognitive/rq2" 

  for L in ${LAYERS//,/ }; do
    echo "[CLT] L=${L} hook=resid_post"
    python -m mechdiff.experiments.cognitive.rq2.run_rq2_clt \
      --pair "$PAIR" --layer $L --hook resid_post \
      --k1_decision --solver procrustes_scaled --shrink 0.05 --alpha auto
    echo "[OK ] CLT L=${L} hook=resid_post"
  done

  for L in 26 30; do
    echo "[CLT] L=${L} hook=attn_out"
    python -m mechdiff.experiments.cognitive.rq2.run_rq2_clt \
      --pair "$PAIR" --layer $L --hook attn_out \
      --k1_decision --solver procrustes_scaled --shrink 0.05 --alpha auto
    echo "[OK ] CLT L=${L} hook=attn_out"

    echo "[CLT] L=${L} hook=mlp_out"
    python -m mechdiff.experiments.cognitive.rq2.run_rq2_clt \
      --pair "$PAIR" --layer $L --hook mlp_out \
      --k1_decision --solver procrustes_scaled --shrink 0.05 --alpha auto
    echo "[OK ] CLT L=${L} hook=mlp_out"
  done

  echo "[DONE] $(date -Is) RQ2 cognitive CLT finished."
' >"$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "Tail: tail -f $LOG"


