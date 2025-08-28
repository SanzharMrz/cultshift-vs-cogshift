mkdir -p mechdiff/logs
nohup bash -c '
  for L in 26 24; do
    stdbuf -oL -eL python -m mechdiff.experiments.rq2.run_rq2_clt \
      --pair mechdiff/pairs/pair_cultural.py \
      --layer $L --hook attn_out --k1_decision \
      --solver procrustes_scaled --shrink 0.05 --alpha auto;
  done;
  stdbuf -oL -eL python mechdiff/experiments/rq4/run_and_summarize.py
' > mechdiff/logs/rq4_full_$(date +%F_%H%M%S).log 2>&1 &
echo $! > mechdiff/logs/rq4_full.pid