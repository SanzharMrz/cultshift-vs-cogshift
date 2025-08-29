mkdir -p mechdiff/logs
nohup bash -c '
    for L in 30 26 24; do
    python -m mechdiff.experiments.cognitive.rq2.run_rq2_clt \
        --pair mechdiff/pairs/pair_cognitive.py \
        --layer $L --hook resid_post \
        --k1_decision --solver procrustes_scaled --pca 0 \
        --shrink 0.05 --device cuda
  done;
  stdbuf -oL -eL python mechdiff/experiments/cognitive/rq4/run_and_summarize_cognitive.py
' > mechdiff/logs/rq4_cognitive_$(date +%F_%H%M%S).log 2>&1 &
echo $! > mechdiff/logs/rq4_cognitive.pid