#!/usr/bin/env bash
set -euo pipefail

# Rank-k CLT + RQ3 alpha-sweeps, aggregation, and plots
# Outputs:
# - Per-run JSONs under mechdiff/artifacts/rq3/ranks/
# - Aggregated tables under mechdiff/artifacts/rq3/
# - Figures under mechdiff/artifacts/rq3/fig/

PAIR="${PAIR:-mechdiff/pairs/pair_cultural.py}"
# Ensure CUDA by default; override with DEVICE=cpu to force CPU
DEVICE="${DEVICE:-cuda}"
# Optional: control batch size for GPU activation collection in CLT
BATCH_SIZE="${BATCH_SIZE:-16}"
# Optional: control alpha grid globally (comma-separated)
ALPHAS_GLOBAL="${ALPHAS_GLOBAL:-0.3,0.5,0.7,1.0}"

# If CUDA_VISIBLE_DEVICES is unset and DEVICE=cuda, default to GPU:0
if [[ "${DEVICE}" == "cuda" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
RQ3_DIR="mechdiff/artifacts/rq3"
RQ3_RANKS_DIR="${RQ3_DIR}/ranks"
mkdir -p "${RQ3_RANKS_DIR}"

run_rankk_target() {
  local LAYER="$1"        # e.g., 26
  local HOOK="$2"         # e.g., attn_out | resid_post | mlp_out
  local ALPHAS_CSV="${3:-$ALPHAS_GLOBAL}"   # e.g., "0.3,0.5,0.7,1.0"

  IFS=',' read -r -a ALPHAS <<< "${ALPHAS_CSV}"

  for K in 1 8 32 0; do  # 0 = full (no PCA)
    echo "[CLT] Train map: L=${LAYER} hook=${HOOK} K=${K} (0=full)"
    python -m mechdiff.experiments.rq2.run_rq2_clt \
      --pair "${PAIR}" \
      --layer "${LAYER}" --hook "${HOOK}" \
      --k1_decision --solver procrustes_scaled --pca "${K}" --shrink 0.05 \
      --batch_size "${BATCH_SIZE}" --device "${DEVICE}"

    echo "[CLT] Fetch map_path for L=${LAYER} hook=${HOOK} K=${K}"
    MAP_PT=$(L_ENV="${LAYER}" H_ENV="${HOOK}" K_ENV="${K}" \
    python - <<'PY'
import json, glob, os
L = int(os.environ["L_ENV"])  # layer
H = os.environ["H_ENV"].strip()  # hook
K = int(os.environ["K_ENV"])  # pca_q (0=full)
matches = sorted(glob.glob(f"mechdiff/artifacts/rq2/rq2_clt_L{L}_*procrustes_scaled*.json"), reverse=True)
for j in matches:
    try:
        d = json.load(open(j))
    except Exception:
        continue
    if int(d.get("layer", -1)) == L and str(d.get("hook", "")) == H and int(d.get("pca_q") or 0) == K:
        mp = d.get("map_path", "")
        if mp:
            print(mp)
        break
PY
    )
    if [[ -z "${MAP_PT:-}" || ! -f "${MAP_PT}" ]]; then
      echo "[WARN] No map bundle found for L=${LAYER} hook=${HOOK} K=${K}; skipping"
      continue
    fi

    NAME_K="${K}"
    [[ "${K}" == "0" ]] && NAME_K="full"
    for A in "${ALPHAS[@]}"; do
      echo "[RQ3] α-sweep: L=${LAYER} hook=${HOOK} K=${NAME_K} alpha=${A}"
      python -m mechdiff.experiments.rq2.run_rq2_mapped_patch \
        --pair "${PAIR}" --layer "${LAYER}" --hook "${HOOK}" \
        --k1_decision --map_file "${MAP_PT}" --alpha "${A}" --device "${DEVICE}"

      SRC="mechdiff/artifacts/rq2/mapped_patch_L${LAYER}_val.json"
      DST="${RQ3_RANKS_DIR}/mapped_patch_L${LAYER}_${HOOK}_k${NAME_K}_alpha${A}.json"
      if [[ -f "${SRC}" ]]; then
        mv "${SRC}" "${DST}"
        echo "[RQ3] Saved ${DST}"
      else
        echo "[WARN] Expected ${SRC} not found"
      fi
    done
  done
}

# Targets from the detailed RQ3 task
run_rankk_target 26 attn_out "0.3,0.5,0.7,1.0"
run_rankk_target 24 resid_post "0.3,0.5,0.7,1.0"

# Aggregate and plot
python -m mechdiff.experiments.rq3.run_rq3_aggregate
python -m mechdiff.experiments.rq3.plot_rq3_results

echo "Done. Artifacts in: ${RQ3_DIR} and ${RQ3_RANKS_DIR}"


