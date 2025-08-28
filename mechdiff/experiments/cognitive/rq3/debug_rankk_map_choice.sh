#!/usr/bin/env bash
set -euo pipefail

# Tiny debugger to verify which CLT map JSON and map_path are selected per rank k
# Mirrors the selection logic in run_cognitive_rankk.sh

LAYERS=( ${LAYERS:-"24 26 30"} )
HOOKS=( ${HOOKS:-"resid_post"} )
K_LIST=( ${K_LIST:-"1 8 32 0"} )  # 0 = full

for L in "${LAYERS[@]}"; do
  for H in "${HOOKS[@]}"; do
    echo "=== L=$L H=$H ==="
    for K in "${K_LIST[@]}"; do
      echo "-- K=$K --"
      INFO=$(python - "$L" "$H" "$K" <<'PY'
import json,glob,os,sys
L,H,K=sys.argv[1],sys.argv[2],sys.argv[3]
cands=sorted(glob.glob(f"mechdiff/artifacts/cognitive/rq2/rq2_clt_L{L}_*procrustes_scaled*.json"), reverse=True)
print(f"CANDIDATES={len(cands)}")
chosen=None
for j in cands:
    try:
        d=json.load(open(j))
    except Exception:
        continue
    if str(d.get("layer"))==L and d.get("hook")==H and int(d.get("pca_q") or 0)==int(K):
        mp=d.get("map_path")
        print(f"META={j}")
        print(f"PCA_Q={d.get('pca_q')}")
        print(f"MAP={mp if mp else ''}")
        chosen=j
        break
if not chosen:
    print("META=")
    print("PCA_Q=")
    print("MAP=")
PY
)
      CANDS=$(echo "$INFO" | awk -F= '/^CANDIDATES=/{print $2}')
      META=$(echo "$INFO"  | awk -F= '/^META=/{print $2}')
      PCAQ=$(echo "$INFO"  | awk -F= '/^PCA_Q=/{print $2}')
      MAP=$(echo "$INFO"   | awk -F= '/^MAP=/{print $2}')
      echo "candidates=$CANDS"
      echo "meta_json=$META"
      echo "pca_q=$PCAQ"
      echo "map_path=$MAP"
      if [[ -n "$MAP" && -f "$MAP" ]]; then
        SIZE=$(stat -c%s "$MAP" 2>/dev/null || echo "-")
        MD5=$(md5sum "$MAP" 2>/dev/null | awk '{print $1}')
        echo "map_exists=true size=$SIZE md5=$MD5"
      else
        echo "map_exists=false"
      fi
    done
  done
done

echo "Done."


