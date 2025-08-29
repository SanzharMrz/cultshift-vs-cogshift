## CultShift vs CogShift (LLM Mechanistic Differences)

### What this repo is
- Side-by-side study of cultural vs cognitive fine-tunes in LLMs.
- Focus: representation drift (CKA), causal effects (layer/heads), and linear transport (CLT, rank-k).

### TL;DR findings
- Cultural: late-layer, low-rank, attention-centric (few heads can cover ~all effect).
- Cognitive (math): early–mid rep drift, late causal effects, high-rank/distributed; residual transport strong, attention transport weak.

### Repo layout
- `mechdiff/experiments/cultural/*` and `mechdiff/experiments/cognitive/*`: RQ1–RQ4 runners.
- `mechdiff/data/**`: curated prompts and positions.
- `mechdiff/artifacts/**`: results (JSON/CSV/figures), split by domain and RQ.
- `mechdiff/scripts/*`: orchestration and helpers.

### Run flow (high level)
1) Data curation (cognitive)
   - Prepare MATH-500 prompts + positions → `mechdiff/data/cognitive/rq2/`.
2) RQ1: Baselines & causal patch
   - Run CKA and layer-patch; save to `mechdiff/artifacts/cognitive/rq1/`.
3) RQ2: CLT + mapped-patch
   - Train Procrustes maps at late layers; validate on val; write to `mechdiff/artifacts/cognitive/rq2/`.
4) RQ3: Rank-k
   - Sweep PCA ranks (k∈{1,8,32,full}) and aggregate; write to `mechdiff/artifacts/cognitive/rq3/`.
5) RQ4: Head-level coverage
   - Head-mask at `attn_out`; summarize and plot to `mechdiff/artifacts/cognitive/rq4/`.

### Key scripts (pointers)
- Curation: `mechdiff/experiments/cognitive/prep_prompts_rq2.py` (or `scripts/curate_prompts_cognitive_math500.py`).
- RQ1: `mechdiff/experiments/cognitive/rq1/run_*`.
- RQ2: `mechdiff/experiments/cognitive/rq2/run_rq2_clt.py`, `run_rq2_mapped_patch.py`, plus `scripts/run_rq2_cognitive*.sh`.
- RQ3: `mechdiff/experiments/cognitive/rq3/run_cognitive_rankk.sh`, aggregator/plots under same folder.
- RQ4: `mechdiff/experiments/cognitive/rq4/run_and_summarize_cognitive.py`, `summarize_artifacts.py`, `plot_rq4.py`.

### Headline cognitive results (val)
- RQ1: early–mid CKA dip (min ≈0.56 @ L12); late causal effects (L24–L30).
- RQ2: residual maps at L24/L26/L30 drop KL by ~41–50%; attention maps hurt.
- RQ3: full-rank maps help; low-rank harms → distributed change.
- RQ4: best single head ≈25% of FULL; top-8 ≈77% → no single-head conduit.

### Requirements
- Python 3.10+, PyTorch, transformers, datasets, pandas, matplotlib, seaborn.


