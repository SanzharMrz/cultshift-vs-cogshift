# Culture Shift vs Cognitive Shift

## Changelog

### 2025-08-23

- RQ2 prep and stability
  - Added `scripts/prep_prompts_rq2.py` with category filters; writes prompts and positions to `mechdiff/data/rq2/`.
  - Updated positions to chat-aware last-K sampling (K≈16, tail excluded); wrote stats to `positions_stats.json`.
- Hooks and collectors
  - Implemented multi-site hooks: `resid_pre | resid_post | attn_out | mlp_out` in `mechdiff/utils/hooks.py`.
  - `collect_last_token_resids(...)` now accepts `hook_type`; `run_rq1_baseline.py` and `run_rq2_clt.py` expose `--hook`.
- RQ1 robustness refresh
  - Added `--bootstrap` to CKA; saved `cka_boot.json` and aligned RQ1 with CLT’s hook/positions protocol.
  - Layer patch script now outputs mean±std KL to `artifacts/rq1/patch_kl.json`.
- RQ2 CLT overhaul
  - SVD-based ridge with variance filtering; diagnostics: fallback_rate, raw CKA, train/val/shuffled R².
  - Added `--pca` and `--procrustes`; added `--logit_subspace` (tuned unembedding top-q directions).
  - Method-aware outputs: `artifacts/rq2/rq2_clt_L{layer}_<suffix>.json`.
- Documentation
  - Updated `docs/research_journal.md` with RQ1 refresh and RQ2 interim results/issues and planned next steps.

### 2025-08-22

- Environment and structure
  - Virtualenv created; dependencies installed; HF auth configured.
  - Moved run scripts under `mechdiff/experiments/{rq1,rq2}/`; artifacts split into `artifacts/rq1/` and `artifacts/rq2/`.
- RQ1 (first pass)
  - Curated KK prompts with category include/exclude and fairness option; ran baseline CKA (stride=2).
  - Causal sanity via cross-model layer patch: KL@L24/26 ≫ L10; artifacts saved under `artifacts/rq1/`.
- RQ2 (initial)
  - Implemented initial CLT runner; identified leakage from duplicated prompts; switched to prompt-identity split and multiple positions per prompt.

## Current status (high level)

- RQ1: Late-layer divergence (L24–26) replicated; causal patch shows large KL. Hooks aligned.
- RQ2: Global linear transport at L24 is weak under current protocol; investigating logit-subspace and component-wise mappings, and validating with mapped-patch KL.
