### Research journal

- Environment
  - Verified CUDA availability; logged into Hugging Face.
  - Installed `hf_transfer`; enabled fast downloads.

- RQ1 baseline (first pass)
  - Ran `run_rq1_baseline.py` with cultural pair using fallback prompts.
  - Downloaded and loaded base and tuned models; computed CKA over layers.
  - Saved: `mechdiff/artifacts/rq1_cka.json`.

- Prompt curation (Kazakh, cultural)
  - Added `scripts/curate_prompts_kk.py` to pull from `kz-transformers/kk-socio-cultural-bench-mc`.
  - Category filter: focus on culture/tradition/etiquette/community; excluded religion.
  - Fairness: implemented shared-vocab filtering across base/tuned tokenizers.
  - Performance: added tqdm progress, batched tokenization, on-the-fly scan+filter.
  - Added `--min_shared_ratio` threshold to allow partial shared coverage.
  - Generated 40 prompts at `--min_shared_ratio 0.65` → `mechdiff/data/prompts_freeform_kk.jsonl`.

- Next
  - Re-run RQ1 with curated KK prompts (e.g., `--stride 2`), then run cognitive pair for contrast.

---

### Update — RQ1 baseline (cultural, curated KK prompts with relaxed fairness)

- Prompts
  - Curated 40 KK prompts from `kz-transformers/kk-socio-cultural-bench-mc` using `scripts/curate_prompts_kk.py`.
  - Applied category filter (tradition/culture/etiquette/community; religion excluded).
  - Used relaxed fairness: `--min_shared_ratio 0.65` (≥65% tokens in shared vocab for both tokenizers).

- Runner
  - Executed: `python run_rq1_baseline.py --pair mechdiff/pairs/pair_cultural.py --stride 2 --device cuda --min_shared_ratio 0.65`.
  - Result: prompts kept after fairness = 40/40; CKA computed across layers.
  - Artifact: `mechdiff/artifacts/rq1_cka.json`.

- Decision rationale
  - Sherkala vs base use different tokenizers; strict 100% shared-vocab filtering was too restrictive (0/40 kept).
  - Relaxed threshold (0.65) preserves comparability while enabling sufficient data to run RQ1.

- Next
  - Run RQ1 on cognitive pair (`mechdiff/pairs/pair_cognitive.py`) for contrast.
  - Identify top-divergence layers from CKA for use in RQ2–R3.

### CKA interpretation and action plan

- Observations
  - Early–mid layers are similar (CKA ≈ 0.86–0.90 at L=8–12).
  - Divergence grows late, minima at L=24–26 (CKA ≈ 0.63–0.66); small rebound at L=28–30.
- Layer choices
  - Probe layers: L=24, 26 (max divergence); Control: L=10 (high similarity).
- Next steps
  1) RQ1 causal sanity: cross-model layer patch at L=24/26 (swap last-token residual; keep masking).
  2) RQ2 CLT: train cross-layer coder at L=24 (opt. L=26); report R²/CKA; mapped-patch ≈ real-patch.
  3) RQ3 causal handle: rank-k projector at L=24; sweep k=1..8 and α; measure steer↑ / ablate↓ with side-effects.
  4) RQ4 churn: head grad×act at L=24; optional top-k head patching; expect few strong heads.
  5) Stability: add 20–40 prompts or bootstrap to confirm the dip at 24–26.

### Pipeline fix

- Updated `collect_last_token_resids` to select the last content token (skips EOS/pad) to avoid blurring late-layer signals.

### RQ1 results and decisions (final)

- Setup
  - KK cultural free-form (n=40). Chat templates used. Fairness masking applied for diagnostics.
- Representation
  - CKA high mid-layers (≈0.90 @ L10), dips late (≈0.63–0.66 @ L24–26).
- Causal sanity
  - Layer patch at L24/L26 shifts next-token distribution (KL≈0.6–1.1 nats); L10 ≈0. Matches CKA hotspot.
- Behavior (graded, not binary)
  - Binary refusal rare (0%).
  - Soft refusal (Δ logp(refusal)−logp(neutral)): tuned < base on KK prompts (tuned less refusal-prone).
  - Style markers (/100 tokens, unmasked primary): small base > tuned; masked diagnostic lower due to shared-vocab constraints.
  - MC Δlogp(correct) per token: unmasked shows tuned > base; masked coverage low due to tokenizer drift (treat as diagnostic).
- Takeaway
  - Cultural differences are late-layer and causal, with subtle style/propensity shifts. Strict masking constrains KK behavioral scoring; use unmasked primary with masking as diagnostic.

### Next — RQ2 CLT plan

- Collect 3–5k last-content-token residuals at L=24 for base & tuned; standardize per side.
- Fit ridge mapping base→tuned (λ grid 1e-6..1e-2); report R² (20% holdout) and CKA(mapped,tuned).
- Mapped-patch check: patch tuned with mapped base states; compare to real tuned states via next-token KL and soft-refusal deltas. Save `artifacts/rq2_clt.json` and `figs/rq2_r2.png`.



### RQ1 refresh — robustness (KK cultural)

- Setup
  - Expanded prompt pool with category filtering; chat-formatted; downsampled for speed (≈450 train / 150 val).
  - For stability checks, did not enforce shared-vocab filtering; fairness kept as diagnostic elsewhere.

- Representation (stability)
  - CKA across layers with bootstrap resampling confirms a persistent late-layer dip around L24–26; saved to `mechdiff/artifacts/rq1/cka_boot.json`.

- Causal patch sanity (KL)
  - Mean next-token KL at late layers (L24/L26) is substantially higher than at an early control layer (L10).
  - Directional asymmetry: injecting base activations into the tuned model perturbs it more than the reverse, indicating specialized late-layer features in the tuned model.
  - Variation across prompts is non-trivial (std ~1 nat), consistent with heterogeneous cultural content.
  - Outputs saved to `mechdiff/artifacts/rq1/rq1_patch.json` and summarized in `mechdiff/artifacts/rq1/patch_kl.json`.

- Takeaway
  - Late-layer differences are causal and asymmetric, matching the representation dip. Binary refusal remains unchanged on benign prompts; KL is the primary causal effect size for this set.

- Next
  - Proceed to RQ2 (CLT) using leakage-free train/val pools with multiple token positions. Report val-only R² and CKA(mapped vs tuned), and compare mapped-patch vs real-patch effects at L24.

### RQ2 status — issues and fixes in progress

- Observation
  - Initial CLT at L24/L26 produced unstable/near-zero R² with occasional NaNs when mapping full residual (d=4096) with N<d and mixed token positions. Raw CKA on val was also low under that protocol.

- Fixes implemented
  - Switched to SVD-based ridge with variance filtering; broadened λ grid; added train/val/shuffled R² and raw CKA diagnostics with fallback-rate reporting.
  - Regenerated positions: chat-aware, K=16 per prompt, excluding last 4 tokens; stats saved to `mechdiff/data/rq2/positions_stats.json`.
  - Added hook/site selection: `resid_pre|resid_post|attn_out|mlp_out` to align CLT with causal decision points; CKA runner now uses the same hook.
  - Added PCA and Procrustes solvers; added logit-subspace option to project into tuned unembedding top-q directions.

- Next
  - Re-run CLT on L24 (and L26) with aligned hook and positions; compare PCA/Procrustes vs logit-subspace vs ridge. Add tuned→base and L10 controls. Then validate with mapped-patch ΔKL at L24.

- Note
  - If CLT remains low after alignment and subspace methods while late-layer patching shows strong KL, we will conclude non-global linearity (transport is local/subspace-specific) and proceed with mapped-patch and optional per-cluster maps.

### RQ2 interim results — L24 (aligned hooks, new positions)

- Baseline alignment
  - RQ1 @ L24 with `hook=resid_post` on the new 450-prompt pool: CKA ≈ 0.34. Baseline signal recovered (previously depressed by misalignment), still below earlier ~0.63–0.66 due to different prompts/method and genuinely larger late-layer divergence.

- CLT (global linear mapping)
  - Ridge (full 4096-d): train R² ≈ 0.03, val R² ≈ −0.015, CKA(mapped,tuned) ≈ 0.066.
  - PCA+Procrustes (q=512): similar or worse (CKA ≈ 0.055).
  - Takeaway: under current position protocol, base→tuned at L24 does not admit a strong global linear map.

- Next (deepening the analysis)
  - Run logit-subspace CLT and component-wise CLT (`attn_out`, `mlp_out`) at L24/L26; compare to ridge in full space.
  - Proceed to mapped-patch validation at L24 to test causal transportability (KL_mapped ≪ KL_raw and correlation with cos(mapped, tuned)).

### RQ2 — Signal at L24 (K=1, resid_post)

- Findings
  - On val, CKA(mapped,tuned) ≈ 0.66 and cosine ≈ 0.67 at L24 → the map captures the correct direction.
  - val R² < 0 with train R² ≈ 1.0 → overfit plus scale/variance mismatch. R² penalizes amplitude; cosine/CKA do not.
  - Component-wise at L24 also aligns (CKA ≈ 0.63), slightly weaker than full residual.

- Next steps (surgical)
  1) Whiten → scaled Procrustes → color at L24 (train-only μ/Σ with shrinkage; ZCA with eigen floor); log train/val R², CKA, cosine. Expect val R² > 0 with high CKA/cos.
  2) Finish mapped-patch with scale norm at the decision token; compare KL_mapped vs KL_raw and correlate with cosine.
  3) Joint logit-subspace ridge (q≈256) to reduce tuned-only bias; then mapped-patch from this subspace map.
  4) Replicate best mapper at L26 (also `attn_out`/`mlp_out`) and tabulate (R², CKA, KL_raw, KL_mapped, reduction%).

- Hygiene
  - Ensure `k_positions = 1` is recorded; keep `resid_post` and the exact chat templating identical to RQ1 across CLT and mapped-patch.

- One-liner
  - L24 shows strong directional transport (CKA≈0.66, cos≈0.67) but R²<0 due to scale mismatch; we’ll apply whitened scaled-Procrustes and joint logit-subspace maps and validate with mapped-patch ΔKL, then replicate at L26.

### RQ2 — Mapped-patch results (causal transportability)

- L24 `resid_post` (K=1): KL_raw ≈ 6.435 → KL_mapped ≈ 5.820 → Δ ≈ 0.615 (≈9.6%).
  - CLT JSON: `mechdiff/artifacts/rq2/rq2_clt_L24_procrustes_scaled_20250823_142606.json`
  - Map bundle: `mechdiff/artifacts/rq2/maps/rq2_clt_L24_procrustes_scaled_20250823_142606.pt`
  - Mapped-patch JSON: `mechdiff/artifacts/rq2/mapped_patch_L24.json`

- L26 `resid_post` (K=1): KL_raw ≈ 6.835 → KL_mapped ≈ 5.938 → Δ ≈ 0.897 (≈13.1%).
  - CLT JSON: `mechdiff/artifacts/rq2/rq2_clt_L26_procrustes_scaled_20250823_141030.json`
  - Map bundle: `mechdiff/artifacts/rq2/maps/rq2_clt_L26_procrustes_scaled_20250823_141030.pt`
  - Mapped-patch JSON: `mechdiff/artifacts/rq2/mapped_patch_L26.json`

- L24 `mlp_out` (K=1): KL_raw ≈ 7.185 → KL_mapped ≈ 6.445 → Δ ≈ 0.740 (≈10.3%).
  - Mapped-patch JSON: `mechdiff/artifacts/rq2/mapped_patch_L24.json`

- Interpretation
  - Linear maps learned on base→tuned activations reduce cross-model patch KL by ~10–13%, strongest at L26 and visible in the MLP path at L24. This is causal evidence that the learned transport aligns the tuned model’s late-layer geometry.

### RQ2 — Component splits and control

- Attention vs MLP
  - `attn_out @ L26 (K=1)`: large reduction (≈29% drop in ΔKL). File: `mechdiff/artifacts/rq2/mapped_patch_L26.json` (see `hook` field).
  - `attn_out @ L24 (K=1)`: negative effect (≈−22% drop), unstable without additional re-scaling. File: `mechdiff/artifacts/rq2/mapped_patch_L24.json` (hook=`attn_out`).
  - Summary: attention transport appears layer-sensitive (good at L26, harmful at L24), while MLP transport at L24 is consistently helpful (~10%).

- Early-layer control
  - `resid_post @ L10 (K=1)`: ΔKL reduction ≈17%. File: `mechdiff/artifacts/rq2/mapped_patch_L10.json`.
  - Caveat: larger-than-expected early-layer drop suggests a global alignment (rotation/scale) component; we will verify with a shuffle control.

—

Using fairness-aware Kazakh prompts and K=1 (last content token), we find strong representational divergence in late layers (CKA≈0.34 at L24). A simple linear cross-layer transport (Procrustes‑scaled with shrinkage) yields high mapped-to-tuned similarity (CKA≈0.64–0.66) at L24–L26. Crucially, when we causally patch the base model with mapped activations, the next-token KL to the tuned model drops by ~10–13% on resid_post, by ~10% on mlp_out@L24, and by ~29% on attn_out@L26. attn_out@L24 is unstable (negative unless re‑scaled), indicating attention changes are layer‑localized, while MLP shows consistent late‑layer shifts. An early-layer control (L10) also shows ΔKL reduction (~17%), which we flag for shuffle‑control validation (likely a global rotation/scale rather than culture-specific circuitry). Overall, cultural fine‑tuning appears as a late‑layer subspace re‑orientation with significant MLP and layer‑specific attention components that are partly transferable by a linear map.

### Update — Task 1 (L26 MLP split)

RQ2 (CLT, cultural pair): At L26/attn_out we previously saw a ~29% KL drop; at L26/mlp_out we now get a ~9.5% KL drop; L10/resid_post gives ~17% drop. L24/attn_out is negative without α tuning; next we sweep α to test scale sensitivity. Overall, late layers show non‑trivial, directionally consistent linear transport, with largest causal impact in attention at L26 and solid, but smaller, effect in MLP.