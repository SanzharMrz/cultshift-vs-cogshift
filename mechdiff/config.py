"""Global defaults for the mechdiff experiments.

These defaults lock in model IDs, fairness controls, dataset sizes, scanning
strategy, CLT settings, steering/ablation sweeps, churn thresholds, and seed.
"""

DEFAULTS = dict(
    # Models
    base_id="meta-llama/Llama-3.1-8B-Instruct",
    cultural_id="inceptionai/Llama-3.1-Sherkala-8B-Chat",
    cognitive_id="nvidia/OpenMath2-Llama3.1-8B",

    # Fairness controls
    enforce_shared_vocab=True,
    mask_logits=True,
    post_embedding_only=True,

    # Data sizes / splits
    n_freeform_cultural=40,
    n_gsm8k=300,
    n_math=100,

    # RQ1 representation metrics
    rep_metrics=["CKA"],  # keep simple; add "PWCCA" only if needed
    layer_stride=2,        # evaluate every 2 layers by default

    # RQ2 CLT
    clt_tokens_per_layer=8000,  # target ~8–10k positions; OK to lower if needed
    clt_val_frac=0.2,

    # RQ3 steering (rank-k projector)
    k_sweep=[1, 2, 3, 4, 6, 8],
    alpha_grid=[0.5, 1.0, 2.0, 3.0],
    side_effect_eps_pp=5.0,   # ≤5 pp neutral drop
    side_effect_eps_ppl=0.05, # ≤5% perplexity increase

    # RQ4 churn thresholds
    head_z_threshold=2.0,
    head_abs_delta_nat=0.5,  # math
    head_abs_delta_pp=5.0,   # cultural refusal (percentage points)

    # Layer fallback if scanning is skipped
    fallback_layers=[22, 26],

    # Reproducibility
    seed=17,
)


