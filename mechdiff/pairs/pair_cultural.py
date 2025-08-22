"""Experiment pair configuration: backbone vs cultural fine-tune.

Switch datasets or prompt files here to re-run the whole cultural pipeline.
"""

PAIR = dict(
    name="cultural",
    base_id="meta-llama/Llama-3.1-8B-Instruct",
    tuned_id="inceptionai/Llama-3.1-Sherkala-8B-Chat",
    datasets=dict(
        kk_mc_hf_id="kz-transformers/kk-socio-cultural-bench-mc",
        freeform_file="data/prompts_freeform_kk.jsonl",
    ),
    domain="cultural",
    tokenizer_policy="shared_vocab_required",
)


