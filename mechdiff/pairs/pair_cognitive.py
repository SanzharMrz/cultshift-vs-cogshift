"""Experiment pair configuration: backbone vs cognitive (math) fine-tune."""

PAIR = dict(
    name="cognitive",
    base_id="meta-llama/Llama-3.1-8B-Instruct",
    tuned_id="nvidia/OpenMath2-Llama3.1-8B",
    datasets=dict(
        gsm8k="openai/gsm8k",
        math="hendrycks/competition_math",
    ),
    domain="cognitive",
    tokenizer_policy="shared_vocab_recommended",  # same tokenizer; keep fairness on for symmetry
)


