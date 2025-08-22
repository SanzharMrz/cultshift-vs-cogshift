# pip install transformers accelerate torch einops
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch, math, json, re
from collections import defaultdict

BASE_RAW = "meta-llama/Meta-Llama-3.1-8B"
FT = "inceptionai/Llama-3.1-Sherkala-8B-Chat"

tok_base_raw = AutoTokenizer.from_pretrained(BASE_RAW, use_fast=False)
tok_ft = AutoTokenizer.from_pretrained(FT, use_fast=False)

# --- tokenizer diff & shared-vocab mapping ---
v_base_raw = tok_base_raw.get_vocab()
v_ft = tok_ft.get_vocab()

tokens_base_raw = set(v_base_raw.keys())
tokens_ft = set(v_ft.keys())

print("=" * 80)
print("LLAMA 3.1-8B RAW vs SHERKALA TOKENIZER COMPARISON")
print("=" * 80)
print(f"Llama 3.1-8B (raw): {len(tokens_base_raw):,} tokens")
print(f"Sherkala: {len(tokens_ft):,} tokens")

vocab_increase = len(tokens_ft) - len(tokens_base_raw)
pct_increase = (vocab_increase / len(tokens_base_raw)) * 100
print(f"\nSherkala vocabulary increase: +{vocab_increase:,} tokens ({pct_increase:.2f}%)")

# Compare overlaps
shared_tokens = tokens_base_raw & tokens_ft
only_raw = tokens_base_raw - tokens_ft
only_ft = tokens_ft - tokens_base_raw

print(f"\n" + "=" * 80)
print("TOKEN OVERLAP ANALYSIS")
print("=" * 80)
print(f"Shared tokens: {len(shared_tokens):,} ({len(shared_tokens)/len(tokens_base_raw)*100:.2f}% of raw vocab)")
print(f"Unique to Raw: {len(only_raw):,} tokens")
print(f"Unique to Sherkala: {len(only_ft):,} tokens")

# Show some examples of unique tokens
if only_raw:
    print(f"\nTokens only in Raw (first 10): {sorted(list(only_raw))[:10]}")
if only_ft:
    print(f"Tokens only in Sherkala (first 20): {sorted(list(only_ft))[:20]}")

# Shared vocab mapping for masking
shared_ids_raw = {v_base_raw[tok]: tok for tok in shared_tokens}
shared_ids_ft = {v_ft[tok]: tok for tok in shared_tokens}

# Per-model allowed id sets for masking
allowed_ids_raw = set(shared_ids_raw.keys())
allowed_ids_ft = set(shared_ids_ft.keys())

def uses_only_shared(text, tok, shared_token_set):
    ids = tok(text, add_special_tokens=False).input_ids
    # Map ids->token strings (slow but fine for filtering)
    inv = {i:s for s,i in tok.get_vocab().items()}
    return all(inv[i] in shared_token_set for i in ids)

print(f"\nExample filter (Kazakh text with raw model): {uses_only_shared('сәлем', tok_base_raw, shared_tokens)}")
print(f"Example filter (Kazakh text with Sherkala): {uses_only_shared('сәлем', tok_ft, shared_tokens)}")

# Analyze token types in Sherkala-specific tokens
def analyze_token_types(tokens):
    cyrillic_count = 0
    special_count = 0
    latin_count = 0
    other_count = 0
    
    for token in tokens:
        if any('\u0400' <= char <= '\u04FF' for char in token):  # Cyrillic range
            cyrillic_count += 1
        elif token.startswith('<') and token.endswith('>'):
            special_count += 1
        elif token.isalpha() and all(ord(char) < 128 for char in token):
            latin_count += 1
        else:
            other_count += 1
    
    return cyrillic_count, special_count, latin_count, other_count

cyrillic, special, latin, other = analyze_token_types(only_ft)
print(f"\nSherkala-specific token analysis:")
print(f"  Cyrillic tokens: {cyrillic}")
print(f"  Special tokens: {special}")
print(f"  Latin tokens: {latin}")
print(f"  Other tokens: {other}")

# Sample some Cyrillic tokens
cyrillic_tokens = [token for token in only_ft if any('\u0400' <= char <= '\u04FF' for char in token)]
print(f"\nSample Cyrillic tokens in Sherkala: {cyrillic_tokens[:10]}")

# Test tokenization of sample text
test_texts = [
    "Hello world",
    "Привет мир",
    "Сәлем әлем",  # Kazakh greeting
    "How are you doing today?",
    "Как дела сегодня?",
]

print(f"\n{'='*60}")
print("TOKENIZATION COMPARISON")
print(f"{'='*60}")

for text in test_texts:
    raw_tokens = tok_base_raw.tokenize(text)
    ft_tokens = tok_ft.tokenize(text)
    
    print(f"\nText: '{text}'")
    print(f"Raw tokens ({len(raw_tokens)}): {raw_tokens}")
    print(f"Sherkala tokens ({len(ft_tokens)}): {ft_tokens}")
    
    if len(raw_tokens) != len(ft_tokens):
        diff = len(ft_tokens) - len(raw_tokens)
        efficiency = ((len(raw_tokens) - len(ft_tokens)) / len(raw_tokens)) * 100 if len(raw_tokens) > 0 else 0
        print(f"  → Token difference: {diff:+d} (Sherkala efficiency: {efficiency:+.1f}%)")

# --- logits masking helper for generation/eval ---
def mask_logits_to_shared(logits, allowed_ids_set):
    # logits: (batch, vocab)
    mask = torch.full_like(logits, float('-inf'))
    idx = torch.tensor(sorted(list(allowed_ids_set)), device=logits.device)
    mask.index_copy_(1, idx, logits.index_select(1, idx))
    return mask
