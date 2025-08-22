from typing import List, Set

import torch


def uses_only_shared(text: str, tok, shared: Set[str]) -> bool:
    """Return True if every token in text belongs to the shared token set.

    Tokenization excludes special tokens to reflect true lexical coverage.
    """
    vocab = tok.get_vocab()  # token string -> id
    inv = {i: s for s, i in vocab.items()}  # id -> token string
    ids = tok(text, add_special_tokens=False).input_ids
    return all(inv.get(i, None) in shared for i in ids)


def filter_shared(prompts: List[str], tok_base, tok_tuned, shared: Set[str]) -> List[str]:
    """Filter prompts that tokenize entirely within the shared vocabulary for both tokenizers."""
    kept: List[str] = []
    for p in prompts:
        if uses_only_shared(p, tok_base, shared) and uses_only_shared(p, tok_tuned, shared):
            kept.append(p)
    return kept


def mask_logits_to_allowed(logits: torch.Tensor, allowed_ids: Set[int]) -> torch.Tensor:
    """Mask logits outside allowed_ids to -inf.

    Supports shapes:
      - (B, V)
      - (B, T, V)
    Returns a new tensor with masked values; input is not modified in-place.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")

    allowed_idx = torch.tensor(sorted(allowed_ids), device=logits.device, dtype=torch.long)
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)

    if logits.dim() == 2:
        batch, vocab = logits.shape
        masked = torch.full_like(logits, neg_inf)
        masked.index_copy_(1, allowed_idx, logits.index_select(1, allowed_idx))
        return masked
    elif logits.dim() == 3:
        batch, time, vocab = logits.shape
        masked = torch.full_like(logits, neg_inf)
        # Flatten time for efficient gather/index_copy, then reshape back
        flat = logits.reshape(batch * time, vocab)
        flat_masked = torch.full_like(flat, neg_inf)
        flat_masked.index_copy_(1, allowed_idx, flat.index_select(1, allowed_idx))
        return flat_masked.reshape(batch, time, vocab)
    else:
        raise ValueError("Unsupported logits shape; expected (B,V) or (B,T,V)")


