from typing import List, Set

import torch


def _format_chat(tok, user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_text


def uses_only_shared(text: str, tok, shared: Set[str], use_chat_template: bool = False) -> bool:
    """Return True if every token in text belongs to the shared token set.

    Tokenization excludes special tokens to reflect true lexical coverage.
    """
    vocab = tok.get_vocab()  # token string -> id
    inv = {i: s for s, i in vocab.items()}  # id -> token string
    t = _format_chat(tok, text) if use_chat_template else text
    ids = tok(t, add_special_tokens=False).input_ids
    return all(inv.get(i, None) in shared for i in ids)


def filter_shared(prompts: List[str], tok_base, tok_tuned, shared: Set[str], use_chat_template: bool = False) -> List[str]:
    """Filter prompts that tokenize entirely within the shared vocabulary for both tokenizers."""
    kept: List[str] = []
    for p in prompts:
        if uses_only_shared(p, tok_base, shared, use_chat_template) and uses_only_shared(p, tok_tuned, shared, use_chat_template):
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



def filter_shared_ratio(
    prompts: List[str],
    tok_base,
    tok_tuned,
    allowed_base: Set[int],
    allowed_tuned: Set[int],
    min_shared_ratio: float,
    use_chat_template: bool = False,
) -> List[str]:
    """Keep prompts where both tokenizers have ≥ min_shared_ratio tokens inside shared ids.

    Ratio is computed as (# tokens in allowed set) / (# tokens), excluding special tokens.
    """
    kept: List[str] = []
    for p in prompts:
        t_b = _format_chat(tok_base, p) if use_chat_template else p
        t_t = _format_chat(tok_tuned, p) if use_chat_template else p
        ids_b = tok_base(t_b, add_special_tokens=False).input_ids
        ids_t = tok_tuned(t_t, add_special_tokens=False).input_ids
        if not ids_b or not ids_t:
            continue
        shared_b = sum(1 for i in ids_b if i in allowed_base)
        shared_t = sum(1 for j in ids_t if j in allowed_tuned)
        ratio_b = shared_b / max(1, len(ids_b))
        ratio_t = shared_t / max(1, len(ids_t))
        if ratio_b >= min_shared_ratio and ratio_t >= min_shared_ratio:
            kept.append(p)
    return kept

