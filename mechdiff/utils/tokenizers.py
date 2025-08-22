from typing import Tuple, Set, Dict

from transformers import AutoTokenizer


def load_tokenizers(base_id: str, tuned_id: str):
    """Load slow tokenizers for deterministic behavior.

    Returns the pair (tok_base, tok_tuned).
    """
    tok_base = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    tok_tuned = AutoTokenizer.from_pretrained(tuned_id, use_fast=False)
    return tok_base, tok_tuned


def shared_vocab_maps(tok_base, tok_tuned):
    """Compute shared token strings and ID sets for both tokenizers.

    Returns:
        shared: set[str] — token strings in both vocabs
        ids_b: dict[int,str] — base tokenizer ID -> token string for shared tokens
        ids_t: dict[int,str] — tuned tokenizer ID -> token string for shared tokens
        allowed_b: set[int] — base IDs allowed (shared only)
        allowed_t: set[int] — tuned IDs allowed (shared only)
    """
    vocab_b: Dict[str, int] = tok_base.get_vocab()
    vocab_t: Dict[str, int] = tok_tuned.get_vocab()

    shared: Set[str] = set(vocab_b.keys()) & set(vocab_t.keys())
    ids_b: Dict[int, str] = {vocab_b[s]: s for s in shared}
    ids_t: Dict[int, str] = {vocab_t[s]: s for s in shared}

    allowed_b: Set[int] = set(ids_b.keys())
    allowed_t: Set[int] = set(ids_t.keys())
    return shared, ids_b, ids_t, allowed_b, allowed_t


