from typing import List

import torch

from .hooks import cache_layer


def _format_chat(tok, user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_text


def collect_last_token_resids(model, tok, texts: List[str], layer_idx: int, device: str = "cuda", use_chat_template: bool = True):
    """Return (N, d_model) residuals at layer_idx for the last CONTENT token per text.

    Skips trailing special tokens (e.g., EOS) when selecting the index. Falls back to
    the final position if no content token is found. Runs per-text (batch=1).
    """
    model.eval()
    feats = []
    with torch.no_grad():
        for t in texts:
            prompt = _format_chat(tok, t) if use_chat_template else t
            enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids[0]
            # Find last non-special token index by skipping EOS/pad ids
            special_ids = set(tok.all_special_ids) if hasattr(tok, "all_special_ids") else set()
            last_idx = input_ids.shape[0] - 1
            sel_idx = last_idx
            for i in range(last_idx, -1, -1):
                if int(input_ids[i]) not in special_ids:
                    sel_idx = i
                    break
            enc = {k: v.to(device) for k, v in enc.items()}
            with cache_layer(model, layer_idx) as c:
                _ = model(**enc)
            h = c.acts[-1][:, sel_idx, :].float().cpu()  # last content token
            feats.append(h)
    return torch.cat(feats, dim=0)  # (N, d_model)


