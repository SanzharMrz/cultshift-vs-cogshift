from typing import Optional

import torch


def _last_content_index(input_ids: torch.Tensor, special_ids: set) -> int:
    last = input_ids.size(1) - 1
    for i in range(last, -1, -1):
        if int(input_ids[0, i]) not in special_ids:
            return i
    return last


def layer_patch_crossmodel(model, tok, user_text: str, layer_idx: int, h_src_vec: torch.Tensor):
    """Run model on a chat-formatted prompt and overwrite residual at layer_idx for last content token.

    Returns (logits, patched_residual_vec_at_L).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    msgs = [{"role": "user", "content": user_text}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = user_text
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    last_idx = _last_content_index(input_ids, special_ids)

    h_src_vec = h_src_vec.to(device=device, dtype=dtype).view(1, 1, -1)
    patched_vec_captured = None

    block = model.model.layers[layer_idx]

    def hook(module, inputs, output):
        nonlocal patched_vec_captured
        out = output[0] if isinstance(output, tuple) else output  # (B, T, d)
        out = out.clone()
        out[:, last_idx:last_idx + 1, :] = h_src_vec
        patched_vec_captured = out[:, last_idx:last_idx + 1, :].detach().clone()
        return (out,) if isinstance(output, tuple) else out

    h = block.register_forward_hook(hook)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids)
    h.remove()

    logits = out.logits
    return logits, patched_vec_captured.squeeze(1)


