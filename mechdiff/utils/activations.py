from typing import List

import torch

from .hooks import cache_layer


def collect_last_token_resids(model, tok, texts: List[str], layer_idx: int, device: str = "cuda"):
    """Return (N, d_model) residuals at layer_idx for the last token per text.

    Runs the model per text (batch=1) to limit memory pressure and ensures
    deterministic capture using forward hooks. No gradients are computed.
    """
    model.eval()
    feats = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to(device)
            with cache_layer(model, layer_idx) as c:
                _ = model(**enc)
            h = c.acts[-1][:, -1, :].float().cpu()  # (1, d_model) last token
            feats.append(h)
    return torch.cat(feats, dim=0)  # (N, d_model)


