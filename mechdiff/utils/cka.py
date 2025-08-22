import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA similarity between two (N, d) representations.

    Inputs should correspond to the same N prompts. Zero-mean is performed
    internally. Returns a Python float in [0, 1] (numerically may stray slightly).
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D tensors with shape (N, d)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (N)")

    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    hsic = (K * L).sum()
    denom = torch.sqrt((K * K).sum() * (L * L).sum() + 1e-8)
    val = (hsic / denom).item()
    return float(val)


