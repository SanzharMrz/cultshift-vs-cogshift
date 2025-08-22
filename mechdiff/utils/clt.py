from typing import Tuple

import torch


def standardize(X: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Zero-mean/unit-variance standardization along features.

    Returns standardized X and (mu, sd) for later un/standardization if needed.
    """
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    return (X - mu) / sd, (mu, sd)


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Solve (X^T X + lam I) W = X^T Y; returns W with shape (d_x, d_y)."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D: (N, d_x), (N, d_y)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same N")
    XtX = X.T @ X
    d_x = XtX.shape[0]
    reg = lam * torch.eye(d_x, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, X.T @ Y)
    return W


def map_states(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply linear mapping W to X: (N, d_x) @ (d_x, d_y) -> (N, d_y)."""
    return X @ W


