"""Inductive projection-field utilities.

These helpers turn projection coefficients learned on labeled training anchors
into test-anchor projections by GP conditioning, then evaluate the resulting
projection ensemble with the existing projected JumpGP predictor.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from tqdm.auto import tqdm

from djgp.projections.projected_jumpgp import run_projected_jumpgp


def median_pairwise_lengthscale(X: torch.Tensor, *, floor: float = 1e-3) -> float:
    """Median pairwise Euclidean distance, used as a default RBF lengthscale."""
    if X.shape[0] < 2:
        return 1.0
    pd = torch.cdist(X, X, p=2.0)
    iu = torch.triu_indices(X.shape[0], X.shape[0], offset=1, device=X.device)
    med = pd[iu[0], iu[1]].median().clamp_min(floor)
    return float(med.item())


def rbf_cross_kernel(
    X_left: torch.Tensor,
    X_right: torch.Tensor,
    *,
    lengthscale: float,
    variance: float = 1.0,
) -> torch.Tensor:
    """RBF kernel between two point sets."""
    d2 = torch.cdist(X_left, X_right, p=2.0).pow(2)
    return float(variance) * torch.exp(-0.5 * d2 / (max(float(lengthscale), 1e-8) ** 2))


def gp_condition_coefficients(
    X_anchor: torch.Tensor,
    C_anchor: torch.Tensor,
    X_query: torch.Tensor,
    *,
    ridge: float = 1e-4,
    lengthscale: float | None = None,
    variance: float = 1.0,
) -> dict[str, torch.Tensor | float]:
    """Condition independent coefficient GPs from ``C_anchor`` to ``X_query``.

    Args:
        X_anchor: ``[A, D]`` labeled projection-training anchors.
        C_anchor: ``[A, Q, Rv]`` learned MAP coefficients at those anchors.
        X_query: ``[T, D]`` test anchors.
        ridge: empirical-Bayes noise/jitter for the MAP coefficients.

    Returns a dict with ``mean`` ``[T, Q, Rv]`` and marginal ``var`` ``[T]``.
    The same scalar posterior variance is shared across coefficient dimensions.
    """
    if X_anchor.dim() != 2 or X_query.dim() != 2:
        raise ValueError("X_anchor and X_query must be 2D tensors.")
    if C_anchor.dim() != 3 or C_anchor.shape[0] != X_anchor.shape[0]:
        raise ValueError("C_anchor must be [A, Q, Rv] and align with X_anchor.")

    dtype = C_anchor.dtype
    device = C_anchor.device
    Xa = X_anchor.to(device=device, dtype=dtype)
    Xq = X_query.to(device=device, dtype=dtype)
    if lengthscale is None:
        lengthscale = median_pairwise_lengthscale(Xa)

    A = Xa.shape[0]
    C_flat = C_anchor.reshape(A, -1)
    Kaa = rbf_cross_kernel(Xa, Xa, lengthscale=lengthscale, variance=variance)
    Kaa = Kaa + float(ridge) * torch.eye(A, device=device, dtype=dtype)
    Kqa = rbf_cross_kernel(Xq, Xa, lengthscale=lengthscale, variance=variance)
    L = torch.linalg.cholesky(0.5 * (Kaa + Kaa.T))
    alpha = torch.cholesky_solve(C_flat, L)
    mean_flat = Kqa @ alpha

    v = torch.cholesky_solve(Kqa.T, L)
    var = (float(variance) - (Kqa * v.T).sum(dim=1)).clamp_min(1e-10)
    return {
        "mean": mean_flat.reshape(Xq.shape[0], C_anchor.shape[1], C_anchor.shape[2]),
        "var": var,
        "lengthscale": float(lengthscale),
        "ridge": float(ridge),
        "variance": float(variance),
    }


def sample_induced_coefficients(
    mean_C: torch.Tensor,
    var_query: torch.Tensor,
    *,
    n_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample ``C_*`` with independent coefficient dimensions and marginal GP variance."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    gen = None
    if seed is not None:
        gen = torch.Generator(device=mean_C.device)
        gen.manual_seed(int(seed))
    eps = torch.randn(
        (n_samples,) + tuple(mean_C.shape),
        device=mean_C.device,
        dtype=mean_C.dtype,
        generator=gen,
    )
    std = torch.sqrt(var_query.to(device=mean_C.device, dtype=mean_C.dtype)).view(1, -1, 1, 1)
    return mean_C.unsqueeze(0) + eps * std


def induced_W_from_coefficients(
    C_samples: torch.Tensor,
    W0: torch.Tensor,
    V: torch.Tensor,
    *,
    coeff_scale: float = 1.0,
    row_normalize: bool = True,
    row_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Map coefficient samples to projection matrices.

    ``C_samples`` may be ``[M, T, Q, Rv]`` or ``[T, Q, Rv]``. The returned tensor
    has the corresponding leading dimensions and final shape ``[..., Q, D]``.
    """
    squeeze = False
    if C_samples.dim() == 3:
        C_samples = C_samples.unsqueeze(0)
        squeeze = True
    if C_samples.dim() != 4:
        raise ValueError("C_samples must be [M, T, Q, Rv] or [T, Q, Rv].")
    W = W0.to(C_samples.device, C_samples.dtype).view(1, 1, W0.shape[0], W0.shape[1])
    V_t = V.to(C_samples.device, C_samples.dtype)
    W = W + float(coeff_scale) * torch.einsum("mtqr,dr->mtqd", C_samples, V_t)
    if row_normalize:
        norm = torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(row_norm_eps)
        W = W / norm
    return W.squeeze(0) if squeeze else W


def run_projected_jumpgp_ensemble(
    W_samples: torch.Tensor,
    X_anchors: torch.Tensor,
    X_neighbors_list,
    y_neighbors_list,
    *,
    device: torch.device | None = None,
    progress: bool = True,
    desc: str = "projection ensemble",
) -> dict[str, np.ndarray | float]:
    """Evaluate projected JumpGP for each ``W`` sample and moment-match outputs."""
    if W_samples.dim() == 3:
        W_samples = W_samples.unsqueeze(0)
    if W_samples.dim() != 4:
        raise ValueError("W_samples must be [M, T, Q, D] or [T, Q, D].")

    M, T = W_samples.shape[:2]
    mu_s = np.empty((M, T), dtype=np.float64)
    sig_s = np.empty((M, T), dtype=np.float64)
    iterator = range(M)
    if progress:
        iterator = tqdm(iterator, desc=desc, total=M)

    t0 = time.perf_counter()
    for m in iterator:
        out = run_projected_jumpgp(
            W_samples[m],
            X_anchors,
            X_neighbors_list,
            y_neighbors_list,
            device=device,
            progress=False,
            desc=f"{desc}[{m}]",
        )
        mu_s[m] = out["mu"]
        sig_s[m] = out["sigma"]
    elapsed = time.perf_counter() - t0

    mu = mu_s.mean(axis=0)
    var = np.mean(sig_s**2 + (mu_s - mu[None, :]) ** 2, axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return {
        "mu": mu,
        "sigma": sigma,
        "mu_samples": mu_s,
        "sigma_samples": sig_s,
        "elapsed_sec": float(elapsed),
    }

