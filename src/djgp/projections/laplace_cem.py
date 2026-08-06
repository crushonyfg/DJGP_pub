"""Diagonal-Laplace coefficient posterior for CEM-EM projection fields.

The CEM-EM learner produces a MAP coefficient field ``C_A`` in

    W(x_a) = W0 + C_a V^T.

This module approximates the posterior over training-anchor coefficients with
a diagonal Laplace approximation:

    q(C_A) ~= N(C_hat, diag((diag H_data + diag K_AA^{-1})^{-1})).

The test-anchor coefficient distribution then integrates this approximate
``q(C_A)`` through the GP conditional. This is still an approximation, but it
has the intended empirical-Bayes/Laplace interpretation and separates the
usual GP interpolation variance from uncertainty in the learned MAP
coefficients.
"""

from __future__ import annotations

from typing import Any

import torch

from djgp.projections.cem_em_projection import (
    CemCacheEntry,
    gate_bce_loss,
    gp_anchor_predictive_nll,
    gp_log_marginal_inliers,
)
from djgp.projections.inductive import median_pairwise_lengthscale, rbf_cross_kernel


def _row_normalize(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _single_anchor_cem_loss(
    C_flat: torch.Tensor,
    *,
    Q: int,
    Rv: int,
    W0: torch.Tensor,
    V: torch.Tensor,
    X_nb: torch.Tensor,
    y_nb: torch.Tensor,
    x_anchor: torch.Tensor | None,
    y_anchor: torch.Tensor | None,
    cache: CemCacheEntry,
    coeff_scale: float,
    row_normalize: bool,
    lambda_gate: float,
    lambda_anchor_pred: float,
    anchor_pred_var_floor: float,
) -> torch.Tensor:
    """Differentiable frozen-CEM per-anchor loss as a function of one C_t."""
    C_t = C_flat.view(Q, Rv)
    W = W0 + float(coeff_scale) * (C_t @ V.T)
    if row_normalize:
        W = _row_normalize(W)

    zero = C_flat.sum() * 0.0
    if not cache.success or cache.r is None:
        return zero

    X_nb = X_nb.to(device=C_flat.device, dtype=C_flat.dtype)
    y_nb = y_nb.to(device=C_flat.device, dtype=C_flat.dtype).view(-1)
    Z_nb = X_nb @ W.T
    loss = zero

    try:
        loss = loss + float(lambda_gate) * gate_bce_loss(
            Z_nb,
            cache.w.to(device=C_flat.device, dtype=C_flat.dtype),
            cache.r.to(device=C_flat.device, dtype=C_flat.dtype),
        )
    except Exception:
        pass

    idx = (cache.r.to(device=C_flat.device) > 0.5).nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() < 3:
        return loss
    Z_in = Z_nb[idx]
    y_in = y_nb[idx]
    try:
        gp_ll = gp_log_marginal_inliers(
            Z_in,
            y_in,
            cache.logtheta.to(device=C_flat.device, dtype=C_flat.dtype),
            cache.ms.to(device=C_flat.device, dtype=C_flat.dtype),
        )
        loss = loss - gp_ll
    except RuntimeError:
        pass

    if lambda_anchor_pred > 0 and x_anchor is not None and y_anchor is not None:
        try:
            z_anchor = x_anchor.to(device=C_flat.device, dtype=C_flat.dtype) @ W.T
            anchor_nll = gp_anchor_predictive_nll(
                Z_in,
                y_in,
                z_anchor,
                y_anchor.to(device=C_flat.device, dtype=C_flat.dtype),
                cache.logtheta.to(device=C_flat.device, dtype=C_flat.dtype),
                cache.ms.to(device=C_flat.device, dtype=C_flat.dtype),
                pred_var_floor=anchor_pred_var_floor,
            )
            loss = loss + float(lambda_anchor_pred) * anchor_nll
        except (RuntimeError, ValueError):
            pass
    return loss


def diagonal_cem_data_hessian(
    *,
    C_map: torch.Tensor,
    W0: torch.Tensor,
    V: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    X_anchors: torch.Tensor | None,
    y_anchors: torch.Tensor | None,
    cache: list[CemCacheEntry],
    coeff_scale: float = 1.0,
    row_normalize: bool = True,
    lambda_gate: float = 0.1,
    lambda_anchor_pred: float = 0.0,
    anchor_pred_var_floor: float = 0.05,
    max_anchors: int | None = None,
    hessian_floor: float = 1e-8,
) -> dict[str, Any]:
    """Compute blockwise exact Hessian diagonals for CEM data terms.

    The Hessian is block-diagonalized by anchor for tractability. GP prior
    curvature across anchors is added later in
    :func:`gp_condition_coefficients_laplace_diag`.
    """
    A, Q, Rv = C_map.shape
    n_eval = A if max_anchors is None else min(A, int(max_anchors))
    diag = torch.zeros_like(C_map)
    failures = 0
    for t in range(n_eval):
        C0 = C_map[t].detach().clone().reshape(-1).requires_grad_(True)

        def f(c_flat: torch.Tensor) -> torch.Tensor:
            return _single_anchor_cem_loss(
                c_flat,
                Q=Q,
                Rv=Rv,
                W0=W0.detach(),
                V=V.detach(),
                X_nb=X_neighbors_list[t],
                y_nb=y_neighbors_list[t],
                x_anchor=None if X_anchors is None else X_anchors[t],
                y_anchor=None if y_anchors is None else y_anchors[t],
                cache=cache[t],
                coeff_scale=coeff_scale,
                row_normalize=row_normalize,
                lambda_gate=lambda_gate,
                lambda_anchor_pred=lambda_anchor_pred,
                anchor_pred_var_floor=anchor_pred_var_floor,
            )

        try:
            H = torch.autograd.functional.hessian(f, C0, vectorize=False)
            d = torch.diagonal(H).reshape(Q, Rv)
            diag[t] = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        except Exception:
            failures += 1
            diag[t] = 0.0
    if n_eval < A:
        # Conservative fallback: use the median available curvature for skipped
        # anchors rather than pretending they are curvature-free.
        med = diag[:n_eval].reshape(n_eval, -1).median(dim=0).values.reshape(Q, Rv)
        diag[n_eval:] = med
    diag = diag.clamp_min(float(hessian_floor))
    return {
        "data_hessian_diag": diag,
        "n_hessian_anchors": int(n_eval),
        "n_hessian_failures": int(failures),
        "median_data_hessian_diag": float(diag.median().detach().cpu().item()),
        "mean_data_hessian_diag": float(diag.mean().detach().cpu().item()),
    }


def gp_condition_coefficients_laplace_diag(
    X_anchor: torch.Tensor,
    C_anchor: torch.Tensor,
    X_query: torch.Tensor,
    data_hessian_diag: torch.Tensor,
    *,
    ridge: float = 1e-4,
    lengthscale: float | None = None,
    variance: float = 1.0,
    curvature_floor: float = 1e-8,
) -> dict[str, torch.Tensor | float]:
    """GP-condition coefficients while integrating diagonal-Laplace q(C_A)."""
    if X_anchor.dim() != 2 or X_query.dim() != 2:
        raise ValueError("X_anchor and X_query must be 2D tensors.")
    if C_anchor.shape != data_hessian_diag.shape:
        raise ValueError("C_anchor and data_hessian_diag must have the same shape.")

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
    Kinv = torch.cholesky_inverse(L)
    B = Kqa @ Kinv
    mean_flat = B @ C_flat

    v = torch.cholesky_solve(Kqa.T, L)
    base_var = (float(variance) - (Kqa * v.T).sum(dim=1)).clamp_min(1e-10)

    prior_precision_diag = torch.diagonal(Kinv).view(A, 1, 1)
    post_var_anchor = 1.0 / (
        data_hessian_diag.to(device=device, dtype=dtype).clamp_min(float(curvature_floor))
        + prior_precision_diag
    ).clamp_min(float(curvature_floor))
    extra_var_flat = (B.pow(2) @ post_var_anchor.reshape(A, -1)).reshape(
        Xq.shape[0], C_anchor.shape[1], C_anchor.shape[2]
    )
    var = base_var.view(-1, 1, 1) + extra_var_flat

    return {
        "mean": mean_flat.reshape(Xq.shape[0], C_anchor.shape[1], C_anchor.shape[2]),
        "var": var.clamp_min(1e-10),
        "base_gp_var": base_var,
        "extra_laplace_var_mean": float(extra_var_flat.mean().detach().cpu().item()),
        "posterior_anchor_var_mean": float(post_var_anchor.mean().detach().cpu().item()),
        "posterior_anchor_var_median": float(post_var_anchor.median().detach().cpu().item()),
        "lengthscale": float(lengthscale),
        "ridge": float(ridge),
        "variance": float(variance),
    }


def sample_induced_coefficients_diag(
    mean_C: torch.Tensor,
    var_C: torch.Tensor,
    *,
    n_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample ``C_*`` from coefficient-specific diagonal marginal variances."""
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
    return mean_C.unsqueeze(0) + eps * torch.sqrt(var_C.clamp_min(1e-10)).unsqueeze(0)
