"""
First-layer projection variants for the LMJGP ablation.

Given the variational posterior moment ``A_j = E_q[W_j^T W_j] in R^{D x D}`` (built from
``mu_W, cov_W`` via :func:`djgp.diagnostics.projection_posterior.metric_A_from_moment`),
construct a ``Q x D`` projection ``L_j`` that defines a learned Mahalanobis-like distance
in latent space::

    z_i = L_j x_i,
    || L_j (x_i - x_{i'}) ||^2 = (x_i - x_{i'})^T L_j^T L_j (x_i - x_{i'}).

Three variants share the same trace ``tr(L_j L_j^T) = tr(A_j)`` (per anchor) so the only
difference is the geometry of the projection:

- ``learned``    : ``L_j = Lambda_Q^{1/2} U_Q^T``, top-``Q`` eigendecomposition of ``A_j``.
                   Equivalent (up to rotation in latent space) to projecting onto the
                   directions that carry the largest expected squared distance.
- ``isotropic``  : ``L_j = sqrt(tr(A_j)/Q) * R_j`` with ``R_j R_j^T = I_Q``
                   (random row-orthonormal, full ``Q``-dim ball, no learned direction).
- ``random``     : ``L_j = sqrt(tr(A_j)/Q) * R_j``, ``R_j`` row-normalized random
                   (rows are unit-norm but not mutually orthogonal).

Use ``build_projection_per_anchor`` to produce ``[T, Q, D]`` tensors, then either pass
them to ``djgp.projections.projected_jumpgp.run_projected_jumpgp`` for end-to-end
metrics, or feed ``z_i = L_j x_i`` to a custom local model.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

PROJECTION_MODES = ("learned", "isotropic", "random")


def _assert_psd_dim(A: torch.Tensor) -> None:
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must end in [D, D]; got shape {tuple(A.shape)}.")


def _trace_per_anchor(A: torch.Tensor) -> torch.Tensor:
    return A.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def learned_L_from_A(A: torch.Tensor, Q: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Top-``Q`` eigen-projection: ``L = diag(sqrt(lambda_top))) U_top^T``.

    Parameters
    ----------
    A : torch.Tensor of shape ``[..., D, D]``
        Symmetric PSD metric (e.g. ``E[W_j^T W_j]``).
    Q : int
        Latent dimension of the projection (rows of ``L``).

    Returns
    -------
    L : torch.Tensor of shape ``[..., Q, D]``.

    Notes
    -----
    When ``A`` has rank ``< Q``, missing eigenvalues are clamped to ``eps`` and produce
    rows of vanishing norm. Trace is preserved exactly: ``tr(L L^T) = sum_top_Q lambda``.
    """
    _assert_psd_dim(A)
    A_sym = 0.5 * (A + A.transpose(-2, -1))
    D = A_sym.shape[-1]
    if Q > D:
        raise ValueError(f"Q={Q} cannot exceed D={D}.")
    eigvals, eigvecs = torch.linalg.eigh(A_sym)
    eigvals = torch.clamp(eigvals, min=0.0)
    top_vals = eigvals[..., -Q:]
    top_vecs = eigvecs[..., :, -Q:]
    sqrt_vals = torch.sqrt(torch.clamp(top_vals, min=eps))
    L = (top_vecs * sqrt_vals.unsqueeze(-2)).transpose(-2, -1)
    return L


def _random_orthonormal_rows(
    Q: int,
    D: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``R in R^{Q x D}`` with ``R R^T = I_Q`` from QR of a Gaussian."""
    G = torch.randn((D, Q), generator=generator, device=device, dtype=dtype)
    Qmat, _ = torch.linalg.qr(G, mode="reduced")
    return Qmat.transpose(-2, -1)


def isotropic_orthonormal_L(
    A: torch.Tensor,
    Q: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Per-anchor row-orthonormal projection scaled to match ``tr(A_j)``.

    ``L_j = sqrt(tr(A_j) / Q) * R_j`` with ``R_j R_j^T = I_Q`` so ``tr(L_j L_j^T) = tr(A_j)``.
    """
    _assert_psd_dim(A)
    *batch, D, _ = A.shape
    tr = _trace_per_anchor(A).clamp_min(0.0)
    scale = torch.sqrt(tr / Q)
    Ls = []
    A_flat = A.reshape(-1, D, D)
    scale_flat = scale.reshape(-1)
    for i in range(A_flat.shape[0]):
        R = _random_orthonormal_rows(
            Q,
            D,
            generator=generator,
            device=A.device,
            dtype=A.dtype,
        )
        Ls.append(scale_flat[i] * R)
    return torch.stack(Ls, dim=0).reshape(*batch, Q, D)


def random_rownorm_L(
    A: torch.Tensor,
    Q: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Per-anchor row-normalized random projection scaled to match ``tr(A_j)``.

    Each row drawn ``i.i.d.`` Gaussian then renormalized to unit L2; rows are *not*
    mutually orthogonal. Final scale ``sqrt(tr(A_j)/Q)`` keeps ``tr(L_j L_j^T) = tr(A_j)``.
    """
    _assert_psd_dim(A)
    *batch, D, _ = A.shape
    tr = _trace_per_anchor(A).clamp_min(0.0)
    scale = torch.sqrt(tr / Q)
    G = torch.randn((*batch, Q, D), generator=generator, device=A.device, dtype=A.dtype)
    norms = torch.linalg.norm(G, dim=-1, keepdim=True).clamp_min(1e-12)
    Rn = G / norms
    return scale.unsqueeze(-1).unsqueeze(-1) * Rn


def build_projection_per_anchor(
    A: torch.Tensor,
    Q: int,
    *,
    mode: str,
    seed: int = 0,
) -> torch.Tensor:
    """Dispatch to one of :data:`PROJECTION_MODES` and return ``L of shape [T, Q, D]``."""
    if mode not in PROJECTION_MODES:
        raise ValueError(f"mode must be one of {PROJECTION_MODES}; got {mode!r}.")
    g = torch.Generator(device=A.device)
    g.manual_seed(seed)
    if mode == "learned":
        return learned_L_from_A(A, Q)
    if mode == "isotropic":
        return isotropic_orthonormal_L(A, Q, generator=g)
    return random_rownorm_L(A, Q, generator=g)


def _capture_ratio(A: torch.Tensor, Q: int, eps: float = 1e-12) -> float:
    """Sum of top-``Q`` eigenvalues of ``A`` divided by ``tr(A)`` — average over anchors."""
    A_sym = 0.5 * (A + A.transpose(-2, -1))
    eigvals = torch.clamp(torch.linalg.eigvalsh(A_sym), min=0.0)
    tr = eigvals.sum(dim=-1).clamp_min(eps)
    top = eigvals[..., -Q:].sum(dim=-1)
    return float((top / tr).mean().item())


def _eff_rank_batch(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A_sym = 0.5 * (A + A.transpose(-2, -1))
    eigvals = torch.clamp(torch.linalg.eigvalsh(A_sym), min=0.0)
    sum_l = eigvals.sum(dim=-1)
    sum_l2 = (eigvals**2).sum(dim=-1)
    out = torch.where(
        sum_l2 > eps * eps,
        sum_l.pow(2) / sum_l2.clamp_min(eps),
        torch.full_like(sum_l, float("nan")),
    )
    return out


def projection_quality_diagnostics(
    A: torch.Tensor,
    L_dict: dict[str, torch.Tensor],
    *,
    Q: int,
) -> dict[str, Any]:
    """
    Per-variant median diagnostics computable without running JumpGP.

    Returns a nested dict::

        {
            "metric_A": {"median_trace_A", "median_eff_rank_A", "median_top_eig_A",
                         "median_min_eig_A_positive", "capture_ratio_top_Q"},
            "<mode>": {
                "median_trace_LLt", "trace_ratio_to_A",
                "median_kappa_LLt", "median_eff_rank_LLt",
                "median_row_norm_sq_min", "median_row_norm_sq_max",
            },
            ...
        }
    """
    A_sym = 0.5 * (A + A.transpose(-2, -1))
    eigvals_A = torch.clamp(torch.linalg.eigvalsh(A_sym), min=0.0)
    tr_A = eigvals_A.sum(dim=-1)
    top_eig_A = eigvals_A.max(dim=-1).values
    pos_mask = eigvals_A > 1e-12
    finite_min = torch.where(pos_mask.any(dim=-1), eigvals_A.masked_fill(~pos_mask, float("inf")).min(dim=-1).values, torch.full_like(tr_A, float("nan")))
    eff_rank_A = _eff_rank_batch(A_sym)

    out: dict[str, Any] = {
        "metric_A": {
            "median_trace_A": float(tr_A.median().item()),
            "median_eff_rank_A": float(np.nanmedian(eff_rank_A.cpu().numpy())),
            "median_top_eig_A": float(top_eig_A.median().item()),
            "median_min_eig_A_positive": float(np.nanmedian(finite_min.cpu().numpy())),
            "capture_ratio_top_Q": _capture_ratio(A_sym, Q),
        }
    }

    for mode, L in L_dict.items():
        LLt = torch.einsum("tqd,trd->tqr", L, L)
        tr_LLt = LLt.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        eigvals_LLt = torch.clamp(torch.linalg.eigvalsh(0.5 * (LLt + LLt.transpose(-2, -1))), min=0.0)
        s_max = eigvals_LLt.max(dim=-1).values
        eps = 1e-12
        pos = eigvals_LLt > eps
        s_min_pos = torch.where(
            pos.any(dim=-1),
            eigvals_LLt.masked_fill(~pos, float("inf")).min(dim=-1).values,
            torch.full_like(s_max, float("nan")),
        )
        kappa = torch.where(
            (s_max > eps) & torch.isfinite(s_min_pos),
            s_max / s_min_pos.clamp_min(eps),
            torch.full_like(s_max, float("nan")),
        )
        eff_rank_LLt = _eff_rank_batch(LLt)
        row_norms_sq = (L * L).sum(dim=-1)  # [T, Q]
        rn_min = row_norms_sq.min(dim=-1).values
        rn_max = row_norms_sq.max(dim=-1).values
        ratio = (tr_LLt / tr_A.clamp_min(1e-12)).cpu().numpy()
        out[mode] = {
            "median_trace_LLt": float(tr_LLt.median().item()),
            "trace_ratio_to_A_median": float(np.nanmedian(ratio)),
            "median_kappa_LLt": float(np.nanmedian(kappa.cpu().numpy())),
            "median_eff_rank_LLt": float(np.nanmedian(eff_rank_LLt.cpu().numpy())),
            "median_row_norm_sq_min": float(rn_min.median().item()),
            "median_row_norm_sq_max": float(rn_max.median().item()),
        }
    return out


def project_neighborhoods(
    L: torch.Tensor,
    X_anchors: torch.Tensor,
    X_neighbors_list: Iterable[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Apply per-anchor ``L_t`` to anchor and neighborhood inputs.

    Parameters
    ----------
    L : ``[T, Q, D]``
    X_anchors : ``[T, D]``
    X_neighbors_list : length-``T`` list of ``[n_t, D]``.

    Returns
    -------
    Z_anchors : ``[T, Q]``
    Z_neighbors_list : list of ``[n_t, Q]``.
    """
    Z_anchors = torch.einsum("tqd,td->tq", L, X_anchors)
    Z_neighbors_list = []
    for t, X_nb in enumerate(X_neighbors_list):
        Z_nb = X_nb @ L[t].T
        Z_neighbors_list.append(Z_nb)
    return Z_anchors, Z_neighbors_list
