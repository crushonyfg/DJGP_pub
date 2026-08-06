"""Full-covariance variational coefficient posterior for CEM-EM projections.

The CEM-EM MAP learner estimates coefficients ``C_A`` in

    W_a = W0 + C_a V^T.

This module upgrades the MAP field to a Gaussian variational posterior over
training-anchor coefficient values.  For each coefficient dimension ``p`` it
uses a full anchor covariance

    q(c_{A,p}) = N(m_p, S_p),

with an RBF GP prior ``p(c_{A,p}) = N(0, K_AA)``.  The likelihood term is the
same frozen-CEM predictive surrogate used by the MAP/Laplace utilities, and
test-anchor uncertainty is induced by integrating ``q(C_A)`` through the GP
conditional.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from djgp.projections.cem_em_projection import CemCacheEntry
from djgp.projections.inductive import median_pairwise_lengthscale, rbf_cross_kernel
from djgp.projections.laplace_cem import _single_anchor_cem_loss


def _inverse_softplus(x: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    val = torch.tensor(float(x), device=device, dtype=dtype).clamp_min(1e-8)
    return torch.log(torch.expm1(val))


def _tril_from_raw(raw_tril: torch.Tensor, *, diag_floor: float = 1e-5) -> torch.Tensor:
    """Convert unconstrained raw lower-triangular factors to valid Cholesky factors."""
    L = torch.tril(raw_tril)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos = F.softplus(diag) + float(diag_floor)
    return L - torch.diag_embed(diag) + torch.diag_embed(pos)


def _sample_coefficients(
    mean_flat: torch.Tensor,
    raw_tril: torch.Tensor,
    *,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``C`` as ``[M,A,P]`` and return Cholesky factors ``[P,A,A]``."""
    L = _tril_from_raw(raw_tril)
    P, A, _ = L.shape
    eps = torch.randn(
        n_samples,
        P,
        A,
        device=mean_flat.device,
        dtype=mean_flat.dtype,
    )
    samples_pa = mean_flat.T.unsqueeze(0) + torch.einsum("pab,mpb->mpa", L, eps)
    return samples_pa.permute(0, 2, 1).contiguous(), L


def kl_fullcov_anchor_gp(
    mean_flat: torch.Tensor,
    chol_factors: torch.Tensor,
    Kaa: torch.Tensor,
) -> torch.Tensor:
    """KL(q(C_A)||p(C_A)) for independent coefficient dimensions.

    Args:
        mean_flat: ``[A,P]`` variational means.
        chol_factors: ``[P,A,A]`` lower Cholesky factors of ``S_p``.
        Kaa: GP prior covariance ``[A,A]``.
    """
    A, P = mean_flat.shape
    Kaa = 0.5 * (Kaa + Kaa.T)
    Lk = torch.linalg.cholesky(Kaa)
    Kinv = torch.cholesky_inverse(Lk)
    logdet_k = 2.0 * torch.diagonal(Lk).log().sum()
    S = chol_factors @ chol_factors.transpose(-2, -1)
    trace = torch.einsum("ab,pba->p", Kinv, S)
    quad = torch.einsum("ap,ab,bp->p", mean_flat, Kinv, mean_flat)
    logdet_s = 2.0 * torch.diagonal(chol_factors, dim1=-2, dim2=-1).log().sum(dim=-1)
    return 0.5 * (trace + quad - A + logdet_k - logdet_s).sum()


def train_cem_coefficient_vi(
    *,
    C_init: torch.Tensor,
    W0: torch.Tensor,
    V: torch.Tensor,
    X_anchor: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    cache: list[CemCacheEntry],
    y_anchor: torch.Tensor | None = None,
    coeff_scale: float = 1.0,
    row_normalize: bool = True,
    lambda_gate: float = 0.1,
    lambda_anchor_pred: float = 0.0,
    anchor_pred_var_floor: float = 0.05,
    gp_ridge: float = 1e-4,
    gp_lengthscale: float | None = None,
    gp_variance: float = 1.0,
    n_steps: int = 120,
    lr: float = 5e-3,
    n_mc: int = 2,
    kl_weight: float = 1.0,
    init_std: float = 0.03,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimize a full-anchor-covariance Gaussian VI posterior over ``C_A``."""
    A, Q, Rv = C_init.shape
    P = Q * Rv
    device = C_init.device
    dtype = C_init.dtype
    if gp_lengthscale is None:
        gp_lengthscale = median_pairwise_lengthscale(X_anchor)
    Kaa = rbf_cross_kernel(
        X_anchor.to(device=device, dtype=dtype),
        X_anchor.to(device=device, dtype=dtype),
        lengthscale=float(gp_lengthscale),
        variance=float(gp_variance),
    )
    Kaa = Kaa + float(gp_ridge) * torch.eye(A, device=device, dtype=dtype)

    mean_flat = C_init.detach().reshape(A, P).clone().requires_grad_(True)
    raw_tril = torch.zeros(P, A, A, device=device, dtype=dtype)
    raw_diag = _inverse_softplus(init_std, device=device, dtype=dtype)
    idx = torch.arange(A, device=device)
    raw_tril[:, idx, idx] = raw_diag
    raw_tril.requires_grad_(True)
    opt = torch.optim.Adam([mean_flat, raw_tril], lr=float(lr))
    best: tuple[torch.Tensor, torch.Tensor] | None = None
    best_loss = float("inf")
    history = []
    t0 = time.perf_counter()

    for step in range(1, int(n_steps) + 1):
        opt.zero_grad()
        samples_flat, L = _sample_coefficients(mean_flat, raw_tril, n_samples=int(n_mc))
        data_loss = mean_flat.sum() * 0.0
        for m in range(int(n_mc)):
            C_m = samples_flat[m].reshape(A, Q, Rv)
            for a in range(A):
                data_loss = data_loss + _single_anchor_cem_loss(
                    C_m[a].reshape(-1),
                    Q=Q,
                    Rv=Rv,
                    W0=W0.detach(),
                    V=V.detach(),
                    X_nb=X_neighbors_list[a],
                    y_nb=y_neighbors_list[a],
                    x_anchor=X_anchor[a],
                    y_anchor=None if y_anchor is None else y_anchor[a],
                    cache=cache[a],
                    coeff_scale=coeff_scale,
                    row_normalize=row_normalize,
                    lambda_gate=lambda_gate,
                    lambda_anchor_pred=lambda_anchor_pred,
                    anchor_pred_var_floor=anchor_pred_var_floor,
                )
        data_loss = data_loss / float(max(1, n_mc))
        kl = kl_fullcov_anchor_gp(mean_flat, L, Kaa)
        loss = (data_loss + float(kl_weight) * kl) / float(max(1, A))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([mean_flat, raw_tril], max_norm=10.0)
        opt.step()
        val = float(loss.detach().cpu().item())
        if math.isfinite(val) and val < best_loss:
            best_loss = val
            best = (mean_flat.detach().clone(), _tril_from_raw(raw_tril).detach().clone())
        if verbose and (step == 1 or step == int(n_steps) or step % max(1, int(n_steps) // 4) == 0):
            history.append(
                {
                    "step": step,
                    "loss": val,
                    "data_loss_per_anchor": float((data_loss / max(1, A)).detach().cpu().item()),
                    "kl_per_anchor": float((kl / max(1, A)).detach().cpu().item()),
                }
            )
            print(
                f"[cem-coeff-vi step {step}/{n_steps}] loss={val:.4f} "
                f"data/A={history[-1]['data_loss_per_anchor']:.4f} "
                f"kl/A={history[-1]['kl_per_anchor']:.4f}",
                flush=True,
            )

    if best is None:
        best = (mean_flat.detach().clone(), _tril_from_raw(raw_tril).detach().clone())
    mean_best, chol_best = best
    return {
        "mean": mean_best.reshape(A, Q, Rv),
        "mean_flat": mean_best,
        "chol": chol_best,
        "cov": chol_best @ chol_best.transpose(-2, -1),
        "Kaa": Kaa.detach(),
        "lengthscale": float(gp_lengthscale),
        "ridge": float(gp_ridge),
        "variance": float(gp_variance),
        "best_loss": float(best_loss),
        "train_time_sec": float(time.perf_counter() - t0),
        "history": history,
        "kl": float(kl_fullcov_anchor_gp(mean_best, chol_best, Kaa).detach().cpu().item()),
    }


def gp_condition_coefficients_vi_fullcov(
    X_anchor: torch.Tensor,
    vi: dict[str, Any],
    X_query: torch.Tensor,
    *,
    ridge: float | None = None,
    lengthscale: float | None = None,
    variance: float | None = None,
) -> dict[str, Any]:
    """Induce ``q(C_*)`` by integrating full-covariance ``q(C_A)``."""
    mean_flat = vi["mean_flat"]
    cov = vi["cov"]
    A, P = mean_flat.shape
    Q, Rv = vi["mean"].shape[1:]
    dtype = mean_flat.dtype
    device = mean_flat.device
    Xa = X_anchor.to(device=device, dtype=dtype)
    Xq = X_query.to(device=device, dtype=dtype)
    lengthscale = float(vi["lengthscale"] if lengthscale is None else lengthscale)
    ridge = float(vi["ridge"] if ridge is None else ridge)
    variance = float(vi["variance"] if variance is None else variance)
    Kaa = rbf_cross_kernel(Xa, Xa, lengthscale=lengthscale, variance=variance)
    Kaa = Kaa + ridge * torch.eye(A, device=device, dtype=dtype)
    Kqa = rbf_cross_kernel(Xq, Xa, lengthscale=lengthscale, variance=variance)
    Kqq = rbf_cross_kernel(Xq, Xq, lengthscale=lengthscale, variance=variance)
    Lk = torch.linalg.cholesky(0.5 * (Kaa + Kaa.T))
    Kinv = torch.cholesky_inverse(Lk)
    B = Kqa @ Kinv
    mean_q = B @ mean_flat
    base_cov = Kqq - Kqa @ Kinv @ Kqa.T
    base_cov = 0.5 * (base_cov + base_cov.T)
    cov_q = base_cov.unsqueeze(0) + torch.einsum("ta,pab,ub->ptu", B, cov, B)
    eye = torch.eye(Xq.shape[0], device=device, dtype=dtype)
    cov_q = 0.5 * (cov_q + cov_q.transpose(-2, -1)) + 1e-8 * eye.unsqueeze(0)
    diag = torch.diagonal(cov_q, dim1=-2, dim2=-1).T.reshape(Xq.shape[0], Q, Rv)
    return {
        "mean": mean_q.reshape(Xq.shape[0], Q, Rv),
        "cov": cov_q,
        "var": diag.clamp_min(1e-10),
        "base_gp_cov": base_cov,
        "lengthscale": lengthscale,
        "ridge": ridge,
        "variance": variance,
        "posterior_anchor_var_mean": float(torch.diagonal(cov, dim1=-2, dim2=-1).mean().detach().cpu().item()),
        "query_var_mean": float(diag.mean().detach().cpu().item()),
    }


def sample_induced_coefficients_fullcov(
    mean_C: torch.Tensor,
    cov_by_dim: torch.Tensor,
    *,
    n_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample ``C_*`` from full query covariance per coefficient dimension."""
    T, Q, Rv = mean_C.shape
    P = Q * Rv
    mean_flat = mean_C.reshape(T, P)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=mean_C.device)
        gen.manual_seed(int(seed))
    eye = torch.eye(T, device=mean_C.device, dtype=mean_C.dtype).unsqueeze(0)
    cov = 0.5 * (cov_by_dim + cov_by_dim.transpose(-2, -1))
    L = None
    for jitter_val in (1e-8, 1e-6, 1e-4, 1e-2):
        try:
            L = torch.linalg.cholesky(cov + jitter_val * eye)
            break
        except RuntimeError:
            L = None
    if L is None:
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(1e-6)
        cov_psd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-2, -1)
        L = torch.linalg.cholesky(cov_psd + 1e-6 * eye)
    eps = torch.randn(
        int(n_samples),
        P,
        T,
        device=mean_C.device,
        dtype=mean_C.dtype,
        generator=gen,
    )
    samples_pt = mean_flat.T.unsqueeze(0) + torch.einsum("ptu,mpu->mpt", L, eps)
    return samples_pt.permute(0, 2, 1).reshape(int(n_samples), T, Q, Rv)
