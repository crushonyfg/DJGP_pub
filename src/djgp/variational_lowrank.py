"""Low-rank residual projection-field VI for LMJGP.

This variant replaces the full mean-zero projection GP

    W(x) in R^{Q x D}

with a fixed deterministic projection mean plus a low-rank residual field:

    W(x) = W0 + C(x) V^T,

where ``W0`` is fixed ``[Q, D]``, ``V`` is a fixed orthonormal basis
``[D, R]``, and each coefficient ``c_{q,r}(x)`` has an independent GP prior.

The key point is that ``C_j`` is Gaussian under the inducing-variable VI, and
``W_j`` is a linear transform of ``C_j``. Therefore ``q(W_j)`` is still
Gaussian and the existing LMJGP expected-kernel formulas can be reused with the
transformed mean/covariance.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from djgp.projections.inductive import run_projected_jumpgp_ensemble
from djgp.variational import (
    _LOG_2PI,
    expected_Kfu,
    expected_KufKfu,
    expected_log_sigmoid_gh_batch,
    jitter,
    kl_q_u_batch,
    kl_qp,
)


def qC_from_qU(
    X: torch.Tensor,
    Z: torch.Tensor,
    mu_U: torch.Tensor,
    sigma_U: torch.Tensor,
    lengthscales: torch.Tensor,
    var_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Induce ``q(C(X))`` from coefficient inducing variables.

    Args:
        X: query anchors ``[T, D]``.
        Z: inducing inputs ``[m2, D]``.
        mu_U, sigma_U: variational inducing coefficients ``[m2, Q, R]``.

    Returns:
        ``mu_C [T,Q,R]`` and diagonal covariance ``cov_C [T,Q,R,R]``.
    """
    T = X.shape[0]
    m2, Q, R = mu_U.shape
    device = X.device
    dtype = X.dtype
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    XZ2 = (X.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    mu_C = torch.empty((T, Q, R), device=device, dtype=dtype)
    sigma_C = torch.empty((T, Q, R), device=device, dtype=dtype)
    for q in range(Q):
        ell = lengthscales[q] if lengthscales.ndim > 0 else lengthscales
        vc = var_c[q] if getattr(var_c, "ndim", 0) > 0 else var_c
        Kzz = vc * torch.exp(-0.5 * ZZ2 / ell.pow(2))
        Kxz = vc * torch.exp(-0.5 * XZ2 / ell.pow(2))
        I = torch.eye(m2, device=device, dtype=dtype)
        Kzz = 0.5 * (Kzz + Kzz.T) + jitter * I
        L = torch.linalg.cholesky(Kzz)
        Kinv = torch.cholesky_inverse(L)
        A = Kxz @ Kinv
        diag_cross = (A * Kxz).sum(dim=1)
        U = Kinv @ Kxz.T
        for r in range(R):
            mu_C[:, q, r] = A @ mu_U[:, q, r]
            s2 = sigma_U[:, q, r].pow(2).unsqueeze(1)
            diag_var = (U.pow(2) * s2).sum(dim=0)
            sigma_C[:, q, r] = torch.sqrt((vc - diag_cross + diag_var).clamp_min(1e-12))
    return mu_C, torch.diag_embed(sigma_C.pow(2))


def qW_lowrank_residual(
    X: torch.Tensor,
    Z: torch.Tensor,
    mu_U: torch.Tensor,
    sigma_U: torch.Tensor,
    lengthscales: torch.Tensor,
    var_c: torch.Tensor,
    W0: torch.Tensor,
    V: torch.Tensor,
    *,
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``q(W(X))`` induced by ``q(C(X))``."""
    mu_C, cov_C = qC_from_qU(X, Z, mu_U, sigma_U, lengthscales, var_c)
    W0 = W0.to(device=X.device, dtype=X.dtype)
    V = V.to(device=X.device, dtype=X.dtype)
    mu_W = W0.unsqueeze(0) + float(residual_scale) * torch.einsum("tqr,dr->tqd", mu_C, V)
    cov_W = (float(residual_scale) ** 2) * torch.einsum("dr,tqrs,es->tqde", V, cov_C, V)
    return mu_W, cov_W, mu_C, cov_C


def _stack_sigma_u(u_params: list[dict[str, torch.Tensor]], m1: int) -> torch.Tensor:
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L = torch.tril(raw)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + 1e-6
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    return L @ L.transpose(-2, -1)


def compute_ELBO_lowrank_residual(
    regions: list[dict[str, torch.Tensor]],
    C_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Same ELBO as full LMJGP, with ``q(W)`` induced by low-rank q(C)."""
    T = len(regions)
    device = regions[0]["X"].device
    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    C_inducing_local = torch.stack([r["C"] for r in regions], dim=0)
    m1 = C_inducing_local.shape[1]

    U_logit = torch.stack([u["U_logit"] for u in u_params], dim=0).view(-1, 1)
    U = torch.sigmoid(U_logit)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    Sigma_u = _stack_sigma_u(u_params, m1)
    omega = torch.stack([u["omega"] for u in u_params], dim=0)

    Z = hyperparams["Z"]
    X_test = hyperparams["X_test"]
    lengthscales = hyperparams["lengthscales"]
    var_c = hyperparams["var_c"]
    W0 = hyperparams["W0"]
    V = hyperparams["V"]
    residual_scale = float(hyperparams.get("residual_scale", 1.0))

    mu_C_U = C_params["mu_C"]
    sigma_C_U = C_params["sigma_C"]
    KL_C = kl_qp(Z, mu_C_U, sigma_C_U, lengthscales, var_c)
    mu_W, cov_W, _, _ = qW_lowrank_residual(
        X_test,
        Z,
        mu_C_U,
        sigma_C_U,
        lengthscales,
        var_c,
        W0,
        V,
        residual_scale=residual_scale,
    )

    Kfu = expected_Kfu(mu_W, cov_W, X, C_inducing_local, sigma_k)
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C_inducing_local, sigma_k)
    KL_u = kl_q_u_batch(C_inducing_local, mu_u, Sigma_u, torch.tensor(1.0, device=device)).sum()

    d2 = (C_inducing_local.unsqueeze(2) - C_inducing_local.unsqueeze(1)).pow(2).sum(-1)
    Kuu = torch.exp(-0.5 * d2)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)
    V1 = sigma_k.view(-1, 1).pow(2) - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    v = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    E_fu = torch.einsum("tni,ti->tn", Kfu, v)
    Sigma_u_tilde = Sigma_u + torch.einsum("ti,tj->tij", mu_u, mu_u)
    T3 = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, Sigma_u_tilde)
    quad = (y.pow(2) - 2 * y * E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1, 1).pow(2))

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T1 = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise.view(-1, 1).pow(2)) + elog_sig - quad
    T2 = torch.log(U) + elog_one_minus
    region_elbo = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum(-1)
    data_term = region_elbo.sum()
    return (data_term - KL_C - KL_u) / max(T, 1)


def train_vi_lowrank_residual(
    regions: list[dict[str, torch.Tensor]],
    C_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
    *,
    lr: float = 1e-2,
    num_steps: int = 300,
    log_interval: int = 50,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, Any]]:
    """Optimize the low-rank residual ELBO."""
    params = [C_params["mu_C"], C_params["sigma_C"]]
    for u in u_params:
        params += [u["U_logit"], u["mu_u"], u["Sigma_u"], u["sigma_noise"], u["omega"], u["sigma_k"]]
    params += [hyperparams["Z"], hyperparams["lengthscales"], hyperparams["var_c"]]
    optimizer = torch.optim.Adam(params, lr=lr)
    best = None
    best_elbo = -float("inf")
    history = []
    t0 = time.perf_counter()
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        elbo = compute_ELBO_lowrank_residual(regions, C_params, u_params, hyperparams)
        (-elbo).backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            C_params["sigma_C"].clamp_(min=1e-3, max=10.0)
            hyperparams["var_c"].clamp_(min=1e-3, max=10.0)
            hyperparams["lengthscales"].clamp_(min=1e-4, max=1e3)
            for u in u_params:
                u["sigma_noise"].clamp_(min=1e-2)
                u["sigma_k"].clamp_(min=1e-2)
        val = float(elbo.detach().item())
        if val > best_elbo:
            best_elbo = val
            best = (
                {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in C_params.items()},
                copy.deepcopy(u_params),
                {
                    k: (v.detach().clone().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v)
                    for k, v in hyperparams.items()
                },
            )
        if step == 1 or step % max(1, log_interval) == 0:
            history.append({"step": step, "elbo": val})
            print(f"[lowrank-residual step {step}/{num_steps}] ELBO={val:.4f}", flush=True)
    train_time = time.perf_counter() - t0
    if best is not None:
        C_params, u_params, hyperparams = best
    diag = {"best_elbo": best_elbo, "train_time_sec": float(train_time), "history": history}
    return C_params, u_params, hyperparams, diag


def sample_W_lowrank_residual(
    X_query: torch.Tensor,
    C_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    *,
    M: int,
    seed: int = 0,
    row_normalize: bool = True,
) -> torch.Tensor:
    """Sample ``W`` from the low-rank residual posterior at query anchors."""
    _, _, mu_C, cov_C = qW_lowrank_residual(
        X_query,
        hyperparams["Z"],
        C_params["mu_C"],
        C_params["sigma_C"],
        hyperparams["lengthscales"],
        hyperparams["var_c"],
        hyperparams["W0"],
        hyperparams["V"],
        residual_scale=float(hyperparams.get("residual_scale", 1.0)),
    )
    R = mu_C.shape[-1]
    eye = torch.eye(R, device=X_query.device, dtype=X_query.dtype).view(1, 1, R, R)
    L = torch.linalg.cholesky(cov_C + 1e-8 * eye)
    gen = torch.Generator(device=X_query.device)
    gen.manual_seed(int(seed))
    eps = torch.randn((M,) + tuple(mu_C.shape), device=X_query.device, dtype=X_query.dtype, generator=gen)
    C = mu_C.unsqueeze(0) + torch.einsum("tqrs,mtqs->mtqr", L, eps)
    W0 = hyperparams["W0"].to(device=X_query.device, dtype=X_query.dtype)
    V = hyperparams["V"].to(device=X_query.device, dtype=X_query.dtype)
    W = W0.view(1, 1, *W0.shape) + float(hyperparams.get("residual_scale", 1.0)) * torch.einsum(
        "mtqr,dr->mtqd",
        C,
        V,
    )
    if row_normalize:
        W = W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(1e-6)
    return W


def predict_vi_lowrank_residual(
    regions: list[dict[str, torch.Tensor]],
    C_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    *,
    M: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Projected-JumpGP mixture prediction with sampled low-rank residual W."""
    X_query = hyperparams["X_test"]
    W_samples = sample_W_lowrank_residual(X_query, C_params, hyperparams, M=M, seed=seed)
    X_nb_list = [r["X"] for r in regions]
    y_nb_list = [r["y"] for r in regions]
    out = run_projected_jumpgp_ensemble(
        W_samples,
        X_query,
        X_nb_list,
        y_nb_list,
        device=X_query.device,
        progress=False,
        desc="lowrank_residual_lmjgp",
    )
    return (
        torch.as_tensor(out["mu"], device=X_query.device, dtype=X_query.dtype),
        torch.as_tensor(out["sigma"], device=X_query.device, dtype=X_query.dtype).pow(2),
        float(out["elapsed_sec"]),
    )
