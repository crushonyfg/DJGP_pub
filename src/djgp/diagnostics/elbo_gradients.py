"""Training-time ELBO gradient diagnostics for the original LMJGP objective."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from djgp.variational import (
    expected_Kfu,
    expected_KufKfu,
    expected_log_sigmoid_gh_batch,
    kl_q_u_batch,
    kl_qp,
    qW_from_qV,
)

_LOG_2PI = math.log(2.0 * math.pi)


def _stack_sigma_u(u_params: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    lower = torch.tril(raw)
    diag = torch.diagonal(lower, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + 1e-6
    lower = lower - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    return lower @ lower.transpose(-2, -1)


def compute_lmjgp_elbo_parts(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return tensor-valued ELBO pieces without detaching the autograd graph."""

    device = regions[0]["X"].device
    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)

    U_logit = torch.stack([u["U_logit"] for u in u_params], dim=0).view(-1, 1)
    U = torch.sigmoid(U_logit)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    Sigma_u = _stack_sigma_u(u_params)
    omega = torch.stack([u["omega"] for u in u_params], dim=0)

    Z = hyperparams["Z"]
    X_test = hyperparams["X_test"]
    lengthscales = hyperparams["lengthscales"]
    var_w = hyperparams["var_w"]
    mu_V = V_params["mu_V"]
    sigma_V = V_params["sigma_V"]

    KL_V = kl_qp(Z, mu_V, sigma_V, lengthscales, var_w)
    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)
    Kfu = expected_Kfu(mu_W, cov_W, X, C, sigma_k)
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)

    KL_u_vec = kl_q_u_batch(C, mu_u, Sigma_u, torch.tensor(1.0, device=device))
    KL_u = KL_u_vec.sum()

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = torch.exp(-0.5 * d2)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)

    V1 = sigma_k.view(-1, 1) ** 2 - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    v = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    E_fu = torch.einsum("tni,ti->tn", Kfu, v)
    Sigma_u_tilde = Sigma_u + torch.einsum("ti,tj->tij", mu_u, mu_u)
    T3 = torch.einsum(
        "tij,tnjk,tkm,tmi->tn",
        Kuu_inv,
        KufKfu,
        Kuu_inv,
        Sigma_u_tilde,
    )
    quad = (y.pow(2) - 2.0 * y * E_fu + (V1 + T3)) / (
        2.0 * sigma_noise.view(-1, 1) ** 2
    )

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T1 = (
        -0.5 * _LOG_2PI
        - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2)
        + elog_sig
        - quad
    )
    T2 = torch.log(U) + elog_one_minus
    logits = torch.stack([T1, T2], dim=0)
    region_elbo = torch.logsumexp(logits, dim=0).sum(-1)
    data_term = region_elbo.sum()
    rho = torch.softmax(logits, dim=0)[0]
    rho_entropy = -(
        rho.clamp_min(1e-12).log() * rho
        + (1.0 - rho).clamp_min(1e-12).log() * (1.0 - rho)
    )

    return {
        "data_term": data_term,
        "KL_V": KL_V,
        "KL_u": KL_u,
        "region_elbo": region_elbo,
        "rho": rho,
        "rho_entropy": rho_entropy,
        "local_y_var": y.var(dim=1, unbiased=False),
        "mu_W": mu_W,
        "cov_W": cov_W,
    }


def _zero_like_param(param: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(param, memory_format=torch.preserve_format)


def _grad_pair(
    scalar: torch.Tensor,
    mu_V: torch.Tensor,
    sigma_V: torch.Tensor,
    *,
    retain_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    grads = torch.autograd.grad(
        scalar,
        (mu_V, sigma_V),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    g_mu = grads[0] if grads[0] is not None else _zero_like_param(mu_V)
    g_sigma = grads[1] if grads[1] is not None else _zero_like_param(sigma_V)
    return g_mu, g_sigma


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.detach()).item())


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / max(den, eps))


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().reshape(-1)
    bf = b.detach().reshape(-1)
    den = torch.linalg.norm(af) * torch.linalg.norm(bf)
    if float(den.item()) <= eps:
        return float("nan")
    return float((af @ bf / den).item())


def _grad_wrt_variance_proxy(g_sigma: torch.Tensor, sigma_V: torch.Tensor) -> torch.Tensor:
    return g_sigma / (2.0 * sigma_V.detach().clamp_min(1e-12))


def compute_lmjgp_qv_gradient_diagnostics(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
    *,
    step: int,
    beta_W: float = 1.0,
    boundary_quantile: float = 0.75,
) -> dict[str, Any]:
    """Split q(V) gradients into data, KL, and boundary-proxy contributions.

    The norms use the same per-anchor scaling as the training objective:
    ``data_term / T - beta_W * KL_V / T - KL_u / T``.
    """

    parts = compute_lmjgp_elbo_parts(regions, V_params, u_params, hyperparams)
    T = max(1, len(regions))
    mu_V = V_params["mu_V"]
    sigma_V = V_params["sigma_V"]

    data_obj = parts["data_term"] / T
    kl_obj = parts["KL_V"] / T
    kl_scaled_obj = float(beta_W) * kl_obj
    elbo_obj = data_obj - kl_scaled_obj - parts["KL_u"] / T

    g_data_mu, g_data_sigma = _grad_pair(data_obj, mu_V, sigma_V)
    g_kl_mu, g_kl_sigma = _grad_pair(kl_obj, mu_V, sigma_V)
    g_total_mu, g_total_sigma = _grad_pair(elbo_obj, mu_V, sigma_V)

    y_var = parts["local_y_var"].detach()
    if T > 1:
        thresh = torch.quantile(y_var, float(boundary_quantile))
        boundary_mask = y_var >= thresh
        if bool(boundary_mask.all()):
            boundary_mask = y_var == y_var.max()
    else:
        thresh = y_var[0]
        boundary_mask = torch.ones_like(y_var, dtype=torch.bool)
    interior_mask = ~boundary_mask

    zero = data_obj * 0.0
    boundary_obj = (
        parts["region_elbo"][boundary_mask].sum() / T if bool(boundary_mask.any()) else zero
    )
    interior_obj = (
        parts["region_elbo"][interior_mask].sum() / T if bool(interior_mask.any()) else zero
    )
    g_boundary_mu, g_boundary_sigma = _grad_pair(boundary_obj, mu_V, sigma_V)
    g_interior_mu, g_interior_sigma = _grad_pair(interior_obj, mu_V, sigma_V)

    data_mu_norm = _norm(g_data_mu)
    data_sigma_norm = _norm(g_data_sigma)
    kl_mu_norm = _norm(g_kl_mu)
    kl_sigma_norm = _norm(g_kl_sigma)
    scaled_kl_mu_norm = abs(float(beta_W)) * kl_mu_norm
    scaled_kl_sigma_norm = abs(float(beta_W)) * kl_sigma_norm
    boundary_mu_norm = _norm(g_boundary_mu)
    interior_mu_norm = _norm(g_interior_mu)
    boundary_sigma_norm = _norm(g_boundary_sigma)
    interior_sigma_norm = _norm(g_interior_sigma)

    g_data_s2 = _grad_wrt_variance_proxy(g_data_sigma, sigma_V)
    g_kl_s2 = _grad_wrt_variance_proxy(g_kl_sigma, sigma_V)
    g_boundary_s2 = _grad_wrt_variance_proxy(g_boundary_sigma, sigma_V)
    g_interior_s2 = _grad_wrt_variance_proxy(g_interior_sigma, sigma_V)
    data_s2_norm = _norm(g_data_s2)
    kl_s2_norm = _norm(g_kl_s2)
    boundary_s2_norm = _norm(g_boundary_s2)
    interior_s2_norm = _norm(g_interior_s2)

    rho = parts["rho"].detach()
    rho_entropy = parts["rho_entropy"].detach()
    mu_W = parts["mu_W"].detach()
    cov_W = parts["cov_W"].detach()
    sigma_W = torch.sqrt(torch.diagonal(cov_W, dim1=-2, dim2=-1).clamp_min(1e-12))
    row_second = mu_W.pow(2).sum(dim=-1) + sigma_W.pow(2).sum(dim=-1)

    result: dict[str, Any] = {
        "step": int(step),
        "beta_W": float(beta_W),
        "boundary_quantile": float(boundary_quantile),
        "n_anchors": int(T),
        "n_boundary": int(boundary_mask.sum().item()),
        "n_interior": int(interior_mask.sum().item()),
        "boundary_y_var_threshold": float(thresh.item()),
        "data_per_anchor": float(data_obj.detach().item()),
        "KL_V_per_anchor": float(kl_obj.detach().item()),
        "KL_V_scaled_per_anchor": float(kl_scaled_obj.detach().item()),
        "KL_u_per_anchor": float((parts["KL_u"] / T).detach().item()),
        "elbo_per_anchor": float(elbo_obj.detach().item()),
        "rho_mean": float(rho.mean().item()),
        "rho_entropy_mean": float(rho_entropy.mean().item()),
        "rho_hard_frac_005_095": float(((rho < 0.05) | (rho > 0.95)).float().mean().item()),
        "rho_boundary_mean": float(rho[boundary_mask].mean().item()) if bool(boundary_mask.any()) else float("nan"),
        "rho_boundary_entropy_mean": float(rho_entropy[boundary_mask].mean().item()) if bool(boundary_mask.any()) else float("nan"),
        "rho_interior_mean": float(rho[interior_mask].mean().item()) if bool(interior_mask.any()) else float("nan"),
        "rho_interior_entropy_mean": float(rho_entropy[interior_mask].mean().item()) if bool(interior_mask.any()) else float("nan"),
        "mu_W_frob_median": float(torch.linalg.norm(mu_W, dim=(-2, -1)).median().item()),
        "mu_W_frob_mean": float(torch.linalg.norm(mu_W, dim=(-2, -1)).mean().item()),
        "sigma_W_mean": float(sigma_W.mean().item()),
        "row_second_moment_median": float(row_second.median().item()),
        "mu_data_grad_norm": data_mu_norm,
        "mu_KL_grad_norm": kl_mu_norm,
        "mu_KL_scaled_grad_norm": scaled_kl_mu_norm,
        "mu_total_elbo_grad_norm": _norm(g_total_mu),
        "mu_data_over_KL": _safe_ratio(data_mu_norm, kl_mu_norm),
        "mu_data_over_scaled_KL": _safe_ratio(data_mu_norm, scaled_kl_mu_norm),
        "mu_data_cos_neg_KL": _cosine(g_data_mu, -g_kl_mu),
        "mu_boundary_data_grad_norm": boundary_mu_norm,
        "mu_interior_data_grad_norm": interior_mu_norm,
        "mu_boundary_share_of_split_norms": _safe_ratio(
            boundary_mu_norm, boundary_mu_norm + interior_mu_norm
        ),
        "mu_boundary_over_scaled_KL": _safe_ratio(boundary_mu_norm, scaled_kl_mu_norm),
        "sigma_data_grad_norm": data_sigma_norm,
        "sigma_KL_grad_norm": kl_sigma_norm,
        "sigma_KL_scaled_grad_norm": scaled_kl_sigma_norm,
        "sigma_total_elbo_grad_norm": _norm(g_total_sigma),
        "sigma_data_over_KL": _safe_ratio(data_sigma_norm, kl_sigma_norm),
        "sigma_data_over_scaled_KL": _safe_ratio(data_sigma_norm, scaled_kl_sigma_norm),
        "sigma_data_cos_neg_KL": _cosine(g_data_sigma, -g_kl_sigma),
        "sigma_boundary_data_grad_norm": boundary_sigma_norm,
        "sigma_interior_data_grad_norm": interior_sigma_norm,
        "sigma_boundary_share_of_split_norms": _safe_ratio(
            boundary_sigma_norm, boundary_sigma_norm + interior_sigma_norm
        ),
        "sigma_boundary_over_scaled_KL": _safe_ratio(boundary_sigma_norm, scaled_kl_sigma_norm),
        "sigma2proxy_data_grad_norm": data_s2_norm,
        "sigma2proxy_KL_grad_norm": kl_s2_norm,
        "sigma2proxy_KL_scaled_grad_norm": abs(float(beta_W)) * kl_s2_norm,
        "sigma2proxy_data_over_scaled_KL": _safe_ratio(
            data_s2_norm, abs(float(beta_W)) * kl_s2_norm
        ),
        "sigma2proxy_boundary_data_grad_norm": boundary_s2_norm,
        "sigma2proxy_interior_data_grad_norm": interior_s2_norm,
        "sigma2proxy_boundary_share_of_split_norms": _safe_ratio(
            boundary_s2_norm, boundary_s2_norm + interior_s2_norm
        ),
        "kl_mu_pull_dot": float(((-g_kl_mu.detach()) * mu_V.detach()).sum().item()),
        "kl_mu_pull_dot_scaled": float(
            ((-float(beta_W) * g_kl_mu.detach()) * mu_V.detach()).sum().item()
        ),
    }
    return result


__all__ = [
    "compute_lmjgp_elbo_parts",
    "compute_lmjgp_qv_gradient_diagnostics",
]
