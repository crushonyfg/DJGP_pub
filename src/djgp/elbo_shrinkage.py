"""ELBO residual-GP shrinkage diagnostics: decompose ``G`` vs ``sigma_n^2 K^{-1}``.

Stationary mean (fixed ``Sigma_u`` in ``G``):

    mu_u* = (beta_G G + beta_K sigma_n^2 K^{-1})^{-1} Phi^T r,
    zeta* = Phi_amp @ mu_u*.

Sweep ``beta_G, beta_K`` and ``sigma_n`` caps to locate what suppresses ``zeta*/|r|``.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from djgp.gp_feature_alignment import (
    _stack_mu_u,
    alignment_residual,
    compute_elbo_kernels,
    ridge_coefficients,
)


def _zeta_ratio(zeta: torch.Tensor, r: torch.Tensor, eps: float = 1e-8) -> float:
    return float(zeta.std().item() / (r.std().item() + eps))


def elbo_star_solve(
    phi_amp: torch.Tensor,
    r: torch.Tensor,
    G: torch.Tensor,
    Kuu_inv: torch.Tensor,
    sigma_n: torch.Tensor,
    *,
    beta_G: float = 1.0,
    beta_K: float = 1.0,
    sigma_n_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Solve tempered ELBO star; shapes ``phi_amp`` ``[T,n,m1]``, ``r`` ``[T,n]``."""
    sn = (sigma_n.view(-1) * float(sigma_n_scale)).clamp_min(1e-6)
    m1 = G.shape[-1]
    eye = torch.eye(m1, device=G.device, dtype=G.dtype).unsqueeze(0)
    A = float(beta_G) * G + float(beta_K) * (sn.view(-1, 1, 1).pow(2)) * Kuu_inv
    rhs = torch.einsum("tnj,tn->tj", phi_amp, r)
    if float(beta_G) + float(beta_K) < 1e-12:
        mu_star = torch.zeros_like(rhs)
    else:
        A = A + 1e-8 * eye
        mu_star = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    zeta_star = torch.einsum("tnj,tj->tn", phi_amp, mu_star)
    Kinvt_mu = torch.einsum("tij,tj->ti", Kuu_inv, mu_star)
    return {
        "mu_star": mu_star,
        "zeta_star": zeta_star,
        "Kinvt_mu_star": Kinvt_mu,
        "sigma_n_eff": sn,
    }


@torch.no_grad()
def elbo_star_metrics(
    regions,
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
    u_params,
    *,
    beta_G: float = 1.0,
    beta_K: float = 1.0,
    sigma_n_scale: float = 1.0,
    sigma_n_cap: float | None = None,
    sigma_n_fixed: float | None = None,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
    beta_u_train: float | None = None,
) -> dict[str, float]:
    """Scalar metrics for one ``(beta_G, beta_K, sigma_n)`` setting.

    If ``beta_u_train`` is set, also reports ``ratio_zeta_star_beta_u_r`` using
    ``beta_K = beta_u_train`` (consistent with tempered KL training).
    """
    from experiments.synthetic.local_gp_elbo_mj import stack_m_j

    y = torch.stack([r["y"] for r in regions], dim=0)
    m_j = stack_m_j(u_params)
    r = alignment_residual(y, use_m_j=True, m_j=m_j)
    sigma_n = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    if sigma_n_fixed is not None:
        sigma_n = torch.full_like(sigma_n, float(sigma_n_fixed))
    elif sigma_n_cap is not None:
        sigma_n = sigma_n.clamp(max=float(sigma_n_cap))

    kern = compute_elbo_kernels(
        regions, proj, hyperparams, u_params,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    phi_amp = torch.einsum("tni,tij->tnj", kern["Kfu"], kern["Kuu_inv"])
    G = torch.einsum("tij,tnjk,tkm->tim", kern["Kuu_inv"], kern["KufKfu"], kern["Kuu_inv"])

    star = elbo_star_solve(
        phi_amp, r, G, kern["Kuu_inv"], sigma_n,
        beta_G=beta_G, beta_K=beta_K, sigma_n_scale=sigma_n_scale,
    )
    mu_u = _stack_mu_u(u_params)
    z_learn = torch.einsum("tnj,tj->tn", phi_amp, mu_u)
    b_ridge = ridge_coefficients(phi_amp, r, ridge_lambda=1e-2)
    z_ridge = torch.einsum("tnj,tj->tn", phi_amp, b_ridge)

    z_star = star["zeta_star"]
    Kinvt_learn = torch.einsum("tij,tj->ti", kern["Kuu_inv"], mu_u)
    r_std = float(r.std().item()) + 1e-8

    out = {
        "beta_G": float(beta_G),
        "beta_K": float(beta_K),
        "sigma_n_scale": float(sigma_n_scale),
        "sigma_n_cap": float(sigma_n_cap) if sigma_n_cap is not None else float("nan"),
        "sigma_n_fixed": float(sigma_n_fixed) if sigma_n_fixed is not None else float("nan"),
        "sigma_n_mean": float(star["sigma_n_eff"].mean().item()),
        "ratio_zeta_star_r": _zeta_ratio(z_star, r),
        "ratio_zeta_learned_r": _zeta_ratio(z_learn, r),
        "ratio_zeta_ridge_r": _zeta_ratio(z_ridge, r),
        "Kinvt_mu_star_norm_mean": float(star["Kinvt_mu_star"].norm(dim=-1).mean().item()),
        "Kinvt_mu_learned_norm_mean": float(Kinvt_learn.norm(dim=-1).mean().item()),
        "mu_star_norm_mean": float(star["mu_star"].norm(dim=-1).mean().item()),
        "mu_u_norm_mean": float(mu_u.norm(dim=-1).mean().item()),
        "r_std": r_std,
        "G_fro_mean": float(G.norm(dim=(1, 2)).mean().item()),
        "phiTr_norm_mean": float(
            torch.einsum("tnj,tn->tj", phi_amp, r).norm(dim=-1).mean().item()
        ),
    }
    if beta_u_train is not None and float(beta_u_train) != float(beta_K):
        star_bu = elbo_star_solve(
            phi_amp, r, G, kern["Kuu_inv"], sigma_n,
            beta_G=beta_G, beta_K=float(beta_u_train), sigma_n_scale=sigma_n_scale,
        )
        out["ratio_zeta_star_beta_u_r"] = _zeta_ratio(star_bu["zeta_star"], r)
        out["beta_u_train"] = float(beta_u_train)
    return out


@torch.no_grad()
def beta_shrinkage_grid(
    regions,
    proj,
    hyperparams,
    u_params,
    *,
    betas: Iterable[float] = (0.0, 0.1, 0.3, 1.0),
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> list[dict[str, float]]:
    """Full grid over ``(beta_G, beta_K)`` plus 1D marginals at baseline."""
    beta_list = list(betas)
    rows: list[dict[str, float]] = []
    for bG in beta_list:
        for bK in beta_list:
            row = elbo_star_metrics(
                regions, proj, hyperparams, u_params,
                beta_G=bG, beta_K=bK,
                use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
            )
            row["sweep"] = "grid"
            rows.append(row)
    for bK in beta_list:
        row = elbo_star_metrics(
            regions, proj, hyperparams, u_params,
            beta_G=1.0, beta_K=bK,
            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
        )
        row["sweep"] = "beta_K_only"
        rows.append(row)
    for bG in beta_list:
        row = elbo_star_metrics(
            regions, proj, hyperparams, u_params,
            beta_G=bG, beta_K=1.0,
            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
        )
        row["sweep"] = "beta_G_only"
        rows.append(row)
    return rows


@torch.no_grad()
def sigma_n_shrinkage_sweep(
    regions,
    proj,
    hyperparams,
    u_params,
    *,
    caps: Iterable[float] | None = None,
    scales: Iterable[float] = (0.3, 0.5, 0.7, 1.0),
    use_r_std: bool = True,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> list[dict[str, float]]:
    """Cap or scale ``sigma_n`` at eval (no retrain)."""
    from experiments.synthetic.local_gp_elbo_mj import stack_m_j

    y = torch.stack([r["y"] for r in regions], dim=0)
    r = alignment_residual(y, use_m_j=True, m_j=stack_m_j(u_params))
    r_std = float(r.std().item()) + 1e-8

    rows: list[dict[str, float]] = []
    row0 = elbo_star_metrics(
        regions, proj, hyperparams, u_params,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    row0["sweep"] = "learned_sigma_n"
    rows.append(row0)

    if caps is None:
        caps = (0.3, 0.5, 0.8, 1.0)
    for cap in caps:
        val = float(cap) * r_std if use_r_std else float(cap)
        row = elbo_star_metrics(
            regions, proj, hyperparams, u_params,
            sigma_n_cap=val,
            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
        )
        row["sweep"] = "sigma_n_cap"
        row["sigma_n_cap_value"] = val
        rows.append(row)

    for sc in scales:
        row = elbo_star_metrics(
            regions, proj, hyperparams, u_params,
            sigma_n_scale=float(sc),
            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
        )
        row["sweep"] = "sigma_n_scale"
        rows.append(row)

    for fixed in (0.3, 0.5, 0.8, 1.0):
        val = float(fixed) * r_std if use_r_std else float(fixed)
        row = elbo_star_metrics(
            regions, proj, hyperparams, u_params,
            sigma_n_fixed=val,
            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
        )
        row["sweep"] = "sigma_n_fixed"
        row["sigma_n_fixed_value"] = val
        rows.append(row)
    return rows


def format_beta_grid_table(rows: list[dict[str, float]], betas: tuple[float, ...]) -> str:
    """Pretty-print ``z_star/r`` for grid slice.

    Read rows at fixed ``beta_G`` (vary ``beta_K`` left-to-right): primary K-lock test.
    Read columns at fixed ``beta_K`` (vary ``beta_G`` top-to-bottom): secondary G test.
    """
    lines = ["beta_G\\beta_K"] + [f"{bK:>8g}" for bK in betas]
    header = "".join(f"{h:>10s}" for h in lines)
    out = [header]
    for bG in betas:
        cells = [f"{bG:>10g}"]
        for bK in betas:
            hit = next(
                (r for r in rows if r.get("sweep") == "grid"
                 and r["beta_G"] == bG and r["beta_K"] == bK),
                None,
            )
            val = hit["ratio_zeta_star_r"] if hit else float("nan")
            cells.append(f"{val:>10.4f}")
        out.append("".join(cells))
    return "\n".join(out)
