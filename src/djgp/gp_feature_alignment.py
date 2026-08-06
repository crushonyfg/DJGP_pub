"""GP feature–residual alignment diagnostics and auxiliary losses.



Definitions (per region ``j``):



- ``r^c = H(y - mean(y))``  (centered residual; alignment target detached)

- ``Phi = Psi1_base @ Kuu^{-1}`` with **unit** kernel amplitude (no ``sigma_k``)

- ``zeta = Phi @ mu_u`` (learned); ridge ``zeta_ridge = Phi^c b*``



ELBO-consistent inducing mean (pure inlier, fixed ``Sigma_u`` in ``G``):



``mu_u* = (G + sigma_n^2 K^{-1})^{-1} Phi_amp^T r`` with

``G = sum_n K^{-1} Psi_{2,n} K^{-1}``, ``Phi_amp = Kfu @ K^{-1}``, ``r = y - m_j``.

"""



from __future__ import annotations



from typing import Any



import torch



from djgp.structured_projection import get_mu_W_cov_W

from djgp.variational import expected_Kfu, kl_q_u_batch





def center_rows(mat: torch.Tensor) -> torch.Tensor:

    """Apply ``H = I - 11^T/n`` on the neighbour dimension (dim=-2)."""

    return mat - mat.mean(dim=-2, keepdim=True)





def _stack_mu_u(u_params) -> torch.Tensor:

    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)

    if mu_u.dim() > 2:

        mu_u = mu_u.squeeze(-1)

    if mu_u.dim() > 2:

        mu_u = mu_u.squeeze(1)

    return mu_u





def alignment_residual(

    y: torch.Tensor,

    *,

    use_m_j: bool = False,

    m_j: torch.Tensor | None = None,

) -> torch.Tensor:

    """Alignment target; detached from learnable ``m_j`` when requested."""

    if use_m_j and m_j is not None:

        r = y - m_j.view(-1, 1)

    else:

        r = y - y.mean(dim=-1, keepdim=True)

    return r.detach()





def compute_psi1_phi(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    *,

    tau: float = 1.0,

    fixed_w: bool = False,

    detach_inducing: bool = False,

) -> dict[str, torch.Tensor]:

    """Base ``Psi1`` (``sigma_k=1``) and ``Phi = Psi1 @ Kuu^{-1}`` per region.



    ``tau`` scales ``cov(W)`` before analytic kernel expectation (diag C).

    """

    device = regions[0]["X"].device

    X = torch.stack([r["X"] for r in regions], dim=0)

    C = torch.stack([r["C"] for r in regions], dim=0)

    if detach_inducing:

        C = C.detach()

    T, n, _ = X.shape

    m1 = C.shape[1]



    mu_W, cov_W = get_mu_W_cov_W(proj, hyperparams["X_test"], hyperparams)

    if fixed_w or tau <= 0.0:

        cov_W = torch.zeros_like(cov_W)

    elif tau != 1.0:

        cov_W = cov_W * float(tau)



    ones_sk = torch.ones(T, device=device, dtype=X.dtype)

    psi1 = expected_Kfu(mu_W, cov_W, X, C, ones_sk)



    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)

    eye = torch.eye(m1, device=device, dtype=C.dtype).unsqueeze(0)

    kuu = torch.exp(-0.5 * d2) + 1e-6 * eye

    kuu_inv = torch.cholesky_inverse(torch.linalg.cholesky(0.5 * (kuu + kuu.transpose(-1, -2))))



    phi = torch.einsum("tni,tij->tnj", psi1, kuu_inv)

    return {

        "psi1": psi1,

        "phi": phi,

        "kuu_inv": kuu_inv,

        "mu_W": mu_W,

        "cov_W": cov_W,

        "X": X,

        "C": C,

    }





def compute_elbo_kernels(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    *,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, torch.Tensor]:

    """ELBO amplitude ``Kfu``, ``KufKfu``, ``Kuu_inv`` on training neighbours."""

    from djgp.structured_projection import expected_Kfu_mc

    from experiments.synthetic.local_gp_elbo_mj import gp_residual_moments



    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}

    mom = gp_residual_moments(

        regions, V_dummy, u_params, hyperparams,

        proj=proj, fixed_w=False, at_test=False,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    return {

        "Kfu": mom["Kfu"],

        "KufKfu": mom["KufKfu"],

        "Kuu_inv": mom["Kuu_inv"],

    }





@torch.no_grad()

def elbo_star_mu_u(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    *,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, torch.Tensor]:

    """``mu_u*`` and ``zeta* = Phi_amp @ mu_u*`` from ELBO quadratic in ``mu_u``.



    ``(G + sigma_n^2 K^{-1}) mu_u* = Phi_amp^T r`` with ``r = y - m_j`` (detached).

    """

    from experiments.synthetic.local_gp_elbo_mj import stack_m_j



    y = torch.stack([r["y"] for r in regions], dim=0)

    m_j = stack_m_j(u_params)

    r = alignment_residual(y, use_m_j=True, m_j=m_j)

    sigma_n = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)



    kern = compute_elbo_kernels(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    Kfu = kern["Kfu"]

    KufKfu = kern["KufKfu"]

    Kuu_inv = kern["Kuu_inv"]



    phi_amp = torch.einsum("tni,tij->tnj", Kfu, Kuu_inv)

    rhs = torch.einsum("tnj,tn->tj", phi_amp, r)

    G = torch.einsum("tij,tnjk,tkm->tim", Kuu_inv, KufKfu, Kuu_inv)



    sn2 = sigma_n.view(-1, 1, 1).pow(2)

    A = G + sn2 * Kuu_inv

    mu_star = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)

    zeta_star = torch.einsum("tnj,tj->tn", phi_amp, mu_star)

    v_star = torch.einsum("tij,tj->ti", Kuu_inv, mu_star)

    return {

        "mu_star": mu_star,

        "zeta_star": zeta_star,

        "phi_amp": phi_amp,

        "r_elbo": r,

        "Kinvt_mu_star": v_star,

    }





@torch.no_grad()

def elbo_star_mu_u_custom_rho(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    rho: torch.Tensor,

    *,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, torch.Tensor]:

    """Star solve with arbitrary per-point weights ``rho`` ``[T,n]``."""

    from experiments.synthetic.local_gp_elbo_mj import stack_m_j



    y = torch.stack([r["y"] for r in regions], dim=0)

    m_j = stack_m_j(u_params)

    r = alignment_residual(y, use_m_j=True, m_j=m_j)

    sigma_n = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)

    kern = compute_elbo_kernels(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    Kfu = kern["Kfu"]

    KufKfu = kern["KufKfu"]

    Kuu_inv = kern["Kuu_inv"]

    phi_amp = torch.einsum("tni,tij->tnj", Kfu, Kuu_inv)

    wr = (rho.to(dtype=phi_amp.dtype) * r)

    rhs = torch.einsum("tnj,tn->tj", phi_amp, wr)

    G_rho = torch.einsum(

        "tij,tn,tnjk,tkm->tim", Kuu_inv, rho.to(dtype=Kuu_inv.dtype), KufKfu, Kuu_inv,

    )

    sn2 = sigma_n.view(-1, 1, 1).pow(2)

    m1 = G_rho.shape[-1]

    eye = torch.eye(m1, device=G_rho.device, dtype=G_rho.dtype).unsqueeze(0)

    A = G_rho + sn2 * Kuu_inv + 1e-8 * eye

    mu_star = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)

    zeta_star = torch.einsum("tnj,tj->tn", phi_amp, mu_star)

    v_star = torch.einsum("tij,tj->ti", Kuu_inv, mu_star)

    return {

        "mu_star": mu_star,

        "zeta_star": zeta_star,

        "phi_amp": phi_amp,

        "r_elbo": r,

        "rho": rho,

        "Kinvt_mu_star": v_star,

    }





@torch.no_grad()

def elbo_star_mu_u_rho(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    V_params,

    *,

    gate_config: dict[str, Any] | None = None,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, torch.Tensor]:

    """Mixture-consistent star with ELBO posterior ``rho_i``."""

    from experiments.synthetic.local_gp_elbo_mj import mixture_scores_rho_mj



    gate_config = gate_config or {"gate_mode": "learned"}

    _, _, rho = mixture_scores_rho_mj(

        regions, V_params, u_params, hyperparams,

        proj=proj, gate_config=gate_config,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    out = elbo_star_mu_u_custom_rho(

        regions, proj, hyperparams, u_params, rho,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    return out





@torch.no_grad()

def mixture_elbo_star_metrics(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    V_params,

    *,

    gate_config: dict[str, Any] | None = None,

    rho_tilde: torch.Tensor | None = None,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, float]:

    """Pure vs rho-weighted ELBO stars and learned ``zeta`` on ``r_elbo``."""

    def _zr(z: torch.Tensor, r_t: torch.Tensor) -> float:
        return float(z.std().item() / (r_t.std().item() + 1e-8))



    star_pure = elbo_star_mu_u(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    star_rho = elbo_star_mu_u_rho(

        regions, proj, hyperparams, u_params, V_params,

        gate_config=gate_config,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    star_tilde = None

    if rho_tilde is not None:

        star_tilde = elbo_star_mu_u_custom_rho(

            regions, proj, hyperparams, u_params, rho_tilde,

            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

        )

    r = star_pure["r_elbo"]

    mu_u = _stack_mu_u(u_params)

    phi_amp = star_pure["phi_amp"]

    z_learn = torch.einsum("tnj,tj->tn", phi_amp, mu_u)

    cos = torch.nn.functional.cosine_similarity

    rho_w = rho_tilde if star_tilde is not None else star_rho["rho"]

    out: dict[str, float] = {

        "ratio_zeta_star_pure_r": _zr(star_pure["zeta_star"], r),

        "ratio_zeta_star_rho_r": _zr(star_rho["zeta_star"], r),

        "ratio_zeta_learned_r_elbo": _zr(z_learn, r),

        "rho_batch_mean": float(star_rho["rho"].mean().item()),

        "mu_u_cos_star_pure": float(cos(mu_u, star_pure["mu_star"], dim=-1).mean().item()),

        "mu_u_cos_star_rho": float(cos(mu_u, star_rho["mu_star"], dim=-1).mean().item()),

        "mu_star_rho_norm_mean": float(star_rho["mu_star"].norm(dim=-1).mean().item()),

        **mixture_rho_signal_diagnostics(

            star_pure["phi_amp"], star_pure["r_elbo"], rho_w,

            regions, proj, hyperparams, u_params,

            use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

        ),

    }

    if star_tilde is not None:

        out["ratio_zeta_star_rho_tilde_r"] = _zr(star_tilde["zeta_star"], r)

        out["rho_tilde_mean"] = float(rho_tilde.mean().item())

        out["mu_u_cos_star_rho_tilde"] = float(

            cos(mu_u, star_tilde["mu_star"], dim=-1).mean().item()

        )

    return out





def _pearson_1d(x: torch.Tensor, y: torch.Tensor) -> float:

    x = x.reshape(-1).float()

    y = y.reshape(-1).float()

    xc = x - x.mean()

    yc = y - y.mean()

    den = xc.norm() * yc.norm()

    if float(den) < 1e-12:

        return float("nan")

    return float((xc * yc).sum() / den)





@torch.no_grad()

def mixture_rho_signal_diagnostics(

    phi_amp: torch.Tensor,

    r: torch.Tensor,

    rho: torch.Tensor,

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    *,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, float]:

    """Decompose why ``Phi^T D_rho r`` can be much smaller than ``Phi^T r``."""

    r = r.to(dtype=phi_amp.dtype)

    rho = rho.to(dtype=phi_amp.dtype)

    phiTr_pure = torch.einsum("tnj,tn->tj", phi_amp, r)

    phiTr_rho = torch.einsum("tnj,tn->tj", phi_amp, rho * r)

    B_pure = phiTr_pure.norm(dim=-1)

    B_rho = phiTr_rho.norm(dim=-1)

    B_pure_mean = float(B_pure.mean().item())

    B_rho_mean = float(B_rho.mean().item())

    rho_bar = float(rho.mean().item())

    b_ridge = ridge_coefficients(phi_amp, r, ridge_lambda=1e-2)

    z_ridge = torch.einsum("tnj,tj->tn", phi_amp, b_ridge)

    cos_phiTr = float(

        torch.nn.functional.cosine_similarity(

            phiTr_pure.reshape(1, -1), phiTr_rho.reshape(1, -1),

        ).item()

    )

    r_flat = r.reshape(-1)

    rho_flat = rho.reshape(-1)

    abs_r = r_flat.abs()

    r2 = r_flat.pow(2)

    rho_sum = float(rho_flat.sum().clamp_min(1e-8).item())

    wr2_mean = float((rho_flat * r2).sum().item() / rho_sum)

    r2_mean = float(r2.mean().item())



    kern = compute_elbo_kernels(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    Kuu_inv = kern["Kuu_inv"]

    KufKfu = kern["KufKfu"]

    G_pure = torch.einsum("tij,tnjk,tkm->tim", Kuu_inv, KufKfu, Kuu_inv)

    G_rho = torch.einsum(

        "tij,tn,tnjk,tkm->tim", Kuu_inv, rho, KufKfu, Kuu_inv,

    )

    G_pure_fro = float(G_pure.norm(dim=(1, 2)).mean().item())

    G_rho_fro = float(G_rho.norm(dim=(1, 2)).mean().item())



    return {

        "phiTr_pure_norm_mean": B_pure_mean,

        "phiTr_rho_norm_mean": B_rho_mean,

        "phiTr_rho_over_pure": B_rho_mean / max(B_pure_mean, 1e-8),

        "phiTr_rho_over_rhobar_pure": B_rho_mean / max(rho_bar * B_pure_mean, 1e-8),

        "cos_phiTr_pure_rho": cos_phiTr,

        "corr_rho_abs_r": _pearson_1d(rho_flat, abs_r),

        "corr_rho_abs_zridge": _pearson_1d(rho_flat, z_ridge.reshape(-1).abs()),

        "corr_rho_r2": _pearson_1d(rho_flat, r2),

        "weighted_r2_mean": wr2_mean,

        "r2_mean": r2_mean,

        "weighted_r2_over_mean": wr2_mean / max(r2_mean, 1e-8),

        "G_rho_fro_mean": G_rho_fro,

        "G_pure_fro_mean": G_pure_fro,

        "G_rho_over_rhobar_pure": G_rho_fro / max(rho_bar * G_pure_fro, 1e-8),

    }





def signal_strength(phi_c: torch.Tensor, r_c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    """Per-region signal in ``[0,1]`` scale; shape ``[T]``."""

    cross = torch.einsum("tnm,tn->tm", phi_c, r_c)

    num = cross.norm(dim=-1)

    den = phi_c.norm(dim=(1, 2)) * r_c.norm(dim=-1) + eps

    return num / den





def ridge_coefficients(

    phi_c: torch.Tensor,

    r_c: torch.Tensor,

    ridge_lambda: float = 1e-2,

) -> torch.Tensor:

    """``b* = (Phi^{cT} Phi^c + lam I)^{-1} Phi^{cT} r^c``; shape ``[T, m1]``."""

    T, n, m1 = phi_c.shape

    device, dtype = phi_c.device, phi_c.dtype

    gram = torch.einsum("tni,tnj->tij", phi_c, phi_c)

    rhs = torch.einsum("tni,tn->ti", phi_c, r_c)

    eye = torch.eye(m1, device=device, dtype=dtype).unsqueeze(0)

    return torch.linalg.solve(gram + float(ridge_lambda) * eye, rhs)





def ridge_r2(phi_c: torch.Tensor, r_c: torch.Tensor, ridge_lambda: float = 1e-2) -> torch.Tensor:

    """Per-region ridge ``R^2``; shape ``[T]``."""

    eps = 1e-8

    b = ridge_coefficients(phi_c, r_c, ridge_lambda)

    r_hat = torch.einsum("tnm,tm->tn", phi_c, b)

    ss_res = (r_c - r_hat).pow(2).sum(dim=-1)

    ss_tot = r_c.pow(2).sum(dim=-1) + eps

    return 1.0 - ss_res / ss_tot





def cov_alignment_loss(phi_c: torch.Tensor, r_c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    """Normalized ``|Phi^{cT} r^c|^2`` (mean over regions); maximize in ELBO."""

    cross = torch.einsum("tnm,tn->tm", phi_c, r_c)

    num = cross.pow(2).sum(dim=-1)

    den = phi_c.pow(2).sum(dim=(1, 2)) * r_c.pow(2).sum(dim=-1) + eps

    return (num / den).mean()





def ridge_fit_loss(

    phi_c: torch.Tensor,

    r_c: torch.Tensor,

    ridge_lambda: float = 1e-2,

    *,

    detach_b: bool = True,

) -> torch.Tensor:

    """``1 - R^2`` with optional ``detach(b*)`` for stable gradients."""

    b = ridge_coefficients(phi_c, r_c, ridge_lambda)

    if detach_b:

        b = b.detach()

    r_hat = torch.einsum("tnm,tm->tn", phi_c, b)

    eps = 1e-8

    ss_res = (r_c - r_hat).pow(2).sum(dim=-1)

    ss_tot = r_c.pow(2).sum(dim=-1) + eps

    return (ss_res / ss_tot).mean()





@torch.no_grad()

def qu_diagnostics(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    *,

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, float]:

    """``sigma_n``, ``sigma_k``, ``KL_u``, ``|mu_u|``, ``|K^{-1} mu_u|``."""

    from experiments.synthetic.local_gp_elbo_mj import _psd_sigma_u



    C = torch.stack([r["C"] for r in regions], dim=0)

    mu_u = _stack_mu_u(u_params)

    Sigma_u = _psd_sigma_u(u_params)

    sigma_n = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)

    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)



    kern = compute_elbo_kernels(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    Kinvt_mu = torch.einsum("tij,tj->ti", kern["Kuu_inv"], mu_u)



    kl_u = kl_q_u_batch(C, mu_u, Sigma_u, torch.tensor(1.0, device=C.device))



    return {

        "sigma_n_mean": float(sigma_n.mean().item()),

        "sigma_n_std": float(sigma_n.std().item()),

        "sigma_k_mean": float(sigma_k.mean().item()),

        "sigma_k_std": float(sigma_k.std().item()),

        "KL_u_mean": float(kl_u.mean().item()),

        "KL_u_sum": float(kl_u.sum().item()),

        "mu_u_norm_mean": float(mu_u.norm(dim=-1).mean().item()),

        "Kinvt_mu_norm_mean": float(Kinvt_mu.norm(dim=-1).mean().item()),

    }





@torch.no_grad()

def region_alignment_diagnostics(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    u_params,

    *,

    ridge_lambda: float = 1e-2,

    tau_list: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),

    use_mc_kfu: bool = False,

    mc_samples: int = 5,

) -> dict[str, float]:

    """Aggregate diagnostics A/B/C, ELBO ``mu_u*``, and ``q(u)`` scalars."""

    y = torch.stack([r["y"] for r in regions], dim=0)

    mu_u = _stack_mu_u(u_params)

    r_align = alignment_residual(y, use_m_j=False)

    r_c = center_rows(r_align)



    out: dict[str, float] = {}

    for tau in tau_list:

        fixed = tau <= 0.0

        pack = compute_psi1_phi(regions, proj, hyperparams, tau=tau, fixed_w=fixed)

        phi_c = center_rows(pack["phi"])

        sig = signal_strength(phi_c, r_c)

        r2 = ridge_r2(phi_c, r_c, ridge_lambda)

        b = ridge_coefficients(phi_c, r_c, ridge_lambda)

        z_ridge = torch.einsum("tnm,tm->tn", phi_c, b)

        z_learn = torch.einsum("tnm,tm->tn", pack["phi"], mu_u)

        tag = "tau0" if fixed else f"tau{tau:g}"

        out[f"signal_{tag}"] = float(sig.mean().item())

        out[f"r2_{tag}"] = float(r2.mean().item())

        r_std = float(r_c.std().item()) + 1e-8

        out[f"zeta_ridge_std_{tag}"] = float(z_ridge.std().item())

        out[f"zeta_learned_std_{tag}"] = float(z_learn.std().item())

        out[f"ratio_zeta_ridge_r_{tag}"] = float(z_ridge.std().item() / r_std)

        out[f"ratio_zeta_learned_r_{tag}"] = float(z_learn.std().item() / r_std)



    phi1 = compute_psi1_phi(regions, proj, hyperparams, tau=1.0, fixed_w=False)

    cross_mu = torch.einsum("tnm,tm->tn", phi1["phi"], mu_u)

    out["phiTr_norm_mean"] = float(cross_mu.norm(dim=-1).mean().item())



    star = elbo_star_mu_u(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    )

    r_elbo = star["r_elbo"]
    r_elbo_std = float(r_elbo.std().item()) + 1e-8
    phi_amp = star["phi_amp"]
    z_star = star["zeta_star"]
    z_learn_elbo = torch.einsum("tnj,tj->tn", phi_amp, mu_u)
    b_elbo = ridge_coefficients(phi_amp, r_elbo, ridge_lambda)
    z_ridge_elbo = torch.einsum("tnj,tj->tn", phi_amp, b_elbo)
    out["ratio_zeta_elbo_star_r"] = float(z_star.std().item() / r_elbo_std)
    out["ratio_zeta_learned_r_elbo"] = float(z_learn_elbo.std().item() / r_elbo_std)
    out["ratio_zeta_ridge_r_elbo"] = float(z_ridge_elbo.std().item() / r_elbo_std)
    out["r2_ridge_elbo"] = float(ridge_r2(phi_amp, r_elbo, ridge_lambda).mean().item())
    out["mu_star_norm_mean"] = float(star["mu_star"].norm(dim=-1).mean().item())

    out["Kinvt_mu_star_norm_mean"] = float(star["Kinvt_mu_star"].norm(dim=-1).mean().item())

    out["mu_u_vs_star_cos_mean"] = float(

        torch.nn.functional.cosine_similarity(mu_u, star["mu_star"], dim=-1).mean().item()

    )



    out.update(qu_diagnostics(

        regions, proj, hyperparams, u_params,

        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,

    ))

    return out





def alignment_losses_train(

    regions,

    proj: dict[str, Any],

    hyperparams: dict[str, Any],

    *,

    lambda_cov: float = 0.0,

    lambda_ridge: float = 0.0,

    ridge_lambda: float = 1e-2,

    tau: float = 1.0,

    cov_clip: float = 1.0,

    detach_inducing: bool = True,

) -> tuple[torch.Tensor, dict[str, float]]:

    """Differentiable alignment terms to **add** to ELBO (before ``-elbo`` minimization).



    Gradients flow only through projection / ``W`` path when ``detach_inducing=True``.

    """

    y = torch.stack([r["y"] for r in regions], dim=0)

    r_align = alignment_residual(y, use_m_j=False)

    r_c = center_rows(r_align)

    pack = compute_psi1_phi(

        regions, proj, hyperparams, tau=tau, fixed_w=False,

        detach_inducing=detach_inducing,

    )

    phi_c = center_rows(pack["phi"])



    parts: dict[str, float] = {}

    total = torch.tensor(0.0, device=y.device, dtype=y.dtype)

    if lambda_cov > 0:

        lc = cov_alignment_loss(phi_c, r_c)

        if cov_clip > 0:

            lc = lc.clamp(max=float(cov_clip))

        parts["L_cov"] = float(lc.item())

        total = total + float(lambda_cov) * lc

    if lambda_ridge > 0:

        lr = ridge_fit_loss(phi_c, r_c, ridge_lambda, detach_b=True)

        parts["L_ridge"] = float(lr.item())

        total = total + float(lambda_ridge) * (-lr)

    return total, parts





def refresh_mu_u_toward_star(

    u_params,

    mu_star: torch.Tensor,

    alpha: float,

) -> None:

    """``mu_u <- (1-alpha) mu_u + alpha mu_u*`` in-place."""

    a = float(alpha)

    with torch.no_grad():

        for t, u in enumerate(u_params):

            u["mu_u"].mul_(1.0 - a).add_(a * mu_star[t])


