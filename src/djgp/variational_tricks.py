"""
Optional training-time tricks for LMJGP (kept additive to ``djgp.variational``).

Implements four knobs independently or together:

1. **KL warm-up on q(W)**: scale ``KL(q(V) || p(V))`` by ``beta_W(step)`` increasing from a
   small value (e.g. 0.01) to ``1.0``.
2. **Supervised init for mu_V**: initialise ``mu_V[m, q, d] = W_init[q, d]`` for every
   inducing point ``m`` so that ``mu_W ≈ W_init`` after the kernel interpolation. ``W_init``
   comes from PLS or SIR; this breaks the rotation/sign ambiguity that traps ``E[W]`` near 0.
3. **Training-time row normalisation penalty** on ``E[W]``: encourages
   ``||mu_W_{t,q,:}||^2 ≈ 1``, matching what ``predict_vi`` does at sample time.
4. **Auxiliary metric loss**: maximises Pearson correlation between projected pairwise
   squared distances ``|| W_t (x_i - x_j) ||^2`` and observed response squared distances
   ``(y_i - y_j)^2`` per region. Weakly forces ``W_t`` toward locally response-aligned
   directions without distorting the ELBO too much.

The original ``djgp.variational.train_vi`` and ``predict_vi`` are unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.variational import (
    expected_Kfu,
    expected_KufKfu,
    expected_log_sigmoid_gh_batch,
    kl_qp,
    kl_q_u_batch,
    qW_from_qV,
)

_LOG_2PI = math.log(2 * math.pi)


# =====================================================================
# 1. KL warm-up schedule
# =====================================================================
@dataclass(frozen=True)
class KLWarmupSchedule:
    """``beta_W(step)`` ramp."""

    beta_init: float = 0.01
    beta_final: float = 1.0
    warmup_steps: int = 200
    shape: str = "linear"  # "linear" | "cosine"

    def __call__(self, step: int) -> float:
        if step <= 0:
            return float(self.beta_init)
        if step >= self.warmup_steps:
            return float(self.beta_final)
        frac = step / max(1, self.warmup_steps)
        if self.shape == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        return float(self.beta_init + (self.beta_final - self.beta_init) * frac)


# =====================================================================
# 2. Supervised initialisation for mu_V
# =====================================================================
def _pls_directions(X: np.ndarray, y: np.ndarray, Q: int) -> np.ndarray:
    """Return ``W ∈ R^{Q×D}`` from PLSRegression weights (raw X scale)."""
    from sklearn.cross_decomposition import PLSRegression

    pls = PLSRegression(n_components=Q, scale=False)
    pls.fit(X, y.reshape(-1, 1))
    W = np.asarray(pls.x_weights_).T  # [Q, D]
    return W


def _sir_directions(X: np.ndarray, y: np.ndarray, Q: int, H: int = 10) -> np.ndarray:
    """
    Return ``W ∈ R^{Q×D}`` mapping raw X to top-``Q`` SIR directions.

    Implements the same slicing-and-Cov estimator as
    ``experiments.uci.uci_data.sir_reduction`` but without importing it (that module pulls
    in ``ucimlrepo`` at top level).
    """
    from scipy.linalg import eigh
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).ravel()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(np.asarray(X))
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)

    percentiles = np.linspace(0, 100, H + 1)[1:-1]
    bin_edges = np.percentile(y, percentiles)
    y_binned = np.digitize(y, bin_edges)
    means = np.zeros((H, X_std.shape[1]))
    props = np.zeros(H)
    for h in range(H):
        idx = y_binned == h
        if np.any(idx):
            means[h] = np.nan_to_num(X_std[idx].mean(axis=0))
            props[h] = idx.mean()
    s = np.sum(props)
    if s > 0:
        props = props / s
    V = np.zeros((X_std.shape[1], X_std.shape[1]))
    for h in range(H):
        if props[h] > 0:
            V += props[h] * np.outer(means[h], means[h])
    V += 1e-8 * np.eye(V.shape[0])
    try:
        _, eigvecs = eigh(V)
    except np.linalg.LinAlgError:
        _, eigvecs = np.linalg.svd(V)[:2]
        eigvecs = eigvecs.T
    selected = eigvecs[:, -Q:]  # [D, Q]
    inv_scale = 1.0 / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    W = (selected.T * inv_scale[None, :])  # [Q, D] in raw X space
    return np.asarray(W, dtype=np.float64)


def init_mu_V_supervised(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    m2: int,
    Q: int,
    mode: str = "pls",
    sir_H: int = 10,
    perturb_std: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
    return_W: bool = False,
):
    """
    Build ``mu_V_init`` of shape ``[m2, Q, D]`` so that for every inducing point ``m`` the
    projection mean equals (up to small noise) the supervised direction matrix ``W_init``.
    Because ``mu_W = (Kxz Kzz^{-1}) mu_V`` and ``mu_V`` is constant in ``m``, ``mu_W`` ends
    up as ``(row-sum of weights) * W_init`` for an RBF interpolant — which is **only**
    approximately ``W_init`` when the inducing points + lengthscales actually cover the
    test points. When the kernel damps the interpolation (small lengthscales, sparse Z),
    the user must follow this with :func:`calibrate_mu_V_to_target_mu_W` so ``mu_W`` at
    the start of training really matches the supervised directions.

    Parameters
    ----------
    return_W : if True, also return ``W_init`` as a torch tensor on ``device`` (useful for
        post-init calibration).

    Returns
    -------
    mu_V_init : torch.Tensor ``[m2, Q, D]`` (requires_grad=False; caller should clone +
        ``requires_grad_(True)``).
    info : dict with ``W_init`` numpy stats.
    [W_init_tensor] : optional ``[Q, D]`` torch tensor when ``return_W=True``.
    """
    if mode not in ("pls", "sir"):
        raise ValueError(f"mode must be 'pls' or 'sir', got {mode!r}.")
    device = device or X_train.device
    Xn = X_train.detach().cpu().numpy().astype(np.float64)
    yn = y_train.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if mode == "pls":
        W = _pls_directions(Xn, yn, Q=Q)
    else:
        W = _sir_directions(Xn, yn, Q=Q, H=sir_H)
    W_t = torch.from_numpy(W).to(device=device, dtype=X_train.dtype)
    D = W_t.shape[-1]
    mu_V = W_t.unsqueeze(0).expand(m2, Q, D).contiguous().clone()
    if perturb_std > 0:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        mu_V = mu_V + perturb_std * torch.randn(mu_V.shape, generator=g, device=device, dtype=mu_V.dtype)
    info = {
        "mode": mode,
        "W_init_frob": float(np.linalg.norm(W)),
        "W_init_trace_WtW": float((W @ W.T).trace().sum()) if W.shape[0] >= 1 else float("nan"),
        "W_init_shape": tuple(W.shape),
    }
    if return_W:
        return mu_V, info, W_t
    return mu_V, info


def calibrate_mu_V_to_target_mu_W(
    V_params: dict,
    hyperparams: dict,
    X_target: torch.Tensor,
    W_target: torch.Tensor,
    *,
    eps: float = 1e-12,
    per_row: bool = True,
    max_alpha: float = 5.0,
    target_row_norm: float | None = None,
) -> dict[str, float]:
    """
    Scale ``V_params['mu_V']`` so that ``mu_W = qW_from_qV(X_target, ...)`` reaches a
    reasonable magnitude in the direction of ``W_target``.

    The GP interpolation ``A = K_xz K_zz^{-1}`` is **not** row-stochastic for an SE
    kernel; small lengthscales / sparse inducing points make the row-sums tiny and damp
    ``mu_V`` by an unknown factor. We rescale by that factor.

    To avoid blowing up ``KL_V`` we (a) cap the multiplier at ``max_alpha`` and (b) optionally
    target a fixed row-norm instead of matching the (potentially much larger) magnitude of
    ``W_target``. The default ``target_row_norm=None`` matches ``||W_target_q||`` per row;
    set ``target_row_norm=1.0`` for a stable, KL-friendly initialisation.

    Parameters
    ----------
    per_row : if True, scale each latent row ``q`` by its own ratio. Otherwise use a
        single scalar across all rows.
    max_alpha : per-element cap on the multiplier; prevents ``KL_V`` blow-up when the
        kernel essentially zeros ``mu_W``.
    target_row_norm : if set, target this magnitude per row (regardless of ``W_target``
        norms); useful when combined with the second-moment row-norm penalty.
    """
    out: dict[str, float] = {}
    with torch.no_grad():
        mu_W_pre, _ = qW_from_qV(
            X_target,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        out["mu_W_frob_pre"] = float(torch.linalg.norm(mu_W_pre).item())
        Wt = W_target.to(device=mu_W_pre.device, dtype=mu_W_pre.dtype)
        if per_row:
            actual_per_q = torch.linalg.norm(mu_W_pre, dim=-1).mean(dim=0)  # [Q]
            if target_row_norm is None:
                target_per_q = torch.linalg.norm(Wt, dim=-1)
            else:
                target_per_q = torch.full_like(actual_per_q, float(target_row_norm))
            alphas = (target_per_q / actual_per_q.clamp_min(eps)).clamp(max=float(max_alpha))
            V_params["mu_V"].data.mul_(alphas.view(1, -1, 1))
            out["alphas_q"] = [float(a.item()) for a in alphas]
            out["alphas_capped_at"] = float(max_alpha)
        else:
            a_norm = torch.linalg.norm(mu_W_pre, dim=-1).mean()
            if target_row_norm is None:
                t_norm = torch.linalg.norm(Wt) / max(1, Wt.shape[0])
            else:
                t_norm = torch.tensor(float(target_row_norm), device=a_norm.device, dtype=a_norm.dtype)
            alpha = float((t_norm / a_norm.clamp_min(eps)).clamp(max=float(max_alpha)).item())
            V_params["mu_V"].data.mul_(alpha)
            out["alpha"] = alpha
            out["alpha_capped_at"] = float(max_alpha)
        mu_W_post, _ = qW_from_qV(
            X_target,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        out["mu_W_frob_post"] = float(torch.linalg.norm(mu_W_post).item())
        out["target_W_frob"] = float(torch.linalg.norm(Wt).item())
    return out


def init_lengthscales_median_heuristic(
    X_target: torch.Tensor,
    Z: torch.Tensor,
    *,
    Q: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Median pairwise distance between target inputs and inducing points. Used to seed
    ``hyperparams['lengthscales']`` so the SE kernel produces O(1) interpolation weights
    at step 0 — without this the supervised init for ``mu_V`` damps to ~0 in ``mu_W``.

    Returns a ``[Q]`` tensor (same value across all latent dims).
    """
    with torch.no_grad():
        pd = torch.cdist(X_target, Z, p=2.0)
        med = pd.median().clamp_min(eps)
    return torch.full((Q,), float(med.item()), device=Z.device, dtype=Z.dtype)


# =====================================================================
# 3. Row-norm penalties (mu_W or full second moment)
# =====================================================================
def row_norm_penalty_mu_W(mu_W: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    """``mean((||mu_W_{t,q,:}||^2 - target)^2)`` — first-moment only."""
    n2 = (mu_W * mu_W).sum(dim=-1)
    return ((n2 - target) ** 2).mean()


def row_norm_penalty_E_norm_sq(
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    target: float = 1.0,
) -> torch.Tensor:
    """
    Penalty on the **second moment** ``E[||W_{t,q,:}||^2] = ||mu_W_{t,q,:}||^2 + sum_d Var(W_{t,q,d})``.

    Matches what ``predict_vi`` actually uses (it samples ``W = mu_W + sigma_W * eps`` then
    row-normalises). When ``mu_W ≈ 0`` the gradient of the first-moment penalty vanishes,
    but this version still has a non-trivial gradient through ``cov_W`` and through
    ``mu_V, sigma_V, lengthscales, var_w, Z`` (because both ``mu_W`` and ``cov_W`` are
    differentiable functions of those parameters).
    """
    n2_mean = (mu_W * mu_W).sum(dim=-1)  # [T, Q]
    var_sum = cov_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [T, Q]
    return ((n2_mean + var_sum - target) ** 2).mean()


# =====================================================================
# 4. Auxiliary metric-learning losses
# =====================================================================
def _pair_indices(
    n: int,
    *,
    max_pairs: int | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (i, j) index pairs (i != j) for the given neighbourhood size."""
    if n < 3:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty
    n_pairs_full = n * (n - 1) // 2
    if max_pairs is None or max_pairs >= n_pairs_full:
        iu = torch.triu_indices(n, n, offset=1, device=device)
        return iu[0], iu[1]
    i_idx = torch.randint(0, n, (max_pairs,), generator=generator, device=device)
    j_idx = torch.randint(0, n, (max_pairs,), generator=generator, device=device)
    keep = i_idx != j_idx
    return i_idx[keep], j_idx[keep]


def auxiliary_metric_corr_loss(
    mu_W: torch.Tensor,
    regions: list[dict],
    *,
    max_pairs_per_region: int = 64,
    eps: float = 1e-12,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Mean per-region negative Pearson correlation between projected sq-distances
    ``||mu_W_t (x_i - x_j)||^2`` and ``(y_i - y_j)^2`` over sampled pairs.

    Note: this uses **only the posterior mean** ``mu_W``. When ``mu_W ≈ 0`` the projected
    sq-distances are nearly constant and the gradient is weak — prefer
    :func:`auxiliary_metric_corr_loss_A` in that regime.
    """
    losses: list[torch.Tensor] = []
    g = None
    if seed is not None:
        g = torch.Generator(device=mu_W.device)
        g.manual_seed(int(seed))
    for t, r in enumerate(regions):
        X = r["X"]
        y = r["y"]
        n = X.shape[0]
        i_idx, j_idx = _pair_indices(n, max_pairs=max_pairs_per_region, generator=g, device=X.device)
        if i_idx.numel() < 3:
            continue
        Wt = mu_W[t]
        Z = X @ Wt.T
        d_proj = (Z[i_idx] - Z[j_idx]).pow(2).sum(dim=-1)
        d_y = (y[i_idx] - y[j_idx]).pow(2)
        a = d_proj - d_proj.mean()
        b = d_y - d_y.mean()
        denom = (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(eps)
        corr = (a * b).mean() / denom
        losses.append(-corr)
    if not losses:
        return torch.zeros((), device=mu_W.device, dtype=mu_W.dtype)
    return torch.stack(losses).mean()


def auxiliary_metric_corr_loss_A(
    A: torch.Tensor,
    regions: list[dict],
    *,
    max_pairs_per_region: int = 64,
    eps: float = 1e-12,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Same idea as :func:`auxiliary_metric_corr_loss` but uses the full Mahalanobis metric
    ``A_t = E_q[W_t^T W_t] in R^{D×D}`` instead of ``mu_W``::

        d^2_{A_t}(x_i, x_j) = (x_i - x_j)^T A_t (x_i - x_j).

    Because ``A_t`` includes posterior variance, gradients flow through both ``mu_W`` and
    ``cov_W``; this still has signal even when ``E[W] ≈ 0``.
    """
    losses: list[torch.Tensor] = []
    g = None
    if seed is not None:
        g = torch.Generator(device=A.device)
        g.manual_seed(int(seed))
    for t, r in enumerate(regions):
        X = r["X"]
        y = r["y"]
        n = X.shape[0]
        i_idx, j_idx = _pair_indices(n, max_pairs=max_pairs_per_region, generator=g, device=X.device)
        if i_idx.numel() < 3:
            continue
        delta = X[i_idx] - X[j_idx]  # [P, D]
        At = A[t]
        d_proj = torch.einsum("pd,de,pe->p", delta, At, delta)
        d_y = (y[i_idx] - y[j_idx]).pow(2)
        a = d_proj - d_proj.mean()
        b = d_y - d_y.mean()
        denom = (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(eps)
        corr = (a * b).mean() / denom
        losses.append(-corr)
    if not losses:
        return torch.zeros((), device=A.device, dtype=A.dtype)
    return torch.stack(losses).mean()


def metric_A_from_moment_local(mu_W: torch.Tensor, cov_W: torch.Tensor) -> torch.Tensor:
    """``A_j = mu_W^T mu_W + sum_q cov_W_{j,q,:,:}`` — same definition as
    :func:`djgp.diagnostics.projection_posterior.metric_A_from_moment` but kept here so
    this module is self-contained and avoids circular imports."""
    A_mean = torch.einsum("tqd,tqe->tde", mu_W, mu_W)
    A_cov = cov_W.sum(dim=1)
    return A_mean + A_cov


# =====================================================================
# 5. Objective with knobs (mirrors compute_ELBO)
# =====================================================================
def compute_objective_with_tricks(
    regions: list[dict],
    V_params: dict,
    u_params: list[dict],
    hyperparams: dict,
    *,
    beta_W: float = 1.0,
    row_norm_weight: float = 0.0,
    row_norm_mode: str = "second_moment",
    row_norm_target: float = 1.0,
    aux_metric_weight: float = 0.0,
    aux_metric_mode: str = "A",
    aux_max_pairs: int = 64,
    aux_seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Returns ``(loss_to_minimise, breakdown_dict)`` where
    ``loss = -(data_term/T - beta_W * KL_V/T - KL_u/T) + row_norm_weight * R_row + aux_metric_weight * R_aux``.

    ``row_norm_mode``:
        - ``"mu"``            — ``(||mu_W,k:||^2 - target)^2`` (legacy, has zero gradient at mu_W=0).
        - ``"second_moment"`` — ``(E[||W_,k:||^2] - target)^2`` using ``mu_W^T mu_W + diag(cov_W)``.

    ``aux_metric_mode``:
        - ``"mu"`` — Pearson(d^2_proj_under_mu_W, d^2_y).
        - ``"A"``  — Pearson(d^2_proj_under_A, d^2_y), still informative when E[W]=0.
    """
    if row_norm_mode not in ("mu", "second_moment"):
        raise ValueError(f"row_norm_mode must be 'mu' or 'second_moment', got {row_norm_mode!r}")
    if aux_metric_mode not in ("mu", "A"):
        raise ValueError(f"aux_metric_mode must be 'mu' or 'A', got {aux_metric_mode!r}")
    T = len(regions)
    device = regions[0]["X"].device

    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)

    U_logit = torch.stack([u["U_logit"] for u in u_params], dim=0).view(-1, 1)
    U = torch.sigmoid(U_logit)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)

    eps = 1e-6
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L_u = torch.tril(raw)
    diag = torch.diagonal(L_u, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + eps
    L_u = L_u - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    Sigma_u = L_u @ L_u.transpose(-2, -1)

    omega = torch.stack([u["omega"] for u in u_params], dim=0)

    Z = hyperparams["Z"]
    m2 = Z.shape[0]
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
        "tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, Sigma_u_tilde
    )
    quad = (y.pow(2) - 2 * y * E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1, 1) ** 2)

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T1 = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2) + elog_sig - quad
    T2 = torch.log(U) + elog_one_minus
    region_elbo = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum(-1)
    data_term = region_elbo.sum()

    elbo = (data_term - beta_W * KL_V - KL_u) / T

    R_row = torch.zeros((), device=device, dtype=mu_W.dtype)
    R_aux = torch.zeros((), device=device, dtype=mu_W.dtype)
    if row_norm_weight > 0:
        if row_norm_mode == "mu":
            R_row = row_norm_penalty_mu_W(mu_W, target=row_norm_target)
        else:
            R_row = row_norm_penalty_E_norm_sq(mu_W, cov_W, target=row_norm_target)
    if aux_metric_weight > 0:
        if aux_metric_mode == "mu":
            R_aux = auxiliary_metric_corr_loss(
                mu_W, regions, max_pairs_per_region=aux_max_pairs, seed=aux_seed
            )
        else:
            A = metric_A_from_moment_local(mu_W, cov_W)
            R_aux = auxiliary_metric_corr_loss_A(
                A, regions, max_pairs_per_region=aux_max_pairs, seed=aux_seed
            )

    loss = -elbo + row_norm_weight * R_row + aux_metric_weight * R_aux
    breakdown = {
        "elbo_per_T": float(elbo.detach().item()),
        "data_per_T": float((data_term / T).detach().item()),
        "KL_V_per_T_raw": float((KL_V / T).detach().item()),
        "KL_V_per_T_betaW": float((beta_W * KL_V / T).detach().item()),
        "KL_u_per_T": float((KL_u / T).detach().item()),
        "row_norm": float(R_row.detach().item()),
        "row_norm_mode": row_norm_mode,
        "aux_metric_neg_corr": float(R_aux.detach().item()),
        "aux_metric_mode": aux_metric_mode,
        "beta_W": float(beta_W),
    }
    return loss, breakdown


# =====================================================================
# 6. Training loop with all knobs
# =====================================================================
def _snapshot_metric_state(
    V_params: dict,
    hyperparams: dict,
    *,
    step: int,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Snapshot of the **q(W) posterior** at the current parameters. Captures things that
    the user wants to track over training to tell ``init failed`` apart from
    ``training pulled mu back to 0``::

        median ||mu_W||_F                  — overall projection mean strength
        median ||mu_W,k:||^2               — per-row mean energy
        median sum_d Var(W,k:d)            — per-row posterior variance energy
        median trace(A_j)                  — total Mahalanobis energy
        median eff_rank(A_j)               — directionality of the metric
        median top eigenvalue of A_j       — concentration in top direction
    """
    from djgp.diagnostics.projection_posterior import sym_matrix_stats  # late import

    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            hyperparams["X_test"],
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        A = metric_A_from_moment_local(mu_W, cov_W)
        per_anchor_stats = [sym_matrix_stats(A[t]) for t in range(A.shape[0])]
        rownorm_mu_sq = (mu_W * mu_W).sum(dim=-1)  # [T, Q]
        rownorm_var = cov_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [T, Q]

        snap = {
            "step": float(step),
            "median_frob_mu_W": float(torch.linalg.norm(mu_W, dim=(-2, -1)).median().item()),
            "median_row_norm_sq_mu_W": float(rownorm_mu_sq.median().item()),
            "mean_row_norm_sq_mu_W": float(rownorm_mu_sq.mean().item()),
            "median_row_var_sum": float(rownorm_var.median().item()),
            "median_E_row_norm_sq": float((rownorm_mu_sq + rownorm_var).median().item()),
            "median_trace_A": float(np.nanmedian([s["trace"] for s in per_anchor_stats])),
            "median_eff_rank_A": float(np.nanmedian([s["eff_rank"] for s in per_anchor_stats])),
            "median_top_eig_A": float(np.nanmedian([s["top_eig"] for s in per_anchor_stats])),
            "median_min_eig_A_pos": float(np.nanmedian([s["min_eig_pos"] for s in per_anchor_stats])),
        }
    if extra is not None:
        snap.update(extra)
    return snap


def train_vi_with_tricks(
    regions: list[dict],
    V_params: dict,
    u_params: list[dict],
    hyperparams: dict,
    *,
    lr: float = 1e-2,
    num_steps: int = 400,
    log_interval: int = 50,
    snapshot_steps: list[int] | None = None,
    kl_warmup: KLWarmupSchedule | None = None,
    row_norm_weight: float = 0.0,
    row_norm_mode: str = "second_moment",
    row_norm_target: float = 1.0,
    aux_metric_weight: float = 0.0,
    aux_metric_mode: str = "A",
    aux_max_pairs: int = 64,
    aux_seed: int | None = 0,
    clamp_sigma_V_min: float = 1e-1,
    clamp_var_w_min: float = 1e-1,
    clamp_lengthscales_min: float = 1e-6,
    clamp_sigma_noise_min: float = 1e-1,
    clamp_sigma_k_min: float = 1e-1,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict, list[dict], dict, list[dict[str, float]], list[dict[str, float]]]:
    """
    Returns ``(V_params, u_params, hyperparams, history, snapshots)``.

    - ``history``    — breakdown dicts captured at every ``log_interval`` step.
    - ``snapshots``  — q(W)-posterior diagnostics captured at ``snapshot_steps``
                       (defaults to ``[0, 10, 50, num_steps]``). Step ``0`` is captured
                       *before* the first optimisation step so we can tell whether a
                       supervised init actually reached ``mu_W``.
    """
    params = [V_params["mu_V"], V_params["sigma_V"]]
    for u in u_params:
        params += [
            u["U_logit"],
            u["mu_u"],
            u["Sigma_u"],
            u["sigma_noise"],
            u["omega"],
            u["sigma_k"],
        ]
    params += [
        hyperparams["lengthscales"],
        hyperparams["var_w"],
        hyperparams["Z"],
    ]
    optimizer = torch.optim.Adam(params, lr=lr)
    history: list[dict[str, float]] = []
    snapshots: list[dict[str, float]] = []

    if snapshot_steps is None:
        snapshot_steps = [0, 10, 50, num_steps]
    snap_set = sorted({int(s) for s in snapshot_steps if 0 <= int(s) <= num_steps})

    if 0 in snap_set:
        snapshots.append(_snapshot_metric_state(V_params, hyperparams, step=0))

    for step in range(1, num_steps + 1):
        beta_W = float(kl_warmup(step)) if kl_warmup is not None else 1.0
        optimizer.zero_grad()
        loss, breakdown = compute_objective_with_tricks(
            regions,
            V_params,
            u_params,
            hyperparams,
            beta_W=beta_W,
            row_norm_weight=row_norm_weight,
            row_norm_mode=row_norm_mode,
            row_norm_target=row_norm_target,
            aux_metric_weight=aux_metric_weight,
            aux_metric_mode=aux_metric_mode,
            aux_max_pairs=aux_max_pairs,
            aux_seed=None if aux_seed is None else int(aux_seed) + step,
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            V_params["sigma_V"].clamp_(min=clamp_sigma_V_min)
            hyperparams["var_w"].clamp_(min=clamp_var_w_min)
            hyperparams["lengthscales"].clamp_(min=clamp_lengthscales_min)
            for u in u_params:
                u["sigma_noise"].clamp_(min=clamp_sigma_noise_min)
                u["sigma_k"].clamp_(min=clamp_sigma_k_min)

        if step == 1 or step % log_interval == 0 or step == num_steps:
            log = {"step": float(step), "loss": float(loss.detach().item()), **breakdown}
            history.append(log)
            if progress_callback is not None:
                progress_callback(step, log)
            else:
                print(
                    f"[step {step:>4d}] loss={log['loss']:.4f} "
                    f"elbo/T={log['elbo_per_T']:.4f} betaW={log['beta_W']:.3f} "
                    f"KL_V_raw/T={log['KL_V_per_T_raw']:.4f} "
                    f"row_norm({log['row_norm_mode'][:3]})={log['row_norm']:.4f} "
                    f"aux({log['aux_metric_mode']})={log['aux_metric_neg_corr']:.4f}"
                )

        if step in snap_set:
            snapshots.append(_snapshot_metric_state(V_params, hyperparams, step=step))

    return V_params, u_params, hyperparams, history, snapshots


__all__ = [
    "KLWarmupSchedule",
    "auxiliary_metric_corr_loss",
    "auxiliary_metric_corr_loss_A",
    "calibrate_mu_V_to_target_mu_W",
    "compute_objective_with_tricks",
    "init_lengthscales_median_heuristic",
    "init_mu_V_supervised",
    "metric_A_from_moment_local",
    "row_norm_penalty_E_norm_sq",
    "row_norm_penalty_mu_W",
    "train_vi_with_tricks",
]
