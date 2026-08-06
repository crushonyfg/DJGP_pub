"""Batched JumpGP-VEM prediction with analytic marginalisation of uncertain W.

The implementation follows ``docs/uncertain_w_jgp_marginalization.md`` and keeps the
first version deliberately narrow and closed-form:

* squared-exponential local GP kernel;
* Gaussian ``q(W)`` with independent latent rows and input dimensions;
* anchor-conditioned ``W(x_anchor)`` and pointwise ``W(x_i)`` modes;
* full covariance across local query positions for the pointwise mode;
* Gauss-Hermite quadrature for gate log-sigmoid expectations;
* closed-form kernel-vector covariance correction
  ``alpha^T Cov_W(k_*I) alpha`` for predictive variance.

All local anchors are processed as one batched torch computation.  The local JumpGP
objectives remain independent; batching is only for speed and shape consistency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F


WMode = Literal["anchor", "pointwise"]
OutlierMode = Literal["constant", "background_normal"]
MCCEMObjective = Literal["stopgrad_reparam", "score_function"]
MCWeighting = Literal["uniform", "softmax"]
GateBasis = Literal["linear", "quadratic"]
MCRestoreState = Literal["best_loss", "best_eval", "final"]
RewardAdvantage = Literal["center", "center_std", "rank"]
AnchorWeightMode = Literal["uniform", "spread_std", "spread_range", "top_fraction", "spread_threshold"]


@dataclass(frozen=True)
class UncertainWJGPConfig:
    """Configuration for batched uncertain-W JumpGP-VEM prediction."""

    mode: WMode = "anchor"
    lengthscale: float | tuple[float, ...] = 1.0
    signal_var: float = 1.0
    noise_var: float = 0.05
    vem_iters: int = 8
    rho_init: float = 0.8
    rho_floor: float = 0.02
    damping: float = 0.7
    temperature: float = 1.0
    gh_points: int = 20
    jitter: float = 1e-5
    outlier_mode: OutlierMode = "constant"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    kernel_vector_correction: bool = True
    correction_weight: float = 1.0


@dataclass
class UncertainWJGPResult:
    """Prediction output and diagnostics."""

    mu: torch.Tensor
    var: torch.Tensor
    sigma: torch.Tensor
    rho: torch.Tensor
    mean_const: torch.Tensor
    correction: torch.Tensor
    latent_var: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass
class UncertainWCEMResult:
    """CEM-compatible uncertain-W prediction output."""

    mu: torch.Tensor
    var: torch.Tensor
    sigma: torch.Tensor
    latent_var: torch.Tensor
    correction: torch.Tensor
    n_inliers: torch.Tensor
    mean_const: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class FixedLabelHyperoptConfig:
    """Local hyperparameter optimisation for fixed CEM labels."""

    steps: int = 40
    lr: float = 0.05
    batched: bool = False
    jitter: float = 1e-5
    min_noise_var: float = 1e-6
    max_noise_var: float = 10.0
    min_signal_var: float = 1e-5
    max_signal_var: float = 100.0
    min_lengthscale: float = 1e-3
    max_lengthscale: float = 20.0
    verbose: bool = False


@dataclass(frozen=True)
class FixedLabelGateConfig:
    """Gate learning under fixed CEM labels."""

    steps: int = 80
    lr: float = 0.05
    basis: GateBasis = "linear"
    intercept_only: bool = False
    l2: float = 1e-3
    max_norm: float = 8.0
    gh_points: int = 20
    init_from_label_balance: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class SelfCEMConfig:
    """Self-contained uncertain-W CEM driver configuration."""

    cem_updates: int = 1
    hyperopt: FixedLabelHyperoptConfig = FixedLabelHyperoptConfig(steps=30, lr=0.04)
    gate: FixedLabelGateConfig = FixedLabelGateConfig(steps=60, lr=0.05)
    outlier_mode: OutlierMode = "background_normal"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    learn_outlier_constant: bool = False
    outlier_constant_steps: int = 10
    outlier_constant_lr: float = 0.05
    outlier_constant_l2: float = 5.0
    outlier_constant_min_logp: float = -12.0
    outlier_constant_max_logp: float = -1e-4
    min_inliers: int = 2
    max_inlier_frac: float = 0.95
    refit_after_update: bool = True


@dataclass(frozen=True)
class SelfVEMConfig:
    """Self-contained uncertain-W VEM driver with soft local responsibilities."""

    vem_updates: int = 1
    hyperopt: FixedLabelHyperoptConfig = FixedLabelHyperoptConfig(steps=30, lr=0.04)
    gate: FixedLabelGateConfig = FixedLabelGateConfig(steps=60, lr=0.05)
    outlier_mode: OutlierMode = "background_normal"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    min_inliers: int = 2
    max_inlier_frac: float = 0.95
    temperature: float = 1.0
    responsibility_floor: float = 1e-4
    damping: float = 1.0
    refit_after_update: bool = True


@dataclass(frozen=True)
class UncertainWQRMstepConfig:
    """M-step configuration for training sparse-GP ``q(R)`` under fixed local CEM state."""

    steps: int = 40
    lr: float = 0.01
    beta_kl: float = 0.05
    kl_warmup_steps: int = 20
    composite_temperature: float = 1.0
    init_log_std: float = -3.0
    train_log_std: bool = False
    w_lengthscale: float = 1.0
    w_signal_var: float = 1.0
    include_conditional_residual: bool = True
    outlier_mode: OutlierMode = "background_normal"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    ortho_weight: float = 0.0
    ortho_target_scale: float = 1.0
    ortho_normalize: bool = True
    early_stopping: bool = False
    early_stopping_metric: str = "eval_rmse"
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    gh_points: int = 20
    jitter: float = 1e-5
    grad_clip_norm: float = 10.0
    min_inliers: int = 2
    log_interval: int = 10
    pairwise_weight: float = 0.0
    pairwise_margin: float = 1.0
    pairwise_same_weight: float = 1.0
    pairwise_cross_weight: float = 1.0
    pairwise_normalize: bool = True
    rank_weight: float = 0.0
    rank_temperature: float = 0.25
    rank_normalize: bool = True
    anchor_pred_weight: float = 0.0
    anchor_pred_var_floor: float = 1e-8
    anchor_pred_exclude_self: bool = True
    anchor_self_tol: float = 1e-10
    residual_group_weight: float = 0.0
    residual_group_eps: float = 1e-8
    train_basis_scale: bool = False
    basis_scale_l1_weight: float = 0.0
    basis_scale_log_min: float = -4.0
    basis_scale_log_max: float = 4.0
    ortho_weight: float = 0.0
    ortho_target_scale: float = 1.0
    ortho_normalize: bool = True
    mean_parameterization: str = "free"
    svd_log_scale_min: float = -6.0
    svd_log_scale_max: float = 4.0


@dataclass
class UncertainWQRMstepResult:
    """Sparse-GP q(R) M-step output and diagnostics."""

    R_mu: torch.Tensor
    R_log_std: torch.Tensor
    inducing_X: torch.Tensor
    history: list[dict[str, float]]
    train_sec: float
    diagnostics: dict[str, Any]
    basis_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class MCCEMQRMstepConfig:
    """Monte-Carlo CEM M-step for sparse-GP ``q(R)``.

    ``stopgrad_reparam`` is route A: sample ``W`` by reparameterization, refit
    local self-CEM on detached samples, then differentiate the frozen-CEM
    likelihood through the sampled ``W``.  ``score_function`` is route B:
    treat the frozen-CEM likelihood as a black-box reward and use
    ``reward * grad log q(W)``.
    """

    objective: MCCEMObjective = "stopgrad_reparam"
    steps: int = 8
    lr: float = 0.005
    beta_kl: float = 0.05
    kl_warmup_steps: int = 4
    n_samples: int = 3
    weighting: MCWeighting = "uniform"
    softmax_temperature_start: float = 2.0
    softmax_temperature_end: float = 1.0
    entropy_beta_start: float = 0.0
    entropy_beta_end: float = 0.0
    standardize_rewards: bool = False
    reward_mode: Literal["transductive", "supervised_anchor"] = "transductive"
    anchor_pred_var_floor: float = 1e-8
    anchor_pred_exclude_self: bool = True
    anchor_self_tol: float = 1e-10
    init_log_std: float = -3.0
    train_log_std: bool = True
    w_lengthscale: float = 1.0
    w_signal_var: float = 1.0
    include_conditional_residual: bool = True
    outlier_mode: OutlierMode = "background_normal"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    learn_inducing: bool = False
    inducing_lr_mult: float = 0.1
    inducing_l2: float = 0.0
    ortho_weight: float = 0.0
    ortho_target_scale: float = 1.0
    ortho_normalize: bool = True
    early_stopping: bool = False
    early_stopping_metric: str = "eval_rmse"
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    restore_state: MCRestoreState = "best_loss"
    gh_points: int = 20
    jitter: float = 1e-5
    grad_clip_norm: float = 10.0
    min_inliers: int = 2
    log_interval: int = 1
    log_sample_rewards: bool = False
    reward_advantage: RewardAdvantage = "center_std"
    anchor_weight: AnchorWeightMode = "uniform"
    anchor_weight_top_frac: float = 0.3
    anchor_weight_spread_tau: float = 0.0
    anchor_weight_max: float = 5.0
    anchor_weight_eps: float = 1e-6
    anchor_weight_curriculum_steps: int = 0
    seed: int = 0


@dataclass
class MCCEMQRMstepResult:
    """Monte-Carlo CEM q(R) M-step output and diagnostics."""

    R_mu: torch.Tensor
    R_log_std: torch.Tensor
    inducing_X: torch.Tensor
    history: list[dict[str, float]]
    sample_reward_trace: list[dict[str, Any]]
    train_sec: float
    diagnostics: dict[str, Any]


def _as_float_tensor(x: torch.Tensor | np.ndarray, *, device: torch.device | None = None) -> torch.Tensor:
    out = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if device is not None:
        out = out.to(device)
    if not torch.is_floating_point(out):
        out = out.float()
    return out


def _q_vector(
    value: float | tuple[float, ...] | torch.Tensor,
    Q: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.to(device=device, dtype=dtype)
    elif isinstance(value, tuple):
        out = torch.tensor(value, device=device, dtype=dtype)
    else:
        out = torch.full((Q,), float(value), device=device, dtype=dtype)
    if out.numel() != Q:
        raise ValueError(f"Expected {Q} lengthscale values, got {out.numel()}.")
    return out.clamp_min(1e-8)


def _scalar_tensor(value: float | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)


def _param_for_anchor(value: float | torch.Tensor | tuple[float, ...], t: int, Q: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = _q_vector(value, Q, device=device, dtype=dtype) if not isinstance(value, torch.Tensor) else value.to(device=device, dtype=dtype)
    if v.dim() == 0:
        return v.expand(Q)
    if v.shape == (Q,):
        return v
    if v.dim() == 2 and v.shape[1] == Q:
        return v[t]
    raise ValueError(f"Cannot read anchor-specific Q-vector from shape {tuple(v.shape)}.")


def _scalar_for_anchor(value: float | torch.Tensor, t: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = _scalar_tensor(value, device=device, dtype=dtype) if not isinstance(value, torch.Tensor) else value.to(device=device, dtype=dtype)
    if v.dim() == 0:
        return v
    if v.dim() == 1:
        return v[t]
    raise ValueError(f"Cannot read anchor-specific scalar from shape {tuple(v.shape)}.")


def _symmetrise(K: torch.Tensor) -> torch.Tensor:
    return 0.5 * (K + K.transpose(-1, -2))


def _cholesky_psd(K: torch.Tensor, *, jitter: float) -> torch.Tensor:
    n = K.shape[-1]
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    K = _symmetrise(K)
    last_err: RuntimeError | None = None
    for mult in (1.0, 10.0, 100.0, 1000.0):
        try:
            return torch.linalg.cholesky(K + (float(jitter) * mult) * eye)
        except RuntimeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return torch.linalg.cholesky(K + float(jitter) * eye)


def _rbf_kernel_projected(Z1: torch.Tensor, Z2: torch.Tensor, ell: torch.Tensor, signal_var: float) -> torch.Tensor:
    diff = Z1.unsqueeze(-2) - Z2.unsqueeze(-3)
    d2 = (diff.pow(2) / ell.view(*([1] * (diff.dim() - 1)), -1).pow(2)).sum(dim=-1)
    sig = _scalar_tensor(signal_var, device=Z1.device, dtype=Z1.dtype)
    return sig * torch.exp(-0.5 * d2)


def expected_se_kernel_anchor(
    X1: torch.Tensor,
    X2: torch.Tensor,
    W_mu: torch.Tensor,
    W_var: torch.Tensor,
    *,
    lengthscale: float | tuple[float, ...] = 1.0,
    signal_var: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Closed-form ``E_W[k(Wx_i, Wx_j)]`` for anchor-conditioned ``q(W_a)``.

    Parameters
    ----------
    X1, X2:
        Batched raw inputs ``[T, n1, D]`` and ``[T, n2, D]``.
    W_mu, W_var:
        Mean and diagonal variance of ``q(W_a)`` with shape ``[T, Q, D]``.
    """
    if X1.dim() != 3 or X2.dim() != 3:
        raise ValueError("X1 and X2 must have shape [T, n, D].")
    if W_mu.shape != W_var.shape or W_mu.dim() != 3:
        raise ValueError("W_mu and W_var must have shape [T, Q, D].")
    T, Q, D = W_mu.shape
    if X1.shape[0] != T or X2.shape[0] != T or X1.shape[-1] != D or X2.shape[-1] != D:
        raise ValueError("X/W dimensions are inconsistent.")
    ell = _q_vector(lengthscale, Q, device=X1.device, dtype=X1.dtype)
    ell2 = ell.pow(2).view(1, 1, 1, Q)
    delta = X1.unsqueeze(2) - X2.unsqueeze(1)  # [T,n1,n2,D]
    mu_delta = torch.einsum("tqd,tijd->tijq", W_mu, delta)
    s_delta = torch.einsum("tqd,tijd->tijq", W_var.clamp_min(0.0), delta.pow(2))
    log_det = -0.5 * torch.log1p(s_delta / ell2).sum(dim=-1)
    quad = (mu_delta.pow(2) / (ell2 + s_delta).clamp_min(1e-12)).sum(dim=-1)
    sig = _scalar_tensor(signal_var, device=X1.device, dtype=X1.dtype)
    return sig * torch.exp(log_det - 0.5 * quad)


def expected_se_kernel_anchor_fullcov(
    X1: torch.Tensor,
    X2: torch.Tensor,
    W_mu: torch.Tensor,
    W_cov: torch.Tensor,
    *,
    lengthscale: float | tuple[float, ...] = 1.0,
    signal_var: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Closed-form anchor-conditioned expected SE kernel with full row covariance.

    ``W_cov[t,q]`` is the ``D x D`` covariance of row ``q`` of ``W(c_t)``.
    Rows are still assumed independent, matching the low-rank residual
    coefficient prior used by the q(R) M-step.
    """
    if X1.dim() != 3 or X2.dim() != 3:
        raise ValueError("X1 and X2 must have shape [T, n, D].")
    if W_mu.dim() != 3 or W_cov.dim() != 4:
        raise ValueError("W_mu must be [T,Q,D] and W_cov must be [T,Q,D,D].")
    T, Q, D = W_mu.shape
    if W_cov.shape != (T, Q, D, D):
        raise ValueError(f"W_cov must have shape {(T, Q, D, D)}, got {tuple(W_cov.shape)}.")
    if X1.shape[0] != T or X2.shape[0] != T or X1.shape[-1] != D or X2.shape[-1] != D:
        raise ValueError("X/W dimensions are inconsistent.")
    ell = _q_vector(lengthscale, Q, device=X1.device, dtype=X1.dtype)
    ell2 = ell.pow(2).view(1, 1, 1, Q)
    delta = X1.unsqueeze(2) - X2.unsqueeze(1)
    mu_delta = torch.einsum("tqd,tijd->tijq", W_mu, delta)
    s_delta = torch.einsum("tqde,tijd,tije->tijq", W_cov, delta, delta).clamp_min(0.0)
    log_det = -0.5 * torch.log1p(s_delta / ell2).sum(dim=-1)
    quad = (mu_delta.pow(2) / (ell2 + s_delta).clamp_min(1e-12)).sum(dim=-1)
    sig = _scalar_tensor(signal_var, device=X1.device, dtype=X1.dtype)
    return sig * torch.exp(log_det - 0.5 * quad)


def pointwise_projected_moments(
    X_all: torch.Tensor,
    W_mu_all: torch.Tensor,
    W_cov_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean/covariance of pointwise projections ``z_i = W(x_i)x_i``.

    ``X_all`` is ``[T, m, D]``.  ``W_mu_all`` is ``[T, m, Q, D]``.
    ``W_cov_all`` is ``[T, Q, D, m, m]`` and may include cross-position covariance.

    Returns
    -------
    z_mu:
        ``[T, m, Q]``.
    z_cov:
        ``[T, m, m, Q]`` where ``z_cov[..., i, j, q] =
        Cov(W_q(x_i)x_i, W_q(x_j)x_j)``.
    """
    if X_all.dim() != 3 or W_mu_all.dim() != 4 or W_cov_all.dim() != 5:
        raise ValueError("Expected X_all [T,m,D], W_mu_all [T,m,Q,D], W_cov_all [T,Q,D,m,m].")
    T, m, D = X_all.shape
    if W_mu_all.shape[:2] != (T, m) or W_mu_all.shape[-1] != D:
        raise ValueError("X_all and W_mu_all dimensions are inconsistent.")
    Q = W_mu_all.shape[2]
    if W_cov_all.shape != (T, Q, D, m, m):
        raise ValueError(f"W_cov_all must have shape {(T, Q, D, m, m)}, got {tuple(W_cov_all.shape)}.")
    z_mu = torch.einsum("tmqd,tmd->tmq", W_mu_all, X_all)
    z_cov = torch.einsum("tid,tqdij,tjd->tijq", X_all, W_cov_all, X_all)
    return z_mu, z_cov


def expected_se_kernel_pointwise(
    X_all: torch.Tensor,
    W_mu_all: torch.Tensor,
    W_cov_all: torch.Tensor,
    *,
    lengthscale: float | tuple[float, ...] = 1.0,
    signal_var: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Closed-form ``E_W[k(W(x_i)x_i, W(x_j)x_j)]`` for pointwise ``W(x)``."""
    z_mu, z_cov = pointwise_projected_moments(X_all, W_mu_all, W_cov_all)
    T, m, Q = z_mu.shape
    ell = _q_vector(lengthscale, Q, device=X_all.device, dtype=X_all.dtype)
    ell2 = ell.pow(2).view(1, 1, 1, Q)
    mean_delta = z_mu.unsqueeze(2) - z_mu.unsqueeze(1)
    diag_cov = torch.diagonal(z_cov, dim1=1, dim2=2).permute(0, 2, 1)  # [T,m,Q]
    s_delta = diag_cov.unsqueeze(2) + diag_cov.unsqueeze(1) - 2.0 * z_cov
    s_delta = s_delta.clamp_min(0.0)
    log_det = -0.5 * torch.log1p(s_delta / ell2).sum(dim=-1)
    quad = (mean_delta.pow(2) / (ell2 + s_delta).clamp_min(1e-12)).sum(dim=-1)
    sig = _scalar_tensor(signal_var, device=X_all.device, dtype=X_all.dtype)
    return sig * torch.exp(log_det - 0.5 * quad)


def _gh_nodes_weights(n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.hermite.hermgauss(int(n))
    return (
        torch.as_tensor(nodes, device=device, dtype=dtype),
        torch.as_tensor(weights, device=device, dtype=dtype),
    )


def gh_logsigmoid_expectations(
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    gh_points: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``E[log sigmoid(x)]``, ``E[log sigmoid(-x)]``, and ``E[sigmoid(x)]``."""
    nodes, weights = _gh_nodes_weights(gh_points, device=mean.device, dtype=mean.dtype)
    std = var.clamp_min(1e-12).sqrt()
    z = mean.unsqueeze(-1) + math.sqrt(2.0) * std.unsqueeze(-1) * nodes
    norm = math.sqrt(math.pi)
    elog_pos = (weights * F.logsigmoid(z)).sum(dim=-1) / norm
    elog_neg = (weights * F.logsigmoid(-z)).sum(dim=-1) / norm
    epi = (weights * torch.sigmoid(z)).sum(dim=-1) / norm
    return elog_pos, elog_neg, epi


def gate_moments_anchor(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    W_mu: torch.Tensor,
    W_var: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian moments of ``nu^T[1, W_a(x_i-x_anchor)]`` in anchor mode."""
    delta = X_neighbors - X_anchor.unsqueeze(1)
    proj_mu = torch.einsum("tqd,tnd->tnq", W_mu, delta)
    proj_var = torch.einsum("tqd,tnd->tnq", W_var.clamp_min(0.0), delta.pow(2))
    return _gate_moments_from_projected_diag(proj_mu, proj_var, gate)


def gate_moments_anchor_fullcov(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    W_mu: torch.Tensor,
    W_cov: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian gate moments for anchor-conditioned W with full row covariance."""
    delta = X_neighbors - X_anchor.unsqueeze(1)
    proj_mu = torch.einsum("tqd,tnd->tnq", W_mu, delta)
    proj_var = torch.einsum("tqde,tnd,tne->tnq", W_cov, delta, delta).clamp_min(0.0)
    return _gate_moments_from_projected_diag(proj_mu, proj_var, gate)


def gate_moments_pointwise(
    X_all: torch.Tensor,
    W_mu_all: torch.Tensor,
    W_cov_all: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian moments of ``nu^T[1, W(x_i)x_i - W(x_anchor)x_anchor]``."""
    z_mu, z_cov = pointwise_projected_moments(X_all, W_mu_all, W_cov_all)
    delta_mu = z_mu[:, 1:] - z_mu[:, :1]
    diag_cov = torch.diagonal(z_cov, dim1=1, dim2=2).permute(0, 2, 1)  # [T,m,Q]
    delta_var = diag_cov[:, 1:] + diag_cov[:, :1] - 2.0 * z_cov[:, 1:, 0, :]
    delta_var = delta_var.clamp_min(0.0)
    return _gate_moments_from_projected_diag(delta_mu, delta_var, gate)


def _gate_dim(Q: int, basis: GateBasis) -> int:
    if basis == "linear":
        return Q + 1
    if basis == "quadratic":
        return 1 + Q + Q * (Q + 1) // 2
    raise ValueError(f"Unknown gate basis {basis!r}.")


def _gate_basis_from_dim(width: int, Q: int) -> GateBasis:
    if int(width) == Q + 1:
        return "linear"
    if int(width) == _gate_dim(Q, "quadratic"):
        return "quadratic"
    raise ValueError(f"Cannot infer gate basis from width={width}, Q={Q}.")


def _quadratic_pair_indices(Q: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [(i, j) for i in range(Q) for j in range(i, Q)]
    left = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
    right = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
    return left, right


def _gate_moments_from_projected_diag(
    z_mu: torch.Tensor,
    z_var: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate-score moments for linear or full-quadratic latent boundary.

    The quadratic score is not exactly Gaussian when ``Var(z)>0``.  We use the
    exact mean and a first-order delta-method variance.  For the deterministic
    MC-CEM samples used in routes A/B, ``z_var=0`` and this reduces to the exact
    quadratic boundary score.
    """
    Q = z_mu.shape[-1]
    basis = _gate_basis_from_dim(gate.shape[1], Q)
    slope = gate[:, 1 : 1 + Q]
    mean = gate[:, :1] + (slope.unsqueeze(1) * z_mu).sum(dim=-1)
    grad = slope.unsqueeze(1).expand_as(z_mu).clone()
    if basis == "quadratic":
        left, right = _quadratic_pair_indices(Q, device=z_mu.device)
        coeff = gate[:, 1 + Q :]
        z_left = z_mu[..., left]
        z_right = z_mu[..., right]
        quad_mean = z_left * z_right
        diag_mask = left == right
        if bool(diag_mask.any().item()):
            quad_mean = quad_mean.clone()
            quad_mean[..., diag_mask] = quad_mean[..., diag_mask] + z_var[..., left[diag_mask]]
        mean = mean + (coeff.unsqueeze(1) * quad_mean).sum(dim=-1)
        for k in range(left.numel()):
            i = int(left[k].item())
            j = int(right[k].item())
            c = coeff[:, k].unsqueeze(1)
            if i == j:
                grad[..., i] = grad[..., i] + 2.0 * c * z_mu[..., i]
            else:
                grad[..., i] = grad[..., i] + c * z_mu[..., j]
                grad[..., j] = grad[..., j] + c * z_mu[..., i]
    var = (grad.pow(2) * z_var.clamp_min(0.0)).sum(dim=-1)
    return mean, var


def weighted_gp_train_posterior(
    K: torch.Tensor,
    y: torch.Tensor,
    rho: torch.Tensor,
    noise_var: float | torch.Tensor,
    mean_const: torch.Tensor,
    *,
    rho_floor: float = 1e-3,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Posterior moments at training inputs under weighted Gaussian likelihood."""
    if K.dim() != 3:
        raise ValueError("K must have shape [T,n,n].")
    T, n, _ = K.shape
    sig2 = torch.as_tensor(noise_var, device=K.device, dtype=K.dtype).reshape(()).clamp_min(1e-10)
    rho_c = rho.to(device=K.device, dtype=K.dtype).clamp(float(rho_floor), 1.0)
    y_c = y.to(device=K.device, dtype=K.dtype) - mean_const.view(T, 1)
    Ky = K + torch.diag_embed(sig2 / rho_c)
    L = _cholesky_psd(Ky, jitter=jitter)
    alpha = torch.cholesky_solve(y_c.unsqueeze(-1), L).squeeze(-1)
    mean_f = mean_const.view(T, 1) + torch.einsum("tij,tj->ti", K, alpha)
    solve_K = torch.cholesky_solve(K, L)
    cov = K - K @ solve_K
    diag_cov = torch.diagonal(_symmetrise(cov), dim1=-2, dim2=-1).clamp_min(0.0)
    return mean_f, diag_cov, alpha, L


def _outlier_logprob(
    y: torch.Tensor,
    *,
    noise_var: float,
    mode: OutlierMode,
    outlier_sigma_mult: float,
    background_scale_mult: float,
    constant_logp: torch.Tensor | None = None,
) -> torch.Tensor:
    sig = math.sqrt(max(float(noise_var), 1e-12))
    if mode == "constant":
        if constant_logp is not None:
            logp = constant_logp.to(device=y.device, dtype=y.dtype)
            if logp.dim() == 0:
                return torch.full_like(y, float(logp.detach().cpu().item()))
            if logp.dim() == 1:
                return logp.view(-1, 1).expand_as(y)
            return logp.expand_as(y)
        z = float(outlier_sigma_mult)
        value = -math.log(sig) - 0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
        return torch.full_like(y, float(value))
    if mode == "background_normal":
        loc = y.mean(dim=1, keepdim=True)
        scale = (y.std(dim=1, unbiased=False, keepdim=True) * float(background_scale_mult)).clamp_min(sig)
        return -0.5 * (math.log(2.0 * math.pi) + 2.0 * scale.log() + ((y - loc) / scale).pow(2))
    raise ValueError(f"Unknown outlier mode {mode!r}.")


def _default_constant_outlier_logp(
    y: torch.Tensor,
    *,
    noise_var: float | torch.Tensor,
    outlier_sigma_mult: float,
    min_logp: float,
    max_logp: float,
) -> torch.Tensor:
    """Default per-anchor constant outlier log-density used to initialize learnable 1/u."""
    noise_t = torch.as_tensor(noise_var, device=y.device, dtype=y.dtype).reshape(-1)
    if noise_t.numel() == 1:
        noise_t = noise_t.expand(y.shape[0])
    sig = noise_t.clamp_min(1e-12).sqrt()
    z = float(outlier_sigma_mult)
    value = -sig.log() - 0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
    return value.clamp(float(min_logp), float(max_logp))


def _logp_to_logit(logp: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    prob = logp.exp().clamp(float(eps), 1.0 - float(eps))
    return torch.logit(prob)


def _fit_constant_outlier_logp_from_labels(
    labels: torch.Tensor,
    default_logp: torch.Tensor,
    *,
    steps: int,
    lr: float,
    l2: float,
    min_logp: float,
    max_logp: float,
) -> torch.Tensor:
    """Fit one bounded constant outlier log-density per local anchor.

    A free constant density has no finite MLE under fixed outlier labels.  The
    small quadratic anchor to the legacy value keeps this as a local calibration
    parameter rather than an unbounded shortcut.
    """
    dtype = labels.dtype
    device = labels.device
    default = default_logp.to(device=device, dtype=dtype).reshape(-1)
    if int(steps) <= 0:
        return default.clamp(float(min_logp), float(max_logp)).detach()
    out_count = (1.0 - labels.to(dtype=dtype).clamp(0.0, 1.0)).sum(dim=1)
    active = out_count > 0
    if not bool(active.any().item()):
        return default.clamp(float(min_logp), float(max_logp)).detach()
    logit = _logp_to_logit(default.clamp(float(min_logp), float(max_logp))).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=float(lr))
    for _ in range(max(0, int(steps))):
        opt.zero_grad()
        logp = F.logsigmoid(logit).clamp(float(min_logp), float(max_logp))
        reward = (out_count * logp).mean()
        prior = (logp - default).pow(2).mean()
        loss = -reward + float(l2) * prior
        loss.backward()
        opt.step()
    return F.logsigmoid(logit.detach()).clamp(float(min_logp), float(max_logp))


def _two_quadratic_exp_product(
    mean1: torch.Tensor,
    mean2: torch.Tensor,
    s11: torch.Tensor,
    s22: torch.Tensor,
    s12: torch.Tensor,
    ell: torch.Tensor,
    *,
    signal_var: float | torch.Tensor,
) -> torch.Tensor:
    """Closed form for ``E[k_i k_j]`` from two jointly Gaussian projected deltas.

    Shapes are broadcast-compatible ``[..., Q]``.
    """
    ell2 = ell.pow(2).view(*([1] * (mean1.dim() - 1)), -1)
    b11 = 1.0 + s11 / ell2
    b22 = 1.0 + s22 / ell2
    b12 = s12 / ell2
    det_b = (b11 * b22 - b12.pow(2)).clamp_min(1e-12)
    log_factor = -0.5 * torch.log(det_b)

    c11 = ell2 + s11
    c22 = ell2 + s22
    c12 = s12
    det_c = (c11 * c22 - c12.pow(2)).clamp_min(1e-12)
    quad = (c22 * mean1.pow(2) - 2.0 * c12 * mean1 * mean2 + c11 * mean2.pow(2)) / det_c
    sig = _scalar_tensor(signal_var, device=mean1.device, dtype=mean1.dtype)
    return sig.pow(2) * torch.exp((log_factor - 0.5 * quad).sum(dim=-1))


def kernel_vector_covariance_anchor(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    W_mu: torch.Tensor,
    W_var: torch.Tensor,
    k_star: torch.Tensor,
    *,
    lengthscale: float | tuple[float, ...] = 1.0,
    signal_var: float = 1.0,
) -> torch.Tensor:
    """Closed-form ``Cov_W(k_*I)`` for anchor-conditioned W."""
    T, n, D = X_neighbors.shape
    Q = W_mu.shape[1]
    ell = _q_vector(lengthscale, Q, device=X_neighbors.device, dtype=X_neighbors.dtype)
    delta = X_anchor.unsqueeze(1) - X_neighbors  # [T,n,D]
    mean = torch.einsum("tqd,tnd->tnq", W_mu, delta)
    s_self = torch.einsum("tqd,tnd->tnq", W_var.clamp_min(0.0), delta.pow(2))
    s_cross = torch.einsum("tqd,tid,tjd->tijq", W_var.clamp_min(0.0), delta, delta)
    e_prod = _two_quadratic_exp_product(
        mean.unsqueeze(2),
        mean.unsqueeze(1),
        s_self.unsqueeze(2),
        s_self.unsqueeze(1),
        s_cross,
        ell,
        signal_var=signal_var,
    )
    cov = e_prod - k_star.unsqueeze(2) * k_star.unsqueeze(1)
    return _symmetrise(cov)


def kernel_vector_covariance_anchor_fullcov(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    W_mu: torch.Tensor,
    W_cov: torch.Tensor,
    k_star: torch.Tensor,
    *,
    lengthscale: float | tuple[float, ...] = 1.0,
    signal_var: float = 1.0,
) -> torch.Tensor:
    """Closed-form ``Cov_W(k_*I)`` for anchor-conditioned full row covariance."""
    Q = W_mu.shape[1]
    ell = _q_vector(lengthscale, Q, device=X_neighbors.device, dtype=X_neighbors.dtype)
    delta = X_anchor.unsqueeze(1) - X_neighbors
    mean = torch.einsum("tqd,tnd->tnq", W_mu, delta)
    s_self = torch.einsum("tqde,tnd,tne->tnq", W_cov, delta, delta).clamp_min(0.0)
    s_cross = torch.einsum("tqde,tid,tje->tijq", W_cov, delta, delta)
    e_prod = _two_quadratic_exp_product(
        mean.unsqueeze(2),
        mean.unsqueeze(1),
        s_self.unsqueeze(2),
        s_self.unsqueeze(1),
        s_cross,
        ell,
        signal_var=signal_var,
    )
    cov = e_prod - k_star.unsqueeze(2) * k_star.unsqueeze(1)
    return _symmetrise(cov)


def kernel_vector_covariance_pointwise(
    X_all: torch.Tensor,
    W_mu_all: torch.Tensor,
    W_cov_all: torch.Tensor,
    k_star: torch.Tensor,
    *,
    lengthscale: float | tuple[float, ...] = 1.0,
    signal_var: float = 1.0,
) -> torch.Tensor:
    """Closed-form ``Cov_W(k_*I)`` for pointwise W using full position covariance."""
    z_mu, z_cov = pointwise_projected_moments(X_all, W_mu_all, W_cov_all)
    Q = z_mu.shape[-1]
    ell = _q_vector(lengthscale, Q, device=X_all.device, dtype=X_all.dtype)
    mean = z_mu[:, :1] - z_mu[:, 1:]  # [T,n,Q]
    diag_cov = torch.diagonal(z_cov, dim1=1, dim2=2).permute(0, 2, 1)
    s_self = diag_cov[:, :1] + diag_cov[:, 1:] - 2.0 * z_cov[:, :1, 1:, :].squeeze(1)
    s_self = s_self.clamp_min(0.0)
    cov00 = z_cov[:, 0, 0, :].unsqueeze(1).unsqueeze(2)
    cov0j = z_cov[:, 0, 1:, :].unsqueeze(1)
    covi0 = z_cov[:, 1:, 0, :].unsqueeze(2)
    covij = z_cov[:, 1:, 1:, :]
    s_cross = cov00 - cov0j - covi0 + covij
    e_prod = _two_quadratic_exp_product(
        mean.unsqueeze(2),
        mean.unsqueeze(1),
        s_self.unsqueeze(2),
        s_self.unsqueeze(1),
        s_cross,
        ell,
        signal_var=signal_var,
    )
    cov = e_prod - k_star.unsqueeze(2) * k_star.unsqueeze(1)
    return _symmetrise(cov)


def _single_kernel_blocks(
    t: int,
    *,
    mode: WMode,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    W_mu_anchor: torch.Tensor | None,
    W_var_anchor: torch.Tensor | None,
    W_cov_anchor: torch.Tensor | None,
    W_mu_all: torch.Tensor | None,
    W_cov_all: torch.Tensor | None,
    need_cov_k: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if mode == "anchor":
        if W_mu_anchor is None or (W_var_anchor is None and W_cov_anchor is None):
            raise ValueError("Anchor mode requires W_mu_anchor and W_var_anchor or W_cov_anchor.")
        xa = X_anchor[t : t + 1]
        xn = X_neighbors[t : t + 1]
        wm = W_mu_anchor[t : t + 1]
        if W_cov_anchor is not None:
            wc = W_cov_anchor[t : t + 1]
            K = expected_se_kernel_anchor_fullcov(xn, xn, wm, wc, lengthscale=lengthscale, signal_var=signal_var).squeeze(0)
            k_star = expected_se_kernel_anchor_fullcov(xa.unsqueeze(1), xn, wm, wc, lengthscale=lengthscale, signal_var=signal_var).squeeze(0).squeeze(0)
        else:
            wv = W_var_anchor[t : t + 1]  # type: ignore[index]
            K = expected_se_kernel_anchor(xn, xn, wm, wv, lengthscale=lengthscale, signal_var=signal_var).squeeze(0)
            k_star = expected_se_kernel_anchor(xa.unsqueeze(1), xn, wm, wv, lengthscale=lengthscale, signal_var=signal_var).squeeze(0).squeeze(0)
        cov_k = None
        if need_cov_k:
            if W_cov_anchor is not None:
                cov_k = kernel_vector_covariance_anchor_fullcov(
                    xa,
                    xn,
                    wm,
                    W_cov_anchor[t : t + 1],
                    k_star.view(1, -1),
                    lengthscale=lengthscale,
                    signal_var=signal_var,
                ).squeeze(0)
            else:
                cov_k = kernel_vector_covariance_anchor(
                    xa,
                    xn,
                    wm,
                    W_var_anchor[t : t + 1],  # type: ignore[index]
                    k_star.view(1, -1),
                    lengthscale=lengthscale,
                    signal_var=signal_var,
                ).squeeze(0)
        return K, k_star, cov_k
    if mode == "pointwise":
        if W_mu_all is None or W_cov_all is None:
            raise ValueError("Pointwise mode requires W_mu_all/W_cov_all.")
        x_all = torch.cat([X_anchor[t : t + 1].unsqueeze(1), X_neighbors[t : t + 1]], dim=1)
        wm_all = W_mu_all[t : t + 1]
        wc_all = W_cov_all[t : t + 1]
        Kall = expected_se_kernel_pointwise(x_all, wm_all, wc_all, lengthscale=lengthscale, signal_var=signal_var).squeeze(0)
        K = Kall[1:, 1:]
        k_star = Kall[0, 1:]
        cov_k = None
        if need_cov_k:
            cov_k = kernel_vector_covariance_pointwise(
                x_all,
                wm_all,
                wc_all,
                k_star.view(1, -1),
                lengthscale=lengthscale,
                signal_var=signal_var,
            ).squeeze(0)
        return K, k_star, cov_k
    raise ValueError(f"Unknown mode={mode!r}.")


def _initial_anchor_param_matrix(
    value: float | tuple[float, ...] | torch.Tensor,
    T: int,
    Q: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_value: float,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        v = value.to(device=device, dtype=dtype)
    elif isinstance(value, tuple):
        v = torch.tensor(value, device=device, dtype=dtype)
    else:
        v = torch.tensor(float(value), device=device, dtype=dtype)
    if v.numel() == 1:
        return v.reshape(()).expand(T, Q).clone().clamp_min(float(min_value))
    if v.shape == (Q,):
        return v.reshape(1, Q).expand(T, Q).clone().clamp_min(float(min_value))
    if v.shape == (T, Q):
        return v.clone().clamp_min(float(min_value))
    if v.numel() == T * Q:
        return v.reshape(T, Q).clone().clamp_min(float(min_value))
    raise ValueError(f"Could not broadcast parameter with shape {tuple(v.shape)} to {(T, Q)}.")


def _initial_anchor_param_vector(
    value: float | torch.Tensor,
    T: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_value: float,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        v = value.to(device=device, dtype=dtype)
    else:
        v = torch.tensor(float(value), device=device, dtype=dtype)
    if v.numel() == 1:
        return v.reshape(()).expand(T).clone().clamp_min(float(min_value))
    if v.shape == (T,):
        return v.clone().clamp_min(float(min_value))
    if v.numel() == T:
        return v.reshape(T).clone().clamp_min(float(min_value))
    raise ValueError(f"Could not broadcast parameter with shape {tuple(v.shape)} to {(T,)}.")


def _expected_se_kernel_anchor_batched_params(
    X1: torch.Tensor,
    X2: torch.Tensor,
    W_mu: torch.Tensor,
    W_var: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
) -> torch.Tensor:
    """Anchor-conditioned expected SE kernel with per-anchor lengthscales."""
    T, Q, D = W_mu.shape
    if lengthscale.shape != (T, Q) or signal_var.shape != (T,):
        raise ValueError("Batched kernel expects lengthscale [T,Q] and signal_var [T].")
    ell2 = lengthscale.clamp_min(1e-12).pow(2).view(T, 1, 1, Q)
    delta = X1.unsqueeze(2) - X2.unsqueeze(1)
    mu_delta = torch.einsum("tqd,tijd->tijq", W_mu, delta)
    s_delta = torch.einsum("tqd,tijd->tijq", W_var.clamp_min(0.0), delta.pow(2))
    log_det = -0.5 * torch.log1p(s_delta / ell2).sum(dim=-1)
    quad = (mu_delta.pow(2) / (ell2 + s_delta).clamp_min(1e-12)).sum(dim=-1)
    return signal_var.view(T, 1, 1).clamp_min(1e-12) * torch.exp(log_det - 0.5 * quad)


def _fit_uncertain_kernel_hyperparams_anchor_batched(
    X_anchor_t: torch.Tensor,
    X_neighbors_t: torch.Tensor,
    y_t: torch.Tensor,
    labels_t: torch.Tensor,
    *,
    W_mu_anchor: torch.Tensor,
    W_var_anchor: torch.Tensor,
    init_lengthscale: float | tuple[float, ...] | torch.Tensor,
    init_signal_var: float | torch.Tensor,
    init_noise_var: float | torch.Tensor,
    config: FixedLabelHyperoptConfig,
    min_inliers: int,
) -> dict[str, torch.Tensor]:
    """Batched anchor-mode fixed-label GP hyperparameter fit.

    This keeps per-anchor parameters independent, but performs each optimizer
    step as one batched kernel build and Cholesky solve on the GPU.
    """
    cfg = config
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    T, n, _D = X_neighbors_t.shape
    Q = W_mu_anchor.shape[1]
    active = labels_t > 0.5
    fallback = torch.zeros(T, device=device, dtype=torch.bool)
    for t in range(T):
        if int(active[t].sum().item()) < int(min_inliers):
            fallback[t] = True
            k = min(max(int(min_inliers), 1), n)
            top_idx = torch.topk(labels_t[t], k=k, largest=True).indices
            active[t] = False
            active[t, top_idx] = True
    active_f = active.to(dtype=dtype)
    n_active = active_f.sum(dim=1).clamp_min(1.0)
    ell0 = _initial_anchor_param_matrix(
        init_lengthscale,
        T,
        Q,
        device=device,
        dtype=dtype,
        min_value=float(cfg.min_lengthscale),
    )
    sig0 = _initial_anchor_param_vector(
        init_signal_var,
        T,
        device=device,
        dtype=dtype,
        min_value=float(cfg.min_signal_var),
    )
    noise0 = _initial_anchor_param_vector(
        init_noise_var,
        T,
        device=device,
        dtype=dtype,
        min_value=float(cfg.min_noise_var),
    )
    log_ell = ell0.log().detach().clone().requires_grad_(True)
    log_signal = sig0.log().detach().clone().requires_grad_(True)
    log_noise = noise0.log().detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([log_ell, log_signal, log_noise], lr=float(cfg.lr))
    final_nll = torch.full((T,), float("nan"), device=device, dtype=dtype)
    final_mean = (y_t * active_f).sum(dim=1) / n_active
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    pair_active = active_f.unsqueeze(2) * active_f.unsqueeze(1)
    y_active = y_t * active_f
    for _step in range(max(0, int(cfg.steps))):
        opt.zero_grad()
        ell = log_ell.exp().clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
        sig = log_signal.exp().clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
        noise = log_noise.exp().clamp(float(cfg.min_noise_var), float(cfg.max_noise_var))
        K = _expected_se_kernel_anchor_batched_params(
            X_neighbors_t,
            X_neighbors_t,
            W_mu_anchor,
            W_var_anchor,
            lengthscale=ell,
            signal_var=sig,
        )
        diag = noise.unsqueeze(1) * active_f + (1.0 - active_f) + float(cfg.jitter)
        Ky = K * pair_active + torch.diag_embed(diag)
        L = _cholesky_psd(Ky, jitter=float(cfg.jitter))
        one = active_f
        Ainv_y = torch.cholesky_solve(y_active.unsqueeze(-1), L).squeeze(-1)
        Ainv_one = torch.cholesky_solve(one.unsqueeze(-1), L).squeeze(-1)
        mean = (one * Ainv_y).sum(dim=1) / (one * Ainv_one).sum(dim=1).clamp_min(1e-12)
        resid = (y_t - mean.unsqueeze(1)) * active_f
        alpha = torch.cholesky_solve(resid.unsqueeze(-1), L).squeeze(-1)
        quad = (resid * alpha).sum(dim=1)
        logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=1)
        nll = 0.5 * (quad + logdet + n_active * math.log(2.0 * math.pi))
        loss = nll.mean()
        loss.backward()
        opt.step()
        final_nll = nll.detach()
        final_mean = mean.detach()
    with torch.no_grad():
        ell_out = log_ell.exp().clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
        signal_out = log_signal.exp().clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
        noise_out = log_noise.exp().clamp(float(cfg.min_noise_var), float(cfg.max_noise_var))
        if int(cfg.steps) <= 0:
            final_mean = (y_t * active_f).sum(dim=1) / n_active
            final_nll = torch.full((T,), float("nan"), device=device, dtype=dtype)
    return {
        "lengthscale": ell_out.detach(),
        "signal_var": signal_out.detach(),
        "noise_var": noise_out.detach(),
        "mean_const": final_mean.detach(),
        "nll": final_nll.detach(),
        "fallback": fallback,
    }


def uncertain_w_cem_predict_from_labels(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    lengthscale: float | tuple[float, ...] | torch.Tensor = 1.0,
    signal_var: float | torch.Tensor = 1.0,
    noise_var: float | torch.Tensor = 0.05,
    mean_const: torch.Tensor | np.ndarray | None = None,
    correction_weight: float = 0.0,
    jitter: float = 1e-5,
    min_inliers: int = 2,
) -> UncertainWCEMResult:
    """Predict using fixed CEM labels/hyperparameters and uncertain-W kernels.

    With ``Var(W)=0`` this reduces to ordinary GP prediction on the projected
    CEM inlier set, provided ``labels``, ``mean_const`` and kernel hyperparameters
    come from the same deterministic JumpGP-CEM fit.
    """
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=X_anchor_t.dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=X_anchor_t.dtype)
    labels_t = _as_float_tensor(labels, device=device).to(dtype=X_anchor_t.dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if labels_t.dim() == 3 and labels_t.shape[-1] == 1:
        labels_t = labels_t.squeeze(-1)
    T, n, D = X_neighbors_t.shape
    if X_anchor_t.shape != (T, D) or y_t.shape != (T, n) or labels_t.shape != (T, n):
        raise ValueError("Anchor/neighborhood/y/label dimensions are inconsistent.")
    dtype = X_anchor_t.dtype
    if mode == "anchor":
        W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
        W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
        W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
        Q = W_mu_a.shape[1] if W_mu_a is not None else 0
        W_mu_p = None
        W_cov_p = None
    else:
        W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
        W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
        Q = W_mu_p.shape[2] if W_mu_p is not None else 0
        W_mu_a = None
        W_var_a = None
        W_cov_a = None
    if Q <= 0:
        raise ValueError("Could not infer latent dimension Q from W moments.")

    mean_t = None if mean_const is None else _as_float_tensor(mean_const, device=device).to(dtype=dtype).reshape(T)
    soft_labels = bool(((labels_t > 1e-6) & (labels_t < 1.0 - 1e-6)).any().item())
    mu = torch.empty(T, device=device, dtype=dtype)
    var = torch.empty(T, device=device, dtype=dtype)
    latent_var = torch.empty(T, device=device, dtype=dtype)
    correction = torch.zeros(T, device=device, dtype=dtype)
    n_inliers = torch.empty(T, device=device, dtype=torch.long)
    used_mean = torch.empty(T, device=device, dtype=dtype)
    fallback_count = 0

    for t in range(T):
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, k_star, cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor_t,
            X_neighbors=X_neighbors_t,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            need_cov_k=(float(correction_weight) != 0.0),
        )
        if soft_labels:
            weights = labels_t[t].clamp_min(0.0)
            mask = weights > 1e-6
        else:
            weights = None
            mask = labels_t[t] > 0.5
        if int(mask.sum().item()) < int(min_inliers):
            fallback_count += 1
            k = min(max(int(min_inliers), 1), n)
            top_idx = torch.topk(labels_t[t], k=k, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[top_idx] = True
            if weights is not None:
                weights = weights.clone()
                weights[mask] = weights[mask].clamp_min(1.0)
        n_inliers[t] = int(mask.sum().item())
        y_i = y_t[t, mask]
        K_i = K[mask][:, mask]
        k_i = k_star[mask]
        w_i = None if weights is None else weights[mask].clamp_min(1e-6)
        m = mean_t[t] if mean_t is not None else (y_i.mean() if w_i is None else (w_i * y_i).sum() / w_i.sum().clamp_min(1e-12))
        used_mean[t] = m
        if w_i is None:
            noise_diag = noise_t.expand(K_i.shape[0])
        else:
            noise_diag = noise_t / w_i
        Ky = K_i + torch.diag(noise_diag)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        alpha = torch.cholesky_solve((y_i - m).unsqueeze(-1), L).squeeze(-1)
        mu[t] = m + torch.dot(k_i, alpha)
        v = torch.cholesky_solve(k_i.unsqueeze(-1), L).squeeze(-1)
        latent_var[t] = (sig_t - torch.dot(k_i, v)).clamp_min(0.0)
        if cov_k is not None:
            cov_i = cov_k[mask][:, mask]
            correction[t] = torch.einsum("i,ij,j->", alpha, cov_i, alpha).clamp_min(0.0)
        var[t] = (latent_var[t] + noise_t + float(correction_weight) * correction[t]).clamp_min(1e-12)

    diagnostics = {
        "mode": mode,
        "fallback_count": int(fallback_count),
        "mean_n_inliers": float(n_inliers.to(dtype=dtype).mean().detach().cpu().item()),
        "mean_latent_var": float(latent_var.mean().detach().cpu().item()),
        "mean_kernel_vector_correction": float(correction.mean().detach().cpu().item()),
        "correction_weight": float(correction_weight),
    }
    return UncertainWCEMResult(
        mu=mu,
        var=var,
        sigma=var.sqrt(),
        latent_var=latent_var,
        correction=correction,
        n_inliers=n_inliers,
        mean_const=used_mean,
        diagnostics=diagnostics,
    )


def fit_uncertain_kernel_hyperparams_from_labels(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    init_lengthscale: float | tuple[float, ...] | torch.Tensor = 1.0,
    init_signal_var: float | torch.Tensor = 1.0,
    init_noise_var: float | torch.Tensor = 0.05,
    config: FixedLabelHyperoptConfig | None = None,
    min_inliers: int = 2,
) -> dict[str, torch.Tensor]:
    """Fit per-anchor effective SE hyperparameters under fixed CEM labels.

    The local mean is profiled out in closed form at each optimizer step:
    ``m = 1^T A^-1 y / 1^T A^-1 1``.  Gate parameters are intentionally not
    used here because fixed-label prediction only depends on the final inlier
    set; this function repairs the kernel/noise/mean mismatch caused by
    replacing ``K(W_mean)`` with ``E_W[K(W)]``.
    """
    cfg = config or FixedLabelHyperoptConfig()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=X_anchor_t.dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=X_anchor_t.dtype)
    labels_t = _as_float_tensor(labels, device=device).to(dtype=X_anchor_t.dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if labels_t.dim() == 3 and labels_t.shape[-1] == 1:
        labels_t = labels_t.squeeze(-1)
    T, n, D = X_neighbors_t.shape
    dtype = X_anchor_t.dtype
    if mode == "anchor":
        W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
        W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
        W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
        Q = W_mu_a.shape[1] if W_mu_a is not None else 0
        W_mu_p = None
        W_cov_p = None
    else:
        W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
        W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
        Q = W_mu_p.shape[2] if W_mu_p is not None else 0
        W_mu_a = None
        W_var_a = None
        W_cov_a = None
    if Q <= 0:
        raise ValueError("Could not infer latent dimension Q from W moments.")
    soft_labels = bool(((labels_t > 1e-6) & (labels_t < 1.0 - 1e-6)).any().item())
    if (not soft_labels) and bool(cfg.batched) and mode == "anchor" and W_mu_a is not None and W_var_a is not None and W_cov_a is None:
        return _fit_uncertain_kernel_hyperparams_anchor_batched(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            labels_t,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            init_lengthscale=init_lengthscale,
            init_signal_var=init_signal_var,
            init_noise_var=init_noise_var,
            config=cfg,
            min_inliers=min_inliers,
        )

    ell_out = torch.empty(T, Q, device=device, dtype=dtype)
    signal_out = torch.empty(T, device=device, dtype=dtype)
    noise_out = torch.empty(T, device=device, dtype=dtype)
    mean_out = torch.empty(T, device=device, dtype=dtype)
    nll_out = torch.empty(T, device=device, dtype=dtype)
    fallback = torch.zeros(T, device=device, dtype=torch.bool)

    for t in range(T):
        if soft_labels:
            weights = labels_t[t].clamp_min(0.0)
            mask = weights > 1e-6
        else:
            weights = None
            mask = labels_t[t] > 0.5
        if int(mask.sum().item()) < int(min_inliers):
            fallback[t] = True
            k = min(max(int(min_inliers), 1), n)
            top_idx = torch.topk(labels_t[t], k=k, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[top_idx] = True
            if weights is not None:
                weights = weights.clone()
                weights[mask] = weights[mask].clamp_min(1.0)
        y_i = y_t[t, mask]
        w_i = None if weights is None else weights[mask].clamp_min(1e-6)
        log_ell = torch.log(_param_for_anchor(init_lengthscale, t, Q, device=device, dtype=dtype).clamp_min(cfg.min_lengthscale)).detach().clone().requires_grad_(True)
        log_signal = torch.log(_scalar_for_anchor(init_signal_var, t, device=device, dtype=dtype).clamp_min(cfg.min_signal_var)).detach().clone().requires_grad_(True)
        log_noise = torch.log(_scalar_for_anchor(init_noise_var, t, device=device, dtype=dtype).clamp_min(cfg.min_noise_var)).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([log_ell, log_signal, log_noise], lr=float(cfg.lr))
        final_nll = torch.tensor(float("nan"), device=device, dtype=dtype)
        final_mean = y_i.mean()
        for _step in range(max(0, int(cfg.steps))):
            opt.zero_grad()
            ell = log_ell.exp().clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
            sig = log_signal.exp().clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
            noise = log_noise.exp().clamp(float(cfg.min_noise_var), float(cfg.max_noise_var))
            K, _k_star, _cov_k = _single_kernel_blocks(
                t,
                mode=mode,
                X_anchor=X_anchor_t,
                X_neighbors=X_neighbors_t,
                lengthscale=ell,
                signal_var=sig,
                W_mu_anchor=W_mu_a,
                W_var_anchor=W_var_a,
                W_cov_anchor=W_cov_a,
                W_mu_all=W_mu_p,
                W_cov_all=W_cov_p,
                need_cov_k=False,
            )
            K_i = K[mask][:, mask]
            if w_i is None:
                noise_diag = noise.expand(K_i.shape[0])
                n_eff = torch.as_tensor(y_i.numel(), device=device, dtype=dtype)
            else:
                noise_diag = noise / w_i
                n_eff = w_i.sum().clamp_min(1.0)
            Ky = K_i + torch.diag(noise_diag)
            L = _cholesky_psd(Ky.unsqueeze(0), jitter=float(cfg.jitter)).squeeze(0)
            one = torch.ones_like(y_i)
            Ainv_y = torch.cholesky_solve(y_i.unsqueeze(-1), L).squeeze(-1)
            Ainv_one = torch.cholesky_solve(one.unsqueeze(-1), L).squeeze(-1)
            m = (one @ Ainv_y) / (one @ Ainv_one).clamp_min(1e-12)
            resid = y_i - m
            alpha = torch.cholesky_solve(resid.unsqueeze(-1), L).squeeze(-1)
            logdet = 2.0 * torch.diag(L).log().sum()
            nll = 0.5 * (resid @ alpha + logdet + n_eff * math.log(2.0 * math.pi))
            nll.backward()
            opt.step()
            final_nll = nll.detach()
            final_mean = m.detach()
        with torch.no_grad():
            ell_out[t] = log_ell.exp().clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
            signal_out[t] = log_signal.exp().clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
            noise_out[t] = log_noise.exp().clamp(float(cfg.min_noise_var), float(cfg.max_noise_var))
            if int(cfg.steps) <= 0:
                mean_out[t] = y_i.mean() if w_i is None else (w_i * y_i).sum() / w_i.sum().clamp_min(1e-12)
                nll_out[t] = torch.nan
            else:
                mean_out[t] = final_mean
                nll_out[t] = final_nll

    return {
        "lengthscale": ell_out.detach(),
        "signal_var": signal_out.detach(),
        "noise_var": noise_out.detach(),
        "mean_const": mean_out.detach(),
        "nll": nll_out.detach(),
        "fallback": fallback,
    }


def _gate_moments_for_mode(
    *,
    mode: WMode,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    gate: torch.Tensor,
    W_mu_anchor: torch.Tensor | None = None,
    W_var_anchor: torch.Tensor | None = None,
    W_cov_anchor: torch.Tensor | None = None,
    W_mu_all: torch.Tensor | None = None,
    W_cov_all: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "anchor":
        if W_mu_anchor is None or (W_var_anchor is None and W_cov_anchor is None):
            raise ValueError("Anchor mode requires W_mu_anchor and W_var_anchor or W_cov_anchor.")
        if W_cov_anchor is not None:
            return gate_moments_anchor_fullcov(X_anchor, X_neighbors, W_mu_anchor, W_cov_anchor, gate)
        return gate_moments_anchor(X_anchor, X_neighbors, W_mu_anchor, W_var_anchor, gate)
    if mode == "pointwise":
        if W_mu_all is None or W_cov_all is None:
            raise ValueError("Pointwise mode requires W_mu_all/W_cov_all.")
        X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
        return gate_moments_pointwise(X_all, W_mu_all, W_cov_all, gate)
    raise ValueError(f"Unknown mode={mode!r}.")


def _init_gate_from_labels(labels: torch.Tensor, Q: int, basis: GateBasis = "linear") -> torch.Tensor:
    T = labels.shape[0]
    p = labels.to(dtype=torch.float64).mean(dim=1).clamp(1e-3, 1.0 - 1e-3)
    gate = torch.zeros(T, _gate_dim(Q, basis), device=labels.device, dtype=labels.dtype)
    gate[:, 0] = (p / (1.0 - p)).log().to(dtype=labels.dtype)
    return gate


def fit_uncertain_gate_from_labels(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    gate_init: torch.Tensor | np.ndarray | None = None,
    config: FixedLabelGateConfig | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Fit per-anchor logistic gate parameters under fixed labels.

    The objective uses Gauss-Hermite quadrature for
    ``E_q(W)[log sigmoid(xi)]`` and ``E_q(W)[log sigmoid(-xi)]``.
    """
    cfg = config or FixedLabelGateConfig()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    labels_t = _as_float_tensor(labels, device=device).to(dtype=dtype)
    if labels_t.dim() == 3 and labels_t.shape[-1] == 1:
        labels_t = labels_t.squeeze(-1)
    T, _n, D = X_neighbors_t.shape
    if mode == "anchor":
        W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
        W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
        W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
        Q = W_mu_a.shape[1] if W_mu_a is not None else 0
        W_mu_p = None
        W_cov_p = None
    else:
        W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
        W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
        Q = W_mu_p.shape[2] if W_mu_p is not None else 0
        W_mu_a = None
        W_var_a = None
        W_cov_a = None
    if Q <= 0:
        raise ValueError("Could not infer Q from W moments.")
    if gate_init is None:
        gate0 = _init_gate_from_labels(labels_t, Q, basis=cfg.basis)
    else:
        gate0 = _as_float_tensor(gate_init, device=device).to(dtype=dtype)
        expected_dim = _gate_dim(Q, cfg.basis)
        if gate0.shape != (T, expected_dim):
            if gate0.shape == (T, Q + 1) and cfg.basis == "quadratic":
                padded = torch.zeros(T, expected_dim, device=device, dtype=dtype)
                padded[:, : Q + 1] = gate0
                gate0 = padded
            else:
                raise ValueError(f"gate_init must have shape {(T, expected_dim)}, got {tuple(gate0.shape)}.")
        if cfg.init_from_label_balance:
            # Keep supplied slopes but make intercept compatible with label prevalence.
            gate0 = gate0.clone()
            gate0[:, 0] = _init_gate_from_labels(labels_t, Q, basis=cfg.basis)[:, 0]

    if bool(cfg.intercept_only):
        gate0 = gate0.clone()
        gate0[:, 1:] = 0.0
        gate0_param = gate0[:, 0].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([gate0_param], lr=float(cfg.lr))
    else:
        gate = gate0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([gate], lr=float(cfg.lr))
    history: list[dict[str, float]] = []
    y01 = labels_t.clamp(0.0, 1.0)
    for step in range(max(0, int(cfg.steps))):
        opt.zero_grad()
        if bool(cfg.intercept_only):
            gate = torch.zeros_like(gate0)
            gate[:, 0] = gate0_param
        mean, var = _gate_moments_for_mode(
            mode=mode,
            X_anchor=X_anchor_t,
            X_neighbors=X_neighbors_t,
            gate=gate,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
        )
        elog_pos, elog_neg, _epi = gh_logsigmoid_expectations(mean, var, gh_points=int(cfg.gh_points))
        loglike = (y01 * elog_pos + (1.0 - y01) * elog_neg).mean()
        penalty = gate.pow(2).mean()
        loss = -loglike + float(cfg.l2) * penalty
        loss.backward()
        opt.step()
        if float(cfg.max_norm) > 0 and not bool(cfg.intercept_only):
            with torch.no_grad():
                norm = torch.linalg.norm(gate, dim=1, keepdim=True).clamp_min(1e-12)
                scale = (float(cfg.max_norm) / norm).clamp(max=1.0)
                gate.mul_(scale)
        elif float(cfg.max_norm) > 0:
            with torch.no_grad():
                gate0_param.clamp_(min=-float(cfg.max_norm), max=float(cfg.max_norm))
        if step == 0 or step == int(cfg.steps) - 1:
            history.append({
                "step": float(step + 1),
                "loss": float(loss.detach().cpu().item()),
                "loglike": float(loglike.detach().cpu().item()),
                "gate_norm_mean": float(torch.linalg.norm(gate.detach(), dim=1).mean().cpu().item()),
            })
    if bool(cfg.intercept_only):
        gate = torch.zeros_like(gate0)
        gate[:, 0] = gate0_param.detach()
    return {"gate": gate.detach(), "history": history}


def _predictive_logpdf_from_kernel_row(
    K: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    i: int,
    *,
    mean_const: torch.Tensor,
    noise_var: torch.Tensor,
    signal_var: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    if int(train_mask.sum().item()) < 1:
        pred_mean = mean_const
        pred_var = signal_var + noise_var
    else:
        y_tr = y[train_mask]
        K_tr = K[train_mask][:, train_mask]
        k_i = K[i, train_mask]
        Ky = K_tr + noise_var * torch.eye(K_tr.shape[0], device=K.device, dtype=K.dtype)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        alpha = torch.cholesky_solve((y_tr - mean_const).unsqueeze(-1), L).squeeze(-1)
        pred_mean = mean_const + torch.dot(k_i, alpha)
        v = torch.cholesky_solve(k_i.unsqueeze(-1), L).squeeze(-1)
        pred_var = (K[i, i] - torch.dot(k_i, v)).clamp_min(0.0) + noise_var
    return -0.5 * (math.log(2.0 * math.pi) + pred_var.log() + (y[i] - pred_mean).pow(2) / pred_var)


def _predictive_logpdf_from_kernel_row_weighted(
    K: torch.Tensor,
    y: torch.Tensor,
    train_weight: torch.Tensor,
    i: int,
    *,
    mean_const: torch.Tensor,
    noise_var: torch.Tensor,
    signal_var: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    active = train_weight > 1e-6
    if int(active.sum().item()) < 1:
        pred_mean = mean_const
        pred_var = signal_var + noise_var
    else:
        w = train_weight[active].clamp_min(1e-6)
        y_tr = y[active]
        K_tr = K[active][:, active]
        k_i = K[i, active]
        Ky = K_tr + torch.diag(noise_var / w)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        alpha = torch.cholesky_solve((y_tr - mean_const).unsqueeze(-1), L).squeeze(-1)
        pred_mean = mean_const + torch.dot(k_i, alpha)
        v = torch.cholesky_solve(k_i.unsqueeze(-1), L).squeeze(-1)
        pred_var = (K[i, i] - torch.dot(k_i, v)).clamp_min(0.0) + noise_var
    return -0.5 * (math.log(2.0 * math.pi) + pred_var.log() + (y[i] - pred_mean).pow(2) / pred_var)


def uncertain_w_cem_update_labels(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    gate: torch.Tensor | np.ndarray,
    lengthscale: float | tuple[float, ...] | torch.Tensor,
    signal_var: float | torch.Tensor,
    noise_var: float | torch.Tensor,
    mean_const: torch.Tensor | np.ndarray,
    outlier_mode: OutlierMode = "background_normal",
    outlier_sigma_mult: float = 2.5,
    background_scale_mult: float = 3.0,
    outlier_logp: torch.Tensor | np.ndarray | None = None,
    gh_points: int = 20,
    jitter: float = 1e-5,
    min_inliers: int = 2,
    max_inlier_frac: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Hard C-step under uncertain-W kernel using LOO/candidate GP scores."""
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    labels_t = _as_float_tensor(labels, device=device).to(dtype=dtype)
    gate_t = _as_float_tensor(gate, device=device).to(dtype=dtype)
    mean_t = _as_float_tensor(mean_const, device=device).to(dtype=dtype).reshape(-1)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if labels_t.dim() == 3 and labels_t.shape[-1] == 1:
        labels_t = labels_t.squeeze(-1)
    T, n, D = X_neighbors_t.shape
    if mode == "anchor":
        W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
        W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
        W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
        Q = W_mu_a.shape[1] if W_mu_a is not None else 0
        W_mu_p = None
        W_cov_p = None
    else:
        W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
        W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
        Q = W_mu_p.shape[2] if W_mu_p is not None else 0
        W_mu_a = None
        W_var_a = None
        W_cov_a = None
    gate_mean, gate_var = _gate_moments_for_mode(
        mode=mode,
        X_anchor=X_anchor_t,
        X_neighbors=X_neighbors_t,
        gate=gate_t,
        W_mu_anchor=W_mu_a,
        W_var_anchor=W_var_a,
        W_cov_anchor=W_cov_a,
        W_mu_all=W_mu_p,
        W_cov_all=W_cov_p,
    )
    elog_pos, elog_neg, _epi = gh_logsigmoid_expectations(gate_mean, gate_var, gh_points=int(gh_points))
    # Use a representative noise scale for background when per-anchor noise is supplied.
    noise_ref = float(torch.as_tensor(noise_var).detach().cpu().reshape(-1).median().item())
    outlier_logp_t = None if outlier_logp is None else _as_float_tensor(outlier_logp, device=device).to(dtype=dtype)
    out_logp = _outlier_logprob(
        y_t,
        noise_var=noise_ref,
        mode=outlier_mode,
        outlier_sigma_mult=float(outlier_sigma_mult),
        background_scale_mult=float(background_scale_mult),
        constant_logp=outlier_logp_t,
    )
    new_labels = torch.empty(T, n, device=device, dtype=torch.bool)
    margins = torch.empty(T, n, device=device, dtype=dtype)
    log_in = torch.empty(T, n, device=device, dtype=dtype)
    log_out = elog_neg + out_logp
    for t in range(T):
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, _k_star, _cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor_t,
            X_neighbors=X_neighbors_t,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            need_cov_k=False,
        )
        old = labels_t[t] > 0.5
        for i in range(n):
            train_mask = old.clone()
            if bool(old[i].item()):
                train_mask[i] = False
            if int(train_mask.sum().item()) < int(min_inliers):
                d = torch.sum((X_neighbors_t[t] - X_neighbors_t[t, i]).pow(2), dim=-1)
                d[i] = torch.inf
                k = min(max(int(min_inliers), 1), n - 1)
                idx = torch.topk(-d, k=k, largest=True).indices
                train_mask = torch.zeros(n, device=device, dtype=torch.bool)
                train_mask[idx] = True
            gp_score = _predictive_logpdf_from_kernel_row(
                K,
                y_t[t],
                train_mask,
                i,
                mean_const=mean_t[t],
                noise_var=noise_t,
                signal_var=sig_t,
                jitter=jitter,
            )
            log_in[t, i] = elog_pos[t, i] + gp_score
        margins[t] = log_in[t] - log_out[t]
        row_labels = margins[t] > 0.0
        min_k = min(max(int(min_inliers), 1), n)
        if int(row_labels.sum().item()) < min_k:
            idx = torch.topk(margins[t], k=min_k, largest=True).indices
            row_labels = torch.zeros(n, device=device, dtype=torch.bool)
            row_labels[idx] = True
        if float(max_inlier_frac) < 1.0:
            max_k = max(min_k, int(math.ceil(float(max_inlier_frac) * n)))
            if int(row_labels.sum().item()) > max_k:
                idx = torch.topk(margins[t], k=max_k, largest=True).indices
                row_labels = torch.zeros(n, device=device, dtype=torch.bool)
                row_labels[idx] = True
        new_labels[t] = row_labels
    return {
        "labels": new_labels,
        "margin": margins.detach(),
        "log_in": log_in.detach(),
        "log_out": log_out.detach(),
        "changed_frac": (new_labels != (labels_t > 0.5)).to(dtype=dtype).mean().detach(),
    }


def uncertain_w_vem_update_responsibilities(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    responsibilities: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    gate: torch.Tensor | np.ndarray,
    lengthscale: float | tuple[float, ...] | torch.Tensor,
    signal_var: float | torch.Tensor,
    noise_var: float | torch.Tensor,
    mean_const: torch.Tensor | np.ndarray,
    outlier_mode: OutlierMode = "background_normal",
    outlier_sigma_mult: float = 2.5,
    background_scale_mult: float = 3.0,
    gh_points: int = 20,
    jitter: float = 1e-5,
    min_inliers: int = 2,
    max_inlier_frac: float = 0.95,
    temperature: float = 1.0,
    responsibility_floor: float = 1e-4,
    damping: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Soft V-step under uncertain-W kernels using fractional responsibilities."""
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    resp_t = _as_float_tensor(responsibilities, device=device).to(dtype=dtype)
    gate_t = _as_float_tensor(gate, device=device).to(dtype=dtype)
    mean_t = _as_float_tensor(mean_const, device=device).to(dtype=dtype).reshape(-1)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if resp_t.dim() == 3 and resp_t.shape[-1] == 1:
        resp_t = resp_t.squeeze(-1)
    T, n, _D = X_neighbors_t.shape
    if mode == "anchor":
        W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
        W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
        W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
        W_mu_p = None
        W_cov_p = None
        Q = W_mu_a.shape[1] if W_mu_a is not None else 0
    else:
        W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
        W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
        W_mu_a = None
        W_var_a = None
        W_cov_a = None
        Q = W_mu_p.shape[2] if W_mu_p is not None else 0
    gate_mean, gate_var = _gate_moments_for_mode(
        mode=mode,
        X_anchor=X_anchor_t,
        X_neighbors=X_neighbors_t,
        gate=gate_t,
        W_mu_anchor=W_mu_a,
        W_var_anchor=W_var_a,
        W_cov_anchor=W_cov_a,
        W_mu_all=W_mu_p,
        W_cov_all=W_cov_p,
    )
    elog_pos, elog_neg, _epi = gh_logsigmoid_expectations(gate_mean, gate_var, gh_points=int(gh_points))
    noise_ref = float(torch.as_tensor(noise_var).detach().cpu().reshape(-1).median().item())
    out_logp = elog_neg + _outlier_logprob(
        y_t,
        noise_var=noise_ref,
        mode=outlier_mode,
        outlier_sigma_mult=float(outlier_sigma_mult),
        background_scale_mult=float(background_scale_mult),
    )
    log_in = torch.empty(T, n, device=device, dtype=dtype)
    resp_old = resp_t.clamp(float(responsibility_floor), 1.0 - float(responsibility_floor))
    for t in range(T):
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, _k_star, _cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor_t,
            X_neighbors=X_neighbors_t,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            need_cov_k=False,
        )
        for i in range(n):
            train_weight = resp_old[t].clone()
            train_weight[i] = 0.0
            if int((train_weight > 1e-6).sum().item()) < int(min_inliers):
                d = torch.sum((X_neighbors_t[t] - X_neighbors_t[t, i]).pow(2), dim=-1)
                d[i] = torch.inf
                k = min(max(int(min_inliers), 1), n - 1)
                idx = torch.topk(-d, k=k, largest=True).indices
                train_weight = torch.zeros(n, device=device, dtype=dtype)
                train_weight[idx] = 1.0
            gp_score = _predictive_logpdf_from_kernel_row_weighted(
                K,
                y_t[t],
                train_weight,
                i,
                mean_const=mean_t[t],
                noise_var=noise_t,
                signal_var=sig_t,
                jitter=jitter,
            )
            log_in[t, i] = elog_pos[t, i] + gp_score
    margins = (log_in - out_logp) / max(float(temperature), 1e-6)
    new_resp = torch.sigmoid(margins).clamp(float(responsibility_floor), 1.0 - float(responsibility_floor))
    min_k = min(max(int(min_inliers), 1), n)
    for t in range(T):
        if float(new_resp[t].sum().detach().item()) < float(min_k):
            idx = torch.topk(margins[t], k=min_k, largest=True).indices
            floor = float(responsibility_floor)
            row = torch.full((n,), floor, device=device, dtype=dtype)
            row[idx] = 1.0 - floor
            new_resp[t] = row
        if float(max_inlier_frac) < 1.0:
            target = float(max_inlier_frac) * n
            total = float(new_resp[t].sum().detach().item())
            if total > target:
                new_resp[t] = (new_resp[t] * (target / max(total, 1e-12))).clamp(float(responsibility_floor), 1.0 - float(responsibility_floor))
    if float(damping) < 1.0:
        new_resp = (float(damping) * new_resp + (1.0 - float(damping)) * resp_old).clamp(float(responsibility_floor), 1.0 - float(responsibility_floor))
    return {
        "responsibilities": new_resp.detach(),
        "margin": margins.detach(),
        "log_in": log_in.detach(),
        "log_out": out_logp.detach(),
        "changed_frac": (new_resp - resp_old).abs().mean().detach(),
    }


def run_uncertain_w_self_cem(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    init_labels: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    init_lengthscale: float | tuple[float, ...] | torch.Tensor = 1.0,
    init_signal_var: float | torch.Tensor = 1.0,
    init_noise_var: float | torch.Tensor = 0.05,
    gate_init: torch.Tensor | np.ndarray | None = None,
    config: SelfCEMConfig | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Run a small number of self-contained uncertain-W CEM updates."""
    cfg = config or SelfCEMConfig()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    labels = _as_float_tensor(init_labels, device=device).to(dtype=dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if labels.dim() == 3 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
    W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
    W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
    W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
    W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
    gate = _as_float_tensor(gate_init, device=device).to(dtype=dtype) if gate_init is not None else None
    history: list[dict[str, float]] = []
    hp: dict[str, torch.Tensor] | None = None
    outlier_logp: torch.Tensor | None = None
    for outer in range(max(0, int(cfg.cem_updates))):
        hp = fit_uncertain_kernel_hyperparams_from_labels(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            labels,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            init_lengthscale=init_lengthscale if hp is None else hp["lengthscale"],
            init_signal_var=init_signal_var if hp is None else hp["signal_var"],
            init_noise_var=init_noise_var if hp is None else hp["noise_var"],
            config=cfg.hyperopt,
            min_inliers=cfg.min_inliers,
        )
        gate_fit = fit_uncertain_gate_from_labels(
            X_anchor_t,
            X_neighbors_t,
            labels,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            gate_init=gate,
            config=cfg.gate,
        )
        gate = gate_fit["gate"]  # type: ignore[assignment]
        if cfg.outlier_mode == "constant" and bool(cfg.learn_outlier_constant):
            default_logp = _default_constant_outlier_logp(
                y_t,
                noise_var=hp["noise_var"],
                outlier_sigma_mult=float(cfg.outlier_sigma_mult),
                min_logp=float(cfg.outlier_constant_min_logp),
                max_logp=float(cfg.outlier_constant_max_logp),
            )
            outlier_logp = _fit_constant_outlier_logp_from_labels(
                labels,
                default_logp if outlier_logp is None else outlier_logp,
                steps=int(cfg.outlier_constant_steps),
                lr=float(cfg.outlier_constant_lr),
                l2=float(cfg.outlier_constant_l2),
                min_logp=float(cfg.outlier_constant_min_logp),
                max_logp=float(cfg.outlier_constant_max_logp),
            )
        upd = uncertain_w_cem_update_labels(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            labels,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            gate=gate,
            lengthscale=hp["lengthscale"],
            signal_var=hp["signal_var"],
            noise_var=hp["noise_var"],
            mean_const=hp["mean_const"],
            outlier_mode=cfg.outlier_mode,
            outlier_sigma_mult=cfg.outlier_sigma_mult,
            background_scale_mult=cfg.background_scale_mult,
            outlier_logp=outlier_logp,
            gh_points=cfg.gate.gh_points,
            jitter=cfg.hyperopt.jitter,
            min_inliers=cfg.min_inliers,
            max_inlier_frac=cfg.max_inlier_frac,
        )
        labels_new = upd["labels"].to(dtype=dtype)
        history.append({
            "outer": float(outer + 1),
            "changed_frac": float(upd["changed_frac"].detach().cpu().item()),
            "inlier_frac": float(labels_new.mean().detach().cpu().item()),
            "mean_lengthscale": float(hp["lengthscale"].mean().detach().cpu().item()),
            "mean_signal_var": float(hp["signal_var"].mean().detach().cpu().item()),
            "mean_noise_var": float(hp["noise_var"].mean().detach().cpu().item()),
            "mean_outlier_logp": float(outlier_logp.mean().detach().cpu().item()) if outlier_logp is not None else float("nan"),
        })
        labels = labels_new
    if hp is None or cfg.refit_after_update:
        hp = fit_uncertain_kernel_hyperparams_from_labels(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            labels,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            init_lengthscale=init_lengthscale if hp is None else hp["lengthscale"],
            init_signal_var=init_signal_var if hp is None else hp["signal_var"],
            init_noise_var=init_noise_var if hp is None else hp["noise_var"],
            config=cfg.hyperopt,
            min_inliers=cfg.min_inliers,
        )
        gate_fit = fit_uncertain_gate_from_labels(
            X_anchor_t,
            X_neighbors_t,
            labels,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            gate_init=gate,
            config=cfg.gate,
        )
        gate = gate_fit["gate"]  # type: ignore[assignment]
    if cfg.outlier_mode == "constant" and bool(cfg.learn_outlier_constant):
        default_logp = _default_constant_outlier_logp(
            y_t,
            noise_var=hp["noise_var"],
            outlier_sigma_mult=float(cfg.outlier_sigma_mult),
            min_logp=float(cfg.outlier_constant_min_logp),
            max_logp=float(cfg.outlier_constant_max_logp),
        )
        outlier_logp = _fit_constant_outlier_logp_from_labels(
            labels,
            default_logp if outlier_logp is None else outlier_logp,
            steps=int(cfg.outlier_constant_steps),
            lr=float(cfg.outlier_constant_lr),
            l2=float(cfg.outlier_constant_l2),
            min_logp=float(cfg.outlier_constant_min_logp),
            max_logp=float(cfg.outlier_constant_max_logp),
        )
    return {
        "labels": (labels > 0.5).detach(),
        "lengthscale": hp["lengthscale"].detach(),
        "signal_var": hp["signal_var"].detach(),
        "noise_var": hp["noise_var"].detach(),
        "mean_const": hp["mean_const"].detach(),
        "gate": gate.detach() if isinstance(gate, torch.Tensor) else gate,
        "outlier_logp": outlier_logp.detach() if outlier_logp is not None else torch.empty(0, device=device, dtype=dtype),
        "history": history,
    }


def run_uncertain_w_self_vem(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    init_responsibilities: torch.Tensor | np.ndarray,
    *,
    mode: WMode = "anchor",
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_cov_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    init_lengthscale: float | tuple[float, ...] | torch.Tensor = 1.0,
    init_signal_var: float | torch.Tensor = 1.0,
    init_noise_var: float | torch.Tensor = 0.05,
    gate_init: torch.Tensor | np.ndarray | None = None,
    config: SelfVEMConfig | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Run uncertain-W self-VEM with soft inlier responsibilities."""
    cfg = config or SelfVEMConfig()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    resp = _as_float_tensor(init_responsibilities, device=device).to(dtype=dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if resp.dim() == 3 and resp.shape[-1] == 1:
        resp = resp.squeeze(-1)
    floor = float(cfg.responsibility_floor)
    resp = resp.clamp(floor, 1.0 - floor)
    W_mu_a = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype) if W_mu_anchor is not None else None
    W_var_a = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype) if W_var_anchor is not None else None
    W_cov_a = _as_float_tensor(W_cov_anchor, device=device).to(dtype=dtype) if W_cov_anchor is not None else None
    W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype) if W_mu_all is not None else None
    W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype) if W_cov_all is not None else None
    gate = _as_float_tensor(gate_init, device=device).to(dtype=dtype) if gate_init is not None else None
    history: list[dict[str, float]] = []
    hp: dict[str, torch.Tensor] | None = None
    for outer in range(max(0, int(cfg.vem_updates))):
        hp = fit_uncertain_kernel_hyperparams_from_labels(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            resp,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            init_lengthscale=init_lengthscale if hp is None else hp["lengthscale"],
            init_signal_var=init_signal_var if hp is None else hp["signal_var"],
            init_noise_var=init_noise_var if hp is None else hp["noise_var"],
            config=cfg.hyperopt,
            min_inliers=cfg.min_inliers,
        )
        gate_fit = fit_uncertain_gate_from_labels(
            X_anchor_t,
            X_neighbors_t,
            resp,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            gate_init=gate,
            config=cfg.gate,
        )
        gate = gate_fit["gate"]  # type: ignore[assignment]
        upd = uncertain_w_vem_update_responsibilities(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            resp,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            gate=gate,
            lengthscale=hp["lengthscale"],
            signal_var=hp["signal_var"],
            noise_var=hp["noise_var"],
            mean_const=hp["mean_const"],
            outlier_mode=cfg.outlier_mode,
            outlier_sigma_mult=cfg.outlier_sigma_mult,
            background_scale_mult=cfg.background_scale_mult,
            gh_points=cfg.gate.gh_points,
            jitter=cfg.hyperopt.jitter,
            min_inliers=cfg.min_inliers,
            max_inlier_frac=cfg.max_inlier_frac,
            temperature=cfg.temperature,
            responsibility_floor=cfg.responsibility_floor,
            damping=cfg.damping,
        )
        resp_new = upd["responsibilities"].to(dtype=dtype)
        history.append({
            "outer": float(outer + 1),
            "mean_abs_resp_change": float(upd["changed_frac"].detach().cpu().item()),
            "soft_inlier_frac": float(resp_new.mean().detach().cpu().item()),
            "hard_inlier_frac": float((resp_new > 0.5).to(dtype=dtype).mean().detach().cpu().item()),
            "mean_entropy": float((-(resp_new * resp_new.log() + (1.0 - resp_new) * (1.0 - resp_new).log())).mean().detach().cpu().item()),
            "mean_lengthscale": float(hp["lengthscale"].mean().detach().cpu().item()),
            "mean_signal_var": float(hp["signal_var"].mean().detach().cpu().item()),
            "mean_noise_var": float(hp["noise_var"].mean().detach().cpu().item()),
        })
        resp = resp_new
    if hp is None or cfg.refit_after_update:
        hp = fit_uncertain_kernel_hyperparams_from_labels(
            X_anchor_t,
            X_neighbors_t,
            y_t,
            resp,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            init_lengthscale=init_lengthscale if hp is None else hp["lengthscale"],
            init_signal_var=init_signal_var if hp is None else hp["signal_var"],
            init_noise_var=init_noise_var if hp is None else hp["noise_var"],
            config=cfg.hyperopt,
            min_inliers=cfg.min_inliers,
        )
        gate_fit = fit_uncertain_gate_from_labels(
            X_anchor_t,
            X_neighbors_t,
            resp,
            mode=mode,
            W_mu_anchor=W_mu_a,
            W_var_anchor=W_var_a,
            W_cov_anchor=W_cov_a,
            W_mu_all=W_mu_p,
            W_cov_all=W_cov_p,
            gate_init=gate,
            config=cfg.gate,
        )
        gate = gate_fit["gate"]  # type: ignore[assignment]
    return {
        "labels": resp.detach(),
        "responsibilities": resp.detach(),
        "lengthscale": hp["lengthscale"].detach(),
        "signal_var": hp["signal_var"].detach(),
        "noise_var": hp["noise_var"].detach(),
        "mean_const": hp["mean_const"].detach(),
        "gate": gate.detach() if isinstance(gate, torch.Tensor) else gate,
        "outlier_logp": torch.empty(0, device=device, dtype=dtype),
        "history": history,
    }


def _diag_sparse_gp_kl(
    inducing_X: torch.Tensor,
    R_mu: torch.Tensor,
    R_log_std: torch.Tensor,
    *,
    lengthscale: float,
    signal_var: float,
    jitter: float,
) -> torch.Tensor:
    """KL(q(R)||p(R)) for independent q,d outputs with diagonal q covariance."""
    U = inducing_X
    M, Q, D = R_mu.shape
    K = rbf_kernel(U, U, lengthscale=float(lengthscale), signal_var=float(signal_var))
    K = _symmetrise(K) + float(jitter) * torch.eye(M, device=U.device, dtype=U.dtype)
    L = _cholesky_psd(K.unsqueeze(0), jitter=jitter).squeeze(0)
    K_inv = torch.cholesky_inverse(L)
    diag_inv = torch.diagonal(K_inv)
    logdet_K = 2.0 * torch.diagonal(L).log().sum()
    sigma2 = torch.exp(2.0 * R_log_std).clamp_min(1e-12)
    kl = R_mu.new_zeros(())
    for q in range(Q):
        for d in range(D):
            mu = R_mu[:, q, d]
            s2 = sigma2[:, q, d]
            trace = (diag_inv * s2).sum()
            quad = mu @ (K_inv @ mu)
            logdet_q = s2.log().sum()
            kl = kl + 0.5 * (trace + quad - M + logdet_K - logdet_q)
    return kl


def sparse_gp_uncertain_w_moments_for_mode(
    *,
    mode: WMode,
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    R_mu: torch.Tensor | np.ndarray,
    R_log_std: torch.Tensor | np.ndarray,
    lengthscale: float,
    signal_var: float = 1.0,
    include_conditional_residual: bool = True,
    jitter: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return q(W) moments induced by sparse-GP q(R) in anchor or pointwise mode."""
    Xa = _as_float_tensor(X_anchor)
    device = Xa.device
    dtype = Xa.dtype
    Xn = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    U = _as_float_tensor(inducing_X, device=device).to(dtype=dtype)
    Rm = _as_float_tensor(R_mu, device=device).to(dtype=dtype)
    Rls = _as_float_tensor(R_log_std, device=device).to(dtype=dtype)
    Rv = torch.exp(2.0 * Rls).clamp_min(1e-12)
    if mode == "anchor":
        W_mu_all, W_cov_all = sparse_gp_w_moments_diag(
            Xa.unsqueeze(1),
            U,
            Rm,
            Rv,
            lengthscale=float(lengthscale),
            signal_var=float(signal_var),
            include_conditional_residual=bool(include_conditional_residual),
            jitter=float(jitter),
        )
        return {
            "W_mu_anchor": W_mu_all[:, 0],
            "W_var_anchor": W_cov_all[..., 0, 0],
        }
    if mode == "pointwise":
        X_all = torch.cat([Xa.unsqueeze(1), Xn], dim=1)
        W_mu_all, W_cov_all = sparse_gp_w_moments_diag(
            X_all,
            U,
            Rm,
            Rv,
            lengthscale=float(lengthscale),
            signal_var=float(signal_var),
            include_conditional_residual=bool(include_conditional_residual),
            jitter=float(jitter),
        )
        return {"W_mu_all": W_mu_all, "W_cov_all": W_cov_all}
    raise ValueError(f"Unknown mode={mode!r}.")


def sparse_gp_lowrank_residual_w_moments_for_mode(
    *,
    mode: WMode,
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    C_mu: torch.Tensor | np.ndarray,
    C_log_std: torch.Tensor | np.ndarray,
    base_W: torch.Tensor | np.ndarray,
    basis_V: torch.Tensor | np.ndarray,
    lengthscale: float,
    signal_var: float = 1.0,
    include_conditional_residual: bool = True,
    jitter: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return q(W) moments for ``W(x)=W0+C(x)V^T``.

    The first implementation supports the anchor-conditioned local-constant
    projection used by DJGP.  It keeps the exact full row covariance induced by
    the low-rank residual coefficients, so expected kernels use
    ``delta^T V S_C V^T delta`` rather than a diagonal-W approximation.
    """
    if mode != "anchor":
        raise NotImplementedError("Low-rank residual q(R) currently supports anchor mode only.")
    Xa = _as_float_tensor(X_anchor)
    device = Xa.device
    dtype = Xa.dtype
    U = _as_float_tensor(inducing_X, device=device).to(dtype=dtype)
    Cm = _as_float_tensor(C_mu, device=device).to(dtype=dtype)
    Cls = _as_float_tensor(C_log_std, device=device).to(dtype=dtype)
    W0 = _as_float_tensor(base_W, device=device).to(dtype=dtype)
    V = _as_float_tensor(basis_V, device=device).to(dtype=dtype)
    if Cm.shape != Cls.shape or Cm.dim() != 3:
        raise ValueError("C_mu/C_log_std must have shape [M,Q,r].")
    M, Q, r = Cm.shape
    if W0.dim() != 2 or W0.shape[0] != Q:
        raise ValueError(f"base_W must have shape [Q,D] with Q={Q}.")
    D = W0.shape[1]
    if V.shape != (D, r):
        raise ValueError(f"basis_V must have shape {(D, r)}, got {tuple(V.shape)}.")
    C_var = torch.exp(2.0 * Cls).clamp_min(1e-12)
    C_mu_all, C_cov_all = sparse_gp_w_moments_diag(
        Xa.unsqueeze(1),
        U,
        Cm,
        C_var,
        lengthscale=float(lengthscale),
        signal_var=float(signal_var),
        include_conditional_residual=bool(include_conditional_residual),
        jitter=float(jitter),
    )
    C_mu_a = C_mu_all[:, 0]
    C_var_a = C_cov_all[..., 0, 0].clamp_min(0.0)
    W_mu = W0.unsqueeze(0) + torch.einsum("tqr,dr->tqd", C_mu_a, V)
    W_cov = torch.einsum("tqr,dr,er->tqde", C_var_a, V, V)
    W_var = torch.diagonal(W_cov, dim1=-2, dim2=-1).contiguous()
    return {
        "W_mu_anchor": W_mu,
        "W_var_anchor": W_var,
        "W_cov_anchor": _symmetrise(W_cov),
    }


def _effective_lowrank_basis(basis_V: torch.Tensor, basis_scale: torch.Tensor | None) -> torch.Tensor:
    if basis_scale is None:
        return basis_V
    return basis_V * basis_scale.to(device=basis_V.device, dtype=basis_V.dtype).view(1, -1)


def _residual_column_group_penalty(
    w_moments: dict[str, torch.Tensor],
    base_W: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Group-lasso style penalty on the residual columns of ``W-W0``."""
    W_mu = w_moments["W_mu_anchor"]
    delta_mu = W_mu - base_W.to(device=W_mu.device, dtype=W_mu.dtype).unsqueeze(0)
    usage_sq = delta_mu.pow(2).sum(dim=1).mean(dim=0)
    usage = torch.sqrt(usage_sq + float(eps))
    penalty = usage.sum()
    diag_var = w_moments.get("W_var_anchor")
    if diag_var is None and "W_cov_anchor" in w_moments:
        diag_var = torch.diagonal(w_moments["W_cov_anchor"], dim1=-2, dim2=-1).contiguous()
    relevance_sq = usage_sq if diag_var is None else (delta_mu.pow(2) + diag_var.clamp_min(0.0)).sum(dim=1).mean(dim=0)
    rel_total = relevance_sq.sum().clamp_min(1e-12)
    top_frac = relevance_sq.max() / rel_total
    diag = {
        "residual_group_penalty": penalty.detach(),
        "residual_group_usage_mean": usage.detach().mean(),
        "residual_group_usage_max": usage.detach().max(),
        "residual_group_top_feature_frac": top_frac.detach(),
    }
    return penalty, diag


def _orthogonality_penalty(
    R_mu: torch.Tensor,
    *,
    target_scale: float,
    normalize: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalty that makes each inducing mean row-block close to semi-orthogonal."""
    if R_mu.dim() != 3:
        raise ValueError("R_mu must have shape [M,Q,D].")
    _M, Q, D = R_mu.shape
    gram = torch.matmul(R_mu, R_mu.transpose(-1, -2))
    eye = torch.eye(Q, device=R_mu.device, dtype=R_mu.dtype).unsqueeze(0)
    diff = gram - float(target_scale) * eye
    per_block = diff.pow(2).sum(dim=(-2, -1))
    if bool(normalize):
        per_block = per_block / float(max(1, Q * Q))
    penalty = per_block.mean()
    diag_sq = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(0.0)
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    diag = {
        "ortho_penalty": penalty.detach(),
        "ortho_frob_mean": diff.pow(2).sum(dim=(-2, -1)).sqrt().detach().mean(),
        "ortho_row_norm_mean": diag_sq.sqrt().detach().mean(),
        "ortho_row_norm_sq_mean": diag_sq.detach().mean(),
        "ortho_offdiag_abs_mean": off.detach().abs().mean(),
        "ortho_target_scale": torch.as_tensor(float(target_scale), device=R_mu.device, dtype=R_mu.dtype),
        "ortho_normalized": torch.as_tensor(float(1.0 if normalize else 0.0), device=R_mu.device, dtype=R_mu.dtype),
        "ortho_last_dim": torch.as_tensor(float(D), device=R_mu.device, dtype=R_mu.dtype),
    }
    return penalty, diag


def _cayley_factor(raw: torch.Tensor) -> torch.Tensor:
    """Orthogonal Cayley transform from an unconstrained square matrix."""
    if raw.dim() != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Cayley transform expects a square matrix.")
    skew = raw - raw.transpose(-1, -2)
    eye = torch.eye(raw.shape[0], device=raw.device, dtype=raw.dtype)
    return torch.linalg.solve(eye + skew, eye - skew)


def _initial_shared_svd_factors(R0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initial shared SVD factors for ``R_l = U diag(s_l) V^T``."""
    if R0.dim() != 3:
        raise ValueError("R0 must have shape [M,Q,D].")
    M, Q, D = R0.shape
    if Q > D:
        raise ValueError("shared_svd q(R) mean requires Q <= D.")
    ref = R0.mean(dim=0)
    U0, _S0, Vh0 = torch.linalg.svd(ref, full_matrices=False)
    U0 = U0[:, :Q].contiguous()
    V0 = Vh0[:Q, :].transpose(0, 1).contiguous()
    coeff = torch.einsum("aq,mab,bq->mq", U0, R0, V0).abs().clamp_min(1e-4)
    if M > 0:
        fallback = torch.linalg.svdvals(ref)[:Q].clamp_min(1e-4)
        bad = ~torch.isfinite(coeff)
        if bool(bad.any().item()):
            coeff = torch.where(bad, fallback.view(1, Q).expand_as(coeff), coeff)
    return U0, V0, coeff.log()


def _shared_svd_mean(
    left_raw: torch.Tensor,
    right_raw: torch.Tensor,
    log_scale: torch.Tensor,
    left_base: torch.Tensor,
    right_base: torch.Tensor,
    *,
    log_scale_min: float,
    log_scale_max: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build ``R_mu`` from shared Cayley/Stiefel factors and positive scales."""
    U = _cayley_factor(left_raw) @ left_base
    V = _cayley_factor(right_raw) @ right_base
    s = torch.exp(log_scale.clamp(float(log_scale_min), float(log_scale_max)))
    R_mu = torch.einsum("ab,mb,db->mad", U, s, V)
    eye_u = torch.eye(U.shape[1], device=U.device, dtype=U.dtype)
    eye_v = torch.eye(V.shape[1], device=V.device, dtype=V.dtype)
    diag = {
        "svd_scale_mean": s.detach().mean(),
        "svd_scale_min": s.detach().min(),
        "svd_scale_max": s.detach().max(),
        "svd_left_orth_error": (U.transpose(0, 1) @ U - eye_u).pow(2).sum().sqrt().detach(),
        "svd_right_orth_error": (V.transpose(0, 1) @ V - eye_v).pow(2).sum().sqrt().detach(),
    }
    return R_mu, diag


def _hard_cem_marginal_bound_from_w_moments(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
    gate: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    outlier_mode: OutlierMode,
    outlier_sigma_mult: float,
    background_scale_mult: float,
    outlier_logp: torch.Tensor | None = None,
    gh_points: int,
    jitter: float,
    min_inliers: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Hard-label uncertain-W CEM surrogate bound from precomputed q(W) moments."""
    device = X_anchor.device
    dtype = X_anchor.dtype
    T, n, _D = X_neighbors.shape
    labels_f = labels.to(device=device, dtype=dtype)
    labels_b = labels_f > 0.5
    soft_labels = bool(((labels_f > 1e-6) & (labels_f < 1.0 - 1e-6)).any().item())
    if mode == "anchor":
        if "W_cov_anchor" in w_moments:
            gate_mean, gate_var = gate_moments_anchor_fullcov(
                X_anchor,
                X_neighbors,
                w_moments["W_mu_anchor"],
                w_moments["W_cov_anchor"],
                gate,
            )
        else:
            gate_mean, gate_var = gate_moments_anchor(
                X_anchor,
                X_neighbors,
                w_moments["W_mu_anchor"],
                w_moments["W_var_anchor"],
                gate,
            )
        Q = w_moments["W_mu_anchor"].shape[1]
    else:
        X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
        gate_mean, gate_var = gate_moments_pointwise(
            X_all,
            w_moments["W_mu_all"],
            w_moments["W_cov_all"],
            gate,
        )
        Q = w_moments["W_mu_all"].shape[2]
    elog_pos, elog_neg, _epi = gh_logsigmoid_expectations(gate_mean, gate_var, gh_points=int(gh_points))
    gate_bound = (labels_f * elog_pos + (1.0 - labels_f) * elog_neg).sum(dim=1)
    noise_ref = float(torch.as_tensor(noise_var).detach().reshape(-1).median().cpu().item())
    out_logp = _outlier_logprob(
        y_neighbors,
        noise_var=noise_ref,
        mode=outlier_mode,
        outlier_sigma_mult=float(outlier_sigma_mult),
        background_scale_mult=float(background_scale_mult),
        constant_logp=outlier_logp,
    )
    out_bound = ((1.0 - labels_f) * out_logp).sum(dim=1)
    gp_logs: list[torch.Tensor] = []
    n_in = torch.empty(T, device=device, dtype=dtype)
    fallback = 0
    for t in range(T):
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, _k_star, _cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=w_moments.get("W_mu_anchor"),
            W_var_anchor=w_moments.get("W_var_anchor"),
            W_cov_anchor=w_moments.get("W_cov_anchor"),
            W_mu_all=w_moments.get("W_mu_all"),
            W_cov_all=w_moments.get("W_cov_all"),
            need_cov_k=False,
        )
        if soft_labels:
            weights = labels_f[t].clamp_min(0.0)
            mask = weights > 1e-6
        else:
            weights = None
            mask = labels_b[t]
        if int(mask.sum().item()) < int(min_inliers):
            fallback += 1
            k = min(max(int(min_inliers), 1), n)
            idx = torch.topk(labels_f[t], k=k, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[idx] = True
            if weights is not None:
                weights = weights.clone()
                weights[mask] = weights[mask].clamp_min(1.0)
        n_in[t] = labels_f[t].sum() if soft_labels else mask.to(dtype=dtype).sum()
        y_i = y_neighbors[t, mask]
        m = mean_const[t]
        K_i = K[mask][:, mask]
        if weights is None:
            noise_diag = noise_t.expand(K_i.shape[0])
            n_eff = torch.as_tensor(y_i.numel(), device=device, dtype=dtype)
        else:
            w_i = weights[mask].clamp_min(1e-6)
            noise_diag = noise_t / w_i
            n_eff = w_i.sum().clamp_min(1.0)
        Ky = K_i + torch.diag(noise_diag)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        yc = y_i - m
        alpha = torch.cholesky_solve(yc.unsqueeze(-1), L).squeeze(-1)
        quad = torch.dot(yc, alpha)
        logdet = 2.0 * torch.diagonal(L).log().sum()
        gp_logs.append(-0.5 * (quad + logdet + n_eff * math.log(2.0 * math.pi)))
    gp_bound = torch.stack(gp_logs)
    total = gp_bound + gate_bound + out_bound
    diag = {
        "gp_bound_mean": gp_bound.mean().detach(),
        "gate_bound_mean": gate_bound.mean().detach(),
        "outlier_bound_mean": out_bound.mean().detach(),
        "bound_mean": total.mean().detach(),
        "mean_n_inliers": n_in.mean().detach(),
        "fallback_count": torch.as_tensor(float(fallback), device=device, dtype=dtype),
    }
    return total.mean(), diag


def _hard_cem_bound_per_anchor_from_w_moments(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
    gate: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    outlier_mode: OutlierMode,
    outlier_sigma_mult: float,
    background_scale_mult: float,
    outlier_logp: torch.Tensor | None = None,
    gh_points: int,
    jitter: float,
    min_inliers: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-anchor version of the frozen hard-CEM surrogate bound."""
    device = X_anchor.device
    dtype = X_anchor.dtype
    T, n, _D = X_neighbors.shape
    labels_f = labels.to(device=device, dtype=dtype)
    labels_b = labels_f > 0.5
    soft_labels = bool(((labels_f > 1e-6) & (labels_f < 1.0 - 1e-6)).any().item())
    if mode == "anchor":
        if "W_cov_anchor" in w_moments:
            gate_mean, gate_var = gate_moments_anchor_fullcov(
                X_anchor,
                X_neighbors,
                w_moments["W_mu_anchor"],
                w_moments["W_cov_anchor"],
                gate,
            )
        else:
            gate_mean, gate_var = gate_moments_anchor(
                X_anchor,
                X_neighbors,
                w_moments["W_mu_anchor"],
                w_moments["W_var_anchor"],
                gate,
            )
        Q = w_moments["W_mu_anchor"].shape[1]
    else:
        X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
        gate_mean, gate_var = gate_moments_pointwise(
            X_all,
            w_moments["W_mu_all"],
            w_moments["W_cov_all"],
            gate,
        )
        Q = w_moments["W_mu_all"].shape[2]
    elog_pos, elog_neg, _epi = gh_logsigmoid_expectations(gate_mean, gate_var, gh_points=int(gh_points))
    gate_bound = (labels_f * elog_pos + (1.0 - labels_f) * elog_neg).sum(dim=1)
    noise_ref = float(torch.as_tensor(noise_var).detach().reshape(-1).median().cpu().item())
    out_logp = _outlier_logprob(
        y_neighbors,
        noise_var=noise_ref,
        mode=outlier_mode,
        outlier_sigma_mult=float(outlier_sigma_mult),
        background_scale_mult=float(background_scale_mult),
        constant_logp=outlier_logp,
    )
    out_bound = ((1.0 - labels_f) * out_logp).sum(dim=1)
    gp_logs: list[torch.Tensor] = []
    n_in = torch.empty(T, device=device, dtype=dtype)
    fallback = 0
    for t in range(T):
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, _k_star, _cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=w_moments.get("W_mu_anchor"),
            W_var_anchor=w_moments.get("W_var_anchor"),
            W_cov_anchor=w_moments.get("W_cov_anchor"),
            W_mu_all=w_moments.get("W_mu_all"),
            W_cov_all=w_moments.get("W_cov_all"),
            need_cov_k=False,
        )
        if soft_labels:
            weights = labels_f[t].clamp_min(0.0)
            mask = weights > 1e-6
        else:
            weights = None
            mask = labels_b[t]
        if int(mask.sum().item()) < int(min_inliers):
            fallback += 1
            k = min(max(int(min_inliers), 1), n)
            idx = torch.topk(labels_f[t], k=k, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[idx] = True
            if weights is not None:
                weights = weights.clone()
                weights[mask] = weights[mask].clamp_min(1.0)
        n_in[t] = labels_f[t].sum() if soft_labels else mask.to(dtype=dtype).sum()
        y_i = y_neighbors[t, mask]
        m = mean_const[t]
        K_i = K[mask][:, mask]
        if weights is None:
            noise_diag = noise_t.expand(K_i.shape[0])
            n_eff = torch.as_tensor(y_i.numel(), device=device, dtype=dtype)
        else:
            w_i = weights[mask].clamp_min(1e-6)
            noise_diag = noise_t / w_i
            n_eff = w_i.sum().clamp_min(1.0)
        Ky = K_i + torch.diag(noise_diag)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        yc = y_i - m
        alpha = torch.cholesky_solve(yc.unsqueeze(-1), L).squeeze(-1)
        quad = torch.dot(yc, alpha)
        logdet = 2.0 * torch.diagonal(L).log().sum()
        gp_logs.append(-0.5 * (quad + logdet + n_eff * math.log(2.0 * math.pi)))
    gp_bound = torch.stack(gp_logs)
    total = gp_bound + gate_bound + out_bound
    diag = {
        "gp_bound_mean": gp_bound.mean().detach(),
        "gate_bound_mean": gate_bound.mean().detach(),
        "outlier_bound_mean": out_bound.mean().detach(),
        "bound_mean": total.mean().detach(),
        "mean_n_inliers": n_in.mean().detach(),
        "fallback_count": torch.as_tensor(float(fallback), device=device, dtype=dtype),
    }
    return total, diag


def _expected_neighbor_pair_sqdist(
    X_neighbors: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Expected squared projected distances among local neighbors under q(W)."""
    if mode == "anchor":
        W_mu = w_moments["W_mu_anchor"]
        delta = X_neighbors.unsqueeze(2) - X_neighbors.unsqueeze(1)
        mean_proj = torch.einsum("tqd,tnmd->tnmq", W_mu, delta)
        mean_sq = mean_proj.pow(2).sum(dim=-1)
        if "W_cov_anchor" in w_moments:
            var_sq = torch.einsum("tqde,tnmd,tnme->tnmq", w_moments["W_cov_anchor"], delta, delta).sum(dim=-1)
        else:
            W_var = w_moments["W_var_anchor"]
            var_sq = (W_var[:, None, None, :, :] * delta.pow(2)[:, :, :, None, :]).sum(dim=(-1, -2))
        return (mean_sq + var_sq).clamp_min(0.0)

    if mode == "pointwise":
        W_mu_all = w_moments["W_mu_all"]
        W_cov_all = w_moments["W_cov_all"]
        W_mu_n = W_mu_all[:, 1:]
        Z_mu = (W_mu_n * X_neighbors[:, :, None, :]).sum(dim=-1)
        mean_sq = (Z_mu.unsqueeze(2) - Z_mu.unsqueeze(1)).pow(2).sum(dim=-1)
        cov_nn = W_cov_all[..., 1:, 1:]
        diag_var = torch.diagonal(cov_nn, dim1=-2, dim2=-1).permute(0, 3, 1, 2).contiguous()
        var_z = (diag_var * X_neighbors[:, :, None, :].pow(2)).sum(dim=(-1, -2))
        xprod = X_neighbors.unsqueeze(2) * X_neighbors.unsqueeze(1)
        cross_cov = (cov_nn.permute(0, 3, 4, 1, 2) * xprod[:, :, :, None, :]).sum(dim=(-1, -2))
        var_sq = var_z.unsqueeze(2) + var_z.unsqueeze(1) - 2.0 * cross_cov
        return (mean_sq + var_sq).clamp_min(0.0)

    raise ValueError(f"Unknown mode={mode!r}.")


def _expected_anchor_neighbor_sqdist(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Expected squared projected anchor-to-neighbor distances under q(W)."""
    if mode == "anchor":
        W_mu = w_moments["W_mu_anchor"]
        delta = X_neighbors - X_anchor.unsqueeze(1)
        mean_proj = torch.einsum("tqd,tnd->tnq", W_mu, delta)
        mean_sq = mean_proj.pow(2).sum(dim=-1)
        if "W_cov_anchor" in w_moments:
            var_sq = torch.einsum("tqde,tnd,tne->tnq", w_moments["W_cov_anchor"], delta, delta).sum(dim=-1)
        else:
            W_var = w_moments["W_var_anchor"]
            var_sq = (W_var[:, None, :, :] * delta[:, :, None, :].pow(2)).sum(dim=(-1, -2))
        return (mean_sq + var_sq).clamp_min(0.0)

    if mode == "pointwise":
        W_mu_all = w_moments["W_mu_all"]
        W_cov_all = w_moments["W_cov_all"]
        X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
        z_mu, z_cov = pointwise_projected_moments(X_all, W_mu_all, W_cov_all)
        mean_sq = (z_mu[:, 1:] - z_mu[:, :1]).pow(2).sum(dim=-1)
        diag_cov = torch.diagonal(z_cov, dim1=1, dim2=2).permute(0, 2, 1)
        var_sq = (diag_cov[:, 1:] + diag_cov[:, :1] - 2.0 * z_cov[:, 1:, :1, :].squeeze(2)).sum(dim=-1)
        return (mean_sq + var_sq).clamp_min(0.0)

    raise ValueError(f"Unknown mode={mode!r}.")


def _anchor_retrieval_ranking_loss(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
    temperature: float,
    normalize: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Anchor-centric retrieval loss: inliers should rank ahead of outliers."""
    device = X_neighbors.device
    dtype = X_neighbors.dtype
    dist = _expected_anchor_neighbor_sqdist(X_anchor, X_neighbors, mode=mode, w_moments=w_moments)
    if bool(normalize):
        scale = dist.detach().median().clamp_min(1e-6)
        dist_eff = dist / scale
    else:
        scale = torch.ones((), device=device, dtype=dtype)
        dist_eff = dist
    labels_b = labels.to(device=device).bool()
    pos = labels_b
    neg = ~labels_b
    pair_mask = pos.unsqueeze(2) & neg.unsqueeze(1)
    zero = dist.new_zeros(())
    if not bool(pair_mask.any().item()):
        diag = {
            "rank_loss": zero.detach(),
            "rank_pos_dist_mean": zero.detach(),
            "rank_neg_dist_mean": zero.detach(),
            "rank_pair_accuracy": zero.detach(),
            "rank_scale": scale.detach(),
            "rank_pair_count": torch.as_tensor(0.0, device=device, dtype=dtype),
        }
        return zero, diag
    pos_d = dist_eff.unsqueeze(2)
    neg_d = dist_eff.unsqueeze(1)
    logits = (neg_d - pos_d) / max(float(temperature), 1e-6)
    vals = F.softplus(-logits)[pair_mask]
    loss = vals.mean()
    pos_vals = dist_eff[pos]
    neg_vals = dist_eff[neg]
    acc = (neg_d > pos_d).to(dtype=dtype)[pair_mask].mean()
    diag = {
        "rank_loss": loss.detach(),
        "rank_pos_dist_mean": pos_vals.detach().mean() if pos_vals.numel() else zero.detach(),
        "rank_neg_dist_mean": neg_vals.detach().mean() if neg_vals.numel() else zero.detach(),
        "rank_pair_accuracy": acc.detach(),
        "rank_scale": scale.detach(),
        "rank_pair_count": torch.as_tensor(float(vals.numel()), device=device, dtype=dtype),
    }
    return loss, diag


def _pairwise_projection_auxiliary_loss(
    X_neighbors: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
    margin: float,
    same_weight: float,
    cross_weight: float,
    normalize: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrastive auxiliary loss on q(W)-induced neighbor geometry.

    Inlier-inlier pairs are pulled together.  Inlier-outlier pairs are pushed
    apart up to ``margin``.  Outlier-outlier pairs are ignored because a local
    CEM outlier set may contain multiple regimes.
    """
    device = X_neighbors.device
    dtype = X_neighbors.dtype
    T, n, _D = X_neighbors.shape
    dist = _expected_neighbor_pair_sqdist(X_neighbors, mode=mode, w_moments=w_moments)
    upper = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0)
    labels_b = labels.to(device=device).bool()
    same = labels_b.unsqueeze(2) & labels_b.unsqueeze(1) & upper
    cross = (labels_b.unsqueeze(2) ^ labels_b.unsqueeze(1)) & upper
    if bool(normalize):
        upper_vals = dist.detach()[upper.expand(T, n, n)]
        scale = upper_vals.median().clamp_min(1e-6) if upper_vals.numel() else dist.detach().median().clamp_min(1e-6)
        dist_eff = dist / scale
    else:
        scale = torch.ones((), device=device, dtype=dtype)
        dist_eff = dist
    zero = dist.new_zeros(())
    same_vals = dist_eff[same]
    cross_vals = dist_eff[cross]
    same_loss = same_vals.mean() if same_vals.numel() else zero
    cross_gap = torch.relu(float(margin) - cross_vals)
    cross_loss = cross_gap.pow(2).mean() if cross_vals.numel() else zero
    loss = float(same_weight) * same_loss + float(cross_weight) * cross_loss
    diag = {
        "pairwise_loss": loss.detach(),
        "pairwise_same_loss": same_loss.detach(),
        "pairwise_cross_loss": cross_loss.detach(),
        "pairwise_same_dist_mean": same_vals.detach().mean() if same_vals.numel() else zero.detach(),
        "pairwise_cross_dist_mean": cross_vals.detach().mean() if cross_vals.numel() else zero.detach(),
        "pairwise_cross_violation_rate": (cross_vals.detach() < float(margin)).to(dtype=dtype).mean()
        if cross_vals.numel()
        else zero.detach(),
        "pairwise_scale": scale.detach(),
        "pairwise_same_count": torch.as_tensor(float(same_vals.numel()), device=device, dtype=dtype),
        "pairwise_cross_count": torch.as_tensor(float(cross_vals.numel()), device=device, dtype=dtype),
    }
    return loss, diag


def _anchor_predictive_nll_from_w_moments(
    X_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    jitter: float,
    min_inliers: int,
    pred_var_floor: float,
    exclude_self: bool,
    self_tol: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean leave-anchor-out predictive NLL for supervised train anchors."""
    device = X_anchor.device
    dtype = X_anchor.dtype
    T, n, _D = X_neighbors.shape
    losses: list[torch.Tensor] = []
    pred_vars: list[torch.Tensor] = []
    used_counts: list[torch.Tensor] = []
    fallback = 0
    labels_t = labels.to(device=device, dtype=dtype)
    for t in range(T):
        if mode == "anchor":
            Q = w_moments["W_mu_anchor"].shape[1]
        else:
            Q = w_moments["W_mu_all"].shape[2]
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, k_star, _cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=w_moments.get("W_mu_anchor"),
            W_var_anchor=w_moments.get("W_var_anchor"),
            W_cov_anchor=w_moments.get("W_cov_anchor"),
            W_mu_all=w_moments.get("W_mu_all"),
            W_cov_all=w_moments.get("W_cov_all"),
            need_cov_k=False,
        )
        allowed = torch.ones(n, device=device, dtype=torch.bool)
        if bool(exclude_self):
            raw_d2 = (X_neighbors[t] - X_anchor[t]).pow(2).sum(dim=-1)
            allowed = raw_d2 > float(self_tol)
        mask = (labels_t[t] > 0.5) & allowed
        if int(mask.sum().item()) < int(min_inliers):
            fallback += 1
            k = min(max(int(min_inliers), 1), int(allowed.sum().item()))
            if k <= 0:
                allowed = torch.ones(n, device=device, dtype=torch.bool)
                k = min(max(int(min_inliers), 1), n)
            score = labels_t[t].masked_fill(~allowed, -1.0)
            top_idx = torch.topk(score, k=k, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[top_idx] = True
        y_i = y_neighbors[t, mask]
        K_i = K[mask][:, mask]
        k_i = k_star[mask]
        m = mean_const[t]
        Ky = K_i + noise_t * torch.eye(K_i.shape[0], device=device, dtype=dtype)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        alpha = torch.cholesky_solve((y_i - m).unsqueeze(-1), L).squeeze(-1)
        pred_mean = m + torch.dot(k_i, alpha)
        v = torch.cholesky_solve(k_i.unsqueeze(-1), L).squeeze(-1)
        pred_var = (sig_t - torch.dot(k_i, v) + noise_t).clamp_min(float(pred_var_floor))
        err = y_anchor[t] - pred_mean
        losses.append(0.5 * (torch.log(2.0 * torch.pi * pred_var) + err.pow(2) / pred_var))
        pred_vars.append(pred_var)
        used_counts.append(torch.as_tensor(float(mask.sum().item()), device=device, dtype=dtype))
    loss = torch.stack(losses).mean() if losses else X_anchor.new_zeros(())
    pred_var_t = torch.stack(pred_vars) if pred_vars else X_anchor.new_zeros((0,))
    used_t = torch.stack(used_counts) if used_counts else X_anchor.new_zeros((0,))
    diag = {
        "anchor_pred_nll": loss.detach(),
        "anchor_pred_var_mean": pred_var_t.detach().mean() if pred_var_t.numel() else X_anchor.new_zeros(()),
        "anchor_pred_mean_n_inliers": used_t.detach().mean() if used_t.numel() else X_anchor.new_zeros(()),
        "anchor_pred_fallback_count": torch.as_tensor(float(fallback), device=device, dtype=dtype),
    }
    return loss, diag


def _anchor_predictive_logpdf_per_anchor_from_w_moments(
    X_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: WMode,
    w_moments: dict[str, torch.Tensor],
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    jitter: float,
    min_inliers: int,
    pred_var_floor: float,
    exclude_self: bool,
    self_tol: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-anchor supervised predictive log-density for score-function rewards."""
    device = X_anchor.device
    dtype = X_anchor.dtype
    T, n, _D = X_neighbors.shape
    vals: list[torch.Tensor] = []
    pred_vars: list[torch.Tensor] = []
    used_counts: list[torch.Tensor] = []
    fallback = 0
    labels_t = labels.to(device=device, dtype=dtype)
    for t in range(T):
        Q = w_moments["W_mu_anchor"].shape[1] if mode == "anchor" else w_moments["W_mu_all"].shape[2]
        ell_t = _param_for_anchor(lengthscale, t, Q, device=device, dtype=dtype)
        sig_t = _scalar_for_anchor(signal_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        noise_t = _scalar_for_anchor(noise_var, t, device=device, dtype=dtype).clamp_min(1e-12)
        K, k_star, _cov_k = _single_kernel_blocks(
            t,
            mode=mode,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            lengthscale=ell_t,
            signal_var=sig_t,
            W_mu_anchor=w_moments.get("W_mu_anchor"),
            W_var_anchor=w_moments.get("W_var_anchor"),
            W_cov_anchor=w_moments.get("W_cov_anchor"),
            W_mu_all=w_moments.get("W_mu_all"),
            W_cov_all=w_moments.get("W_cov_all"),
            need_cov_k=False,
        )
        allowed = torch.ones(n, device=device, dtype=torch.bool)
        if bool(exclude_self):
            raw_d2 = (X_neighbors[t] - X_anchor[t]).pow(2).sum(dim=-1)
            allowed = raw_d2 > float(self_tol)
        mask = (labels_t[t] > 0.5) & allowed
        if int(mask.sum().item()) < int(min_inliers):
            fallback += 1
            k = min(max(int(min_inliers), 1), int(allowed.sum().item()))
            if k <= 0:
                allowed = torch.ones(n, device=device, dtype=torch.bool)
                k = min(max(int(min_inliers), 1), n)
            score = labels_t[t].masked_fill(~allowed, -1.0)
            top_idx = torch.topk(score, k=k, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[top_idx] = True
        y_i = y_neighbors[t, mask]
        K_i = K[mask][:, mask]
        k_i = k_star[mask]
        m = mean_const[t]
        Ky = K_i + noise_t * torch.eye(K_i.shape[0], device=device, dtype=dtype)
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        alpha = torch.cholesky_solve((y_i - m).unsqueeze(-1), L).squeeze(-1)
        pred_mean = m + torch.dot(k_i, alpha)
        v = torch.cholesky_solve(k_i.unsqueeze(-1), L).squeeze(-1)
        pred_var = (sig_t - torch.dot(k_i, v) + noise_t).clamp_min(float(pred_var_floor))
        err = y_anchor[t] - pred_mean
        vals.append(-0.5 * (torch.log(2.0 * torch.pi * pred_var) + err.pow(2) / pred_var))
        pred_vars.append(pred_var)
        used_counts.append(torch.as_tensor(float(mask.sum().item()), device=device, dtype=dtype))
    rewards = torch.stack(vals) if vals else X_anchor.new_zeros((T,))
    pred_var_t = torch.stack(pred_vars) if pred_vars else X_anchor.new_zeros((0,))
    used_t = torch.stack(used_counts) if used_counts else X_anchor.new_zeros((0,))
    diag = {
        "supervised_reward_mean": rewards.detach().mean(),
        "supervised_pred_var_mean": pred_var_t.detach().mean() if pred_var_t.numel() else X_anchor.new_zeros(()),
        "supervised_mean_n_inliers": used_t.detach().mean() if used_t.numel() else X_anchor.new_zeros(()),
        "supervised_fallback_count": torch.as_tensor(float(fallback), device=device, dtype=dtype),
    }
    return rewards, diag


def train_qr_uncertain_w_cem_mstep(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    y_anchor: torch.Tensor | np.ndarray | None = None,
    mode: WMode,
    inducing_X: torch.Tensor | np.ndarray,
    init_W: torch.Tensor | np.ndarray | None = None,
    init_R_mu: torch.Tensor | np.ndarray | None = None,
    init_basis_scale: torch.Tensor | np.ndarray | None = None,
    residual_basis: torch.Tensor | np.ndarray | None = None,
    gate: torch.Tensor | np.ndarray,
    lengthscale: float | tuple[float, ...] | torch.Tensor,
    signal_var: float | torch.Tensor,
    noise_var: float | torch.Tensor,
    mean_const: torch.Tensor | np.ndarray,
    config: UncertainWQRMstepConfig | None = None,
) -> UncertainWQRMstepResult:
    """Train sparse-GP ``q(R)`` with the moment-matched uncertain-W CEM M-step.

    The C-step state (labels, local hyperparameters, mean constants, and gate)
    is fixed.  The differentiable M-step optimizes ``q(R)`` through
    ``q(R) -> q(W) -> E[K(W)]`` and gate Gauss-Hermite terms, minus
    ``beta KL(q(R)||p(R))``.
    """
    cfg = config or UncertainWQRMstepConfig()
    Xa = _as_float_tensor(X_anchor)
    device = Xa.device
    dtype = Xa.dtype
    Xn = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    ya = None if y_anchor is None else _as_float_tensor(y_anchor, device=device).to(dtype=dtype).reshape(-1)
    lab = _as_float_tensor(labels, device=device).to(dtype=dtype)
    U = _as_float_tensor(inducing_X, device=device).to(dtype=dtype)
    gate_t = _as_float_tensor(gate, device=device).to(dtype=dtype)
    mean_t = _as_float_tensor(mean_const, device=device).to(dtype=dtype).reshape(-1)
    if y.dim() == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if lab.dim() == 3 and lab.shape[-1] == 1:
        lab = lab.squeeze(-1)
    T, n, D = Xn.shape
    M = U.shape[0]
    lowrank_basis = _as_float_tensor(residual_basis, device=device).to(dtype=dtype) if residual_basis is not None else None
    base_W = _as_float_tensor(init_W, device=device).to(dtype=dtype) if init_W is not None else None
    if lowrank_basis is not None:
        if mode != "anchor":
            raise NotImplementedError("Low-rank residual qR M-step currently supports anchor mode only.")
        if base_W is None or base_W.dim() != 2 or base_W.shape[1] != D:
            raise ValueError(f"Low-rank residual qR requires init_W/base_W with shape [Q,{D}].")
        if lowrank_basis.dim() != 2 or lowrank_basis.shape[0] != D:
            raise ValueError(f"residual_basis must have shape [D,r] with D={D}.")
        r = lowrank_basis.shape[1]
        if init_R_mu is not None:
            R0 = _as_float_tensor(init_R_mu, device=device).to(dtype=dtype)
            if R0.shape != (M, base_W.shape[0], r):
                raise ValueError(f"init_R_mu must have shape {(M, base_W.shape[0], r)} for low-rank qR.")
        else:
            R0 = torch.zeros(M, base_W.shape[0], r, device=device, dtype=dtype)
        if init_basis_scale is not None:
            scale0 = _as_float_tensor(init_basis_scale, device=device).to(dtype=dtype).reshape(-1)
            if scale0.shape != (r,):
                raise ValueError(f"init_basis_scale must have shape {(r,)} for low-rank qR.")
        else:
            scale0 = torch.ones(r, device=device, dtype=dtype)
    else:
        if init_basis_scale is not None:
            raise ValueError("init_basis_scale is only valid for low-rank residual qR.")
        if init_R_mu is not None:
            R0 = _as_float_tensor(init_R_mu, device=device).to(dtype=dtype)
            if R0.shape[:1] != (M,):
                raise ValueError("init_R_mu first dimension must match inducing_X.")
        elif base_W is not None:
            if base_W.dim() != 2 or base_W.shape[1] != D:
                raise ValueError(f"init_W must have shape [Q,{D}], got {tuple(base_W.shape)}.")
            R0 = base_W.unsqueeze(0).expand(M, -1, -1).contiguous()
        else:
            raise ValueError("Either init_W or init_R_mu is required.")
    Q = R0.shape[1]
    mean_param = str(cfg.mean_parameterization).lower()
    if mean_param not in {"free", "shared_svd"}:
        raise ValueError(f"Unknown qR mean_parameterization {cfg.mean_parameterization!r}.")
    if mean_param == "shared_svd" and lowrank_basis is not None:
        raise NotImplementedError("shared_svd qR mean is only implemented for full R, not low-rank residual C.")

    R_mu_free: torch.nn.Parameter | None = None
    svd_left_raw: torch.nn.Parameter | None = None
    svd_right_raw: torch.nn.Parameter | None = None
    svd_log_scale: torch.nn.Parameter | None = None
    svd_left_base: torch.Tensor | None = None
    svd_right_base: torch.Tensor | None = None
    if mean_param == "shared_svd":
        left_base, right_base, log_scale0 = _initial_shared_svd_factors(R0.detach())
        svd_left_base = left_base.detach()
        svd_right_base = right_base.detach()
        svd_left_raw = torch.nn.Parameter(torch.zeros(Q, Q, device=device, dtype=dtype))
        svd_right_raw = torch.nn.Parameter(torch.zeros(D, D, device=device, dtype=dtype))
        svd_log_scale = torch.nn.Parameter(log_scale0.detach().clone())
        R_shape = R0.shape
    else:
        R_mu_free = torch.nn.Parameter(R0.detach().clone())
        R_shape = R_mu_free.shape

    R_log_std = torch.nn.Parameter(torch.full(R_shape, float(cfg.init_log_std), device=device, dtype=dtype))
    R_log_std.requires_grad_(bool(cfg.train_log_std))
    params: list[torch.nn.Parameter] = []
    if mean_param == "shared_svd":
        params.extend([svd_left_raw, svd_right_raw, svd_log_scale])  # type: ignore[list-item]
    else:
        params.append(R_mu_free)  # type: ignore[arg-type]
    if R_log_std.requires_grad:
        params.append(R_log_std)
    basis_log_scale: torch.nn.Parameter | None = None
    if lowrank_basis is not None and bool(cfg.train_basis_scale):
        init_scale = scale0.clamp_min(1e-6)
        basis_log_scale = torch.nn.Parameter(init_scale.log())
        params.append(basis_log_scale)
    opt = torch.optim.Adam(params, lr=float(cfg.lr))
    ell_t = _as_float_tensor(lengthscale, device=device).to(dtype=dtype)
    sig_t = _as_float_tensor(signal_var, device=device).to(dtype=dtype)
    noise_t = _as_float_tensor(noise_var, device=device).to(dtype=dtype)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: list[torch.Tensor] | None = None
    t0 = time.perf_counter()
    for step in range(1, int(cfg.steps) + 1):
        opt.zero_grad()
        svd_diag: dict[str, torch.Tensor] = {}
        if mean_param == "shared_svd":
            R_mu, svd_diag = _shared_svd_mean(
                svd_left_raw,  # type: ignore[arg-type]
                svd_right_raw,  # type: ignore[arg-type]
                svd_log_scale,  # type: ignore[arg-type]
                svd_left_base,  # type: ignore[arg-type]
                svd_right_base,  # type: ignore[arg-type]
                log_scale_min=float(cfg.svd_log_scale_min),
                log_scale_max=float(cfg.svd_log_scale_max),
            )
        else:
            R_mu = R_mu_free  # type: ignore[assignment]
        if lowrank_basis is not None:
            basis_scale = (
                torch.exp(
                    basis_log_scale.clamp(
                        min=float(cfg.basis_scale_log_min),
                        max=float(cfg.basis_scale_log_max),
                    )
                )
                if basis_log_scale is not None
                else scale0
            )
            lowrank_basis_eff = _effective_lowrank_basis(lowrank_basis, basis_scale)
            w_moments = sparse_gp_lowrank_residual_w_moments_for_mode(
                mode=mode,
                X_anchor=Xa,
                X_neighbors=Xn,
                inducing_X=U,
                C_mu=R_mu,
                C_log_std=R_log_std,
                base_W=base_W,  # type: ignore[arg-type]
                basis_V=lowrank_basis_eff,
                lengthscale=float(cfg.w_lengthscale),
                signal_var=float(cfg.w_signal_var),
                include_conditional_residual=bool(cfg.include_conditional_residual),
                jitter=float(cfg.jitter),
            )
        else:
            w_moments = sparse_gp_uncertain_w_moments_for_mode(
                mode=mode,
                X_anchor=Xa,
                X_neighbors=Xn,
                inducing_X=U,
                R_mu=R_mu,
                R_log_std=R_log_std,
                lengthscale=float(cfg.w_lengthscale),
                signal_var=float(cfg.w_signal_var),
                include_conditional_residual=bool(cfg.include_conditional_residual),
                jitter=float(cfg.jitter),
            )
        bound, bound_diag = _hard_cem_marginal_bound_from_w_moments(
            Xa,
            Xn,
            y,
            lab,
            mode=mode,
            w_moments=w_moments,
            gate=gate_t,
            lengthscale=ell_t,
            signal_var=sig_t,
            noise_var=noise_t,
            mean_const=mean_t,
            outlier_mode=cfg.outlier_mode,
            outlier_sigma_mult=float(cfg.outlier_sigma_mult),
            background_scale_mult=float(cfg.background_scale_mult),
            gh_points=int(cfg.gh_points),
            jitter=float(cfg.jitter),
            min_inliers=int(cfg.min_inliers),
        )
        data_loss = -float(cfg.composite_temperature) * bound
        anchor_pred_loss = R_mu.new_zeros(())
        anchor_pred_term = R_mu.new_zeros(())
        anchor_pred_diag: dict[str, torch.Tensor] = {}
        if float(cfg.anchor_pred_weight) != 0.0:
            if ya is None:
                raise ValueError("anchor_pred_weight requires y_anchor.")
            anchor_pred_loss, anchor_pred_diag = _anchor_predictive_nll_from_w_moments(
                Xa,
                ya,
                Xn,
                y,
                lab,
                mode=mode,
                w_moments=w_moments,
                lengthscale=ell_t,
                signal_var=sig_t,
                noise_var=noise_t,
                mean_const=mean_t,
                jitter=float(cfg.jitter),
                min_inliers=int(cfg.min_inliers),
                pred_var_floor=float(cfg.anchor_pred_var_floor),
                exclude_self=bool(cfg.anchor_pred_exclude_self),
                self_tol=float(cfg.anchor_self_tol),
            )
            anchor_pred_term = float(cfg.anchor_pred_weight) * anchor_pred_loss
        pairwise_loss = R_mu.new_zeros(())
        pairwise_term = R_mu.new_zeros(())
        pairwise_diag: dict[str, torch.Tensor] = {}
        if float(cfg.pairwise_weight) != 0.0:
            pairwise_loss, pairwise_diag = _pairwise_projection_auxiliary_loss(
                Xn,
                lab,
                mode=mode,
                w_moments=w_moments,
                margin=float(cfg.pairwise_margin),
                same_weight=float(cfg.pairwise_same_weight),
                cross_weight=float(cfg.pairwise_cross_weight),
                normalize=bool(cfg.pairwise_normalize),
            )
            pairwise_term = float(cfg.pairwise_weight) * pairwise_loss
        rank_loss = R_mu.new_zeros(())
        rank_term = R_mu.new_zeros(())
        rank_diag: dict[str, torch.Tensor] = {}
        if float(cfg.rank_weight) != 0.0:
            rank_loss, rank_diag = _anchor_retrieval_ranking_loss(
                Xa,
                Xn,
                lab,
                mode=mode,
                w_moments=w_moments,
                temperature=float(cfg.rank_temperature),
                normalize=bool(cfg.rank_normalize),
            )
            rank_term = float(cfg.rank_weight) * rank_loss
        residual_group_loss = R_mu.new_zeros(())
        residual_group_term = R_mu.new_zeros(())
        residual_group_diag: dict[str, torch.Tensor] = {}
        if lowrank_basis is not None and float(cfg.residual_group_weight) != 0.0:
            residual_group_loss, residual_group_diag = _residual_column_group_penalty(
                w_moments,
                base_W,  # type: ignore[arg-type]
                eps=float(cfg.residual_group_eps),
            )
            residual_group_term = float(cfg.residual_group_weight) * residual_group_loss
        basis_scale_l1 = R_mu.new_zeros(())
        basis_scale_term = R_mu.new_zeros(())
        if lowrank_basis is not None and basis_log_scale is not None and float(cfg.basis_scale_l1_weight) != 0.0:
            basis_scale_l1 = basis_scale.sum()
            basis_scale_term = float(cfg.basis_scale_l1_weight) * basis_scale_l1
        ortho_loss = R_mu.new_zeros(())
        ortho_term = R_mu.new_zeros(())
        ortho_diag: dict[str, torch.Tensor] = {}
        if float(cfg.ortho_weight) != 0.0:
            ortho_loss, ortho_diag = _orthogonality_penalty(
                R_mu,
                target_scale=float(cfg.ortho_target_scale),
                normalize=bool(cfg.ortho_normalize),
            )
            ortho_term = float(cfg.ortho_weight) * ortho_loss
        kl = _diag_sparse_gp_kl(
            U,
            R_mu,
            R_log_std,
            lengthscale=float(cfg.w_lengthscale),
            signal_var=float(cfg.w_signal_var),
            jitter=float(cfg.jitter),
        ) / max(1, T)
        beta = float(cfg.beta_kl) * min(1.0, step / max(1, int(cfg.kl_warmup_steps)))
        loss = data_loss + anchor_pred_term + pairwise_term + rank_term + residual_group_term + basis_scale_term + ortho_term + beta * kl
        should_record = step == 1 or step % int(cfg.log_interval) == 0 or step == int(cfg.steps)
        grad_diag: dict[str, float] = {}
        if should_record:
            g_data_mu = torch.autograd.grad(data_loss, R_mu, retain_graph=True, allow_unused=True)[0]
            g_anchor_mu = (
                torch.autograd.grad(anchor_pred_term, R_mu, retain_graph=True, allow_unused=True)[0]
                if anchor_pred_term.requires_grad
                else None
            )
            g_kl_mu = torch.autograd.grad(beta * kl, R_mu, retain_graph=True, allow_unused=True)[0]
            g_pair_mu = (
                torch.autograd.grad(pairwise_term, R_mu, retain_graph=True, allow_unused=True)[0]
                if pairwise_term.requires_grad
                else None
            )
            g_rank_mu = (
                torch.autograd.grad(rank_term, R_mu, retain_graph=True, allow_unused=True)[0]
                if rank_term.requires_grad
                else None
            )
            g_residual_group_mu = (
                torch.autograd.grad(residual_group_term, R_mu, retain_graph=True, allow_unused=True)[0]
                if residual_group_term.requires_grad
                else None
            )
            g_basis_scale_mu = (
                torch.autograd.grad(basis_scale_term, R_mu, retain_graph=True, allow_unused=True)[0]
                if basis_scale_term.requires_grad
                else None
            )
            g_ortho_mu = (
                torch.autograd.grad(ortho_term, R_mu, retain_graph=True, allow_unused=True)[0]
                if ortho_term.requires_grad
                else None
            )
            data_mu = float(torch.linalg.norm(g_data_mu).detach().cpu().item()) if g_data_mu is not None else 0.0
            anchor_mu = float(torch.linalg.norm(g_anchor_mu).detach().cpu().item()) if g_anchor_mu is not None else 0.0
            kl_mu = float(torch.linalg.norm(g_kl_mu).detach().cpu().item()) if g_kl_mu is not None else 0.0
            pair_mu = float(torch.linalg.norm(g_pair_mu).detach().cpu().item()) if g_pair_mu is not None else 0.0
            rank_mu = float(torch.linalg.norm(g_rank_mu).detach().cpu().item()) if g_rank_mu is not None else 0.0
            residual_group_mu = (
                float(torch.linalg.norm(g_residual_group_mu).detach().cpu().item())
                if g_residual_group_mu is not None
                else 0.0
            )
            basis_scale_mu = (
                float(torch.linalg.norm(g_basis_scale_mu).detach().cpu().item())
                if g_basis_scale_mu is not None
                else 0.0
            )
            ortho_mu = float(torch.linalg.norm(g_ortho_mu).detach().cpu().item()) if g_ortho_mu is not None else 0.0
            if R_log_std.requires_grad:
                g_data_std = torch.autograd.grad(data_loss, R_log_std, retain_graph=True, allow_unused=True)[0]
                g_anchor_std = (
                    torch.autograd.grad(anchor_pred_term, R_log_std, retain_graph=True, allow_unused=True)[0]
                    if anchor_pred_term.requires_grad
                    else None
                )
                g_kl_std = torch.autograd.grad(beta * kl, R_log_std, retain_graph=True, allow_unused=True)[0]
                g_pair_std = (
                    torch.autograd.grad(pairwise_term, R_log_std, retain_graph=True, allow_unused=True)[0]
                    if pairwise_term.requires_grad
                    else None
                )
                g_rank_std = (
                    torch.autograd.grad(rank_term, R_log_std, retain_graph=True, allow_unused=True)[0]
                    if rank_term.requires_grad
                    else None
                )
                g_residual_group_std = (
                    torch.autograd.grad(residual_group_term, R_log_std, retain_graph=True, allow_unused=True)[0]
                    if residual_group_term.requires_grad
                    else None
                )
                g_ortho_std = (
                    torch.autograd.grad(ortho_term, R_log_std, retain_graph=True, allow_unused=True)[0]
                    if ortho_term.requires_grad
                    else None
                )
                data_std = float(torch.linalg.norm(g_data_std).detach().cpu().item()) if g_data_std is not None else 0.0
                anchor_std = float(torch.linalg.norm(g_anchor_std).detach().cpu().item()) if g_anchor_std is not None else 0.0
                kl_std = float(torch.linalg.norm(g_kl_std).detach().cpu().item()) if g_kl_std is not None else 0.0
                pair_std = float(torch.linalg.norm(g_pair_std).detach().cpu().item()) if g_pair_std is not None else 0.0
                rank_std = float(torch.linalg.norm(g_rank_std).detach().cpu().item()) if g_rank_std is not None else 0.0
                residual_group_std = (
                    float(torch.linalg.norm(g_residual_group_std).detach().cpu().item())
                    if g_residual_group_std is not None
                    else 0.0
                )
                ortho_std = float(torch.linalg.norm(g_ortho_std).detach().cpu().item()) if g_ortho_std is not None else 0.0
                std_ratio = data_std / (kl_std + 1e-12)
            else:
                data_std = 0.0
                anchor_std = 0.0
                kl_std = 0.0
                pair_std = 0.0
                rank_std = 0.0
                residual_group_std = 0.0
                ortho_std = 0.0
                std_ratio = float("nan")
            grad_diag = {
                "grad_data_R_mu": data_mu,
                "grad_anchor_pred_R_mu": anchor_mu,
                "grad_pairwise_R_mu": pair_mu,
                "grad_rank_R_mu": rank_mu,
                "grad_residual_group_R_mu": residual_group_mu,
                "grad_basis_scale_l1_R_mu": basis_scale_mu,
                "grad_ortho_R_mu": ortho_mu,
                "grad_scaled_KL_R_mu": kl_mu,
                "grad_data_over_scaled_KL_R_mu": data_mu / (kl_mu + 1e-12),
                "grad_anchor_pred_over_data_R_mu": anchor_mu / (data_mu + 1e-12),
                "grad_anchor_pred_over_scaled_KL_R_mu": anchor_mu / (kl_mu + 1e-12),
                "grad_pairwise_over_data_R_mu": pair_mu / (data_mu + 1e-12),
                "grad_pairwise_over_scaled_KL_R_mu": pair_mu / (kl_mu + 1e-12),
                "grad_rank_over_data_R_mu": rank_mu / (data_mu + 1e-12),
                "grad_rank_over_anchor_pred_R_mu": rank_mu / (anchor_mu + 1e-12),
                "grad_rank_over_scaled_KL_R_mu": rank_mu / (kl_mu + 1e-12),
                "grad_residual_group_over_data_R_mu": residual_group_mu / (data_mu + 1e-12),
                "grad_residual_group_over_scaled_KL_R_mu": residual_group_mu / (kl_mu + 1e-12),
                "grad_ortho_over_data_R_mu": ortho_mu / (data_mu + 1e-12),
                "grad_ortho_over_scaled_KL_R_mu": ortho_mu / (kl_mu + 1e-12),
                "grad_data_R_log_std": data_std,
                "grad_anchor_pred_R_log_std": anchor_std,
                "grad_pairwise_R_log_std": pair_std,
                "grad_rank_R_log_std": rank_std,
                "grad_residual_group_R_log_std": residual_group_std,
                "grad_ortho_R_log_std": ortho_std,
                "grad_scaled_KL_R_log_std": kl_std,
                "grad_data_over_scaled_KL_R_log_std": std_ratio,
                "grad_anchor_pred_over_data_R_log_std": anchor_std / (data_std + 1e-12),
                "grad_pairwise_over_data_R_log_std": pair_std / (data_std + 1e-12),
                "grad_rank_over_anchor_pred_R_log_std": rank_std / (anchor_std + 1e-12),
            }
        loss.backward()
        if float(cfg.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.grad_clip_norm))
        opt.step()
        with torch.no_grad():
            R_log_std.clamp_(-10.0, 2.0)
            if basis_log_scale is not None:
                basis_log_scale.clamp_(float(cfg.basis_scale_log_min), float(cfg.basis_scale_log_max))
            if svd_log_scale is not None:
                svd_log_scale.clamp_(float(cfg.svd_log_scale_min), float(cfg.svd_log_scale_max))

        val = float(loss.detach().cpu().item())
        if torch.isfinite(loss.detach()) and val < best_loss:
            best_loss = val
            best_state = [p.detach().clone() for p in params]
        if should_record:
            rec = {
                "step": float(step),
                "loss": val,
                "data_loss": float(data_loss.detach().cpu().item()),
                "anchor_pred_term": float(anchor_pred_term.detach().cpu().item()),
                "anchor_pred_weight": float(cfg.anchor_pred_weight),
                "pairwise_term": float(pairwise_term.detach().cpu().item()),
                "pairwise_weight": float(cfg.pairwise_weight),
                "rank_term": float(rank_term.detach().cpu().item()),
                "rank_weight": float(cfg.rank_weight),
                "residual_group_term": float(residual_group_term.detach().cpu().item()),
                "residual_group_weight": float(cfg.residual_group_weight),
                "basis_scale_l1_term": float(basis_scale_term.detach().cpu().item()),
                "basis_scale_l1_weight": float(cfg.basis_scale_l1_weight),
                "ortho_term": float(ortho_term.detach().cpu().item()),
                "ortho_weight": float(cfg.ortho_weight),
                "bound": float(bound.detach().cpu().item()),
                "kl_per_anchor": float(kl.detach().cpu().item()),
                "beta": float(beta),
                "R_std_median": float(torch.exp(R_log_std.detach()).median().cpu().item()),
            }
            rec.update({k: float(v.detach().cpu().item()) for k, v in bound_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in anchor_pred_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in pairwise_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in rank_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in residual_group_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in ortho_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in svd_diag.items()})
            if lowrank_basis is not None:
                rec.update(
                    {
                        "basis_scale_mean": float(basis_scale.detach().mean().cpu().item()),
                        "basis_scale_min": float(basis_scale.detach().min().cpu().item()),
                        "basis_scale_max": float(basis_scale.detach().max().cpu().item()),
                    }
                )
            rec.update(grad_diag)
            history.append(rec)
    if best_state is not None:
        with torch.no_grad():
            for p, state in zip(params, best_state):
                p.copy_(state)
    train_sec = time.perf_counter() - t0
    final_svd_diag: dict[str, torch.Tensor] = {}
    if mean_param == "shared_svd":
        R_mu_final, final_svd_diag = _shared_svd_mean(
            svd_left_raw,  # type: ignore[arg-type]
            svd_right_raw,  # type: ignore[arg-type]
            svd_log_scale,  # type: ignore[arg-type]
            svd_left_base,  # type: ignore[arg-type]
            svd_right_base,  # type: ignore[arg-type]
            log_scale_min=float(cfg.svd_log_scale_min),
            log_scale_max=float(cfg.svd_log_scale_max),
        )
    else:
        R_mu_final = R_mu_free  # type: ignore[assignment]
    final_basis_scale = None
    if lowrank_basis is not None:
        if basis_log_scale is not None:
            final_basis_scale = torch.exp(
                basis_log_scale.detach().clamp(
                    min=float(cfg.basis_scale_log_min),
                    max=float(cfg.basis_scale_log_max),
                )
            )
        else:
            final_basis_scale = scale0.detach().clone()
    with torch.no_grad():
        final_ortho, final_ortho_diag = _orthogonality_penalty(
            R_mu_final.detach(),
            target_scale=float(cfg.ortho_target_scale),
            normalize=bool(cfg.ortho_normalize),
        )
    diag = {
        "train_sec": float(train_sec),
        "best_loss": float(best_loss),
        "R_std_median": float(torch.exp(R_log_std.detach()).median().cpu().item()),
        "kl_per_anchor": float(
            (
                _diag_sparse_gp_kl(
                    U,
                    R_mu_final.detach(),
                    R_log_std.detach(),
                    lengthscale=float(cfg.w_lengthscale),
                    signal_var=float(cfg.w_signal_var),
                    jitter=float(cfg.jitter),
                )
                / max(1, T)
            )
            .detach()
            .cpu()
            .item()
        ),
        "w_lengthscale": float(cfg.w_lengthscale),
        "w_signal_var": float(cfg.w_signal_var),
        "include_conditional_residual": bool(cfg.include_conditional_residual),
        "train_log_std": bool(cfg.train_log_std),
        "mean_parameterization": mean_param,
        "svd_log_scale_min": float(cfg.svd_log_scale_min),
        "svd_log_scale_max": float(cfg.svd_log_scale_max),
        "lowrank_residual": bool(lowrank_basis is not None),
        "lowrank_rank": int(lowrank_basis.shape[1]) if lowrank_basis is not None else 0,
        "pairwise_weight": float(cfg.pairwise_weight),
        "pairwise_margin": float(cfg.pairwise_margin),
        "pairwise_normalize": bool(cfg.pairwise_normalize),
        "rank_weight": float(cfg.rank_weight),
        "rank_temperature": float(cfg.rank_temperature),
        "rank_normalize": bool(cfg.rank_normalize),
        "anchor_pred_weight": float(cfg.anchor_pred_weight),
        "anchor_pred_var_floor": float(cfg.anchor_pred_var_floor),
        "residual_group_weight": float(cfg.residual_group_weight),
        "residual_group_eps": float(cfg.residual_group_eps),
        "train_basis_scale": bool(cfg.train_basis_scale),
        "basis_scale_l1_weight": float(cfg.basis_scale_l1_weight),
        "ortho_weight": float(cfg.ortho_weight),
        "ortho_target_scale": float(cfg.ortho_target_scale),
        "ortho_normalize": bool(cfg.ortho_normalize),
        "ortho_penalty": float(final_ortho.detach().cpu().item()),
        "history_tail": history[-5:],
    }
    diag.update({k: float(v.detach().cpu().item()) for k, v in final_ortho_diag.items()})
    diag.update({k: float(v.detach().cpu().item()) for k, v in final_svd_diag.items()})
    if final_basis_scale is not None:
        diag.update(
            {
                "basis_scale_mean": float(final_basis_scale.mean().cpu().item()),
                "basis_scale_min": float(final_basis_scale.min().cpu().item()),
                "basis_scale_max": float(final_basis_scale.max().cpu().item()),
                "basis_scale_values": [float(v) for v in final_basis_scale.cpu().reshape(-1).tolist()],
            }
        )
    if history:
        for key, value in history[-1].items():
            if (
                key.startswith("grad_")
                or key.startswith("pairwise_")
                or key.startswith("rank_")
                or key.startswith("anchor_pred_")
                or key.startswith("residual_group_")
                or key.startswith("basis_scale_")
                or key.startswith("ortho_")
                or key.startswith("svd_")
            ):
                diag[f"{key}_final"] = float(value)
    return UncertainWQRMstepResult(
        R_mu=R_mu_final.detach(),
        R_log_std=R_log_std.detach(),
        inducing_X=U.detach(),
        history=history,
        train_sec=float(train_sec),
        diagnostics=diag,
        basis_scale=None if final_basis_scale is None else final_basis_scale.detach(),
    )


def _linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    frac = (int(step) - 1) / max(1, int(total_steps) - 1)
    return float(start) + float(frac) * (float(end) - float(start))


def _diag_anchor_entropy(W_var: torch.Tensor) -> torch.Tensor:
    var = W_var.clamp_min(1e-12)
    return 0.5 * (torch.log(2.0 * torch.pi * math.e * var)).sum(dim=(-1, -2))


def _sample_anchor_w_reparam(
    W_mu: torch.Tensor,
    W_var: torch.Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    eps = torch.randn(
        (int(n_samples), *tuple(W_mu.shape)),
        device=W_mu.device,
        dtype=W_mu.dtype,
        generator=generator,
    )
    return W_mu.unsqueeze(0) + eps * W_var.clamp_min(1e-12).sqrt().unsqueeze(0)


def _per_anchor_rank_advantage(rewards: torch.Tensor, *, normalize: bool, eps: float) -> torch.Tensor:
    """Rank-based advantage per anchor: ``rank_s - (S+1)/2``; optional per-anchor std scaling."""
    s_count = int(rewards.shape[0])
    if s_count <= 1:
        return rewards.new_zeros(rewards.shape)
    order = torch.argsort(rewards, dim=0)
    ranks = torch.empty_like(rewards)
    rank_vals = torch.arange(1, s_count + 1, device=rewards.device, dtype=rewards.dtype).unsqueeze(1)
    ranks.scatter_(0, order, rank_vals.expand(-1, rewards.shape[1]))
    adv = ranks - (s_count + 1) / 2.0
    if normalize:
        return adv / adv.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))
    return adv


def _mc_reinforce_advantage(rewards: torch.Tensor, cfg: MCCEMQRMstepConfig) -> torch.Tensor:
    """Per-anchor advantage across MC samples (shape ``[S, T]``)."""
    mode = str(cfg.reward_advantage)
    eps = float(cfg.anchor_weight_eps)
    if mode == "center":
        return rewards - rewards.mean(dim=0, keepdim=True)
    if mode == "center_std":
        centered = rewards - rewards.mean(dim=0, keepdim=True)
        return centered / centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    if mode == "rank":
        return _per_anchor_rank_advantage(rewards, normalize=False, eps=eps)
    if mode == "rank_norm":
        return _per_anchor_rank_advantage(rewards, normalize=True, eps=eps)
    raise ValueError(f"Unknown reward_advantage {mode!r}.")


def _mc_anchor_weights(
    rewards: torch.Tensor,
    cfg: MCCEMQRMstepConfig,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Non-negative anchor weights ``w_j`` (shape ``[T]``), sum normalized to ``T``."""
    spread_std = rewards.std(dim=0, unbiased=False)
    spread_range = rewards.max(dim=0).values - rewards.min(dim=0).values
    t_count = int(rewards.shape[1])
    mode = str(cfg.anchor_weight)
    eps = float(cfg.anchor_weight_eps)
    w_max = float(cfg.anchor_weight_max)

    if mode == "uniform":
        w = torch.ones(t_count, device=rewards.device, dtype=rewards.dtype)
    elif mode == "spread_std":
        w = (spread_std / (spread_std.median() + eps)).clamp(0.0, w_max)
    elif mode == "spread_range":
        w = (spread_range / (spread_range.median() + eps)).clamp(0.0, w_max)
    elif mode == "top_fraction":
        k = max(1, int(round(float(cfg.anchor_weight_top_frac) * t_count)))
        thresh = spread_std.topk(k).values.min()
        w = (spread_std >= thresh).to(dtype=rewards.dtype)
    elif mode == "spread_threshold":
        tau = float(cfg.anchor_weight_spread_tau)
        if tau <= 0.0:
            tau = float(spread_std.median().item())
        w = (spread_std > tau).to(dtype=rewards.dtype)
    else:
        raise ValueError(f"Unknown anchor_weight {mode!r}.")

    cur_steps = int(cfg.anchor_weight_curriculum_steps)
    if cur_steps > 0:
        alpha = min(1.0, float(step) / float(cur_steps))
        w = (1.0 - alpha) * w + alpha * torch.ones_like(w)

    w = w * (float(t_count) / w.sum().clamp_min(eps))
    diag = {
        "anchor_spread_std_mean": float(spread_std.mean().item()),
        "anchor_spread_std_median": float(spread_std.median().item()),
        "anchor_spread_range_mean": float(spread_range.mean().item()),
        "anchor_weight_mean": float(w.mean().item()),
        "anchor_weight_max": float(w.max().item()),
        "anchor_weight_active_frac": float((w > 1e-8).to(dtype=rewards.dtype).mean().item()),
        "anchor_weight_curriculum_alpha": float(min(1.0, float(step) / float(cur_steps))) if cur_steps > 0 else 1.0,
    }
    return w, diag


def _anchor_diag_log_prob(W_sample: torch.Tensor, W_mu: torch.Tensor, W_var: torch.Tensor) -> torch.Tensor:
    var = W_var.clamp_min(1e-12).unsqueeze(0)
    diff2 = (W_sample - W_mu.unsqueeze(0)).pow(2)
    return -0.5 * ((diff2 / var) + torch.log(2.0 * torch.pi * var)).sum(dim=(-1, -2))


def train_qr_mc_cem_mstep(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    init_labels: torch.Tensor | np.ndarray,
    *,
    y_anchor: torch.Tensor | np.ndarray | None = None,
    inducing_X: torch.Tensor | np.ndarray,
    init_W: torch.Tensor | np.ndarray | None = None,
    init_R_mu: torch.Tensor | np.ndarray | None = None,
    init_R_log_std: torch.Tensor | np.ndarray | None = None,
    init_lengthscale: float | tuple[float, ...] | torch.Tensor = 1.0,
    init_signal_var: float | torch.Tensor = 1.0,
    init_noise_var: float | torch.Tensor = 0.05,
    gate_init: torch.Tensor | np.ndarray | None = None,
    cem_config: SelfCEMConfig | None = None,
    config: MCCEMQRMstepConfig | None = None,
    eval_callback: Any | None = None,
) -> MCCEMQRMstepResult:
    """Train sparse-GP ``q(R)`` with a Monte-Carlo self-CEM M-step.

    This implementation supports anchor-conditioned ``q(W_j)``.  Each step
    samples ``W_j``, runs local self-CEM on detached samples, and then applies
    either route A (reparameterized frozen-CEM gradient) or route B
    (score-function gradient).
    """
    cfg = config or MCCEMQRMstepConfig()
    cem_cfg = cem_config or SelfCEMConfig()
    Xa = _as_float_tensor(X_anchor)
    device = Xa.device
    dtype = Xa.dtype
    Xn = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    labels0 = _as_float_tensor(init_labels, device=device).to(dtype=dtype)
    ya = None if y_anchor is None else _as_float_tensor(y_anchor, device=device).to(dtype=dtype).reshape(-1)
    U0 = _as_float_tensor(inducing_X, device=device).to(dtype=dtype)
    if y.dim() == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if labels0.dim() == 3 and labels0.shape[-1] == 1:
        labels0 = labels0.squeeze(-1)
    T, n, D = Xn.shape
    M_ind = U0.shape[0]
    base_W = _as_float_tensor(init_W, device=device).to(dtype=dtype) if init_W is not None else None
    if init_R_mu is not None:
        R0 = _as_float_tensor(init_R_mu, device=device).to(dtype=dtype)
        if R0.shape[:1] != (M_ind,):
            raise ValueError("init_R_mu first dimension must match inducing_X.")
    elif base_W is not None:
        if base_W.dim() != 2 or base_W.shape[1] != D:
            raise ValueError(f"init_W must have shape [Q,{D}], got {tuple(base_W.shape)}.")
        R0 = base_W.unsqueeze(0).expand(M_ind, -1, -1).contiguous()
    else:
        raise ValueError("Either init_W or init_R_mu is required.")
    Q = R0.shape[1]
    R_mu = torch.nn.Parameter(R0.detach().clone())
    if init_R_log_std is not None:
        Rls0 = _as_float_tensor(init_R_log_std, device=device).to(dtype=dtype)
        if Rls0.shape != R_mu.shape:
            raise ValueError(
                f"init_R_log_std must match R_mu shape {tuple(R_mu.shape)}, got {tuple(Rls0.shape)}."
            )
        R_log_std = torch.nn.Parameter(Rls0.detach().clone())
    else:
        R_log_std = torch.nn.Parameter(torch.full_like(R_mu, float(cfg.init_log_std)))
    R_log_std.requires_grad_(bool(cfg.train_log_std))
    U_param = torch.nn.Parameter(U0.detach().clone()) if bool(cfg.learn_inducing) else U0
    params = [R_mu]
    if R_log_std.requires_grad:
        params.append(R_log_std)
    opt_groups: list[dict[str, Any]] = [{"params": params, "lr": float(cfg.lr)}]
    if isinstance(U_param, torch.nn.Parameter):
        opt_groups.append({"params": [U_param], "lr": float(cfg.lr) * float(cfg.inducing_lr_mult)})
    opt = torch.optim.Adam(opt_groups)
    history: list[dict[str, float]] = []
    sample_reward_trace: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    best_eval = float("inf")
    best_eval_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    best_eval_step = 0
    no_eval_improve = 0
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(cfg.seed))
    n_samples = max(1, int(cfg.n_samples))
    t0 = time.perf_counter()

    for step in range(1, int(cfg.steps) + 1):
        opt.zero_grad()
        reinforce_diag: dict[str, float] = {}
        U = U_param if isinstance(U_param, torch.Tensor) else U0
        w_moments = sparse_gp_uncertain_w_moments_for_mode(
            mode="anchor",
            X_anchor=Xa,
            X_neighbors=Xn,
            inducing_X=U,
            R_mu=R_mu,
            R_log_std=R_log_std,
            lengthscale=float(cfg.w_lengthscale),
            signal_var=float(cfg.w_signal_var),
            include_conditional_residual=bool(cfg.include_conditional_residual),
            jitter=float(cfg.jitter),
        )
        W_mu = w_moments["W_mu_anchor"]
        W_var = w_moments["W_var_anchor"]
        W_samples = _sample_anchor_w_reparam(W_mu, W_var, n_samples=n_samples, generator=gen)
        W_flat_det = W_samples.detach().reshape(n_samples * T, Q, D).contiguous()
        zero_flat = torch.zeros_like(W_flat_det)
        X_anchor_flat = Xa.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * T, D).contiguous()
        X_neighbors_flat = Xn.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * T, n, D).contiguous()
        y_flat = y.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * T, n).contiguous()
        y_anchor_flat = None
        if ya is not None:
            y_anchor_flat = ya.unsqueeze(0).expand(n_samples, -1).reshape(n_samples * T).contiguous()
        labels_flat = labels0.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * T, n).contiguous()

        ell0 = _as_float_tensor(init_lengthscale, device=device).to(dtype=dtype)
        sig0 = _as_float_tensor(init_signal_var, device=device).to(dtype=dtype)
        noise0 = _as_float_tensor(init_noise_var, device=device).to(dtype=dtype)
        if ell0.numel() > 1:
            ell0 = ell0.reshape(T, -1).unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * T, -1)
        if sig0.numel() > 1:
            sig0 = sig0.reshape(T).unsqueeze(0).expand(n_samples, -1).reshape(n_samples * T)
        if noise0.numel() > 1:
            noise0 = noise0.reshape(T).unsqueeze(0).expand(n_samples, -1).reshape(n_samples * T)
        gate_flat = None
        if gate_init is not None:
            gate_t = _as_float_tensor(gate_init, device=device).to(dtype=dtype)
            gate_flat = gate_t.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * T, gate_t.shape[1]).contiguous()

        cem_t0 = time.perf_counter()
        sample_state = run_uncertain_w_self_cem(
            X_anchor_flat,
            X_neighbors_flat,
            y_flat,
            labels_flat,
            mode="anchor",
            W_mu_anchor=W_flat_det,
            W_var_anchor=zero_flat,
            init_lengthscale=ell0,
            init_signal_var=sig0,
            init_noise_var=noise0,
            gate_init=gate_flat,
            config=cem_cfg,
        )
        cem_sec = time.perf_counter() - cem_t0
        sample_outlier_logp = sample_state.get("outlier_logp") if isinstance(sample_state, dict) else None
        if isinstance(sample_outlier_logp, torch.Tensor) and sample_outlier_logp.numel() == 0:
            sample_outlier_logp = None

        if str(cfg.objective) == "stopgrad_reparam":
            W_flat = W_samples.reshape(n_samples * T, Q, D).contiguous()
            rewards_flat, reward_diag = _hard_cem_bound_per_anchor_from_w_moments(
                X_anchor_flat,
                X_neighbors_flat,
                y_flat,
                sample_state["labels"],  # type: ignore[arg-type]
                mode="anchor",
                w_moments={"W_mu_anchor": W_flat, "W_var_anchor": torch.zeros_like(W_flat)},
                gate=sample_state["gate"],  # type: ignore[arg-type]
                lengthscale=sample_state["lengthscale"],  # type: ignore[arg-type]
                signal_var=sample_state["signal_var"],  # type: ignore[arg-type]
                noise_var=sample_state["noise_var"],  # type: ignore[arg-type]
                mean_const=sample_state["mean_const"],  # type: ignore[arg-type]
                outlier_mode=cfg.outlier_mode,
                outlier_sigma_mult=float(cfg.outlier_sigma_mult),
                background_scale_mult=float(cfg.background_scale_mult),
                outlier_logp=sample_outlier_logp,  # type: ignore[arg-type]
                gh_points=int(cfg.gh_points),
                jitter=float(cfg.jitter),
                min_inliers=int(cfg.min_inliers),
            )
            rewards = rewards_flat.reshape(n_samples, T)
            if str(cfg.weighting) == "softmax":
                tau = max(1e-6, _linear_schedule(cfg.softmax_temperature_start, cfg.softmax_temperature_end, step, int(cfg.steps)))
                centered = rewards.detach() - rewards.detach().mean(dim=0, keepdim=True)
                weights = torch.softmax(centered / tau, dim=0)
                data_objective = (weights * rewards).sum(dim=0).mean()
            else:
                tau = float("nan")
                data_objective = rewards.mean()
        elif str(cfg.objective) == "score_function":
            with torch.no_grad():
                if str(cfg.reward_mode) == "supervised_anchor":
                    if y_anchor_flat is None:
                        raise ValueError("reward_mode='supervised_anchor' requires y_anchor.")
                    rewards_flat, reward_diag = _anchor_predictive_logpdf_per_anchor_from_w_moments(
                        X_anchor_flat,
                        y_anchor_flat,
                        X_neighbors_flat,
                        y_flat,
                        sample_state["labels"],  # type: ignore[arg-type]
                        mode="anchor",
                        w_moments={"W_mu_anchor": W_flat_det, "W_var_anchor": zero_flat},
                        lengthscale=sample_state["lengthscale"],  # type: ignore[arg-type]
                        signal_var=sample_state["signal_var"],  # type: ignore[arg-type]
                        noise_var=sample_state["noise_var"],  # type: ignore[arg-type]
                        mean_const=sample_state["mean_const"],  # type: ignore[arg-type]
                        jitter=float(cfg.jitter),
                        min_inliers=int(cfg.min_inliers),
                        pred_var_floor=float(cfg.anchor_pred_var_floor),
                        exclude_self=bool(cfg.anchor_pred_exclude_self),
                        self_tol=float(cfg.anchor_self_tol),
                    )
                else:
                    rewards_flat, reward_diag = _hard_cem_bound_per_anchor_from_w_moments(
                        X_anchor_flat,
                        X_neighbors_flat,
                        y_flat,
                        sample_state["labels"],  # type: ignore[arg-type]
                        mode="anchor",
                        w_moments={"W_mu_anchor": W_flat_det, "W_var_anchor": zero_flat},
                        gate=sample_state["gate"],  # type: ignore[arg-type]
                        lengthscale=sample_state["lengthscale"],  # type: ignore[arg-type]
                        signal_var=sample_state["signal_var"],  # type: ignore[arg-type]
                        noise_var=sample_state["noise_var"],  # type: ignore[arg-type]
                        mean_const=sample_state["mean_const"],  # type: ignore[arg-type]
                        outlier_mode=cfg.outlier_mode,
                        outlier_sigma_mult=float(cfg.outlier_sigma_mult),
                        background_scale_mult=float(cfg.background_scale_mult),
                        outlier_logp=sample_outlier_logp,  # type: ignore[arg-type]
                        gh_points=int(cfg.gh_points),
                        jitter=float(cfg.jitter),
                        min_inliers=int(cfg.min_inliers),
                    )
            rewards = rewards_flat.reshape(n_samples, T).detach()
            rewards_w0 = None
            if (
                bool(cfg.log_sample_rewards)
                and base_W is not None
                and str(cfg.reward_mode) == "supervised_anchor"
            ):
                W0_flat = base_W.unsqueeze(0).expand(n_samples * T, -1, -1).contiguous()
                with torch.no_grad():
                    rewards_w0_flat, _ = _anchor_predictive_logpdf_per_anchor_from_w_moments(
                        X_anchor_flat,
                        y_anchor_flat,
                        X_neighbors_flat,
                        y_flat,
                        sample_state["labels"],  # type: ignore[arg-type]
                        mode="anchor",
                        w_moments={"W_mu_anchor": W0_flat, "W_var_anchor": zero_flat},
                        lengthscale=sample_state["lengthscale"],  # type: ignore[arg-type]
                        signal_var=sample_state["signal_var"],  # type: ignore[arg-type]
                        noise_var=sample_state["noise_var"],  # type: ignore[arg-type]
                        mean_const=sample_state["mean_const"],  # type: ignore[arg-type]
                        jitter=float(cfg.jitter),
                        min_inliers=int(cfg.min_inliers),
                        pred_var_floor=float(cfg.anchor_pred_var_floor),
                        exclude_self=bool(cfg.anchor_pred_exclude_self),
                        self_tol=float(cfg.anchor_self_tol),
                    )
                rewards_w0 = rewards_w0_flat.reshape(n_samples, T).detach()
            adv = _mc_reinforce_advantage(rewards, cfg)
            anchor_w, reinforce_diag = _mc_anchor_weights(rewards, cfg, step=int(step))
            log_prob = _anchor_diag_log_prob(W_samples.detach(), W_mu, W_var)
            per_term = adv.detach() * log_prob
            data_objective = (per_term * anchor_w.unsqueeze(0)).sum() / anchor_w.sum().clamp_min(1e-12)
            tau = float("nan")
        else:
            raise ValueError(f"Unknown MC-CEM qR objective {cfg.objective!r}.")

        if bool(cfg.log_sample_rewards):
            r_cpu = rewards.detach().cpu()
            spread_std = rewards.std(dim=0, unbiased=False)
            trace_row: dict[str, Any] = {
                "step": int(step),
                "n_samples": int(n_samples),
                "n_anchors": int(T),
                "reward_mean_per_sample": [float(r_cpu[s].mean().item()) for s in range(n_samples)],
                "reward_std_per_sample": [float(r_cpu[s].std(unbiased=False).item()) for s in range(n_samples)],
                "reward_min_per_sample": [float(r_cpu[s].min().item()) for s in range(n_samples)],
                "reward_max_per_sample": [float(r_cpu[s].max().item()) for s in range(n_samples)],
                "reward_per_sample_anchor": r_cpu.numpy().tolist(),
                "reward_spread_mean_over_anchors": float(
                    (r_cpu.max(dim=0).values - r_cpu.min(dim=0).values).mean().item()
                ),
                "reward_spread_max_over_anchors": float(
                    (r_cpu.max(dim=0).values - r_cpu.min(dim=0).values).max().item()
                ),
                "anchor_spread_std": spread_std.cpu().numpy().tolist(),
            }
            if str(cfg.objective) == "score_function":
                a_cpu = adv.detach().cpu()
                trace_row["adv_mean_per_sample"] = [float(a_cpu[s].mean().item()) for s in range(n_samples)]
                trace_row["adv_std_per_sample"] = [float(a_cpu[s].std(unbiased=False).item()) for s in range(n_samples)]
                trace_row["anchor_weights"] = anchor_w.detach().cpu().numpy().tolist()
                k_top = max(1, int(round(float(cfg.anchor_weight_top_frac) * T)))
                _, top_idx = torch.topk(spread_std, k=k_top, largest=True)
                trace_row["top_anchor_indices"] = top_idx.cpu().numpy().astype(int).tolist()
            if rewards_w0 is not None:
                w0_cpu = rewards_w0.cpu()
                delta_cpu = (rewards - rewards_w0).cpu()
                trace_row["reward_w0_per_sample_anchor"] = w0_cpu.numpy().tolist()
                trace_row["delta_reward_per_sample_anchor"] = delta_cpu.numpy().tolist()
                trace_row["reward_w0_mean_per_anchor"] = [float(w0_cpu[:, j].mean().item()) for j in range(T)]
                trace_row["delta_reward_best_per_anchor"] = [
                    float(delta_cpu[:, j].max().item()) for j in range(T)
                ]
            sample_reward_trace.append(trace_row)

        entropy = _diag_anchor_entropy(W_var).mean()
        entropy_beta = _linear_schedule(cfg.entropy_beta_start, cfg.entropy_beta_end, step, int(cfg.steps))
        kl = _diag_sparse_gp_kl(
            U,
            R_mu,
            R_log_std,
            lengthscale=float(cfg.w_lengthscale),
            signal_var=float(cfg.w_signal_var),
            jitter=float(cfg.jitter),
        ) / max(1, T)
        beta_kl = float(cfg.beta_kl) * min(1.0, step / max(1, int(cfg.kl_warmup_steps)))
        ortho_loss = R_mu.new_zeros(())
        ortho_diag: dict[str, torch.Tensor] = {}
        if float(cfg.ortho_weight) != 0.0:
            ortho_loss, ortho_diag = _orthogonality_penalty(
                R_mu,
                target_scale=float(cfg.ortho_target_scale),
                normalize=bool(cfg.ortho_normalize),
            )
        ortho_term = float(cfg.ortho_weight) * ortho_loss
        inducing_l2 = R_mu.new_zeros(())
        if bool(cfg.learn_inducing) and float(cfg.inducing_l2) != 0.0:
            inducing_l2 = (U - U0).pow(2).mean()
        inducing_term = float(cfg.inducing_l2) * inducing_l2
        loss = -data_objective - float(entropy_beta) * entropy + beta_kl * kl + ortho_term + inducing_term
        loss.backward()
        if float(cfg.grad_clip_norm) > 0:
            clip_params = params + ([U_param] if isinstance(U_param, torch.nn.Parameter) else [])
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(cfg.grad_clip_norm))
        opt.step()
        with torch.no_grad():
            R_log_std.clamp_(-10.0, 2.0)
        val = float(loss.detach().cpu().item())
        if torch.isfinite(loss.detach()) and val < best_loss:
            best_loss = val
            best_state = (R_mu.detach().clone(), R_log_std.detach().clone(), U.detach().clone())
        should_record = step == 1 or step % int(cfg.log_interval) == 0 or step == int(cfg.steps)
        should_stop = False
        reinforce_diag_step = reinforce_diag
        if should_record:
            rec = {
                "step": float(step),
                "loss": val,
                "data_objective": float(data_objective.detach().cpu().item()),
                "reward_mean": float(rewards.detach().mean().cpu().item()),
                "reward_std": float(rewards.detach().std(unbiased=False).cpu().item()),
                "entropy_mean": float(entropy.detach().cpu().item()),
                "entropy_beta": float(entropy_beta),
                "kl_per_anchor": float(kl.detach().cpu().item()),
                "beta_kl": float(beta_kl),
                "ortho_term": float(ortho_term.detach().cpu().item()),
                "ortho_weight": float(cfg.ortho_weight),
                "inducing_term": float(inducing_term.detach().cpu().item()),
                "inducing_shift_mean": float(torch.linalg.norm((U - U0).detach(), dim=1).mean().cpu().item()),
                "inducing_shift_max": float(torch.linalg.norm((U - U0).detach(), dim=1).max().cpu().item()),
                "R_std_median": float(torch.exp(R_log_std.detach()).median().cpu().item()),
                "cem_sec": float(cem_sec),
                "softmax_temperature": float(tau),
            }
            rec.update({k: float(v.detach().cpu().item()) for k, v in reward_diag.items()})
            rec.update({k: float(v.detach().cpu().item()) for k, v in ortho_diag.items()})
            rec.update(reinforce_diag_step)
            if str(cfg.objective) == "score_function":
                rec["mc_reward_advantage"] = float(
                    {"center": 0.0, "center_std": 1.0, "rank": 2.0}.get(str(cfg.reward_advantage), -1.0)
                )
                rec["mc_anchor_weight_code"] = float(
                    {
                        "uniform": 0.0,
                        "spread_std": 1.0,
                        "spread_range": 2.0,
                        "top_fraction": 3.0,
                        "spread_threshold": 4.0,
                    }.get(str(cfg.anchor_weight), -1.0)
                )
            if eval_callback is not None:
                try:
                    eval_diag = eval_callback(step, R_mu.detach(), R_log_std.detach(), U.detach())
                except TypeError:
                    eval_diag = eval_callback(step, R_mu.detach(), R_log_std.detach())
                if eval_diag:
                    rec.update({str(k): float(v) for k, v in eval_diag.items()})
                    metric_name = str(cfg.early_stopping_metric)
                    if bool(cfg.early_stopping) and metric_name in eval_diag:
                        metric = float(eval_diag[metric_name])
                        if metric < best_eval - float(cfg.early_stopping_min_delta):
                            best_eval = metric
                            best_eval_step = int(step)
                            best_eval_state = (R_mu.detach().clone(), R_log_std.detach().clone(), U.detach().clone())
                            no_eval_improve = 0
                        else:
                            no_eval_improve += 1
                        rec["best_eval_metric"] = float(best_eval)
                        rec["best_eval_step"] = float(best_eval_step)
                        rec["early_stop_no_improve"] = float(no_eval_improve)
                        if no_eval_improve >= max(1, int(cfg.early_stopping_patience)):
                            should_stop = True
            history.append(rec)
        if should_stop:
            break
    if str(cfg.restore_state) == "final":
        pass
    elif str(cfg.restore_state) == "best_eval" and best_eval_state is not None:
        with torch.no_grad():
            R_mu.copy_(best_eval_state[0])
            R_log_std.copy_(best_eval_state[1])
            if isinstance(U_param, torch.nn.Parameter):
                U_param.copy_(best_eval_state[2])
    elif str(cfg.restore_state) in {"best_loss", "best_eval"} and best_state is not None:
        with torch.no_grad():
            R_mu.copy_(best_state[0])
            R_log_std.copy_(best_state[1])
            if isinstance(U_param, torch.nn.Parameter):
                U_param.copy_(best_state[2])
    train_sec = time.perf_counter() - t0
    U_final = U_param.detach() if isinstance(U_param, torch.Tensor) else U0.detach()
    final_kl = _diag_sparse_gp_kl(
        U_final,
        R_mu.detach(),
        R_log_std.detach(),
        lengthscale=float(cfg.w_lengthscale),
        signal_var=float(cfg.w_signal_var),
        jitter=float(cfg.jitter),
    ) / max(1, T)
    diag = {
        "train_sec": float(train_sec),
        "best_loss": float(best_loss),
        "R_std_median": float(torch.exp(R_log_std.detach()).median().cpu().item()),
        "kl_per_anchor": float(final_kl.detach().cpu().item()),
        "objective": str(cfg.objective),
        "reward_mode": str(cfg.reward_mode),
        "n_samples": int(n_samples),
        "weighting": str(cfg.weighting),
        "standardize_rewards": bool(cfg.standardize_rewards),
        "entropy_beta_start": float(cfg.entropy_beta_start),
        "entropy_beta_end": float(cfg.entropy_beta_end),
        "ortho_weight": float(cfg.ortho_weight),
        "ortho_target_scale": float(cfg.ortho_target_scale),
        "ortho_normalize": bool(cfg.ortho_normalize),
        "learn_inducing": bool(cfg.learn_inducing),
        "inducing_lr_mult": float(cfg.inducing_lr_mult),
        "inducing_l2": float(cfg.inducing_l2),
        "inducing_shift_mean": float(torch.linalg.norm((U_final - U0).detach(), dim=1).mean().cpu().item()),
        "inducing_shift_max": float(torch.linalg.norm((U_final - U0).detach(), dim=1).max().cpu().item()),
        "early_stopping": bool(cfg.early_stopping),
        "early_stopping_metric": str(cfg.early_stopping_metric),
        "restore_state": str(cfg.restore_state),
        "best_eval_metric": float(best_eval),
        "best_eval_step": int(best_eval_step),
        "history_tail": history[-5:],
    }
    return MCCEMQRMstepResult(
        R_mu=R_mu.detach(),
        R_log_std=R_log_std.detach(),
        inducing_X=U_final.detach(),
        history=history,
        sample_reward_trace=sample_reward_trace,
        train_sec=float(train_sec),
        diagnostics=diag,
    )


def _default_gate(T: int, Q: int, cfg: UncertainWJGPConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rho0 = min(max(float(cfg.rho_init), 1e-4), 1.0 - 1e-4)
    intercept = math.log(rho0 / (1.0 - rho0))
    gate = torch.zeros(T, Q + 1, device=device, dtype=dtype)
    gate[:, 0] = intercept
    return gate


def uncertain_w_jgp_vem_predict(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    *,
    config: UncertainWJGPConfig,
    W_mu_anchor: torch.Tensor | np.ndarray | None = None,
    W_var_anchor: torch.Tensor | np.ndarray | None = None,
    W_mu_all: torch.Tensor | np.ndarray | None = None,
    W_cov_all: torch.Tensor | np.ndarray | None = None,
    gate: torch.Tensor | np.ndarray | None = None,
) -> UncertainWJGPResult:
    """Fit a batched uncertain-W JGP-VEM local predictor and predict at anchors.

    Anchor mode expects ``W_mu_anchor``/``W_var_anchor`` with shape ``[T,Q,D]``.
    Pointwise mode expects ``W_mu_all``/``W_cov_all`` for
    ``X_all = [x_anchor, X_neighbors]`` with shapes ``[T,n+1,Q,D]`` and
    ``[T,Q,D,n+1,n+1]``.
    """
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device)
    y_t = _as_float_tensor(y_neighbors, device=device)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if X_anchor_t.dim() != 2 or X_neighbors_t.dim() != 3 or y_t.dim() != 2:
        raise ValueError("Expected X_anchor [T,D], X_neighbors [T,n,D], y_neighbors [T,n].")
    T, n, D = X_neighbors_t.shape
    if X_anchor_t.shape != (T, D) or y_t.shape != (T, n):
        raise ValueError("Anchor/neighborhood/y dimensions are inconsistent.")
    dtype = X_neighbors_t.dtype

    if config.mode == "anchor":
        if W_mu_anchor is None or W_var_anchor is None:
            raise ValueError("Anchor mode requires W_mu_anchor and W_var_anchor.")
        W_mu = _as_float_tensor(W_mu_anchor, device=device).to(dtype=dtype)
        W_var = _as_float_tensor(W_var_anchor, device=device).to(dtype=dtype)
        Q = W_mu.shape[1]
        K = expected_se_kernel_anchor(
            X_neighbors_t,
            X_neighbors_t,
            W_mu,
            W_var,
            lengthscale=config.lengthscale,
            signal_var=config.signal_var,
        )
        k_star = expected_se_kernel_anchor(
            X_anchor_t.unsqueeze(1),
            X_neighbors_t,
            W_mu,
            W_var,
            lengthscale=config.lengthscale,
            signal_var=config.signal_var,
        ).squeeze(1)
        gate_t = _default_gate(T, Q, config, device=device, dtype=dtype) if gate is None else _as_float_tensor(gate, device=device).to(dtype=dtype)
        gate_mean, gate_var = gate_moments_anchor(X_anchor_t, X_neighbors_t, W_mu, W_var, gate_t)
        cov_k = None
        if config.kernel_vector_correction:
            cov_k = kernel_vector_covariance_anchor(
                X_anchor_t,
                X_neighbors_t,
                W_mu,
                W_var,
                k_star,
                lengthscale=config.lengthscale,
                signal_var=config.signal_var,
            )
    elif config.mode == "pointwise":
        if W_mu_all is None or W_cov_all is None:
            raise ValueError("Pointwise mode requires W_mu_all and W_cov_all.")
        X_all = torch.cat([X_anchor_t.unsqueeze(1), X_neighbors_t], dim=1)
        W_mu_p = _as_float_tensor(W_mu_all, device=device).to(dtype=dtype)
        W_cov_p = _as_float_tensor(W_cov_all, device=device).to(dtype=dtype)
        Q = W_mu_p.shape[2]
        K_all = expected_se_kernel_pointwise(
            X_all,
            W_mu_p,
            W_cov_p,
            lengthscale=config.lengthscale,
            signal_var=config.signal_var,
        )
        K = K_all[:, 1:, 1:]
        k_star = K_all[:, 0, 1:]
        gate_t = _default_gate(T, Q, config, device=device, dtype=dtype) if gate is None else _as_float_tensor(gate, device=device).to(dtype=dtype)
        gate_mean, gate_var = gate_moments_pointwise(X_all, W_mu_p, W_cov_p, gate_t)
        cov_k = None
        if config.kernel_vector_correction:
            cov_k = kernel_vector_covariance_pointwise(
                X_all,
                W_mu_p,
                W_cov_p,
                k_star,
                lengthscale=config.lengthscale,
                signal_var=config.signal_var,
            )
    else:
        raise ValueError(f"Unknown config.mode={config.mode!r}.")

    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    K = _symmetrise(K) + float(config.jitter) * eye
    elog_pos, elog_neg, epi = gh_logsigmoid_expectations(
        gate_mean,
        gate_var,
        gh_points=int(config.gh_points),
    )
    gate_log_odds = elog_pos - elog_neg
    out_logp = _outlier_logprob(
        y_t,
        noise_var=float(config.noise_var),
        mode=config.outlier_mode,
        outlier_sigma_mult=float(config.outlier_sigma_mult),
        background_scale_mult=float(config.background_scale_mult),
    )

    rho = torch.full((T, n), float(config.rho_init), device=device, dtype=dtype)
    rho = rho.clamp(float(config.rho_floor), 1.0 - float(config.rho_floor))
    sig2 = torch.as_tensor(float(config.noise_var), device=device, dtype=dtype).reshape(())

    history: list[dict[str, float]] = []
    mean_const = y_t.mean(dim=1)
    for it in range(int(config.vem_iters)):
        denom = rho.sum(dim=1).clamp_min(1e-8)
        mean_const = (rho * y_t).sum(dim=1) / denom
        mean_f, diag_f, _alpha, _L = weighted_gp_train_posterior(
            K,
            y_t,
            rho,
            sig2,
            mean_const,
            rho_floor=float(config.rho_floor),
            jitter=float(config.jitter),
        )
        ell_in = -0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(sig2)
            + ((y_t - mean_f).pow(2) + diag_f) / sig2
        )
        logit = (gate_log_odds + ell_in - out_logp) / max(float(config.temperature), 1e-6)
        rho_new = torch.sigmoid(logit).clamp(float(config.rho_floor), 1.0 - float(config.rho_floor))
        rho = ((1.0 - float(config.damping)) * rho + float(config.damping) * rho_new).clamp(
            float(config.rho_floor),
            1.0 - float(config.rho_floor),
        )
        history.append({
            "iter": float(it + 1),
            "rho_mean": float(rho.mean().detach().cpu().item()),
            "rho_entropy": float((-(rho * rho.log() + (1.0 - rho) * (1.0 - rho).log())).mean().detach().cpu().item()),
            "mean_train_var": float(diag_f.mean().detach().cpu().item()),
        })

    denom = rho.sum(dim=1).clamp_min(1e-8)
    mean_const = (rho * y_t).sum(dim=1) / denom
    _mean_f, _diag_f, alpha, L = weighted_gp_train_posterior(
        K,
        y_t,
        rho,
        sig2,
        mean_const,
        rho_floor=float(config.rho_floor),
        jitter=float(config.jitter),
    )
    mu = mean_const + (k_star * alpha).sum(dim=-1)
    v = torch.cholesky_solve(k_star.unsqueeze(-1), L).squeeze(-1)
    kss = torch.full((T,), float(config.signal_var), device=device, dtype=dtype)
    latent_var = (kss - (k_star * v).sum(dim=-1)).clamp_min(0.0)

    correction = torch.zeros_like(latent_var)
    if cov_k is not None and float(config.correction_weight) != 0.0:
        correction = torch.einsum("ti,tij,tj->t", alpha, cov_k, alpha).clamp_min(0.0)
    var = (latent_var + sig2 + float(config.correction_weight) * correction).clamp_min(1e-12)
    sigma = var.sqrt()
    diagnostics = {
        "mode": config.mode,
        "rho_mean": float(rho.mean().detach().cpu().item()),
        "rho_min": float(rho.min().detach().cpu().item()),
        "rho_max": float(rho.max().detach().cpu().item()),
        "rho_entropy": float((-(rho * rho.log() + (1.0 - rho) * (1.0 - rho).log())).mean().detach().cpu().item()),
        "gate_pi_mean": float(epi.mean().detach().cpu().item()),
        "mean_kernel_diag": float(torch.diagonal(K, dim1=-2, dim2=-1).mean().detach().cpu().item()),
        "mean_kernel_offdiag": float(((K.sum(dim=(-1, -2)) - torch.diagonal(K, dim1=-2, dim2=-1).sum(dim=-1)) / max(1, n * (n - 1))).mean().detach().cpu().item()),
        "mean_latent_var": float(latent_var.mean().detach().cpu().item()),
        "mean_kernel_vector_correction": float(correction.mean().detach().cpu().item()),
        "history": history,
    }
    return UncertainWJGPResult(
        mu=mu,
        var=var,
        sigma=sigma,
        rho=rho,
        mean_const=mean_const,
        correction=correction,
        latent_var=latent_var,
        diagnostics=diagnostics,
    )


def rbf_kernel(X1: torch.Tensor, X2: torch.Tensor, *, lengthscale: float, signal_var: float = 1.0) -> torch.Tensor:
    """RBF kernel for sparse W-field helper."""
    diff = X1.unsqueeze(-2) - X2.unsqueeze(-3)
    d2 = diff.pow(2).sum(dim=-1)
    return float(signal_var) * torch.exp(-0.5 * d2 / max(float(lengthscale) ** 2, 1e-12))


def sparse_gp_w_moments_diag(
    X_query: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    R_mu: torch.Tensor | np.ndarray,
    R_var: torch.Tensor | np.ndarray,
    *,
    lengthscale: float = 1.0,
    signal_var: float = 1.0,
    include_conditional_residual: bool = True,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginal ``q(W(x))`` moments from diagonal sparse-GP inducing ``q(R)``.

    Parameters
    ----------
    X_query:
        ``[T,m,Dx]`` query positions.  Use ``m=1`` for anchor-conditioned W and
        ``m=n+1`` for pointwise W at ``[anchor, neighbors]``.
    inducing_X:
        ``[M,Dx]`` inducing locations.
    R_mu, R_var:
        ``[M,Q,D]`` inducing mean and diagonal variance for each W entry.

    Returns
    -------
    W_mu:
        ``[T,m,Q,D]``.
    W_cov:
        ``[T,Q,D,m,m]`` covariance across query positions for each independent
        ``(q,d)`` entry.
    """
    Xq = _as_float_tensor(X_query)
    device = Xq.device
    U = _as_float_tensor(inducing_X, device=device).to(dtype=Xq.dtype)
    Rm = _as_float_tensor(R_mu, device=device).to(dtype=Xq.dtype)
    Rv = _as_float_tensor(R_var, device=device).to(dtype=Xq.dtype).clamp_min(0.0)
    if Xq.dim() != 3 or U.dim() != 2 or Rm.dim() != 3 or Rv.shape != Rm.shape:
        raise ValueError("Expected X_query [T,m,Dx], inducing_X [M,Dx], R_mu/R_var [M,Q,D].")
    T, m, Dx = Xq.shape
    M, Q, D = Rm.shape
    if U.shape != (M, Dx):
        raise ValueError("inducing_X dimensions are inconsistent with X_query/R_mu.")

    Kuu = rbf_kernel(U, U, lengthscale=lengthscale, signal_var=signal_var)
    Kuu = Kuu + float(jitter) * torch.eye(M, device=device, dtype=Xq.dtype)
    Luu = _cholesky_psd(Kuu.unsqueeze(0), jitter=jitter).squeeze(0)
    Xflat = Xq.reshape(T * m, Dx)
    KxU = rbf_kernel(Xflat, U, lengthscale=lengthscale, signal_var=signal_var)
    A = torch.cholesky_solve(KxU.T, Luu).T  # [T*m,M]
    W_mu = torch.einsum("pm,mqd->pqd", A, Rm).reshape(T, m, Q, D)

    A_b = A.reshape(T, m, M)
    Kxx = rbf_kernel(Xflat, Xflat, lengthscale=lengthscale, signal_var=signal_var)
    Kxx = Kxx.reshape(T, m, T, m)
    # We only need within-local covariance blocks.
    Kxx_local = torch.stack([Kxx[t, :, t, :] for t in range(T)], dim=0)
    Kux = KxU.reshape(T, m, M)
    cond = Kxx_local - torch.einsum("tia,ab,tjb->tij", Kux, torch.cholesky_inverse(Luu), Kux)
    cond = _symmetrise(cond) if include_conditional_residual else torch.zeros_like(Kxx_local)
    cov_qd = torch.einsum("tim,mqd,tjm->tqdij", A_b, Rv, A_b)
    W_cov = cov_qd + cond.unsqueeze(1).unsqueeze(1)
    return W_mu, _symmetrise(W_cov)


__all__ = [
    "FixedLabelHyperoptConfig",
    "FixedLabelGateConfig",
    "MCCEMQRMstepConfig",
    "MCCEMQRMstepResult",
    "SelfCEMConfig",
    "SelfVEMConfig",
    "UncertainWQRMstepConfig",
    "UncertainWQRMstepResult",
    "UncertainWJGPConfig",
    "UncertainWJGPResult",
    "UncertainWCEMResult",
    "expected_se_kernel_anchor",
    "expected_se_kernel_anchor_fullcov",
    "expected_se_kernel_pointwise",
    "fit_uncertain_kernel_hyperparams_from_labels",
    "fit_uncertain_gate_from_labels",
    "gate_moments_anchor",
    "gate_moments_anchor_fullcov",
    "gate_moments_pointwise",
    "gh_logsigmoid_expectations",
    "kernel_vector_covariance_anchor",
    "kernel_vector_covariance_anchor_fullcov",
    "kernel_vector_covariance_pointwise",
    "pointwise_projected_moments",
    "sparse_gp_w_moments_diag",
    "sparse_gp_lowrank_residual_w_moments_for_mode",
    "sparse_gp_uncertain_w_moments_for_mode",
    "train_qr_mc_cem_mstep",
    "train_qr_uncertain_w_cem_mstep",
    "run_uncertain_w_self_cem",
    "run_uncertain_w_self_vem",
    "uncertain_w_cem_update_labels",
    "uncertain_w_vem_update_responsibilities",
    "uncertain_w_cem_predict_from_labels",
    "uncertain_w_jgp_vem_predict",
    "weighted_gp_train_posterior",
]
