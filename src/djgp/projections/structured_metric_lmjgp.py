"""Structured-metric LMJGP training for CEM-style composite likelihood.

This module keeps the local JumpGP/CEM head from ``uncertain_w_jgp`` but replaces
the free ``Q x D`` sparse-GP projection family with a structured sparse GP:

    R_l in R^Q at inducing inputs,     eta_j | R ~ GP conditioning,
    W_j = diag(exp(eta_j / 2)) U.T,    U.T @ U = I.

The training loop is stochastic over anchors and uses a detached self-CEM state
for each sampled W, then differentiates the frozen CEM likelihood through the
sampled structured metric.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import torch

from djgp.projections.uncertain_w_jgp import (
    SelfCEMConfig,
    _as_float_tensor,
    _diag_sparse_gp_kl,
    _hard_cem_bound_per_anchor_from_w_moments,
    gh_logsigmoid_expectations,
    run_uncertain_w_self_cem,
    sparse_gp_w_moments_diag,
)


EvalCallback = Callable[[int, dict[str, torch.Tensor], dict[str, float]], dict[str, float]]
AnalyticEvalCallback = Callable[
    [int, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]],
    dict[str, float],
]


@dataclass(frozen=True)
class StructuredMetricLMJGPConfig:
    """Configuration for the structured metric composite-likelihood M-step."""

    steps: int = 20
    lr: float = 0.01
    batch_anchors: int = 40
    n_samples: int = 2
    beta_kl: float = 0.02
    kl_warmup_steps: int = 10
    eta_prior_std: float = 1.0
    eta_init_log_std: float = -3.0
    train_eta_log_std: bool = True
    n_inducing: int | None = None
    w_lengthscale: float | None = None
    w_signal_var: float = 1.0
    include_conditional_residual: bool = True
    trace_target: float | None = None
    trace_normalize: bool = True
    trace_penalty_weight: float = 0.01
    outlier_mode: str = "background_normal"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    gh_points: int = 20
    jitter: float = 1e-5
    min_inliers: int = 2
    grad_clip_norm: float = 10.0
    log_interval: int = 1
    eval_interval: int = 1
    restore_state: Literal["final", "best_loss", "best_direct"] = "final"
    seed: int = 0


@dataclass
class StructuredMetricLMJGPResult:
    """Structured metric training output."""

    U: torch.Tensor
    R_mu: torch.Tensor
    R_log_std: torch.Tensor
    inducing_X: torch.Tensor
    eta_mu: torch.Tensor
    eta_log_std: torch.Tensor
    w_lengthscale: float
    w_signal_var: float
    include_conditional_residual: bool
    history: list[dict[str, float]]
    train_sec: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AnalyticStructuredMetricLMJGPConfig:
    """Configuration for analytic ISM-LMJGP with marginalised W."""

    steps: int = 40
    lr: float = 0.01
    batch_anchors: int = 40
    beta_kl_R: float = 0.02
    beta_kl_u: float = 1.0
    kl_warmup_steps: int = 10
    eta_prior_std: float = 1.0
    eta_init_log_std: float = -3.0
    train_eta_log_std: bool = True
    # Floor on the q(eta) inducing log-std. The data likelihood drives q(W) to a
    # near point-mass (eta_std collapses to ~0.02), which removes the metric-scale
    # uncertainty from the predictive. A floor keeps a minimum q(W) spread so the
    # predictive variance reflects residual metric uncertainty. Default -8 = off.
    r_log_std_min: float = -8.0
    n_inducing_R: int | None = None
    w_lengthscale: float | None = None
    w_signal_var: float = 1.0
    include_conditional_residual: bool = True
    local_signal_var: float = 1.0
    # The observation noise used to be pinned tightly at 0.1. The local GP overfits
    # the neighbour responses, so the data-optimal noise collapses to ~0.1 and the
    # inlier predictive (anchor not in the training neighbours) is far too narrow.
    # Anchor the noise at a calibration-sensible ~0.25 so test-point intervals match
    # the residual scale; the prior is the effective noise floor here.
    obs_noise_init: float = 0.25
    obs_noise_prior_std: float = 0.3
    obs_noise_prior_weight: float = 3.0
    # Observation-noise prior mode. "fixed" anchors every anchor's noise prior at
    # obs_noise_init (the original behaviour: the noise pins at ~0.25 and the
    # analytic-direct predictive uniformly undercovers on UCI, see
    # docs/ism_lmjgp_analytic.md 8c-calib). "data_driven" instead sets a
    # PER-ANCHOR (heteroscedastic) prior mean from a nearest-neighbour-difference
    # (Rice/Gasser) estimate of the local residual scale in the W0-projected
    # neighbour space, so the learnable observation noise is pulled toward the
    # genuine out-of-sample residual scale. The log-Gaussian prior keeps it
    # Bayesian and ``obs_noise_max`` caps it so it cannot blow up.
    obs_noise_prior_mode: Literal["fixed", "data_driven"] = "fixed"
    obs_noise_data_driven_frac: float = 1.0
    obs_noise_data_driven_floor: float = 0.1
    # ABLATION (default OFF). Use a robust (median) nearest-neighbour-difference Rice
    # noise estimate instead of the mean, so cross-region jump-crossing neighbour pairs
    # do not inflate the observation-noise prior (the cause of over-wide intervals /
    # overcoverage on mild jump data). Pairs the gate-aware inlier noise with a
    # learnable outlier that fits the cross-region points itself.
    obs_noise_robust: bool = False
    obs_noise_max: float = 1.5
    init_noise_at_prior: bool = True
    rho_init: float = 0.8
    rho_prior_strength: float = 2.0
    min_rho_mean: float = 0.55
    min_rho_penalty_weight: float = 50.0
    # Gate mode for the inlier probability rho.
    #   "intercept" (default): rho_t = sigmoid(rho_logit_t) -- one learnable scalar
    #     per anchor, NOT a function of x (the documented intercept-only gate).
    #   "spatial": a JumpGP-style boundary rho_{t,i} = sigmoid(nu_0_t + w_t^T z_{t,i})
    #     where z_{t,i} = W(x_i - x_anchor) is the projected neighbour displacement,
    #     RANDOM under q(W). The mixture data term remains a valid ELBO via the
    #     collapsed-responsibility logsumexp with log(rho)/log(1-rho) replaced by the
    #     Gauss-Hermite expectations E_q[log sigmoid(+/-g)]. Here ``rho_logit`` is
    #     reused as the per-anchor intercept nu_0 and ``gate_w`` holds the slopes.
    gate_mode: Literal["intercept", "spatial"] = "intercept"
    gate_gh_points: int = 20
    gate_w_l2: float = 1e-3
    # 4A (default OFF). Collapse the local inducing values u ANALYTICALLY instead of
    # learning (mu_u, u_log_std) by Adam. Given detached per-neighbour inlier
    # responsibilities r (a persistent buffer, EM-style), the r-weighted Gaussian
    # data term is conjugate in u, so the optimal full-covariance q*(u) has the
    # closed form (Titsias / Bayesian-GPLVM with the existing Psi1/Psi2 statistics)
    #     Sigma* = K B^{-1} K,   mu* = K B^{-1} (Psi1^T (r . y) / sigma_n^2 + c 1),
    #     B = Psi2_r / sigma_n^2 + K,   Psi2_r = sum_i r_i Psi2_i.
    # q*(u) is a deterministic differentiable function of (W, sigma_n), so the ELBO
    # gradient w.r.t. W/gate/noise includes the ENVELOPE term through mu*(W), Sigma*(W)
    # -- the "stale mu_u" / two-timescale problem is removed at the root. The mixture
    # logsumexp data term is unchanged (still a valid bound for ANY q(u); q* is just a
    # near-optimal choice), and the responsibilities that weight q*(u) are refreshed
    # each step from the posterior branch responsibilities (detached, damped).
    collapse_u: bool = False
    # ABLATION (default OFF). Optimize the inducing-point LOCATIONS jointly with the rest,
    # instead of keeping them fixed at their data-derived init.
    #   learn_inducing_R:      the GLOBAL W-field sparse-GP inducing inputs inducing_X
    #                          [M, D] (count = n_inducing_R). Fixed init = _select_inducing_X.
    #   learn_local_inducing:  the LOCAL per-anchor GP inducing locations C [T, m, Q]
    #                          (count = m_inducing), in the projected z-space. Only takes
    #                          effect when reproject_local_inducing is False (otherwise C is
    #                          re-derived from the current W each step and has no free params).
    learn_inducing_R: bool = False
    learn_local_inducing: bool = False
    # Damping of the responsibility-buffer update r <- (1-d) r + d r_new. Buffer inits
    # at 1 (pure-inlier warmup, per docs/lmjgp_elbo_diagnostics_summary.md).
    collapse_u_resp_damping: float = 0.5
    # Tempered responsibilities for the q*(u) weights: r_used = (1-b) * 1 + b * r_buf.
    # b=1 -> full mixture responsibilities; b=0.5 -> the "alpha -> 0.5" blend that the
    # ELBO diagnostics found avoids selective signal deletion.
    collapse_u_resp_blend: float = 1.0
    # Gate GP prior (default OFF; spatial mode only). Replace the FREE per-anchor gate
    # slopes omega_t [T,Q] and intercept nu0_t [T] with smooth fields read out from a
    # sparse GP over anchor locations (kriging readout on the SAME inducing inputs as
    # the eta metric GP): omega(x) = K_xz Kzz^{-1} V_omega, nu0(x) = logit(rho_init) +
    # K_xz Kzz^{-1} v_nu0, with a GP quadratic prior 0.5 v^T (var * Kzz)^{-1} v on the
    # inducing values. Nearby anchors then SHARE boundary evidence instead of each
    # re-learning a Q-dim direction from ~n neighbours (the reason spatial ~ intercept).
    gate_gp: bool = False
    gate_gp_lengthscale: float | None = None  # None -> the eta GP's w_lengthscale
    gate_gp_slope_var: float = 4.0
    gate_gp_nu0_var: float = 4.0
    gate_gp_prior_weight: float = 1.0
    # POST-TRAINING (default OFF). Freeze the global W field (U/R for structured,
    # V for free) at its (warm-started) init and train ONLY the per-anchor local
    # layers (mu_u / noise / gate). Used for append-only test points: the W field
    # learned on the original run is reused as-is.
    freeze_w_field: bool = False
    # W family (default "structured" = diag(exp(eta/2)) U^T with a GP on the Q
    # log-scales). "free": the ORIGINAL entrywise Q x D sparse-GP projection family
    # (variational.py / uncertain_w_jgp), in DEVIATION form W(x) = W0 + G(x) with
    # G ~ independent per-entry GP(0, k) over anchor locations (inducing values on
    # the same inducing_X; w_lengthscale / w_signal_var reused). The KL shrinks
    # W -> W0 rather than -> 0, so no capacity is wasted smoothing the rotation
    # gauge. Motivation: with collapse_u the W gradient carries the envelope term,
    # so the richer family may be learnable where joint Adam previously could not.
    w_family: Literal["structured", "free"] = "structured"
    outlier_scale_mult: float = 3.0
    trace_target: float | None = None
    trace_normalize: bool = True
    trace_penalty_weight: float = 0.01
    grad_clip_norm: float = 10.0
    log_interval: int = 1
    eval_interval: int = 5
    restore_state: Literal["final", "best_loss", "best_direct"] = "final"
    seed: int = 0
    jitter: float = 1e-5
    # ABLATION (default OFF = original behaviour). When True, the local inducing
    # inputs are NOT frozen at the W0 projection; instead their raw input-space
    # locations are re-projected by the *current* metric W (detached) at every
    # step and at predict time, so they track W's rotation/scaling and stay inside
    # the current projected neighbourhood. Requires ``X_inducing_raw`` to be passed
    # to ``train_analytic_structured_metric_lmjgp``.
    reproject_local_inducing: bool = False
    # ABLATION (default "zero" = original behaviour). Prior mean of the log-metric
    # GP on eta.
    #   "zero": p(R) = N(0, Kuu); the prior shrinks eta -> 0, i.e. an isotropic
    #           (Lambda = I) metric on the U subspace. R_mu inits at eta0 (W0's
    #           log-scales) and the KL erodes that anisotropy.
    #   "init": prior mean = eta0 (the W0 log-row-power). Implemented as a fixed
    #           offset eta = eta0 + g(x), g ~ GP(0, Kuu): the zero-mean machinery
    #           is reused on the deviation g (R inits at 0), so the prior shrinks
    #           eta -> eta0 (keep W0's anisotropy) instead of -> isotropic. A
    #           consistent non-zero-mean GP, not a KL-only hack.
    eta_prior_mean_mode: Literal["zero", "init"] = "zero"
    # ABLATION (default OFF). Data-driven warm start for the spatial gate + local
    # inducing values: per anchor, roughly split the neighbours into two y-clusters,
    # init the gate slope toward the anchor's cluster (anchor forced inlier) and init
    # mu_u to the anchor's region y-level. Breaks the all-inlier fixed point that
    # otherwise leaves the spatial gate at chance. Only active when gate_mode=="spatial".
    gate_warm_start: bool = False
    # ABLATION (default OFF). Make the local outlier component's mean + variance
    # learnable per anchor (instead of a fixed broad data-statistic background), so
    # the ELBO can actually "explain" cross-region neighbours by an outlier Gaussian
    # that sits on the other cluster -- giving the gate a reason to separate. The
    # variance is floored at the inlier noise to prevent collapse onto inliers.
    learn_outlier: bool = False
    outlier_var_prior_weight: float = 0.1  # pull out_log_var toward broad data default
    # ABLATION (default OFF). Freeze the spatial gate (gate_w, rho_logit) at its
    # warm-start init -- it is not optimized. Isolates "how good is the init gate if
    # training cannot pull it back to all-inlier?".
    freeze_gate: bool = False
    # ABLATION (default OFF). Freeze ONLY the gate slope omega (gate_w) at its
    # warm-start init, but keep the per-anchor intercept nu0 (rho_logit) learnable.
    # Rationale: omega's direction is the part that collapses to noise; fixing it
    # removes that failure mode, while the shared metric W (still learned) adapts the
    # effective boundary W^T omega and nu0 is an easy per-anchor scalar to learn.
    gate_fix_omega: bool = False
    # ABLATION (default "zero"). Prior mean of the local GP inducing values:
    #   "zero":   p(u)=N(0,Kuu); shrinks the local prediction toward 0 (global mean).
    #   "inlier": p(u)=N(c,Kuu) with c = anchor's nearest-neighbour y level; shrinks
    #             toward the region level instead of 0 (fixes the zero-mean shrinkage).
    #             The conditional/predictive mean uses the matching non-zero-mean GP
    #             form c + Kfu Kuu^{-1}(mu_u - c 1).
    #   "profile": the local GP gets a FREE per-anchor constant mean mu_j (an intercept
    #             that the KL does NOT penalise), profiled out in closed form from the
    #             responsibility-weighted quadratic term:
    #                 mu_j = sum_i rho_i (y_i - zeta_i) / sum_i rho_i,
    #             with zeta_i = E_q[f_i - mu_j] = Kfu Kuu^{-1} mu_u the residual mean and
    #             rho_i the (detached) inlier responsibility. In BOTH gate modes rho does
    #             not depend on the local likelihood (intercept: a per-anchor constant;
    #             spatial: a function of the gate only), so this is a non-circular exact
    #             coordinate maximum, not an EM iterate. By the envelope theorem the
    #             gradient w.r.t. the other parameters is unaffected by the mu_j path.
    #             Model: f_j = mu_j + GP(0, Kuu); the KL still shrinks mu_u -> 0, so the
    #             region LEVEL is carried by the unpenalised intercept instead of by
    #             penalised inducing values (that is the whole point).
    local_mean_mode: Literal["zero", "inlier", "profile"] = "zero"
    # ABLATION (default 1.0/1.0 = off). Deterministic-annealing temperature on the
    # inlier/outlier logsumexp: data term = T*logsumexp([a,b]/T). T is annealed
    # geometrically from gate_temp_init to gate_temp_final over training; T<1 sharpens
    # toward hard CEM (excludes low-responsibility cross-region points from the GP fit).
    gate_temp_init: float = 1.0
    gate_temp_final: float = 1.0
    # ADAPTIVE per-anchor annealing (default OFF). When on, each anchor's temperature is
    # set from the entropy of its T=1 posterior inlier/outlier responsibilities: a CLEAN
    # neighbourhood (polarised responsibilities, low entropy) is hardened toward
    # gate_temp_min (commit -> sharp CEM); an AMBIGUOUS/multi-region neighbourhood (diffuse
    # responsibilities, high entropy) stays soft (T=1, honest wide mixture). This keeps the
    # easy/medium gains of annealing without the hard-boundary damage of a global schedule.
    gate_temp_adaptive: bool = False
    gate_temp_min: float = 0.3
    # Adaptive-temperature signal: "entropy" (T=1 responsibility entropy; circular when the
    # gate is poor) or "bimodality" (Otsu separability of neighbour responses; gate-
    # independent -> hardens only where a real jump exists, stays soft on smooth/multi-region).
    gate_temp_adaptive_signal: Literal["entropy", "bimodality"] = "entropy"
    # EM-style gate supervision (default 0 = off). Add a BCE loss pushing the spatial gate
    # logit g(z_i) to fit the LIKELIHOOD-ONLY responsibility r_i = sigma(loglik_in - loglik_out)
    # (detached). This gives the gate a strong, direct learning signal -- the explicit M-step
    # for the mixing weights -- instead of relying on the weak ELBO gradient through omega.
    # Best paired with a constrained local GP (small m_inducing) so r_i is discriminative.
    gate_supervise_weight: float = 0.0
    # PRINCIPLED MODEL CHANGE (default OFF). Replace the flat single-Gaussian outlier
    # with a SECOND local-GP expert B (own inducing values, second moment, noise),
    # sharing the inducing locations C and the W-marginalised Psi statistics with the
    # inlier expert A. Adds KL(q(u_B)||p(u_B)). The gate becomes the soft A-vs-B
    # boundary; predict marginalises the anchor's A/B membership.
    two_expert: bool = False
    # ABLATION #2 (default OFF). Enforce the spatial gate intercept nu_0 >= 0 so the
    # anchor (test point, at z=0) is always on the inlier side: g(anchor)=nu_0>=0. The
    # variational analog of JumpGP's hard LinearConstraint pinning the test point inlier;
    # it gives the half-space gate a fixed inlier anchor (resolves the side-sign ambiguity
    # that otherwise leaves spatial ~ intercept). Implemented as nu_0 = softplus(rho_logit)
    # in the gate, only when gate_mode=="spatial".
    gate_nu0_nonneg: bool = False
    # ABLATION #4 (default OFF). Replace the broad-Gaussian local outlier with a
    # JumpGP-style FIXED density floor tied to the inlier noise sigma: every neighbour's
    # outlier log-density = log N(k*sigma; 0, sigma) = -0.5 log(2 pi sigma^2) - 0.5 k^2,
    # a per-anchor constant independent of y_i. A neighbour is inlier iff its standardized
    # residual < k. Unlike the broad Gaussian (whose density a smoothed GP can beat even
    # on cross-region points) this is a crisp rejection threshold; unlike a second expert
    # it cannot collapse / over-explain a clean region.
    outlier_floor: bool = False
    outlier_floor_k: float = 2.5
    # Paper's flat uniform outlier density 1/u_j (u_j=range(y) floored). Constant-in-y.
    uniform_outlier: bool = False
    # ABLATION #1 (default OFF). Decouple the local-GP lengthscale from the shared metric
    # W and floor it so the local GP CANNOT shrink its lengthscale to bend across a jump.
    # W still defines the gate geometry/anisotropy (unchanged); the local kernel uses
    # W / ell_loc with a per-anchor learnable ell_loc whose FLOOR is driven by the
    # (gate-independent) Otsu separability of the neighbour responses:
    #   ell_min_t = ell_lo + (ell_hi - ell_lo) * otsu_t ;  ell_loc_t = ell_min_t + softplus(raw_t)
    # A jump neighbourhood (high Otsu) gets a high floor -> forced-smooth GP -> large
    # cross-region residuals -> the outlier floor rejects them -> the gate separates. A
    # smooth/wiggly neighbourhood (low Otsu) gets a low floor -> the GP can still use a
    # short lengthscale for within-region detail (no underfit, unlike cutting m_inducing).
    # Requires the transductive setup (anchors == test points) so ell_loc aligns by index
    # at predict time.
    local_ell_floor: bool = False
    local_ell_lo: float = 1.0
    local_ell_hi: float = 3.0


@dataclass
class AnalyticStructuredMetricLMJGPResult:
    """Analytic structured metric training output."""

    U: torch.Tensor
    R_mu: torch.Tensor
    R_log_std: torch.Tensor
    inducing_X: torch.Tensor
    mu_u: torch.Tensor
    u_log_std: torch.Tensor
    log_noise_var: torch.Tensor
    rho_logit: torch.Tensor
    C: torch.Tensor
    w_lengthscale: float
    w_signal_var: float
    local_signal_var: float
    include_conditional_residual: bool
    history: list[dict[str, float]]
    train_sec: float
    diagnostics: dict[str, Any]
    gate_mode: str = "intercept"
    gate_w: torch.Tensor | None = None
    # Raw input-space local inducing locations [T, m, D]; only set (and used by
    # predict) when ``reproject_local_inducing`` is True.
    X_inducing_raw: torch.Tensor | None = None
    reproject_local_inducing: bool = False
    # Constant per-coordinate prior-mean offset added to eta(x) [Q]; None when
    # eta_prior_mean_mode == "zero".
    eta_mean_offset: torch.Tensor | None = None
    # Learnable per-anchor outlier mean / log-var [T]; None when learn_outlier is off.
    out_mean: torch.Tensor | None = None
    out_log_var: torch.Tensor | None = None
    # Second-expert (B) local-GP parameters [T,m]/[T,m]/[T]; None unless two_expert.
    mu_u_B: torch.Tensor | None = None
    u_log_std_B: torch.Tensor | None = None
    log_noise_var_B: torch.Tensor | None = None
    # Per-anchor local-GP lengthscale [T]; only set when local_ell_floor is on. Applied
    # at predict by scaling the local kernel's W / C by 1 / local_ell (anchors aligned
    # by index in the transductive setup).
    local_ell: torch.Tensor | None = None
    # Full-covariance q(u) [T,m,m]; only set when collapse_u is on (the closed-form
    # Sigma* is full, not diagonal). predict uses it for the exact second moment.
    Sigma_u: torch.Tensor | None = None
    # Direct per-anchor W moments [T,Q,D] / [T,Q,D,D]; only set for w_family="free"
    # (transductive: anchors == the predict anchors). predict uses them verbatim
    # instead of the structured eta-GP readout.
    W_mu_direct: torch.Tensor | None = None
    W_cov_direct: torch.Tensor | None = None
    # Free-family GP state (deviation inducing values + base W0) so W can be read
    # out at ARBITRARY query points (e.g. validation anchors); None for structured.
    V_mu: torch.Tensor | None = None
    V_log_std: torch.Tensor | None = None
    W0_free: torch.Tensor | None = None
    # Constant per-anchor local mean of f [T]; set when local_mean_mode != "zero".
    # predict adds it to the residual GP mean; it does NOT enter the variance.
    # The mode matters for how mu_u relates to it: under "inlier" mu_u is the inducing
    # mean of f itself (deviation = mu_u - c), under "profile" mu_u is already the
    # zero-mean residual's inducing mean (deviation = mu_u).
    local_mean_c: torch.Tensor | None = None
    local_mean_mode: str = "zero"


def _repeat_first(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.unsqueeze(0).expand(int(n), *tuple(x.shape)).reshape(
        int(n) * x.shape[0], *tuple(x.shape[1:])
    ).contiguous()


def _orthonormal_columns(raw: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(raw, mode="reduced")
    sign = torch.sign(torch.diagonal(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.view(1, -1)


def init_structured_metric_from_w0(
    W0: torch.Tensor | np.ndarray,
    *,
    n_anchors: int,
    trace_target: float | None = None,
    eta_init_log_std: float = -3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize ``U`` and per-anchor ``q(eta)`` from a fixed projection W0."""
    W0_t = _as_float_tensor(W0)
    if W0_t.dim() != 2:
        raise ValueError("W0 must have shape [Q,D].")
    Q, D = W0_t.shape
    if D < Q:
        raise ValueError("Structured metric requires observed dimension D >= Q.")
    U0 = _orthonormal_columns(W0_t.transpose(0, 1))
    target = float(Q if trace_target is None else trace_target)
    row_power = W0_t.pow(2).sum(dim=1).clamp_min(1e-8)
    row_power = row_power * (target / row_power.sum().clamp_min(1e-8))
    eta0 = row_power.log().view(1, Q).expand(int(n_anchors), Q).clone()
    eta_log_std0 = torch.full_like(eta0, float(eta_init_log_std))
    return U0, eta0, eta_log_std0


def _eta_trace(
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
) -> torch.Tensor:
    return _eta_trace_from_var(eta_mu, eta_log_std.exp().pow(2))


def _eta_trace_from_var(
    eta_mu: torch.Tensor,
    eta_var: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(eta_mu + 0.5 * eta_var).sum(dim=-1)


def _default_sparse_gp_lengthscale(X: torch.Tensor) -> float:
    if X.shape[0] <= 1:
        return 1.0
    with torch.no_grad():
        d = torch.pdist(X.detach())
        d = d[torch.isfinite(d) & (d > 0)]
        if d.numel() == 0:
            return 1.0
        return float(d.median().clamp_min(1e-3).cpu().item())


def _select_inducing_X(
    X_anchor: torch.Tensor,
    n_inducing: int | None,
) -> torch.Tensor:
    T = int(X_anchor.shape[0])
    M = T if n_inducing is None else max(1, min(int(n_inducing), T))
    if M == T:
        return X_anchor.detach().clone()
    idx = torch.linspace(0, T - 1, steps=M, device=X_anchor.device).round().long().unique()
    if idx.numel() < M:
        remaining = torch.arange(T, device=X_anchor.device)
        keep = torch.ones(T, device=X_anchor.device, dtype=torch.bool)
        keep[idx] = False
        idx = torch.cat([idx, remaining[keep][: M - idx.numel()]])
    return X_anchor[idx[:M]].detach().clone()


def structured_metric_eta_moments_from_R(
    X_query: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    R_mu: torch.Tensor | np.ndarray,
    R_log_std: torch.Tensor | np.ndarray,
    *,
    lengthscale: float,
    signal_var: float = 1.0,
    include_conditional_residual: bool = True,
    jitter: float = 1e-6,
    eta_mean_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return marginal ``q(eta(x))`` moments induced by structured ``q(R)``.

    ``R`` stores only the local log-diagonal metric coordinates, so it has shape
    ``[M,Q]``.  The shared subspace ``U`` is global and not part of this GP.

    ``eta_mean_offset`` (``[Q]``) is a constant prior-mean function added to the
    zero-mean GP output, i.e. ``eta(x) = offset + GP``; ``None`` is the zero-mean
    default.
    """
    Xq = _as_float_tensor(X_query)
    if Xq.dim() != 2:
        raise ValueError("X_query must have shape [T,D].")
    device = Xq.device
    dtype = Xq.dtype
    U = _as_float_tensor(inducing_X, device=device).to(dtype=dtype)
    Rm = _as_float_tensor(R_mu, device=device).to(dtype=dtype)
    Rls = _as_float_tensor(R_log_std, device=device).to(dtype=dtype)
    if Rm.dim() != 2 or Rls.shape != Rm.shape:
        raise ValueError("R_mu/R_log_std must have shape [M,Q].")
    if U.shape[0] != Rm.shape[0] or U.shape[1] != Xq.shape[1]:
        raise ValueError("inducing_X dimensions are inconsistent with X_query/R.")
    eta_mu_all, eta_cov_all = sparse_gp_w_moments_diag(
        Xq.unsqueeze(1),
        U,
        Rm.unsqueeze(-1),
        torch.exp(2.0 * Rls).clamp_min(1e-12).unsqueeze(-1),
        lengthscale=float(lengthscale),
        signal_var=float(signal_var),
        include_conditional_residual=bool(include_conditional_residual),
        jitter=float(jitter),
    )
    eta_mu = eta_mu_all[:, 0, :, 0]
    eta_var = eta_cov_all[:, :, 0, 0, 0].clamp_min(1e-12)
    if eta_mean_offset is not None:
        eta_mu = eta_mu + eta_mean_offset.to(device=device, dtype=dtype).reshape(1, -1)
    return eta_mu, eta_var


def structured_metric_w_moments(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
    *,
    trace_target: float,
    trace_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``W_mu`` and full row covariance for the structured q(W)."""
    return structured_metric_w_moments_from_eta_moments(
        U,
        eta_mu,
        eta_log_std.exp().pow(2),
        trace_target=trace_target,
        trace_normalize=trace_normalize,
    )


def structured_metric_w_moments_from_eta_moments(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_var: torch.Tensor,
    *,
    trace_target: float,
    trace_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``q(W)`` moments from marginal ``q(eta)`` moments."""
    eta_eff = eta_mu
    if bool(trace_normalize):
        trace = torch.exp(eta_mu + 0.5 * eta_var).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        eta_eff = eta_mu - torch.log(trace / float(trace_target))
    scale_mean = torch.exp(0.5 * eta_eff + 0.125 * eta_var)
    scale_second = torch.exp(eta_eff + 0.5 * eta_var)
    scale_var = (scale_second - scale_mean.pow(2)).clamp_min(0.0)
    W_mu = torch.einsum("tq,dq->tqd", scale_mean, U)
    UU = torch.einsum("dq,eq->qde", U, U)
    W_cov = scale_var[:, :, None, None] * UU.unsqueeze(0)
    trace_raw = _eta_trace_from_var(eta_mu, eta_var)
    diag = {
        "trace_raw_mean": float(trace_raw.detach().mean().cpu().item()),
        "trace_raw_std": float(trace_raw.detach().std(unbiased=False).cpu().item()),
        "eta_std_median": float(eta_var.detach().sqrt().median().cpu().item()),
        "D_entropy_mean": float(
            (
                0.5 * (1.0 + math.log(2.0 * math.pi))
                + 0.5 * torch.log(eta_var.clamp_min(1e-12))
            ).sum(dim=1).mean().detach().cpu().item()
        ),
    }
    return W_mu, W_cov, diag


def structured_metric_w_moments_from_R(
    U: torch.Tensor,
    X_query: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    R_mu: torch.Tensor | np.ndarray,
    R_log_std: torch.Tensor | np.ndarray,
    *,
    lengthscale: float,
    signal_var: float,
    include_conditional_residual: bool,
    trace_target: float,
    trace_normalize: bool,
    jitter: float = 1e-6,
    eta_mean_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``q(W(x))`` and ``q(eta(x))`` moments induced by ``q(R)``."""
    eta_mu, eta_var = structured_metric_eta_moments_from_R(
        X_query,
        inducing_X,
        R_mu,
        R_log_std,
        lengthscale=float(lengthscale),
        signal_var=float(signal_var),
        include_conditional_residual=bool(include_conditional_residual),
        jitter=float(jitter),
        eta_mean_offset=eta_mean_offset,
    )
    W_mu, W_cov, diag = structured_metric_w_moments_from_eta_moments(
        U,
        eta_mu,
        eta_var,
        trace_target=float(trace_target),
        trace_normalize=bool(trace_normalize),
    )
    return W_mu, W_cov, eta_mu, eta_var, diag


def sample_structured_metric_w(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
    *,
    n_samples: int,
    trace_target: float,
    trace_normalize: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample structured W with shape ``[S,T,Q,D]``."""
    eps = torch.randn(
        (int(n_samples), *tuple(eta_mu.shape)),
        device=eta_mu.device,
        dtype=eta_mu.dtype,
        generator=generator,
    )
    eta = eta_mu.unsqueeze(0) + eps * eta_log_std.exp().unsqueeze(0)
    if bool(trace_normalize):
        trace = torch.exp(eta).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        eta = eta - torch.log(trace / float(trace_target))
    scale = torch.exp(0.5 * eta)
    return torch.einsum("stq,dq->stqd", scale, U)


def sample_structured_metric_w_from_eta_moments(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_var: torch.Tensor,
    *,
    n_samples: int,
    trace_target: float,
    trace_normalize: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample structured W from marginal ``q(eta)`` moments."""
    eps = torch.randn(
        (int(n_samples), *tuple(eta_mu.shape)),
        device=eta_mu.device,
        dtype=eta_mu.dtype,
        generator=generator,
    )
    eta = eta_mu.unsqueeze(0) + eps * eta_var.clamp_min(1e-12).sqrt().unsqueeze(0)
    if bool(trace_normalize):
        trace = torch.exp(eta).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        eta = eta - torch.log(trace / float(trace_target))
    scale = torch.exp(0.5 * eta)
    return torch.einsum("stq,dq->stqd", scale, U)


def _eta_kl(
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
    *,
    trace_target: float,
    prior_std: float,
) -> torch.Tensor:
    Q = eta_mu.shape[1]
    prior_mu = math.log(float(trace_target) / float(Q))
    prior_var = float(prior_std) ** 2
    var = eta_log_std.exp().pow(2)
    return 0.5 * (
        (var + (eta_mu - prior_mu).pow(2)) / prior_var
        - 1.0
        + math.log(prior_var)
        - torch.log(var.clamp_min(1e-12))
    ).sum()


def build_fixed_local_inducing_from_w0(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    W0: torch.Tensor | np.ndarray,
    *,
    m_inducing: int = 10,
    return_raw: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose fixed local inducing locations near each anchor in ``W0 x`` space.

    The first inducing location is the projected anchor.  Remaining locations
    are the nearest projected raw-X neighbors.  The returned ``mu_u`` initializer
    uses the corresponding neighbor responses, with the anchor slot initialized
    from the nearest-neighbor response.

    If ``return_raw`` is True, additionally return the raw input-space locations
    ``X_ind_raw`` ``[T, m, D]`` (anchor + the selected neighbors, same order and
    padding as ``C``).  These let a caller re-project the inducing inputs with the
    *current* metric ``W`` during training (``reproject_local_inducing``) instead
    of freezing them at the ``W0`` projection.
    """
    Xa = _as_float_tensor(X_anchor)
    device = Xa.device
    dtype = Xa.dtype
    Xn = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    W = _as_float_tensor(W0, device=device).to(dtype=dtype)
    if y.dim() == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    T, n, D = Xn.shape
    Q = W.shape[0]
    m = max(1, int(m_inducing))
    za = torch.einsum("qd,td->tq", W, Xa)
    zn = torch.einsum("qd,tnd->tnq", W, Xn)
    dist = (zn - za.unsqueeze(1)).pow(2).sum(dim=-1)
    k = min(n, max(1, m - 1))
    idx = torch.topk(dist, k=k, largest=False).indices
    gather_idx = idx.unsqueeze(-1).expand(T, k, Q)
    C_tail = torch.gather(zn, 1, gather_idx)
    y_tail = torch.gather(y, 1, idx)
    C = torch.cat([za.unsqueeze(1), C_tail], dim=1)
    mu_u0 = torch.cat([y_tail[:, :1], y_tail], dim=1)
    # Raw input-space locations matching C's order: anchor, then selected neighbors.
    Xn_tail = torch.gather(Xn, 1, idx.unsqueeze(-1).expand(T, k, D))
    X_raw = torch.cat([Xa.unsqueeze(1), Xn_tail], dim=1)
    if C.shape[1] < m:
        pad = m - C.shape[1]
        C = torch.cat([C, C[:, -1:].expand(T, pad, Q)], dim=1)
        mu_u0 = torch.cat([mu_u0, mu_u0[:, -1:].expand(T, pad)], dim=1)
        X_raw = torch.cat([X_raw, X_raw[:, -1:].expand(T, pad, D)], dim=1)
    if return_raw:
        return C.contiguous(), mu_u0.contiguous(), X_raw.contiguous()
    return C.contiguous(), mu_u0.contiguous()


def _local_kl_diag_u(
    C: torch.Tensor,
    mu_u: torch.Tensor,
    u_var: torch.Tensor,
    *,
    jitter: float,
    signal_var: float = 1.0,
    prior_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    T, m, _Q = C.shape
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    K = float(signal_var) * torch.exp(-0.5 * d2) + float(jitter) * torch.eye(m, device=C.device, dtype=C.dtype).unsqueeze(0)
    L = torch.linalg.cholesky(K)
    K_inv = torch.cholesky_inverse(L)
    diag_inv = torch.diagonal(K_inv, dim1=-2, dim2=-1)
    logdet_K = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    trace = (diag_inv * u_var).sum(dim=-1)
    # Non-zero prior mean c (per anchor, broadcast over inducing slots): the KL
    # shrinks mu_u toward c instead of 0, so the local prediction is pulled toward
    # the region level rather than the global mean.
    dev = mu_u if prior_mean is None else (mu_u - prior_mean.view(T, 1))
    quad = torch.einsum("ti,tij,tj->t", dev, K_inv, dev)
    logdet_q = torch.log(u_var.clamp_min(1e-12)).sum(dim=-1)
    return 0.5 * (trace + quad - float(m) + logdet_K - logdet_q)


def _free_w_moments(
    X_query: torch.Tensor,
    inducing_X: torch.Tensor,
    W0: torch.Tensor,
    V_mu: torch.Tensor,
    V_log_std: torch.Tensor,
    *,
    lengthscale: float,
    signal_var: float,
    include_conditional_residual: bool,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor-mode entrywise sparse-GP W: W(x) = W0 + A(x) V, independent (q,d) entries.

    V_mu / V_log_std: [M,Q,D] deviation inducing values. Returns W_mu [T,Q,D] and
    diagonal row covariance W_cov [T,Q,D,D].
    """
    T = X_query.shape[0]
    M = inducing_X.shape[0]
    ls2 = float(lengthscale) ** 2
    d2_zz = (inducing_X.unsqueeze(1) - inducing_X.unsqueeze(0)).pow(2).sum(-1)
    Kzz = float(signal_var) * torch.exp(-0.5 * d2_zz / ls2) + float(jitter) * torch.eye(
        M, device=X_query.device, dtype=X_query.dtype)
    L = torch.linalg.cholesky(0.5 * (Kzz + Kzz.T))
    d2_xz = (X_query.unsqueeze(1) - inducing_X.unsqueeze(0)).pow(2).sum(-1)
    Kxz = float(signal_var) * torch.exp(-0.5 * d2_xz / ls2)
    A = torch.cholesky_solve(Kxz.T, L).T  # [T,M]
    W_mu = W0.unsqueeze(0) + torch.einsum("tm,mqd->tqd", A, V_mu)
    V_var = torch.exp(2.0 * V_log_std).clamp_min(1e-12)
    var_entries = torch.einsum("tm,mqd->tqd", A.pow(2), V_var)
    if include_conditional_residual:
        cond = (float(signal_var) - (A * Kxz).sum(-1)).clamp_min(0.0)  # [T]
        var_entries = var_entries + cond.view(T, 1, 1)
    return W_mu, torch.diag_embed(var_entries)


def _free_w_kl(
    inducing_X: torch.Tensor,
    V_mu: torch.Tensor,
    V_log_std: torch.Tensor,
    *,
    lengthscale: float,
    signal_var: float,
    jitter: float,
) -> torch.Tensor:
    """KL(q(V)||p(V)) for the entrywise deviation GP (zero-mean prior, shared Kzz)."""
    M, Q, D = V_mu.shape
    ls2 = float(lengthscale) ** 2
    d2_zz = (inducing_X.unsqueeze(1) - inducing_X.unsqueeze(0)).pow(2).sum(-1)
    Kzz = float(signal_var) * torch.exp(-0.5 * d2_zz / ls2) + float(jitter) * torch.eye(
        M, device=inducing_X.device, dtype=inducing_X.dtype)
    L = torch.linalg.cholesky(0.5 * (Kzz + Kzz.T))
    K_inv = torch.cholesky_inverse(L)
    logdet_K = 2.0 * torch.log(torch.diagonal(L)).sum()
    s2 = torch.exp(2.0 * V_log_std).clamp_min(1e-12)
    trace = torch.einsum("m,mqd->", torch.diagonal(K_inv), s2)
    quad = torch.einsum("mqd,mk,kqd->", V_mu, K_inv, V_mu)
    logdet_q = torch.log(s2).sum()
    return 0.5 * (trace + quad - float(M * Q * D) + float(Q * D) * logdet_K - logdet_q)


def _local_kl_full_u(
    Kuu: torch.Tensor,
    Luu: torch.Tensor,
    mu_u: torch.Tensor,
    Sigma_u: torch.Tensor,
    *,
    prior_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL( N(mu_u, Sigma_u) || N(c 1, Kuu) ) with FULL covariance, batched over anchors.

    Kuu/Luu are the (jittered) prior covariance and its Cholesky [T,m,m]; Sigma_u is
    assumed PD (the collapsed Sigma* = K B^{-1} K is by construction).
    """
    T, m, _ = Kuu.shape
    K_inv = torch.cholesky_inverse(Luu)
    logdet_K = 2.0 * torch.log(torch.diagonal(Luu, dim1=-2, dim2=-1)).sum(dim=-1)
    trace = torch.einsum("tij,tji->t", K_inv, Sigma_u)
    dev = mu_u if prior_mean is None else (mu_u - prior_mean.view(T, 1))
    quad = torch.einsum("ti,tij,tj->t", dev, K_inv, dev)
    Sig_sym = 0.5 * (Sigma_u + Sigma_u.transpose(-2, -1))
    L_S = torch.linalg.cholesky(
        Sig_sym + 1e-10 * torch.eye(m, device=Kuu.device, dtype=Kuu.dtype).unsqueeze(0)
    )
    logdet_S = 2.0 * torch.log(torch.diagonal(L_S, dim1=-2, dim2=-1)).sum(dim=-1)
    return 0.5 * (trace + quad - float(m) + logdet_K - logdet_S)


def _otsu_separability(y: torch.Tensor) -> torch.Tensor:
    """Per-anchor Otsu separability eta = max_split sigma2_between / sigma2_total in [0,1].

    y: [T, n] neighbour responses. eta -> 1 when the responses split into two tight,
    well-separated clusters (a clear jump); eta -> 0 when unimodal/smooth. A multi-region
    neighbourhood gives a moderate eta (the 2-class split leaves high within-class spread),
    so this also down-weights non-binary cases. Cheap, vectorised, DETACHED (a temperature
    signal, not a learnable). Used to drive adaptive annealing: harden where eta is high.
    """
    T, n = y.shape
    ys, _ = torch.sort(y, dim=1)                          # [T,n]
    csum = torch.cumsum(ys, dim=1)                        # [T,n]
    total = csum[:, -1:]                                  # [T,1]
    k = torch.arange(1, n, device=y.device, dtype=y.dtype).view(1, n - 1)  # split after k pts
    c = csum[:, :n - 1]                                   # sum of left part  [T,n-1]
    w1 = k / float(n)
    mu1 = c / k
    mu2 = (total - c) / (float(n) - k)
    sigma_between = (w1 * (1.0 - w1) * (mu1 - mu2).pow(2)).max(dim=1).values  # [T]
    var_total = y.var(dim=1, unbiased=False).clamp_min(1e-12)                 # [T]
    return (sigma_between / var_total).clamp(0.0, 1.0)


def _analytic_local_terms(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    C: torch.Tensor,
    W_mu: torch.Tensor,
    W_cov: torch.Tensor,
    mu_u: torch.Tensor,
    u_log_std: torch.Tensor,
    log_noise_var: torch.Tensor,
    rho_logit: torch.Tensor,
    *,
    local_signal_var: float,
    outlier_scale_mult: float,
    jitter: float,
    gate_w: torch.Tensor | None = None,
    gate_mode: str = "intercept",
    gate_gh_points: int = 20,
    out_mean: torch.Tensor | None = None,
    out_log_var: torch.Tensor | None = None,
    fixed_rho: torch.Tensor | None = None,
    local_prior_mean: torch.Tensor | None = None,
    profile_mean: bool = False,
    gate_temp: float = 1.0,
    adaptive_temp_min: float | None = None,
    adaptive_signal: str = "entropy",
    mu_u_B: torch.Tensor | None = None,
    u_log_std_B: torch.Tensor | None = None,
    log_noise_var_B: torch.Tensor | None = None,
    gate_nu0_nonneg: bool = False,
    outlier_floor: bool = False,
    outlier_floor_k: float = 2.5,
    uniform_outlier: bool = False,
    ell_loc: torch.Tensor | None = None,
    collapse_resp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if y_neighbors.dim() == 3 and y_neighbors.shape[-1] == 1:
        y_neighbors = y_neighbors.squeeze(-1)
    T, n, _D = X_neighbors.shape
    m = C.shape[1]
    sigma_f = torch.full((T,), math.sqrt(float(local_signal_var)), device=X_neighbors.device, dtype=X_neighbors.dtype)
    # ABLATION #1: decouple the local-GP lengthscale from W by scaling the kernel's W / C
    # by 1 / ell_loc (per anchor). Mathematically equivalent to an RBF lengthscale ell_loc
    # in z-space, so the existing unit-lengthscale moment formulas are reused verbatim. The
    # spatial gate below keeps the ORIGINAL (full-scale) W -- only the local GP is rescaled.
    if ell_loc is not None:
        inv_l = (1.0 / ell_loc.clamp_min(1e-6))
        W_mu_k = W_mu * inv_l.view(T, 1, 1)
        W_cov_k = W_cov * inv_l.view(T, 1, 1, 1).pow(2)
        C_k = C * inv_l.view(T, 1, 1)
    else:
        W_mu_k, W_cov_k, C_k = W_mu, W_cov, C
    Kfu = _expected_Kfu_structured(W_mu_k, W_cov_k, X_neighbors, C_k, sigma_f)
    KufKfu = _expected_KufKfu_structured(W_mu_k, W_cov_k, X_neighbors, C_k, sigma_f)
    d2 = (C_k.unsqueeze(2) - C_k.unsqueeze(1)).pow(2).sum(-1)
    Kuu = float(local_signal_var) * torch.exp(-0.5 * d2) + float(jitter) * torch.eye(m, device=C.device, dtype=C.dtype).unsqueeze(0)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)
    noise_var = log_noise_var.exp().clamp_min(1e-8).view(T, 1)
    Sigma_u_star = None
    if collapse_resp is not None:
        # 4A: closed-form optimal q*(u) under the (detached) responsibility-weighted
        # Gaussian data term:  Sigma* = K B^{-1} K,  mu* = K B^{-1} (b1/sigma^2 + c 1),
        # B = Psi2_r / sigma^2 + K,  Psi2_r = sum_i r_i Psi2_i,  b1 = Psi1^T (r . y).
        # Differentiable through Psi1/Psi2 (-> W) and sigma -> the ELBO's W-gradient
        # carries the envelope term through q*(u).
        r_w = collapse_resp.detach().clamp(0.0, 1.0)
        Psi2_r = torch.einsum("tn,tnij->tij", r_w, KufKfu)
        # With a non-zero prior mean c the whole problem is the zero-mean one on the
        # DEVIATION u - c 1 driven by the centred response y - c, so centre b1 too and
        # add c back at the end (the old "rhs += c" form did not match the conditional).
        y_for_b1 = y_neighbors if local_prior_mean is None else (y_neighbors - local_prior_mean.view(T, 1))
        b1 = torch.einsum("tn,tni->ti", r_w * y_for_b1, Kfu)
        B = Psi2_r / noise_var.view(T, 1, 1) + Kuu
        B = 0.5 * (B + B.transpose(-2, -1))
        L_B = torch.linalg.cholesky(B)
        rhs = b1 / noise_var
        mu_u = torch.einsum("tij,tj->ti", Kuu, torch.cholesky_solve(rhs.unsqueeze(-1), L_B).squeeze(-1))
        if local_prior_mean is not None:
            mu_u = mu_u + local_prior_mean.view(T, 1)
        Sigma_u_star = torch.einsum("tij,tjk->tik", Kuu, torch.cholesky_solve(Kuu, L_B))
        Sigma_u_star = 0.5 * (Sigma_u_star + Sigma_u_star.transpose(-2, -1))
        u_cov = Sigma_u_star
    else:
        u_var = torch.exp(2.0 * u_log_std).clamp_min(1e-12)
        u_cov = torch.diag_embed(u_var)
    # Constant local mean of f. "zero": none. "inlier": the fixed prior level c, in which
    # case the GP machinery runs on the deviation u - c 1 (this is the conditional that
    # MATCHES the KL's non-zero prior mean). "profile": a free intercept solved for below,
    # for which u is already the zero-mean residual's inducing vector (dev = mu_u).
    dev_u = mu_u if local_prior_mean is None else (mu_u - local_prior_mean.view(T, 1))
    u_second = u_cov + torch.einsum("ti,tj->tij", dev_u, dev_u)
    alpha = torch.einsum("tij,tj->ti", Kuu_inv, dev_u)
    # zeta = E_q[f - m_loc]: the RESIDUAL predictive mean (excludes the constant mean).
    zeta = torch.einsum("tni,ti->tn", Kfu, alpha)
    v1 = float(local_signal_var) - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    t3 = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, u_second)
    # second_f = E_q[(f - m_loc)^2]; the constant mean does not enter it.
    second_f = (v1 + t3).clamp_min(1e-12)
    g_mean = None  # gate logit mean (set in the spatial branch; used for gate supervision)
    if fixed_rho is not None:
        # Frozen per-neighbour inlier probability (e.g. an ORACLE gate from true
        # region labels). Bypasses the learned gate entirely -- isolates "if the
        # responsibilities were perfect, does the prediction improve?".
        rho_eff = fixed_rho.clamp(1e-4, 1.0 - 1e-4)
        e_log_rho = torch.log(rho_eff)
        e_log_1mrho = torch.log1p(-rho_eff)
    elif str(gate_mode) == "spatial":
        # Spatial JumpGP boundary g = nu_0 + w^T z with z = W(x_i - x_anchor)
        # RANDOM under q(W). The structured W has independent rows, so the
        # projected coordinate is diagonal-Gaussian: z_mu_q = (W_mu delta)_q and
        # z_var_q = delta^T W_cov[:,q] delta. The scalar gate score is therefore
        # Gaussian g ~ N(g_mean, g_var); the logistic-Gaussian expectations
        # E_q[log sigmoid(+/-g)] are evaluated by Gauss-Hermite quadrature so the
        # collapsed-responsibility logsumexp below remains a valid ELBO term.
        nu0 = (torch.nn.functional.softplus(rho_logit) if gate_nu0_nonneg else rho_logit).view(T, 1)
        Q = W_mu.shape[1]
        w_gate = gate_w if gate_w is not None else torch.zeros((T, Q), device=W_mu.device, dtype=W_mu.dtype)
        delta = X_neighbors - X_anchor.unsqueeze(1)
        z_mu = torch.einsum("tqd,tnd->tnq", W_mu, delta)
        z_var = torch.einsum("tqde,tnd,tne->tnq", W_cov, delta, delta).clamp_min(0.0)
        g_mean = nu0 + (w_gate.unsqueeze(1) * z_mu).sum(dim=-1)
        g_var = (w_gate.unsqueeze(1).pow(2) * z_var).sum(dim=-1).clamp_min(0.0)
        e_log_rho, e_log_1mrho, rho_eff = gh_logsigmoid_expectations(
            g_mean, g_var, gh_points=int(gate_gh_points)
        )
    else:
        rho = torch.sigmoid(rho_logit).view(T, 1).clamp(1e-4, 1.0 - 1e-4)
        e_log_rho = torch.log(rho)
        e_log_1mrho = torch.log1p(-rho)
        rho_eff = rho.expand(T, n)
    # --- constant local mean m_loc, and the resulting inlier likelihood -------------
    # rho_eff never depends on the local likelihood (intercept: per-anchor constant;
    # spatial: a function of the gate; fixed_rho: frozen), so the responsibility-weighted
    # coordinate maximum for a free intercept is available in closed form right here --
    # no EM iteration, no circularity. rho is detached; zeta stays live (the envelope
    # theorem makes the mu_j path contribute nothing to the other gradients anyway).
    if profile_mean:
        w_in = rho_eff.detach().clamp_min(1e-6)
        m_loc = (w_in * (y_neighbors - zeta)).sum(dim=1) / w_in.sum(dim=1)
    elif local_prior_mean is not None:
        m_loc = local_prior_mean.view(T)
    else:
        m_loc = None
    if m_loc is None:
        y_c = y_neighbors
        mean_f = zeta
    else:
        y_c = y_neighbors - m_loc.view(T, 1)
        mean_f = zeta + m_loc.view(T, 1)
    sqerr = y_c.pow(2) - 2.0 * y_c * zeta + second_f
    log_inlier = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(noise_var) - 0.5 * sqerr / noise_var
    local_var = y_neighbors.var(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    if mu_u_B is not None and u_log_std_B is not None and log_noise_var_B is not None:
        # Second local-GP expert B as the "outlier" branch -- same machinery as A,
        # sharing the inducing locations C and the (W-marginalised) Psi statistics
        # Kfu / KufKfu / Kuu_inv / v1, with its own values, second moment and noise.
        # This replaces the flat single-Gaussian outlier so that a cross-region point
        # assigned to B gets a proper high GP likelihood (B fits its own region).
        u_var_B = torch.exp(2.0 * u_log_std_B).clamp_min(1e-12)
        u_second_B = torch.diag_embed(u_var_B) + torch.einsum("ti,tj->tij", mu_u_B, mu_u_B)
        alpha_B = torch.einsum("tij,tj->ti", Kuu_inv, mu_u_B)
        zeta_B = torch.einsum("tni,ti->tn", Kfu, alpha_B)
        t3_B = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, u_second_B)
        second_f_B = (v1 + t3_B).clamp_min(1e-12)
        noise_var_B = log_noise_var_B.exp().clamp_min(1e-8).view(T, 1)
        # Expert B gets its OWN intercept, profiled against the outlier responsibilities.
        if profile_mean:
            w_out = (1.0 - rho_eff).detach().clamp_min(1e-6)
            m_loc_B = (w_out * (y_neighbors - zeta_B)).sum(dim=1) / w_out.sum(dim=1)
            y_c_B = y_neighbors - m_loc_B.view(T, 1)
            mean_f_B = zeta_B + m_loc_B.view(T, 1)
        else:
            m_loc_B = None
            y_c_B = y_neighbors
            mean_f_B = zeta_B
        sqerr_B = y_c_B.pow(2) - 2.0 * y_c_B * zeta_B + second_f_B
        log_outlier = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(noise_var_B) - 0.5 * sqerr_B / noise_var_B
    elif out_mean is not None and out_log_var is not None:
        # Learnable per-anchor outlier Gaussian. Floor its variance at the inlier
        # observation noise so it stays a genuine broad alternative and cannot
        # collapse to a sharp spike that steals inlier points; its mean is free to
        # move onto the other (cross-region) cluster.
        om = out_mean.view(T, 1)
        out_var = out_log_var.view(T, 1).exp().clamp_min(1e-4)
        log_outlier = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(out_var) - 0.5 * (y_neighbors - om).pow(2) / out_var
    elif outlier_floor:
        # ABLATION #4: JumpGP-style fixed density floor. log N(k*sigma; 0, sigma) is a
        # per-anchor constant (independent of y_i); a neighbour is inlier iff its
        # standardized residual < k, i.e. log_inlier - log_outlier = 0.5(k^2 - sqerr/sigma^2).
        k = float(outlier_floor_k)
        log_outlier = (-0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(noise_var)
                       - 0.5 * k * k).expand(T, n)
    elif uniform_outlier:
        # Paper's flat (uniform) outlier density p(y_i|v=0) = 1/u_j, with
        # u_j = max(range(y^(j)), eps_u) the local response range floored away from 0.
        # log_outlier = -log(u_j) is a per-anchor CONSTANT (independent of y_i): a pure
        # detection threshold, contributing no gradient through the outlier branch.
        u_j = (y_neighbors.max(dim=1, keepdim=True).values
               - y_neighbors.min(dim=1, keepdim=True).values).clamp_min(1e-6)
        log_outlier = (-torch.log(u_j)).expand(T, n)
    else:
        local_mean = y_neighbors.mean(dim=1, keepdim=True)
        out_var = (float(outlier_scale_mult) ** 2) * local_var
        log_outlier = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(out_var) - 0.5 * (y_neighbors - local_mean).pow(2) / out_var
    a_branch = e_log_rho + log_inlier
    b_branch = e_log_1mrho + log_outlier
    stacked = torch.stack([a_branch, b_branch], dim=0)
    if adaptive_temp_min is not None:
        # Per-anchor adaptive temperature. Two signals (DETACHED -> no degenerate
        # "drive T->0" gradient):
        #   "entropy"    : entropy of the T=1 posterior responsibilities. CIRCULAR when the
        #                  gate is poor (diffuse r everywhere -> never hardens).
        #   "bimodality" : Otsu separability of the neighbour responses (gate-independent).
        #                  Clear binary jump -> harden; smooth/multi-region -> stay soft.
        with torch.no_grad():
            if str(adaptive_signal) == "bimodality":
                eta = _otsu_separability(y_neighbors)                       # [T] in [0,1]
                s = eta                                                     # high -> harden
            else:
                r = torch.sigmoid(a_branch - b_branch).clamp(1e-6, 1.0 - 1e-6)
                h = -(r * torch.log(r) + (1.0 - r) * torch.log1p(-r))      # [T,n] binary entropy (nats)
                s = 1.0 - (h.mean(dim=1) / math.log(2.0)).clamp(0.0, 1.0)  # low entropy -> harden
            T_vec = (1.0 - (1.0 - float(adaptive_temp_min)) * s).clamp_min(1e-3)  # s=1 -> T_min; s=0 -> 1
        T_b = T_vec.view(1, T, 1)
        point_logp = T_vec.view(T, 1) * torch.logsumexp(stacked / T_b, dim=0)
    elif float(gate_temp) == 1.0:
        point_logp = torch.logsumexp(stacked, dim=0)
    else:
        # Deterministic-annealing tempered free energy: T*logsumexp([a,b]/T).
        # T=1 -> soft EM (marginal); T->0 -> max(a,b) = hard CEM (sharp assignment).
        point_logp = float(gate_temp) * torch.logsumexp(stacked / float(gate_temp), dim=0)
    region_elbo = point_logp.sum(dim=1)
    if Sigma_u_star is not None:
        kl_u = _local_kl_full_u(Kuu, Luu, mu_u, Sigma_u_star, prior_mean=local_prior_mean)
    else:
        kl_u = _local_kl_diag_u(C_k, mu_u, u_var, jitter=jitter, signal_var=local_signal_var,
                                prior_mean=local_prior_mean)
    diag = {
        "region_elbo_mean": region_elbo.mean().detach(),
        "kl_u_mean": kl_u.mean().detach(),
        "rho_mean": rho_eff.mean().detach(),
        "rho_anchor": rho_eff.mean(dim=1).detach(),
        "rho_within_std": rho_eff.std(dim=1).mean().detach(),
        "noise_std_mean": noise_var.sqrt().mean().detach(),
        "mean_f_abs": mean_f.abs().mean().detach(),
        "mean_second_f": second_f.mean().detach(),
        "pure_lik_logit": (log_inlier - log_outlier).detach(),  # likelihood-only resp. logit
    }
    # Per-anchor constant local mean actually used (None in "zero" mode). Exported so the
    # trainer can freeze it into the result and predict can add it back.
    diag["local_mean_c"] = None if m_loc is None else m_loc.detach()
    if m_loc is not None:
        diag["local_mean_abs"] = m_loc.abs().mean().detach()
    diag["gate_score"] = g_mean   # [T,n] gate logit mean (spatial), kept WITH grad; None otherwise
    if Sigma_u_star is not None:
        # EM-style refresh targets for the persistent responsibility buffer + the
        # collapsed q*(u) moments (all detached; T=1 posterior responsibilities).
        diag["resp_new"] = torch.sigmoid(a_branch - b_branch).detach()
        diag["mu_u_star"] = mu_u.detach()
        diag["Sigma_u_star"] = Sigma_u_star.detach()
    return region_elbo, kl_u, diag


def _expected_Kfu_structured(mu_W: torch.Tensor, cov_W: torch.Tensor, X: torch.Tensor, C: torch.Tensor, sigma_f: torch.Tensor) -> torch.Tensor:
    T, _n, _D = X.shape
    s = torch.einsum("tnd,tqde,tne->tnq", X, cov_W, X).clamp_min(0.0)
    den = torch.sqrt(s + 1.0)
    mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
    diff = mu_proj.unsqueeze(2) - C.unsqueeze(1)
    exp_term = torch.exp(-0.5 * diff.pow(2) / (s.unsqueeze(2) + 1.0))
    # RBF amplitude is sigma_f**2; Psi1 = E_W[k(z,C)] carries one factor of sigma_f**2.
    return sigma_f.view(T, 1, 1).pow(2) * exp_term.prod(dim=-1) / den.prod(dim=-1, keepdim=True)


def _expected_KufKfu_structured(mu_W: torch.Tensor, cov_W: torch.Tensor, X: torch.Tensor, C: torch.Tensor, sigma_f: torch.Tensor) -> torch.Tensor:
    T, n, _D = X.shape
    s = torch.einsum("tnd,tqde,tne->tnq", X, cov_W, X).clamp_min(0.0)
    mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
    mid = 0.5 * (C.unsqueeze(2) + C.unsqueeze(1))
    diff = mu_proj.unsqueeze(2).unsqueeze(3) - mid.unsqueeze(1)
    # Product of two RBF kernels doubles the curvature: the marginal over a
    # Gaussian z ~ N(mu, s) of exp(-(z-mid)^2) integrates to
    # (1/sqrt(1+2s)) exp(-(mu-mid)^2/(1+2s)), so the denominator is (2s+1),
    # not (s+1) as in the single-kernel Psi1 statistic.
    denom = 2.0 * s + 1.0
    exp_term = torch.exp(-diff.pow(2) / denom.unsqueeze(2).unsqueeze(2))
    num = exp_term.prod(dim=-1)
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    prior = torch.exp(-0.25 * d2)
    den_prod = torch.sqrt(denom).prod(dim=-1)
    # Psi2 = E_W[k(z,C_i) k(z,C_j)] carries sigma_f**4 (two kernel factors).
    return sigma_f.view(T, 1, 1, 1).pow(4) * prior.unsqueeze(1) * (num / den_prod.view(T, n, 1, 1))


def train_structured_metric_lmjgp(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    init_labels: torch.Tensor | np.ndarray,
    *,
    init_W: torch.Tensor | np.ndarray,
    init_lengthscale: torch.Tensor | np.ndarray,
    init_signal_var: torch.Tensor | np.ndarray,
    init_noise_var: torch.Tensor | np.ndarray,
    gate_init: torch.Tensor | np.ndarray,
    cem_config: SelfCEMConfig,
    config: StructuredMetricLMJGPConfig,
    eval_callback: EvalCallback | None = None,
) -> StructuredMetricLMJGPResult:
    """Train structured ``q(eta), U`` with stochastic composite likelihood."""
    t0 = time.perf_counter()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    labels_t = _as_float_tensor(init_labels, device=device).to(dtype=dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if labels_t.dim() == 3 and labels_t.shape[-1] == 1:
        labels_t = labels_t.squeeze(-1)
    T, _n, D = X_neighbors_t.shape
    W0 = _as_float_tensor(init_W, device=device).to(dtype=dtype)
    Q = W0.shape[0]
    trace_target = float(Q if config.trace_target is None else config.trace_target)
    U0, eta0, eta_log_std0 = init_structured_metric_from_w0(
        W0,
        n_anchors=T,
        trace_target=trace_target,
        eta_init_log_std=float(config.eta_init_log_std),
    )
    inducing_X = _select_inducing_X(X_anchor_t, config.n_inducing).to(device=device, dtype=dtype)
    M = int(inducing_X.shape[0])
    R0 = eta0[:1].expand(M, Q).contiguous()
    U_raw = torch.nn.Parameter(U0.to(device=device, dtype=dtype).clone())
    R_mu = torch.nn.Parameter(R0.to(device=device, dtype=dtype).clone())
    R_log_std = torch.nn.Parameter(
        eta_log_std0[:1].expand(M, Q).contiguous().to(device=device, dtype=dtype).clone()
    )
    params: list[torch.nn.Parameter] = [U_raw, R_mu]
    if bool(config.train_eta_log_std):
        params.append(R_log_std)
    opt = torch.optim.Adam(params, lr=float(config.lr))
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(config.seed))
    init_lengthscale_t = _as_float_tensor(init_lengthscale, device=device).to(dtype=dtype)
    init_signal_var_t = _as_float_tensor(init_signal_var, device=device).to(dtype=dtype)
    init_noise_var_t = _as_float_tensor(init_noise_var, device=device).to(dtype=dtype)
    gate_init_t = _as_float_tensor(gate_init, device=device).to(dtype=dtype)
    w_lengthscale = (
        _default_sparse_gp_lengthscale(X_anchor_t)
        if config.w_lengthscale is None
        else float(config.w_lengthscale)
    )
    w_signal_var = float(config.w_signal_var) * float(config.eta_prior_std) ** 2
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_direct_rmse = float("inf")
    best_direct_state: dict[str, torch.Tensor] | None = None
    for step in range(1, int(config.steps) + 1):
        opt.zero_grad()
        B = min(int(config.batch_anchors), T)
        idx = torch.randperm(T, device=device, generator=gen)[:B]
        U = _orthonormal_columns(U_raw)
        eta_mu_b, eta_var_b = structured_metric_eta_moments_from_R(
            X_anchor_t[idx],
            inducing_X,
            R_mu,
            R_log_std,
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            jitter=float(config.jitter),
        )
        W_s = sample_structured_metric_w_from_eta_moments(
            U,
            eta_mu_b,
            eta_var_b,
            n_samples=int(config.n_samples),
            trace_target=trace_target,
            trace_normalize=bool(config.trace_normalize),
            generator=gen,
        )
        S = W_s.shape[0]
        W_flat = W_s.reshape(S * B, Q, D).contiguous()
        zero_var = torch.zeros_like(W_flat)
        X_anchor_b = _repeat_first(X_anchor_t[idx], S)
        X_neighbors_b = _repeat_first(X_neighbors_t[idx], S)
        y_b = _repeat_first(y_t[idx], S)
        labels_b = _repeat_first(labels_t[idx], S)
        state = run_uncertain_w_self_cem(
            X_anchor_b,
            X_neighbors_b,
            y_b,
            labels_b,
            mode="anchor",
            W_mu_anchor=W_flat.detach(),
            W_var_anchor=zero_var.detach(),
            init_lengthscale=_repeat_first(init_lengthscale_t[idx], S),
            init_signal_var=_repeat_first(init_signal_var_t[idx], S),
            init_noise_var=_repeat_first(init_noise_var_t[idx], S),
            gate_init=_repeat_first(gate_init_t[idx], S),
            config=cem_config,
        )
        gate = state["gate"]
        if isinstance(gate, torch.Tensor):
            gate = gate.clone()
            gate[:, 1:] = 0.0
        outlier_logp = state.get("outlier_logp") if isinstance(state, dict) else None
        if isinstance(outlier_logp, torch.Tensor) and outlier_logp.numel() == 0:
            outlier_logp = None
        rewards, reward_diag = _hard_cem_bound_per_anchor_from_w_moments(
            X_anchor_b,
            X_neighbors_b,
            y_b,
            state["labels"],
            mode="anchor",
            w_moments={"W_mu_anchor": W_flat, "W_var_anchor": zero_var},
            gate=gate,
            lengthscale=state["lengthscale"],
            signal_var=state["signal_var"],
            noise_var=state["noise_var"],
            mean_const=state["mean_const"],
            outlier_mode=str(config.outlier_mode),
            outlier_sigma_mult=float(config.outlier_sigma_mult),
            background_scale_mult=float(config.background_scale_mult),
            outlier_logp=outlier_logp,
            gh_points=int(config.gh_points),
            jitter=float(config.jitter),
            min_inliers=int(config.min_inliers),
        )
        data_objective = rewards.mean()
        kl = _diag_sparse_gp_kl(
            inducing_X,
            R_mu.unsqueeze(-1),
            R_log_std.unsqueeze(-1),
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            jitter=float(config.jitter),
        ) / float(T)
        beta_kl = float(config.beta_kl) * min(1.0, step / max(1, int(config.kl_warmup_steps)))
        eta_mu_all, eta_var_all = structured_metric_eta_moments_from_R(
            X_anchor_t,
            inducing_X,
            R_mu,
            R_log_std,
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            jitter=float(config.jitter),
        )
        trace_raw = _eta_trace_from_var(eta_mu_all, eta_var_all)
        trace_penalty = (trace_raw - trace_target).pow(2).mean()
        loss = -data_objective + beta_kl * kl + float(config.trace_penalty_weight) * trace_penalty
        loss.backward()
        if float(config.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
        opt.step()
        with torch.no_grad():
            R_log_std.clamp_(min=-8.0, max=2.0)
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "U_raw": U_raw.detach().clone(),
                "R_mu": R_mu.detach().clone(),
                "R_log_std": R_log_std.detach().clone(),
            }
        if step == 1 or step % int(config.log_interval) == 0 or step == int(config.steps):
            with torch.no_grad():
                U_eval = _orthonormal_columns(U_raw)
                W_mu, W_cov, _eta_mu_eval, _eta_var_eval, moment_diag = structured_metric_w_moments_from_R(
                    U_eval,
                    X_anchor_t,
                    inducing_X,
                    R_mu,
                    R_log_std,
                    lengthscale=w_lengthscale,
                    signal_var=w_signal_var,
                    include_conditional_residual=bool(config.include_conditional_residual),
                    trace_target=trace_target,
                    trace_normalize=bool(config.trace_normalize),
                    jitter=float(config.jitter),
                )
            eval_diag: dict[str, float] = {}
            if eval_callback is not None and (
                step == 1 or step % max(1, int(config.eval_interval)) == 0 or step == int(config.steps)
            ):
                eval_diag = eval_callback(
                    step,
                    {"W_mu_anchor": W_mu.detach(), "W_cov_anchor": W_cov.detach()},
                    moment_diag,
                )
                if "direct_eval_rmse" in eval_diag:
                    direct_rmse = float(eval_diag["direct_eval_rmse"])
                    if direct_rmse < best_direct_rmse:
                        best_direct_rmse = direct_rmse
                        best_direct_state = {
                            "U_raw": U_raw.detach().clone(),
                            "R_mu": R_mu.detach().clone(),
                            "R_log_std": R_log_std.detach().clone(),
                        }
            row = {
                "step": float(step),
                "loss": loss_value,
                "data_objective": float(data_objective.detach().cpu().item()),
                "kl_per_anchor": float(kl.detach().cpu().item()),
                "beta_kl": float(beta_kl),
                "trace_penalty": float(trace_penalty.detach().cpu().item()),
                "mean_n_inliers": float(reward_diag["mean_n_inliers"].detach().cpu().item()),
                "fallback_count": float(reward_diag["fallback_count"].detach().cpu().item()),
            }
            row.update(moment_diag)
            row.update(eval_diag)
            history.append(row)
    if str(config.restore_state) == "best_loss" and best_state is not None:
        U_raw.data.copy_(best_state["U_raw"])
        R_mu.data.copy_(best_state["R_mu"])
        R_log_std.data.copy_(best_state["R_log_std"])
    elif str(config.restore_state) == "best_direct" and best_direct_state is not None:
        U_raw.data.copy_(best_direct_state["U_raw"])
        R_mu.data.copy_(best_direct_state["R_mu"])
        R_log_std.data.copy_(best_direct_state["R_log_std"])
    U_final = _orthonormal_columns(U_raw).detach()
    W_mu_final, _W_cov_final, eta_mu_final, eta_var_final, moment_diag = structured_metric_w_moments_from_R(
        U_final,
        X_anchor_t,
        inducing_X,
        R_mu.detach(),
        R_log_std.detach(),
        lengthscale=w_lengthscale,
        signal_var=w_signal_var,
        include_conditional_residual=bool(config.include_conditional_residual),
        trace_target=trace_target,
        trace_normalize=bool(config.trace_normalize),
        jitter=float(config.jitter),
    )
    diagnostics = {
        "best_loss": float(best_loss),
        "trace_target": float(trace_target),
        "trace_normalize": bool(config.trace_normalize),
        "restore_state": str(config.restore_state),
        "best_direct_rmse": float(best_direct_rmse),
        "final_W_row_norm_mean": float(torch.linalg.norm(W_mu_final, dim=-1).mean().cpu().item()),
        "R_n_inducing": int(M),
        "R_w_lengthscale": float(w_lengthscale),
        "R_w_signal_var": float(w_signal_var),
        "R_std_median": float(R_log_std.detach().exp().median().cpu().item()),
        "include_conditional_residual": bool(config.include_conditional_residual),
    }
    diagnostics.update(moment_diag)
    return StructuredMetricLMJGPResult(
        U=U_final,
        R_mu=R_mu.detach().clone(),
        R_log_std=R_log_std.detach().clone(),
        inducing_X=inducing_X.detach().clone(),
        eta_mu=eta_mu_final.detach().clone(),
        eta_log_std=eta_var_final.clamp_min(1e-12).sqrt().log().detach().clone(),
        w_lengthscale=float(w_lengthscale),
        w_signal_var=float(w_signal_var),
        include_conditional_residual=bool(config.include_conditional_residual),
        history=history,
        train_sec=float(time.perf_counter() - t0),
        diagnostics=diagnostics,
    )


def _warm_start_gate_and_mu_u(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y: torch.Tensor,
    W0: torch.Tensor,
    m: int,
    *,
    rho_init: float,
    k_near: int = 5,
    score_gap: float = 4.0,
    gap_ratio: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Data-driven warm start (class-mean-difference) for the spatial gate + local outlier/inducing.

    Per anchor: (1) estimate the anchor region y-level ``y0`` from nearest neighbours;
    (2) split neighbours into two y-clusters at the largest gap (only if clearly bimodal,
    > ``gap_ratio`` x median gap; else single-region, intercept-only gate); (3) init the
    gate slope toward the inlier-cluster mean in ``z=W0(x-xa)`` space (LDA-style class-mean
    difference), 0.5 boundary at the cluster midpoint, anchor forced inlier (nu0>0);
    (4) init mu_u to y0 and the learnable outlier mean/log-var to the other cluster's
    actual mean/variance.
    Returns (gate_w0 [T,Q], rho_logit0 [T], mu_u0 [T,m], out_mean0 [T], out_log_var0 [T]).
    """
    device, dtype = X_anchor.device, X_anchor.dtype
    T, n, _D = X_neighbors.shape
    Q = W0.shape[0]
    delta = X_neighbors - X_anchor.unsqueeze(1)
    z = torch.einsum("qd,tnd->tnq", W0, delta)
    d2 = (delta * delta).sum(-1)
    knear = max(1, min(int(k_near), n))
    near_idx = torch.topk(d2, k=knear, largest=False, dim=1).indices
    y0 = torch.gather(y, 1, near_idx).median(dim=1).values
    gate_w = torch.zeros((T, Q), device=device, dtype=dtype)
    nu0 = torch.full((T,), math.log(rho_init / (1.0 - rho_init)), device=device, dtype=dtype)
    out_mean0 = y.mean(dim=1).clone()
    out_log_var0 = y.var(dim=1, unbiased=False).clamp_min(1e-4).log().clone()
    for t in range(T):
        yt = y[t]
        if n < 2:
            continue
        s, _ = torch.sort(yt)
        gaps = s[1:] - s[:-1]
        gi = int(torch.argmax(gaps).item())
        max_gap = float(gaps[gi].item())
        med_gap = float(gaps.median().item()) + 1e-12
        if max_gap < float(gap_ratio) * med_gap:
            continue
        thr = 0.5 * (s[gi] + s[gi + 1])
        low = yt < thr
        inlier = low if bool((y0[t] < thr).item()) else (~low)
        n_in = int(inlier.sum().item())
        if n_in == 0 or n_in == n:
            continue
        mu_in = z[t][inlier].mean(0)
        mu_out = z[t][~inlier].mean(0)
        dvec = mu_in - mu_out
        norm2 = float((dvec * dvec).sum().item()) + 1e-8
        omega = (float(score_gap) / norm2) * dvec
        z_b = 0.5 * (mu_in + mu_out)
        b0 = -float((omega * z_b).sum().item())
        gate_w[t] = omega
        nu0[t] = max(b0, 0.5)
        y_out = yt[~inlier]
        out_mean0[t] = y_out.mean()
        if y_out.numel() > 1:
            out_log_var0[t] = y_out.var(unbiased=False).clamp_min(1e-4).log()
    mu_u0 = y0.unsqueeze(1).expand(T, int(m)).contiguous()
    return gate_w, nu0, mu_u0, out_mean0, out_log_var0


def train_analytic_structured_metric_lmjgp(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    C: torch.Tensor | np.ndarray,
    *,
    init_W: torch.Tensor | np.ndarray,
    init_mu_u: torch.Tensor | np.ndarray | None = None,
    init_U: torch.Tensor | np.ndarray | None = None,
    init_R_mu: torch.Tensor | np.ndarray | None = None,
    init_R_log_std: torch.Tensor | np.ndarray | None = None,
    init_V_mu: torch.Tensor | np.ndarray | None = None,
    init_V_log_std: torch.Tensor | np.ndarray | None = None,
    global_inducing_X: torch.Tensor | np.ndarray | None = None,
    X_inducing_raw: torch.Tensor | np.ndarray | None = None,
    fixed_rho: torch.Tensor | np.ndarray | None = None,
    init_rho_logit: torch.Tensor | np.ndarray | None = None,
    init_gate_w: torch.Tensor | np.ndarray | None = None,
    config: AnalyticStructuredMetricLMJGPConfig,
    eval_callback: AnalyticEvalCallback | None = None,
) -> AnalyticStructuredMetricLMJGPResult:
    """Train analytic ISM-LMJGP using marginalised W and fixed local inducing locations.

    ``init_U`` / ``init_R_mu`` / ``init_R_log_std`` allow **warm-starting** the global
    metric layer from a previously trained model (e.g. for dynamic/append-only test
    sets). ``init_R_*`` must have shape ``[M, Q]`` matching the inducing set; a natural
    choice is the previous model's eta-GP posterior evaluated at this run's inducing
    inputs (``structured_metric_eta_moments_from_R``). Defaults (None) reproduce the
    cold ``init_W``-derived initialisation.
    """
    t0 = time.perf_counter()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    C_t = _as_float_tensor(C, device=device).to(dtype=dtype)
    reproject = bool(config.reproject_local_inducing) and X_inducing_raw is not None
    learn_local_inducing = bool(getattr(config, "learn_local_inducing", False)) and not reproject
    if learn_local_inducing:
        # ABLATION: learn the local per-anchor GP inducing LOCATIONS C [T, m, Q] in the
        # projected z-space. C feeds Kfu / KufKfu / Kuu; all are differentiable in C.
        C_t = torch.nn.Parameter(C_t.clone())
    X_ind_raw_t = (
        _as_float_tensor(X_inducing_raw, device=device).to(dtype=dtype)
        if X_inducing_raw is not None else None
    )
    fixed_rho_t = (
        _as_float_tensor(fixed_rho, device=device).to(dtype=dtype)
        if fixed_rho is not None else None
    )
    # Per-anchor local-GP prior mean c (nearest-neighbour y level) for local_mean_mode="inlier".
    _mean_mode = str(getattr(config, "local_mean_mode", "zero"))
    profile_mean = _mean_mode == "profile"
    if profile_mean and bool(getattr(config, "collapse_u", False)):
        # b1 would have to be built from y - mu_j while mu_j is built from the resulting
        # q*(u): a genuine fixed point. Not implemented rather than silently approximated.
        raise NotImplementedError(
            'local_mean_mode="profile" is not supported together with collapse_u=True '
            "(the profiled intercept and the closed-form q*(u) are mutually circular)."
        )
    if profile_mean and bool(getattr(config, "two_expert", False)):
        # Expert B does get its own profiled intercept in the ELBO, but predict has no
        # field to carry it, so the two would silently disagree.
        raise NotImplementedError(
            'local_mean_mode="profile" is not supported together with two_expert=True '
            "(expert B's intercept is not exported to predict)."
        )
    if _mean_mode == "inlier":
        with torch.no_grad():
            _d2x = (X_neighbors_t - X_anchor_t.unsqueeze(1)).pow(2).sum(-1)
            _kn = max(1, min(5, X_neighbors_t.shape[1]))
            _ni = torch.topk(_d2x, k=_kn, largest=False, dim=1).indices
            c_local = torch.gather(y_t, 1, _ni).median(dim=1).values
    else:
        c_local = None
    T, _n, _D = X_neighbors_t.shape
    m = C_t.shape[1]
    W0 = _as_float_tensor(init_W, device=device).to(dtype=dtype)
    Q = W0.shape[0]
    trace_target = float(Q if config.trace_target is None else config.trace_target)
    U0, eta0, eta_log_std0 = init_structured_metric_from_w0(
        W0,
        n_anchors=T,
        trace_target=trace_target,
        eta_init_log_std=float(config.eta_init_log_std),
    )
    if global_inducing_X is not None:
        # User-provided W-field inducing locations (same standardized-X space as
        # X_anchor); overrides n_inducing_R selection.
        inducing_X = _as_float_tensor(global_inducing_X, device=device).to(dtype=dtype)
    else:
        inducing_X = _select_inducing_X(X_anchor_t, config.n_inducing_R).to(device=device, dtype=dtype)
    if bool(getattr(config, "learn_inducing_R", False)):
        # ABLATION: learn the global W-field inducing LOCATIONS. Standard sparse-GP
        # inducing-input optimization; the free-W KL / moments already take inducing_X
        # as an argument so the gradient flows with no other change.
        inducing_X = torch.nn.Parameter(inducing_X.clone())
    M = int(inducing_X.shape[0])
    # Non-zero prior mean: eta = eta0 + g, g ~ GP(0, Kuu). The GP machinery runs
    # zero-mean on the deviation g (R = inducing values of g), so R inits at 0 and
    # the KL shrinks g -> 0 (eta -> eta0). "zero" keeps the original eta0 init with
    # a zero offset (KL shrinks eta -> 0).
    if str(config.eta_prior_mean_mode) == "init":
        eta_mean_offset = eta0[0].detach().clone()
        R_mu_default = torch.zeros((M, Q), device=device, dtype=dtype)
    else:
        eta_mean_offset = None
        R_mu_default = eta0[:1].expand(M, Q).contiguous()
    R_mu0 = (R_mu_default if init_R_mu is None
             else _as_float_tensor(init_R_mu, device=device).to(dtype=dtype))
    R_log_std0 = (eta_log_std0[:1].expand(M, Q).contiguous() if init_R_log_std is None
                  else _as_float_tensor(init_R_log_std, device=device).to(dtype=dtype))
    R_mu = torch.nn.Parameter(R_mu0.to(device=device, dtype=dtype).clone())
    R_log_std = torch.nn.Parameter(R_log_std0.to(device=device, dtype=dtype).clone())
    R_log_std.requires_grad_(bool(config.train_eta_log_std))
    U_init = U0 if init_U is None else _as_float_tensor(init_U, device=device).to(dtype=dtype)
    U_raw = torch.nn.Parameter(U_init.to(device=device, dtype=dtype).clone())
    # Free entrywise W family: deviation inducing values V (W(x) = W0 + A(x) V).
    free_w = str(getattr(config, "w_family", "structured")) == "free"
    D_x = int(W0.shape[1])
    V_mu = torch.nn.Parameter(torch.zeros((M, Q, D_x), device=device, dtype=dtype))
    V_log_std = torch.nn.Parameter(torch.full(
        (M, Q, D_x), float(config.eta_init_log_std), device=device, dtype=dtype))
    # Warm-start the free deviation field (e.g. the previous run's posterior read out
    # at THIS run's inducing set) -- the free-family analogue of init_R_mu.
    with torch.no_grad():
        if init_V_mu is not None:
            V_mu.data.copy_(_as_float_tensor(init_V_mu, device=device).to(dtype=dtype))
        if init_V_log_std is not None:
            V_log_std.data.copy_(_as_float_tensor(init_V_log_std, device=device).to(dtype=dtype))
    if init_mu_u is None:
        mu_u0 = torch.zeros((T, m), device=device, dtype=dtype)
    else:
        mu_u0 = _as_float_tensor(init_mu_u, device=device).to(dtype=dtype)
    mu_u = torch.nn.Parameter(mu_u0.clone())
    u_log_std = torch.nn.Parameter(torch.full((T, m), -2.0, device=device, dtype=dtype))
    noise0 = max(float(config.obs_noise_init) ** 2, 1e-8)
    noise_max_logvar = math.log(max(float(config.obs_noise_max), 1e-3) ** 2)
    if str(config.obs_noise_prior_mode) == "data_driven":
        # Per-anchor nearest-neighbour-difference (Rice) estimate of the local
        # residual scale in the W0-projected neighbour space; this reflects
        # out-of-sample residuals, unlike the in-sample local-GP fit which the
        # model interpolates to ~0.
        with torch.no_grad():
            z_nb = torch.einsum("qd,tnd->tnq", W0, X_neighbors_t)
            d2_nb = (z_nb.unsqueeze(2) - z_nb.unsqueeze(1)).pow(2).sum(-1)
            big = torch.eye(z_nb.shape[1], device=device, dtype=dtype).unsqueeze(0) * 1e12
            nn_idx = (d2_nb + big).argmin(dim=2)
            y_nn = torch.gather(y_t, 1, nn_idx)
            sq_nn = 0.5 * (y_t - y_nn).pow(2)                 # [T,n] per-neighbour Rice term
            if bool(getattr(config, "obs_noise_robust", False)):
                # Robust (median) Rice estimate: cross-region neighbour pairs produce
                # huge jump-sized diffs that inflate the MEAN; the median (for mild
                # neighbourhoods with <50% cross-region) lands on within-region pairs,
                # so the noise prior reflects within-region noise, not the jump.
                sig_t = sq_nn.median(dim=1).values.clamp_min(1e-8).sqrt()
            else:
                sig_t = sq_nn.mean(dim=1).clamp_min(1e-8).sqrt()
            sig_t = (sig_t * float(config.obs_noise_data_driven_frac)).clamp(
                float(config.obs_noise_data_driven_floor), float(config.obs_noise_max)
            )
        prior_log_noise = sig_t.pow(2).log().to(device=device, dtype=dtype)
    else:
        prior_log_noise = torch.full((T,), math.log(noise0), device=device, dtype=dtype)
    init_log_noise = prior_log_noise.clone() if bool(config.init_noise_at_prior) else torch.full(
        (T,), math.log(noise0), device=device, dtype=dtype
    )
    log_noise_var = torch.nn.Parameter(init_log_noise.clone())
    rho0 = min(max(float(config.rho_init), 1e-4), 1.0 - 1e-4)
    rho_logit = torch.nn.Parameter(
        torch.full((T,), math.log(rho0 / (1.0 - rho0)), device=device, dtype=dtype)
    )
    # Spatial-gate slopes (per-anchor w_t in R^Q). Zero-init => the spatial gate
    # equals the intercept-only gate at step 0, so the only difference from the
    # baseline is what the slopes learn. Always allocated for a consistent state
    # dict; only optimised when gate_mode == "spatial".
    gate_w = torch.nn.Parameter(torch.zeros((T, Q), device=device, dtype=dtype))
    spatial_gate = str(config.gate_mode) == "spatial"
    # Learnable outlier component (default OFF -> _analytic_local_terms uses the fixed
    # broad data-statistic background). Init mean at the neighbourhood mean, log-var at
    # the broad default (outlier_scale_mult^2 * local var).
    learn_outlier = bool(getattr(config, "learn_outlier", False))
    with torch.no_grad():
        data_local_mean = y_t.mean(dim=1)
        data_local_var = y_t.var(dim=1, unbiased=False).clamp_min(1e-4)
    out_log_var_prior = (float(config.outlier_scale_mult) ** 2 * data_local_var).log()
    out_mean = torch.nn.Parameter(data_local_mean.clone())
    out_log_var = torch.nn.Parameter(out_log_var_prior.clone())
    if bool(getattr(config, "gate_warm_start", False)) and spatial_gate:
        with torch.no_grad():
            gw0, nu0_ws, muu0_ws, outm0_ws, outlv0_ws = _warm_start_gate_and_mu_u(
                X_anchor_t, X_neighbors_t, y_t, W0, m, rho_init=rho0)
            gate_w.data.copy_(gw0)
            rho_logit.data.copy_(nu0_ws)
            mu_u.data.copy_(muu0_ws)
            out_mean.data.copy_(outm0_ws)
            if learn_outlier:
                out_log_var.data.copy_(outlv0_ws)
                # prior target = data-driven other-cluster spread (not the broad default),
                # so the prior keeps the outlier tight on the other cluster, not flat.
                out_log_var_prior = outlv0_ws.clone()
    # Second-expert (B) local GP: own inducing values / log-std / noise, shared C.
    # Init mu_u_B at the neighbourhood mean (different from A's nearest-neighbour init
    # => symmetry broken so the gate can separate); warm-start overrides with the
    # other-cluster level. Default OFF.
    two_expert = bool(getattr(config, "two_expert", False))
    mu_u_B = torch.nn.Parameter(data_local_mean.view(T, 1).expand(T, m).contiguous().clone())
    u_log_std_B = torch.nn.Parameter(torch.full((T, m), -2.0, device=device, dtype=dtype))
    log_noise_var_B = torch.nn.Parameter(init_log_noise.clone())
    if two_expert and bool(getattr(config, "gate_warm_start", False)) and spatial_gate:
        with torch.no_grad():
            mu_u_B.data.copy_(outm0_ws.view(T, 1).expand(T, m).contiguous())
    # External gate init (e.g. seeded from a JumpGP CEM solve): overrides the warm-start
    # / zero init. rho_logit is the per-anchor intercept nu_0, gate_w the slope omega.
    with torch.no_grad():
        if init_rho_logit is not None:
            rho_logit.data.copy_(
                _as_float_tensor(init_rho_logit, device=device).to(dtype=dtype).reshape(T))
        if init_gate_w is not None and spatial_gate:
            gate_w.data.copy_(
                _as_float_tensor(init_gate_w, device=device).to(dtype=dtype).reshape(T, Q))
    # Gate GP prior: omega / nu0 become smooth fields over anchor locations, read out
    # by kriging from inducing values on the eta-GP inducing inputs. Only meaningful in
    # spatial mode (intercept mode has no slopes to smooth).
    gate_gp = bool(getattr(config, "gate_gp", False)) and spatial_gate
    rho0_logit_const = math.log(rho0 / (1.0 - rho0))
    if gate_gp:
        gate_ls = (
            float(config.gate_gp_lengthscale)
            if getattr(config, "gate_gp_lengthscale", None) is not None
            else _default_sparse_gp_lengthscale(X_anchor_t)
        )
        with torch.no_grad():
            _d2_zz = (inducing_X.unsqueeze(1) - inducing_X.unsqueeze(0)).pow(2).sum(-1)
            Kzz_gate = torch.exp(-0.5 * _d2_zz / gate_ls**2) + float(config.jitter) * torch.eye(
                M, device=device, dtype=dtype)
            _d2_xz = (X_anchor_t.unsqueeze(1) - inducing_X.unsqueeze(0)).pow(2).sum(-1)
            Kxz_gate = torch.exp(-0.5 * _d2_xz / gate_ls**2)
            Kzz_gate_inv = torch.cholesky_inverse(torch.linalg.cholesky(Kzz_gate))
            A_gate = Kxz_gate @ Kzz_gate_inv  # [T, M] kriging readout weights
        gate_w_ind = torch.nn.Parameter(torch.zeros((M, Q), device=device, dtype=dtype))
        nu0_ind = torch.nn.Parameter(torch.zeros((M,), device=device, dtype=dtype))
        # Project any warm-start / external per-anchor gate init onto the inducing
        # values (GP-prior-regularized least squares so A v ~ per-anchor target).
        with torch.no_grad():
            _has_gate_init = bool(gate_w.abs().max() > 0) or bool(
                (rho_logit - rho0_logit_const).abs().max() > 1e-8)
            if _has_gate_init:
                _AtA = A_gate.T @ A_gate + 1e-2 * Kzz_gate_inv + 1e-8 * torch.eye(
                    M, device=device, dtype=dtype)
                _L_p = torch.linalg.cholesky(0.5 * (_AtA + _AtA.T))
                gate_w_ind.data.copy_(torch.cholesky_solve(A_gate.T @ gate_w.data, _L_p))
                nu0_ind.data.copy_(torch.cholesky_solve(
                    (A_gate.T @ (rho_logit.data - rho0_logit_const)).unsqueeze(-1), _L_p
                ).squeeze(-1))
    else:
        gate_w_ind = None
        nu0_ind = None
        A_gate = None
        Kzz_gate_inv = None
    # 4A: collapse local q(u) -> closed-form (mu*, Sigma*) given a persistent detached
    # responsibility buffer (init 1 = pure-inlier warmup). mu_u / u_log_std stay as
    # NON-optimized buffers mirroring the latest collapsed moments (for eval/result).
    collapse_u = bool(getattr(config, "collapse_u", False))
    if collapse_u and two_expert:
        raise ValueError("collapse_u is not supported together with two_expert.")
    if collapse_u:
        resp_buf = torch.ones((T, _n), device=device, dtype=dtype)
        Sigma_u_buf = torch.zeros((T, m, m), device=device, dtype=dtype)
        resp_damp = float(getattr(config, "collapse_u_resp_damping", 0.5))
        resp_blend = float(getattr(config, "collapse_u_resp_blend", 1.0))
    else:
        resp_buf = None
        Sigma_u_buf = None
    # ABLATION #1: per-anchor local-GP lengthscale with an Otsu-driven floor.
    local_ell_floor = bool(getattr(config, "local_ell_floor", False))
    if local_ell_floor:
        with torch.no_grad():
            otsu_t = _otsu_separability(y_t).clamp(0.0, 1.0)  # [T] detached jump signal
        ell_lo = float(getattr(config, "local_ell_lo", 1.0))
        ell_hi = float(getattr(config, "local_ell_hi", 3.0))
        ell_min_t = (ell_lo + (ell_hi - ell_lo) * otsu_t).to(device=device, dtype=dtype)  # [T]
        raw_ell = torch.nn.Parameter(torch.zeros((T,), device=device, dtype=dtype))
    else:
        ell_min_t = None
        raw_ell = None
    freeze_gate = bool(getattr(config, "freeze_gate", False)) and spatial_gate
    fix_omega = bool(getattr(config, "gate_fix_omega", False)) and spatial_gate
    gate_nu0_nonneg = bool(getattr(config, "gate_nu0_nonneg", False))
    outlier_floor = bool(getattr(config, "outlier_floor", False))
    outlier_floor_k = float(getattr(config, "outlier_floor_k", 2.5))
    uniform_outlier = bool(getattr(config, "uniform_outlier", False))
    freeze_w_field = bool(getattr(config, "freeze_w_field", False))
    if freeze_w_field:
        params: list[torch.nn.Parameter] = [log_noise_var]   # W field stays at init
    elif free_w:
        params = [V_mu, V_log_std, log_noise_var]
    else:
        params = [U_raw, R_mu, log_noise_var]
    if not collapse_u:
        params.extend([mu_u, u_log_std])    # collapse_u: (mu*, Sigma*) are closed-form
    if not freeze_gate:
        if gate_gp:
            params.append(nu0_ind)          # gate GP: optimize the inducing values instead
            if not fix_omega:
                params.append(gate_w_ind)
        else:
            params.append(rho_logit)        # in spatial mode rho_logit is the gate intercept nu0
            if spatial_gate and not fix_omega:
                params.append(gate_w)       # fix_omega: keep nu0 learnable, freeze slope omega
    if learn_outlier:
        params.extend([out_mean, out_log_var])
    if two_expert:
        params.extend([mu_u_B, u_log_std_B, log_noise_var_B])
    if local_ell_floor:
        params.append(raw_ell)
    if not freeze_w_field and not free_w and R_log_std.requires_grad:
        params.append(R_log_std)
    if isinstance(inducing_X, torch.nn.Parameter):
        params.append(inducing_X)       # learn_inducing_R: global W-field locations
    if learn_local_inducing:
        params.append(C_t)              # learn_local_inducing: local GP locations
    opt = torch.optim.Adam(params, lr=float(config.lr))
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(config.seed))
    w_lengthscale = (
        _default_sparse_gp_lengthscale(X_anchor_t)
        if config.w_lengthscale is None
        else float(config.w_lengthscale)
    )
    w_signal_var = float(config.w_signal_var) * float(config.eta_prior_std) ** 2
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_direct_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_direct_state: dict[str, torch.Tensor] | None = None
    prior_rho_logit = torch.tensor(math.log(rho0 / (1.0 - rho0)), device=device, dtype=dtype)

    def _param_snapshot() -> dict[str, torch.Tensor]:
        snap = {
            "U_raw": U_raw,
            "R_mu": R_mu,
            "R_log_std": R_log_std,
            "mu_u": mu_u,
            "u_log_std": u_log_std,
            "log_noise_var": log_noise_var,
            "rho_logit": rho_logit,
            "gate_w": gate_w,
        }
        if gate_gp:
            snap["gate_w_ind"] = gate_w_ind
            snap["nu0_ind"] = nu0_ind
        if free_w:
            snap["V_mu"] = V_mu
            snap["V_log_std"] = V_log_std
        return {name: t.detach().clone() for name, t in snap.items()}

    _t0_temp = float(getattr(config, "gate_temp_init", 1.0))
    _t1_temp = float(getattr(config, "gate_temp_final", 1.0))
    for step in range(1, int(config.steps) + 1):
        # geometric annealing of the gate temperature from init -> final
        _frac = (step - 1) / max(1, int(config.steps) - 1)
        _gate_temp_step = float(_t0_temp * (_t1_temp / _t0_temp) ** _frac) if _t0_temp > 0 and _t1_temp > 0 else 1.0
        opt.zero_grad()
        B = min(int(config.batch_anchors), T)
        idx = torch.randperm(T, device=device, generator=gen)[:B]
        if free_w:
            W_mu_b, W_cov_b = _free_w_moments(
                X_anchor_t[idx], inducing_X, W0, V_mu, V_log_std,
                lengthscale=w_lengthscale, signal_var=w_signal_var,
                include_conditional_residual=bool(config.include_conditional_residual),
                jitter=float(config.jitter))
        else:
            U = _orthonormal_columns(U_raw)
            W_mu_b, W_cov_b, _eta_mu_b, _eta_var_b, _moment_diag_b = structured_metric_w_moments_from_R(
                U,
                X_anchor_t[idx],
                inducing_X,
                R_mu,
                R_log_std,
                lengthscale=w_lengthscale,
                signal_var=w_signal_var,
                include_conditional_residual=bool(config.include_conditional_residual),
                trace_target=trace_target,
                trace_normalize=bool(config.trace_normalize),
                jitter=float(config.jitter),
                eta_mean_offset=eta_mean_offset,
            )
        C_b = (
            torch.einsum("tqd,tmd->tmq", W_mu_b.detach(), X_ind_raw_t[idx])
            if reproject else C_t[idx]
        )
        ell_loc_b = (ell_min_t[idx] + torch.nn.functional.softplus(raw_ell[idx])
                     if local_ell_floor else None)
        if gate_gp:
            gate_w_b = A_gate[idx] @ gate_w_ind
            rho_logit_b = rho0_logit_const + A_gate[idx] @ nu0_ind
        else:
            gate_w_b = gate_w[idx]
            rho_logit_b = rho_logit[idx]
        if collapse_u:
            resp_b = resp_buf[idx]
            if resp_blend < 1.0:
                resp_b = (1.0 - resp_blend) + resp_blend * resp_b
        else:
            resp_b = None
        region_elbo_b, kl_u_b, local_diag = _analytic_local_terms(
            X_anchor_t[idx],
            X_neighbors_t[idx],
            y_t[idx],
            C_b,
            W_mu_b,
            W_cov_b,
            mu_u[idx],
            u_log_std[idx],
            log_noise_var[idx],
            rho_logit_b,
            local_signal_var=float(config.local_signal_var),
            outlier_scale_mult=float(config.outlier_scale_mult),
            jitter=float(config.jitter),
            gate_w=gate_w_b,
            gate_mode=str(config.gate_mode),
            gate_gh_points=int(config.gate_gh_points),
            out_mean=(out_mean[idx] if learn_outlier else None),
            out_log_var=(out_log_var[idx] if learn_outlier else None),
            fixed_rho=(fixed_rho_t[idx] if fixed_rho_t is not None else None),
            local_prior_mean=(c_local[idx] if c_local is not None else None),
            profile_mean=profile_mean,
            gate_temp=_gate_temp_step,
            adaptive_temp_min=(float(config.gate_temp_min)
                               if bool(getattr(config, "gate_temp_adaptive", False)) else None),
            adaptive_signal=str(getattr(config, "gate_temp_adaptive_signal", "entropy")),
            mu_u_B=(mu_u_B[idx] if two_expert else None),
            u_log_std_B=(u_log_std_B[idx] if two_expert else None),
            log_noise_var_B=(log_noise_var_B[idx] if two_expert else None),
            gate_nu0_nonneg=gate_nu0_nonneg,
            outlier_floor=outlier_floor,
            uniform_outlier=uniform_outlier,
            outlier_floor_k=outlier_floor_k,
            ell_loc=ell_loc_b,
            collapse_resp=resp_b,
        )
        if collapse_u:
            # EM-style refresh: damped responsibility-buffer update + mirror the
            # collapsed q*(u) moments into the (non-optimized) mu_u / u_log_std /
            # Sigma_u buffers so eval callbacks and the final result stay meaningful.
            with torch.no_grad():
                resp_buf[idx] = (1.0 - resp_damp) * resp_buf[idx] + resp_damp * local_diag["resp_new"]
                mu_u.data[idx] = local_diag["mu_u_star"]
                u_log_std.data[idx] = 0.5 * torch.log(
                    torch.diagonal(local_diag["Sigma_u_star"], dim1=-2, dim2=-1).clamp_min(1e-12))
                Sigma_u_buf[idx] = local_diag["Sigma_u_star"]
        if two_expert:
            kl_u_B_b = _local_kl_diag_u(
                C_b, mu_u_B[idx], torch.exp(2.0 * u_log_std_B[idx]).clamp_min(1e-12),
                jitter=float(config.jitter), signal_var=float(config.local_signal_var))
            kl_u_b = kl_u_b + kl_u_B_b  # symmetric KL for the second expert
        data_objective = region_elbo_b.mean()
        kl_u_mean = kl_u_b.mean()
        beta_R = float(config.beta_kl_R) * min(1.0, step / max(1, int(config.kl_warmup_steps)))
        if free_w:
            kl_R = _free_w_kl(
                inducing_X, V_mu, V_log_std,
                lengthscale=w_lengthscale, signal_var=w_signal_var,
                jitter=float(config.jitter)) / float(T)
            trace_penalty = kl_R.new_zeros(())  # trace normalization is a structured-family concept
        else:
            kl_R = _diag_sparse_gp_kl(
                inducing_X,
                R_mu.unsqueeze(-1),
                R_log_std.unsqueeze(-1),
                lengthscale=w_lengthscale,
                signal_var=w_signal_var,
                jitter=float(config.jitter),
            ) / float(T)
            eta_mu_all, eta_var_all = structured_metric_eta_moments_from_R(
                X_anchor_t,
                inducing_X,
                R_mu,
                R_log_std,
                lengthscale=w_lengthscale,
                signal_var=w_signal_var,
                include_conditional_residual=bool(config.include_conditional_residual),
                jitter=float(config.jitter),
                eta_mean_offset=eta_mean_offset,
            )
            trace_penalty = (_eta_trace_from_var(eta_mu_all, eta_var_all) - trace_target).pow(2).mean()
        beta_u = float(config.beta_kl_u)
        noise_prior = ((log_noise_var[idx] - prior_log_noise[idx]) / max(float(config.obs_noise_prior_std), 1e-8)).pow(2).mean()
        rho = torch.sigmoid(rho_logit_b).clamp(1e-4, 1.0 - 1e-4)
        # In spatial mode rho_logit is the per-anchor intercept nu_0; the floor
        # penalty (which keeps the intercept-only rho from collapsing) would fight
        # the learned spatial slopes, so it is dropped. The intercept prior is
        # kept (anchors nu_0 near the rho_init level) and the slopes get a light
        # L2 ridge instead. With gate_gp, both the intercept deviation and the
        # slopes are governed by the GP quadratic prior on the inducing values.
        if gate_gp:
            rho_prior = rho.new_zeros(())
            min_rho_penalty = rho.new_zeros(())
            quad_w = torch.einsum("mq,mk,kq->", gate_w_ind, Kzz_gate_inv, gate_w_ind)
            quad_nu = torch.einsum("m,mk,k->", nu0_ind, Kzz_gate_inv, nu0_ind)
            gate_w_penalty = float(config.gate_gp_prior_weight) * 0.5 * (
                quad_w / float(config.gate_gp_slope_var)
                + quad_nu / float(config.gate_gp_nu0_var)
            ) / float(T)
        elif spatial_gate:
            rho_prior = (rho_logit[idx] - prior_rho_logit).pow(2).mean()
            min_rho_penalty = rho.new_zeros(())
            gate_w_penalty = float(config.gate_w_l2) * gate_w[idx].pow(2).mean()
        else:
            rho_prior = (rho_logit[idx] - prior_rho_logit).pow(2).mean()
            min_rho_penalty = torch.relu(float(config.min_rho_mean) - rho.mean()).pow(2)
            gate_w_penalty = rho.new_zeros(())
        # Keep the learnable outlier variance broad-ish (toward the data default) so it
        # cannot shrink into a spike; the hard floor at noise_var is enforced in-likelihood.
        outlier_var_prior = (
            (out_log_var[idx] - out_log_var_prior[idx]).pow(2).mean()
            if learn_outlier else rho.new_zeros(())
        )
        loss = (
            -data_objective
            + beta_R * kl_R
            + beta_u * kl_u_mean
            + float(config.obs_noise_prior_weight) * noise_prior
            + float(config.rho_prior_strength) * rho_prior
            + float(config.min_rho_penalty_weight) * min_rho_penalty
            + gate_w_penalty
            + float(config.outlier_var_prior_weight) * outlier_var_prior
            + float(config.trace_penalty_weight) * trace_penalty
        )
        gate_sup_w = float(getattr(config, "gate_supervise_weight", 0.0))
        if gate_sup_w > 0.0 and local_diag.get("gate_score") is not None:
            # EM-style M-step for the gate: fit g(z_i) to the likelihood-only responsibility.
            r_target = torch.sigmoid(local_diag["pure_lik_logit"])
            gate_sup = torch.nn.functional.binary_cross_entropy_with_logits(
                local_diag["gate_score"], r_target)
            loss = loss + gate_sup_w * gate_sup
        loss.backward()
        if float(config.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
        opt.step()
        with torch.no_grad():
            R_log_std.clamp_(float(config.r_log_std_min), 2.0)
            V_log_std.clamp_(float(config.r_log_std_min), 2.0)
            u_log_std.clamp_(-8.0, 2.0)
            log_noise_var.clamp_(math.log(1e-6), noise_max_logvar)
            if learn_outlier:
                out_log_var.clamp_(math.log(1e-6), float(out_log_var_prior.max().item()) + 4.0)
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = _param_snapshot()
        if step == 1 or step % int(config.log_interval) == 0 or step == int(config.steps):
            with torch.no_grad():
                if free_w:
                    U_eval = _orthonormal_columns(U_raw)
                    W_mu_all, W_cov_all = _free_w_moments(
                        X_anchor_t, inducing_X, W0, V_mu, V_log_std,
                        lengthscale=w_lengthscale, signal_var=w_signal_var,
                        include_conditional_residual=bool(config.include_conditional_residual),
                        jitter=float(config.jitter))
                    moment_diag = {"free_w_drift": float(
                        (W_mu_all - W0.unsqueeze(0)).norm(dim=(1, 2)).mean().cpu().item())}
                else:
                    U_eval = _orthonormal_columns(U_raw)
                    W_mu_all, W_cov_all, _eta_mu_eval, _eta_var_eval, moment_diag = structured_metric_w_moments_from_R(
                        U_eval,
                        X_anchor_t,
                        inducing_X,
                        R_mu,
                        R_log_std,
                        lengthscale=w_lengthscale,
                        signal_var=w_signal_var,
                        include_conditional_residual=bool(config.include_conditional_residual),
                        trace_target=trace_target,
                        trace_normalize=bool(config.trace_normalize),
                        jitter=float(config.jitter),
                        eta_mean_offset=eta_mean_offset,
                    )
            eval_diag: dict[str, float] = {}
            if eval_callback is not None and (
                step == 1 or step % max(1, int(config.eval_interval)) == 0 or step == int(config.steps)
            ):
                eval_diag = eval_callback(
                    step,
                    {"W_mu_anchor": W_mu_all.detach(), "W_cov_anchor": W_cov_all.detach()},
                    {
                        "U": U_eval.detach(),
                        "R_mu": R_mu.detach(),
                        "R_log_std": R_log_std.detach(),
                        "inducing_X": inducing_X.detach(),
                        "mu_u": mu_u.detach(),
                        "u_log_std": u_log_std.detach(),
                        "log_noise_var": log_noise_var.detach(),
                        "rho_logit": rho_logit.detach(),
                        **({"Sigma_u": Sigma_u_buf.detach()} if collapse_u else {}),
                        **({"W_mu_direct": W_mu_all.detach(),
                            "W_cov_direct": W_cov_all.detach()} if free_w else {}),
                        "C": C_t.detach(),
                        "w_lengthscale": torch.as_tensor(float(w_lengthscale), device=device, dtype=dtype),
                        "w_signal_var": torch.as_tensor(float(w_signal_var), device=device, dtype=dtype),
                        "include_conditional_residual": torch.as_tensor(
                            1.0 if bool(config.include_conditional_residual) else 0.0,
                            device=device,
                            dtype=dtype,
                        ),
                    },
                    moment_diag,
                )
                if "direct_eval_rmse" in eval_diag:
                    direct_rmse = float(eval_diag["direct_eval_rmse"])
                    if direct_rmse < best_direct_rmse:
                        best_direct_rmse = direct_rmse
                        best_direct_state = _param_snapshot()
            row = {
                "step": float(step),
                "loss": loss_value,
                "data_objective": float(data_objective.detach().cpu().item()),
                "kl_R_per_anchor": float(kl_R.detach().cpu().item()),
                "beta_kl_R": float(beta_R),
                "kl_u_mean": float(local_diag["kl_u_mean"].cpu().item()),
                "noise_prior": float(noise_prior.detach().cpu().item()),
                "rho_prior": float(rho_prior.detach().cpu().item()),
                "min_rho_penalty": float(min_rho_penalty.detach().cpu().item()),
                "trace_penalty": float(trace_penalty.detach().cpu().item()),
                "rho_mean": float(local_diag["rho_mean"].cpu().item()),
                "rho_within_std": float(local_diag["rho_within_std"].cpu().item()),
                "noise_std_mean": float(local_diag["noise_std_mean"].cpu().item()),
            }
            row.update(moment_diag)
            row.update(eval_diag)
            history.append(row)
    restore = best_state if str(config.restore_state) == "best_loss" else None
    if str(config.restore_state) == "best_direct":
        restore = best_direct_state
    if restore is not None:
        U_raw.data.copy_(restore["U_raw"])
        R_mu.data.copy_(restore["R_mu"])
        R_log_std.data.copy_(restore["R_log_std"])
        mu_u.data.copy_(restore["mu_u"])
        u_log_std.data.copy_(restore["u_log_std"])
        log_noise_var.data.copy_(restore["log_noise_var"])
        rho_logit.data.copy_(restore["rho_logit"])
        if "gate_w" in restore:
            gate_w.data.copy_(restore["gate_w"])
        if gate_gp and "gate_w_ind" in restore:
            gate_w_ind.data.copy_(restore["gate_w_ind"])
            nu0_ind.data.copy_(restore["nu0_ind"])
        if free_w and "V_mu" in restore:
            V_mu.data.copy_(restore["V_mu"])
            V_log_std.data.copy_(restore["V_log_std"])
    if gate_gp:
        # Materialize the per-anchor gate readouts into the standard result fields so
        # downstream consumers (predict, enrichment eval) are agnostic to gate_gp.
        with torch.no_grad():
            gate_w.data.copy_(A_gate @ gate_w_ind)
            rho_logit.data.copy_(rho0_logit_const + A_gate @ nu0_ind)
    if local_ell_floor:
        with torch.no_grad():
            local_ell_final = (ell_min_t + torch.nn.functional.softplus(raw_ell)).detach().clone()
    else:
        local_ell_final = None
    U_final = _orthonormal_columns(U_raw).detach()
    if free_w:
        with torch.no_grad():
            _W_mu_final, _W_cov_final = _free_w_moments(
                X_anchor_t, inducing_X, W0, V_mu, V_log_std,
                lengthscale=w_lengthscale, signal_var=w_signal_var,
                include_conditional_residual=bool(config.include_conditional_residual),
                jitter=float(config.jitter))
        moment_diag = {"free_w_drift": float(
            (_W_mu_final - W0.unsqueeze(0)).norm(dim=(1, 2)).mean().cpu().item())}
    else:
        _W_mu_final, _W_cov_final, _eta_mu_final, _eta_var_final, moment_diag = structured_metric_w_moments_from_R(
            U_final,
            X_anchor_t,
            inducing_X,
            R_mu.detach(),
            R_log_std.detach(),
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            trace_target=trace_target,
            trace_normalize=bool(config.trace_normalize),
            jitter=float(config.jitter),
            eta_mean_offset=eta_mean_offset,
        )
    Sigma_u_final = None
    if collapse_u:
        # Final full-batch collapsed pass with the (restored) final parameters so the
        # exported q(u) moments exactly match the exported W / gate / noise.
        with torch.no_grad():
            C_full = (
                torch.einsum("tqd,tmd->tmq", _W_mu_final.detach(), X_ind_raw_t)
                if reproject else C_t
            )
            resp_full = resp_buf
            if resp_blend < 1.0:
                resp_full = (1.0 - resp_blend) + resp_blend * resp_full
            _, _, _diag_full = _analytic_local_terms(
                X_anchor_t,
                X_neighbors_t,
                y_t,
                C_full,
                _W_mu_final,
                _W_cov_final,
                mu_u,
                u_log_std,
                log_noise_var,
                rho_logit,
                local_signal_var=float(config.local_signal_var),
                outlier_scale_mult=float(config.outlier_scale_mult),
                jitter=float(config.jitter),
                gate_w=gate_w,
                gate_mode=str(config.gate_mode),
                gate_gh_points=int(config.gate_gh_points),
                out_mean=(out_mean if learn_outlier else None),
                out_log_var=(out_log_var if learn_outlier else None),
                fixed_rho=fixed_rho_t,
                local_prior_mean=c_local,
                profile_mean=profile_mean,
                gate_nu0_nonneg=gate_nu0_nonneg,
                outlier_floor=outlier_floor,
                uniform_outlier=uniform_outlier,
                outlier_floor_k=outlier_floor_k,
                ell_loc=local_ell_final,
                collapse_resp=resp_full,
            )
            mu_u.data.copy_(_diag_full["mu_u_star"])
            u_log_std.data.copy_(0.5 * torch.log(
                torch.diagonal(_diag_full["Sigma_u_star"], dim1=-2, dim2=-1).clamp_min(1e-12)))
            Sigma_u_final = _diag_full["Sigma_u_star"].clone()
    # Freeze the constant local mean for ALL anchors under the final parameters, so
    # predict uses exactly the intercept the final model was fitted with. "inlier" uses
    # the fixed level directly; "profile" needs one full-batch pass to solve for it.
    local_mean_final = c_local
    if profile_mean:
        with torch.no_grad():
            C_full_m = (
                torch.einsum("tqd,tmd->tmq", _W_mu_final.detach(), X_ind_raw_t)
                if reproject else C_t
            )
            _, _, _diag_mean = _analytic_local_terms(
                X_anchor_t, X_neighbors_t, y_t, C_full_m,
                _W_mu_final, _W_cov_final, mu_u, u_log_std, log_noise_var, rho_logit,
                local_signal_var=float(config.local_signal_var),
                outlier_scale_mult=float(config.outlier_scale_mult),
                jitter=float(config.jitter),
                gate_w=gate_w,
                gate_mode=str(config.gate_mode),
                gate_gh_points=int(config.gate_gh_points),
                out_mean=(out_mean if learn_outlier else None),
                out_log_var=(out_log_var if learn_outlier else None),
                fixed_rho=fixed_rho_t,
                local_prior_mean=None,
                profile_mean=True,
                gate_nu0_nonneg=gate_nu0_nonneg,
                outlier_floor=outlier_floor,
                uniform_outlier=uniform_outlier,
                outlier_floor_k=outlier_floor_k,
                ell_loc=local_ell_final,
            )
            local_mean_final = _diag_mean["local_mean_c"]
    diagnostics = {
        "best_loss": float(best_loss),
        "best_direct_rmse": float(best_direct_rmse),
        "trace_target": float(trace_target),
        "trace_normalize": bool(config.trace_normalize),
        "restore_state": str(config.restore_state),
        "R_n_inducing": int(M),
        "R_w_lengthscale": float(w_lengthscale),
        "R_w_signal_var": float(w_signal_var),
        "R_std_median": float(R_log_std.detach().exp().median().cpu().item()),
        "rho_mean": float(torch.sigmoid(rho_logit.detach()).mean().cpu().item()),
        "noise_std_mean": float(log_noise_var.detach().exp().sqrt().mean().cpu().item()),
        "noise_prior_anchor_median": float(prior_log_noise.detach().exp().sqrt().median().cpu().item()),
        "obs_noise_prior_mode": str(config.obs_noise_prior_mode),
        "local_m_inducing": int(m),
        "collapse_u": bool(collapse_u),
        "gate_gp": bool(gate_gp),
        "w_family": str(getattr(config, "w_family", "structured")),
    }
    if collapse_u:
        diagnostics["resp_buf_mean"] = float(resp_buf.mean().cpu().item())
    diagnostics.update(moment_diag)
    return AnalyticStructuredMetricLMJGPResult(
        U=U_final,
        R_mu=R_mu.detach().clone(),
        R_log_std=R_log_std.detach().clone(),
        inducing_X=inducing_X.detach().clone(),
        mu_u=mu_u.detach().clone(),
        u_log_std=u_log_std.detach().clone(),
        log_noise_var=log_noise_var.detach().clone(),
        rho_logit=rho_logit.detach().clone(),
        C=C_t.detach().clone(),
        w_lengthscale=float(w_lengthscale),
        w_signal_var=float(w_signal_var),
        local_signal_var=float(config.local_signal_var),
        include_conditional_residual=bool(config.include_conditional_residual),
        history=history,
        train_sec=float(time.perf_counter() - t0),
        diagnostics=diagnostics,
        gate_mode=str(config.gate_mode),
        gate_w=gate_w.detach().clone(),
        X_inducing_raw=(X_ind_raw_t.detach().clone() if reproject else None),
        reproject_local_inducing=bool(reproject),
        eta_mean_offset=(eta_mean_offset.detach().clone() if eta_mean_offset is not None else None),
        out_mean=(out_mean.detach().clone() if learn_outlier else None),
        out_log_var=(out_log_var.detach().clone() if learn_outlier else None),
        mu_u_B=(mu_u_B.detach().clone() if two_expert else None),
        u_log_std_B=(u_log_std_B.detach().clone() if two_expert else None),
        log_noise_var_B=(log_noise_var_B.detach().clone() if two_expert else None),
        local_ell=local_ell_final,
        Sigma_u=Sigma_u_final,
        W_mu_direct=(_W_mu_final.detach().clone() if free_w else None),
        W_cov_direct=(_W_cov_final.detach().clone() if free_w else None),
        V_mu=(V_mu.detach().clone() if free_w else None),
        V_log_std=(V_log_std.detach().clone() if free_w else None),
        W0_free=(W0.detach().clone() if free_w else None),
        local_mean_c=(local_mean_final.detach().clone() if local_mean_final is not None else None),
        local_mean_mode=_mean_mode,
    )


def free_w_moments_from_result(
    result: AnalyticStructuredMetricLMJGPResult,
    X_query: torch.Tensor | np.ndarray,
    *,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q(W) moments at arbitrary query points for a w_family="free" result."""
    Xq = _as_float_tensor(X_query, device=result.V_mu.device).to(dtype=result.V_mu.dtype)
    return _free_w_moments(
        Xq, result.inducing_X, result.W0_free, result.V_mu, result.V_log_std,
        lengthscale=float(result.w_lengthscale), signal_var=float(result.w_signal_var),
        include_conditional_residual=bool(result.include_conditional_residual),
        jitter=float(jitter))


def predict_analytic_structured_metric_lmjgp(
    X_anchor: torch.Tensor | np.ndarray,
    result: AnalyticStructuredMetricLMJGPResult,
    *,
    trace_target: float,
    trace_normalize: bool,
    mixture: bool = False,
    outlier_mean: torch.Tensor | np.ndarray | None = None,
    outlier_var: torch.Tensor | np.ndarray | None = None,
    predict_noise_var: torch.Tensor | np.ndarray | float | None = None,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Direct VI prediction from an analytic ISM-LMJGP result.

    By default this returns the *inlier* GP predictive ``N(mu_f, var_f + noise)``
    for the anchor's own response. The ``rho`` / outlier component models
    contaminated neighbours inside the local composite likelihood during
    training; mixing a broad background into the predictive of the test point's
    response only inflates the intervals (heavy over-coverage). Set
    ``mixture=True`` (and pass ``outlier_mean``/``outlier_var`` matching the
    training background) only if a contaminated predictive is explicitly wanted.
    """
    Xa = _as_float_tensor(X_anchor, device=result.U.device).to(dtype=result.U.dtype)
    sigma2 = float(getattr(result, "local_signal_var", 1.0))
    if getattr(result, "W_mu_direct", None) is not None:
        # w_family="free": transductive direct per-anchor W moments (anchors must
        # match the training anchors by index).
        W_mu = result.W_mu_direct
        W_cov = result.W_cov_direct
    else:
        W_mu, W_cov, _eta_mu, _eta_var, _moment_diag = structured_metric_w_moments_from_R(
            result.U,
            Xa,
            result.inducing_X,
            result.R_mu,
            result.R_log_std,
            lengthscale=float(result.w_lengthscale),
            signal_var=float(result.w_signal_var),
            include_conditional_residual=bool(result.include_conditional_residual),
            trace_target=float(trace_target),
            trace_normalize=bool(trace_normalize),
            jitter=float(jitter),
            eta_mean_offset=getattr(result, "eta_mean_offset", None),
        )
    if getattr(result, "reproject_local_inducing", False) and getattr(result, "X_inducing_raw", None) is not None:
        C = torch.einsum("tqd,tmd->tmq", W_mu.detach(), result.X_inducing_raw)
    else:
        C = result.C
    # ABLATION #1: apply the per-anchor local-GP lengthscale by scaling the local
    # kernel's W / C by 1 / ell_loc (anchors aligned by index; transductive setup).
    # Only the local GP is affected -- predict has no gate, so this is self-contained.
    local_ell = getattr(result, "local_ell", None)
    if local_ell is not None:
        inv_l = (1.0 / _as_float_tensor(local_ell, device=Xa.device).to(dtype=Xa.dtype)
                 .reshape(-1).clamp_min(1e-6))
        W_mu = W_mu * inv_l.view(-1, 1, 1)
        W_cov = W_cov * inv_l.view(-1, 1, 1, 1).pow(2)
        C = C * inv_l.view(-1, 1, 1)
    Xq = Xa.unsqueeze(1)
    sigma_f = torch.full((Xa.shape[0],), math.sqrt(sigma2), device=Xa.device, dtype=Xa.dtype)
    m_W = _expected_Kfu_structured(W_mu, W_cov, Xq, C, sigma_f).squeeze(1)
    S_W = _expected_KufKfu_structured(W_mu, W_cov, Xq, C, sigma_f).squeeze(1)
    m = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = sigma2 * torch.exp(-0.5 * d2) + float(jitter) * torch.eye(m, device=Xa.device, dtype=Xa.dtype).unsqueeze(0)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)
    # Constant local mean. ``dev_u`` is the inducing mean of the ZERO-mean residual
    # process: under "inlier" the exported mu_u is the inducing mean of f itself, so the
    # level must be subtracted; under "profile" mu_u is already the residual's.
    c_add = getattr(result, "local_mean_c", None)
    if c_add is not None:
        c_add = c_add.to(device=Xa.device, dtype=Xa.dtype).reshape(-1)
        dev_u = (result.mu_u - c_add.view(-1, 1)
                 if str(getattr(result, "local_mean_mode", "zero")) == "inlier"
                 else result.mu_u)
    else:
        dev_u = result.mu_u
    Sigma_u_res = getattr(result, "Sigma_u", None)
    if Sigma_u_res is not None:
        # collapse_u: exported q(u) covariance is full -- use the exact second moment.
        u_second = Sigma_u_res + torch.einsum("ti,tj->tij", dev_u, dev_u)
    else:
        u_var = torch.exp(2.0 * result.u_log_std).clamp_min(1e-12)
        u_second = torch.diag_embed(u_var) + torch.einsum("ti,tj->tij", dev_u, dev_u)
    alpha = torch.einsum("tij,tj->ti", Kuu_inv, dev_u)
    # zeta = E[f* - c]: the RESIDUAL predictive mean.
    zeta = (m_W * alpha).sum(dim=1)
    mu_f = zeta if c_add is None else (zeta + c_add)
    v1 = sigma2 - torch.einsum("tij,tij->t", Kuu_inv, S_W)
    t3 = torch.einsum("tij,tjk,tkm,tmi->t", Kuu_inv, S_W, Kuu_inv, u_second)
    # v1 + t3 = E[(f* - c)^2], so the mean square subtracted must be the RESIDUAL mean
    # zeta, NOT mu_f. A deterministic constant contributes nothing to the variance.
    var_f = (v1 + t3 - zeta.pow(2)).clamp_min(1e-10)
    noise_var = result.log_noise_var.exp().clamp_min(1e-8)
    if predict_noise_var is not None:
        # Decouple training robustness from predictive calibration: mu_f was fit under the
        # (possibly inflated) training noise that down-weights within-neighbourhood
        # contaminants -> robust mean; but the test point's OWN observation noise is the
        # small within-region scale. Use that here so intervals are not inflated by the
        # jump-contamination absorbed during training. Mean is unchanged.
        if isinstance(predict_noise_var, (float, int)):
            noise_var = torch.full_like(noise_var, float(predict_noise_var)).clamp_min(1e-8)
        else:
            noise_var = _as_float_tensor(predict_noise_var, device=Xa.device).to(
                dtype=Xa.dtype).reshape(-1).clamp_min(1e-8)
            if noise_var.numel() == 1:
                noise_var = noise_var.expand_as(mu_f)
    rho = torch.sigmoid(result.rho_logit).clamp(1e-4, 1.0 - 1e-4)
    if getattr(result, "mu_u_B", None) is not None:
        # Two-expert mixture: the outlier branch is a second local GP (expert B).
        # The test point's own A/B membership is marginalised, giving a calibrated
        # predictive: a bimodal-aware variance with the between-expert spread term
        # rho(1-rho)(mu_A-mu_B)^2 added on top of each expert's own var+noise.
        u_var_B = torch.exp(2.0 * result.u_log_std_B).clamp_min(1e-12)
        u_second_B = torch.diag_embed(u_var_B) + torch.einsum("ti,tj->tij", result.mu_u_B, result.mu_u_B)
        alpha_B = torch.einsum("tij,tj->ti", Kuu_inv, result.mu_u_B)
        mu_f_B = (m_W * alpha_B).sum(dim=1)
        t3_B = torch.einsum("tij,tjk,tkm,tmi->t", Kuu_inv, S_W, Kuu_inv, u_second_B)
        var_f_B = (v1 + t3_B - mu_f_B.pow(2)).clamp_min(1e-10)
        noise_var_B = result.log_noise_var_B.exp().clamp_min(1e-8)
        mu_y = rho * mu_f + (1.0 - rho) * mu_f_B
        var_y = (rho * (var_f + noise_var) + (1.0 - rho) * (var_f_B + noise_var_B)
                 + rho * (1.0 - rho) * (mu_f - mu_f_B).pow(2)).clamp_min(1e-10)
        diag = {
            "rho_mean": float(rho.mean().detach().cpu().item()),
            "noise_std_mean": float(noise_var.sqrt().mean().detach().cpu().item()),
            "var_f_mean": float(var_f.mean().detach().cpu().item()),
            "mu_gap_mean": float((mu_f - mu_f_B).abs().mean().detach().cpu().item()),
            "noise_std_B_mean": float(noise_var_B.sqrt().mean().detach().cpu().item()),
        }
        return mu_y, var_y, diag
    if not mixture:
        mu_y = mu_f
        var_y = (var_f + noise_var).clamp_min(1e-10)
    else:
        if outlier_mean is None:
            out_mu = torch.zeros_like(mu_f)
        else:
            out_mu = _as_float_tensor(outlier_mean, device=Xa.device).to(dtype=Xa.dtype).reshape_as(mu_f)
        if outlier_var is None:
            out_var = torch.ones_like(mu_f)
        else:
            out_var = _as_float_tensor(outlier_var, device=Xa.device).to(dtype=Xa.dtype).reshape_as(mu_f).clamp_min(1e-8)
        mu_y = rho * mu_f + (1.0 - rho) * out_mu
        second_y = rho * (var_f + noise_var + mu_f.pow(2)) + (1.0 - rho) * (out_var + out_mu.pow(2))
        var_y = (second_y - mu_y.pow(2)).clamp_min(1e-10)
    diag = {
        "rho_mean": float(rho.mean().detach().cpu().item()),
        "noise_std_mean": float(noise_var.sqrt().mean().detach().cpu().item()),
        "var_f_mean": float(var_f.mean().detach().cpu().item()),
    }
    return mu_y, var_y, diag


__all__ = [
    "AnalyticStructuredMetricLMJGPConfig",
    "AnalyticStructuredMetricLMJGPResult",
    "StructuredMetricLMJGPConfig",
    "StructuredMetricLMJGPResult",
    "build_fixed_local_inducing_from_w0",
    "free_w_moments_from_result",
    "init_structured_metric_from_w0",
    "predict_analytic_structured_metric_lmjgp",
    "sample_structured_metric_w",
    "sample_structured_metric_w_from_eta_moments",
    "structured_metric_eta_moments_from_R",
    "structured_metric_w_moments",
    "structured_metric_w_moments_from_R",
    "structured_metric_w_moments_from_eta_moments",
    "train_analytic_structured_metric_lmjgp",
    "train_structured_metric_lmjgp",
]
