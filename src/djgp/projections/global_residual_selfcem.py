"""Global GP + local residual self-CEM additive model.

Implements the five recommended variants from
``docs/residual_ablation.md`` on top of the best 5-seed LH self-CEM
configuration (``self_cem2_nstd0.1_kcov0.1``, see
``docs/experiments_log.md`` ``uncertain_w_cem_qr_hybrid_ensemble_lh_5seeds_20260517``):

- ``diagvar``  -- Variant A: pre-residualised local self-CEM with the
  per-neighbor global posterior variances added to the local covariance
  diagonal.
- ``blockvar`` -- Variant B: local CEM hyperparameter optimisation uses the
  full global posterior block ``Sigma_g(X_a, X_a)`` inside ``K_h + Sigma_g +
  sigma_a^2 I``.
- ``oob_diagvar`` -- Variant C: identical to A but ``mu_g``/``v_g`` for
  training points are produced by ``predict_oob_kfold`` so training
  responses are not double-counted.
- ``alt``      -- Variant D: alternate ``q(g)`` <-> ``{rho_a, theta_a}``
  updates.  Each global GP refit uses heteroscedastic pseudo-observations
  ``y_i - hat h_i`` with noise ``sigma_eps^2 + hat v_{h,i}``.
- ``alt_ensW`` -- Variant E: Variant D with a sampled-W deterministic-CEM
  ensemble at prediction time.

The local CEM machinery is provided by ``djgp.projections.uncertain_w_jgp``
and is reused as much as possible.  Per-anchor heteroscedastic noise
floors (and the optional block-covariance variant) are implemented by a
small custom hyperparameter optimiser that loops over anchors.

All variants share the additive predictive form ``mu_y = mu_g + mu_h``,
``v_y = v_g + v_h`` (mean-field approximation; sigma_eps^2 is already
folded into ``v_h`` via the per-anchor noise term, so it is **not** added
twice).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import torch

from djgp.baselines.global_sparse_gp import (
    GlobalGPConfig,
    GlobalGPModel,
    predict_oob_ensemble,
    predict_oob_kfold,
)
from djgp.projections.uncertain_w_jgp import (
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    expected_se_kernel_anchor,
    fit_uncertain_gate_from_labels,
    kernel_vector_covariance_anchor,
    uncertain_w_cem_update_labels,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


VARIANTS = ("diagvar", "blockvar", "oob_diagvar", "alt", "alt_ensW")


@dataclass(frozen=True)
class GlobalResidualSelfCEMConfig:
    """End-to-end configuration for ``run_global_residual_self_cem``."""

    variant: str = "diagvar"
    n_neighbors: int = 35
    w_var: float = 0.02
    cem_updates: int = 2
    kcov: float = 0.1
    local_noise_std_floor: float = 0.1  # nstd=0.1 of the best self-CEM2 baseline
    hyperopt: FixedLabelHyperoptConfig = FixedLabelHyperoptConfig(
        steps=30, lr=0.04, jitter=1e-5
    )
    gate: FixedLabelGateConfig = FixedLabelGateConfig(steps=60, lr=0.05)
    min_inliers: int = 2
    max_inlier_frac: float = 0.95
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0

    # Variant-specific options.
    alt_outer_cycles: int = 2
    oob_k_folds: int = 5
    ensemble_samples: int = 5
    block_variance_scale: float = 1.0
    diag_variance_inflation: float = 1.0
    use_oob_for_test: bool = True
    # When >0 and < N_train, subsample this many training anchors for the
    # local-residual estimate used by the global GP refit (Variant D).  Other
    # training points are still passed to the refit but with a constant
    # ``sigma_eps^2`` pseudo-noise (treated as having no local-residual
    # information).  Default 0 means "use all training points" (slow for
    # large N_train).
    alt_train_anchor_subsample: int = 0
    # Reduce hyperopt steps for the train-anchor pass used by Variant D; the
    # local CEM there only needs rough h_i/v_h_i estimates.
    alt_train_hyper_steps: int | None = None
    alt_train_gate_steps: int | None = None

    # Numerical guards.
    min_noise_var_floor: float = 1e-6


# ---------------------------------------------------------------------------
# Helper: per-anchor heteroscedastic hyperopt
# ---------------------------------------------------------------------------


def _cholesky_psd(K: torch.Tensor, *, jitter: float) -> torch.Tensor:
    last_err: RuntimeError | None = None
    n = K.shape[-1]
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    K = 0.5 * (K + K.transpose(-1, -2))
    for mult in (1.0, 10.0, 100.0, 1000.0):
        try:
            return torch.linalg.cholesky(K + (float(jitter) * mult) * eye)
        except RuntimeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return torch.linalg.cholesky(K + float(jitter) * eye)


def _expected_se_kernel_single_anchor(
    X_anchor_t: torch.Tensor,
    X_neighbors_t: torch.Tensor,
    W_mu_t: torch.Tensor,
    W_var_t: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ``E_W[K(W X_n, W X_n)]`` for a single anchor (n x n matrix)."""
    K = expected_se_kernel_anchor(
        X_neighbors_t.unsqueeze(0),
        X_neighbors_t.unsqueeze(0),
        W_mu_t.unsqueeze(0),
        W_var_t.unsqueeze(0),
        lengthscale=lengthscale,
        signal_var=signal_var,
    )
    return K.squeeze(0)


def _expected_se_kernel_anchor_to_neighbors(
    X_anchor_t: torch.Tensor,
    X_neighbors_t: torch.Tensor,
    W_mu_t: torch.Tensor,
    W_var_t: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
) -> torch.Tensor:
    """``E_W[K(W x_*, W X_n)]`` for a single anchor (returns length-n vector)."""
    K = expected_se_kernel_anchor(
        X_anchor_t.reshape(1, 1, -1),
        X_neighbors_t.unsqueeze(0),
        W_mu_t.unsqueeze(0),
        W_var_t.unsqueeze(0),
        lengthscale=lengthscale,
        signal_var=signal_var,
    )
    return K.squeeze(0).squeeze(0)


def _fit_hyper_per_anchor_with_extras(
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_resid_neighbors: torch.Tensor,
    labels: torch.Tensor,
    W_mu_anchor: torch.Tensor,
    W_var_anchor: torch.Tensor,
    init_lengthscale: torch.Tensor,
    init_signal_var: torch.Tensor,
    init_noise_var: torch.Tensor,
    min_noise_var_per_anchor: torch.Tensor,
    extra_noise_diag: torch.Tensor | None,
    extra_block_cov: torch.Tensor | None,
    cfg: FixedLabelHyperoptConfig,
    min_inliers: int = 2,
) -> dict[str, torch.Tensor]:
    """Per-anchor hyperopt that supports per-point noise diagonal **or** full block ``Sigma_g``.

    For each anchor ``t`` the local Gaussian log-likelihood maximised is::

        log N(y_t,I; m 1, K_h + Sigma_g + sigma_a^2 I + diag(extra_noise))

    where ``extra_noise_diag[t, i]`` (if provided) adds a per-point diagonal
    (Variant A / C) and ``extra_block_cov[t]`` (if provided) adds the full
    block (Variant B).  Either is optional; combining both is allowed and
    additive but is not used by the standard variants.
    """
    T, n, D = X_neighbors.shape
    Q = int(W_mu_anchor.shape[1])
    device = X_anchor.device
    dtype = X_anchor.dtype

    ell_out = torch.empty(T, Q, device=device, dtype=dtype)
    signal_out = torch.empty(T, device=device, dtype=dtype)
    noise_out = torch.empty(T, device=device, dtype=dtype)
    mean_out = torch.empty(T, device=device, dtype=dtype)
    fallback_out = torch.zeros(T, device=device, dtype=torch.bool)

    for t in range(T):
        mask = labels[t].bool()
        if int(mask.sum().item()) < int(min_inliers):
            fallback_out[t] = True
            k_top = min(max(int(min_inliers), 1), n)
            idx = torch.topk(labels[t].to(dtype=dtype), k=k_top, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[idx] = True
        y_i = y_resid_neighbors[t, mask]
        n_i = int(y_i.numel())
        if n_i < 1:
            ell_out[t] = init_lengthscale[t].clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
            signal_out[t] = init_signal_var[t].clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
            noise_out[t] = init_noise_var[t].clamp(float(cfg.min_noise_var), float(cfg.max_noise_var))
            mean_out[t] = torch.zeros((), device=device, dtype=dtype)
            continue
        floor_t = float(max(float(min_noise_var_per_anchor[t].item()), float(cfg.min_noise_var)))
        init_noise_t = float(max(float(init_noise_var[t].item()), floor_t))
        log_ell = torch.log(
            init_lengthscale[t].clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
        ).detach().clone().requires_grad_(True)
        log_signal = torch.log(
            init_signal_var[t].clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
        ).detach().clone().requires_grad_(True)
        log_noise = torch.log(
            torch.tensor(init_noise_t, device=device, dtype=dtype).clamp(
                floor_t, float(cfg.max_noise_var)
            )
        ).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([log_ell, log_signal, log_noise], lr=float(cfg.lr))
        final_mean = y_i.mean()
        for _ in range(int(cfg.steps)):
            opt.zero_grad()
            ell = log_ell.exp().clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
            sig = log_signal.exp().clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
            noise = log_noise.exp().clamp(floor_t, float(cfg.max_noise_var))
            K_full = _expected_se_kernel_single_anchor(
                X_anchor[t], X_neighbors[t], W_mu_anchor[t], W_var_anchor[t], ell, sig
            )
            K_i = K_full[mask][:, mask]
            Ky = K_i + noise * torch.eye(n_i, device=device, dtype=dtype)
            if extra_noise_diag is not None:
                Ky = Ky + torch.diag(extra_noise_diag[t, mask])
            if extra_block_cov is not None:
                block_t = extra_block_cov[t][mask][:, mask]
                Ky = Ky + block_t
            L = _cholesky_psd(Ky.unsqueeze(0), jitter=float(cfg.jitter)).squeeze(0)
            one = torch.ones_like(y_i)
            Ainv_y = torch.cholesky_solve(y_i.unsqueeze(-1), L).squeeze(-1)
            Ainv_one = torch.cholesky_solve(one.unsqueeze(-1), L).squeeze(-1)
            m = (one @ Ainv_y) / (one @ Ainv_one).clamp_min(1e-12)
            resid = y_i - m
            alpha = torch.cholesky_solve(resid.unsqueeze(-1), L).squeeze(-1)
            logdet = 2.0 * torch.diag(L).log().sum()
            nll = 0.5 * (resid @ alpha + logdet + n_i * math.log(2.0 * math.pi))
            nll.backward()
            opt.step()
            final_mean = m.detach()
        with torch.no_grad():
            ell_out[t] = log_ell.exp().clamp(float(cfg.min_lengthscale), float(cfg.max_lengthscale))
            signal_out[t] = log_signal.exp().clamp(float(cfg.min_signal_var), float(cfg.max_signal_var))
            noise_out[t] = log_noise.exp().clamp(floor_t, float(cfg.max_noise_var))
            mean_out[t] = final_mean if int(cfg.steps) > 0 else y_i.mean()
    return {
        "lengthscale": ell_out,
        "signal_var": signal_out,
        "noise_var": noise_out,
        "mean_const": mean_out,
        "fallback": fallback_out,
    }


def _predict_per_anchor_with_extras(
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_resid_neighbors: torch.Tensor,
    labels: torch.Tensor,
    W_mu_anchor: torch.Tensor,
    W_var_anchor: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    extra_noise_diag: torch.Tensor | None,
    extra_block_cov: torch.Tensor | None,
    correction_weight: float = 0.0,
    jitter: float = 1e-5,
    min_inliers: int = 2,
) -> dict[str, torch.Tensor]:
    """Per-anchor residual predictive distribution with per-point/block ``Sigma_g``.

    Returns ``mu`` and ``var`` on the **residual y scale** (i.e. for h_a only;
    the caller adds ``mu_g``, ``v_g`` of the test points to get the full
    additive prediction).
    """
    T, n, D = X_neighbors.shape
    device = X_anchor.device
    dtype = X_anchor.dtype
    mu_out = torch.empty(T, device=device, dtype=dtype)
    var_out = torch.empty(T, device=device, dtype=dtype)
    correction_accum = torch.zeros(T, device=device, dtype=dtype)
    n_inliers = torch.empty(T, device=device, dtype=torch.long)
    fallback_count = 0
    need_cov_k = float(correction_weight) != 0.0
    for t in range(T):
        mask = labels[t].bool()
        if int(mask.sum().item()) < int(min_inliers):
            fallback_count += 1
            k_top = min(max(int(min_inliers), 1), n)
            idx = torch.topk(labels[t].to(dtype=dtype), k=k_top, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[idx] = True
        n_inliers[t] = int(mask.sum().item())
        ell_t = lengthscale[t]
        sig_t = signal_var[t]
        noise_t = noise_var[t]
        K_full = _expected_se_kernel_single_anchor(
            X_anchor[t], X_neighbors[t], W_mu_anchor[t], W_var_anchor[t], ell_t, sig_t
        )
        k_star = _expected_se_kernel_anchor_to_neighbors(
            X_anchor[t], X_neighbors[t], W_mu_anchor[t], W_var_anchor[t], ell_t, sig_t
        )
        cov_k = None
        if need_cov_k:
            cov_k = kernel_vector_covariance_anchor(
                X_anchor[t : t + 1],
                X_neighbors[t : t + 1],
                W_mu_anchor[t : t + 1],
                W_var_anchor[t : t + 1],
                k_star.view(1, -1),
                lengthscale=ell_t,
                signal_var=sig_t,
            ).squeeze(0)
        K_i = K_full[mask][:, mask]
        k_i = k_star[mask]
        y_i = y_resid_neighbors[t, mask]
        m = mean_const[t]
        Ky = K_i + noise_t * torch.eye(K_i.shape[0], device=device, dtype=dtype)
        if extra_noise_diag is not None:
            Ky = Ky + torch.diag(extra_noise_diag[t, mask])
        if extra_block_cov is not None:
            Ky = Ky + extra_block_cov[t][mask][:, mask]
        L = _cholesky_psd(Ky.unsqueeze(0), jitter=jitter).squeeze(0)
        alpha = torch.cholesky_solve((y_i - m).unsqueeze(-1), L).squeeze(-1)
        mu_out[t] = m + torch.dot(k_i, alpha)
        v = torch.cholesky_solve(k_i.unsqueeze(-1), L).squeeze(-1)
        latent_var = (sig_t - torch.dot(k_i, v)).clamp_min(0.0)
        corr_t = torch.zeros((), device=device, dtype=dtype)
        if cov_k is not None:
            cov_i = cov_k[mask][:, mask]
            corr_t = torch.einsum("i,ij,j->", alpha, cov_i, alpha).clamp_min(0.0)
            correction_accum[t] = corr_t
        var_out[t] = (
            latent_var + noise_t + float(correction_weight) * corr_t
        ).clamp_min(1e-12)
    return {
        "mu": mu_out,
        "var": var_out,
        "n_inliers": n_inliers,
        "fallback_count": int(fallback_count),
        "mean_kernel_vector_correction": float(correction_accum.mean().item()),
    }


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------


def _neighbor_stack(
    X_query: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        d = torch.cdist(X_query, X_train)
        idx = torch.topk(
            d, k=min(int(n_neighbors), int(X_train.shape[0])), largest=False, dim=1
        ).indices
    return X_train[idx].contiguous(), y_train[idx].contiguous(), idx.contiguous()


def _projected_neighbor_stack_from_raw_pool(
    X_query: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    W: torch.Tensor,
    n_neighbors: int,
    raw_pool: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-rank a raw candidate pool by projected distance under a deterministic ``W``."""
    n_train = int(X_train.shape[0])
    raw_pool = min(max(int(raw_pool), int(n_neighbors)), n_train)
    with torch.no_grad():
        raw_d = torch.cdist(X_query, X_train)
        raw_idx = torch.topk(raw_d, k=raw_pool, largest=False, dim=1).indices
        X_pool = X_train[raw_idx]
        diff = X_pool - X_query.unsqueeze(1)
        proj = torch.einsum("tnd,qd->tnq", diff, W)
        proj_d = proj.pow(2).sum(dim=-1)
        rank_idx = torch.topk(
            proj_d,
            k=min(int(n_neighbors), raw_pool),
            largest=False,
            dim=1,
        ).indices
        idx = torch.gather(raw_idx, 1, rank_idx)
    return X_train[idx].contiguous(), y_train[idx].contiguous(), idx.contiguous()


def _make_anchor_w_moments(
    W0: torch.Tensor, X_anchor: torch.Tensor, X_neighbors: torch.Tensor, *, w_var: float
) -> dict[str, torch.Tensor]:
    T = int(X_anchor.shape[0])
    Q, D = int(W0.shape[0]), int(W0.shape[1])
    return {
        "W_mu_anchor": W0.reshape(1, Q, D).expand(T, Q, D).contiguous(),
        "W_var_anchor": torch.full(
            (T, Q, D), float(w_var), device=X_anchor.device, dtype=X_anchor.dtype
        ),
    }


def _meanw_default_hypers(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    labels: torch.Tensor,
    W0: torch.Tensor,
    *,
    min_inliers: int = 2,
) -> dict[str, torch.Tensor]:
    """Cheap label-conditioned defaults: pairwise-median lengthscale in projected Z."""
    T, n, _D = X_neighbors.shape
    Q = int(W0.shape[0])
    device = X_anchor.device
    dtype = X_anchor.dtype
    Z_neighbors = torch.einsum("tnd,qd->tnq", X_neighbors, W0)
    Z_anchor = torch.einsum("td,qd->tq", X_anchor, W0)
    ell = torch.empty(T, Q, device=device, dtype=dtype)
    signal = torch.empty(T, device=device, dtype=dtype)
    noise = torch.empty(T, device=device, dtype=dtype)
    mean = torch.empty(T, device=device, dtype=dtype)
    for t in range(T):
        mask = labels[t].bool()
        if int(mask.sum().item()) < int(min_inliers):
            k_top = min(max(int(min_inliers), 1), n)
            idx = torch.topk(labels[t].to(dtype=dtype), k=k_top, largest=True).indices
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[idx] = True
        z = Z_neighbors[t, mask]
        if z.shape[0] > 1:
            pd = torch.cdist(z, z)
            vals = pd[pd > 0]
            med = (
                vals.median().clamp(0.05, 10.0)
                if vals.numel()
                else torch.tensor(1.0, device=z.device, dtype=z.dtype)
            )
        else:
            med = torch.linalg.norm(Z_neighbors[t] - Z_anchor[t], dim=1).median().clamp(
                0.05, 10.0
            )
        y_i = y_neighbors[t, mask]
        mean[t] = y_i.mean()
        var = y_i.var(unbiased=False).clamp_min(0.05)
        signal[t] = var.clamp(1e-3, 20.0)
        noise[t] = (0.10 * var).clamp(1e-4, 2.0)
        ell[t] = med
    return {
        "lengthscale": ell,
        "signal_var": signal,
        "noise_var": noise,
        "mean_const": mean,
    }


def _initial_residual_labels(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_resid_neighbors: torch.Tensor,
    *,
    seed_frac: float = 0.25,
    inlier_frac: float = 0.70,
    min_inliers: int = 2,
) -> torch.Tensor:
    """Seed labels using the response-based heuristic (anchor-local median y)."""
    T, n, _D = X_neighbors.shape
    labels = torch.zeros(T, n, device=X_neighbors.device, dtype=torch.bool)
    seed_k = min(max(int(round(float(seed_frac) * n)), 1), n)
    keep_k = min(max(int(round(float(inlier_frac) * n)), int(min_inliers)), n)
    with torch.no_grad():
        d = torch.sum((X_neighbors - X_anchor.unsqueeze(1)).pow(2), dim=-1)
        seed_idx = torch.topk(d, k=seed_k, largest=False, dim=1).indices
        seed_y = torch.gather(y_resid_neighbors, 1, seed_idx)
        center = seed_y.median(dim=1).values
        dy = (y_resid_neighbors - center.unsqueeze(1)).abs()
        keep = torch.topk(dy, k=keep_k, largest=False, dim=1).indices
        labels.scatter_(1, keep, True)
    return labels


def _run_local_self_cem_with_extras(
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_resid_neighbors: torch.Tensor,
    init_labels: torch.Tensor,
    W_mu_anchor: torch.Tensor,
    W_var_anchor: torch.Tensor,
    init_lengthscale: torch.Tensor,
    init_signal_var: torch.Tensor,
    init_noise_var: torch.Tensor,
    min_noise_var_per_anchor: torch.Tensor,
    extra_noise_diag: torch.Tensor | None,
    extra_block_cov: torch.Tensor | None,
    cfg: GlobalResidualSelfCEMConfig,
    gate_init: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run ``cem_updates`` rounds of (hyperopt -> gate -> relabel) on residual y."""
    labels = init_labels.clone()
    gate = gate_init
    history: list[dict[str, float]] = []
    hp: dict[str, torch.Tensor] | None = None
    for outer in range(max(0, int(cfg.cem_updates))):
        hp = _fit_hyper_per_anchor_with_extras(
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            y_resid_neighbors=y_resid_neighbors,
            labels=labels,
            W_mu_anchor=W_mu_anchor,
            W_var_anchor=W_var_anchor,
            init_lengthscale=init_lengthscale if hp is None else hp["lengthscale"],
            init_signal_var=init_signal_var if hp is None else hp["signal_var"],
            init_noise_var=init_noise_var if hp is None else hp["noise_var"],
            min_noise_var_per_anchor=min_noise_var_per_anchor,
            extra_noise_diag=extra_noise_diag,
            extra_block_cov=extra_block_cov,
            cfg=cfg.hyperopt,
            min_inliers=cfg.min_inliers,
        )
        gate_fit = fit_uncertain_gate_from_labels(
            X_anchor,
            X_neighbors,
            labels,
            mode="anchor",
            W_mu_anchor=W_mu_anchor,
            W_var_anchor=W_var_anchor,
            gate_init=gate,
            config=cfg.gate,
        )
        gate = gate_fit["gate"]
        upd = uncertain_w_cem_update_labels(
            X_anchor,
            X_neighbors,
            y_resid_neighbors,
            labels,
            mode="anchor",
            W_mu_anchor=W_mu_anchor,
            W_var_anchor=W_var_anchor,
            gate=gate,
            lengthscale=hp["lengthscale"],
            signal_var=hp["signal_var"],
            noise_var=hp["noise_var"],
            mean_const=hp["mean_const"],
            outlier_mode="background_normal",
            outlier_sigma_mult=cfg.outlier_sigma_mult,
            background_scale_mult=cfg.background_scale_mult,
            gh_points=cfg.gate.gh_points,
            jitter=cfg.hyperopt.jitter,
            min_inliers=cfg.min_inliers,
            max_inlier_frac=cfg.max_inlier_frac,
        )
        labels = upd["labels"].to(dtype=torch.float64) if upd["labels"].dtype == torch.bool else upd["labels"].to(dtype=labels.dtype)
        history.append(
            {
                "outer": float(outer + 1),
                "changed_frac": float(upd["changed_frac"].detach().cpu().item()),
                "inlier_frac": float(labels.to(dtype=torch.float64).mean().detach().cpu().item()),
                "mean_lengthscale": float(hp["lengthscale"].mean().detach().cpu().item()),
                "mean_signal_var": float(hp["signal_var"].mean().detach().cpu().item()),
                "mean_noise_var": float(hp["noise_var"].mean().detach().cpu().item()),
            }
        )
    # Final refit after relabel (mirrors run_uncertain_w_self_cem.refit_after_update=True).
    hp = _fit_hyper_per_anchor_with_extras(
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        labels=labels,
        W_mu_anchor=W_mu_anchor,
        W_var_anchor=W_var_anchor,
        init_lengthscale=init_lengthscale if hp is None else hp["lengthscale"],
        init_signal_var=init_signal_var if hp is None else hp["signal_var"],
        init_noise_var=init_noise_var if hp is None else hp["noise_var"],
        min_noise_var_per_anchor=min_noise_var_per_anchor,
        extra_noise_diag=extra_noise_diag,
        extra_block_cov=extra_block_cov,
        cfg=cfg.hyperopt,
        min_inliers=cfg.min_inliers,
    )
    gate_fit = fit_uncertain_gate_from_labels(
        X_anchor,
        X_neighbors,
        labels,
        mode="anchor",
        W_mu_anchor=W_mu_anchor,
        W_var_anchor=W_var_anchor,
        gate_init=gate,
        config=cfg.gate,
    )
    gate = gate_fit["gate"]
    return {
        "labels": (labels > 0.5).detach(),
        "lengthscale": hp["lengthscale"].detach(),
        "signal_var": hp["signal_var"].detach(),
        "noise_var": hp["noise_var"].detach(),
        "mean_const": hp["mean_const"].detach(),
        "gate": gate.detach() if isinstance(gate, torch.Tensor) else gate,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Aggregation utilities for Variant D
# ---------------------------------------------------------------------------


def _train_local_residual_estimates(
    *,
    X_train_t: torch.Tensor,
    y_train_resid_t: torch.Tensor,
    W0: torch.Tensor,
    cfg: GlobalResidualSelfCEMConfig,
    mu_g_train: torch.Tensor,
    v_g_train: torch.Tensor,
    block_g_train_neighbors: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Run local residual self-CEM with each (subsampled) training point as anchor.

    Used by Variant D to estimate per-training-point ``hat h_i`` and ``hat
    v_{h,i}`` for the global-GP refit pseudo-observations.  When
    ``cfg.alt_train_anchor_subsample > 0`` and is less than ``N_train``,
    only a random subset of training points is used as anchors; the
    remaining training points get ``hat h_i = 0`` (so the global GP refit
    sees them as ``tilde y_i = y_i - 0 = y_i``) and a constant pseudo-noise
    ``sigma_eps^2``.  This trades some bias for a dramatic speedup when
    ``N_train`` is large.
    """
    device = X_train_t.device
    dtype = X_train_t.dtype
    n_train = int(X_train_t.shape[0])
    Q = int(W0.shape[0])
    sub_n = int(cfg.alt_train_anchor_subsample)
    if sub_n > 0 and sub_n < n_train:
        rng = torch.Generator(device=device).manual_seed(int(seed) + 31337)
        anchor_idx = torch.randperm(n_train, generator=rng, device=device)[:sub_n]
    else:
        anchor_idx = torch.arange(n_train, device=device)
    X_anchor = X_train_t[anchor_idx].contiguous()
    n_anchors = int(X_anchor.shape[0])
    X_neighbors, y_resid_neighbors, neighbor_idx = _neighbor_stack(
        X_anchor,
        X_train_t,
        y_train_resid_t,
        n_neighbors=int(cfg.n_neighbors),
    )
    w_moments = _make_anchor_w_moments(W0, X_anchor, X_neighbors, w_var=float(cfg.w_var))
    labels0 = _initial_residual_labels(
        X_anchor, X_neighbors, y_resid_neighbors, min_inliers=cfg.min_inliers
    )
    defaults = _meanw_default_hypers(
        X_anchor, X_neighbors, y_resid_neighbors, labels0, W0, min_inliers=cfg.min_inliers
    )
    extra_noise_diag = v_g_train[neighbor_idx].clamp_min(0.0)
    noise_std_var = float(cfg.local_noise_std_floor) ** 2
    min_noise_var_per_anchor = torch.full(
        (n_anchors,),
        max(noise_std_var, float(cfg.min_noise_var_floor)),
        device=device,
        dtype=dtype,
    )
    # Optionally use cheaper hyperopt/gate for the train-anchor pass.
    train_cfg = cfg
    if cfg.alt_train_hyper_steps is not None or cfg.alt_train_gate_steps is not None:
        train_hyperopt = (
            replace(cfg.hyperopt, steps=int(cfg.alt_train_hyper_steps))
            if cfg.alt_train_hyper_steps is not None
            else cfg.hyperopt
        )
        train_gate = (
            replace(cfg.gate, steps=int(cfg.alt_train_gate_steps))
            if cfg.alt_train_gate_steps is not None
            else cfg.gate
        )
        train_cfg = replace(cfg, hyperopt=train_hyperopt, gate=train_gate)
    state = _run_local_self_cem_with_extras(
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        init_labels=labels0,
        W_mu_anchor=w_moments["W_mu_anchor"],
        W_var_anchor=w_moments["W_var_anchor"],
        init_lengthscale=defaults["lengthscale"],
        init_signal_var=defaults["signal_var"],
        init_noise_var=defaults["noise_var"].clamp_min(min_noise_var_per_anchor),
        min_noise_var_per_anchor=min_noise_var_per_anchor,
        extra_noise_diag=extra_noise_diag,
        extra_block_cov=block_g_train_neighbors,
        cfg=train_cfg,
    )
    pred = _predict_per_anchor_with_extras(
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        labels=state["labels"],
        W_mu_anchor=w_moments["W_mu_anchor"],
        W_var_anchor=w_moments["W_var_anchor"],
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        extra_noise_diag=extra_noise_diag,
        extra_block_cov=block_g_train_neighbors,
        correction_weight=float(cfg.kcov),
        jitter=cfg.hyperopt.jitter,
        min_inliers=cfg.min_inliers,
    )
    # Expand back to all training points: anchors get their estimate, the rest
    # are filled with zero mean and an inflated pseudo-noise so the global GP
    # refit treats them as ordinary noisy observations.
    mu_h_train = torch.zeros(n_train, device=device, dtype=dtype)
    var_h_train = torch.full(
        (n_train,),
        float(noise_std_var),
        device=device,
        dtype=dtype,
    )
    mu_h_train[anchor_idx] = pred["mu"]
    var_h_train[anchor_idx] = pred["var"]
    return {
        "mu_h_train": mu_h_train,
        "var_h_train": var_h_train,
        "labels_train": state["labels"],
        "inlier_rate_train": state["labels"].to(dtype=torch.float64).mean(),
        "n_train_anchors": int(n_anchors),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _projection_W0_pls(
    X_train: np.ndarray, y_train: np.ndarray, Q: int, *, seed: int = 0
) -> np.ndarray:
    """PLS projection used by the best self-CEM2 baseline (see ``_projection_W0``)."""
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(seed)
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    Q = max(1, int(Q))
    Q = min(Q, X.shape[1])
    pls = PLSRegression(n_components=Q)
    pls.fit(X, y)
    W = pls.x_loadings_.T  # [Q, D]
    # Normalize rows for stability.
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    W = W / norms
    return W


def _gaussian_pi_calibration(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *, level: float = 0.9
) -> tuple[float, float]:
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2.0))
    sigma = np.maximum(sigma, 1e-12)
    cov = float(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))
    width = float(np.mean(2.0 * z * sigma))
    return cov, width


@dataclass
class GlobalResidualSelfCEMResult:
    """Output of ``run_global_residual_self_cem``."""

    mu_y: np.ndarray
    var_y: np.ndarray
    sigma_y: np.ndarray
    mu_g_test: np.ndarray
    var_g_test: np.ndarray
    mu_h_test: np.ndarray
    var_h_test: np.ndarray
    labels_test: torch.Tensor
    diagnostics: dict[str, Any]


def run_global_residual_self_cem(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    Q: int,
    cfg: GlobalResidualSelfCEMConfig,
    global_gp_cfg: GlobalGPConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> GlobalResidualSelfCEMResult:
    """End-to-end runner returning ``(mu_y, var_y, ...)`` and rich diagnostics."""
    if cfg.variant not in VARIANTS:
        raise ValueError(f"Unknown residual variant {cfg.variant!r}; allowed: {VARIANTS}.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag: dict[str, Any] = {"variant": str(cfg.variant), "backend": str(global_gp_cfg.backend)}
    t_start = time.perf_counter()

    X_train_np = np.asarray(X_train, dtype=np.float64)
    y_train_raw = np.asarray(y_train, dtype=np.float64).reshape(-1)
    X_test_np = np.asarray(X_test, dtype=np.float64)
    n_train = int(X_train_np.shape[0])
    n_test = int(X_test_np.shape[0])

    # Standardize y so that all local-CEM hyperparameter bounds (lengthscale,
    # signal/noise variance) are on the same numerical scale as the existing
    # ``self_cem2_nstd0.1_kcov0.1`` baseline.  Everything inside the driver
    # operates on standardized y; predictions are rescaled to the original
    # scale before returning.
    y_train_mean = float(y_train_raw.mean())
    y_train_std_val = float(y_train_raw.std() if y_train_raw.std() > 1e-8 else 1.0)
    y_train_std_np = ((y_train_raw - y_train_mean) / y_train_std_val).astype(np.float64)

    W0_np = _projection_W0_pls(X_train_np, y_train_std_np, Q, seed=int(seed) + 513)
    W0 = torch.as_tensor(W0_np, device=device, dtype=dtype)

    # --- 1) Train global GP on standardized y. ---
    # GlobalGPModel(standardize_y=True) re-standardizes internally; pass a
    # config with ``standardize_y=False`` since we already standardized.
    global_gp_cfg_std = replace(global_gp_cfg, standardize_y=False, seed=int(seed))
    t_global = time.perf_counter()
    oob_models: list[GlobalGPModel] = []
    if cfg.variant == "oob_diagvar":
        mu_g_train_t, v_g_train_t, oob_models = predict_oob_kfold(
            global_gp_cfg_std,
            X_train_np,
            y_train_std_np,
            k_folds=int(cfg.oob_k_folds),
            device=device,
            dtype=torch.float32,
        )
        if cfg.use_oob_for_test:
            mu_g_test_t, v_g_test_t = predict_oob_ensemble(oob_models, X_test_np)
            global_model = oob_models[0]
        else:
            global_model = GlobalGPModel(global_gp_cfg_std, device=device, dtype=torch.float32)
            global_model.train(X_train_np, y_train_std_np)
            mu_g_test_t, v_g_test_t = global_model.predict(X_test_np, include_noise=False)
    else:
        global_model = GlobalGPModel(global_gp_cfg_std, device=device, dtype=torch.float32)
        train_res = global_model.train(X_train_np, y_train_std_np)
        diag["global_gp_train_sec"] = float(train_res.train_sec)
        diag["global_gp_final_loss"] = float(train_res.final_loss)
        diag["global_gp_num_data"] = int(train_res.num_data)
        mu_g_train_t, v_g_train_t = global_model.predict(X_train_np, include_noise=False)
        mu_g_test_t, v_g_test_t = global_model.predict(X_test_np, include_noise=False)
    diag["global_gp_total_sec"] = float(time.perf_counter() - t_global)

    mu_g_train = mu_g_train_t.to(device=device, dtype=dtype)
    v_g_train = v_g_train_t.to(device=device, dtype=dtype).clamp_min(0.0)
    mu_g_test = mu_g_test_t.to(device=device, dtype=dtype)
    v_g_test = v_g_test_t.to(device=device, dtype=dtype).clamp_min(0.0)

    diag["y_train_mean"] = float(y_train_mean)
    diag["y_train_std"] = float(y_train_std_val)
    diag["global_v_g_train_mean"] = float(v_g_train.mean().item())
    diag["global_v_g_test_mean"] = float(v_g_test.mean().item())
    diag["global_resid_train_std"] = float(
        torch.tensor(y_train_std_np, device=device, dtype=dtype).sub(mu_g_train).std().item()
    )

    # --- 2) Residualize training y (standardized scale). ---
    y_train_t = torch.as_tensor(y_train_std_np, device=device, dtype=dtype)
    y_train_resid_t = y_train_t - mu_g_train

    # --- 3) Build test neighborhoods and per-neighborhood global GP outputs. ---
    X_train_t = torch.as_tensor(X_train_np, device=device, dtype=dtype)
    X_test_t = torch.as_tensor(X_test_np, device=device, dtype=dtype)
    X_neighbors, y_resid_neighbors, test_neighbor_idx = _neighbor_stack(
        X_test_t, X_train_t, y_train_resid_t, n_neighbors=int(cfg.n_neighbors)
    )
    v_g_neighbors = v_g_train[test_neighbor_idx]

    block_neighbors: torch.Tensor | None = None
    if cfg.variant == "blockvar":
        nb = global_model.predict_neighborhood(
            X_neighbors.to(torch.float32), include_block=True, include_noise=False
        )
        block_neighbors = nb["block"].to(device=device, dtype=dtype).clamp_min(0.0)
        block_neighbors = 0.5 * (block_neighbors + block_neighbors.transpose(-1, -2))
        block_neighbors = block_neighbors * float(cfg.block_variance_scale)
        # Use the block diagonals as the source of v_g over neighbors (more accurate
        # for the same neighborhood than the pointwise predict).
        diag_block = torch.diagonal(block_neighbors, dim1=-2, dim2=-1).clamp_min(0.0)
        v_g_neighbors = diag_block

    v_g_neighbors = v_g_neighbors * float(cfg.diag_variance_inflation)
    # The per-anchor local-noise floor encodes sigma_eps^2 only.  Global GP
    # uncertainty enters the local likelihood through ``extra_noise_diag``
    # (Variant A / C / D) or ``extra_block_cov`` (Variant B); adding it to
    # the noise floor as well would double-count the global variance.
    noise_std_var = float(cfg.local_noise_std_floor) ** 2
    min_noise_var_per_anchor = torch.full(
        (n_test,),
        max(noise_std_var, float(cfg.min_noise_var_floor)),
        device=device,
        dtype=dtype,
    )

    # --- 4) Seed local labels using residual-based heuristic. ---
    init_labels = _initial_residual_labels(
        X_test_t, X_neighbors, y_resid_neighbors, min_inliers=cfg.min_inliers
    )
    defaults = _meanw_default_hypers(
        X_test_t, X_neighbors, y_resid_neighbors, init_labels, W0, min_inliers=cfg.min_inliers
    )
    w_moments_test = _make_anchor_w_moments(
        W0, X_test_t, X_neighbors, w_var=float(cfg.w_var)
    )

    extra_noise_diag = None
    extra_block_cov = None
    if cfg.variant in ("diagvar", "oob_diagvar"):
        extra_noise_diag = v_g_neighbors
    elif cfg.variant == "blockvar":
        extra_block_cov = block_neighbors
    elif cfg.variant in ("alt", "alt_ensW"):
        extra_noise_diag = v_g_neighbors

    t_local = time.perf_counter()
    local_state = _run_local_self_cem_with_extras(
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        init_labels=init_labels,
        W_mu_anchor=w_moments_test["W_mu_anchor"],
        W_var_anchor=w_moments_test["W_var_anchor"],
        init_lengthscale=defaults["lengthscale"],
        init_signal_var=defaults["signal_var"],
        init_noise_var=defaults["noise_var"].clamp_min(min_noise_var_per_anchor),
        min_noise_var_per_anchor=min_noise_var_per_anchor,
        extra_noise_diag=extra_noise_diag,
        extra_block_cov=extra_block_cov,
        cfg=cfg,
    )
    diag["local_self_cem_sec"] = float(time.perf_counter() - t_local)

    pred_local = _predict_per_anchor_with_extras(
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        labels=local_state["labels"],
        W_mu_anchor=w_moments_test["W_mu_anchor"],
        W_var_anchor=w_moments_test["W_var_anchor"],
        lengthscale=local_state["lengthscale"],
        signal_var=local_state["signal_var"],
        noise_var=local_state["noise_var"],
        mean_const=local_state["mean_const"],
        extra_noise_diag=extra_noise_diag,
        extra_block_cov=extra_block_cov,
        correction_weight=float(cfg.kcov),
        jitter=cfg.hyperopt.jitter,
        min_inliers=cfg.min_inliers,
    )
    mu_h_test = pred_local["mu"]
    var_h_test = pred_local["var"]
    diag["fallback_count"] = int(pred_local["fallback_count"])
    diag["mean_kernel_vector_correction"] = float(
        pred_local.get("mean_kernel_vector_correction", 0.0)
    )
    diag["mean_n_inliers"] = float(pred_local["n_inliers"].to(dtype=dtype).mean().item())
    diag["final_inlier_rate"] = float(
        local_state["labels"].to(dtype=torch.float64).mean().item()
    )
    diag["mean_lengthscale"] = float(local_state["lengthscale"].mean().item())
    diag["mean_signal_var"] = float(local_state["signal_var"].mean().item())
    diag["mean_noise_var"] = float(local_state["noise_var"].mean().item())

    # --- 5) Variant D / E outer cycles. ---
    if cfg.variant in ("alt", "alt_ensW"):
        outer_history: list[dict[str, float]] = [
            {
                "outer": 0.0,
                "mean_noise_var": diag["mean_noise_var"],
                "mean_n_inliers": diag["mean_n_inliers"],
            }
        ]
        for outer in range(max(1, int(cfg.alt_outer_cycles))):
            # 5a) Compute per-training-point residual estimates.
            train_est = _train_local_residual_estimates(
                X_train_t=X_train_t,
                y_train_resid_t=y_train_resid_t,
                W0=W0,
                cfg=cfg,
                mu_g_train=mu_g_train,
                v_g_train=v_g_train,
                block_g_train_neighbors=None,
                seed=int(seed) + 100 * (outer + 1),
            )
            hat_h_train = train_est["mu_h_train"]
            hat_v_h_train = train_est["var_h_train"].clamp_min(
                float(cfg.min_noise_var_floor)
            )
            tilde_y = y_train_t - hat_h_train
            tilde_noise_var = (hat_v_h_train + noise_std_var).clamp_min(
                float(cfg.min_noise_var_floor)
            )
            # 5b) Refit global GP with heteroscedastic pseudo-observations.
            global_model = GlobalGPModel(
                replace(global_gp_cfg, seed=int(seed) + 1000 * (outer + 1)),
                device=device,
                dtype=torch.float32,
            )
            global_model.train(
                X_train_np,
                tilde_y.detach().cpu().numpy(),
                noise_var_per_point=tilde_noise_var.detach().cpu().numpy(),
            )
            mu_g_train_t, v_g_train_t = global_model.predict(X_train_np, include_noise=False)
            mu_g_train = mu_g_train_t.to(device=device, dtype=dtype)
            v_g_train = v_g_train_t.to(device=device, dtype=dtype).clamp_min(0.0)
            y_train_resid_t = y_train_t - mu_g_train
            # 5c) Refresh test inputs derived from training residuals.
            X_neighbors, y_resid_neighbors, test_neighbor_idx = _neighbor_stack(
                X_test_t, X_train_t, y_train_resid_t, n_neighbors=int(cfg.n_neighbors)
            )
            v_g_neighbors = v_g_train[test_neighbor_idx]
            v_g_neighbors = v_g_neighbors * float(cfg.diag_variance_inflation)
            min_noise_var_per_anchor = torch.full(
                (n_test,),
                max(noise_std_var, float(cfg.min_noise_var_floor)),
                device=device,
                dtype=dtype,
            )
            init_labels = _initial_residual_labels(
                X_test_t, X_neighbors, y_resid_neighbors, min_inliers=cfg.min_inliers
            )
            defaults = _meanw_default_hypers(
                X_test_t, X_neighbors, y_resid_neighbors, init_labels, W0, min_inliers=cfg.min_inliers
            )
            extra_noise_diag = v_g_neighbors
            local_state = _run_local_self_cem_with_extras(
                X_anchor=X_test_t,
                X_neighbors=X_neighbors,
                y_resid_neighbors=y_resid_neighbors,
                init_labels=init_labels,
                W_mu_anchor=w_moments_test["W_mu_anchor"],
                W_var_anchor=w_moments_test["W_var_anchor"],
                init_lengthscale=defaults["lengthscale"],
                init_signal_var=defaults["signal_var"],
                init_noise_var=defaults["noise_var"].clamp_min(min_noise_var_per_anchor),
                min_noise_var_per_anchor=min_noise_var_per_anchor,
                extra_noise_diag=extra_noise_diag,
                extra_block_cov=None,
                cfg=cfg,
            )
            pred_local = _predict_per_anchor_with_extras(
                X_anchor=X_test_t,
                X_neighbors=X_neighbors,
                y_resid_neighbors=y_resid_neighbors,
                labels=local_state["labels"],
                W_mu_anchor=w_moments_test["W_mu_anchor"],
                W_var_anchor=w_moments_test["W_var_anchor"],
                lengthscale=local_state["lengthscale"],
                signal_var=local_state["signal_var"],
                noise_var=local_state["noise_var"],
                mean_const=local_state["mean_const"],
                extra_noise_diag=extra_noise_diag,
                extra_block_cov=None,
                correction_weight=float(cfg.kcov),
                jitter=cfg.hyperopt.jitter,
                min_inliers=cfg.min_inliers,
            )
            mu_h_test = pred_local["mu"]
            var_h_test = pred_local["var"]
            outer_history.append(
                {
                    "outer": float(outer + 1),
                    "mean_noise_var": float(local_state["noise_var"].mean().item()),
                    "mean_n_inliers": float(
                        pred_local["n_inliers"].to(dtype=dtype).mean().item()
                    ),
                    "global_v_g_train_mean": float(v_g_train.mean().item()),
                    "fallback_count": int(pred_local["fallback_count"]),
                    "train_inlier_rate": float(train_est["inlier_rate_train"].item()),
                }
            )
            mu_g_test_t, v_g_test_t = global_model.predict(X_test_np, include_noise=False)
            mu_g_test = mu_g_test_t.to(device=device, dtype=dtype)
            v_g_test = v_g_test_t.to(device=device, dtype=dtype).clamp_min(0.0)
        diag["alt_outer_history"] = outer_history

    # --- 6) Variant E: sampled-W deterministic CEM ensemble at test time. ---
    if cfg.variant == "alt_ensW" and int(cfg.ensemble_samples) > 0:
        S = int(cfg.ensemble_samples)
        Q_dim, D_dim = int(W0.shape[0]), int(W0.shape[1])
        ensemble_raw_pool = max(int(cfg.n_neighbors), min(200, n_train))
        rng = torch.Generator(device=device).manual_seed(int(seed) + 7919)
        mu_h_samples: list[torch.Tensor] = []
        var_h_samples: list[torch.Tensor] = []
        for s in range(S):
            eps = torch.randn(Q_dim, D_dim, generator=rng, device=device, dtype=dtype)
            W_s = W0 + math.sqrt(max(float(cfg.w_var), 0.0)) * eps
            X_neighbors_s, y_resid_neighbors_s, neighbor_idx_s = _projected_neighbor_stack_from_raw_pool(
                X_test_t,
                X_train_t,
                y_train_resid_t,
                W=W_s,
                n_neighbors=int(cfg.n_neighbors),
                raw_pool=ensemble_raw_pool,
            )
            v_g_neighbors_s = v_g_train[neighbor_idx_s] * float(cfg.diag_variance_inflation)
            w_moments_s = {
                "W_mu_anchor": W_s.reshape(1, Q_dim, D_dim).expand(n_test, Q_dim, D_dim).contiguous(),
                "W_var_anchor": torch.zeros(n_test, Q_dim, D_dim, device=device, dtype=dtype),
            }
            init_labels_s = _initial_residual_labels(
                X_test_t, X_neighbors_s, y_resid_neighbors_s, min_inliers=cfg.min_inliers
            )
            defaults_s = _meanw_default_hypers(
                X_test_t,
                X_neighbors_s,
                y_resid_neighbors_s,
                init_labels_s,
                W_s,
                min_inliers=cfg.min_inliers,
            )
            state_s = _run_local_self_cem_with_extras(
                X_anchor=X_test_t,
                X_neighbors=X_neighbors_s,
                y_resid_neighbors=y_resid_neighbors_s,
                init_labels=init_labels_s,
                W_mu_anchor=w_moments_s["W_mu_anchor"],
                W_var_anchor=w_moments_s["W_var_anchor"],
                init_lengthscale=defaults_s["lengthscale"],
                init_signal_var=defaults_s["signal_var"],
                init_noise_var=defaults_s["noise_var"].clamp_min(min_noise_var_per_anchor),
                min_noise_var_per_anchor=min_noise_var_per_anchor,
                extra_noise_diag=v_g_neighbors_s,
                extra_block_cov=None,
                cfg=cfg,
            )
            pred_s = _predict_per_anchor_with_extras(
                X_anchor=X_test_t,
                X_neighbors=X_neighbors_s,
                y_resid_neighbors=y_resid_neighbors_s,
                labels=state_s["labels"],
                W_mu_anchor=w_moments_s["W_mu_anchor"],
                W_var_anchor=w_moments_s["W_var_anchor"],
                lengthscale=state_s["lengthscale"],
                signal_var=state_s["signal_var"],
                noise_var=state_s["noise_var"],
                mean_const=state_s["mean_const"],
                extra_noise_diag=v_g_neighbors_s,
                extra_block_cov=None,
                correction_weight=float(cfg.kcov),
                jitter=cfg.hyperopt.jitter,
                min_inliers=cfg.min_inliers,
            )
            mu_h_samples.append(pred_s["mu"])
            var_h_samples.append(pred_s["var"])
        mu_h_stack = torch.stack(mu_h_samples, dim=0)
        var_h_stack = torch.stack(var_h_samples, dim=0)
        mu_h_test = mu_h_stack.mean(dim=0)
        within = var_h_stack.mean(dim=0)
        between = mu_h_stack.var(dim=0, unbiased=False)
        var_h_test = (within + between).clamp_min(float(cfg.min_noise_var_floor))
        diag["ensemble_samples"] = S
        diag["ensemble_raw_pool"] = int(ensemble_raw_pool)
        diag["ensemble_within_var_mean"] = float(within.mean().item())
        diag["ensemble_between_var_mean"] = float(between.mean().item())

    # --- 7) Combine global + residual into final additive predictive
    # (standardized scale) and rescale back to the original y units.
    mu_y_std = (mu_g_test + mu_h_test).detach().cpu().numpy().reshape(-1)
    var_y_std = (v_g_test + var_h_test).clamp_min(float(cfg.min_noise_var_floor))
    var_y_std_np = var_y_std.detach().cpu().numpy().reshape(-1)
    mu_y = mu_y_std * y_train_std_val + y_train_mean
    var_y = var_y_std_np * (y_train_std_val ** 2)
    sigma_y = np.sqrt(np.maximum(var_y, 1e-12))

    mu_g_orig = mu_g_test.detach().cpu().numpy().reshape(-1) * y_train_std_val + y_train_mean
    var_g_orig = v_g_test.detach().cpu().numpy().reshape(-1) * (y_train_std_val ** 2)
    mu_h_orig = mu_h_test.detach().cpu().numpy().reshape(-1) * y_train_std_val
    var_h_orig = var_h_test.detach().cpu().numpy().reshape(-1) * (y_train_std_val ** 2)

    diag["wall_sec"] = float(time.perf_counter() - t_start)
    diag["final_mu_g_mean"] = float(mu_g_orig.mean())
    diag["final_mu_h_mean"] = float(mu_h_orig.mean())
    diag["final_v_g_mean"] = float(var_g_orig.mean())
    diag["final_v_h_mean"] = float(var_h_orig.mean())
    diag["final_v_y_mean"] = float(var_y.mean())

    return GlobalResidualSelfCEMResult(
        mu_y=mu_y,
        var_y=var_y,
        sigma_y=sigma_y,
        mu_g_test=mu_g_orig,
        var_g_test=var_g_orig,
        mu_h_test=mu_h_orig,
        var_h_test=var_h_orig,
        labels_test=local_state["labels"],
        diagnostics=diag,
    )


__all__ = [
    "GlobalResidualSelfCEMConfig",
    "GlobalResidualSelfCEMResult",
    "VARIANTS",
    "run_global_residual_self_cem",
]
