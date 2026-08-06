"""
Compare transductive and supervised-inductive projection learning on UCI data.

The four main methods are:

* ``lmjgp_transductive``: existing ``train_vi`` + ``predict_vi`` with true test
  inputs used as projection anchors during variational training.
* ``cem_em_transductive``: existing CEM-EM coefficient learner with true test
  inputs used as optimization anchors.
* ``lmjgp_supervised``: train ``q(V)`` only on labeled training anchors, then
  induce ``q(W_*)`` at test inputs by GP conditioning and run projected JumpGP.
  Its training objective is a strict leave-one-anchor predictive VI objective:
  neighbors come only from ``D_base`` and ``y_a`` appears only in
  ``log p(y_a | N(a), W_a)``.
* ``cem_em_supervised``: train coefficient MAP values ``C_A`` on labeled
  training anchors using a leave-one-anchor predictive term, then GP-condition
  and sample ``C_*`` at test inputs before projected JumpGP ensemble prediction.
* ``xgboost``: XGBoost regressor trained on the same bounded training budget.
  Its predictive standard deviation is a constant residual-scale plug-in, so
  RMSE is the primary comparison metric and CRPS/calibration are only rough
  Gaussian-reference diagnostics.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test: all four methods, tiny Wine slice.
    python experiments/uci/compare_inductive_projection.py ^
        --num_exp 1 --datasets "Wine Quality" ^
        --max_test_anchors 12 --max_train_anchors 24 ^
        --n 12 --lmjgp_steps 20 --n_outer 1 --n_m_steps 10 --MC_num 2

    # Larger comparison on the three UCI datasets.
    python experiments/uci/compare_inductive_projection.py ^
        --num_exp 3 ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_test_anchors 128 --max_train_anchors 128 ^
        --n 35 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --MC_num 3 ^
        --lmjgp_supervised_train_mc 2 --scaling_preset uci_empirical ^
        --out_dir experiments/uci/inductive_projection_compare

    # Small-data comparison against XGBoost.
    python experiments/uci/compare_inductive_projection.py ^
        --num_exp 1 --methods "lmjgp_transductive,cem_em_transductive,lmjgp_supervised,cem_em_supervised,xgboost" ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128 ^
        --n 35 --lmjgp_steps 200 --n_outer 5 --n_m_steps 80 --MC_num 3 ^
        --lmjgp_supervised_train_mc 2 --xgb_boots 30 --scaling_preset uci_empirical ^
        --out_dir experiments/uci/inductive_projection_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation import mean_gaussian_crps_torch  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    gp_condition_coefficients,
    induced_W_from_coefficients,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    run_projected_jumpgp_ensemble,
    sample_induced_coefficients,
    train_projection_cem_em,
)
from djgp.variational import predict_vi, qW_from_qV, train_vi  # noqa: E402
from djgp.variational import kl_qp  # noqa: E402
from experiments.uci.uci_data import load_uci_split  # noqa: E402


METHODS_DEFAULT = (
    "lmjgp_transductive,cem_em_transductive,lmjgp_supervised,cem_em_supervised"
)
METHODS_ALLOWED = METHODS_DEFAULT.split(",") + ["xgboost"]

SCALING_PRESETS = {
    "Wine Quality": (True, False),
    "Parkinsons Telemonitoring": (False, False),
    "Appliances Energy Prediction": (True, True),
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subsample(X: np.ndarray, y: np.ndarray, max_n: int | None, seed: int):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n = X.shape[0]
    if max_n is None or max_n >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
    return X[idx], y[idx], idx


def _split_base_anchor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_train_anchors: int,
    seed: int,
):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)
    n = X_train.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 training rows to split base and anchor sets.")
    n_anchor = min(max(1, int(max_train_anchors)), n - 2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    anchor_idx = np.sort(perm[:n_anchor])
    base_idx = np.sort(perm[n_anchor:])
    disjoint = np.intersect1d(base_idx, anchor_idx).size == 0
    return (
        X_train[base_idx],
        y_train[base_idx],
        X_train[anchor_idx],
        y_train[anchor_idx],
        {
            "n_base": int(base_idx.size),
            "n_anchor": int(anchor_idx.size),
            "base_anchor_disjoint": bool(disjoint),
        },
    )


def _apply_scaling(
    X_train: np.ndarray,
    X_base: np.ndarray,
    X_anchor: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_base: np.ndarray,
    y_anchor: np.ndarray,
    y_test: np.ndarray,
    *,
    standardize_x: bool,
    standardize_y: bool,
):
    x_scaler = StandardScaler() if standardize_x else None
    y_scaler = StandardScaler() if standardize_y else None

    if x_scaler is not None:
        x_scaler.fit(X_train)
        X_train_s = x_scaler.transform(X_train)
        X_base_s = x_scaler.transform(X_base)
        X_anchor_s = x_scaler.transform(X_anchor)
        X_test_s = x_scaler.transform(X_test)
    else:
        X_train_s = np.asarray(X_train, dtype=np.float64)
        X_base_s = np.asarray(X_base, dtype=np.float64)
        X_anchor_s = np.asarray(X_anchor, dtype=np.float64)
        X_test_s = np.asarray(X_test, dtype=np.float64)

    y_train_s = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_base_s = np.asarray(y_base, dtype=np.float64).reshape(-1)
    y_anchor_s = np.asarray(y_anchor, dtype=np.float64).reshape(-1)
    y_test_s = np.asarray(y_test, dtype=np.float64).reshape(-1)
    if y_scaler is not None:
        y_scaler.fit(y_train_s.reshape(-1, 1))
        y_train_s = y_scaler.transform(y_train_s.reshape(-1, 1)).reshape(-1)
        y_base_s = y_scaler.transform(y_base_s.reshape(-1, 1)).reshape(-1)
        y_anchor_s = y_scaler.transform(y_anchor_s.reshape(-1, 1)).reshape(-1)
        y_test_s = y_scaler.transform(y_test_s.reshape(-1, 1)).reshape(-1)

    return (
        X_train_s,
        X_base_s,
        X_anchor_s,
        X_test_s,
        y_train_s,
        y_base_s,
        y_anchor_s,
        y_test_s,
        x_scaler,
        y_scaler,
    )


def _build_regions(
    X_anchors: torch.Tensor,
    y_anchors: torch.Tensor | None,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    device: torch.device,
    m1: int,
    Q: int,
    n: int,
    include_anchor_observation: bool = False,
):
    neighborhoods = find_neighborhoods(X_anchors.cpu(), X_pool.cpu(), y_pool.cpu(), M=n)
    regions = []
    X_nb_list = []
    y_nb_list = []
    for i in range(X_anchors.shape[0]):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device).view(-1)
        X_region = X_nb
        y_region = y_nb
        if include_anchor_observation:
            if y_anchors is None:
                raise ValueError("y_anchors is required when include_anchor_observation=True.")
            X_region = torch.cat([X_nb, X_anchors[i : i + 1].to(device)], dim=0)
            y_region = torch.cat([y_nb, y_anchors[i].to(device).view(1)], dim=0)
        regions.append({"X": X_region, "y": y_region, "C": torch.randn(m1, Q, device=device)})
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return regions, X_nb_list, y_nb_list


def _init_lmjgp_params(
    *,
    X_inducing_pool: torch.Tensor,
    X_anchors: torch.Tensor,
    T: int,
    D: int,
    Q: int,
    m1: int,
    m2: int,
    device: torch.device,
):
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    u_params = []
    for _ in range(T):
        u_params.append(
            {
                "U_logit": torch.zeros(1, device=device, requires_grad=True),
                "mu_u": torch.randn(m1, device=device, requires_grad=True),
                "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
                "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
                "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
                "omega": torch.randn(Q + 1, device=device, requires_grad=True),
            }
        )
    X_mean = X_inducing_pool.mean(dim=0)
    X_std = X_inducing_pool.std(dim=0).clamp_min(1e-6)
    hyperparams = {
        "Z": X_mean + torch.randn(m2, D, device=device) * X_std,
        "X_test": X_anchors,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    return V_params, u_params, hyperparams


def _train_lmjgp_on_regions(
    *,
    regions,
    X_anchor_for_qw: torch.Tensor,
    X_inducing_pool: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    num_steps: int,
    lr: float,
    device: torch.device,
):
    T, D = X_anchor_for_qw.shape
    V_params, u_params, hyperparams = _init_lmjgp_params(
        X_inducing_pool=X_inducing_pool,
        X_anchors=X_anchor_for_qw,
        T=T,
        D=D,
        Q=Q,
        m1=m1,
        m2=m2,
        device=device,
    )
    t0 = time.perf_counter()
    V_params, _, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=num_steps,
        log_interval=max(10, num_steps // 4),
    )
    train_time = time.perf_counter() - t0
    return V_params, hyperparams, float(train_time)


def _init_lmjgp_qv_params(
    *,
    X_inducing_pool: torch.Tensor,
    X_anchors: torch.Tensor,
    Q: int,
    m2: int,
    device: torch.device,
):
    T, D = X_anchors.shape
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": (0.1 + 0.1 * torch.rand(m2, Q, D, device=device)).requires_grad_(True),
    }
    X_mean = X_inducing_pool.mean(dim=0)
    X_std = X_inducing_pool.std(dim=0).clamp_min(1e-6)
    hyperparams = {
        "Z": (X_mean + torch.randn(m2, D, device=device) * X_std).requires_grad_(True),
        "X_test": X_anchors,
        "lengthscales": torch.ones(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    return V_params, hyperparams


def _anchor_gp_predictive_nll_batch(
    W_samples: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    X_anchors: torch.Tensor,
    y_anchors: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_signal_var: torch.Tensor,
    log_noise_var: torch.Tensor,
    *,
    pred_var_floor: float,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """MC leave-one-anchor GP predictive NLL.

    ``W_samples`` is ``[M, T, Q, D]``. The anchor target is never included in
    ``X_neighbors`` / ``y_neighbors``; it appears only in the predictive density.
    """
    M, T, Q, _ = W_samples.shape
    n = X_neighbors.shape[1]
    dtype = W_samples.dtype
    device = W_samples.device

    Z_nb = torch.einsum("tnd,mtqd->mtnq", X_neighbors.to(dtype=dtype), W_samples)
    z_anchor = torch.einsum("td,mtqd->mtq", X_anchors.to(dtype=dtype), W_samples)

    ell = torch.exp(log_lengthscale).to(device=device, dtype=dtype).view(1, 1, 1, 1, Q).clamp_min(1e-6)
    signal_var = torch.exp(log_signal_var).to(device=device, dtype=dtype).clamp_min(1e-8)
    noise_var = torch.exp(log_noise_var).to(device=device, dtype=dtype).clamp_min(1e-8)

    diff = (Z_nb.unsqueeze(3) - Z_nb.unsqueeze(2)) / ell
    d2 = (diff * diff).sum(dim=-1)
    K = signal_var * torch.exp(-0.5 * d2)
    eye = torch.eye(n, device=device, dtype=dtype).view(1, 1, n, n)
    Ky = K + (noise_var + jitter) * eye
    Ky = 0.5 * (Ky + Ky.transpose(-1, -2))
    L = torch.linalg.cholesky(Ky.reshape(M * T, n, n))

    y_nb = y_neighbors.to(device=device, dtype=dtype)
    local_mean = y_nb.mean(dim=1)
    y_center = y_nb - local_mean[:, None]
    rhs = y_center.view(1, T, n, 1).expand(M, T, n, 1).reshape(M * T, n, 1)
    alpha = torch.cholesky_solve(rhs, L).reshape(M, T, n)

    d2_star = (((Z_nb - z_anchor.unsqueeze(2)) / ell.squeeze(2)) ** 2).sum(dim=-1)
    k_star = signal_var * torch.exp(-0.5 * d2_star)
    pred_mean = local_mean.view(1, T) + (k_star * alpha).sum(dim=-1)

    v = torch.cholesky_solve(k_star.reshape(M * T, n, 1), L).reshape(M, T, n)
    pred_var = (signal_var + noise_var - (k_star * v).sum(dim=-1)).clamp_min(float(pred_var_floor))
    err2 = (y_anchors.to(device=device, dtype=dtype).view(1, T) - pred_mean).pow(2)
    return 0.5 * (torch.log(2.0 * torch.pi * pred_var) + err2 / pred_var).mean()


def _qv_anchor_predictive_nll(
    *,
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_nb: torch.Tensor,
    y_nb: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_signal_var: torch.Tensor,
    log_noise_var: torch.Tensor,
    eps: torch.Tensor,
    pred_var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized predictive NLL and KL/T for supervised LMJGP."""
    T = X_anchor_t.shape[0]
    mu_W, cov_W = qW_from_qV(
        X_anchor_t,
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    # Use a differentiable Cholesky reparameterization of q(W_a), not a
    # detached diagonal std plug-in.  The current qW_from_qV returns diagonal
    # covariance, but keeping the Cholesky path makes the supervised objective
    # correct if q(W) is later upgraded to a low-rank/full covariance.
    D = cov_W.shape[-1]
    eye = torch.eye(D, device=cov_W.device, dtype=cov_W.dtype).view(1, 1, D, D)
    L_W = torch.linalg.cholesky(cov_W + 1e-8 * eye)
    W_samples = mu_W.unsqueeze(0) + torch.einsum("tqde,mtqe->mtqd", L_W, eps)
    W_samples = W_samples / torch.linalg.norm(W_samples, dim=-1, keepdim=True).clamp_min(1e-6)
    pred_nll = _anchor_gp_predictive_nll_batch(
        W_samples,
        X_nb,
        y_nb,
        X_anchor_t,
        y_anchor_t,
        log_lengthscale,
        log_signal_var,
        log_noise_var,
        pred_var_floor=pred_var_floor,
    )
    kl_v = kl_qp(
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    ) / max(T, 1)
    return pred_nll, kl_v


def _clone_lmjgp_predictive_state(
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    log_lengthscale: torch.Tensor,
    log_signal_var: torch.Tensor,
    log_noise_var: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    vp = {
        "mu_V": V_params["mu_V"].detach().clone().requires_grad_(True),
        "sigma_V": V_params["sigma_V"].detach().clone().requires_grad_(True),
    }
    hp = {
        "Z": hyperparams["Z"].detach().clone().requires_grad_(True),
        "lengthscales": hyperparams["lengthscales"].detach().clone().requires_grad_(True),
        "var_w": hyperparams["var_w"].detach().clone().requires_grad_(True),
        "X_test": hyperparams["X_test"],
    }
    return (
        vp,
        hp,
        log_lengthscale.detach().clone().requires_grad_(True),
        log_signal_var.detach().clone().requires_grad_(True),
        log_noise_var.detach().clone().requires_grad_(True),
    )


def _lmjgp_supervised_gradient_check(
    *,
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_nb: torch.Tensor,
    y_nb: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_signal_var: torch.Tensor,
    log_noise_var: torch.Tensor,
    train_mc: int,
    pred_var_floor: float,
    lr: float,
) -> dict[str, float | bool]:
    """Check that predictive NLL alone gives q(V) gradients and can decrease."""
    vp, hp, ll, lsv, lnv = _clone_lmjgp_predictive_state(
        V_params, hyperparams, log_lengthscale, log_signal_var, log_noise_var,
    )
    params = [vp["mu_V"], vp["sigma_V"], hp["Z"], hp["lengthscales"], hp["var_w"], ll, lsv, lnv]
    opt = torch.optim.Adam(params, lr=min(float(lr), 5e-3))
    eps = torch.randn(
        (train_mc, X_anchor_t.shape[0], vp["mu_V"].shape[1], X_anchor_t.shape[1]),
        device=X_anchor_t.device,
        dtype=X_anchor_t.dtype,
    )

    opt.zero_grad()
    loss0, _ = _qv_anchor_predictive_nll(
        V_params=vp,
        hyperparams=hp,
        X_anchor_t=X_anchor_t,
        y_anchor_t=y_anchor_t,
        X_nb=X_nb,
        y_nb=y_nb,
        log_lengthscale=ll,
        log_signal_var=lsv,
        log_noise_var=lnv,
        eps=eps,
        pred_var_floor=pred_var_floor,
    )
    loss0.backward()
    grad_mu = float(vp["mu_V"].grad.detach().norm().item()) if vp["mu_V"].grad is not None else 0.0
    grad_sigma = float(vp["sigma_V"].grad.detach().norm().item()) if vp["sigma_V"].grad is not None else 0.0
    grad_z = float(hp["Z"].grad.detach().norm().item()) if hp["Z"].grad is not None else 0.0
    opt.step()

    with torch.no_grad():
        vp["sigma_V"].clamp_(min=1e-3, max=10.0)
        hp["var_w"].clamp_(min=1e-3, max=10.0)
        hp["lengthscales"].clamp_(min=1e-4, max=1e3)

    loss1, _ = _qv_anchor_predictive_nll(
        V_params=vp,
        hyperparams=hp,
        X_anchor_t=X_anchor_t,
        y_anchor_t=y_anchor_t,
        X_nb=X_nb,
        y_nb=y_nb,
        log_lengthscale=ll,
        log_signal_var=lsv,
        log_noise_var=lnv,
        eps=eps,
        pred_var_floor=pred_var_floor,
    )
    before = float(loss0.detach().item())
    after = float(loss1.detach().item())
    return {
        "predictive_only_loss_before": before,
        "predictive_only_loss_after_one_step": after,
        "predictive_only_loss_decreased": bool(after <= before),
        "mu_V_grad_norm": grad_mu,
        "sigma_V_grad_norm": grad_sigma,
        "Z_grad_norm": grad_z,
        "has_qV_gradient": bool(grad_mu > 0 or grad_sigma > 0),
    }


def _train_lmjgp_supervised_predictive(
    *,
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_anchor_neighbors: list[torch.Tensor],
    y_anchor_neighbors: list[torch.Tensor],
    X_inducing_t: torch.Tensor,
    Q: int,
    m2: int,
    num_steps: int,
    lr: float,
    train_mc: int,
    kl_weight: float,
    pred_var_floor: float,
    grad_check: bool,
    device: torch.device,
):
    """Train q(V) with a strict leave-one-anchor predictive objective."""
    T = X_anchor_t.shape[0]
    X_nb = torch.stack([x.to(device) for x in X_anchor_neighbors], dim=0)
    y_nb = torch.stack([y.to(device).view(-1) for y in y_anchor_neighbors], dim=0)
    V_params, hyperparams = _init_lmjgp_qv_params(
        X_inducing_pool=X_inducing_t,
        X_anchors=X_anchor_t,
        Q=Q,
        m2=m2,
        device=device,
    )
    y_var = torch.var(y_anchor_t.detach()).clamp_min(1e-4)
    log_lengthscale = torch.zeros(Q, device=device, requires_grad=True)
    log_signal_var = torch.log(y_var.detach()).clone().requires_grad_(True)
    log_noise_var = torch.log((0.05 * y_var).clamp_min(1e-4).detach()).clone().requires_grad_(True)

    params = [
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["Z"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
        log_lengthscale,
        log_signal_var,
        log_noise_var,
    ]
    optimizer = torch.optim.Adam(params, lr=lr)
    best = None
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    gradient_check: dict[str, float | bool] = {}
    if grad_check:
        gradient_check = _lmjgp_supervised_gradient_check(
            V_params=V_params,
            hyperparams=hyperparams,
            X_anchor_t=X_anchor_t,
            y_anchor_t=y_anchor_t,
            X_nb=X_nb,
            y_nb=y_nb,
            log_lengthscale=log_lengthscale,
            log_signal_var=log_signal_var,
            log_noise_var=log_noise_var,
            train_mc=train_mc,
            pred_var_floor=pred_var_floor,
            lr=lr,
        )
        print(
            "[LMJGP supervised grad-check] "
            f"pred_only={gradient_check['predictive_only_loss_before']:.4f}"
            f"->{gradient_check['predictive_only_loss_after_one_step']:.4f} "
            f"mu_grad={gradient_check['mu_V_grad_norm']:.3e} "
            f"sigma_grad={gradient_check['sigma_V_grad_norm']:.3e} "
            f"Z_grad={gradient_check['Z_grad_norm']:.3e} "
            f"has_qV_grad={gradient_check['has_qV_gradient']}",
            flush=True,
        )
    t0 = time.perf_counter()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        eps = torch.randn((train_mc, T, Q, X_anchor_t.shape[1]), device=device, dtype=X_anchor_t.dtype)
        pred_nll, kl_v = _qv_anchor_predictive_nll(
            V_params=V_params,
            hyperparams=hyperparams,
            X_anchor_t=X_anchor_t,
            y_anchor_t=y_anchor_t,
            X_nb=X_nb,
            y_nb=y_nb,
            log_lengthscale=log_lengthscale,
            log_signal_var=log_signal_var,
            log_noise_var=log_noise_var,
            eps=eps,
            pred_var_floor=pred_var_floor,
        )
        loss = pred_nll + float(kl_weight) * kl_v
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            V_params["sigma_V"].clamp_(min=1e-3, max=10.0)
            hyperparams["var_w"].clamp_(min=1e-3, max=10.0)
            hyperparams["lengthscales"].clamp_(min=1e-4, max=1e3)
            log_lengthscale.clamp_(min=-6.0, max=6.0)
            log_signal_var.clamp_(min=-10.0, max=10.0)
            log_noise_var.clamp_(min=-12.0, max=8.0)

        loss_val = float(loss.detach().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best = {
                "V_params": {
                    "mu_V": V_params["mu_V"].detach().clone().requires_grad_(True),
                    "sigma_V": V_params["sigma_V"].detach().clone().requires_grad_(True),
                },
                "hyperparams": {
                    "Z": hyperparams["Z"].detach().clone().requires_grad_(True),
                    "lengthscales": hyperparams["lengthscales"].detach().clone().requires_grad_(True),
                    "var_w": hyperparams["var_w"].detach().clone().requires_grad_(True),
                    "X_test": X_anchor_t,
                },
                "log_lengthscale": log_lengthscale.detach().clone(),
                "log_signal_var": log_signal_var.detach().clone(),
                "log_noise_var": log_noise_var.detach().clone(),
            }
        if step == 1 or step % max(10, num_steps // 4) == 0:
            row = {
                "step": float(step),
                "loss": loss_val,
                "pred_nll": float(pred_nll.detach().item()),
                "kl_v_per_anchor": float(kl_v.detach().item()),
            }
            history.append(row)
            print(
                f"[LMJGP supervised step {step}/{num_steps}] "
                f"loss={row['loss']:.4f} pred_nll={row['pred_nll']:.4f} "
                f"kl/T={row['kl_v_per_anchor']:.4f}",
                flush=True,
            )

    train_time = time.perf_counter() - t0
    if best is None:
        best = {
            "V_params": V_params,
            "hyperparams": hyperparams,
            "log_lengthscale": log_lengthscale.detach(),
            "log_signal_var": log_signal_var.detach(),
            "log_noise_var": log_noise_var.detach(),
        }
    diag = {
        "train_time_sec": float(train_time),
        "best_loss": float(best_loss),
        "train_mc": int(train_mc),
        "kl_weight": float(kl_weight),
        "pred_var_floor": float(pred_var_floor),
        "local_gp_lengthscale": [float(v) for v in torch.exp(best["log_lengthscale"]).detach().cpu()],
        "local_gp_signal_var": float(torch.exp(best["log_signal_var"]).detach().cpu().item()),
        "local_gp_noise_var": float(torch.exp(best["log_noise_var"]).detach().cpu().item()),
        "history": history,
        "strict_leave_one_anchor_objective": True,
        "kl_scaled_by_n_anchors": True,
        "anchor_label_used_as_region_observation": False,
        "training_neighbor_pool": "D_base",
        "gradient_check": gradient_check,
    }
    return best["V_params"], best["hyperparams"], diag


def _predict_lmjgp_from_qv(
    *,
    regions,
    V_params,
    hyperparams,
    X_query: torch.Tensor,
    y_query_np: np.ndarray,
    MC_num: int,
):
    pred_hyp = dict(hyperparams)
    pred_hyp["X_test"] = X_query
    t0 = time.perf_counter()
    mu_pred, var_pred = predict_vi(regions, V_params, pred_hyp, M=MC_num)
    elapsed = time.perf_counter() - t0
    sigma = torch.sqrt(torch.clamp(var_pred, min=1e-12))
    mu_np = mu_pred.detach().cpu().numpy()
    sig_np = sigma.detach().cpu().numpy()
    return mu_np, sig_np, float(elapsed)


def _pack_metrics(y_eval: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    y_eval = np.asarray(y_eval, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - y_eval) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.from_numpy(y_eval).float(),
            torch.from_numpy(mu).float(),
            torch.from_numpy(sigma).float(),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y_eval, mu, sigma)


def _pack_metrics_maybe_original_y(
    y_scaler: StandardScaler | None,
    y_eval_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    wall_sec: float,
) -> list[float]:
    if y_scaler is None:
        return _pack_metrics(y_eval_scaled, mu_scaled, sigma_scaled, wall_sec)
    scale = float(np.squeeze(y_scaler.scale_))
    if not np.isfinite(scale) or scale < 1e-20:
        scale = 1.0
    y_eval = y_scaler.inverse_transform(np.asarray(y_eval_scaled).reshape(-1, 1)).reshape(-1)
    mu = y_scaler.inverse_transform(np.asarray(mu_scaled).reshape(-1, 1)).reshape(-1)
    sigma = np.asarray(sigma_scaled, dtype=np.float64).reshape(-1) * abs(scale)
    return _pack_metrics(y_eval, mu, sigma, wall_sec)


def _run_lmjgp_transductive(
    *,
    X_test_t: torch.Tensor,
    y_test_np: np.ndarray,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    X_inducing_t: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    steps: int,
    lr: float,
    MC_num: int,
    device: torch.device,
):
    regions, _, _ = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    V_params, hyperparams, train_time = _train_lmjgp_on_regions(
        regions=regions,
        X_anchor_for_qw=X_test_t,
        X_inducing_pool=X_inducing_t,
        Q=Q,
        m1=m1,
        m2=m2,
        num_steps=steps,
        lr=lr,
        device=device,
    )
    mu, sig, pred_time = _predict_lmjgp_from_qv(
        regions=regions,
        V_params=V_params,
        hyperparams=hyperparams,
        X_query=X_test_t,
        y_query_np=y_test_np,
        MC_num=MC_num,
    )
    with torch.no_grad():
        mu_W, _ = qW_from_qV(
            X_test_t,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
    return mu, sig, pred_time, {"train_time_sec": train_time, "mu_W_shape": list(mu_W.shape)}


def _run_lmjgp_supervised(
    *,
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test_np: np.ndarray,
    X_base_t: torch.Tensor,
    y_base_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    X_inducing_t: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    steps: int,
    lr: float,
    MC_num: int,
    train_mc: int,
    kl_weight: float,
    pred_var_floor: float,
    grad_check: bool,
    device: torch.device,
):
    _, X_anchor_nb_list, y_anchor_nb_list = _build_regions(
        X_anchor_t,
        None,
        X_base_t,
        y_base_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    V_params, hyperparams, train_diag = _train_lmjgp_supervised_predictive(
        X_anchor_t=X_anchor_t,
        y_anchor_t=y_anchor_t,
        X_anchor_neighbors=X_anchor_nb_list,
        y_anchor_neighbors=y_anchor_nb_list,
        X_inducing_t=X_inducing_t,
        Q=Q,
        m2=m2,
        num_steps=steps,
        lr=lr,
        train_mc=train_mc,
        kl_weight=kl_weight,
        pred_var_floor=pred_var_floor,
        grad_check=grad_check,
        device=device,
    )
    test_regions, _, _ = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    mu, sig, pred_time = _predict_lmjgp_from_qv(
        regions=test_regions,
        V_params=V_params,
        hyperparams=hyperparams,
        X_query=X_test_t,
        y_query_np=y_test_np,
        MC_num=MC_num,
    )
    with torch.no_grad():
        mu_W, _ = qW_from_qV(
            X_test_t,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
    diag = dict(train_diag)
    diag["mu_W_shape"] = list(mu_W.shape)
    return mu, sig, pred_time, diag


def _run_cem_em_transductive(
    *,
    X_test_t: torch.Tensor,
    y_test_np: np.ndarray,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    V_t: torch.Tensor,
    W_pls_t: torch.Tensor,
    Q: int,
    m1: int,
    n: int,
    cfg: CemEmConfig,
    init_C_scale: float,
    coeff_scale: float,
    row_normalize_W: bool,
    device: torch.device,
):
    _, X_nb_list, y_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    model = CoeffProjection(
        T=X_test_t.shape[0],
        Q=Q,
        D=X_test_t.shape[1],
        V=V_t,
        W0_init=W_pls_t,
        learn_W0=True,
        init_C_scale=init_C_scale,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).to(device)
    cfg_local = CemEmConfig(**{**cfg.__dict__, "lambda_anchor_pred": 0.0})
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test_t,
        device=device,
        cfg=cfg_local,
        label="cem_em_transductive",
    )
    out = run_projected_jumpgp(
        res.W,
        X_test_t,
        X_nb_list,
        y_nb_list,
        device=device,
        progress=True,
        desc="cem_em_transductive",
    )
    diag = per_anchor_W_diagnostics(res.W, W_ref=W_pls_t)
    diag.update(
        {
            "train_time_sec": float(res.train_time_sec),
            "jgp_elapsed_sec": float(out["elapsed_sec"]),
            "best_outer": int(res.best_outer),
        }
    )
    return out["mu"], out["sigma"], float(out["elapsed_sec"]), diag


def _run_cem_em_supervised(
    *,
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test_np: np.ndarray,
    X_base_t: torch.Tensor,
    y_base_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    V_t: torch.Tensor,
    W_pls_t: torch.Tensor,
    Q: int,
    m1: int,
    n: int,
    cfg: CemEmConfig,
    init_C_scale: float,
    coeff_scale: float,
    row_normalize_W: bool,
    MC_num: int,
    coeff_gp_ridge: float,
    coeff_gp_lengthscale: float | None,
    seed: int,
    device: torch.device,
):
    _, X_anchor_nb_list, y_anchor_nb_list = _build_regions(
        X_anchor_t,
        None,
        X_base_t,
        y_base_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    model = CoeffProjection(
        T=X_anchor_t.shape[0],
        Q=Q,
        D=X_anchor_t.shape[1],
        V=V_t,
        W0_init=W_pls_t,
        learn_W0=True,
        init_C_scale=init_C_scale,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).to(device)
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_anchor_nb_list,
        y_neighbors_list=y_anchor_nb_list,
        X_anchors=X_anchor_t,
        y_anchors=y_anchor_t,
        device=device,
        cfg=cfg,
        label="cem_em_supervised",
    )
    cond = gp_condition_coefficients(
        X_anchor_t,
        res.C,
        X_test_t,
        ridge=coeff_gp_ridge,
        lengthscale=coeff_gp_lengthscale,
    )
    C_samples = sample_induced_coefficients(
        cond["mean"],
        cond["var"],
        n_samples=MC_num,
        seed=seed,
    )
    W_samples = induced_W_from_coefficients(
        C_samples,
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    )
    _, X_test_nb_list, y_test_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    out = run_projected_jumpgp_ensemble(
        W_samples,
        X_test_t,
        X_test_nb_list,
        y_test_nb_list,
        device=device,
        progress=True,
        desc="cem_em_supervised",
    )
    W_mean = induced_W_from_coefficients(
        cond["mean"],
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    )
    diag = per_anchor_W_diagnostics(W_mean, W_ref=W_pls_t)
    diag.update(
        {
            "train_time_sec": float(res.train_time_sec),
            "jgp_elapsed_sec": float(out["elapsed_sec"]),
            "best_outer": int(res.best_outer),
            "coeff_gp_lengthscale": float(cond["lengthscale"]),
            "coeff_gp_ridge": float(cond["ridge"]),
            "mean_coeff_gp_var": float(cond["var"].mean().detach().cpu().item()),
        }
    )
    return out["mu"], out["sigma"], float(out["elapsed_sec"]), diag


def _run_xgboost_bootstrap(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int,
    seed: int,
):
    raise RuntimeError('xgboost baseline removed from the public release')
    # (unreachable; kept for signature compatibility)
    rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        n_bootstrap=n_bootstrap,
        random_state=seed,
    )
    diag = {
        "train_time_sec": float(elapsed),
        "bootstrap_models": int(n_bootstrap),
        "scaled_rmse": float(rmse),
        "scaled_crps": float(crps),
        "sigma_a2": float(details.get("sigma_a2", np.nan)),
    }
    return details["mu"], details["sigma"], float(elapsed), diag


def parse_args():
    p = argparse.ArgumentParser(description="UCI transductive vs supervised-inductive DJGP projection comparison.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--datasets", type=str, default="Wine Quality")
    p.add_argument("--methods", type=str, default=METHODS_DEFAULT)
    p.add_argument("--max_total_train", type=int, default=None,
                   help="Optional cap on total labeled training rows before base/anchor splitting.")
    p.add_argument("--max_test_anchors", type=int, default=128)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--neighbor_pool", type=str, default="base", choices=("base", "train"),
                   help="Pool for final test neighbourhoods. 'base' keeps anchor labels out of local fits.")
    p.add_argument("--scaling_preset", type=str, default="global", choices=("global", "uci_empirical"),
                   help="Use global standardize_x/y flags, or empirical per-UCI scaling preferences.")
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=1)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", type=int, default=35)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lmjgp_supervised_grad_check", type=int, default=1,
                   help="If 1, run a predictive-only q(V) gradient check before supervised LMJGP training.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_scale", type=float, default=0.0)
    p.add_argument("--lambda_anchor_pred", type=float, default=1.0)
    p.add_argument("--anchor_pred_var_floor", type=float, default=0.05)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--row_normalize_W", type=int, default=1)
    p.add_argument("--normalize_smoothness", type=int, default=1)
    p.add_argument("--track_best_W", type=int, default=1)
    p.add_argument("--coeff_gp_ridge", type=float, default=1e-4)
    p.add_argument("--coeff_gp_lengthscale", type=float, default=None)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--out_dir", type=str, default="experiments/uci/inductive_projection_compare")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "method", "metric", "value"])
        for row in rows:
            for i, key in enumerate(RESULT_ROW_KEYS):
                w.writerow([row["dataset"], row["seed"], row["method"], key, row["metrics"][i]])


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = set(METHODS_ALLOWED)
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json) if args.out_json else out_dir / f"results_{stamp}.json"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"results_{stamp}.csv"

    base_cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        lambda_anchor_pred=args.lambda_anchor_pred,
        anchor_pred_var_floor=args.anchor_pred_var_floor,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=True,
    )

    results: dict = {
        "_meta": {
            "methods": methods,
            "num_exp": int(args.num_exp),
            "datasets": [s.strip() for s in args.datasets.split(",") if s.strip()],
            "max_total_train": args.max_total_train,
            "max_test_anchors": args.max_test_anchors,
            "max_train_anchors": args.max_train_anchors,
            "neighbor_pool": args.neighbor_pool,
            "scaling_preset": args.scaling_preset,
            "global_standardize_x_flag": bool(args.standardize_x),
            "global_standardize_y_flag": bool(args.standardize_y),
            "n": args.n,
            "m1": args.m1,
            "m2": args.m2,
            "MC_num": args.MC_num,
            "lmjgp_supervised_train_mc": args.lmjgp_supervised_train_mc,
            "lmjgp_supervised_kl_weight": args.lmjgp_supervised_kl_weight,
            "lmjgp_supervised_pred_var_floor": args.lmjgp_supervised_pred_var_floor,
            "lmjgp_supervised_grad_check": bool(args.lmjgp_supervised_grad_check),
            "xgb_boots": int(args.xgb_boots),
            "cem_em": sanitize_for_json(base_cfg.__dict__),
        }
    }
    csv_rows: list[dict] = []
    dataset_list = results["_meta"]["datasets"]

    for dataset_name in dataset_list:
        results[dataset_name] = {}
        for seed in range(args.num_exp):
            _set_seed(seed)
            print(f"\n{'=' * 72}\n{dataset_name} | seed={seed}\n{'=' * 72}", flush=True)
            X_train_full, X_test_full, y_train_full, y_test_full, K_default = load_uci_split(dataset_name, seed)
            X_train_full, y_train_full, train_idx = _subsample(
                X_train_full,
                y_train_full,
                args.max_total_train,
                seed + 500,
            )
            if args.scaling_preset == "uci_empirical":
                standardize_x, standardize_y = SCALING_PRESETS.get(
                    dataset_name, (bool(args.standardize_x), bool(args.standardize_y))
                )
            else:
                standardize_x, standardize_y = bool(args.standardize_x), bool(args.standardize_y)
            X_test_sub, y_test_sub, test_idx = _subsample(X_test_full, y_test_full, args.max_test_anchors, seed + 1000)
            X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
                X_train_full,
                y_train_full,
                max_train_anchors=args.max_train_anchors,
                seed=seed + 2000,
            )
            (
                X_train_s,
                X_base_s,
                X_anchor_s,
                X_test_s,
                y_train_s,
                y_base_s,
                y_anchor_s,
                y_test_s,
                _,
                y_scaler,
            ) = _apply_scaling(
                X_train_full,
                X_base,
                X_anchor,
                X_test_sub,
                y_train_full,
                y_base,
                y_anchor,
                y_test_sub,
                standardize_x=standardize_x,
                standardize_y=standardize_y,
            )

            Q = int(args.Q if args.Q is not None else K_default)
            X_train_t = torch.from_numpy(X_train_s).float().to(device)
            y_train_t = torch.from_numpy(y_train_s).float().to(device)
            X_base_t = torch.from_numpy(X_base_s).float().to(device)
            y_base_t = torch.from_numpy(y_base_s).float().to(device)
            X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
            y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
            X_test_t = torch.from_numpy(X_test_s).float().to(device)

            if args.neighbor_pool == "train":
                X_pool_t, y_pool_t = X_train_t, y_train_t
            else:
                X_pool_t, y_pool_t = X_base_t, y_base_t

            V_np, V_info = build_V_basis(
                X_train_s,
                y_train_s,
                n_pls=args.n_pls,
                n_pca=args.n_pca,
                n_random=args.n_random,
                seed=seed,
            )
            W_pls_np = compute_pls_W0(X_train_s, y_train_s, Q=Q)
            V_t = torch.from_numpy(V_np).float().to(device)
            W_pls_t = torch.from_numpy(W_pls_np).float().to(device)

            seed_block = {
                "dataset": dataset_name,
                "seed": int(seed),
                "Q": Q,
                "D": int(X_train_s.shape[1]),
                "split": {
                    **split_info,
                    "n_train_total": int(X_train_full.shape[0]),
                    "train_indices": train_idx.tolist(),
                    "n_test": int(X_test_t.shape[0]),
                    "test_indices": test_idx.tolist(),
                },
                "scaling": {"standardize_x": bool(standardize_x), "standardize_y": bool(standardize_y)},
                "V_basis": V_info,
                "metrics": {},
                "diagnostics": {},
            }

            method_fns = {
                "lmjgp_transductive": lambda: _run_lmjgp_transductive(
                    X_test_t=X_test_t,
                    y_test_np=y_test_s,
                    X_pool_t=X_pool_t,
                    y_pool_t=y_pool_t,
                    X_inducing_t=X_train_t,
                    Q=Q,
                    m1=args.m1,
                    m2=args.m2,
                    n=args.n,
                    steps=args.lmjgp_steps,
                    lr=args.lr,
                    MC_num=args.MC_num,
                    device=device,
                ),
                "lmjgp_supervised": lambda: _run_lmjgp_supervised(
                    X_anchor_t=X_anchor_t,
                    y_anchor_t=y_anchor_t,
                    X_test_t=X_test_t,
                    y_test_np=y_test_s,
                    X_base_t=X_base_t,
                    y_base_t=y_base_t,
                    X_pool_t=X_pool_t,
                    y_pool_t=y_pool_t,
                    X_inducing_t=X_train_t,
                    Q=Q,
                    m1=args.m1,
                    m2=args.m2,
                    n=args.n,
                    steps=args.lmjgp_steps,
                    lr=args.lr,
                    MC_num=args.MC_num,
                    train_mc=args.lmjgp_supervised_train_mc,
                    kl_weight=args.lmjgp_supervised_kl_weight,
                    pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                    grad_check=bool(args.lmjgp_supervised_grad_check),
                    device=device,
                ),
                "cem_em_transductive": lambda: _run_cem_em_transductive(
                    X_test_t=X_test_t,
                    y_test_np=y_test_s,
                    X_pool_t=X_pool_t,
                    y_pool_t=y_pool_t,
                    V_t=V_t,
                    W_pls_t=W_pls_t,
                    Q=Q,
                    m1=args.m1,
                    n=args.n,
                    cfg=base_cfg,
                    init_C_scale=args.init_C_scale,
                    coeff_scale=args.coeff_scale,
                    row_normalize_W=bool(args.row_normalize_W),
                    device=device,
                ),
                "cem_em_supervised": lambda: _run_cem_em_supervised(
                    X_anchor_t=X_anchor_t,
                    y_anchor_t=y_anchor_t,
                    X_test_t=X_test_t,
                    y_test_np=y_test_s,
                    X_base_t=X_base_t,
                    y_base_t=y_base_t,
                    X_pool_t=X_pool_t,
                    y_pool_t=y_pool_t,
                    V_t=V_t,
                    W_pls_t=W_pls_t,
                    Q=Q,
                    m1=args.m1,
                    n=args.n,
                    cfg=base_cfg,
                    init_C_scale=args.init_C_scale,
                    coeff_scale=args.coeff_scale,
                    row_normalize_W=bool(args.row_normalize_W),
                    MC_num=args.MC_num,
                    coeff_gp_ridge=args.coeff_gp_ridge,
                    coeff_gp_lengthscale=args.coeff_gp_lengthscale,
                    seed=seed + 3000,
                    device=device,
                ),
                "xgboost": lambda: _run_xgboost_bootstrap(
                    X_train=X_train_s,
                    y_train=y_train_s,
                    X_test=X_test_s,
                    y_test=y_test_s,
                    n_bootstrap=args.xgb_boots,
                    seed=seed,
                ),
            }

            for method in methods:
                print(f"\n[{dataset_name} seed={seed}] running {method}", flush=True)
                try:
                    t0 = time.perf_counter()
                    mu, sig, pred_wall, diag = method_fns[method]()
                    total_wall = time.perf_counter() - t0
                    row = _pack_metrics_maybe_original_y(y_scaler, y_test_s, mu, sig, pred_wall)
                    diag = dict(diag)
                    diag["total_wall_sec"] = float(total_wall)
                    seed_block["metrics"][method] = row
                    seed_block["diagnostics"][method] = sanitize_for_json(diag)
                    csv_rows.append({"dataset": dataset_name, "seed": seed, "method": method, "metrics": row})
                    print(
                        f"  {method}: rmse={row[0]:.4f} crps={row[1]:.4f} "
                        f"pred_wall={row[2]:.2f}s total={total_wall:.2f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [error] {method}: {exc}", flush=True)
                    seed_block["metrics"][method] = [float("nan")] * len(RESULT_ROW_KEYS)
                    seed_block["diagnostics"][method] = {"error": str(exc)}

            results[dataset_name][str(seed)] = seed_block
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(results), f, indent=2)
            _write_csv(out_csv, csv_rows)
            print(f"\nPartial outputs saved to {out_json} and {out_csv}", flush=True)

    print(f"\nSaved JSON -> {out_json}\nSaved CSV  -> {out_csv}", flush=True)
    return results


if __name__ == "__main__":
    main()
