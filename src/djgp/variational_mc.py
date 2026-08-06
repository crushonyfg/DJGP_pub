"""MC-ELBO variational inference for LMJGP (Choice B: per-sample optimization).

How to run the self-test from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
    python djgp/variational_mc.py

Key idea
--------
Instead of a single shared gate parameter nu_j for all W_j samples, the
MC-ELBO samples W_j^{(s)} from q(R)->p(W|R), then **independently optimizes**
the local variational parameters (gate nu, inducing u, noise, etc.) for each
sample via a few inner gradient steps.  The gate can therefore learn the correct
split boundary *for that particular projection*, fixing the problem where the
original ELBO's gate degrades to pi~0.5 under high q(W) variance.

This module provides:
- ``compute_mc_elbo``: the MC-ELBO objective
- ``train_mc_vi``: training loop that optimizes q(R) params via reparameterized
  gradients through the MC-ELBO
- ``predict_mc_vi``: prediction that is *consistent* with the training objective
  (sample W, run JGP-CEM per sample, moment-match)
- ``predict_mc_vi_direct``: direct VI prediction using per-sample optimized local
  params (no JGP-CEM refit)
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_LOG_2PI = math.log(2 * math.pi)

# Gauss-Hermite quadrature for sigmoid expectations
_GH_POINTS = 20
_nodes_np, _weights_np = np.polynomial.hermite.hermgauss(_GH_POINTS)
_gh_nodes = torch.from_numpy(_nodes_np).to(device)
_gh_weights = torch.from_numpy(_weights_np).to(device)
_gh_factor = 1.0 / math.sqrt(math.pi)

JITTER = 1e-5


# ---------------------------------------------------------------------------
# Shared kernel helpers
# ---------------------------------------------------------------------------

def _safe_cholesky(K: torch.Tensor, jitter: float = JITTER) -> torch.Tensor:
    n = K.size(-1)
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    K = 0.5 * (K + K.transpose(-2, -1)) + jitter * I
    try:
        return torch.linalg.cholesky(K)
    except RuntimeError:
        for scale in [10, 100, 1000, 10000]:
            try:
                return torch.linalg.cholesky(K + (jitter * scale) * I)
            except RuntimeError:
                continue
        return torch.linalg.cholesky(K + 1e-2 * I)


# ---------------------------------------------------------------------------
# q(W) from q(R) — same as variational.py but self-contained
# ---------------------------------------------------------------------------

def qW_from_qR(
    X_test: torch.Tensor,  # [T, D]
    Z: torch.Tensor,       # [m2, D]
    mu_V: torch.Tensor,    # [m2, Q, D]
    sigma_V: torch.Tensor, # [m2, Q, D]
    lengthscales: torch.Tensor,  # [Q]
    var_w: torch.Tensor,   # scalar or [Q]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute q(W_j) = N(mu_W_j, diag(sigma_W_j^2)) for each test point j."""
    T, D = X_test.shape
    m2, Q, _ = mu_V.shape
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)  # [m2, m2]
    XZ2 = (X_test.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)  # [T, m2]
    mu_W = torch.empty((T, Q, D), device=X_test.device)
    sigma_W = torch.empty((T, Q, D), device=X_test.device)
    for q in range(Q):
        l = lengthscales[q]
        Kzz = var_w * torch.exp(-0.5 * ZZ2 / (l ** 2))
        Kxz = var_w * torch.exp(-0.5 * XZ2 / (l ** 2))
        L = _safe_cholesky(Kzz)
        Kzz_inv = torch.cholesky_inverse(L)
        A = Kxz @ Kzz_inv  # [T, m2]
        diag_cross = (A * Kxz).sum(dim=1)  # [T]
        U = Kzz_inv @ Kxz.t()  # [m2, T]
        for d in range(D):
            mu_W[:, q, d] = A @ mu_V[:, q, d]
            s2 = sigma_V[:, q, d].pow(2).unsqueeze(1)  # [m2, 1]
            diag3 = (U.pow(2) * s2).sum(dim=0)  # [T]
            sigma_W[:, q, d] = torch.sqrt(var_w - diag_cross + diag3 + 1e-12)
    return mu_W, sigma_W  # both [T, Q, D]


def sample_W(mu_W: torch.Tensor, sigma_W: torch.Tensor, S: int) -> torch.Tensor:
    """Draw S samples from the diagonal Gaussian q(W_j).

    Returns: [S, T, Q, D]
    """
    T, Q, D = mu_W.shape
    eps = torch.randn(S, T, Q, D, device=mu_W.device)
    W = mu_W.unsqueeze(0) + eps * sigma_W.unsqueeze(0)
    # Normalize rows to unit norm
    norms = W.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    return W / norms


# ---------------------------------------------------------------------------
# KL(q(R) || p(R)) — same as kl_qp in variational.py
# ---------------------------------------------------------------------------

def kl_qR(
    Z: torch.Tensor,
    mu_V: torch.Tensor,
    sigma_V: torch.Tensor,
    lengthscales: torch.Tensor,
    var_w: torch.Tensor,
) -> torch.Tensor:
    m2, Q, D = mu_V.shape
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    kl = torch.tensor(0.0, device=Z.device)
    for q in range(Q):
        ell = lengthscales[q]
        Kq = var_w * torch.exp(-0.5 * d2 / (ell ** 2))
        L = _safe_cholesky(Kq)
        Kq_inv = torch.cholesky_inverse(L)
        diag_inv = torch.diagonal(Kq_inv)
        logdet_Kq = 2.0 * torch.log(torch.diagonal(L)).sum()
        for d in range(D):
            mu_qd = mu_V[:, q, d]
            s2 = sigma_V[:, q, d].pow(2)
            trace_term = (diag_inv * s2).sum()
            quad_term = mu_qd @ (Kq_inv @ mu_qd)
            logdet_S = torch.log(s2 + 1e-12).sum()
            kl += 0.5 * (trace_term + quad_term - m2 + logdet_Kq - logdet_S)
    return kl


# ---------------------------------------------------------------------------
# Per-sample local ELBO (for a single region j with a fixed W_j)
# ---------------------------------------------------------------------------

def _local_elbo_one_region(
    X_j: torch.Tensor,    # [n, D] neighbor features
    y_j: torch.Tensor,    # [n]
    W_j: torch.Tensor,    # [Q, D] — fixed for this sample
    x_test_j: torch.Tensor,  # [D]
    C_j: torch.Tensor,    # [m1, Q]
    mu_u_j: torch.Tensor, # [m1]
    raw_L_j: torch.Tensor,  # [m1, m1] — raw lower-tri for Sigma_u
    sigma_noise_j: torch.Tensor,  # scalar
    sigma_k_j: torch.Tensor,      # scalar
    omega_j: torch.Tensor,         # [Q+1]
    U_logit_j: torch.Tensor,       # scalar
) -> torch.Tensor:
    """Compute the local ELBO for region j given a fixed W_j sample.

    This includes the data log-likelihood term and KL(q(u_j)||p(u_j)).
    """
    n, D = X_j.shape
    Q = W_j.shape[0]
    m1 = C_j.shape[0]

    # Project data through W_j
    Z_proj = X_j @ W_j.T  # [n, Q]

    # --- Build Kuu, Kfu, Kff_diag in projected space ---
    d2_uu = (C_j.unsqueeze(1) - C_j.unsqueeze(0)).pow(2).sum(-1)  # [m1, m1]
    Kuu = sigma_k_j ** 2 * torch.exp(-0.5 * d2_uu)
    L_kuu = _safe_cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(L_kuu)

    d2_fu = (Z_proj.unsqueeze(1) - C_j.unsqueeze(0)).pow(2).sum(-1)  # [n, m1]
    Kfu = sigma_k_j ** 2 * torch.exp(-0.5 * d2_fu)  # [n, m1]

    # Build PSD Sigma_u from raw_L
    L_sigma = torch.tril(raw_L_j)
    diag = torch.diagonal(L_sigma)
    pos_diag = F.softplus(diag) + 1e-6
    L_sigma = L_sigma - torch.diag(diag) + torch.diag(pos_diag)
    Sigma_u = L_sigma @ L_sigma.T

    # Predictive mean/var: E_q[f] and Var_q[f]
    alpha = Kuu_inv @ mu_u_j  # [m1]
    E_f = Kfu @ alpha  # [n]

    # Var_q[f_i] = k(z_i,z_i) - kfu Kuu^{-1} kuf + kfu Kuu^{-1} Sigma_u Kuu^{-1} kuf
    Kff_diag = sigma_k_j ** 2 * torch.ones(n, device=X_j.device)
    v = Kuu_inv @ Kfu.T  # [m1, n]
    var_prior = Kff_diag - (Kfu * (Kfu @ Kuu_inv)).sum(dim=1)  # [n]
    Sigma_u_tilde = Sigma_u + mu_u_j.unsqueeze(1) * mu_u_j.unsqueeze(0)
    var_posterior = (v * (Sigma_u_tilde @ v)).sum(dim=0)  # [n]
    var_f = var_prior + var_posterior  # [n]

    # Gate: sigmoid(omega^T [1, W_j x_i])
    gate_input = omega_j[0] + (omega_j[1:] * Z_proj).sum(dim=1)  # [n]
    log_sig = F.logsigmoid(gate_input)
    log_one_minus_sig = F.logsigmoid(-gate_input)

    # U (outlier constant)
    U_val = torch.sigmoid(U_logit_j)

    # Inlier log-likelihood: N(y|E_f, sigma_noise^2 + var_f)
    quad = (y_j - E_f).pow(2) / (2 * sigma_noise_j ** 2) + var_f / (2 * sigma_noise_j ** 2)
    T1 = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise_j ** 2 + 1e-12) + log_sig - quad

    # Outlier log-likelihood
    T2 = torch.log(U_val + 1e-12) + log_one_minus_sig

    data_ll = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum()

    # KL(q(u_j) || p(u_j))
    logdet_Kuu = 2.0 * torch.log(torch.diagonal(L_kuu)).sum()
    trace_term = (torch.diagonal(Kuu_inv) * torch.diagonal(Sigma_u)).sum()  # simplified
    # More precise: tr(Kuu_inv @ Sigma_u)
    trace_term = torch.trace(Kuu_inv @ Sigma_u)
    quad_kl = mu_u_j @ (Kuu_inv @ mu_u_j)
    L_sigma_diag = torch.diagonal(L_sigma)
    logdet_Sigma = 2.0 * torch.log(L_sigma_diag + 1e-12).sum()
    kl_u = 0.5 * (trace_term + quad_kl - m1 + logdet_Kuu - logdet_Sigma)

    return data_ll - kl_u


def _optimize_local_params(
    X_j: torch.Tensor,
    y_j: torch.Tensor,
    W_j: torch.Tensor,      # [Q, D], detached
    x_test_j: torch.Tensor, # [D]
    Q: int,
    m1: int,
    inner_steps: int = 10,
    inner_lr: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Optimize local variational params for region j given a fixed W_j sample.

    Returns dict of optimized local params (all detached).
    """
    dev = X_j.device

    # Initialize local params
    C_j = torch.randn(m1, Q, device=dev, requires_grad=True)
    mu_u_j = torch.randn(m1, device=dev, requires_grad=True)
    raw_L_j = torch.eye(m1, device=dev, requires_grad=True)
    sigma_noise_j = torch.tensor(0.5, device=dev, requires_grad=True)
    sigma_k_j = torch.tensor(0.5, device=dev, requires_grad=True)
    omega_j = torch.randn(Q + 1, device=dev, requires_grad=True)
    U_logit_j = torch.zeros(1, device=dev, requires_grad=True)

    local_params = [C_j, mu_u_j, raw_L_j, sigma_noise_j, sigma_k_j, omega_j, U_logit_j]
    opt = torch.optim.Adam(local_params, lr=inner_lr)

    for _ in range(inner_steps):
        opt.zero_grad()
        elbo = _local_elbo_one_region(
            X_j, y_j, W_j, x_test_j, C_j, mu_u_j, raw_L_j,
            sigma_noise_j, sigma_k_j, omega_j, U_logit_j,
        )
        (-elbo).backward()
        opt.step()
        with torch.no_grad():
            sigma_noise_j.clamp_(min=0.1)
            sigma_k_j.clamp_(min=0.1)

    return {
        "C": C_j.detach(),
        "mu_u": mu_u_j.detach(),
        "raw_L": raw_L_j.detach(),
        "sigma_noise": sigma_noise_j.detach(),
        "sigma_k": sigma_k_j.detach(),
        "omega": omega_j.detach(),
        "U_logit": U_logit_j.detach(),
    }


# ---------------------------------------------------------------------------
# MC-ELBO: the full objective
# ---------------------------------------------------------------------------

def compute_mc_elbo(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    S: int = 3,
    inner_steps: int = 10,
    inner_lr: float = 0.05,
    m1: int = 2,
    verbose: bool = False,
) -> tuple[torch.Tensor, list[list[dict[str, torch.Tensor]]]]:
    """Compute the MC-ELBO.

    Returns:
        elbo: scalar tensor (differentiable w.r.t. V_params, hyperparams)
        all_local_params: [S][T] list of per-sample-per-region local params
    """
    T = len(regions)
    Z = hyperparams["Z"]
    X_test = hyperparams["X_test"]
    mu_V = V_params["mu_V"]
    sigma_V = V_params["sigma_V"]
    Q = mu_V.shape[1]

    # 1) Compute q(W) from q(R)
    mu_W, sigma_W_diag = qW_from_qR(
        X_test, Z, mu_V, sigma_V,
        hyperparams["lengthscales"], hyperparams["var_w"],
    )  # [T, Q, D], [T, Q, D]

    # 2) Sample W: [S, T, Q, D]
    W_samples = sample_W(mu_W, sigma_W_diag, S)

    # 3) For each sample s and region j, optimize local params (stop-grad on local)
    #    then evaluate local ELBO with the W_j^{(s)} that *is* in the graph
    all_local_params: list[list[dict[str, torch.Tensor]]] = []
    mc_sum = torch.tensor(0.0, device=Z.device)

    for s in range(S):
        sample_locals = []
        for j in range(T):
            X_j = regions[j]["X"]
            y_j = regions[j]["y"]
            W_j_s = W_samples[s, j]  # [Q, D] — in the graph through reparameterization
            x_test_j = X_test[j]

            # Inner optimization with detached W
            local_opt = _optimize_local_params(
                X_j, y_j, W_j_s.detach(), x_test_j,
                Q=Q, m1=m1, inner_steps=inner_steps, inner_lr=inner_lr,
            )
            sample_locals.append(local_opt)

            # Evaluate local ELBO with the *differentiable* W_j_s but stop-grad locals
            elbo_j = _local_elbo_one_region(
                X_j, y_j, W_j_s, x_test_j,
                local_opt["C"], local_opt["mu_u"], local_opt["raw_L"],
                local_opt["sigma_noise"], local_opt["sigma_k"],
                local_opt["omega"], local_opt["U_logit"],
            )
            mc_sum = mc_sum + elbo_j

        all_local_params.append(sample_locals)

    # 4) KL(q(R) || p(R))
    kl_R = kl_qR(Z, mu_V, sigma_V, hyperparams["lengthscales"], hyperparams["var_w"])

    # 5) MC-ELBO = (1/S) sum_s sum_j L_j(W_j^{(s)}) - KL(q(R)||p(R))
    elbo = mc_sum / S - kl_R

    return elbo / T, all_local_params


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mc_vi(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    *,
    S: int = 3,
    inner_steps: int = 10,
    inner_lr: float = 0.05,
    m1: int = 2,
    lr: float = 1e-2,
    num_steps: int = 200,
    log_interval: int = 20,
    lr_factor: float = 0.5,
    lr_patience: int = 20,
    early_stop_patience: int = 40,
    min_lr: float = 1e-4,
    elbo_tol: float = 1e-3,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[list[dict[str, torch.Tensor]]]]:
    """Train the MC-ELBO.

    Returns:
        V_params, hyperparams (optimized), and last set of per-sample local params.
    """
    # Global params to optimize
    params = [V_params["mu_V"], V_params["sigma_V"]]
    params += [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor,
        patience=lr_patience, min_lr=min_lr,
    )

    best_elbo = -float("inf")
    steps_no_improve = 0
    best_V = best_hyp = None
    last_local_params = None

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        elbo, local_params = compute_mc_elbo(
            regions, V_params, hyperparams,
            S=S, inner_steps=inner_steps, inner_lr=inner_lr, m1=m1,
        )
        (-elbo).backward()
        optimizer.step()

        with torch.no_grad():
            V_params["sigma_V"].clamp_(min=0.1)
            hyperparams["var_w"].clamp_(min=0.1)
            hyperparams["lengthscales"].clamp_(min=1e-6)

        val = elbo.item()
        scheduler.step(val)
        last_local_params = local_params

        if step % log_interval == 0 or step == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[MC-VI Step {step}/{num_steps}] ELBO={val:.4f}, LR={lr_now:.2e}")

            if val > best_elbo + elbo_tol:
                best_elbo = val
                steps_no_improve = 0
                best_V = {
                    "mu_V": V_params["mu_V"].detach().clone().requires_grad_(True),
                    "sigma_V": V_params["sigma_V"].detach().clone().requires_grad_(True),
                }
                best_hyp = {
                    "Z": hyperparams["Z"].detach().clone().requires_grad_(True),
                    "lengthscales": hyperparams["lengthscales"].detach().clone().requires_grad_(True),
                    "var_w": hyperparams["var_w"].detach().clone().requires_grad_(True),
                    "X_test": hyperparams["X_test"],
                }
            else:
                steps_no_improve += log_interval

            if steps_no_improve >= early_stop_patience:
                print(f"Early stopping at step {step}, best ELBO={best_elbo:.4f}")
                break

    if best_V is None:
        return V_params, hyperparams, last_local_params or []
    return best_V, best_hyp, last_local_params or []


# ---------------------------------------------------------------------------
# Prediction: sample W + JGP CEM (consistent with MC-ELBO training)
# ---------------------------------------------------------------------------

def predict_mc_vi(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    M: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict by sampling W from q(W) and running JGP-CEM per sample.

    This is the same procedure as predict_vi in variational.py.
    """
    T = len(regions)
    X_test = hyperparams["X_test"]
    mu_W, sigma_W_diag = qW_from_qR(
        X_test, hyperparams["Z"],
        V_params["mu_V"], V_params["sigma_V"],
        hyperparams["lengthscales"], hyperparams["var_w"],
    )
    W_all = sample_W(mu_W, sigma_W_diag, M)  # [M, T, Q, D]

    mu_s = torch.zeros((M, T), device=device)
    var_s = torch.zeros((M, T), device=device)

    for m in range(M):
        Wm = W_all[m]  # [T, Q, D]
        for j, r in enumerate(regions):
            Xj = r["X"]  # [n, D]
            yj = r["y"]  # [n]
            Wj = Wm[j]   # [Q, D]
            Xn = Xj @ Wj.T
            yn = yj.view(-1, 1)
            xt = (X_test[j].view(1, -1) @ Wj.T)
            try:
                mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
                    Xn, yn, xt, mode="CEM", flag=False, device=device,
                )
                mu_s[m, j] = mu_t.view(-1)
                var_s[m, j] = sig2_t.view(-1)
            except Exception:
                mu_s[m, j] = 0.0
                var_s[m, j] = 1.0

    mu_pred = mu_s.mean(dim=0)
    within_var = var_s.mean(dim=0)
    between_var = ((mu_s - mu_pred) ** 2).mean(dim=0)
    var_pred = within_var + between_var

    mu_pred = torch.nan_to_num(mu_pred, nan=0.0)
    var_pred = torch.nan_to_num(var_pred, nan=1e-6)

    return mu_pred, var_pred


# ---------------------------------------------------------------------------
# Direct VI prediction using per-sample optimized local params
# ---------------------------------------------------------------------------

def predict_mc_vi_direct(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    M: int = 5,
    inner_steps: int = 15,
    inner_lr: float = 0.05,
    m1: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct prediction: sample W, optimize local params, compute predictive moments.

    For each sample m and region j:
    1. Sample W_j from q(W_j)
    2. Optimize local params (gate, inducing) for this W_j
    3. Compute predictive mean/var at the test point

    Moment-match across samples.
    """
    T = len(regions)
    X_test = hyperparams["X_test"]
    Q = V_params["mu_V"].shape[1]
    mu_W, sigma_W_diag = qW_from_qR(
        X_test, hyperparams["Z"],
        V_params["mu_V"], V_params["sigma_V"],
        hyperparams["lengthscales"], hyperparams["var_w"],
    )
    W_all = sample_W(mu_W, sigma_W_diag, M)  # [M, T, Q, D]

    mu_s = torch.zeros((M, T), device=device)
    var_s = torch.zeros((M, T), device=device)

    for m in range(M):
        Wm = W_all[m]
        for j, r in enumerate(regions):
            X_j = r["X"]
            y_j = r["y"]
            W_j = Wm[j].detach()
            x_test_j = X_test[j]

            # Optimize local params
            lp = _optimize_local_params(
                X_j, y_j, W_j, x_test_j,
                Q=Q, m1=m1, inner_steps=inner_steps, inner_lr=inner_lr,
            )

            # Compute predictive moments at test point
            Z_proj_test = (x_test_j.unsqueeze(0) @ W_j.T)  # [1, Q]
            C_j = lp["C"]
            mu_u_j = lp["mu_u"]

            d2_uu = (C_j.unsqueeze(1) - C_j.unsqueeze(0)).pow(2).sum(-1)
            Kuu = lp["sigma_k"] ** 2 * torch.exp(-0.5 * d2_uu)
            L_kuu = _safe_cholesky(Kuu)
            Kuu_inv = torch.cholesky_inverse(L_kuu)

            d2_test = (Z_proj_test - C_j.unsqueeze(0)).pow(2).sum(-1)  # [1, m1]
            Kfu_test = lp["sigma_k"] ** 2 * torch.exp(-0.5 * d2_test)  # [1, m1]

            alpha = Kuu_inv @ mu_u_j
            mu_f = (Kfu_test @ alpha).squeeze()

            # Build PSD Sigma_u
            L_sigma = torch.tril(lp["raw_L"])
            diag = torch.diagonal(L_sigma)
            pos_diag = F.softplus(diag) + 1e-6
            L_sigma = L_sigma - torch.diag(diag) + torch.diag(pos_diag)
            Sigma_u = L_sigma @ L_sigma.T

            v = Kuu_inv @ Kfu_test.T  # [m1, 1]
            var_prior = lp["sigma_k"] ** 2 - (Kfu_test @ Kuu_inv @ Kfu_test.T).squeeze()
            Sigma_u_tilde = Sigma_u + mu_u_j.unsqueeze(1) * mu_u_j.unsqueeze(0)
            var_post = (v.T @ Sigma_u_tilde @ v).squeeze()
            var_f = var_prior + var_post

            # Gate
            gate_in = lp["omega"][0] + (lp["omega"][1:] * Z_proj_test.squeeze()).sum()
            pi = torch.sigmoid(gate_in)
            U_val = torch.sigmoid(lp["U_logit"])

            mu_y = pi * mu_f + (1 - pi) * U_val
            E_y2 = pi * (mu_f ** 2 + var_f + lp["sigma_noise"] ** 2) + (1 - pi) * U_val ** 2
            var_y = (E_y2 - mu_y ** 2).clamp_min(1e-12)

            mu_s[m, j] = mu_y.detach()
            var_s[m, j] = var_y.detach()

    mu_pred = mu_s.mean(dim=0)
    within_var = var_s.mean(dim=0)
    between_var = ((mu_s - mu_pred) ** 2).mean(dim=0)
    var_pred = within_var + between_var

    mu_pred = torch.nan_to_num(mu_pred, nan=0.0)
    var_pred = torch.nan_to_num(var_pred, nan=1e-6)

    return mu_pred, var_pred


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Using device: {device}")
    T, n, m1_, m2, Q, D = 3, 8, 2, 5, 2, 4
    regions = []
    for _ in range(T):
        regions.append({
            "X": torch.randn(n, D, device=device),
            "y": torch.randn(n, device=device),
            "C": torch.randn(m1_, Q, device=device),
        })
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    hyperparams = {
        "Z": torch.randn(m2, D, device=device, requires_grad=True),
        "X_test": torch.randn(T, D, device=device),
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    print("Testing compute_mc_elbo...")
    elbo, lp = compute_mc_elbo(regions, V_params, hyperparams, S=2, inner_steps=5, m1=m1_)
    print(f"MC-ELBO = {elbo.item():.4f}")

    print("\nTesting train_mc_vi (5 steps)...")
    V_params, hyperparams, _ = train_mc_vi(
        regions, V_params, hyperparams,
        S=2, inner_steps=5, m1=m1_, num_steps=5, log_interval=2,
    )
    print("Training done.")
