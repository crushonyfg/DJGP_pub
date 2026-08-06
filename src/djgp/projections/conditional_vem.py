"""Conditional-W exact local JGP-VEM utilities.

This module implements a practical stochastic generalized-VEM objective for
learning local projections before a JumpGP-CEM prediction refit.

The key difference from the older analytic DJGP ELBO is that sampled
``W_j`` enters the full local covariance ``K(W_j X_j, W_j X_j)`` directly.
The local variational responsibilities ``gamma`` are updated from the
conditional GP posterior ``q(f_j | W_j, gamma_j)`` and the training objective
averages the soft collapsed local JGP bound over samples from ``q(W_j)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from djgp.projections.cem_em_projection import laplacian_smoothness
from djgp.projections.lowrank_factorization import anchor_rbf_kernel, per_anchor_W_diagnostics


def _row_normalize(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


class GaussianCoeffProjection(nn.Module):
    """Diagonal Gaussian posterior over per-anchor low-rank coefficients.

    ``W_j = row_norm(W0 + coeff_scale * C_j V^T)``, where
    ``q(C_j) = N(C_mu_j, diag(C_std_j^2))``.  With ``posterior=False`` this
    reduces to a deterministic MAP coefficient model.
    """

    def __init__(
        self,
        *,
        T: int,
        Q: int,
        D: int,
        V: torch.Tensor,
        W0_init: torch.Tensor,
        posterior: bool,
        init_log_std: float = -3.0,
        coeff_scale: float = 1.0,
        row_norm_eps: float = 1e-8,
    ):
        super().__init__()
        if V.shape[0] != D:
            raise ValueError(f"V must have {D} rows; got {tuple(V.shape)}.")
        if W0_init.shape != (Q, D):
            raise ValueError(f"W0_init must be [{Q}, {D}], got {tuple(W0_init.shape)}.")
        self.T = int(T)
        self.Q = int(Q)
        self.D = int(D)
        self.Rv = int(V.shape[1])
        self.posterior = bool(posterior)
        self.coeff_scale = float(coeff_scale)
        self.row_norm_eps = float(row_norm_eps)
        self.register_buffer("V", V.contiguous())
        self.W0 = nn.Parameter(_row_normalize(W0_init.detach().clone(), eps=row_norm_eps))
        self.C_mu = nn.Parameter(torch.zeros(T, Q, self.Rv, device=V.device, dtype=V.dtype))
        log_std = torch.full((T, Q, self.Rv), float(init_log_std), device=V.device, dtype=V.dtype)
        if self.posterior:
            self.C_log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("C_log_std", log_std)

    def sample_C(self, n_samples: int, *, mean_only: bool = False) -> torch.Tensor:
        if mean_only or not self.posterior:
            return self.C_mu.unsqueeze(0).expand(int(n_samples), -1, -1, -1)
        eps = torch.randn(
            int(n_samples),
            self.T,
            self.Q,
            self.Rv,
            device=self.C_mu.device,
            dtype=self.C_mu.dtype,
        )
        std = torch.exp(self.C_log_std).clamp(1e-5, 10.0)
        return self.C_mu.unsqueeze(0) + std.unsqueeze(0) * eps

    def W_from_C(self, C: torch.Tensor) -> torch.Tensor:
        W = self.W0.unsqueeze(0).unsqueeze(0) + self.coeff_scale * torch.einsum("stqr,dr->stqd", C, self.V)
        return _row_normalize(W, eps=self.row_norm_eps)

    def sample_W(self, n_samples: int, *, mean_only: bool = False) -> torch.Tensor:
        return self.W_from_C(self.sample_C(n_samples, mean_only=mean_only))

    def mean_W(self) -> torch.Tensor:
        return self.sample_W(1, mean_only=True).squeeze(0)

    def kl_coefficients(self, *, prior_std: float = 1.0) -> torch.Tensor:
        if not self.posterior:
            return torch.zeros((), device=self.C_mu.device, dtype=self.C_mu.dtype)
        prior_var = float(prior_std) ** 2
        var = torch.exp(2.0 * self.C_log_std).clamp(1e-10, 1e6)
        mu2 = self.C_mu.pow(2)
        kl = 0.5 * ((var + mu2) / prior_var - 1.0 - torch.log(var / prior_var))
        return kl.sum() / max(1, self.T)


def rbf_kernel_from_W(
    X: torch.Tensor,
    W: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_amp: torch.Tensor,
    *,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(K, Z)`` for ``W: [S,T,Q,D]`` and ``X: [T,n,D]``."""
    Z = torch.einsum("tnd,stqd->stnq", X, W)
    l2 = torch.exp(2.0 * log_lengthscale).clamp_min(1e-12)
    amp2 = torch.exp(2.0 * log_amp).clamp_min(1e-12)
    d2 = (Z.unsqueeze(3) - Z.unsqueeze(2)).pow(2).sum(-1)
    K = amp2 * torch.exp(-0.5 * d2 / l2)
    n = X.shape[1]
    eye = torch.eye(n, device=X.device, dtype=X.dtype).view(1, 1, n, n)
    K = K + float(jitter) * eye
    return K, Z


def conditional_gp_moments(
    K: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    noise_var: torch.Tensor,
    *,
    gamma_floor: float = 1e-3,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional ``q(f | W, gamma)`` moments for weighted Gaussian likelihood."""
    S, T, n, _ = K.shape
    dtype = K.dtype
    device = K.device
    gamma_c = gamma.to(device=device, dtype=dtype).clamp(float(gamma_floor), 1.0)
    sig2 = noise_var.to(device=device, dtype=dtype).reshape(()).clamp_min(1e-8)
    diag_noise = sig2 / gamma_c
    eye = torch.eye(n, device=device, dtype=dtype).view(1, 1, n, n)
    Ky = K + torch.diag_embed(diag_noise).unsqueeze(0) + float(jitter) * eye
    L = torch.linalg.cholesky(0.5 * (Ky + Ky.transpose(-1, -2)))
    y_b = y.to(device=device, dtype=dtype).view(1, T, n, 1).expand(S, -1, -1, -1)
    alpha = torch.cholesky_solve(y_b, L)
    mean = (K @ alpha).squeeze(-1)
    solve_K = torch.cholesky_solve(K, L)
    cov = K - K @ solve_K
    diag_cov = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0.0)
    return mean, diag_cov


def update_gamma_conditional_w(
    *,
    Z: torch.Tensor,
    K: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    noise_var: torch.Tensor,
    outlier_logprob: torch.Tensor,
    temperature: float = 1.0,
    damping: float = 1.0,
    gamma_floor: float = 1e-3,
) -> torch.Tensor:
    """One conditional-W VEM responsibility update, averaged over W samples."""
    mean_f, diag_cov = conditional_gp_moments(K, y, gamma, noise_var, gamma_floor=gamma_floor)
    sig2 = noise_var.reshape(()).clamp_min(1e-8)
    y_b = y.view(1, y.shape[0], y.shape[1])
    ell_in = -0.5 * (math.log(2.0 * math.pi) + torch.log(sig2) + ((y_b - mean_f).pow(2) + diag_cov) / sig2)
    gate_intercept = gate[:, :1]
    gate_slope = gate[:, 1:]
    gate_logit = gate_intercept.view(1, -1, 1) + torch.einsum("stnq,tq->stn", Z, gate_slope)
    logit = (gate_logit + ell_in - outlier_logprob.reshape(())).mean(dim=0)
    gamma_new = torch.sigmoid(logit / max(float(temperature), 1e-6))
    gamma_new = gamma_new.clamp(float(gamma_floor), 1.0 - float(gamma_floor))
    return ((1.0 - float(damping)) * gamma + float(damping) * gamma_new).clamp(
        float(gamma_floor),
        1.0 - float(gamma_floor),
    )


def collapsed_soft_jgp_bound_per_anchor(
    *,
    Z: torch.Tensor,
    K: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    noise_var: torch.Tensor,
    outlier_logprob: torch.Tensor,
    gamma_floor: float = 1e-3,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """Soft collapsed local JGP-VEM bound per W-sample and anchor."""
    S, T, n, _ = K.shape
    dtype = K.dtype
    device = K.device
    gamma_c = gamma.to(device=device, dtype=dtype).clamp(float(gamma_floor), 1.0 - float(gamma_floor))
    sig2 = noise_var.to(device=device, dtype=dtype).reshape(()).clamp_min(1e-8)
    eye = torch.eye(n, device=device, dtype=dtype).view(1, 1, n, n)
    Ky = K + torch.diag_embed(sig2 / gamma_c).unsqueeze(0) + float(jitter) * eye
    L = torch.linalg.cholesky(0.5 * (Ky + Ky.transpose(-1, -2)))
    y_b = y.to(device=device, dtype=dtype).view(1, T, n, 1).expand(S, -1, -1, -1)
    alpha = torch.cholesky_solve(y_b, L).squeeze(-1)
    quad = (y_b.squeeze(-1) * alpha).sum(dim=-1)
    logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)
    gp_log = -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))

    gate_intercept = gate[:, :1]
    gate_slope = gate[:, 1:]
    gate_logit = gate_intercept.view(1, T, 1) + torch.einsum("stnq,tq->stn", Z, gate_slope)
    gate_log = gamma_c.unsqueeze(0) * F.logsigmoid(gate_logit)
    gate_log = gate_log + (1.0 - gamma_c).unsqueeze(0) * F.logsigmoid(-gate_logit)
    gate_log = gate_log.sum(dim=-1)
    entropy = -(gamma_c * gamma_c.log() + (1.0 - gamma_c) * (1.0 - gamma_c).log()).sum(dim=-1)
    outlier = ((1.0 - gamma_c) * outlier_logprob.reshape(())).sum(dim=-1)
    return gp_log + gate_log + entropy.unsqueeze(0) + outlier.unsqueeze(0)


def local_response_variance_weights(y: torch.Tensor) -> torch.Tensor:
    var = y.var(dim=1, unbiased=False)
    return (var / var.mean().clamp_min(1e-12)).clamp(0.25, 4.0).detach()


@dataclass
class VemWConfig:
    posterior: bool = False
    n_steps: int = 100
    vem_steps: int = 1
    mc_train: int = 1
    mc_gamma: int = 1
    lr: float = 0.01
    coeff_scale: float = 1.0
    init_log_std: float = -3.0
    prior_std: float = 1.0
    kl_weight: float = 0.1
    kl_warmup_steps: int = 50
    lambda_smooth: float = 0.01
    lambda_coeff: float = 1e-4
    lambda_gate_l2: float = 1e-4
    gamma_init: float = 0.7
    gamma_floor: float = 1e-3
    gamma_damping: float = 0.7
    temperature: float = 1.5
    temperature_final: float = 1.0
    boundary_weighted: bool = False
    train_kernel: bool = False
    train_noise: bool = False
    init_lengthscale: float = 1.0
    init_amp: float = 1.0
    init_noise_var: float = 0.1
    outlier_sigma_mult: float = 2.5
    jitter: float = 1e-5
    log_interval: int = 25
    verbose: bool = False


@dataclass
class VemWResult:
    model: GaussianCoeffProjection
    gamma: torch.Tensor
    gate: torch.Tensor
    W_mean: torch.Tensor
    W_samples: torch.Tensor
    history: list[dict[str, float]]
    train_time_sec: float
    diagnostics: dict[str, float]


def train_conditional_vem_w(
    *,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    X_anchors: torch.Tensor,
    V: torch.Tensor,
    W0_init: torch.Tensor,
    cfg: VemWConfig,
) -> VemWResult:
    """Train a local projection posterior with conditional-W exact JGP-VEM."""
    device = X_neighbors.device
    dtype = X_neighbors.dtype
    T, _, D = X_neighbors.shape
    Q = W0_init.shape[0]
    model = GaussianCoeffProjection(
        T=T,
        Q=Q,
        D=D,
        V=V,
        W0_init=W0_init,
        posterior=cfg.posterior,
        init_log_std=cfg.init_log_std,
        coeff_scale=cfg.coeff_scale,
    ).to(device)
    gate = nn.Parameter(torch.zeros(T, Q + 1, device=device, dtype=dtype))
    log_lengthscale = nn.Parameter(torch.tensor(math.log(cfg.init_lengthscale), device=device, dtype=dtype))
    log_amp = nn.Parameter(torch.tensor(math.log(cfg.init_amp), device=device, dtype=dtype))
    log_noise_var = nn.Parameter(torch.tensor(math.log(cfg.init_noise_var), device=device, dtype=dtype))
    if not cfg.train_kernel:
        log_lengthscale.requires_grad_(False)
        log_amp.requires_grad_(False)
    if not cfg.train_noise:
        log_noise_var.requires_grad_(False)

    params: list[torch.Tensor] = [p for p in model.parameters() if p.requires_grad] + [gate]
    for p in (log_lengthscale, log_amp, log_noise_var):
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.Adam(params, lr=cfg.lr)

    gamma = torch.full(
        (T, X_neighbors.shape[1]),
        float(cfg.gamma_init),
        device=device,
        dtype=dtype,
    ).clamp(cfg.gamma_floor, 1.0 - cfg.gamma_floor)
    weights = local_response_variance_weights(y_neighbors) if cfg.boundary_weighted else torch.ones(T, device=device, dtype=dtype)
    K_anchor = None
    if cfg.lambda_smooth > 0:
        K_anchor = anchor_rbf_kernel(X_anchors.detach()).detach()
    outlier_logprob = -0.5 * (
        math.log(2.0 * math.pi)
        + log_noise_var.detach()
        + float(cfg.outlier_sigma_mult) ** 2
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    t0 = time.perf_counter()
    for step in range(1, int(cfg.n_steps) + 1):
        frac = min(1.0, step / max(1, int(cfg.n_steps)))
        temp = float(cfg.temperature) + frac * (float(cfg.temperature_final) - float(cfg.temperature))
        with torch.no_grad():
            for _ in range(int(cfg.vem_steps)):
                W_g = model.sample_W(max(1, int(cfg.mc_gamma)), mean_only=not cfg.posterior)
                K_g, Z_g = rbf_kernel_from_W(
                    X_neighbors,
                    W_g,
                    log_lengthscale.detach(),
                    log_amp.detach(),
                    jitter=cfg.jitter,
                )
                gamma = update_gamma_conditional_w(
                    Z=Z_g,
                    K=K_g,
                    y=y_neighbors,
                    gamma=gamma,
                    gate=gate.detach(),
                    noise_var=torch.exp(log_noise_var.detach()),
                    outlier_logprob=outlier_logprob.detach(),
                    temperature=temp,
                    damping=cfg.gamma_damping,
                    gamma_floor=cfg.gamma_floor,
                )

        opt.zero_grad()
        W_s = model.sample_W(max(1, int(cfg.mc_train)), mean_only=not cfg.posterior)
        K, Z = rbf_kernel_from_W(X_neighbors, W_s, log_lengthscale, log_amp, jitter=cfg.jitter)
        outlier_logprob = -0.5 * (
            math.log(2.0 * math.pi)
            + log_noise_var
            + float(cfg.outlier_sigma_mult) ** 2
        )
        bound = collapsed_soft_jgp_bound_per_anchor(
            Z=Z,
            K=K,
            y=y_neighbors,
            gamma=gamma.detach(),
            gate=gate,
            noise_var=torch.exp(log_noise_var),
            outlier_logprob=outlier_logprob,
            gamma_floor=cfg.gamma_floor,
            jitter=cfg.jitter,
        )
        weighted_bound = (bound.mean(dim=0) * weights).mean()
        loss = -weighted_bound
        kl = model.kl_coefficients(prior_std=cfg.prior_std)
        beta = float(cfg.kl_weight) * min(1.0, step / max(1, int(cfg.kl_warmup_steps)))
        loss = loss + beta * kl
        smooth = torch.zeros((), device=device, dtype=dtype)
        if K_anchor is not None:
            smooth = laplacian_smoothness(model.C_mu, K_anchor, normalize_by_graph_weight=True)
            loss = loss + float(cfg.lambda_smooth) * smooth
        coeff_l2 = model.C_mu.pow(2).mean()
        loss = loss + float(cfg.lambda_coeff) * coeff_l2
        gate_l2 = gate[:, 1:].pow(2).mean()
        loss = loss + float(cfg.lambda_gate_l2) * gate_l2
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().item())
        if loss_val < best_loss and torch.isfinite(loss.detach()):
            best_loss = loss_val
            best_state = {
                "W0": model.W0.detach().clone(),
                "C_mu": model.C_mu.detach().clone(),
                "C_log_std": model.C_log_std.detach().clone(),
                "gate": gate.detach().clone(),
                "log_l": log_lengthscale.detach().clone(),
                "log_amp": log_amp.detach().clone(),
                "log_noise": log_noise_var.detach().clone(),
                "gamma": gamma.detach().clone(),
            }
        if step == 1 or step % int(cfg.log_interval) == 0 or step == int(cfg.n_steps):
            entropy = -(gamma * gamma.log() + (1.0 - gamma) * (1.0 - gamma).log()).mean()
            rec = {
                "step": float(step),
                "loss": loss_val,
                "bound": float(weighted_bound.detach().item()),
                "kl": float(kl.detach().item()),
                "beta": float(beta),
                "smooth": float(smooth.detach().item()),
                "coeff_l2": float(coeff_l2.detach().item()),
                "gate_l2": float(gate_l2.detach().item()),
                "gamma_mean": float(gamma.mean().detach().item()),
                "gamma_entropy": float(entropy.detach().item()),
                "log_lengthscale": float(log_lengthscale.detach().item()),
                "log_amp": float(log_amp.detach().item()),
                "log_noise_var": float(log_noise_var.detach().item()),
            }
            history.append(rec)
            if cfg.verbose:
                print(
                    f"[vem-w step {step:04d}] loss={rec['loss']:.4f} "
                    f"bound={rec['bound']:.4f} kl={rec['kl']:.4f} "
                    f"gamma={rec['gamma_mean']:.3f} H={rec['gamma_entropy']:.3f}",
                    flush=True,
                )

    if best_state is not None:
        with torch.no_grad():
            model.W0.copy_(best_state["W0"])
            model.C_mu.copy_(best_state["C_mu"])
            if model.posterior:
                model.C_log_std.copy_(best_state["C_log_std"])
            gate.copy_(best_state["gate"])
            log_lengthscale.copy_(best_state["log_l"])
            log_amp.copy_(best_state["log_amp"])
            log_noise_var.copy_(best_state["log_noise"])
            gamma = best_state["gamma"]

    train_sec = time.perf_counter() - t0
    with torch.no_grad():
        W_mean = model.mean_W().detach()
        n_eval_samples = max(1, int(cfg.mc_train))
        W_samples = model.sample_W(n_eval_samples, mean_only=not cfg.posterior).detach()
        diag = per_anchor_W_diagnostics(W_mean, W_ref=_row_normalize(W0_init.detach()))
        gamma_entropy = -(gamma * gamma.log() + (1.0 - gamma) * (1.0 - gamma).log()).mean()
        diag.update(
            {
                "train_sec": float(train_sec),
                "best_loss": float(best_loss),
                "gamma_mean": float(gamma.mean().item()),
                "gamma_entropy": float(gamma_entropy.item()),
                "kl_coeff": float(model.kl_coefficients(prior_std=cfg.prior_std).item()),
                "C_mu_frob_median": float(torch.linalg.norm(model.C_mu, dim=(-2, -1)).median().item()),
                "C_std_median": float(torch.exp(model.C_log_std).median().item()),
                "log_lengthscale": float(log_lengthscale.detach().item()),
                "log_amp": float(log_amp.detach().item()),
                "log_noise_var": float(log_noise_var.detach().item()),
            }
        )
    return VemWResult(
        model=model,
        gamma=gamma.detach(),
        gate=gate.detach(),
        W_mean=W_mean,
        W_samples=W_samples,
        history=history,
        train_time_sec=float(train_sec),
        diagnostics=diag,
    )


__all__ = [
    "GaussianCoeffProjection",
    "VemWConfig",
    "VemWResult",
    "collapsed_soft_jgp_bound_per_anchor",
    "conditional_gp_moments",
    "rbf_kernel_from_W",
    "train_conditional_vem_w",
    "update_gamma_conditional_w",
]
