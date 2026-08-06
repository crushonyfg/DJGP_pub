"""Sparse-GP global projection field trained by local JGP composite likelihood.

This module is an experiment-oriented repair of the original LMJGP projection
path.  It replaces independent anchor posteriors with a shared GP field over
projection matrices,

    W_{q,d}(x) = k(x, U) K(U,U)^{-1} R_{:,q,d},

and trains the inducing posterior q(R) with a fixed-neighborhood composite
local JGP/VEM surrogate.  Prediction samples global R values, recomputes the
latent coordinates coherently for all points, reselects local neighbors in the
sampled latent space, and refits JumpGP-CEM.
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

from djgp.projections.lowrank_factorization import per_anchor_W_diagnostics
from djgp.variational import kl_qp
from shared.utils1 import jumpgp_ld_wrapper


def _row_normalize(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _median_lengthscale(X: torch.Tensor, max_points: int = 512) -> float:
    with torch.no_grad():
        Xs = X[: min(max_points, X.shape[0])]
        d = torch.cdist(Xs, Xs)
        vals = d[d > 0]
        if vals.numel() == 0:
            return 1.0
        return float(vals.median().clamp_min(1e-3).item())


class SparseGPProjectionField(nn.Module):
    """Diagonal q(R) sparse-GP projection field with DTC conditionals."""

    def __init__(
        self,
        *,
        U: torch.Tensor,
        Q: int,
        D: int,
        init_W: torch.Tensor | None = None,
        init_R_mu: torch.Tensor | None = None,
        init_log_std: float = -3.0,
        lengthscale: float = 1.0,
        var_w: float = 1.0,
        train_lengthscale: bool = False,
        train_var_w: bool = False,
    ):
        super().__init__()
        if U.dim() != 2 or U.shape[1] != D:
            raise ValueError(f"U must be [M,{D}], got {tuple(U.shape)}")
        self.Q = int(Q)
        self.D = int(D)
        self.M = int(U.shape[0])
        self.register_buffer("U", U.detach().clone())
        if init_R_mu is not None:
            if init_R_mu.shape != (self.M, Q, D):
                raise ValueError(f"init_R_mu must be [{self.M},{Q},{D}], got {tuple(init_R_mu.shape)}")
            R_mu = init_R_mu.to(device=U.device, dtype=U.dtype).clone()
        elif init_W is None:
            R_mu = 0.01 * torch.randn(self.M, Q, D, device=U.device, dtype=U.dtype)
        else:
            if init_W.shape != (Q, D):
                raise ValueError(f"init_W must be [{Q},{D}], got {tuple(init_W.shape)}")
            R_mu = init_W.to(device=U.device, dtype=U.dtype).unsqueeze(0).expand(self.M, -1, -1).clone()
            R_mu = R_mu + 1e-3 * torch.randn_like(R_mu)
        self.R_mu = nn.Parameter(R_mu)
        self.R_log_std = nn.Parameter(torch.full_like(R_mu, float(init_log_std)))
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(lengthscale), device=U.device, dtype=U.dtype))
        self.log_var_w = nn.Parameter(torch.tensor(math.log(var_w), device=U.device, dtype=U.dtype))
        self.log_lengthscale.requires_grad_(bool(train_lengthscale))
        self.log_var_w.requires_grad_(bool(train_var_w))

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale).clamp_min(1e-4)

    @property
    def var_w(self) -> torch.Tensor:
        return torch.exp(self.log_var_w).clamp_min(1e-6)

    def _cross_weights(self, X: torch.Tensor) -> torch.Tensor:
        d2_uu = (self.U.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(-1)
        d2_xu = (X.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(-1)
        Kuu = self.var_w * torch.exp(-0.5 * d2_uu / self.lengthscale.pow(2))
        eye = torch.eye(self.M, device=X.device, dtype=X.dtype)
        L = torch.linalg.cholesky(0.5 * (Kuu + Kuu.T) + 1e-5 * eye)
        Kxu = self.var_w * torch.exp(-0.5 * d2_xu / self.lengthscale.pow(2))
        return torch.cholesky_solve(Kxu.T, L).T

    def sample_R(self, n_samples: int, *, mean_only: bool = False) -> torch.Tensor:
        if mean_only:
            return self.R_mu.unsqueeze(0).expand(int(n_samples), -1, -1, -1)
        eps = torch.randn(
            int(n_samples),
            self.M,
            self.Q,
            self.D,
            device=self.R_mu.device,
            dtype=self.R_mu.dtype,
        )
        return self.R_mu.unsqueeze(0) + torch.exp(self.R_log_std).clamp(1e-5, 10.0).unsqueeze(0) * eps

    def W_from_R(self, X: torch.Tensor, R: torch.Tensor, *, row_normalize: bool = True) -> torch.Tensor:
        squeeze = False
        if R.dim() == 3:
            R = R.unsqueeze(0)
            squeeze = True
        A = self._cross_weights(X)
        W = torch.einsum("nm,smqd->snqd", A, R)
        if row_normalize:
            W = _row_normalize(W)
        return W.squeeze(0) if squeeze else W

    def z_anchor_conditioned(
        self,
        X_local: torch.Tensor,
        X_anchor: torch.Tensor,
        R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_anchor = self.W_from_R(X_anchor, R)
        if W_anchor.dim() == 3:
            W_anchor = W_anchor.unsqueeze(0)
        Z_local = torch.einsum("and,saqd->sanq", X_local, W_anchor)
        z_anchor = torch.einsum("ad,saqd->saq", X_anchor, W_anchor)
        return Z_local, z_anchor, W_anchor

    def z_pointwise(
        self,
        X_local: torch.Tensor,
        X_anchor: torch.Tensor,
        R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S = R.shape[0] if R.dim() == 4 else 1
        A, n, D = X_local.shape
        X_flat = X_local.reshape(A * n, D)
        W_flat = self.W_from_R(X_flat, R)
        if W_flat.dim() == 3:
            W_flat = W_flat.unsqueeze(0)
        W_local = W_flat.reshape(S, A, n, self.Q, self.D)
        Z_local = torch.einsum("and,sanqd->sanq", X_local, W_local)
        W_anchor = self.W_from_R(X_anchor, R)
        if W_anchor.dim() == 3:
            W_anchor = W_anchor.unsqueeze(0)
        z_anchor = torch.einsum("ad,saqd->saq", X_anchor, W_anchor)
        return Z_local, z_anchor, W_anchor

    def kl(self) -> torch.Tensor:
        sigma = torch.exp(self.R_log_std).clamp(1e-5, 10.0)
        lengthscales = self.lengthscale.expand(self.Q)
        return kl_qp(self.U, self.R_mu, sigma, lengthscales, self.var_w)


def rbf_kernel_from_z(
    Z: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_amp: torch.Tensor,
    *,
    jitter: float = 1e-5,
) -> torch.Tensor:
    l2 = torch.exp(2.0 * log_lengthscale).clamp_min(1e-12)
    amp2 = torch.exp(2.0 * log_amp).clamp_min(1e-12)
    d2 = (Z.unsqueeze(3) - Z.unsqueeze(2)).pow(2).sum(-1)
    K = amp2 * torch.exp(-0.5 * d2 / l2)
    n = Z.shape[2]
    eye = torch.eye(n, device=Z.device, dtype=Z.dtype).view(1, 1, n, n)
    return K + float(jitter) * eye


def conditional_gp_moments(
    K: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    noise_var: torch.Tensor,
    *,
    gamma_floor: float,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    S, A, n, _ = K.shape
    gamma_c = gamma.clamp(float(gamma_floor), 1.0)
    sig2 = noise_var.reshape(()).clamp_min(1e-8)
    eye = torch.eye(n, device=K.device, dtype=K.dtype).view(1, 1, n, n)
    Ky = K + torch.diag_embed(sig2 / gamma_c).unsqueeze(0) + float(jitter) * eye
    L = torch.linalg.cholesky(0.5 * (Ky + Ky.transpose(-1, -2)))
    y_b = y.view(1, A, n, 1).expand(S, -1, -1, -1)
    alpha = torch.cholesky_solve(y_b, L)
    mean = (K @ alpha).squeeze(-1)
    solve_K = torch.cholesky_solve(K, L)
    cov = K - K @ solve_K
    return mean, torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0.0)


def update_gamma(
    *,
    Z: torch.Tensor,
    z_anchor: torch.Tensor,
    K: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    noise_var: torch.Tensor,
    outlier_logprob: torch.Tensor,
    temperature: float,
    damping: float,
    gamma_floor: float,
) -> torch.Tensor:
    mean_f, diag_cov = conditional_gp_moments(K, y, gamma, noise_var, gamma_floor=gamma_floor)
    sig2 = noise_var.reshape(()).clamp_min(1e-8)
    y_b = y.view(1, y.shape[0], y.shape[1])
    ell_in = -0.5 * (math.log(2.0 * math.pi) + torch.log(sig2) + ((y_b - mean_f).pow(2) + diag_cov) / sig2)
    delta = Z - z_anchor.unsqueeze(2)
    gate_logit = gate[:, :1].view(1, -1, 1) + torch.einsum("sanq,aq->san", delta, gate[:, 1:])
    logit = (gate_logit + ell_in - outlier_logprob.unsqueeze(0)).mean(dim=0)
    gamma_new = torch.sigmoid(logit / max(float(temperature), 1e-6))
    gamma_new = gamma_new.clamp(float(gamma_floor), 1.0 - float(gamma_floor))
    return ((1.0 - float(damping)) * gamma + float(damping) * gamma_new).clamp(
        float(gamma_floor), 1.0 - float(gamma_floor)
    )


def collapsed_bound(
    *,
    Z: torch.Tensor,
    z_anchor: torch.Tensor,
    K: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    noise_var: torch.Tensor,
    outlier_logprob: torch.Tensor,
    gamma_floor: float,
    jitter: float = 1e-5,
) -> torch.Tensor:
    S, A, n, _ = K.shape
    gamma_c = gamma.clamp(float(gamma_floor), 1.0 - float(gamma_floor))
    sig2 = noise_var.reshape(()).clamp_min(1e-8)
    eye = torch.eye(n, device=K.device, dtype=K.dtype).view(1, 1, n, n)
    Ky = K + torch.diag_embed(sig2 / gamma_c).unsqueeze(0) + float(jitter) * eye
    L = torch.linalg.cholesky(0.5 * (Ky + Ky.transpose(-1, -2)))
    y_b = y.view(1, A, n, 1).expand(S, -1, -1, -1)
    alpha = torch.cholesky_solve(y_b, L).squeeze(-1)
    quad = (y_b.squeeze(-1) * alpha).sum(dim=-1)
    logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)
    gp_log = -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))

    delta = Z - z_anchor.unsqueeze(2)
    gate_logit = gate[:, :1].view(1, A, 1) + torch.einsum("sanq,aq->san", delta, gate[:, 1:])
    gate_log = gamma_c.unsqueeze(0) * F.logsigmoid(gate_logit)
    gate_log = gate_log + (1.0 - gamma_c).unsqueeze(0) * F.logsigmoid(-gate_logit)
    entropy = -(gamma_c * gamma_c.log() + (1.0 - gamma_c) * (1.0 - gamma_c).log())
    outlier = (1.0 - gamma_c).unsqueeze(0) * outlier_logprob.unsqueeze(0)
    return gp_log + gate_log.sum(-1) + entropy.sum(-1).unsqueeze(0) + outlier.sum(-1)


@dataclass
class GlobalWGPCemConfig:
    mode: str = "anchor"  # "anchor" or "pointwise"
    steps: int = 80
    c_refresh: int = 5
    mc_train: int = 1
    lr: float = 0.01
    beta_kl: float = 0.05
    kl_warmup_steps: int = 40
    composite_temperature: float = 0.2
    gamma_init: float = 0.7
    gamma_floor: float = 0.02
    gamma_temperature: float = 3.0
    gamma_damping: float = 0.5
    coverage_weight: float = 0.0
    coverage_min: float = 0.1
    background: str = "uniform"  # "uniform" or "normal"
    background_scale: float = 3.0
    init_log_std: float = -3.0
    init_lengthscale: float = 1.0
    init_w_lengthscale: float | None = None
    init_noise_var: float = 0.1
    init_amp: float = 1.0
    train_kernel: bool = False
    train_noise: bool = False
    train_R_log_std: bool = True
    sample_train_R: bool = True
    train_w_lengthscale: bool = False
    train_w_var: bool = False
    outlier_sigma_mult: float = 2.5
    gate_l2: float = 1e-4
    jitter: float = 1e-5
    verbose: bool = False
    log_interval: int = 20


@dataclass
class GlobalWGPCemResult:
    field: SparseGPProjectionField
    gamma: torch.Tensor
    gate: torch.Tensor
    history: list[dict[str, float]]
    train_sec: float
    diagnostics: dict[str, Any]


def _embedding(
    field: SparseGPProjectionField,
    X_local: torch.Tensor,
    X_anchor: torch.Tensor,
    R: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "anchor":
        return field.z_anchor_conditioned(X_local, X_anchor, R)
    if mode == "pointwise":
        return field.z_pointwise(X_local, X_anchor, R)
    raise ValueError(f"Unknown mode {mode!r}")


def _outlier_logprob(y: torch.Tensor, log_noise_var: torch.Tensor, cfg: GlobalWGPCemConfig) -> torch.Tensor:
    if cfg.background == "normal":
        scale2 = torch.as_tensor(float(cfg.background_scale) ** 2, device=y.device, dtype=y.dtype)
        return -0.5 * (math.log(2.0 * math.pi) + torch.log(scale2) + y.pow(2) / scale2)
    if cfg.background == "uniform":
        return torch.full_like(
            y,
            -0.5 * (math.log(2.0 * math.pi) + float(log_noise_var.detach().item()) + float(cfg.outlier_sigma_mult) ** 2),
        )
    raise ValueError(f"Unknown background {cfg.background!r}")


def _coverage_penalty(gamma: torch.Tensor, neighbor_indices: torch.Tensor, n_train: int, c_min: float) -> torch.Tensor:
    flat_idx = neighbor_indices.reshape(-1)
    flat_g = gamma.reshape(-1)
    sums = torch.zeros(n_train, device=gamma.device, dtype=gamma.dtype)
    counts = torch.zeros(n_train, device=gamma.device, dtype=gamma.dtype)
    sums.scatter_add_(0, flat_idx, flat_g)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_g))
    mask = counts > 0
    cov = sums[mask] / counts[mask].clamp_min(1.0)
    return torch.relu(float(c_min) - cov).pow(2).mean() if cov.numel() else gamma.sum() * 0.0


def train_global_wgp_cem(
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    neighbor_indices: torch.Tensor,
    X_inducing: torch.Tensor,
    n_train: int,
    Q: int,
    init_W: torch.Tensor | None,
    init_R_mu: torch.Tensor | None = None,
    cfg: GlobalWGPCemConfig,
) -> GlobalWGPCemResult:
    device = X_anchor.device
    dtype = X_anchor.dtype
    A, n, D = X_neighbors.shape
    field = SparseGPProjectionField(
        U=X_inducing,
        Q=Q,
        D=D,
        init_W=init_W,
        init_R_mu=init_R_mu,
        init_log_std=cfg.init_log_std,
        lengthscale=float(cfg.init_w_lengthscale if cfg.init_w_lengthscale is not None else cfg.init_lengthscale),
        var_w=1.0,
        train_lengthscale=cfg.train_w_lengthscale,
        train_var_w=cfg.train_w_var,
    ).to(device)
    field.R_log_std.requires_grad_(bool(cfg.train_R_log_std))
    gate = nn.Parameter(torch.zeros(A, Q + 1, device=device, dtype=dtype))
    log_lengthscale = nn.Parameter(torch.tensor(math.log(cfg.init_lengthscale), device=device, dtype=dtype))
    log_amp = nn.Parameter(torch.tensor(math.log(cfg.init_amp), device=device, dtype=dtype))
    log_noise_var = nn.Parameter(torch.tensor(math.log(cfg.init_noise_var), device=device, dtype=dtype))
    if not cfg.train_kernel:
        log_lengthscale.requires_grad_(False)
        log_amp.requires_grad_(False)
    if not cfg.train_noise:
        log_noise_var.requires_grad_(False)
    params = [p for p in field.parameters() if p.requires_grad] + [gate]
    for p in (log_lengthscale, log_amp, log_noise_var):
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.Adam(params, lr=float(cfg.lr))
    gamma = torch.full((A, n), float(cfg.gamma_init), device=device, dtype=dtype)
    gamma = gamma.clamp(float(cfg.gamma_floor), 1.0 - float(cfg.gamma_floor))
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    t0 = time.perf_counter()
    for step in range(1, int(cfg.steps) + 1):
        if step == 1 or (int(cfg.c_refresh) > 0 and (step - 1) % int(cfg.c_refresh) == 0):
            with torch.no_grad():
                R_mean = field.sample_R(1, mean_only=True)
                Z_m, z_a_m, _ = _embedding(field, X_neighbors, X_anchor, R_mean, cfg.mode)
                K_m = rbf_kernel_from_z(Z_m, log_lengthscale.detach(), log_amp.detach(), jitter=cfg.jitter)
                gamma = update_gamma(
                    Z=Z_m,
                    z_anchor=z_a_m,
                    K=K_m,
                    y=y_neighbors,
                    gamma=gamma,
                    gate=gate.detach(),
                    noise_var=torch.exp(log_noise_var.detach()),
                    outlier_logprob=_outlier_logprob(y_neighbors, log_noise_var.detach(), cfg),
                    temperature=float(cfg.gamma_temperature),
                    damping=float(cfg.gamma_damping),
                    gamma_floor=float(cfg.gamma_floor),
                )

        opt.zero_grad()
        R_s = field.sample_R(max(1, int(cfg.mc_train)), mean_only=not bool(cfg.sample_train_R))
        Z, z_anchor, W_anchor = _embedding(field, X_neighbors, X_anchor, R_s, cfg.mode)
        K = rbf_kernel_from_z(Z, log_lengthscale, log_amp, jitter=cfg.jitter)
        out_lp = _outlier_logprob(y_neighbors, log_noise_var, cfg)
        bound = collapsed_bound(
            Z=Z,
            z_anchor=z_anchor,
            K=K,
            y=y_neighbors,
            gamma=gamma.detach(),
            gate=gate,
            noise_var=torch.exp(log_noise_var),
            outlier_logprob=out_lp,
            gamma_floor=float(cfg.gamma_floor),
            jitter=cfg.jitter,
        )
        data_loss = -float(cfg.composite_temperature) * bound.mean()
        beta = float(cfg.beta_kl) * min(1.0, step / max(1, int(cfg.kl_warmup_steps)))
        kl = field.kl() / max(1, A)
        cov_pen = _coverage_penalty(gamma, neighbor_indices, n_train, float(cfg.coverage_min))
        gate_pen = gate[:, 1:].pow(2).mean()
        loss = data_loss + beta * kl + float(cfg.coverage_weight) * cov_pen + float(cfg.gate_l2) * gate_pen
        should_record = step == 1 or step % int(cfg.log_interval) == 0 or step == int(cfg.steps)
        grad_diag: dict[str, float] = {}
        if should_record:
            g_data_mu = torch.autograd.grad(
                data_loss,
                field.R_mu,
                retain_graph=True,
                allow_unused=True,
            )[0]
            g_kl_mu = torch.autograd.grad(
                beta * kl,
                field.R_mu,
                retain_graph=True,
                allow_unused=True,
            )[0]
            data_mu = float(torch.linalg.norm(g_data_mu).detach().item()) if g_data_mu is not None else 0.0
            kl_mu = float(torch.linalg.norm(g_kl_mu).detach().item()) if g_kl_mu is not None else 0.0
            if field.R_log_std.requires_grad:
                g_data_std = torch.autograd.grad(
                    data_loss,
                    field.R_log_std,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                g_kl_std = torch.autograd.grad(
                    beta * kl,
                    field.R_log_std,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                data_std = float(torch.linalg.norm(g_data_std).detach().item()) if g_data_std is not None else 0.0
                kl_std = float(torch.linalg.norm(g_kl_std).detach().item()) if g_kl_std is not None else 0.0
                std_ratio = data_std / (kl_std + 1e-12)
            else:
                data_std = 0.0
                kl_std = 0.0
                std_ratio = float("nan")
            grad_diag = {
                "grad_data_R_mu": data_mu,
                "grad_scaled_KL_R_mu": kl_mu,
                "grad_data_over_scaled_KL_R_mu": data_mu / (kl_mu + 1e-12),
                "grad_data_R_log_std": data_std,
                "grad_scaled_KL_R_log_std": kl_std,
                "grad_data_over_scaled_KL_R_log_std": std_ratio,
            }
        loss.backward()
        opt.step()

        val = float(loss.detach().item())
        if val < best_loss and torch.isfinite(loss.detach()):
            best_loss = val
            best_state = {
                "R_mu": field.R_mu.detach().clone(),
                "R_log_std": field.R_log_std.detach().clone(),
                "log_w_l": field.log_lengthscale.detach().clone(),
                "log_w_var": field.log_var_w.detach().clone(),
                "gate": gate.detach().clone(),
                "log_l": log_lengthscale.detach().clone(),
                "log_amp": log_amp.detach().clone(),
                "log_noise": log_noise_var.detach().clone(),
                "gamma": gamma.detach().clone(),
            }
        if should_record:
            ent = -(gamma * gamma.log() + (1.0 - gamma) * (1.0 - gamma).log()).mean()
            rec = {
                "step": float(step),
                "loss": val,
                "data_loss": float(data_loss.detach().item()),
                "kl_per_anchor": float(kl.detach().item()),
                "beta": float(beta),
                "coverage_penalty": float(cov_pen.detach().item()),
                "gate_l2": float(gate_pen.detach().item()),
                "gamma_mean": float(gamma.mean().detach().item()),
                "gamma_entropy": float(ent.detach().item()),
                "R_std_median": float(torch.exp(field.R_log_std).median().detach().item()),
            }
            rec.update(grad_diag)
            history.append(rec)
            if cfg.verbose:
                print(
                    f"[global-wgp {cfg.mode} step {step:04d}] loss={rec['loss']:.4f} "
                    f"data={rec['data_loss']:.4f} kl={rec['kl_per_anchor']:.4f} "
                    f"cov={rec['coverage_penalty']:.4f} H={rec['gamma_entropy']:.3f}",
                    flush=True,
                )

    if best_state is not None:
        with torch.no_grad():
            field.R_mu.copy_(best_state["R_mu"])
            field.R_log_std.copy_(best_state["R_log_std"])
            field.log_lengthscale.copy_(best_state["log_w_l"])
            field.log_var_w.copy_(best_state["log_w_var"])
            gate.copy_(best_state["gate"])
            log_lengthscale.copy_(best_state["log_l"])
            log_amp.copy_(best_state["log_amp"])
            log_noise_var.copy_(best_state["log_noise"])
            gamma = best_state["gamma"]

    train_sec = time.perf_counter() - t0
    with torch.no_grad():
        R_mean = field.sample_R(1, mean_only=True)
        _, _, W_anchor_mean = _embedding(field, X_neighbors, X_anchor, R_mean, cfg.mode)
        W_diag = W_anchor_mean.squeeze(0)
        diag = per_anchor_W_diagnostics(W_diag, W_ref=init_W if init_W is not None else None)
        ent = -(gamma * gamma.log() + (1.0 - gamma) * (1.0 - gamma).log()).mean()
        diag.update(
            {
                "train_sec": float(train_sec),
                "best_loss": float(best_loss),
                "gamma_mean": float(gamma.mean().item()),
                "gamma_entropy": float(ent.item()),
                "R_std_median": float(torch.exp(field.R_log_std).median().item()),
                "kl_per_anchor": float((field.kl() / max(1, A)).item()),
                "log_w_lengthscale": float(field.log_lengthscale.detach().item()),
                "log_w_var": float(field.log_var_w.detach().item()),
                "log_lengthscale": float(log_lengthscale.detach().item()),
                "log_amp": float(log_amp.detach().item()),
                "log_noise_var": float(log_noise_var.detach().item()),
                "history_tail": history[-5:],
                "mode": cfg.mode,
                "background": cfg.background,
                "coverage_weight": float(cfg.coverage_weight),
                "gamma_floor": float(cfg.gamma_floor),
                "composite_temperature": float(cfg.composite_temperature),
            }
        )
    return GlobalWGPCemResult(
        field=field,
        gamma=gamma.detach(),
        gate=gate.detach(),
        history=history,
        train_sec=float(train_sec),
        diagnostics=diag,
    )


def predict_global_wgp_jumpgp(
    *,
    field: SparseGPProjectionField,
    X_test: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    n_neighbors: int,
    mode: str,
    n_samples: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    """Sample global R, rebuild latent KNN per sample, and refit JumpGP-CEM."""
    T, C, D = X_candidate.shape
    Q = field.Q
    mu_s = np.empty((int(n_samples), T), dtype=np.float64)
    sig_s = np.empty((int(n_samples), T), dtype=np.float64)
    t0 = time.perf_counter()
    with torch.no_grad():
        R_samples = field.sample_R(int(n_samples), mean_only=False)
    for s in range(int(n_samples)):
        R = R_samples[s : s + 1]
        if mode == "anchor":
            W_test = field.W_from_R(X_test, R).squeeze(0)
            Z_candidate = torch.einsum("tcd,tqd->tcq", X_candidate, W_test)
            z_test = torch.einsum("td,tqd->tq", X_test, W_test)
        elif mode == "pointwise":
            W_test = field.W_from_R(X_test, R).squeeze(0)
            z_test = torch.einsum("td,tqd->tq", X_test, W_test)
            X_flat = X_candidate.reshape(T * C, D)
            W_flat = field.W_from_R(X_flat, R).squeeze(0)
            Z_candidate = torch.einsum("tcd,tcqd->tcq", X_candidate, W_flat.reshape(T, C, Q, D))
        else:
            raise ValueError(mode)
        d2 = (Z_candidate - z_test.unsqueeze(1)).pow(2).sum(-1)
        idx = torch.topk(d2, k=int(n_neighbors), largest=False, dim=1).indices
        for t in range(T):
            Z_nb = Z_candidate[t, idx[t]].contiguous()
            y_nb = y_candidate[t, idx[t]].contiguous()
            mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
                Z_nb,
                y_nb.view(-1, 1),
                z_test[t].view(1, -1),
                mode="CEM",
                flag=False,
                device=device,
            )
            mu_s[s, t] = float(mu_t.detach().cpu().reshape(-1)[0].item())
            sig_s[s, t] = float(torch.sqrt(torch.clamp(sig2_t, min=1e-12)).detach().cpu().reshape(-1)[0].item())
    elapsed = time.perf_counter() - t0
    mu = mu_s.mean(axis=0)
    var = np.mean(sig_s**2 + (mu_s - mu[None, :]) ** 2, axis=0)
    return {
        "mu": mu,
        "sigma": np.sqrt(np.maximum(var, 1e-12)),
        "mu_samples": mu_s,
        "sigma_samples": sig_s,
        "elapsed_sec": float(elapsed),
    }


__all__ = [
    "GlobalWGPCemConfig",
    "GlobalWGPCemResult",
    "SparseGPProjectionField",
    "predict_global_wgp_jumpgp",
    "train_global_wgp_cem",
]
