"""
CEM-aligned generalized EM for the local projection ``W_j ∈ R^{Q × D}``.

The current LMJGP trains a variational ELBO that approximates a different model
than the CEM-JumpGP refit used at prediction time. This module implements a
**generalized EM** procedure whose target *is exactly* the CEM-JumpGP predictor:

    1. **C-step (frozen W)** — for each anchor ``j`` call ``jumpgp_ld_wrapper(...,
       mode='CEM')`` on ``Z_j = X_j W_j^T`` and cache the CEM internals
       ``(r_j, w_j, logtheta_j, ms_j)``.

    2. **M-step (frozen CEM internals)** — update ``W = (W_0, {C_j})`` to maximise

           sum_j  GP_logmarginal(Z_in,j; logtheta_j, ms_j, sigma_j)
                + λ_g · sum_j  -BCE(sigmoid(w_{j,0} + Z_j w_{j,1:}), r_j)
                - λ_s · sum_jj' S_jj' ‖C_j - C_j'‖_F^2
                - λ_r · sum_j sum_q (‖W_{j,q,:}‖^2 - 1)^2

       on the inlier subset ``I_j = {i : r_{ji} = 1}``.

The projection itself is parameterised in the low-rank coefficient form

    W_j = W_0 + C_j V^T,   V ∈ R^{D × Rv} fixed (PLS+PCA+random orthonormalised),

so the per-anchor degrees of freedom drop from ``QD`` to ``QR_v``. Setting
``V = I_D`` recovers the unconstrained "free per-anchor W" model used as an
upper-bound ablation.

The differentiable surrogate kernel uses the *same* RBF parameterisation as
JumpGP_LD (``ell = exp(logtheta[:Q])``, ``s_f = exp(logtheta[Q])``,
``sigma_n = exp(logtheta[-1])``) so the M-step reuses the kernel hyperparameters
that the C-step just selected.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Coefficient-form projection model
# =====================================================================
class CoeffProjection(nn.Module):
    """``W_j = rowNorm(W_0 + α · C_j V^T)`` (row-normalisation optional).

    Args:
        T: number of test anchors.
        Q: latent dimension.
        D: input dimension.
        V: ``[D, Rv]`` fixed input-side basis (set ``V = I_D`` for the free per-anchor model).
        W0_init: optional ``[Q, D]`` initial value for ``W_0``. When ``row_normalize=True``
            we recommend passing a non-zero ``W0_init`` (e.g. ``W_pls``); the rows are
            normalised lazily inside :meth:`W_all`, so the raw magnitude of the init
            does not matter.
        learn_W0: whether ``W_0`` is a learnable parameter.
        init_C_scale: standard deviation of the Gaussian initialising ``C``.
        coeff_scale: ``α``, the multiplier applied to ``C_j V^T`` *before* row
            normalisation. With row normalisation on, ``α`` controls how much the
            local coefficient can rotate ``W_0`` (raw row norms are removed). Try
            ``α ∈ {1, 3, 10}``.
        row_normalize: if ``True`` (recommended), each output row is divided by its
            L2 norm. This decouples the M-step from the absolute scale of ``W_0``
            (e.g. PLS in raw-X scale) and lets the JumpGP-CEM pick its own
            lengthscale unconfounded by the projection magnitude.
        row_norm_eps: clamp added to the row norm for numerical stability.
    """

    def __init__(
        self,
        *,
        T: int,
        Q: int,
        D: int,
        V: torch.Tensor,
        W0_init: torch.Tensor | None = None,
        learn_W0: bool = True,
        init_C_scale: float = 0.0,
        coeff_scale: float = 1.0,
        row_normalize: bool = False,
        row_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if V.dim() != 2 or V.shape[0] != D:
            raise ValueError(f"V must be [D, Rv]=[{D}, *]; got {tuple(V.shape)}.")
        Rv = V.shape[1]
        self.T, self.Q, self.D, self.Rv = int(T), int(Q), int(D), int(Rv)
        self.register_buffer("V", V.contiguous())
        if W0_init is None:
            W0_init = torch.zeros(Q, D)
        if W0_init.shape != (Q, D):
            raise ValueError(f"W0_init must be [Q, D]=[{Q}, {D}]; got {tuple(W0_init.shape)}.")
        # When row_normalize is on, pre-normalise W_0_init so that ||W_0_q|| = 1.
        # Otherwise α C_j V^T (O(1) magnitude) is dwarfed by raw-X PLS rows
        # (O(10^4)) and rowNorm(W_0 + α C V^T) ≈ rowNorm(W_0), killing local
        # adaptation entirely. We do this at init only — W_0 is then free to drift
        # during training but stays in a comparable scale band.
        if row_normalize:
            row_norm = torch.linalg.norm(W0_init, dim=-1, keepdim=True).clamp_min(row_norm_eps)
            W0_init = W0_init / row_norm
        if learn_W0:
            self.W0 = nn.Parameter(W0_init.contiguous())
        else:
            self.register_buffer("W0", W0_init.contiguous())
        if init_C_scale > 0:
            C0 = init_C_scale * torch.randn(T, Q, Rv)
        else:
            C0 = torch.zeros(T, Q, Rv)
        self.C = nn.Parameter(C0.contiguous())
        self.coeff_scale = float(coeff_scale)
        self.row_normalize = bool(row_normalize)
        self.row_norm_eps = float(row_norm_eps)

    def _maybe_row_normalize(self, W: torch.Tensor) -> torch.Tensor:
        if not self.row_normalize:
            return W
        norm = torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(self.row_norm_eps)
        return W / norm

    def W_all(self) -> torch.Tensor:
        W = self.W0.unsqueeze(0) + self.coeff_scale * torch.einsum("tqr,dr->tqd", self.C, self.V)
        return self._maybe_row_normalize(W)

    def W_one(self, t: int) -> torch.Tensor:
        W = self.W0 + self.coeff_scale * (self.C[t] @ self.V.T)
        return self._maybe_row_normalize(W)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# Differentiable RBF kernel matching JumpGP_LD's parameterisation
# =====================================================================
def _rbf_kernel_jumpgp(Z: torch.Tensor, logtheta: torch.Tensor, *, jitter: float = 1e-6) -> torch.Tensor:
    """``K[i, j] = s_f^2 exp(-0.5 ‖(z_i - z_j) / ell‖^2)`` with ``ell = exp(logtheta[:Q])``,
    ``s_f = exp(logtheta[Q])``. Diagonal jitter added for stability.

    The dtype follows ``Z``; ``logtheta`` is cast to ``Z.dtype`` so gradients flow into ``Z``.
    """
    M, Q = Z.shape
    lt = logtheta.to(dtype=Z.dtype, device=Z.device)
    if lt.numel() < Q + 1:
        raise ValueError(f"logtheta has {lt.numel()} entries, need >= {Q + 1}.")
    ell = torch.exp(lt[:Q]).clamp_min(1e-6)
    s_f = torch.exp(lt[Q]).clamp_min(1e-6)
    Z_scaled = Z / ell
    diff = Z_scaled.unsqueeze(1) - Z_scaled.unsqueeze(0)
    d2 = (diff * diff).sum(dim=-1)
    K = s_f.pow(2) * torch.exp(-0.5 * d2)
    K = K + jitter * torch.eye(M, device=Z.device, dtype=Z.dtype)
    return K


def gp_log_marginal_inliers(
    Z_in: torch.Tensor,
    y_in: torch.Tensor,
    logtheta: torch.Tensor,
    ms: torch.Tensor,
    *,
    jitter: float = 1e-6,
    sigma_n_floor: float = 1e-3,
) -> torch.Tensor:
    """Differentiable GP log marginal likelihood with frozen ``logtheta`` / ``ms``.

    Uses ``Ky = K(Z_in) + sigma_n^2 I``; ``sigma_n = exp(logtheta[-1])``.
    """
    n = Z_in.shape[0]
    K = _rbf_kernel_jumpgp(Z_in, logtheta, jitter=jitter)
    sigma_n = torch.exp(logtheta[-1].to(dtype=Z_in.dtype, device=Z_in.device)).clamp_min(sigma_n_floor)
    Ky = K + sigma_n.pow(2) * torch.eye(n, device=Z_in.device, dtype=Z_in.dtype)
    L = torch.linalg.cholesky(Ky)
    mean = ms.to(dtype=Z_in.dtype, device=Z_in.device) * torch.ones(n, device=Z_in.device, dtype=Z_in.dtype)
    diff = (y_in - mean).view(-1, 1)
    alpha = torch.cholesky_solve(diff, L).squeeze(-1)
    quad = (diff.squeeze(-1) * alpha).sum()
    logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum()
    return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


def gp_anchor_predictive_nll(
    Z_in: torch.Tensor,
    y_in: torch.Tensor,
    z_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    logtheta: torch.Tensor,
    ms: torch.Tensor,
    *,
    jitter: float = 1e-6,
    sigma_n_floor: float = 1e-3,
    pred_var_floor: float = 1e-8,
) -> torch.Tensor:
    """Negative log predictive density for a labeled anchor."""
    n = Z_in.shape[0]
    if n < 2:
        raise ValueError("Need at least two inlier neighbours for anchor predictive NLL.")
    dtype = Z_in.dtype
    device = Z_in.device
    lt = logtheta.to(dtype=dtype, device=device)
    Q = Z_in.shape[1]
    ell = torch.exp(lt[:Q]).clamp_min(1e-6)
    s_f = torch.exp(lt[Q]).clamp_min(1e-6)
    sigma_n = torch.exp(lt[-1]).clamp_min(sigma_n_floor)

    K = _rbf_kernel_jumpgp(Z_in, lt, jitter=jitter)
    Ky = K + sigma_n.pow(2) * torch.eye(n, device=device, dtype=dtype)
    L = torch.linalg.cholesky(Ky)

    mean0 = ms.to(dtype=dtype, device=device).reshape(()).expand(n)
    diff_y = (y_in.to(dtype=dtype, device=device).view(-1) - mean0).view(-1, 1)
    alpha = torch.cholesky_solve(diff_y, L).squeeze(-1)

    z = z_anchor.to(dtype=dtype, device=device).view(1, Q)
    d2 = ((Z_in / ell) - (z / ell)).pow(2).sum(dim=-1)
    k_star = s_f.pow(2) * torch.exp(-0.5 * d2)
    pred_mean = ms.to(dtype=dtype, device=device).reshape(()) + k_star @ alpha
    v = torch.cholesky_solve(k_star.view(-1, 1), L).squeeze(-1)
    pred_var = (s_f.pow(2) + sigma_n.pow(2) - (k_star * v).sum()).clamp_min(float(pred_var_floor))
    err2 = (y_anchor.to(dtype=dtype, device=device).reshape(()) - pred_mean).pow(2)
    return 0.5 * (torch.log(2.0 * math.pi * pred_var) + err2 / pred_var)


def gate_bce_loss(Z_nb: torch.Tensor, w_gate: torch.Tensor, r_targets: torch.Tensor) -> torch.Tensor:
    """``BCE(sigmoid(w[0] + Z @ w[1:]), r)`` with ``Z: [n, Q]``, ``w_gate: [1+Q]``, ``r: [n]``."""
    w = w_gate.to(dtype=Z_nb.dtype, device=Z_nb.device).flatten()
    if w.numel() != Z_nb.shape[1] + 1:
        raise ValueError(f"w_gate has {w.numel()} entries, expected {Z_nb.shape[1] + 1}.")
    logits = w[0] + Z_nb @ w[1:]
    target = r_targets.to(dtype=Z_nb.dtype, device=Z_nb.device).clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


# =====================================================================
# C-step: cache CEM internals from JumpGP_LD
# =====================================================================
@dataclass
class CemCacheEntry:
    success: bool
    r: torch.Tensor | None = None  # [n] in {0, 1}
    w: torch.Tensor | None = None  # [1 + Q]
    logtheta: torch.Tensor | None = None  # [Q + 2]
    ms: torch.Tensor | None = None  # scalar
    mu_t: float = float("nan")
    sig2_t: float = float("nan")
    n_inliers: int = 0
    error: str | None = None


def cem_step_all(
    model: CoeffProjection,
    *,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    X_anchors: torch.Tensor,
    device: torch.device,
    progress: bool = False,
    desc: str | None = None,
) -> list[CemCacheEntry]:
    """Run JumpGP-CEM on the projected coordinates with the *current* (detached) ``W``."""
    from shared.utils1 import jumpgp_ld_wrapper

    T = X_anchors.shape[0]
    if len(X_neighbors_list) != T or len(y_neighbors_list) != T:
        raise ValueError("X_anchors / X_neighbors_list / y_neighbors_list must agree on T.")
    with torch.no_grad():
        W_all = model.W_all().detach()

    iterator: Iterable[int] = range(T)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc=desc or "CEM step", total=T)

    cache: list[CemCacheEntry] = []
    for t in iterator:
        Wj = W_all[t]
        X_nb = X_neighbors_list[t]
        y_nb = y_neighbors_list[t].view(-1)
        x_test = X_anchors[t]
        Z_nb = X_nb @ Wj.T
        z_test = x_test @ Wj.T
        try:
            mu_t, sig2_t, jgp_model, _ = jumpgp_ld_wrapper(
                Z_nb,
                y_nb.view(-1, 1),
                z_test.view(1, -1),
                mode="CEM",
                flag=False,
                device=device,
            )
            r_raw = jgp_model["r"].flatten()
            r = (r_raw > 0.5).to(dtype=torch.float32) if r_raw.dtype == torch.bool else r_raw.float()
            w = jgp_model["w"].float().flatten()
            lt = jgp_model["logtheta"].float().flatten()
            ms_raw = jgp_model.get("ms")
            if torch.is_tensor(ms_raw):
                ms = ms_raw.float().flatten()[0]
            else:
                ms = torch.tensor(float(ms_raw), dtype=torch.float32, device=device)
            cache.append(
                CemCacheEntry(
                    success=True,
                    r=r.to(device),
                    w=w.to(device),
                    logtheta=lt.to(device),
                    ms=ms.to(device),
                    mu_t=float(mu_t.detach().cpu().reshape(-1)[0].item()),
                    sig2_t=float(sig2_t.detach().cpu().reshape(-1)[0].item()),
                    n_inliers=int((r > 0.5).sum().item()),
                )
            )
        except Exception as exc:  # noqa: BLE001
            cache.append(CemCacheEntry(success=False, error=str(exc)))
    return cache


# =====================================================================
# Anchor smoothness penalty on C
# =====================================================================
def anchor_rbf_kernel(X_anchors: torch.Tensor, *, lengthscale: float | None = None) -> torch.Tensor:
    pd = torch.cdist(X_anchors, X_anchors, p=2.0)
    if lengthscale is None:
        T = pd.shape[0]
        if T < 2:
            return torch.eye(T, device=pd.device, dtype=pd.dtype)
        iu = torch.triu_indices(T, T, offset=1, device=pd.device)
        med = pd[iu[0], iu[1]].median().clamp_min(1e-3)
        lengthscale = float(med.item())
    return torch.exp(-0.5 * pd.pow(2) / (max(lengthscale, 1e-6) ** 2))


def laplacian_smoothness(
    C: torch.Tensor,
    K_anchor: torch.Tensor,
    *,
    normalize_by_graph_weight: bool = True,
) -> torch.Tensor:
    """``Σ_{j,j'} K_{jj'} ‖C_j - C_{j'}‖_F^2`` via graph Laplacian.

    With ``normalize_by_graph_weight=True`` divides by ``Σ_{j,j'} K_{jj'}`` so the
    penalty is O(1) regardless of the number of anchors ``T`` (otherwise it scales
    as ``O(T^2)``).
    """
    T = C.shape[0]
    C_flat = C.reshape(T, -1)
    deg = K_anchor.sum(dim=1)
    L_C = deg.unsqueeze(-1) * C_flat - K_anchor @ C_flat
    val = 2.0 * (L_C * C_flat).sum()
    if normalize_by_graph_weight:
        denom = K_anchor.sum().clamp_min(1e-12)
        val = val / denom
    return val


def row_norm_penalty(W_all: torch.Tensor, *, target_row_norm_sq: float = 1.0) -> torch.Tensor:
    """``mean over (t, q) of ((‖W_{t,q,:}‖² - target) / target)²``.

    Dividing by ``target`` keeps the penalty scale-aware so it stays O(1) even
    when ``W`` rows have large absolute magnitudes (e.g. PLS in raw-X scale on
    unstandardised UCI features). Set ``target_row_norm_sq = 1`` to recover the
    classic ``(‖W‖² - 1)²`` form.
    """
    row_sq = (W_all * W_all).sum(dim=-1)  # [T, Q]
    target = max(float(target_row_norm_sq), 1e-12)
    return ((row_sq / target - 1.0).pow(2)).mean()


# =====================================================================
# M-step loss
# =====================================================================
def m_step_loss(
    model: CoeffProjection,
    *,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    X_anchors: torch.Tensor | None = None,
    y_anchors: torch.Tensor | None = None,
    cache: list[CemCacheEntry],
    K_anchor: torch.Tensor | None,
    lambda_gate: float,
    lambda_smooth: float,
    lambda_scale: float,
    lambda_anchor_pred: float = 0.0,
    anchor_pred_var_floor: float = 1e-8,
    target_row_norm_sq: float = 1.0,
    normalize_smoothness: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean per-anchor M-step loss = ``-GP_NLL + λ_g BCE + λ_s smooth + λ_r row``."""
    W_all = model.W_all()  # [T, Q, D]
    device = W_all.device
    dtype = W_all.dtype
    total_gp = torch.zeros((), device=device, dtype=dtype)
    total_gate = torch.zeros((), device=device, dtype=dtype)
    total_anchor = torch.zeros((), device=device, dtype=dtype)
    n_gp = 0
    n_gate = 0
    n_anchor = 0
    n_skip = 0

    for t, c in enumerate(cache):
        if not c.success or c.r is None:
            n_skip += 1
            continue
        X_nb = X_neighbors_list[t].to(device=device, dtype=dtype)
        y_nb = y_neighbors_list[t].to(device=device, dtype=dtype).view(-1)
        Wj = W_all[t]
        Z_nb = X_nb @ Wj.T

        # Gate BCE on the full neighborhood (all n points contribute).
        try:
            gate_val = gate_bce_loss(Z_nb, c.w, c.r)
            total_gate = total_gate + gate_val
            n_gate += 1
        except Exception:
            pass

        # GP log marginal on the inlier subset only.
        idx = (c.r > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 3:
            continue
        Z_in = Z_nb[idx]
        y_in = y_nb[idx]
        try:
            gp_ll = gp_log_marginal_inliers(Z_in, y_in, c.logtheta, c.ms)
            total_gp = total_gp + gp_ll
            n_gp += 1
        except RuntimeError:
            continue

        if lambda_anchor_pred > 0 and X_anchors is not None and y_anchors is not None:
            try:
                z_anchor = X_anchors[t].to(device=device, dtype=dtype) @ Wj.T
                anchor_nll = gp_anchor_predictive_nll(
                    Z_in,
                    y_in,
                    z_anchor,
                    y_anchors[t],
                    c.logtheta,
                    c.ms,
                    pred_var_floor=anchor_pred_var_floor,
                )
                total_anchor = total_anchor + anchor_nll
                n_anchor += 1
            except (RuntimeError, ValueError):
                pass

    obj_gp = total_gp / max(n_gp, 1)
    obj_gate = total_gate / max(n_gate, 1)
    obj_anchor = total_anchor / max(n_anchor, 1)

    if K_anchor is not None and lambda_smooth > 0:
        smooth_pen = laplacian_smoothness(
            model.C, K_anchor, normalize_by_graph_weight=normalize_smoothness,
        )
    else:
        smooth_pen = torch.zeros((), device=device, dtype=dtype)
    if lambda_scale > 0:
        row_pen = row_norm_penalty(W_all, target_row_norm_sq=target_row_norm_sq)
    else:
        row_pen = torch.zeros((), device=device, dtype=dtype)

    loss = (
        -obj_gp
        + float(lambda_gate) * obj_gate
        + float(lambda_smooth) * smooth_pen
        + float(lambda_scale) * row_pen
        + float(lambda_anchor_pred) * obj_anchor
    )
    info = {
        "gp_ll_mean": float(obj_gp.detach().item()),
        "gate_bce_mean": float(obj_gate.detach().item()),
        "anchor_pred_nll_mean": float(obj_anchor.detach().item()),
        "smoothness": float(smooth_pen.detach().item()),
        "row_norm": float(row_pen.detach().item()),
        "loss": float(loss.detach().item()),
        "n_gp": int(n_gp),
        "n_gate": int(n_gate),
        "n_anchor": int(n_anchor),
        "n_skip": int(n_skip),
    }
    return loss, info


# =====================================================================
# Snapshot diagnostics on (W, C)
# =====================================================================
def _snapshot(model: CoeffProjection, *, K_anchor: torch.Tensor | None, label: str, step: int) -> dict[str, float]:
    with torch.no_grad():
        W = model.W_all()
        WWT = torch.einsum("tqd,trd->tqr", W, W)
        eig = torch.linalg.eigvalsh(0.5 * (WWT + WWT.transpose(-2, -1))).clamp_min(0.0)
        tr = eig.sum(dim=-1)
        sum_sq = (eig ** 2).sum(dim=-1).clamp_min(1e-18)
        eff = (eig.sum(dim=-1) ** 2) / sum_sq
        row_sq = (W * W).sum(dim=-1)
        W_mean = W.mean(dim=0, keepdim=True)
        disp = torch.linalg.norm(W - W_mean, dim=(-2, -1))
        snap = {
            "label": label,
            "step": float(step),
            "median_trace_WWT": float(tr.median().item()),
            "median_eff_rank_WWT": float(eff.median().item()),
            "median_row_norm_sq_W": float(row_sq.median().item()),
            "median_frob_W": float(torch.linalg.norm(W, dim=(-2, -1)).median().item()),
            "anchor_dispersion_frob": float(disp.median().item()),
            "median_frob_C": float(torch.linalg.norm(model.C, dim=(-2, -1)).median().item()),
            "frob_W0": float(torch.linalg.norm(model.W0).item()),
        }
        if K_anchor is not None:
            snap["smoothness_value"] = float(laplacian_smoothness(model.C, K_anchor).item())
    return snap


# =====================================================================
# Outer EM training loop
# =====================================================================
@dataclass
class CemEmConfig:
    n_outer: int = 5
    n_m_steps: int = 100
    lr: float = 0.01
    lambda_gate: float = 0.1
    lambda_smooth: float = 0.05
    lambda_scale: float = 0.01
    lambda_anchor_pred: float = 0.0
    anchor_pred_var_floor: float = 0.05
    target_row_norm_sq: float = 1.0  # only used when row_normalize_W=False
    normalize_smoothness: bool = True  # divide Laplacian by Σ K_jj' (T-invariant)
    grad_clip: float = 10.0
    anchor_kernel_lengthscale: float | None = None
    snapshot_after_each_outer: bool = True
    cem_progress: bool = False
    verbose: bool = True
    # Per-outer evaluation + best-W tracking. ``track_best_W`` keeps the W with
    # the lowest M-step loss seen during training (set False to use the final W).
    # ``eval_callback(W) -> dict`` is run after each outer for monitoring; if it
    # returns a key ``best_metric`` it overrides the loss criterion (lower=better).
    track_best_W: bool = True
    best_metric_key: str = "loss"  # column name in eval_callback output, or 'loss'


@dataclass
class CemEmResult:
    W: torch.Tensor  # [T, Q, D] — best W (per cfg.track_best_W) or final W
    W_final: torch.Tensor  # [T, Q, D] — last-outer W regardless
    C: torch.Tensor
    C_final: torch.Tensor
    W0: torch.Tensor
    W0_final: torch.Tensor
    final_cem_cache: list[CemCacheEntry]
    history: list[dict[str, float]]
    snapshots: list[dict[str, float]] = field(default_factory=list)
    train_time_sec: float = 0.0
    num_trainable_params: int = 0
    best_outer: int = -1  # -1 if no tracking happened
    best_metric_value: float = float("nan")


def train_projection_cem_em(
    model: CoeffProjection,
    *,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    X_anchors: torch.Tensor,
    y_anchors: torch.Tensor | None = None,
    device: torch.device,
    cfg: CemEmConfig | None = None,
    label: str = "cem_em",
    eval_callback=None,  # Optional[Callable[[torch.Tensor, int], dict[str, float]]]
) -> CemEmResult:
    """Outer EM loop. Optionally tracks the best ``W`` (over outer iterations) by
    either the M-step loss or by an external ``eval_callback(W, outer)`` returning
    a metric named ``cfg.best_metric_key`` (lower=better)."""
    cfg = cfg or CemEmConfig()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )
    K_anchor: torch.Tensor | None = None
    if cfg.lambda_smooth > 0:
        K_anchor = anchor_rbf_kernel(
            X_anchors.to(device=device), lengthscale=cfg.anchor_kernel_lengthscale,
        ).detach()

    history: list[dict[str, float]] = []
    snapshots: list[dict[str, float]] = [_snapshot(model, K_anchor=K_anchor, label=label, step=0)]

    best_value = float("inf")
    best_outer = -1
    best_W: torch.Tensor | None = None
    best_C: torch.Tensor | None = None
    best_W0: torch.Tensor | None = None

    t0 = time.perf_counter()
    cache: list[CemCacheEntry] = []
    for outer in range(cfg.n_outer):
        cache = cem_step_all(
            model,
            X_neighbors_list=X_neighbors_list,
            y_neighbors_list=y_neighbors_list,
            X_anchors=X_anchors,
            device=device,
            progress=cfg.cem_progress,
            desc=f"{label}/CEM outer={outer + 1}",
        )
        n_success = sum(1 for c in cache if c.success)
        last_info: dict[str, float] = {}
        for step in range(1, cfg.n_m_steps + 1):
            optimizer.zero_grad()
            loss, info = m_step_loss(
                model,
                X_neighbors_list=X_neighbors_list,
                y_neighbors_list=y_neighbors_list,
                X_anchors=X_anchors,
                y_anchors=y_anchors,
                cache=cache,
                K_anchor=K_anchor,
                lambda_gate=cfg.lambda_gate,
                lambda_smooth=cfg.lambda_smooth,
                lambda_scale=cfg.lambda_scale,
                lambda_anchor_pred=cfg.lambda_anchor_pred,
                anchor_pred_var_floor=cfg.anchor_pred_var_floor,
                target_row_norm_sq=cfg.target_row_norm_sq,
                normalize_smoothness=cfg.normalize_smoothness,
            )
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=cfg.grad_clip,
                )
            optimizer.step()
            last_info = info
        last_info["outer"] = float(outer + 1)
        last_info["n_cem_success"] = float(n_success)

        # Optional external eval (e.g. a real JumpGP-CEM evaluation on test data).
        eval_metrics: dict[str, float] = {}
        if eval_callback is not None:
            with torch.no_grad():
                W_eval = model.W_all().detach()
            try:
                em = eval_callback(W_eval, outer + 1)
                if isinstance(em, dict):
                    eval_metrics = {f"eval_{k}": float(v) for k, v in em.items() if isinstance(v, (int, float))}
            except Exception as exc:  # noqa: BLE001
                eval_metrics = {"eval_error": -1.0}
                if cfg.verbose:
                    print(f"[{label}] eval_callback failed at outer {outer + 1}: {exc}")
        last_info.update(eval_metrics)
        history.append(last_info)

        if cfg.track_best_W:
            metric_key = cfg.best_metric_key
            if metric_key == "loss":
                metric_val = float(last_info.get("loss", float("inf")))
            else:
                # External metric (eval_callback). Try `eval_<key>` first, then `<key>`.
                metric_val = float(
                    last_info.get(f"eval_{metric_key}", last_info.get(metric_key, float("inf")))
                )
            if metric_val < best_value:
                best_value = metric_val
                best_outer = outer + 1
                with torch.no_grad():
                    best_W = model.W_all().detach().clone()
                    best_C = model.C.detach().clone()
                    best_W0 = model.W0.detach().clone()

        if cfg.verbose:
            extra = ""
            if eval_metrics:
                extra = " " + " ".join(f"{k.replace('eval_', '')}={v:.4f}" for k, v in eval_metrics.items())
            print(
                f"[{label}] outer={outer + 1}/{cfg.n_outer} "
                f"cem_ok={n_success}/{len(cache)} "
                f"loss={last_info['loss']:.4f} gp_ll={last_info['gp_ll_mean']:.4f} "
                f"gate={last_info['gate_bce_mean']:.4f} smooth={last_info['smoothness']:.4f} "
                f"row={last_info['row_norm']:.4f} anchor={last_info['anchor_pred_nll_mean']:.4f}" + extra
            )
        if cfg.snapshot_after_each_outer:
            snapshots.append(_snapshot(model, K_anchor=K_anchor, label=label, step=outer + 1))
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        W_final = model.W_all().detach()
        C_final = model.C.detach().clone()
        W0_final = model.W0.detach().clone()
    if best_W is None:
        best_W = W_final
    if best_C is None:
        best_C = C_final
    if best_W0 is None:
        best_W0 = W0_final
    return CemEmResult(
        W=best_W,
        W_final=W_final,
        C=best_C,
        C_final=C_final,
        W0=best_W0,
        W0_final=W0_final,
        final_cem_cache=cache,
        history=history,
        snapshots=snapshots,
        train_time_sec=float(elapsed),
        num_trainable_params=model.num_trainable(),
        best_outer=int(best_outer),
        best_metric_value=float(best_value),
    )


__all__ = [
    "CemCacheEntry",
    "CemEmConfig",
    "CemEmResult",
    "CoeffProjection",
    "anchor_rbf_kernel",
    "cem_step_all",
    "gate_bce_loss",
    "gp_anchor_predictive_nll",
    "gp_log_marginal_inliers",
    "laplacian_smoothness",
    "m_step_loss",
    "row_norm_penalty",
    "train_projection_cem_em",
]
