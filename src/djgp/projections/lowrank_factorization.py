"""
Low-rank tensor factorisation of the local projection ``W_j ∈ R^{Q × D}``.

Instead of giving every entry ``w_{j,k,d}`` an independent GP prior across anchors
(``Q × D`` GP coefficient functions, the current LMJGP setup), we factor::

    W_j = (C_0 + ΔC_j) V^T,   V ∈ R^{D × Rv}, C_0 ∈ R^{Q × Rv}, ΔC_j ∈ R^{Q × Rv}

so that the projection lives in the column span of a fixed input-side basis ``V`` and
only the much smaller coefficient tensor varies with anchor. This module implements:

- :func:`build_V_basis` — orthonormal ``V`` from ``[PLS, PCA, random]`` in raw-X scale.
- :func:`compute_pls_W0` — supervised initial ``W_pls ∈ R^{Q × D}``.
- :class:`LowRankProjectionModel` — three variants chosen by ``mode``:

    * ``"pls_only"``    — ``W_j = W_pls`` (no learnable parameters, anchor-independent
                          deterministic baseline).
    * ``"global_C"``    — ``W_j = C_0 V^T``, ``C_0`` learned globally (still
                          anchor-independent, but lives in the V column span).
    * ``"local_C_smooth"`` — ``W_j = (C_0 + ΔC_j) V^T``; ``ΔC_j`` per anchor with a
                          graph-Laplacian smoothness penalty over neighbouring anchors,
                          a deterministic surrogate for "GP prior on each c_{q,ℓ}(·)".

- :func:`train_lowrank_projection` — Adam + local-RBF-GP NLL on each neighbourhood,
  optional anchor-Laplacian smoothness penalty.

The output ``W: [T, Q, D]`` plugs directly into
:func:`djgp.projections.projected_jumpgp.run_projected_jumpgp` so we reuse the same
JumpGP-CEM evaluation as the metric-variant ablation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

LOWRANK_MODES = ("pls_only", "global_C", "local_C_smooth")


# =====================================================================
# 1. Basis construction (PLS + PCA + random, orthonormalised)
# =====================================================================
def _pls_directions_raw(X: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """``W ∈ R^{D × K}`` — PLS x_weights converted to raw-X scale."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pls = PLSRegression(n_components=K, scale=False)
    pls.fit(X_std, y.reshape(-1, 1))
    V_std = np.asarray(pls.x_weights_)  # [D, K]
    inv = 1.0 / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    return inv[:, None] * V_std


def _pca_directions_raw(X: np.ndarray, K: int) -> np.ndarray:
    """``V ∈ R^{D × K}`` — PCA components in raw-X scale."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=min(K, X_std.shape[1]))
    pca.fit(X_std)
    V_std = pca.components_.T  # [D, K]
    inv = 1.0 / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    return inv[:, None] * V_std


def build_V_basis(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_pls: int = 0,
    n_pca: int = 0,
    n_random: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build an orthonormal column basis ``V ∈ R^{D × Rv}`` from PLS + PCA + random
    directions in raw-X scale.

    The components are concatenated then QR-orthonormalised. ``Rv = min(D, n_pls + n_pca + n_random)``.

    Returns ``(V, info)`` where ``info`` carries the per-source counts and the column
    indices of the original PLS / PCA / random parts before the QR.
    """
    if n_pls < 0 or n_pca < 0 or n_random < 0:
        raise ValueError("n_pls, n_pca, n_random must be non-negative.")
    parts: list[np.ndarray] = []
    sources: list[str] = []
    D = X.shape[1]
    if n_pls > 0:
        V_pls = _pls_directions_raw(np.asarray(X, dtype=np.float64), np.asarray(y).ravel(), K=min(n_pls, D))
        parts.append(V_pls)
        sources.extend(["pls"] * V_pls.shape[1])
    if n_pca > 0:
        V_pca = _pca_directions_raw(np.asarray(X, dtype=np.float64), K=min(n_pca, D))
        parts.append(V_pca)
        sources.extend(["pca"] * V_pca.shape[1])
    if n_random > 0:
        rng = np.random.default_rng(seed)
        V_rand = rng.standard_normal((D, min(n_random, D)))
        parts.append(V_rand)
        sources.extend(["random"] * V_rand.shape[1])
    if not parts:
        raise ValueError("At least one of n_pls / n_pca / n_random must be > 0.")
    V_tilde = np.concatenate(parts, axis=1)  # [D, Rv_total]
    if V_tilde.shape[1] > D:
        # Cap at D, keep first D columns (PLS first, then PCA, then random).
        V_tilde = V_tilde[:, :D]
        sources = sources[:D]
    Q_, _ = np.linalg.qr(V_tilde, mode="reduced")  # [D, Rv]
    Rv = Q_.shape[1]
    info = {
        "n_pls": int(n_pls),
        "n_pca": int(n_pca),
        "n_random": int(n_random),
        "Rv": int(Rv),
        "source_per_column": sources[:Rv],
        "V_tilde_frob": float(np.linalg.norm(V_tilde)),
    }
    return Q_.astype(np.float64), info


def compute_pls_W0(X: np.ndarray, y: np.ndarray, *, Q: int) -> np.ndarray:
    """``W_pls ∈ R^{Q × D}`` from PLS x_weights, in raw-X scale (rows = directions)."""
    return _pls_directions_raw(np.asarray(X, dtype=np.float64), np.asarray(y).ravel(), K=Q).T


# =====================================================================
# 2. Differentiable local-GP NLL surrogate
# =====================================================================
def local_rbf_gp_nll_batch(
    Z: torch.Tensor,
    y: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_noise_var: torch.Tensor,
    *,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """
    Mean per-anchor neg-log-marginal-likelihood of a zero-mean RBF GP::

        K_t = exp(-0.5 * ||z_i - z_j||^2 / l^2) + (sigma^2 + jitter) I

    Args:
        Z: ``[T, n, Q]`` projected neighbourhood coordinates.
        y: ``[T, n]`` neighbour responses.
        log_lengthscale, log_noise_var: scalar tensors (shared across anchors here for
            simplicity; this is enough as a *training surrogate* — JumpGP at inference
            re-estimates per-anchor hyperparameters).

    Returns scalar mean NLL.
    """
    T, n, _ = Z.shape
    l2 = torch.exp(2.0 * log_lengthscale).clamp_min(1e-12)
    sig2 = torch.exp(log_noise_var).clamp_min(1e-12)
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(-1)
    K = torch.exp(-0.5 * d2 / l2)
    I = torch.eye(n, device=Z.device, dtype=Z.dtype).unsqueeze(0)
    K = K + (sig2 + jitter) * I
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
    quad = (y * alpha).sum(dim=-1)
    logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)
    nll = 0.5 * (quad + logdet + n * math.log(2.0 * math.pi))
    return nll.mean()


# =====================================================================
# 3. Anchor-graph smoothness penalty on ΔC_j
# =====================================================================
def anchor_rbf_kernel(X_anchors: torch.Tensor, *, lengthscale: float | None = None) -> torch.Tensor:
    """RBF kernel ``[T, T]``; lengthscale defaults to median pairwise distance."""
    pd = torch.cdist(X_anchors, X_anchors, p=2.0)
    if lengthscale is None:
        iu = torch.triu_indices(pd.shape[0], pd.shape[0], offset=1, device=pd.device)
        med = pd[iu[0], iu[1]].median().clamp_min(1e-3)
        lengthscale = float(med.item())
    return torch.exp(-0.5 * pd.pow(2) / (max(lengthscale, 1e-6) ** 2))


def laplacian_smoothness_penalty(C_local: torch.Tensor, K_anchor: torch.Tensor) -> torch.Tensor:
    """
    ``Σ_{j,j'} K_{jj'} || ΔC_j - ΔC_{j'} ||_F^2`` computed in ``O(T^2)`` via the graph
    Laplacian ``L = diag(K 1) - K``::

        sum_jj K_jj ||C_j - C_j'||^2 = 2 * trace(C_flat^T L C_flat).
    """
    T = C_local.shape[0]
    C_flat = C_local.reshape(T, -1)
    deg = K_anchor.sum(dim=1)
    L_C = deg.unsqueeze(-1) * C_flat - K_anchor @ C_flat
    return 2.0 * (L_C * C_flat).sum()


# =====================================================================
# 4. Low-rank projection model
# =====================================================================
@dataclass
class LowRankConfig:
    mode: str  # one of LOWRANK_MODES
    Q: int
    Rv: int
    smoothness_lambda: float = 0.1
    anchor_kernel_lengthscale: float | None = None
    init_C0_from_pls: bool = True

    def __post_init__(self):
        if self.mode not in LOWRANK_MODES:
            raise ValueError(f"mode must be one of {LOWRANK_MODES}, got {self.mode!r}")


class LowRankProjectionModel(nn.Module):
    """
    Holds ``C_0, [ΔC_j], log_l, log_noise`` for the chosen variant. The projection
    ``W_j ∈ R^{Q × D}`` is built via :meth:`projections`.
    """

    def __init__(
        self,
        cfg: LowRankConfig,
        V: torch.Tensor,
        *,
        T: int,
        D: int,
        W_pls: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if V.shape[0] != D:
            raise ValueError(f"V must have {D} rows; got {V.shape[0]}.")
        if V.shape[1] != cfg.Rv:
            raise ValueError(f"V must have {cfg.Rv} columns; got {V.shape[1]}.")
        device = device or V.device
        self.cfg = cfg
        self.register_buffer("V", V.to(device=device).contiguous())
        self.T = int(T)
        self.D = int(D)

        # log GP hyperparams for the local-NLL surrogate.
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(0.1), device=device))

        if cfg.mode == "pls_only":
            if W_pls is None:
                raise ValueError("pls_only requires W_pls.")
            # Frozen W_pls broadcast across anchors.
            W_b = W_pls.to(device=device).contiguous()
            self.register_buffer("W_pls", W_b)
            self.C0 = None
            self.dC = None
        else:
            if cfg.init_C0_from_pls and W_pls is not None:
                # Least-squares init: C_0 = W_pls @ V (V has orthonormal columns).
                C0_init = W_pls.to(device=device) @ V.to(device=device)
            else:
                C0_init = torch.zeros(cfg.Q, cfg.Rv, device=device)
            self.C0 = nn.Parameter(C0_init.contiguous())
            if cfg.mode == "local_C_smooth":
                self.dC = nn.Parameter(torch.zeros(self.T, cfg.Q, cfg.Rv, device=device))
            else:
                self.dC = None

    # -----------------------------------------------------------------
    def projections(self) -> torch.Tensor:
        """``W: [T, Q, D]`` per-anchor projection."""
        if self.cfg.mode == "pls_only":
            return self.W_pls.unsqueeze(0).expand(self.T, -1, -1).contiguous()
        if self.cfg.mode == "global_C":
            W_q = self.C0 @ self.V.T  # [Q, D]
            return W_q.unsqueeze(0).expand(self.T, -1, -1).contiguous()
        # local_C_smooth
        C = self.C0.unsqueeze(0) + self.dC  # [T, Q, Rv]
        return C @ self.V.T  # [T, Q, D]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# 5. Training loop
# =====================================================================
@dataclass
class LowRankTrainingResult:
    W: torch.Tensor  # [T, Q, D] final projection
    history: list[dict[str, float]]
    snapshots: list[dict[str, float]]
    train_time_sec: float
    log_lengthscale: float
    log_noise_var: float
    smoothness_lambda: float
    num_trainable_params: int


def _snapshot(model: LowRankProjectionModel, step: int, *, K_anchor: torch.Tensor | None) -> dict[str, float]:
    with torch.no_grad():
        W = model.projections()  # [T, Q, D]
        WWT = torch.einsum("tqd,trd->tqr", W, W)  # [T, Q, Q]
        eigvals = torch.linalg.eigvalsh(0.5 * (WWT + WWT.transpose(-2, -1)))
        eigvals = torch.clamp(eigvals, min=0.0)
        tr = eigvals.sum(dim=-1)
        sum_sq = (eigvals ** 2).sum(dim=-1).clamp_min(1e-18)
        eff = (eigvals.sum(dim=-1) ** 2) / sum_sq
        row_sq = (W * W).sum(dim=-1)  # [T, Q]
        snap = {
            "step": float(step),
            "median_trace_WWT": float(tr.median().item()),
            "median_eff_rank_WWT": float(eff.median().item()),
            "median_row_norm_sq_W": float(row_sq.median().item()),
            "median_frob_W": float(torch.linalg.norm(W, dim=(-2, -1)).median().item()),
        }
        if model.cfg.mode == "local_C_smooth":
            dC = model.dC
            snap["median_frob_dC"] = float(torch.linalg.norm(dC, dim=(-2, -1)).median().item())
            snap["median_frob_C0"] = float(torch.linalg.norm(model.C0).item())
            if K_anchor is not None:
                snap["smoothness_value"] = float(laplacian_smoothness_penalty(dC, K_anchor).item())
    return snap


def train_lowrank_projection(
    model: LowRankProjectionModel,
    *,
    X_neighbors_stack: torch.Tensor,  # [T, n, D]
    y_neighbors_stack: torch.Tensor,  # [T, n]
    X_anchors: torch.Tensor,  # [T, D]
    num_steps: int = 200,
    lr: float = 0.01,
    log_interval: int = 50,
    snapshot_steps: list[int] | None = None,
    verbose: bool = True,
) -> LowRankTrainingResult:
    """
    Optimise ``model.parameters()`` to minimise per-anchor local-RBF-GP NLL summed over
    neighbourhoods, with optional anchor-Laplacian smoothness on ``ΔC_j``.

    For ``mode='pls_only'`` there are no model parameters — we still update
    ``log_lengthscale, log_noise_var`` so the same diagnostic NLL is comparable across
    variants.
    """
    cfg = model.cfg
    device = X_neighbors_stack.device
    if snapshot_steps is None:
        snapshot_steps = [0, 10, 50, num_steps]
    snap_set = sorted({int(s) for s in snapshot_steps if 0 <= int(s) <= num_steps})

    K_anchor: torch.Tensor | None = None
    if cfg.mode == "local_C_smooth" and cfg.smoothness_lambda > 0:
        K_anchor = anchor_rbf_kernel(X_anchors, lengthscale=cfg.anchor_kernel_lengthscale).detach()

    # Always expose at least the GP hyperparams; some variants have no other params.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )

    snapshots: list[dict[str, float]] = []
    history: list[dict[str, float]] = []

    if 0 in snap_set:
        snapshots.append(_snapshot(model, 0, K_anchor=K_anchor))

    t0 = time.perf_counter()
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        W = model.projections()  # [T, Q, D]
        Z = torch.einsum("tnd,tqd->tnq", X_neighbors_stack, W)
        nll = local_rbf_gp_nll_batch(Z, y_neighbors_stack, model.log_lengthscale, model.log_noise_var)
        loss = nll
        smoothness_val = 0.0
        if cfg.mode == "local_C_smooth" and cfg.smoothness_lambda > 0 and K_anchor is not None:
            smoothness = laplacian_smoothness_penalty(model.dC, K_anchor)
            loss = loss + float(cfg.smoothness_lambda) * smoothness
            smoothness_val = float(smoothness.detach().item())
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == num_steps:
            log = {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "nll": float(nll.detach().item()),
                "smoothness": smoothness_val,
                "log_l": float(model.log_lengthscale.detach().item()),
                "log_sig2": float(model.log_noise_var.detach().item()),
            }
            history.append(log)
            if verbose:
                print(
                    f"[{cfg.mode:<14} step {step:>4d}] loss={log['loss']:.4f} "
                    f"nll={log['nll']:.4f} smooth={smoothness_val:.4f} "
                    f"log_l={log['log_l']:.3f} log_sig2={log['log_sig2']:.3f}"
                )
        if step in snap_set:
            snapshots.append(_snapshot(model, step, K_anchor=K_anchor))
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        W_final = model.projections().detach()
    return LowRankTrainingResult(
        W=W_final,
        history=history,
        snapshots=snapshots,
        train_time_sec=float(elapsed),
        log_lengthscale=float(model.log_lengthscale.detach().item()),
        log_noise_var=float(model.log_noise_var.detach().item()),
        smoothness_lambda=float(cfg.smoothness_lambda),
        num_trainable_params=model.num_trainable_params(),
    )


# =====================================================================
# 6. Diagnostics that compare W variation across anchors
# =====================================================================
def per_anchor_W_diagnostics(W: torch.Tensor, *, W_ref: torch.Tensor | None = None) -> dict[str, float]:
    """
    Summarise ``W: [T, Q, D]``::

        median_trace_WWT, median_eff_rank_WWT, median_frob_W,
        anchor_dispersion_frob = median_t || W_t - mean_t W_t ||_F,
        cosine_to_ref_median   = median_t ||W_t W_ref^T||_F / (||W_t||_F ||W_ref||_F)
            (only when W_ref provided; checks how close to a fixed reference projection).
    """
    with torch.no_grad():
        WWT = torch.einsum("tqd,trd->tqr", W, W)
        eig = torch.linalg.eigvalsh(0.5 * (WWT + WWT.transpose(-2, -1))).clamp_min(0.0)
        tr = eig.sum(dim=-1)
        sum_sq = (eig ** 2).sum(dim=-1).clamp_min(1e-18)
        eff = (eig.sum(dim=-1) ** 2) / sum_sq
        frob = torch.linalg.norm(W, dim=(-2, -1))
        W_mean = W.mean(dim=0, keepdim=True)
        disp = torch.linalg.norm(W - W_mean, dim=(-2, -1))
        out = {
            "median_trace_WWT": float(tr.median().item()),
            "median_eff_rank_WWT": float(eff.median().item()),
            "median_frob_W": float(frob.median().item()),
            "anchor_dispersion_frob": float(disp.median().item()),
        }
        if W_ref is not None:
            W_ref_b = W_ref.to(device=W.device, dtype=W.dtype)
            cos_vals = []
            ref_norm = float(torch.linalg.norm(W_ref_b).item())
            for t in range(W.shape[0]):
                num = float(torch.linalg.norm(W[t] @ W_ref_b.T).item())
                den = float(frob[t].item()) * ref_norm + 1e-12
                cos_vals.append(num / den)
            out["cosine_to_ref_median"] = float(np.median(cos_vals))
    return out


__all__ = [
    "LOWRANK_MODES",
    "LowRankConfig",
    "LowRankProjectionModel",
    "LowRankTrainingResult",
    "anchor_rbf_kernel",
    "build_V_basis",
    "compute_pls_W0",
    "laplacian_smoothness_penalty",
    "local_rbf_gp_nll_batch",
    "per_anchor_W_diagnostics",
    "train_lowrank_projection",
]
