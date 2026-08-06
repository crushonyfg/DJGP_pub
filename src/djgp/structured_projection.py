"""W0-centered low-rank residual projection for structured VI (Remedy D).

[
W_j = W_{0,j} + B_j U^\\top, \\quad B_j \\in \\mathbb R^{Q\\times r}, \\; U \\in \\mathbb R^{D\\times r}
]

Variational coefficients live on inducing points: ``mu_B, sigma_B`` with shape ``[m2, Q, r]``.
GP conditioning propagates each scalar field ``B_{q,r}(\\cdot)`` from ``Z`` to target ``X``.
KL is centered at zero on ``B`` (penalises deviation from ``W_0``, not ``W_0`` itself).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from djgp.variational import expected_Kfu, expected_KufKfu, kl_qp, qW_from_qV

_JITTER = 1e-5


def kl_qB(
    Z: torch.Tensor,
    mu_B: torch.Tensor,
    sigma_B: torch.Tensor,
    lengthscales: torch.Tensor,
    var_w: torch.Tensor,
) -> torch.Tensor:
    """KL(q(B)||p(B)) with GP prior on inducing coefficients (same as ``kl_qp``)."""
    return kl_qp(Z, mu_B, sigma_B, lengthscales, var_w)


def _gp_scalar_cond(
    X: torch.Tensor,
    Z: torch.Tensor,
    mu_z: torch.Tensor,
    sigma_z: torch.Tensor,
    lengthscale: float | torch.Tensor,
    var_w: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition scalar GP field from inducing ``(mu_z, sigma_z)`` at ``Z`` to points ``X``."""
    m2 = Z.shape[0]
    T = X.shape[0]
    device = X.device
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    XZ2 = (X.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    ell = float(lengthscale) if not torch.is_tensor(lengthscale) else lengthscale
    vw = float(var_w) if not torch.is_tensor(var_w) else var_w

    Kzz = vw * torch.exp(-0.5 * ZZ2 / (ell**2))
    Kxz = vw * torch.exp(-0.5 * XZ2 / (ell**2))
    eye = torch.eye(m2, device=device, dtype=Kzz.dtype)
    Kzz = 0.5 * (Kzz + Kzz.T) + _JITTER * eye
    L = torch.linalg.cholesky(Kzz)
    Kzz_inv = torch.cholesky_inverse(L)
    A = Kxz @ Kzz_inv
    diag_cross = (A * Kxz).sum(dim=1)
    Umat = Kzz_inv @ Kxz.t()
    mu_x = A @ mu_z
    s2 = sigma_z.pow(2).unsqueeze(1)
    diag3 = (Umat.pow(2) * s2).sum(dim=0)
    sigma_x = torch.sqrt((vw - diag_cross + diag3).clamp_min(1e-12))
    return mu_x, sigma_x


def qW_lowrank_residual(
    X: torch.Tensor,
    Z: torch.Tensor,
    W0: torch.Tensor,
    mu_B: torch.Tensor,
    sigma_B: torch.Tensor,
    U: torch.Tensor,
    lengthscales: torch.Tensor,
    var_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mu_W, cov_W`` with ``W = W0 + B U^T`` and low-rank GP posterior on ``B``.

    ``X``: ``[T, D]`` or ``[T, n, D]`` (neighbours flattened per region).
    ``W0``: ``[Q, D]`` shared warm start.
    ``mu_B, sigma_B``: ``[m2, Q, r]``.
    ``U``: ``[D, r]``.
    """
    orig_3d = X.ndim == 3
    if orig_3d:
        T, n, D = X.shape
        X2 = X.reshape(T * n, D)
    else:
        X2 = X
        T = X2.shape[0]
    device = X2.device
    Q, D0 = W0.shape
    r = U.shape[1]
    assert U.shape[0] == D0

    mu_W = W0.unsqueeze(0).expand(T if not orig_3d else T, Q, D0).clone()
    if orig_3d:
        mu_W = mu_W.unsqueeze(1).expand(T, n, Q, D0).reshape(T * n, Q, D0)
    var_W = torch.zeros_like(mu_W)

    for q in range(Q):
        ell = lengthscales[q]
        vw = var_w[q] if var_w.ndim > 0 else var_w
        for ri in range(r):
            mu_x, sig_x = _gp_scalar_cond(
                X2, Z, mu_B[:, q, ri], sigma_B[:, q, ri], ell, vw,
            )
            mu_W[:, q, :] = mu_W[:, q, :] + mu_x.unsqueeze(1) * U[:, ri].unsqueeze(0)
            var_W[:, q, :] = var_W[:, q, :] + (sig_x.pow(2)).unsqueeze(1) * U[:, ri].pow(2).unsqueeze(0)

    sigma_W = torch.sqrt(var_W.clamp_min(1e-12))
    cov_W = torch.diag_embed(sigma_W.pow(2))
    if orig_3d:
        mu_W = mu_W.reshape(T, n, Q, D0)
        cov_W = cov_W.reshape(T, n, Q, D0, D0)
    return mu_W, cov_W


def qW_full_centered(
    X: torch.Tensor,
    Z: torch.Tensor,
    W0: torch.Tensor,
    mu_V: torch.Tensor,
    sigma_V: torch.Tensor,
    lengthscales: torch.Tensor,
    var_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``W = W0 + Delta_W`` with full ``Delta_W`` from standard ``qW_from_qV``."""
    mu_W, cov_W = qW_from_qV(X, Z, mu_V, sigma_V, lengthscales, var_w)
    return mu_W + W0.unsqueeze(0), cov_W


def get_mu_W_cov_W(
    proj: dict[str, Any],
    X: torch.Tensor,
    hyperparams: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch projection parameter dict to ``(mu_W, cov_W)``."""
    Z = hyperparams["Z"]
    lengthscales = hyperparams["lengthscales"]
    var_w = hyperparams["var_w"]
    kind = proj["kind"]
    W0 = proj["W0"]
    if kind == "full":
        return qW_from_qV(
            X, Z, proj["mu_V"], proj["sigma_V"], lengthscales, var_w,
        )
    if kind == "full_centered":
        return qW_full_centered(
            X, Z, W0, proj["mu_V"], proj["sigma_V"], lengthscales, var_w,
        )
    if kind == "lowrank":
        return qW_lowrank_residual(
            X, Z, W0, proj["mu_B"], proj["sigma_B"], proj["U"],
            lengthscales, var_w,
        )
    raise ValueError(f"Unknown projection kind {kind!r}")


def s_penalty_log_lowrank(
    X: torch.Tensor,
    U: torch.Tensor,
    sigma_B_at_X: torch.Tensor,
) -> torch.Tensor:
    """Mean over regions/neighbours/q of ``log(1 + s)`` with ``s = (U^T x)^T diag(sigma_B^2) (U^T x)``."""
    # X [T,n,D], sigma_B_at_X [T,n,Q,r]
    UtX = torch.einsum("dr,tnd->tnr", U, X)
    s = (UtX.pow(2) * sigma_B_at_X.pow(2)).sum(dim=-1)
    return torch.log1p(s).mean()


def propagate_sigma_B_to_X(
    X: torch.Tensor,
    Z: torch.Tensor,
    sigma_B: torch.Tensor,
    lengthscales: torch.Tensor,
    var_w: torch.Tensor,
) -> torch.Tensor:
    """``sigma_B`` at neighbour points, shape ``[T,n,Q,r]`` (or ``[T,Q,r]`` if ``X`` is 2d)."""
    if X.ndim == 2:
        T = X.shape[0]
        Q, r = sigma_B.shape[1], sigma_B.shape[2]
        out = torch.empty(T, Q, r, device=X.device, dtype=X.dtype)
        for q in range(Q):
            ell = lengthscales[q]
            vw = var_w[q] if var_w.ndim > 0 else var_w
            for ri in range(r):
                _, sig = _gp_scalar_cond(
                    X, Z, torch.zeros(m2 := Z.shape[0], device=X.device),
                    sigma_B[:, q, ri], ell, vw,
                )
                out[:, q, ri] = sig
        return out
    T, n, _ = X.shape
    Q, r = sigma_B.shape[1], sigma_B.shape[2]
    out = torch.empty(T, n, Q, r, device=X.device, dtype=X.dtype)
    Xf = X.reshape(T * n, -1)
    flat = propagate_sigma_B_to_X(
        Xf, Z, sigma_B, lengthscales, var_w,
    )
    return flat.reshape(T, n, Q, r)


def compute_s_penalty_proj(
    regions,
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
    *,
    use_log1p: bool = True,
) -> torch.Tensor:
    """``log(1+s)`` (or mean ``s``) penalty at training neighbours."""
    X = torch.stack([r["X"] for r in regions], dim=0)
    T, n, D = X.shape
    Xf = X.reshape(T * n, D)
    _, cov_W = get_mu_W_cov_W(proj, Xf, hyperparams)
    var_W = torch.diagonal(cov_W, dim1=-2, dim2=-1).reshape(T, n, -1, D)
    s = (X.pow(2).unsqueeze(2) * var_W).sum(dim=-1)
    if use_log1p:
        return torch.log1p(s).mean()
    return s.mean()


def deterministic_Kfu(
    W: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    sigma_f: torch.Tensor,
) -> torch.Tensor:
    """RBF kernel without ``E_q[W]`` — ``W`` shape ``[T,Q,D]``, ``X`` ``[T,n,D]``."""
    mu_proj = torch.einsum("tqd,tnd->tnq", W, X)
    diff = mu_proj.unsqueeze(2) - C.unsqueeze(1)
    exp_term = torch.exp(-0.5 * diff.pow(2))
    num = exp_term.prod(dim=-1)
    return sigma_f.view(-1, 1, 1) * num


def expected_Kfu_mc(
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    sigma_f: torch.Tensor,
    *,
    n_samples: int = 5,
) -> torch.Tensor:
    """MC estimate of ``E_q[K_{fu}]`` by sampling diagonal ``W``."""
    T, Q, D = mu_W.shape
    n = X.shape[1]
    device = mu_W.device
    std = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3).clamp_min(1e-12))
    acc = torch.zeros(T, n, C.shape[1], device=device, dtype=mu_W.dtype)
    for _ in range(n_samples):
        eps = torch.randn(T, Q, D, device=device, dtype=mu_W.dtype)
        W_s = mu_W + eps * std
        acc = acc + deterministic_Kfu(W_s, X, C, sigma_f)
    return acc / float(n_samples)


@torch.no_grad()
def psi1_diagnostics(
    mu_W: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    sigma_k: torch.Tensor,
) -> dict[str, float]:
    """Row-std and effective rank of ``Psi1`` (deterministic kernel at ``mu_W``)."""
    Kfu = deterministic_Kfu(mu_W, X, C, sigma_k)
    row_std = float(Kfu.std(dim=1).mean().item())
    mat = Kfu.reshape(-1, Kfu.shape[-1])
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        effrank = 1.0
    else:
        s = torch.linalg.svdvals(mat)
        s = s / s.sum().clamp_min(1e-12)
        effrank = float(torch.exp(-(s * torch.log(s.clamp_min(1e-12))).sum()).item())
    return {"psi1_row_std": row_std, "psi1_effrank": effrank}


def init_inducing_from_w0x(
    regions: list[dict[str, torch.Tensor]],
    W0: torch.Tensor,
    *,
    seed: int = 0,
) -> None:
    """Set ``C`` to subset of ``W0 @ x`` in each neighbourhood (latent coords)."""
    g = torch.Generator()
    g.manual_seed(seed)
    for r in regions:
        X = r["X"]
        n, D = X.shape
        m1, Q = r["C"].shape
        proj = X @ W0.T
        if n >= m1:
            idx = torch.randperm(n, generator=g)[:m1]
        else:
            idx = torch.randint(0, n, (m1,), generator=g)
        r["C"] = proj[idx].detach()


def build_W0_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    Q: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    from experiments.synthetic.compare_uncertain_w_cem_l2_lh import _projection_W0

    return _projection_W0(X_train, y_train, Q, mode, seed=seed)


def build_U_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    W0: np.ndarray,
    r: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    from experiments.synthetic.compare_uncertain_w_cem_l2_lh import _residual_basis_V

    return _residual_basis_V(X_train, y_train, W0, r, mode=mode, seed=seed)


def build_proj_params(
    *,
    kind: str,
    device: torch.device,
    m2: int,
    Q: int,
    D: int,
    r: int,
    W0_np: np.ndarray,
    U_np: np.ndarray | None = None,
    sigma_init: float = 0.1,
    sigma_max: float | None = None,
) -> dict[str, Any]:
    W0 = torch.from_numpy(W0_np.astype(np.float32)).to(device)
    out: dict[str, Any] = {
        "kind": kind,
        "W0": W0,
        "sigma_max": sigma_max,
    }
    if kind in ("full", "full_centered"):
        out["mu_V"] = torch.randn(m2, Q, D, device=device, requires_grad=True) * 0.01
        sig = torch.full((m2, Q, D), float(sigma_init), device=device)
        if sigma_max is not None:
            sig = sig.clamp(max=float(sigma_max))
        out["sigma_V"] = sig.requires_grad_(True)
    elif kind == "lowrank":
        if U_np is None:
            raise ValueError("lowrank requires U_np")
        r = min(r, U_np.shape[1])
        U = torch.from_numpy(U_np[:, :r].astype(np.float32)).to(device)
        out["U"] = U
        out["mu_B"] = torch.randn(m2, Q, r, device=device, requires_grad=True) * 0.01
        sig = torch.full((m2, Q, r), float(sigma_init), device=device)
        if sigma_max is not None:
            sig = sig.clamp(max=float(sigma_max))
        out["sigma_B"] = sig.requires_grad_(True)
    else:
        raise ValueError(kind)
    return out


def proj_trainable_tensors(proj: dict[str, Any]) -> list[torch.Tensor]:
    kind = proj["kind"]
    if kind in ("full", "full_centered"):
        return [proj["mu_V"], proj["sigma_V"]]
    if kind == "lowrank":
        return [proj["mu_B"], proj["sigma_B"]]
    raise ValueError(kind)


def kl_projection(proj: dict[str, Any], Z: torch.Tensor, lengthscales, var_w) -> torch.Tensor:
    kind = proj["kind"]
    if kind in ("full", "full_centered"):
        return kl_qp(Z, proj["mu_V"], proj["sigma_V"], lengthscales, var_w)
    if kind == "lowrank":
        return kl_qB(Z, proj["mu_B"], proj["sigma_B"], lengthscales, var_w)
    raise ValueError(kind)


def clamp_projection(proj: dict[str, Any], sigma_max: float | None) -> None:
    if sigma_max is None:
        return
    smax = float(sigma_max)
    with torch.no_grad():
        if proj["kind"] in ("full", "full_centered"):
            proj["sigma_V"].clamp_(min=1e-3, max=smax)
        elif proj["kind"] == "lowrank":
            proj["sigma_B"].clamp_(min=1e-3, max=smax)


@torch.no_grad()
def sample_W_from_proj(
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
    *,
    M: int,
    seed: int = 0,
) -> torch.Tensor:
    """Sample ``W`` at test anchors from diagonal ``q(W)`` — shape ``[M, T, Q, D]``."""
    X = hyperparams["X_test"]
    mu_W, cov_W = get_mu_W_cov_W(proj, X, hyperparams)
    std = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3).clamp_min(1e-12))
    gen = torch.Generator(device=mu_W.device)
    gen.manual_seed(int(seed))
    eps = torch.randn(
        (M,) + tuple(mu_W.shape), device=mu_W.device, dtype=mu_W.dtype, generator=gen,
    )
    return mu_W.unsqueeze(0) + eps * std.unsqueeze(0)


@torch.no_grad()
def projection_w_drift_metrics(
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
) -> dict[str, float]:
    """Drift of ``E_q[W]`` at test anchors vs fixed ``W0`` and metric ``W^T W``."""
    X = hyperparams["X_test"]
    mu_W, _ = get_mu_W_cov_W(proj, X, hyperparams)
    W0 = proj["W0"]
    delta = mu_W - W0.unsqueeze(0)
    out: dict[str, float] = {
        "W_drift_frob_mean": float(torch.linalg.norm(delta, dim=(-2, -1)).mean().item()),
        "W_row_norm_mean": float(torch.linalg.norm(mu_W, dim=-1).mean().item()),
        "W0_row_norm_mean": float(torch.linalg.norm(W0, dim=-1).mean().item()),
    }
    G = torch.einsum("tqd,tqe->tde", mu_W, mu_W)
    G0 = W0.transpose(0, 1) @ W0
    out["W_metric_drift_frob_mean"] = float(
        torch.linalg.norm(G - G0.unsqueeze(0), dim=(-2, -1)).mean().item(),
    )
    evals = torch.linalg.eigvalsh(G)
    out["W_metric_top_eig_mean"] = float(evals[..., -1].mean().item())
    out["W0_metric_top_eig"] = float(torch.linalg.eigvalsh(G0.unsqueeze(0))[0, -1].item())
    if proj["kind"] == "lowrank" and "mu_B" in proj:
        mu_B = proj["mu_B"]
        mu_B0 = hyperparams.get("mu_B_init")
        if mu_B0 is not None:
            out["B_drift_frob"] = float(torch.linalg.norm(mu_B - mu_B0).item())
    return out
