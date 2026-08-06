"""Gate with optional fixed slope ``v`` (only ``nu_0`` / ``omega0`` learned).

Modes:
- ``learned``: full ``omega`` in ``[Q+1]`` (legacy).
- ``fixed_ones``: ``v = 1/Q``.
- ``fixed_e1``: ``v = e_1``.
- ``fixed_e1_temp``: ``v = e_1`` with learnable ``tau_g = softplus(log_tau_gate)``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from djgp.variational import expected_log_sigmoid_gh_batch, expected_sigmoid_gh

_GH_POINTS = 20
_nodes: torch.Tensor | None = None
_weights: torch.Tensor | None = None
_factor: float = 1.0 / math.sqrt(math.pi)


def _gh(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    global _nodes, _weights
    if _nodes is None or _nodes.device != device:
        import numpy as np

        nodes_np, weights_np = np.polynomial.hermite.hermgauss(_GH_POINTS)
        _nodes = torch.from_numpy(nodes_np).to(device=device, dtype=dtype)
        _weights = torch.from_numpy(weights_np).to(device=device, dtype=dtype)
    return _nodes, _weights


def fixed_slope_vector(mode: str, Q: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``v`` with shape ``[Q]``."""
    if mode in ("fixed_ones",):
        return torch.full((Q,), 1.0 / Q, device=device, dtype=dtype)
    if mode in ("fixed_e1", "fixed_e1_temp"):
        v = torch.zeros(Q, device=device, dtype=dtype)
        v[0] = 1.0
        return v
    raise ValueError(f"No fixed slope for gate mode {mode!r}")


def gate_mode_uses_full_omega(mode: str) -> bool:
    return mode == "learned"


def init_gate_params(
    Q: int,
    device: torch.device,
    gate_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Gate-related tensors for one region."""
    mode = gate_config.get("gate_mode", "learned")
    omega0_init = float(gate_config.get("omega0_init", 1.0))
    if gate_mode_uses_full_omega(mode):
        omega = torch.randn(Q + 1, device=device)
        omega[0] = omega0_init
        return {"omega": omega.requires_grad_(True)}
    out: dict[str, torch.Tensor] = {
        "omega0": torch.tensor(omega0_init, device=device, requires_grad=True),
    }
    if mode == "fixed_e1_temp":
        t0 = float(gate_config.get("gate_temp_init", 1.0))
        out["log_tau_gate"] = torch.tensor(
            math.log(math.expm1(max(t0 - 1e-4, 1e-4))), device=device, requires_grad=True,
        )
    return out


def gate_trainable_params(u: dict[str, torch.Tensor], gate_config: dict[str, Any]) -> list[torch.Tensor]:
    mode = gate_config.get("gate_mode", "learned")
    if gate_mode_uses_full_omega(mode):
        return [u["omega"]]
    params = [u["omega0"]]
    if mode == "fixed_e1_temp" and "log_tau_gate" in u:
        params.append(u["log_tau_gate"])
    return params


def gate_temperature(u_params, gate_config: dict[str, Any]) -> torch.Tensor | float:
    """Per-region ``tau_g``; shape ``[T]`` or scalar ``1.0``."""
    mode = gate_config.get("gate_mode", "learned")
    if mode == "fixed_e1_temp":
        tau = F.softplus(torch.stack([u["log_tau_gate"] for u in u_params], dim=0)) + 1e-6
        return tau
    if "gate_temp" in gate_config and gate_config["gate_temp"] is not None:
        return float(gate_config["gate_temp"])
    return 1.0


def stack_omega0(u_params) -> torch.Tensor:
    return torch.stack([u["omega0"] for u in u_params], dim=0)


def stack_omega_full(u_params) -> torch.Tensor:
    return torch.stack([u["omega"] for u in u_params], dim=0)


def _proj_variance_s(X: torch.Tensor, cov_W: torch.Tensor) -> torch.Tensor:
    """``s[t,n,q] = x^T Sigma_{t,q} x``; shape ``[T,n,Q]``."""
    if cov_W.dim() == 3:
        return torch.einsum("tnd,tqd,tne->tnq", X, cov_W, X)
    return torch.einsum("tnd,tqde,tne->tnq", X, cov_W, X)


def gate_xi_moments(
    omega0: torch.Tensor,
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    X: torch.Tensor,
    v: torch.Tensor,
    gate_temp: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mu_z``, ``tau_z`` for ``xi = omega0 + tau * sum_q v_q * mu_{W,q}^T x``."""
    mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
    if isinstance(gate_temp, torch.Tensor):
        tau = gate_temp.view(-1, 1)
    else:
        tau = float(gate_temp)
    mu_slope = torch.einsum("q,tnq->tn", v, mu_proj)
    mu_z = omega0.unsqueeze(1) + tau * mu_slope
    s = _proj_variance_s(X, cov_W)
    tau2 = (tau ** 2) * torch.einsum("q,tnq->tn", v.pow(2), s)
    tau_z = torch.sqrt(tau2 + 1e-12)
    return mu_z, tau_z


def expected_log_sigmoid_gh_from_xi(mu_z: torch.Tensor, tau_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GH expectation of ``log sigma(xi)`` and ``log(1-sigma(xi))``."""
    device, dtype = mu_z.device, mu_z.dtype
    nodes, weights = _gh(device, dtype)
    z = mu_z.unsqueeze(2) + math.sqrt(2.0) * tau_z.unsqueeze(2) * nodes.view(1, 1, -1)
    log_sig = F.logsigmoid(z)
    log_one_minus = F.logsigmoid(-z)
    e1 = _factor * (weights.view(1, 1, -1) * log_sig).sum(dim=-1)
    e2 = _factor * (weights.view(1, 1, -1) * log_one_minus).sum(dim=-1)
    return e1, e2


def expected_sigmoid_gh_from_xi(mu_z: torch.Tensor, tau_z: torch.Tensor) -> torch.Tensor:
    device, dtype = mu_z.device, mu_z.dtype
    nodes, weights = _gh(device, dtype)
    z = mu_z.unsqueeze(2) + math.sqrt(2.0) * tau_z.unsqueeze(2) * nodes.view(1, 1, -1)
    sig = torch.sigmoid(z)
    return _factor * (weights.view(1, 1, -1) * sig).sum(dim=-1)


def expected_log_sigmoid_gate(
    u_params,
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    X: torch.Tensor,
    gate_config: dict[str, Any],
    Q: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = gate_config.get("gate_mode", "learned")
    if gate_mode_uses_full_omega(mode):
        omega = stack_omega_full(u_params)
        return expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)

    device, dtype = mu_W.device, mu_W.dtype
    omega0 = stack_omega0(u_params)
    v = fixed_slope_vector(mode, Q, device, dtype)
    tau = gate_temperature(u_params, gate_config)
    if mode == "fixed_e1_temp":
        mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
        if isinstance(tau, torch.Tensor):
            t = tau.view(-1, 1)
        else:
            t = float(tau)
        mu_z = omega0.unsqueeze(1) + t * mu_proj[..., 0]
        s = _proj_variance_s(X, cov_W)
        tau_z = torch.sqrt((t ** 2) * s[..., 0] + 1e-12)
    else:
        mu_z, tau_z = gate_xi_moments(omega0, mu_W, cov_W, X, v, tau)
    return expected_log_sigmoid_gh_from_xi(mu_z, tau_z)


def expected_sigmoid_gate(
    u_params,
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    X: torch.Tensor,
    gate_config: dict[str, Any],
    Q: int,
) -> torch.Tensor:
    mode = gate_config.get("gate_mode", "learned")
    if gate_mode_uses_full_omega(mode):
        omega = stack_omega_full(u_params)
        return expected_sigmoid_gh(omega, mu_W, cov_W, X)

    device, dtype = mu_W.device, mu_W.dtype
    omega0 = stack_omega0(u_params)
    v = fixed_slope_vector(mode, Q, device, dtype)
    tau = gate_temperature(u_params, gate_config)
    if mode == "fixed_e1_temp":
        mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
        if isinstance(tau, torch.Tensor):
            t = tau.view(-1, 1)
        else:
            t = float(tau)
        mu_z = omega0.unsqueeze(1) + t * mu_proj[..., 0]
        s = _proj_variance_s(X, cov_W)
        tau_z = torch.sqrt((t ** 2) * s[..., 0] + 1e-12)
    else:
        mu_z, tau_z = gate_xi_moments(omega0, mu_W, cov_W, X, v, tau)
    return expected_sigmoid_gh_from_xi(mu_z, tau_z)


@torch.no_grad()
def gate_diagnostics(
    u_params,
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    X: torch.Tensor,
    X_test: torch.Tensor,
    gate_config: dict[str, Any],
    Q: int,
) -> dict[str, float]:
    """``mu_xi``, ``pi``, ``omega0`` stats for logging."""
    mode = gate_config.get("gate_mode", "learned")
    pi = expected_sigmoid_gate(u_params, mu_W, cov_W, X, gate_config, Q)
    pi_test = expected_sigmoid_gate(
        u_params, mu_W, cov_W, X_test.unsqueeze(1), gate_config, Q,
    ).squeeze(1)

    if gate_mode_uses_full_omega(mode):
        omega = stack_omega_full(u_params)
        mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
        mu_z = omega[:, 0].unsqueeze(1) + (omega[:, 1:].unsqueeze(1) * mu_proj).sum(dim=2)
        s = _proj_variance_s(X, cov_W)
        tau2 = (omega[:, 1:].unsqueeze(1).pow(2) * s).sum(dim=2)
        omega0_mean = float(omega[:, 0].mean().item())
        slope_norm_mean = float(omega[:, 1:].norm(dim=-1).mean().item())
    else:
        omega0 = stack_omega0(u_params)
        v = fixed_slope_vector(mode, Q, mu_W.device, mu_W.dtype)
        tau = gate_temperature(u_params, gate_config)
        mu_z, tau_z = gate_xi_moments(omega0, mu_W, cov_W, X, v, tau)
        if mode == "fixed_e1_temp":
            mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
            if isinstance(tau, torch.Tensor):
                t = tau.view(-1, 1)
            else:
                t = float(tau)
            mu_z = omega0.unsqueeze(1) + t * mu_proj[..., 0]
            s = _proj_variance_s(X, cov_W)
            tau2 = (t ** 2) * s[..., 0]
        else:
            tau2 = tau_z.pow(2)
        omega0_mean = float(omega0.mean().item())
        slope_norm_mean = float(v.norm().item())

    w_norm = float(mu_W.norm(dim=-1).mean().item())
    out = {
        "mu_xi_mean": float(mu_z.mean().item()),
        "mu_xi_std": float(mu_z.std().item()),
        "tau2_xi_mean": float(tau2.mean().item()),
        "pi_gate_mean": float(pi.mean().item()),
        "pi_gate_std": float(pi.std().item()),
        "pi_gate_test_mean": float(pi_test.mean().item()),
        "omega0_mean": omega0_mean,
        "gate_slope_norm": slope_norm_mean,
        "mu_W_norm_mean": w_norm,
    }
    if mode == "fixed_e1_temp":
        tau = gate_temperature(u_params, gate_config)
        if isinstance(tau, torch.Tensor):
            out["tau_gate_mean"] = float(tau.mean().item())
    return out
