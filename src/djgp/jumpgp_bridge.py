"""Torch/Numpy bridge helpers for the vendored JumpGP implementation."""

from __future__ import annotations

import torch

from shared.utils1 import jumpgp_ld_wrapper


def center_linear_gate(w: torch.Tensor, z_anchor: torch.Tensor) -> torch.Tensor:
    """Convert ``w0 + w_slope^T z`` to ``b + w_slope^T (z - z_anchor)``."""
    gate = w.reshape(-1).clone()
    anchor = z_anchor.reshape(-1).to(device=gate.device, dtype=gate.dtype)
    if gate.numel() != anchor.numel() + 1:
        raise ValueError(f"Expected gate dim {anchor.numel() + 1}, got {gate.numel()}.")
    gate[0] = gate[0] + torch.dot(gate[1:], anchor)
    return gate


__all__ = ["center_linear_gate", "jumpgp_ld_wrapper"]
