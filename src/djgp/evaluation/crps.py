"""Gaussian CRPS (closed form), consistent with JumpGP_test.compute_metrics."""

from __future__ import annotations

import math

import torch


def gaussian_crps_torch(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    CRPS for N(mu, sigma^2) vs observation y (same shape broadcast).
    Returns per-element CRPS.
    """
    y = torch.as_tensor(y, dtype=torch.float32)
    mu = torch.as_tensor(mu, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    sigma = torch.clamp(sigma, min=1e-8)
    z = (y - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def mean_gaussian_crps_torch(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(gaussian_crps_torch(y, mu, sigma))
