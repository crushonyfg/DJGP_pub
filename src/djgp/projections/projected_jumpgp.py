"""
Run the vendored ``JumpGP_LD`` (CEM mode) on coordinates obtained from a fixed per-anchor
projection ``L_j``. The first-layer metric is therefore decoupled from JumpGP's own local
hyperparameter learning.

Identical neighborhoods (raw ``X_neighbors``) are used across all variants so that the only
difference is the projection that maps ``X -> Z``.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from shared.utils1 import jumpgp_ld_wrapper


def run_projected_jumpgp(
    L: torch.Tensor,
    X_anchors: torch.Tensor,
    X_neighbors_list: Iterable[torch.Tensor],
    y_neighbors_list: Iterable[torch.Tensor],
    *,
    device: torch.device | None = None,
    progress: bool = True,
    desc: str | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Per-anchor JumpGP on projected coordinates ``Z = X L_j^T``.

    Returns a dict with ``mu`` ``[T]``, ``sigma`` ``[T]`` (predictive standard deviation),
    and ``elapsed_sec`` (wall clock for the JumpGP loop only).
    """
    device = device or torch.device("cpu")
    X_neighbors_list = list(X_neighbors_list)
    y_neighbors_list = list(y_neighbors_list)
    T = len(X_neighbors_list)
    if X_anchors.shape[0] != T or L.shape[0] != T:
        raise ValueError("L, X_anchors, and X_neighbors_list must agree on T.")
    Q = L.shape[1]

    mus = np.empty(T, dtype=np.float64)
    sigmas = np.empty(T, dtype=np.float64)

    iterator = range(T)
    if progress:
        iterator = tqdm(iterator, desc=desc or "projected JGP")

    t0 = time.perf_counter()
    for t in iterator:
        L_t = L[t]
        X_nb = X_neighbors_list[t]
        y_nb = y_neighbors_list[t]
        x_test = X_anchors[t]
        Z_nb = X_nb @ L_t.T
        z_test = x_test @ L_t.T
        if Z_nb.dim() != 2 or Z_nb.shape[-1] != Q:
            raise ValueError(f"Bad Z_nb shape {tuple(Z_nb.shape)} (expected [n,{Q}]).")
        mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
            Z_nb,
            y_nb.view(-1, 1),
            z_test.view(1, -1),
            mode="CEM",
            flag=False,
            device=device,
        )
        mus[t] = float(mu_t.detach().cpu().reshape(-1)[0].item())
        sigmas[t] = float(torch.sqrt(torch.clamp(sig2_t, min=1e-12)).detach().cpu().reshape(-1)[0].item())
    elapsed = time.perf_counter() - t0

    return {"mu": mus, "sigma": sigmas, "elapsed_sec": float(elapsed)}
