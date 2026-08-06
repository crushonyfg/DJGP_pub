"""
LMJGP training-time tricks ablation on **synthetic data with known low-rank Mahalanobis
structure**. Mirrors :mod:`experiments.uci.compare_lmjgp_tricks` but generates the data
in-process so:

- We know the ground-truth subspace ``W_true ∈ R^{Q_true x D}`` and can measure
  ``capture ratio = sum_top_Q(eigvals(A_j)) / tr(A_j)``,
  ``subspace alignment = ||W_true L_j^T (L_j L_j^T)^-1 L_j W_true^T||_F / Q_true``,
  and the angle between learned ``A_j`` and ``A_true = W_true^T W_true``.
- We can compare ``learned vs isotropic vs random`` projection variants on the *same*
  data + neighbourhoods so KL-warm-up / PLS init / row-norm / aux-metric tricks can be
  directly evaluated against a baseline that *should* learn a low-dim metric if the
  variational layer is doing its job.

Generator (``"lowrank_jump"``):

    Sample W_true ∈ R^{Q_true x D} as a row-orthonormal random matrix.
    Sample X ∈ R^{N x D} from N(0, I_D).
    Compute Z = X W_true^T ∈ R^{N x Q_true}.
    Smooth part f(Z) = sin(2 Z[:,0]) + 0.5 Z[:,1]^2 + 0.3 sum(Z[:,2:]) (if Q_true >= 3).
    Jump   g(Z) = jump_amp * (Z[:,0] > 0).
    y = f(Z) + g(Z) + noise_std * N(0, 1).

This deliberately puts the response on the first two latent dims with a sharp jump on
the first, which is what LMJGP's local model is designed to recover.

How to run (repository root, conda env: ``jumpGP``):

  conda activate jumpGP

  # Smoke test
  python experiments/synthetic/compare_lmjgp_tricks_synth.py \\
      --num_exp 1 --num_steps 200 --max_anchors 32 \\
      --variants baseline,kl_warmup,pls_init,pls_kl

    python experiments/synthetic/compare_lmjgp_tricks_synth.py ^
      --num_exp 1 --num_steps 500 --max_anchors 64 ^
      --variants baseline,kl_warmup,pls_init,pls_kl

  # Full ablation
  python experiments/synthetic/compare_lmjgp_tricks_synth.py \\
      --num_exp 3 --N_train 2000 --N_test 200 --D 12 --Q_true 2 --Q 2 \\
      --variants baseline,kl_warmup,pls_init,sir_init,row_norm,aux_metric,pls_kl,combo \\
      --with_projection_ablation

Agents: update this docstring when CLI defaults / variant names change.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import compute_metrics, find_neighborhoods  # noqa: E402
from djgp.diagnostics.projection_posterior import (  # noqa: E402
    metric_A_from_moment,
    sanitize_for_json,
    sym_matrix_stats,
)
from djgp.evaluation.calibration import filtered_gaussian_uq_metrics_dict, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    PROJECTION_MODES,
    build_projection_per_anchor,
    projection_quality_diagnostics,
    run_projected_jumpgp,
)
from djgp.variational import predict_vi, qW_from_qV  # noqa: E402
from djgp.variational_tricks import (  # noqa: E402
    KLWarmupSchedule,
    calibrate_mu_V_to_target_mu_W,
    init_lengthscales_median_heuristic,
    init_mu_V_supervised,
    train_vi_with_tricks,
)

VARIANT_NAMES = (
    "baseline",
    "kl_warmup",
    "pls_init",
    "sir_init",
    "row_norm",
    "aux_metric",
    "pls_kl",
    "combo",
    "combo_full",
)


# =====================================================================
# Synthetic data generator
# =====================================================================
@dataclass
class SyntheticDataset:
    X_train: np.ndarray  # [N_train, D]
    y_train: np.ndarray  # [N_train]
    X_test: np.ndarray
    y_test: np.ndarray
    W_true: np.ndarray  # [Q_true, D]
    info: dict[str, Any]


def generate_lowrank_jump(
    *,
    N_train: int,
    N_test: int,
    D: int,
    Q_true: int,
    jump_amp: float = 1.5,
    noise_std: float = 0.1,
    seed: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    # Row-orthonormal Q_true x D — top-Q_true principal directions of A_true.
    M = rng.standard_normal((D, Q_true))
    Q_, _ = np.linalg.qr(M)  # D x Q_true, columns orthonormal
    W_true = Q_.T  # Q_true x D, rows orthonormal

    def make_one(N: int) -> tuple[np.ndarray, np.ndarray]:
        X = rng.standard_normal((N, D))
        Z = X @ W_true.T  # N x Q_true
        f_smooth = np.sin(2.0 * Z[:, 0]) + 0.5 * (Z[:, 1] ** 2 if Q_true >= 2 else 0.0)
        if Q_true >= 3:
            f_smooth = f_smooth + 0.3 * Z[:, 2:].sum(axis=1)
        jump = jump_amp * (Z[:, 0] > 0.0).astype(np.float64)
        noise = noise_std * rng.standard_normal(N)
        y = f_smooth + jump + noise
        return X.astype(np.float32), y.astype(np.float32)

    X_tr, y_tr = make_one(N_train)
    X_te, y_te = make_one(N_test)
    info = {
        "name": "lowrank_jump",
        "D": D,
        "Q_true": Q_true,
        "jump_amp": jump_amp,
        "noise_std": noise_std,
        "seed": seed,
        "W_true_frob": float(np.linalg.norm(W_true)),
        "W_true_trace_WtW": float((W_true.T @ W_true).trace()),
    }
    return SyntheticDataset(X_tr, y_tr, X_te, y_te, W_true, info)


# =====================================================================
# Subspace-alignment diagnostics (uses W_true)
# =====================================================================
def subspace_alignment_to_truth(
    A: torch.Tensor,
    W_true: torch.Tensor,
    Q_truth: int,
) -> dict[str, float]:
    """
    For each anchor compute alignment of the learned top-``Q_truth`` eigenspace of
    ``A_j`` with the row-span of ``W_true``.

    Returns medians of:
        ``capture_ratio_truth``  : ``sum_{i<=Q_truth} eigvals(A_j) / tr(A_j)``,
        ``subspace_principal_angles_mean_cos`` : mean cosine of principal angles
            between top-``Q_truth`` eigenspace of ``A_j`` and row-span of ``W_true``
            (1.0 = perfect alignment, 0.0 = orthogonal).
    """
    A_sym = 0.5 * (A + A.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(A_sym)
    eigvals = torch.clamp(eigvals, min=0.0)
    top_vals = eigvals[..., -Q_truth:]
    top_vecs = eigvecs[..., :, -Q_truth:]  # [T, D, Q_truth]
    tr = eigvals.sum(dim=-1).clamp_min(1e-12)
    capture = top_vals.sum(dim=-1) / tr  # [T]

    # Orthonormal basis for W_true row-span
    Wt = W_true.to(dtype=A.dtype, device=A.device)  # [Q_true, D]
    Q_basis, _ = torch.linalg.qr(Wt.T, mode="reduced")  # [D, Q_true]

    cos_means = []
    for t in range(A.shape[0]):
        U = top_vecs[t]  # [D, Q_truth]
        M = U.T @ Q_basis  # [Q_truth, Q_true]
        s = torch.linalg.svdvals(M).clamp(min=0.0, max=1.0)
        cos_means.append(float(s.mean().item()))
    return {
        "median_capture_ratio_top_Q_truth": float(np.nanmedian(capture.detach().cpu().numpy())),
        "median_subspace_cos_to_truth": float(np.nanmedian(cos_means)),
    }


# =====================================================================
# Plumbing reused from compare_lmjgp_tricks (kept inline to avoid coupling)
# =====================================================================
def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_regions(X_test, X_train, Y_train, device, m1: int, Q: int, n: int):
    nbr = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions = []
    X_nb_list, y_nb_list = [], []
    T = X_test.shape[0]
    for i in range(T):
        X_nb = nbr[i]["X_neighbors"].to(device)
        y_nb = nbr[i]["y_neighbors"].to(device)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return regions, X_nb_list, y_nb_list


def _make_initial_params(
    X_train, Y_train, X_test, *, Q, m1, m2, T, D, device, mu_V_mode, init_seed,
):
    info: dict[str, Any] = {"mu_V_mode": mu_V_mode}
    W_init_tensor: torch.Tensor | None = None
    if mu_V_mode == "random":
        mu_V_init = torch.randn(m2, Q, D, device=device)
    elif mu_V_mode in ("pls", "sir"):
        mu_V_init, sup_info, W_init_tensor = init_mu_V_supervised(
            X_train, Y_train, m2=m2, Q=Q,
            mode=mu_V_mode, sir_H=10, perturb_std=1e-3, seed=init_seed, device=device,
            return_W=True,
        )
        info["supervised_init"] = sup_info
    else:
        raise ValueError(f"Unknown mu_V_mode={mu_V_mode!r}.")

    V_params = {
        "mu_V": mu_V_init.detach().clone().requires_grad_(True),
        "sigma_V": torch.rand(m2, Q, D, device=device).requires_grad_(True),
    }
    u_params = []
    for _ in range(T):
        u_params.append({
            "U_logit": torch.zeros(1, device=device, requires_grad=True),
            "mu_u": torch.randn(m1, device=device, requires_grad=True),
            "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
            "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
            "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
            "omega": torch.randn(Q + 1, device=device, requires_grad=True),
        })
    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std
    ls_init = init_lengthscales_median_heuristic(X_test, Z, Q=Q)
    hyperparams = {
        "Z": Z.requires_grad_(True),
        "X_test": X_test,
        "lengthscales": ls_init.clone().requires_grad_(True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    info["lengthscale_init_median_heuristic"] = float(ls_init[0].item())
    if W_init_tensor is not None:
        info["calibration"] = calibrate_mu_V_to_target_mu_W(
            V_params, hyperparams, X_test, W_init_tensor,
            per_row=True, max_alpha=5.0, target_row_norm=1.0,
        )
    return V_params, u_params, hyperparams, info


def _post_train_metric_block(V_params, hyperparams, X_test):
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
    A = metric_A_from_moment(mu_W.detach(), cov_W.detach())
    per_anchor_stats = [sym_matrix_stats(A[t]) for t in range(A.shape[0])]
    block = {
        "median_trace_A": float(np.nanmedian([s["trace"] for s in per_anchor_stats])),
        "median_eff_rank_A": float(np.nanmedian([s["eff_rank"] for s in per_anchor_stats])),
        "median_top_eig_A": float(np.nanmedian([s["top_eig"] for s in per_anchor_stats])),
        "median_min_eig_A_pos": float(np.nanmedian([s["min_eig_pos"] for s in per_anchor_stats])),
    }
    EW_row_sq = (mu_W * mu_W).sum(dim=-1)
    block["median_row_norm_sq_mu_W"] = float(EW_row_sq.median().item())
    block["mean_row_norm_sq_mu_W"] = float(EW_row_sq.mean().item())
    block["median_frob_mu_W"] = float(torch.linalg.norm(mu_W, dim=(-2, -1)).median().item())
    return mu_W, cov_W, A, block


def _evaluate_predict_vi(regions, V_params, hyperparams, Y_test, *, MC_num):
    t0 = time.perf_counter()
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=MC_num)
    elapsed = time.perf_counter() - t0
    sigmas = torch.sqrt(torch.clamp(var_pred, min=1e-12))
    rmse, crps = compute_metrics(mu_pred, sigmas, Y_test)
    rmse_v = float(rmse.detach().item())
    crps_v = float(crps.detach().item())
    pack = pack_gaussian_uq_list(
        rmse_v, crps_v, float(elapsed),
        Y_test.detach().cpu().numpy(),
        mu_pred.detach().cpu().numpy(),
        sigmas.detach().cpu().numpy(),
    )
    return pack, mu_pred.detach(), sigmas.detach()


def _mean_crps_numpy(y, mu, sigma):
    return float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        )
        .mean()
        .item()
    )


def _variant_config(
    variant: str,
    *,
    num_steps: int,
    warmup_frac: float,
    row_norm_weight: float,
    row_norm_mode: str,
    aux_metric_weight: float,
    aux_metric_mode: str,
):
    warm = KLWarmupSchedule(
        beta_init=0.01, beta_final=1.0,
        warmup_steps=max(1, int(num_steps * warmup_frac)), shape="linear",
    )
    cfg = dict(
        mu_V_mode="random",
        kl_warmup=None,
        row_norm_weight=0.0,
        row_norm_mode=row_norm_mode,
        aux_metric_weight=0.0,
        aux_metric_mode=aux_metric_mode,
    )
    if variant == "baseline":
        return cfg
    if variant == "kl_warmup":
        cfg["kl_warmup"] = warm; return cfg
    if variant == "pls_init":
        cfg["mu_V_mode"] = "pls"; return cfg
    if variant == "sir_init":
        cfg["mu_V_mode"] = "sir"; return cfg
    if variant == "row_norm":
        cfg["row_norm_weight"] = float(row_norm_weight); return cfg
    if variant == "aux_metric":
        cfg["aux_metric_weight"] = float(aux_metric_weight); return cfg
    if variant == "pls_kl":
        cfg["mu_V_mode"] = "pls"; cfg["kl_warmup"] = warm; return cfg
    if variant == "combo":
        cfg["mu_V_mode"] = "pls"; cfg["kl_warmup"] = warm
        cfg["row_norm_weight"] = float(row_norm_weight)
        cfg["aux_metric_weight"] = float(aux_metric_weight)
        return cfg
    if variant == "combo_full":
        cfg["mu_V_mode"] = "pls"; cfg["kl_warmup"] = warm
        cfg["row_norm_weight"] = float(row_norm_weight); cfg["row_norm_mode"] = "mu"
        cfg["aux_metric_weight"] = float(aux_metric_weight); cfg["aux_metric_mode"] = "mu"
        return cfg
    raise ValueError(f"Unknown variant {variant!r}.")


# =====================================================================
# Per-variant trainer (with optional learned/iso/random projection ablation)
# =====================================================================
def _train_one_variant(
    variant: str,
    *,
    X_train, Y_train, X_test, Y_test, W_true,
    Q: int, Q_true: int,
    m1: int, m2: int, n: int,
    num_steps: int, lr: float, log_interval: int, MC_num: int,
    warmup_frac: float, row_norm_weight: float, row_norm_mode: str,
    aux_metric_weight: float, aux_metric_mode: str, aux_max_pairs: int,
    snapshot_steps: list[int],
    init_seed: int,
    device: torch.device,
    with_projection_ablation: bool,
    proj_seed_offset: int,
) -> dict[str, Any]:
    T, D = X_test.shape
    cfg = _variant_config(
        variant,
        num_steps=num_steps, warmup_frac=warmup_frac,
        row_norm_weight=row_norm_weight, row_norm_mode=row_norm_mode,
        aux_metric_weight=aux_metric_weight, aux_metric_mode=aux_metric_mode,
    )

    _seed_all(init_seed)
    regions, X_nb_list, y_nb_list = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)
    try:
        V_params, u_params, hyperparams, init_info = _make_initial_params(
            X_train, Y_train, X_test,
            Q=Q, m1=m1, m2=m2, T=T, D=D, device=device,
            mu_V_mode=cfg["mu_V_mode"], init_seed=init_seed,
        )
    except Exception as exc:  # noqa: BLE001
        return {"variant": variant, "error": f"init_failed: {type(exc).__name__}: {exc}"}

    t0 = time.perf_counter()
    V_params, u_params, hyperparams, history, snapshots = train_vi_with_tricks(
        regions=regions,
        V_params=V_params, u_params=u_params, hyperparams=hyperparams,
        lr=lr, num_steps=num_steps, log_interval=log_interval,
        snapshot_steps=snapshot_steps,
        kl_warmup=cfg["kl_warmup"],
        row_norm_weight=cfg["row_norm_weight"],
        row_norm_mode=cfg["row_norm_mode"],
        aux_metric_weight=cfg["aux_metric_weight"],
        aux_metric_mode=cfg["aux_metric_mode"],
        aux_max_pairs=aux_max_pairs,
        aux_seed=init_seed,
    )
    train_time = time.perf_counter() - t0

    mu_W, cov_W, A, metric_block = _post_train_metric_block(V_params, hyperparams, X_test)
    align = subspace_alignment_to_truth(
        A.detach(),
        torch.from_numpy(W_true).to(device=device, dtype=mu_W.dtype),
        Q_truth=min(Q_true, A.shape[-1]),
    )
    metric_block.update(align)

    pack_pv, _mu_pv, _sig_pv = _evaluate_predict_vi(
        regions, V_params, hyperparams, Y_test, MC_num=MC_num,
    )
    method_results: dict[str, list[float]] = {"lmjgp_predict_vi": pack_pv}
    pointfiltered_results = {
        "lmjgp_predict_vi": filtered_gaussian_uq_metrics_dict(
            Y_test.detach().cpu().numpy(),
            _mu_pv.detach().cpu().numpy(),
            _sig_pv.detach().cpu().numpy(),
            elapsed_sec=float(pack_pv[2]),
            mean_crps_fn=_mean_crps_numpy,
        )
    }
    diag_proj: dict[str, Any] = {}

    if with_projection_ablation:
        L_dict: dict[str, torch.Tensor] = {}
        for mode in PROJECTION_MODES:
            L_dict[mode] = build_projection_per_anchor(
                A.detach(), Q, mode=mode,
                seed=int(init_seed) * 13 + proj_seed_offset + hash(mode) % 1000,
            )
        diag_proj = projection_quality_diagnostics(A.detach(), L_dict, Q=Q)
        from djgp.evaluation import mean_gaussian_crps_torch
        y_test_np = Y_test.detach().cpu().numpy()
        for mode in PROJECTION_MODES:
            out = run_projected_jumpgp(
                L_dict[mode], X_test, X_nb_list, y_nb_list,
                device=device, progress=False, desc=f"{variant}/{mode}",
            )
            mu_np = out["mu"]; sig_np = out["sigma"]
            rmse = float(np.sqrt(np.mean((mu_np - y_test_np) ** 2)))
            crps = float(
                mean_gaussian_crps_torch(
                    torch.from_numpy(y_test_np).float(),
                    torch.from_numpy(mu_np).float(),
                    torch.from_numpy(sig_np).float(),
                ).item()
            )
            method_results[f"jgp_metric_{mode}"] = pack_gaussian_uq_list(
                rmse, crps, float(out["elapsed_sec"]), y_test_np, mu_np, sig_np,
            )

    return {
        "variant": variant,
        "init_info": init_info,
        "hyperparams_summary": {
            "kl_warmup": None if cfg["kl_warmup"] is None else {
                "beta_init": cfg["kl_warmup"].beta_init,
                "beta_final": cfg["kl_warmup"].beta_final,
                "warmup_steps": cfg["kl_warmup"].warmup_steps,
                "shape": cfg["kl_warmup"].shape,
            },
            "row_norm_weight": cfg["row_norm_weight"],
            "row_norm_mode": cfg["row_norm_mode"],
            "aux_metric_weight": cfg["aux_metric_weight"],
            "aux_metric_mode": cfg["aux_metric_mode"],
        },
        "train_time_sec": float(train_time),
        "metric_A_summary": metric_block,
        "method_results": method_results,
        "pointfiltered_results": pointfiltered_results,
        "projection_diagnostics": diag_proj,
        "snapshots": snapshots,
        "history_tail": history[-5:] if history else [],
    }


def _per_seed_run(
    seed: int,
    *,
    variants: list[str],
    N_train: int, N_test: int, D: int, Q_true: int, jump_amp: float, noise_std: float,
    Q: int, m1: int, m2: int, n: int,
    num_steps: int, log_interval: int, MC_num: int,
    warmup_frac: float, row_norm_weight: float, row_norm_mode: str,
    aux_metric_weight: float, aux_metric_mode: str, aux_max_pairs: int,
    max_anchors: int | None,
    snapshot_steps: list[int],
    with_projection_ablation: bool,
    proj_seed_offset: int,
    device: torch.device,
):
    ds = generate_lowrank_jump(
        N_train=N_train, N_test=N_test, D=D, Q_true=Q_true,
        jump_amp=jump_amp, noise_std=noise_std, seed=seed,
    )
    X_train = torch.from_numpy(ds.X_train).float().to(device)
    Y_train = torch.from_numpy(ds.y_train).float().to(device)
    X_test = torch.from_numpy(ds.X_test).float().to(device)
    Y_test = torch.from_numpy(ds.y_test).float().to(device)
    if max_anchors is not None:
        k = min(max_anchors, X_test.shape[0])
        X_test = X_test[:k]
        Y_test = Y_test[:k]

    init_seed = int(seed) * 1019 + 11
    out_variants: dict[str, Any] = {}
    for variant in variants:
        try:
            out_variants[variant] = _train_one_variant(
                variant,
                X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
                W_true=ds.W_true, Q=Q, Q_true=Q_true,
                m1=m1, m2=m2, n=n,
                num_steps=num_steps, lr=0.01, log_interval=log_interval, MC_num=MC_num,
                warmup_frac=warmup_frac,
                row_norm_weight=row_norm_weight, row_norm_mode=row_norm_mode,
                aux_metric_weight=aux_metric_weight, aux_metric_mode=aux_metric_mode,
                aux_max_pairs=aux_max_pairs,
                snapshot_steps=snapshot_steps,
                init_seed=init_seed, device=device,
                with_projection_ablation=with_projection_ablation,
                proj_seed_offset=proj_seed_offset,
            )
        except Exception as exc:  # noqa: BLE001
            out_variants[variant] = {"variant": variant, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "label": "synth:lowrank_jump",
        "seed": int(seed),
        "Q": Q,
        "Q_true": Q_true,
        "data_info": ds.info,
        "n_test_anchors_used": int(X_test.shape[0]),
        "variants": out_variants,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="LMJGP training-time tricks ablation on synthetic low-rank jump data."
    )
    p.add_argument("--num_exp", type=int, default=3)
    p.add_argument("--variants", type=str, default=",".join(VARIANT_NAMES))
    p.add_argument("--N_train", type=int, default=1500)
    p.add_argument("--N_test", type=int, default=200)
    p.add_argument("--D", type=int, default=12)
    p.add_argument("--Q_true", type=int, default=2, help="True latent dim of the jump function.")
    p.add_argument("--Q", type=int, default=2, help="Latent dim Q used by LMJGP.")
    p.add_argument("--jump_amp", type=float, default=1.5)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--warmup_frac", type=float, default=0.5)
    p.add_argument("--row_norm_weight", type=float, default=0.1)
    p.add_argument(
        "--row_norm_mode", type=str, default="second_moment", choices=("mu", "second_moment")
    )
    p.add_argument("--aux_metric_weight", type=float, default=0.1)
    p.add_argument("--aux_metric_mode", type=str, default="A", choices=("mu", "A"))
    p.add_argument("--aux_max_pairs", type=int, default=64)
    p.add_argument(
        "--snapshot_steps", type=str, default="0,10,50",
        help="Comma-separated steps; final step is always added.",
    )
    p.add_argument(
        "--with_projection_ablation", action="store_true",
        help="Also run learned/isotropic/random projected JumpGP per variant (slow).",
    )
    p.add_argument("--proj_seed_offset", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def _fmt(x):
    return f"{x:7.4f}" if isinstance(x, float) and np.isfinite(x) else "    nan"


def _print_summary(seed: int, res: dict[str, Any]) -> None:
    print(f"\n[synth:lowrank_jump] seed={seed} (Q_true={res['Q_true']} Q={res['Q']})")
    print(
        "  variant         rmse    crps    cov90   w90     cov95   w95    "
        "med_eff_A  med_E||W||^2  capt_top   cos_truth"
    )
    for variant, info in res["variants"].items():
        if "error" in info:
            print(f"  {variant:<14}  ERROR: {info['error']}")
            continue
        pack = info["method_results"]["lmjgp_predict_vi"]
        rmse, crps, _w, c90, w90, c95, w95 = pack
        m = info["metric_A_summary"]
        snaps = info.get("snapshots", [])
        e_norm = snaps[-1].get("median_E_row_norm_sq", float("nan")) if snaps else float("nan")
        print(
            f"  {variant:<14} {_fmt(rmse)} {_fmt(crps)} {_fmt(c90)} {_fmt(w90)} {_fmt(c95)} {_fmt(w95)} "
            f"{_fmt(m['median_eff_rank_A'])} {_fmt(e_norm)} "
            f"{_fmt(m['median_capture_ratio_top_Q_truth'])} {_fmt(m['median_subspace_cos_to_truth'])}"
        )
        if info["method_results"].get("jgp_metric_learned"):
            print("    projection ablation (rmse / crps / cov95):")
            for mode in PROJECTION_MODES:
                row = info["method_results"][f"jgp_metric_{mode}"]
                print(
                    f"      {mode:<10}  rmse={_fmt(row[0])}  crps={_fmt(row[1])}  cov95={_fmt(row[5])}"
                )


def _print_snapshots(res: dict[str, Any]) -> None:
    print("  step-by-step q(W) snapshots")
    print(
        "    variant         step    frob_muW  med_muW^2  med_var_sum  med_E_W_sq  "
        "med_tr_A  med_eff_A"
    )
    for variant, info in res["variants"].items():
        if "error" in info:
            continue
        for snap in info.get("snapshots", []):
            print(
                f"    {variant:<14} {int(snap['step']):>5d}  "
                f"{_fmt(snap['median_frob_mu_W'])} "
                f"{_fmt(snap['median_row_norm_sq_mu_W'])} "
                f"{_fmt(snap['median_row_var_sum'])} "
                f"{_fmt(snap['median_E_row_norm_sq'])} "
                f"{_fmt(snap['median_trace_A'])} "
                f"{_fmt(snap['median_eff_rank_A'])}"
            )


def main():
    args = parse_args()
    variant_list = [s.strip() for s in args.variants.split(",") if s.strip()]
    unknown = [v for v in variant_list if v not in VARIANT_NAMES]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Choose from {VARIANT_NAMES}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, variants={variant_list}, ablation={args.with_projection_ablation}")
    snapshot_steps = [int(s) for s in args.snapshot_steps.split(",") if s.strip()]

    total_res: dict = {"args": vars(args), "results": {}}
    for seed in range(args.num_exp):
        res = _per_seed_run(
            seed,
            variants=variant_list,
            N_train=args.N_train, N_test=args.N_test, D=args.D, Q_true=args.Q_true,
            jump_amp=args.jump_amp, noise_std=args.noise_std,
            Q=args.Q, m1=args.m1, m2=args.m2, n=args.n,
            num_steps=args.num_steps, log_interval=args.log_interval, MC_num=args.MC_num,
            warmup_frac=args.warmup_frac,
            row_norm_weight=args.row_norm_weight, row_norm_mode=args.row_norm_mode,
            aux_metric_weight=args.aux_metric_weight, aux_metric_mode=args.aux_metric_mode,
            aux_max_pairs=args.aux_max_pairs,
            max_anchors=args.max_anchors,
            snapshot_steps=snapshot_steps,
            with_projection_ablation=args.with_projection_ablation,
            proj_seed_offset=args.proj_seed_offset,
            device=device,
        )
        total_res["results"][seed] = res
        _print_summary(seed, res)
        _print_snapshots(res)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"synth_lmjgp_tricks_{stamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(total_res, f)
    print(f"Saved pickle to {out_path}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(total_res), f, indent=2)
        print(f"Saved JSON to {args.out_json}")
    return total_res


if __name__ == "__main__":
    main()
