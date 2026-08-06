"""Compare baselines on the paper-style L2/LH synthetic generators.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Corrected one-seed pilot on the paper synthetic families.
    python experiments/synthetic/compare_paper_synthetic_baselines.py ^
        --num_exp 1 ^
        --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500 ^
        --out_dir experiments/synthetic/paper_l2_lh_xstd_train200_500_seed1

    # Five-seed table, resumable.
    python experiments/synthetic/compare_paper_synthetic_baselines.py ^
        --num_exp 5 ^
        --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500 ^
        --resume ^
        --out_dir experiments/synthetic/paper_l2_lh_xstd_train200_500_5seeds

    # Train=1000 extension on the same paper synthetic families.
    python experiments/synthetic/compare_paper_synthetic_baselines.py ^
        --num_exp 5 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --resume ^
        --out_dir experiments/synthetic/paper_l2_lh_xstd_train1000_5seeds

This runner is intentionally separate from ``compare_synthetic_baselines.py``.
It uses the historical paper data-generation code:

- L2/phantom: ``new_highdata_gen.generate_data_phantom`` with image boundary
  cases from ``data_generate.generate_Y_from_image``.
- LH: ``new_highdata_gen_utils.generate_data`` with ``caseno=0`` behavior.
- High-dimensional observed features: the Appendix-C RFF expansion from
  ``new_highdata_gen.sample_rff_params`` and ``make_rff_features``.

All reported methods use train-only X standardization after the expansion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from experiments.synthetic.compare_lmjgp_tricks_synth import _train_one_variant  # noqa: E402
from experiments.synthetic.compare_synthetic_baselines import (  # noqa: E402
    _run_cem_transductive_vi_fullcov,
    _run_jgp_pca,
)
from experiments.uci.compare_uci_baselines import _run_djgp_block  # noqa: E402
from data_gen.highdata import generate_data_phantom, generate_data_phantom_gp, make_rff_features, sample_rff_params  # noqa: E402
from data_gen.highdata_utils import generate_data as generate_lh_data  # noqa: E402


METHOD_ORDER = (
    "jgp_pca",
    "lmjgp_baseline",
    "lmjgp_pls_kl",
    "cem_em_transductive_vi_fullcov_ensemble",
)

SETTING_PRESETS: dict[str, dict[str, Any]] = {
    "l2_q2_train200": {
        "family": "L2_phantom",
        "N_train": 200,
        "N_test": 200,
        "d": 2,
        "H": 20,
        "Q_true": 2,
        "Q": 2,
        "n": 25,
        "caseno": 6,
        "noise_std": 0.5,
        "noise_var": 0.25,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "l2_q2_train500": {
        "family": "L2_phantom",
        "N_train": 500,
        "N_test": 200,
        "d": 2,
        "H": 20,
        "Q_true": 2,
        "Q": 2,
        "n": 25,
        "caseno": 6,
        "noise_std": 0.5,
        "noise_var": 0.25,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "l2_q2_train1000": {
        "family": "L2_phantom",
        "N_train": 1000,
        "N_test": 200,
        "d": 2,
        "H": 20,
        "Q_true": 2,
        "Q": 2,
        "n": 25,
        "caseno": 6,
        "noise_std": 0.5,
        "noise_var": 0.25,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "lh_q5_train200": {
        "family": "LH",
        "N_train": 200,
        "N_test": 200,
        "d": 5,
        "H": 30,
        "Q_true": 5,
        "Q": 5,
        "n": 35,
        "caseno": 0,
        "noise_std": 2.0,
        "noise_var": 4.0,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "lh_q5_train500": {
        "family": "LH",
        "N_train": 500,
        "N_test": 200,
        "d": 5,
        "H": 30,
        "Q_true": 5,
        "Q": 5,
        "n": 35,
        "caseno": 0,
        "noise_std": 2.0,
        "noise_var": 4.0,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "lh_q5_train1000": {
        "family": "LH",
        "N_train": 1000,
        "N_test": 200,
        "d": 5,
        "H": 30,
        "Q_true": 5,
        "Q": 5,
        "n": 35,
        "caseno": 0,
        "noise_std": 2.0,
        "noise_var": 4.0,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
}


# --- Main-paper grid: L2 (latent d=2) and LH (latent d=5) lifted to ambient D in
# {30,50} via four expansions {rff, poly, manifold, noise}.  Registered as named
# presets ``{l2,lh}_d{D}_{exp}`` so both this baseline runner and the ISM runner
# (which imports SETTING_PRESETS) can address them.  train=1000 / test=200.
_GRID_BASE = {
    "l2": {"family": "L2_phantom", "d": 2, "Q_true": 2, "Q": 2, "n": 25,
           "caseno": 6, "noise_std": 0.5, "noise_var": 0.25},
    "lh": {"family": "LH", "d": 5, "Q_true": 5, "Q": 5, "n": 35,
           "caseno": 0, "noise_std": 2.0, "noise_var": 4.0},
}
_GRID_EXPANSIONS = ("rff", "poly", "manifold", "noise")
_GRID_DIMS = {"l2": (30,), "lh": (30, 50)}


def register_grid_presets() -> list[str]:
    """Add ``{l2,lh}_d{D}_{exp}`` presets to SETTING_PRESETS; return their names."""
    names: list[str] = []
    for fam, base in _GRID_BASE.items():
        for D in _GRID_DIMS[fam]:
            for exp in _GRID_EXPANSIONS:
                name = f"{fam}_d{D}_{exp}"
                SETTING_PRESETS[name] = {
                    **base, "N_train": 1000, "N_test": 200, "H": int(D),
                    "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
                }
                names.append(name)
    return names


GRID_PRESET_NAMES = register_grid_presets()


# --- User benchmark grid: 6-method comparison (run by experiments/synthetic/
# _exp_paper_grid_6methods.py).  L2 (latent d=2) at ambient D=20, and LH at
# (D, latent z) in {(20,4),(30,5),(50,7)}, each lifted by five expansions
# {rff, poly, ae, randproj, manifold}.  train=1000 / test=200.
_USER_EXPANSIONS = ("rff", "poly", "ae", "randproj", "manifold")
_USER_LH_CONFIGS = ((20, 3), (20, 4), (30, 5), (50, 7))  # (ambient D, latent z)


def register_user_grid_presets() -> list[str]:
    """Add the 6-method-benchmark presets to SETTING_PRESETS; return their names."""
    names: list[str] = []
    for D in (20, 30):
        for exp in _USER_EXPANSIONS:
            name = f"l2_d{D}_{exp}"
            SETTING_PRESETS[name] = {
                # L2 (v6): within-region GP over the same phantom region template
                # (caseno=6); region means -5/+5, SE covariance theta1=9 / theta2=0.05,
                # observation noise sigma^2=1.
                "family": "L2_gp", "N_train": 1000, "N_test": 200,
                "d": 2, "H": int(D), "Q_true": 2, "Q": 2, "n": 25, "caseno": 6,
                "noise_std": 1.0, "noise_var": 1.0,
                "gp_mean_out": -5.0, "gp_mean_in": 5.0, "gp_theta1": 9.0, "gp_theta2": 0.05,
                "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
            }
            names.append(name)
    for D, z in _USER_LH_CONFIGS:
        for exp in _USER_EXPANSIONS:
            name = f"lh_d{D}z{z}_{exp}"
            SETTING_PRESETS[name] = {
                "family": "LH", "N_train": 1000, "N_test": 200,
                "d": int(z), "H": int(D), "Q_true": int(z), "Q": int(z),
                "n": 35, "caseno": 0, "noise_std": 2.0, "noise_var": 4.0,
                "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
            }
            names.append(name)
    return names


USER_GRID_PRESET_NAMES = register_user_grid_presets()


def register_lh_jump_presets() -> list[str]:
    """Jump-regression LH variant (lhj_*): random_mingap separable means (gap>=min_gap, no
    escalating blow-up) + within-region spatial GP structure (theta2=0.3, theta1=25) + bigger
    noise (std 3) + off-boundary test (tol 0.5). Unlike lh/lh_hard this is NOT piecewise-const:
    the local GP genuinely matters. train=1000/test=200. Addressed by explicit --settings."""
    names: list[str] = []
    for D, z in _USER_LH_CONFIGS:
        for exp in _USER_EXPANSIONS:
            name = f"lhj_d{D}z{z}_{exp}"
            SETTING_PRESETS[name] = {
                "family": "LH", "N_train": 1000, "N_test": 200,
                "d": int(z), "H": int(D), "Q_true": int(z), "Q": int(z),
                "n": 35, "caseno": 0, "noise_std": 3.0, "noise_var": 9.0,
                "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
                "lh_theta1": 25.0, "lh_theta2": 0.3, "lh_mu_mode": "random_mingap",
                "lh_min_gap": 6.0, "lh_test_tol": 0.5,
            }
            names.append(name)
    return names


LHJ_PRESET_NAMES = register_lh_jump_presets()


def register_l2gp_presets() -> list[str]:
    """L2_gp: phantom two-region latent where each region is a GP draw (not a flat level).
    y = mean[region] + f + noise, f ~ GP(0, 9*exp(-||z-z'||^2/200)); region means (0, 27);
    obs noise_var=4 (std 2). Same ambient lifting grid as l2 (D in {20,30}) x 5 expansions.
    Addressed by explicit --settings (not part of USER_GRID_PRESET_NAMES / 'all')."""
    names: list[str] = []
    for D in (20, 30):
        for exp in _USER_EXPANSIONS:
            name = f"l2gp_d{D}_{exp}"
            SETTING_PRESETS[name] = {
                "family": "L2_gp", "N_train": 1000, "N_test": 200,
                "d": 2, "H": int(D), "Q_true": 2, "Q": 2, "n": 25, "caseno": 6,
                "noise_std": 2.0, "noise_var": 4.0,
                "gp_mean_out": 0.0, "gp_mean_in": 27.0, "gp_theta1": 9.0, "gp_theta2": 200.0,
                "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
            }
            names.append(name)
    return names


L2GP_PRESET_NAMES = register_l2gp_presets()


def register_lh_d20z3_uniform_presets() -> list[str]:
    """LH escalating d20 z3 but with UNIFORM test sampling (lh_test_tol huge -> the
    boundary filter accepts all candidates, so test points are uniform like train,
    instead of the default boundary-concentrated 'hard' test). Same train/regions/mu as
    lh_d20z3_* at a given seed; only the test set differs. Names `lh_d20z3u_{exp}`."""
    names: list[str] = []
    for exp in _USER_EXPANSIONS:
        name = f"lh_d20z3u_{exp}"
        SETTING_PRESETS[name] = {
            "family": "LH", "N_train": 1000, "N_test": 200,
            "d": 3, "H": 20, "Q_true": 3, "Q": 3,
            "n": 35, "caseno": 0, "noise_std": 2.0, "noise_var": 4.0,
            "lh_test_tol": 1e6,
            "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
        }
        names.append(name)
    return names


LH_D20Z3_UNIFORM_PRESET_NAMES = register_lh_d20z3_uniform_presets()


def register_robust_presets() -> list[str]:
    """Robust-regression variant (rob_*): ONE smooth GP function + inlier noise + a fraction
    of TRAIN points corrupted by heavy noise; clean test. DJGP's inlier-GP + outlier-gate is a
    local robust GP -> should beat global GPs (DKL/DGP) that get dragged by outliers.
    train=1000/test=200. Addressed by explicit --settings."""
    names: list[str] = []
    for D, z in _USER_LH_CONFIGS:
        for exp in _USER_EXPANSIONS:
            name = f"rob_d{D}z{z}_{exp}"
            SETTING_PRESETS[name] = {
                "family": "robust", "N_train": 1000, "N_test": 200,
                "d": int(z), "H": int(D), "Q_true": int(z), "Q": int(z),
                "n": 35, "caseno": 0, "noise_std": 1.0, "noise_var": 1.0,
                "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
                "rob_theta1": 9.0, "rob_theta2": 0.3,
                "rob_outlier_frac": 0.15, "rob_outlier_scale": 10.0,
            }
            names.append(name)
    return names


ROBUST_PRESET_NAMES = register_robust_presets()


# --- Harder LH variant ("lh_hard"): genuine jump-regression regime. ---
# LH-as-shipped is a smooth, very-high-SNR classify-into-region-and-lookup task
# (theta2=200 => nearly flat within a region; mu_k = +/-13.5*k => jumps in the
# thousands; noise std 2) -> global flexible models (DKL/DGP) dominate and local
# jump methods have no edge.  lh_hard makes the jump the hard part and adds real
# local structure: moderate iid region means mu_k ~ N(0, mu_scale^2) (regions
# OVERLAP in y), short within-region lengthscale (theta2=10), and higher noise.
# train=1000/test=200.  Registered separately from the main grid (addressed by
# explicit --settings; NOT part of "all").
_LH_HARD_PARAMS = {"lh_theta2": 30.0, "lh_mu_scale": 6.0, "lh_mu_mode": "random",
                   "lh_test_tol": 0.2}


def register_lh_hard_presets() -> list[str]:
    names: list[str] = []
    for D, z in _USER_LH_CONFIGS:
        for exp in _USER_EXPANSIONS:
            name = f"lh_hard_d{D}z{z}_{exp}"
            SETTING_PRESETS[name] = {
                "family": "LH", "N_train": 1000, "N_test": 200,
                "d": int(z), "H": int(D), "Q_true": int(z), "Q": int(z),
                "n": 35, "caseno": 0, "noise_std": 2.0, "noise_var": 4.0,
                "expansion": exp, "lengthscale": 0.5, "kernel_var": 9.0,
                **_LH_HARD_PARAMS,
            }
            names.append(name)
    return names


LH_HARD_PRESET_NAMES = register_lh_hard_presets()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_numpy_1d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _as_numpy_2d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _expand_poly(z_tr: np.ndarray, z_te: np.ndarray, D: int, seed: int,
                 degree: int = 3, signal_var: float = 9.0) -> tuple[np.ndarray, np.ndarray]:
    """Random degree-``degree`` polynomial lift of the latent, mixed to ``D`` dims."""
    from sklearn.preprocessing import PolynomialFeatures
    rng = np.random.RandomState(int(seed))
    sc = StandardScaler().fit(z_tr)
    ztr, zte = sc.transform(z_tr), sc.transform(z_te)
    pf = PolynomialFeatures(degree=int(degree), include_bias=False)
    Ptr, Pte = pf.fit_transform(ztr), pf.transform(zte)
    sc2 = StandardScaler().fit(Ptr)
    Ptr, Pte = sc2.transform(Ptr), sc2.transform(Pte)
    W = rng.randn(Ptr.shape[1], int(D)) / np.sqrt(Ptr.shape[1])
    s = np.sqrt(float(signal_var))
    return (Ptr @ W) * s, (Pte @ W) * s


def _expand_manifold(z_tr: np.ndarray, z_te: np.ndarray, D: int, seed: int,
                     beta: float = 1.0, signal_var: float = 9.0) -> tuple[np.ndarray, np.ndarray]:
    """Swiss-roll manifold lift (constant-curvature arc of z0) mixed to ``D`` dims.

    Mirrors the projection stressor from ``_exp_betakappa_oracle._lift`` (swissroll).
    """
    rng = np.random.RandomState(int(seed))
    b = max(float(beta), 1e-8)

    def core(z: np.ndarray) -> np.ndarray:
        s, h = z[:, 0], z[:, 1]
        ex = np.sin(b * s) / b
        ey = (1.0 - np.cos(b * s)) / b
        extra = [z[:, j] for j in range(2, z.shape[1])]
        return np.stack([ex, ey, h, *extra], axis=1)

    ctr, cte = core(z_tr), core(z_te)
    A = rng.randn(ctr.shape[1], int(D)) / np.sqrt(ctr.shape[1])
    Xtr, Xte = ctr @ A, cte @ A
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    s = np.sqrt(float(signal_var))
    return ((Xtr - m) / sd) * s, ((Xte - m) / sd) * s


def _expand_noise(z_tr: np.ndarray, z_te: np.ndarray, D: int, seed: int,
                  u_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """X = rotate([z, u]) with u ~ N(0, u_scale^2) filling D-d nuisance dims (y-independent)."""
    rng = np.random.RandomState(int(seed))
    sc = StandardScaler().fit(z_tr)
    ztr, zte = sc.transform(z_tr), sc.transform(z_te)
    d = ztr.shape[1]
    n_noise = max(0, int(D) - d)
    Xtr = np.concatenate([ztr, rng.randn(ztr.shape[0], n_noise) * u_scale], axis=1)
    Xte = np.concatenate([zte, rng.randn(zte.shape[0], n_noise) * u_scale], axis=1)
    Drot = Xtr.shape[1]
    Q_rot, _ = np.linalg.qr(rng.randn(Drot, Drot))
    return Xtr @ Q_rot, Xte @ Q_rot


def _expand_ae(z_tr: np.ndarray, z_te: np.ndarray, D: int, seed: int,
               hidden: int = 64, epochs: int = 600, signal_var: float = 9.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Trained-autoencoder nonlinear lift.

    Jointly fit a decoder g: z (d) -> x (D) and an encoder f: x -> z so that the
    latent is recovered after the round trip, i.e. minimize ||z - f(g(z))||^2
    over the training latents.  This makes z faithfully encoded in x (x lies on a
    d-manifold in R^D from which z is recoverable).  The observed features are
    the decoder image x = g(z).  Deterministic given seed; train/test share the
    trained AE."""
    import torch
    import torch.nn as nn

    sc = StandardScaler().fit(z_tr)
    ztr = sc.transform(z_tr).astype(np.float32)
    zte = sc.transform(z_te).astype(np.float32)
    d = ztr.shape[1]
    g = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))

    def mlp(sizes):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    dec = mlp([d, hidden, hidden, int(D)])   # g: z -> x
    enc = mlp([int(D), hidden, hidden, d])   # f: x -> z
    opt = torch.optim.Adam(list(dec.parameters()) + list(enc.parameters()), lr=1e-3)
    Z = torch.from_numpy(ztr)
    for _ in range(int(epochs)):
        opt.zero_grad()
        x = dec(Z)
        z_rec = enc(x)
        loss = ((z_rec - Z) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xtr = dec(torch.from_numpy(ztr)).numpy()
        Xte = dec(torch.from_numpy(zte)).numpy()
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    s = np.sqrt(float(signal_var))
    return ((Xtr - m) / sd) * s, ((Xte - m) / sd) * s


def _expand_randproj(z_tr: np.ndarray, z_te: np.ndarray, D: int, seed: int,
                     signal_var: float = 9.0) -> tuple[np.ndarray, np.ndarray]:
    """Linear random-projection lift: X = z_std @ R with R ~ N(0, 1/d), a rank-d
    Gaussian random map of the standardized latent to ``D`` observed dims."""
    rng = np.random.RandomState(int(seed))
    sc = StandardScaler().fit(z_tr)
    ztr, zte = sc.transform(z_tr), sc.transform(z_te)
    d = ztr.shape[1]
    R = rng.randn(d, int(D)) / np.sqrt(d)
    Xtr, Xte = ztr @ R, zte @ R
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    s = np.sqrt(float(signal_var))
    return ((Xtr - m) / sd) * s, ((Xte - m) / sd) * s


def _generate_paper_data(cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, np.ndarray]:
    """Generate one split from the paper-style synthetic code path."""
    _set_seed(seed)
    family = str(cfg["family"])
    n_train = int(cfg["N_train"])
    n_test = int(cfg["N_test"])
    d = int(cfg["d"])
    h = int(cfg["H"])

    region_train = region_test = None
    if family == "LH":
        x_train, y_train, x_test, y_test, region_train, region_test = generate_lh_data(
            d=d,
            N=n_train,
            Nt=n_test,
            noise_var=float(cfg["noise_var"]),
            device=str(device),
            theta1=float(cfg.get("lh_theta1", 9.0)),
            theta2=float(cfg.get("lh_theta2", 200.0)),
            mu_scale=float(cfg.get("lh_mu_scale", 13.5)),
            mu_mode=str(cfg.get("lh_mu_mode", "escalating")),
            test_tol=float(cfg.get("lh_test_tol", 0.05)),
            region_mode=str(cfg.get("lh_region_mode", "hyperplane")),
            num_regions=cfg.get("lh_num_regions", None),
            min_gap=float(cfg.get("lh_min_gap", 6.0)),
            return_regions=True,
        )
    elif family == "robust":
        from data_gen.highdata_utils import generate_robust_data  # noqa: PLC0415
        x_train, y_train, x_test, y_test, region_train, region_test = generate_robust_data(
            d=d, N=n_train, Nt=n_test, device=str(device),
            theta1=float(cfg.get("rob_theta1", 9.0)),
            theta2=float(cfg.get("rob_theta2", 0.3)),
            noise_var=float(cfg["noise_var"]),
            outlier_frac=float(cfg.get("rob_outlier_frac", 0.15)),
            outlier_scale=float(cfg.get("rob_outlier_scale", 10.0)),
            return_regions=True,
        )
    elif family == "L2_phantom":
        # data_generate.generate_Y_from_image reads bound*.png from CWD.
        old_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "src" / "JumpGaussianProcess")
            x_train, y_train, x_test, y_test = generate_data_phantom(
                n_train,
                n_test,
                str(device),
                caseno=int(cfg["caseno"]),
                noise_std=float(cfg["noise_std"]),
            )
        finally:
            os.chdir(old_cwd)
    elif family == "L2_gp":
        # Phantom two-region latent, but each region is a GP draw (not a flat level):
        # y = mean[region] + f + noise, f ~ GP(0, theta1*exp(-||z-z'||^2/theta2)).
        # Reads bound*.png from CWD (same as L2_phantom).
        old_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "src" / "JumpGaussianProcess")
            x_train, y_train, x_test, y_test = generate_data_phantom_gp(
                n_train,
                n_test,
                str(device),
                caseno=int(cfg["caseno"]),
                means=(float(cfg.get("gp_mean_out", 0.0)), float(cfg.get("gp_mean_in", 27.0))),
                noise_var=float(cfg["noise_var"]),
                theta1=float(cfg.get("gp_theta1", 9.0)),
                theta2=float(cfg.get("gp_theta2", 200.0)),
            )
        finally:
            os.chdir(old_cwd)
    else:
        raise ValueError(f"Unknown paper synthetic family: {family}")

    expansion = str(cfg.get("expansion", "rff"))
    z_train_np = _as_numpy_2d(x_train)
    z_test_np = _as_numpy_2d(x_test)
    kvar = float(cfg["kernel_var"])
    exp_seed = int(seed) * 7919 + 13  # shared train/test expansion params, varies by seed
    if expansion == "rff":
        omega, phases = sample_rff_params(d, h, float(cfg["lengthscale"]), device)
        X_train = _as_numpy_2d(make_rff_features(x_train.to(device), omega, phases, kvar))
        X_test = _as_numpy_2d(make_rff_features(x_test.to(device), omega, phases, kvar))
    elif expansion == "poly":
        X_train, X_test = _expand_poly(z_train_np, z_test_np, h, exp_seed, signal_var=kvar)
    elif expansion == "manifold":
        X_train, X_test = _expand_manifold(z_train_np, z_test_np, h, exp_seed,
                                           beta=float(cfg.get("manifold_beta", 1.0)), signal_var=kvar)
    elif expansion == "noise":
        X_train, X_test = _expand_noise(z_train_np, z_test_np, h, exp_seed,
                                        u_scale=float(cfg.get("noise_u_scale", 1.0)))
    elif expansion == "ae":
        X_train, X_test = _expand_ae(z_train_np, z_test_np, h, exp_seed, signal_var=kvar)
    elif expansion == "randproj":
        X_train, X_test = _expand_randproj(z_train_np, z_test_np, h, exp_seed, signal_var=kvar)
    else:
        raise NotImplementedError(
            f"Unknown expansion {expansion!r}; choose from rff/poly/manifold/noise/ae/randproj.")

    out = {
        "X_train": _as_numpy_2d(X_train),
        "y_train": _as_numpy_1d(y_train),
        "X_test": _as_numpy_2d(X_test),
        "y_test": _as_numpy_1d(y_test),
        "z_train": z_train_np,
        "z_test": z_test_np,
    }
    if region_train is not None:
        out["region_train"] = _as_numpy_1d(region_train)
        out["region_test"] = _as_numpy_1d(region_test)
    return out


def _run_lmjgp_baseline(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    *,
    Q: int,
    n_neighbors: int,
    steps: int,
    MC_num: int,
    device: torch.device,
) -> list[float]:
    args_ns = Namespace(Q=Q, m1=2, m2=40, n=n_neighbors, num_steps=steps, MC_num=MC_num, lr=0.01)
    return _run_djgp_block(
        torch.from_numpy(X_train_s).float().to(device),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32)).float().to(device),
        torch.from_numpy(X_test_s).float().to(device),
        torch.from_numpy(np.asarray(y_test, dtype=np.float32)).float().to(device),
        args_ns,
        device,
    )


def _dummy_truth(Q: int, D: int) -> np.ndarray:
    W = np.zeros((Q, D), dtype=np.float64)
    for q in range(Q):
        W[q, q % D] = 1.0
    return W


def _run_lmjgp_pls_kl(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    Q: int,
    n_neighbors: int,
    steps: int,
    MC_num: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    t0 = time.perf_counter()
    info = _train_one_variant(
        "pls_kl",
        X_train=torch.from_numpy(X_train_s).float().to(device),
        Y_train=torch.from_numpy(np.asarray(y_train, dtype=np.float32)).float().to(device),
        X_test=torch.from_numpy(X_test_s).float().to(device),
        Y_test=torch.from_numpy(np.asarray(y_test, dtype=np.float32)).float().to(device),
        W_true=_dummy_truth(Q, X_train_s.shape[1]),
        Q=Q,
        Q_true=Q,
        m1=2,
        m2=40,
        n=n_neighbors,
        num_steps=steps,
        lr=0.01,
        log_interval=max(50, steps),
        MC_num=MC_num,
        warmup_frac=0.5,
        row_norm_weight=0.1,
        row_norm_mode="second_moment",
        aux_metric_weight=0.1,
        aux_metric_mode="A",
        aux_max_pairs=64,
        snapshot_steps=[0, min(50, steps), steps],
        init_seed=int(seed) * 1019 + 11,
        device=device,
        with_projection_ablation=False,
        proj_seed_offset=0,
    )
    if "error" in info:
        raise RuntimeError(info["error"])
    pack = list(info["method_results"]["lmjgp_predict_vi"])
    pack[2] = time.perf_counter() - t0
    pointfiltered = dict(info.get("pointfiltered_results", {}).get("lmjgp_predict_vi", {}))
    if pointfiltered:
        pointfiltered["wall_sec"] = float(pack[2])
    return pack, pointfiltered


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_keys(rows: list[dict]) -> list[str]:
    preferred = [
        "setting",
        "family",
        "expansion",
        "caseno",
        "seed",
        "method",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n",
        "standardize_x",
        "noise_std",
        "noise_var",
        "lengthscale",
        "kernel_var",
        *RESULT_ROW_KEYS,
        "n_seeds",
        "status",
        "error",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    return [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))


def _float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out: list[dict] = []
    for (setting, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {"setting": setting, "method": method, "n_seeds": len(group)}
        for meta in (
            "family",
            "expansion",
            "caseno",
            "N_train",
            "N_test",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n",
            "standardize_x",
            "noise_std",
            "noise_var",
            "lengthscale",
            "kernel_var",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        for key in ("pca_explained_variance_sum", "vi_kl", "vi_best_loss"):
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
        out.append(agg)
    return out


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Paper Synthetic L2/LH X-Standardized Baselines

Purpose: corrected baseline comparison on the paper-style synthetic generators,
not the direct low-rank Gaussian-X toy generator.

Settings:

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: exact runner arguments and setting metadata.

All methods use train-only X standardization after the RFF expansion.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style synthetic L2/LH baseline comparison.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument(
        "--settings",
        type=str,
        default="l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500",
    )
    p.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--cem_outer", type=int, default=5)
    p.add_argument("--cem_m_steps", type=int, default=80)
    p.add_argument("--cem_vi_steps", type=int, default=80)
    p.add_argument("--cem_vi_lr", type=float, default=0.005)
    p.add_argument("--cem_vi_mc", type=int, default=2)
    p.add_argument("--cem_vi_kl_weight", type=float, default=1.0)
    p.add_argument("--cem_vi_init_std", type=float, default=0.03)
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/paper_l2_lh_xstd_train200_500_seed1")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings {unknown_settings}; choose from {sorted(SETTING_PRESETS)}")
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown_methods = sorted(set(methods) - set(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"Unknown methods {unknown_methods}; choose from {METHOD_ORDER}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not metric_rows:
        metric_rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (str(r.get("setting")), int(r.get("seed")), str(r.get("method")))
        for r in metric_rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }

    pointfiltered_rows: list[dict[str, Any]] = []

    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        for seed in range(int(args.num_exp)):
            data = _generate_paper_data(cfg, seed, device)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(data["X_train"])
            X_test_s = scaler.transform(data["X_test"])
            y_train = data["y_train"]
            y_test = data["y_test"]

            for method in methods:
                if (setting, seed, method) in completed:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                row: dict[str, Any] = {
                    "setting": setting,
                    "family": cfg["family"],
                    "expansion": cfg["expansion"],
                    "caseno": cfg["caseno"],
                    "seed": seed,
                    "method": method,
                    "N_train": int(cfg["N_train"]),
                    "N_test": int(cfg["N_test"]),
                    "latent_dim": int(cfg["d"]),
                    "observed_dim": int(cfg["H"]),
                    "Q_true": int(cfg["Q_true"]),
                    "Q": int(cfg["Q"]),
                    "n": int(cfg["n"]),
                    "standardize_x": True,
                    "noise_std": float(cfg["noise_std"]),
                    "noise_var": float(cfg["noise_var"]),
                    "lengthscale": float(cfg["lengthscale"]),
                    "kernel_var": float(cfg["kernel_var"]),
                }
                try:
                    if method == "jgp_pca":
                        pack, diag = _run_jgp_pca(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                        )
                    elif method == "lmjgp_baseline":
                        pack = _run_lmjgp_baseline(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            device=device,
                        )
                        diag = {"lmjgp_steps": int(args.lmjgp_steps)}
                    elif method == "lmjgp_pls_kl":
                        pack, pointfiltered = _run_lmjgp_pls_kl(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            seed=seed,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            device=device,
                        )
                        diag = {"lmjgp_steps": int(args.lmjgp_steps), "truth_diag_skipped": True}
                    elif method == "cem_em_transductive_vi_fullcov_ensemble":
                        pack, diag = _run_cem_transductive_vi_fullcov(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            None,
                            seed=seed,
                            Q=int(cfg["Q"]),
                            Q_true=int(cfg["Q_true"]),
                            n_neighbors=int(cfg["n"]),
                            MC_num=int(args.MC_num),
                            cem_outer=int(args.cem_outer),
                            cem_m_steps=int(args.cem_m_steps),
                            cem_vi_steps=int(args.cem_vi_steps),
                            cem_vi_lr=float(args.cem_vi_lr),
                            cem_vi_mc=int(args.cem_vi_mc),
                            cem_vi_kl_weight=float(args.cem_vi_kl_weight),
                            cem_vi_init_std=float(args.cem_vi_init_std),
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unhandled method: {method}")

                    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
                    row.update(diag)
                    row["status"] = "ok"
                    if method == "lmjgp_pls_kl" and pointfiltered:
                        pf_row = dict(row)
                        pf_row.update(pointfiltered)
                        pf_row["status"] = "ok"
                        pointfiltered_rows.append(pf_row)
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR setting={setting} seed={seed} method={method}: {row['error']}", flush=True)
                metric_rows.append(row)
                _write_csv(out_dir / "metrics_partial.csv", metric_rows, _metric_keys(metric_rows))

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(out_dir / "metrics.csv", metric_rows, _metric_keys(metric_rows))
    _write_csv(out_dir / "aggregate.csv", aggregate_rows, _metric_keys(aggregate_rows))
    if pointfiltered_rows:
        existing_pf = _read_csv(out_dir / "metrics_pointfiltered.csv") if args.resume else []
        pf_by_key = {
            (str(r.get("setting")), int(r.get("seed", -1)), str(r.get("method"))): r
            for r in existing_pf
            if str(r.get("seed", "")).strip() != ""
        }
        for r in pointfiltered_rows:
            pf_by_key[(str(r.get("setting")), int(r.get("seed", -1)), str(r.get("method")))] = r
        merged_pf = list(pf_by_key.values())
        _write_csv(out_dir / "metrics_pointfiltered.csv", merged_pf, _metric_keys(merged_pf))
        _write_csv(out_dir / "aggregate_pointfiltered.csv", _aggregate(merged_pf), _metric_keys(_aggregate(merged_pf)))
    summary = {
        "args": vars(args),
        "device": str(device),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "n_rows": len(metric_rows),
        "n_ok": int(sum(1 for r in metric_rows if r.get("status") == "ok")),
        "n_errors": int(sum(1 for r in metric_rows if r.get("status") == "error")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_readme(out_dir, summary)
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": metric_rows, "aggregate": aggregate_rows, "summary": summary}


if __name__ == "__main__":
    main()
