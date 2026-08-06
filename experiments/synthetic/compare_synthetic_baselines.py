"""Representative synthetic low-rank jump baseline comparison.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Appendix-style table: representative and hard synthetic settings.
    python experiments/synthetic/compare_synthetic_baselines.py ^
        --num_exp 5 --settings moderate,hard ^
        --out_dir experiments/synthetic/lowrank_jump_xstd_baselines_5seeds

    # Small-train synthetic table matching the UCI train-size style.
    python experiments/synthetic/compare_synthetic_baselines.py ^
        --num_exp 5 --settings train200,train500 ^
        --out_dir experiments/synthetic/lowrank_jump_xstd_baselines_train200_500_5seeds

    # Same small-train table, but with true/fitted latent dimension Q=5.
    python experiments/synthetic/compare_synthetic_baselines.py ^
        --num_exp 5 --settings train200_q5,train500_q5 ^
        --resume ^
        --out_dir experiments/synthetic/lowrank_jump_qtrue5_xstd_baselines_train200_500_5seeds

    # CEM-EM full-covariance VI ensemble only, for Q=2 and Q=5 synthetic tables.
    python experiments/synthetic/compare_synthetic_baselines.py ^
        --num_exp 5 --settings train200,train500,train200_q5,train500_q5 ^
        --methods cem_em_transductive_vi_fullcov_ensemble ^
        --cem_outer 5 --cem_m_steps 80 --cem_vi_steps 80 --MC_num 3 --resume ^
        --out_dir experiments/synthetic/lowrank_jump_cem_vi_xstd_q2_q5_train200_500_5seeds

    # Smoke test.
    python experiments/synthetic/compare_synthetic_baselines.py ^
        --num_exp 1 --settings smoke --N_test 40 --max_anchors 40 ^
        --xgb_boots 3 --dkl_epochs 5 --lmjgp_steps 20 ^
        --out_dir experiments/synthetic/lowrank_jump_xstd_baselines_smoke

This runner standardizes X with train-only statistics before every method.  It
compares XGBoost-bootstrap, DKL-SVGP, PCA+JumpGP, CEM-EM transductive
full-covariance VI, and LMJGP on the same low-rank jump generator used by
``compare_lmjgp_tricks_synth.py``.  It is a direct high-dimensional Gaussian-X
low-rank jump toy benchmark, not the historical paper L2/LH expansion
generator from ``data_generate.py`` / ``new_highdata_gen.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    cem_step_all,
    compute_pls_W0,
    gp_condition_coefficients_vi_fullcov,
    induced_W_from_coefficients,
    run_projected_jumpgp_ensemble,
    sample_induced_coefficients_fullcov,
    train_cem_coefficient_vi,
    train_projection_cem_em,
)
from experiments.synthetic.compare_lmjgp_tricks_synth import (  # noqa: E402
    _train_one_variant,
    generate_lowrank_jump,
    subspace_alignment_to_truth,
)
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from jumpgp import JumpGP  # noqa: E402


SETTING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"N_train": 120, "D": 12, "Q_true": 2, "Q": 2, "n": 12},
    "train200": {"N_train": 200, "D": 50, "Q_true": 2, "Q": 2, "n": 25},
    "train500": {"N_train": 500, "D": 50, "Q_true": 2, "Q": 2, "n": 25},
    "train200_q5": {"N_train": 200, "D": 50, "Q_true": 5, "Q": 5, "n": 25},
    "train500_q5": {"N_train": 500, "D": 50, "Q_true": 5, "Q": 5, "n": 25},
    "moderate": {"N_train": 1000, "D": 50, "Q_true": 2, "Q": 2, "n": 25},
    "hard": {"N_train": 500, "D": 100, "Q_true": 2, "Q": 2, "n": 25},
}

METHOD_ORDER = (
    "jgp_pca",
    "cem_em_transductive_vi_fullcov_ensemble",
    "lmjgp_baseline",
    "lmjgp_pls_kl",
)


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.strip() == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _run_jgp_pca(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    *,
    Q: int,
    n_neighbors: int,
) -> tuple[list[float], dict[str, float]]:
    t0 = time.perf_counter()
    pca = PCA(n_components=Q)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    jgp = JumpGP(
        np.asarray(X_train_pca, dtype=np.float64),
        np.asarray(y_train, dtype=np.float64).reshape(-1),
        np.asarray(X_test_pca, dtype=np.float64),
        L=1,
        M=int(n_neighbors),
        mode="CEM",
        bVerbose=False,
    )
    jgp.fit()
    elapsed = time.perf_counter() - t0
    mu = np.asarray(jgp.jump_results["mu"], dtype=np.float64).reshape(-1)
    sig2 = np.asarray(jgp.jump_results["sig2"], dtype=np.float64).reshape(-1)
    n_bad_sig2 = int(np.sum(~np.isfinite(sig2) | (sig2 < 0.0)))
    sig2 = np.where(np.isfinite(sig2) & (sig2 >= 0.0), sig2, 1e-12)
    sigma = np.sqrt(np.maximum(sig2, 1e-12))
    y_eval = np.asarray(y_test, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - y_eval) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_eval, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    diag = {
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "n_bad_sig2": n_bad_sig2,
    }
    return pack_gaussian_uq_list(rmse, crps, elapsed, y_eval, mu, sigma), diag


def _run_lmjgp_variant(
    variant: str,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    W_true: np.ndarray,
    *,
    seed: int,
    Q: int,
    Q_true: int,
    n_neighbors: int,
    steps: int,
    MC_num: int,
    max_anchors: int | None,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    y_test_t = torch.from_numpy(np.asarray(y_test, dtype=np.float32)).float().to(device)
    if max_anchors is not None:
        k = min(int(max_anchors), X_test_t.shape[0])
        X_test_t = X_test_t[:k]
        y_test_t = y_test_t[:k]

    variant_map = {"lmjgp_baseline": "baseline", "lmjgp_pls_kl": "pls_kl"}
    internal = variant_map[variant]
    t0 = time.perf_counter()
    info = _train_one_variant(
        internal,
        X_train=X_train_t,
        Y_train=y_train_t,
        X_test=X_test_t,
        Y_test=y_test_t,
        W_true=W_true,
        Q=Q,
        Q_true=Q_true,
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
    elapsed = time.perf_counter() - t0
    if "error" in info:
        raise RuntimeError(info["error"])
    pack = list(info["method_results"]["lmjgp_predict_vi"])
    pack[2] = elapsed
    metric = info.get("metric_A_summary", {})
    diag = {
        "median_eff_rank_A": metric.get("median_eff_rank_A", np.nan),
        "median_capture_ratio_top_Q_truth": metric.get("median_capture_ratio_top_Q_truth", np.nan),
        "median_subspace_cos_to_truth": metric.get("median_subspace_cos_to_truth", np.nan),
    }
    return pack, diag


def _build_neighbor_lists(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    n_neighbors: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    nbr = find_neighborhoods(X_anchor.cpu(), X_pool.cpu(), y_pool.cpu(), M=int(n_neighbors))
    X_nb_list: list[torch.Tensor] = []
    y_nb_list: list[torch.Tensor] = []
    for item in nbr:
        X_nb_list.append(item["X_neighbors"].to(device))
        y_nb_list.append(item["y_neighbors"].to(device))
    return X_nb_list, y_nb_list


def _A_from_W_samples(W_samples: torch.Tensor) -> torch.Tensor:
    if W_samples.dim() == 4:
        W = W_samples.mean(dim=0)
    else:
        W = W_samples
    return torch.einsum("tqd,tqe->tde", W, W)


def _run_cem_transductive_vi_fullcov(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    W_true: np.ndarray | None,
    *,
    seed: int,
    Q: int,
    Q_true: int,
    n_neighbors: int,
    MC_num: int,
    cem_outer: int,
    cem_m_steps: int,
    cem_vi_steps: int,
    cem_vi_lr: float,
    cem_vi_mc: int,
    cem_vi_kl_weight: float,
    cem_vi_init_std: float,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    y_test_np = np.asarray(y_test, dtype=np.float64).reshape(-1)

    V_np, _ = build_V_basis(
        X_train_s,
        y_train,
        n_pls=5,
        n_pca=5,
        n_random=0,
        seed=seed,
    )
    W0_np = compute_pls_W0(X_train_s, y_train, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W0_t = torch.from_numpy(W0_np).float().to(device)
    X_nb_list, y_nb_list = _build_neighbor_lists(
        X_test_t,
        X_train_t,
        y_train_t,
        n_neighbors=n_neighbors,
        device=device,
    )

    cfg = CemEmConfig(
        n_outer=int(cem_outer),
        n_m_steps=int(cem_m_steps),
        lr=0.01,
        lambda_gate=0.1,
        lambda_smooth=0.01,
        lambda_scale=0.0,
        lambda_anchor_pred=0.0,
        anchor_pred_var_floor=0.05,
        normalize_smoothness=True,
        track_best_W=True,
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=False,
    )
    model = CoeffProjection(
        T=X_test_t.shape[0],
        Q=Q,
        D=X_test_t.shape[1],
        V=V_t,
        W0_init=W0_t,
        learn_W0=True,
        init_C_scale=0.0,
        coeff_scale=1.0,
        row_normalize=True,
    ).to(device)
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test_t,
        y_anchors=None,
        device=device,
        cfg=cfg,
        label="cem_em_transductive_synth",
    )

    cache_model = CoeffProjection(
        T=X_test_t.shape[0],
        Q=Q,
        D=X_test_t.shape[1],
        V=V_t,
        W0_init=res.W0.detach(),
        learn_W0=True,
        init_C_scale=0.0,
        coeff_scale=1.0,
        row_normalize=True,
    ).to(device)
    with torch.no_grad():
        cache_model.W0.copy_(res.W0.detach())
        cache_model.C.copy_(res.C.detach())
    cache = cem_step_all(
        cache_model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test_t,
        device=device,
        progress=False,
        desc="cem_transductive_synth_cache",
    )
    vi = train_cem_coefficient_vi(
        C_init=res.C.detach(),
        W0=res.W0.detach(),
        V=V_t.detach(),
        X_anchor=X_test_t,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        cache=cache,
        y_anchor=None,
        coeff_scale=1.0,
        row_normalize=True,
        lambda_gate=cfg.lambda_gate,
        lambda_anchor_pred=0.0,
        anchor_pred_var_floor=cfg.anchor_pred_var_floor,
        gp_ridge=1e-4,
        gp_lengthscale=None,
        n_steps=int(cem_vi_steps),
        lr=float(cem_vi_lr),
        n_mc=int(cem_vi_mc),
        kl_weight=float(cem_vi_kl_weight),
        init_std=float(cem_vi_init_std),
        verbose=False,
    )
    cond_vi = gp_condition_coefficients_vi_fullcov(X_test_t, vi, X_test_t)
    C_vi = sample_induced_coefficients_fullcov(
        cond_vi["mean"],
        cond_vi["cov"],
        n_samples=int(MC_num),
        seed=seed + 43,
    )
    W_vi = induced_W_from_coefficients(
        C_vi,
        res.W0.detach(),
        V_t,
        coeff_scale=1.0,
        row_normalize=True,
    ).detach()
    out = run_projected_jumpgp_ensemble(
        W_vi,
        X_test_t,
        X_nb_list,
        y_nb_list,
        device=device,
        progress=False,
        desc="cem_transductive_vi_fullcov_synth",
    )
    mu = np.asarray(out["mu"], dtype=np.float64)
    sigma = np.asarray(out["sigma"], dtype=np.float64)
    rmse = float(np.sqrt(np.mean((mu - y_test_np) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_test_np, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    wall_sec = float(res.train_time_sec + vi["train_time_sec"] + float(out["elapsed_sec"]))
    pack = pack_gaussian_uq_list(rmse, crps, wall_sec, y_test_np, mu, sigma)
    diag = {
        "cem_map_train_time_sec": float(res.train_time_sec),
        "cem_vi_train_time_sec": float(vi["train_time_sec"]),
        "cem_jgp_elapsed_sec": float(out["elapsed_sec"]),
        "vi_query_var_mean": float(cond_vi["query_var_mean"]),
        "vi_anchor_var_mean": float(cond_vi["posterior_anchor_var_mean"]),
        "vi_kl": float(vi["kl"]),
        "vi_best_loss": float(vi["best_loss"]),
    }
    if W_true is not None:
        W_true_t = torch.from_numpy(W_true).float().to(device)
        align = subspace_alignment_to_truth(
            _A_from_W_samples(W_vi).detach(),
            W_true_t,
            Q_truth=min(Q_true, X_test_t.shape[1]),
        )
        diag.update(
            {
                "median_capture_ratio_top_Q_truth": align["median_capture_ratio_top_Q_truth"],
                "median_subspace_cos_to_truth": align["median_subspace_cos_to_truth"],
            }
        )
    return pack, diag


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out: list[dict] = []
    for (setting, method), group in sorted(groups.items()):
        agg = {"setting": setting, "method": method, "n_seeds": len(group)}
        for meta in ("N_train", "N_test", "D", "Q_true", "Q", "n", "standardize_x"):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([float(r[key]) for r in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        for key in (
            "median_eff_rank_A",
            "median_capture_ratio_top_Q_truth",
            "median_subspace_cos_to_truth",
            "pca_explained_variance_sum",
        ):
            vals = np.asarray([_float_or_nan(r.get(key, np.nan)) for r in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
        out.append(agg)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="X-standardized synthetic low-rank jump baseline comparison.")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--settings", type=str, default="moderate")
    p.add_argument("--N_test", type=int, default=200)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--jump_amp", type=float, default=1.5)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--cem_outer", type=int, default=5)
    p.add_argument("--cem_m_steps", type=int, default=80)
    p.add_argument("--cem_vi_steps", type=int, default=80)
    p.add_argument("--cem_vi_lr", type=float, default=0.005)
    p.add_argument("--cem_vi_mc", type=int, default=2)
    p.add_argument("--cem_vi_kl_weight", type=float, default=1.0)
    p.add_argument("--cem_vi_init_std", type=float, default=0.03)
    p.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/lowrank_jump_xstd_baselines_5seeds")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load metrics_partial.csv/metrics.csv from out_dir and skip completed ok rows.",
    )
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: list[dict] = []
    if args.resume:
        metric_rows = _read_csv_dicts(out_dir / "metrics_partial.csv")
        if not metric_rows:
            metric_rows = _read_csv_dicts(out_dir / "metrics.csv")
    completed = {
        (str(r.get("setting")), int(r.get("seed")), str(r.get("method")))
        for r in metric_rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }

    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        N_train = int(cfg["N_train"])
        D = int(cfg["D"])
        Q_true = int(cfg["Q_true"])
        Q = int(cfg["Q"])
        n_neighbors = int(cfg["n"])
        for seed in range(args.num_exp):
            ds = generate_lowrank_jump(
                N_train=N_train,
                N_test=int(args.N_test),
                D=D,
                Q_true=Q_true,
                jump_amp=float(args.jump_amp),
                noise_std=float(args.noise_std),
                seed=seed,
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(ds.X_train)
            X_test_s = scaler.transform(ds.X_test)
            y_train = np.asarray(ds.y_train, dtype=np.float64).reshape(-1)
            y_test = np.asarray(ds.y_test, dtype=np.float64).reshape(-1)

            if args.max_anchors is not None:
                k = min(int(args.max_anchors), X_test_s.shape[0])
                X_test_eval = X_test_s[:k]
                y_test_eval = y_test[:k]
            else:
                X_test_eval = X_test_s
                y_test_eval = y_test

            for method in methods:
                row_key = (setting, seed, method)
                if row_key in completed:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                row = {
                    "setting": setting,
                    "seed": seed,
                    "method": method,
                    "N_train": N_train,
                    "N_test": int(len(y_test_eval)),
                    "D": D,
                    "Q_true": Q_true,
                    "Q": Q,
                    "n": n_neighbors,
                    "standardize_x": True,
                    "jump_amp": float(args.jump_amp),
                    "noise_std": float(args.noise_std),
                }
                try:
                    if method == "jgp_pca":
                        diag: dict[str, Any] = {}
                        pack, diag = _run_jgp_pca(
                            X_train_s,
                            y_train,
                            X_test_eval,
                            y_test_eval,
                            Q=Q,
                            n_neighbors=n_neighbors,
                        )
                    elif method == "cem_em_transductive_vi_fullcov_ensemble":
                        pack, diag = _run_cem_transductive_vi_fullcov(
                            X_train_s,
                            y_train,
                            X_test_eval,
                            y_test_eval,
                            ds.W_true,
                            seed=seed,
                            Q=Q,
                            Q_true=Q_true,
                            n_neighbors=n_neighbors,
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
                        pack, diag = _run_lmjgp_variant(
                            method,
                            X_train_s,
                            y_train,
                            X_test_eval,
                            y_test_eval,
                            ds.W_true,
                            seed=seed,
                            Q=Q,
                            Q_true=Q_true,
                            n_neighbors=n_neighbors,
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            max_anchors=None,
                            device=device,
                        )
                    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
                    row.update(diag)
                    row["status"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR setting={setting} seed={seed} method={method}: {row['error']}", flush=True)
                metric_rows.append(row)
                _write_csv(out_dir / "metrics_partial.csv", metric_rows, _metric_keys(metric_rows))

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(out_dir / "metrics.csv", metric_rows, _metric_keys(metric_rows))
    _write_csv(out_dir / "aggregate.csv", aggregate_rows, _metric_keys(aggregate_rows))
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
    print(f"Saved results to {out_dir}")
    return {"rows": metric_rows, "aggregate": aggregate_rows, "summary": summary}


def _metric_keys(rows: list[dict]) -> list[str]:
    preferred = [
        "setting",
        "seed",
        "method",
        "N_train",
        "N_test",
        "D",
        "Q_true",
        "Q",
        "n",
        "standardize_x",
        "jump_amp",
        "noise_std",
        *RESULT_ROW_KEYS,
        "n_seeds",
        "status",
        "error",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    return [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Synthetic Low-Rank Jump X-Standardized Baselines

Purpose: appendix-style comparison of generic baselines and LMJGP on a
representative low-rank jump synthetic benchmark.

Settings:

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: run metadata.

All methods use train-only X standardization.  The generator is imported from
`experiments/synthetic/compare_lmjgp_tricks_synth.py`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
