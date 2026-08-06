"""
Compare Parkinsons methods on a subject-grouped split.

This runner uses a stricter Parkinsons Telemonitoring split than the historical
random row split: subjects are split into train/test groups first, so no subject
appears in both sets. It then optionally subsamples rows from the train/test
pools to match small-data experiments.

Methods:

* ``xgboost``: bootstrap XGBoost baseline.
* ``lmjgp_supervised``: strict supervised q(V) learned on labeled training
  anchors, then induced at test inputs.
* ``lmjgp_residual_ridge``: ridge global mean plus supervised LMJGP on residuals.
* ``cem_em_supervised``: supervised CEM-EM coefficient field with GP induction.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test.
    python experiments/uci/compare_parkinsons_grouped_split.py ^
        --max_total_train 80 --max_test_anchors 40 --max_train_anchors 32 ^
        --lmjgp_steps 5 --n_outer 1 --n_m_steps 5 --MC_num 2 --xgb_boots 3 ^
        --out_dir experiments/uci/parkinsons_grouped_split_smoke

    # Main 200-train grouped-subject comparison.
    python experiments/uci/compare_parkinsons_grouped_split.py ^
        --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 ^
        --n 25 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --MC_num 3 ^
        --xgb_boots 30 --out_dir experiments/uci/parkinsons_grouped_split_200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections import CemEmConfig, build_V_basis, compute_pls_W0  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    _apply_scaling,
    _build_regions,
    _pack_metrics_maybe_original_y,
    _predict_lmjgp_from_qv,
    _run_cem_em_supervised,
    _run_lmjgp_supervised,
    _run_xgboost_bootstrap,
    _set_seed,
    _split_base_anchor,
    _subsample,
    _train_lmjgp_supervised_predictive,
)


DEFAULT_METHODS = "xgboost,lmjgp_supervised,lmjgp_residual_ridge,cem_em_supervised"
DATASET = "Parkinsons Telemonitoring"


def _load_parkinsons_grouped_split(
    *,
    seed: int,
    train_subject_frac: float,
) -> dict:
    repo = fetch_ucirepo(id=189)
    X_full = repo.data.features.to_numpy()[:, :-1]  # historical pipeline drops sex
    y_full = repo.data.targets.to_numpy()[:, 0]  # motor_UPDRS
    subject = repo.data.ids["subject#"].to_numpy()

    subjects = np.unique(subject)
    rng = np.random.default_rng(seed)
    subjects_perm = subjects.copy()
    rng.shuffle(subjects_perm)
    n_train_subject = int(np.floor(float(train_subject_frac) * len(subjects_perm)))
    n_train_subject = min(max(1, n_train_subject), len(subjects_perm) - 1)
    train_subjects = np.sort(subjects_perm[:n_train_subject])
    test_subjects = np.sort(subjects_perm[n_train_subject:])
    train_mask = np.isin(subject, train_subjects)
    test_mask = np.isin(subject, test_subjects)
    overlap = np.intersect1d(train_subjects, test_subjects)

    return {
        "X_train": X_full[train_mask],
        "y_train": y_full[train_mask],
        "subject_train": subject[train_mask],
        "X_test": X_full[test_mask],
        "y_test": y_full[test_mask],
        "subject_test": subject[test_mask],
        "K": 5,
        "split_meta": {
            "train_subject_frac": float(train_subject_frac),
            "n_subject_total": int(len(subjects)),
            "n_subject_train": int(len(train_subjects)),
            "n_subject_test": int(len(test_subjects)),
            "subject_overlap_count": int(len(overlap)),
            "train_subjects": train_subjects.astype(int).tolist(),
            "test_subjects": test_subjects.astype(int).tolist(),
        },
    }


def _as_metric_dict(values: list[float]) -> dict[str, float]:
    return {key: float(values[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def _invert_y(y_scaler, y_scaled: np.ndarray, sigma_scaled: np.ndarray | None = None):
    y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
    if y_scaler is None:
        sigma = None if sigma_scaled is None else np.asarray(sigma_scaled, dtype=np.float64).reshape(-1)
        return y_scaled, sigma
    y_orig = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    if sigma_scaled is None:
        return y_orig, None
    scale = float(np.squeeze(y_scaler.scale_))
    if not np.isfinite(scale) or scale < 1e-20:
        scale = 1.0
    return y_orig, np.asarray(sigma_scaled, dtype=np.float64).reshape(-1) * abs(scale)


def _run_lmjgp_residual_ridge(
    *,
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_base_s: np.ndarray,
    y_base_s: np.ndarray,
    X_anchor_s: np.ndarray,
    y_anchor_s: np.ndarray,
    X_test_s: np.ndarray,
    y_test_s: np.ndarray,
    X_train_t: torch.Tensor,
    X_base_t: torch.Tensor,
    X_anchor_t: torch.Tensor,
    X_test_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    steps: int,
    lr: float,
    MC_num: int,
    train_mc: int,
    kl_weight: float,
    pred_var_floor: float,
    ridge_alpha: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    t0 = time.perf_counter()
    ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
    ridge.fit(X_train_s, y_train_s)
    ridge_fit_time = time.perf_counter() - t0

    base_mean = ridge.predict(X_base_s)
    anchor_mean = ridge.predict(X_anchor_s)
    test_mean = ridge.predict(X_test_s)
    pool_mean = ridge.predict(X_train_s if X_pool_t.shape[0] == X_train_s.shape[0] else X_base_s)

    y_base_resid_t = torch.from_numpy(y_base_s - base_mean).float().to(device)
    y_anchor_resid_t = torch.from_numpy(y_anchor_s - anchor_mean).float().to(device)
    y_pool_np = y_train_s if X_pool_t.shape[0] == X_train_s.shape[0] else y_base_s
    y_pool_resid_t = torch.from_numpy(y_pool_np - pool_mean).float().to(device)

    _, X_anchor_nb_list, y_anchor_nb_list = _build_regions(
        X_anchor_t,
        None,
        X_base_t,
        y_base_resid_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    V_params, hyperparams, train_diag = _train_lmjgp_supervised_predictive(
        X_anchor_t=X_anchor_t,
        y_anchor_t=y_anchor_resid_t,
        X_anchor_neighbors=X_anchor_nb_list,
        y_anchor_neighbors=y_anchor_nb_list,
        X_inducing_t=X_train_t,
        Q=Q,
        m2=m2,
        num_steps=steps,
        lr=lr,
        train_mc=train_mc,
        kl_weight=kl_weight,
        pred_var_floor=pred_var_floor,
        grad_check=False,
        device=device,
    )
    test_regions, _, _ = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_resid_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    mu_resid, sig, pred_time = _predict_lmjgp_from_qv(
        regions=test_regions,
        V_params=V_params,
        hyperparams=hyperparams,
        X_query=X_test_t,
        y_query_np=y_test_s - test_mean,
        MC_num=MC_num,
    )
    diag = dict(train_diag)
    diag.update(
        {
            "variant": "ridge_global_mean_plus_lmjgp_residual",
            "ridge_alpha": float(ridge_alpha),
            "ridge_fit_time_sec": float(ridge_fit_time),
        }
    )
    return mu_resid + test_mean, sig, pred_time, diag


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_readme(out_dir: Path, args: argparse.Namespace, metric_rows: list[dict], split_meta: dict) -> None:
    lines = [
        "# Parkinsons Subject-Grouped Split",
        "",
        "This folder was produced by `experiments/uci/compare_parkinsons_grouped_split.py`.",
        "",
        "Split: subject-grouped holdout, so the same `subject#` never appears in both train and test.",
        "",
        f"- train subjects: {split_meta['n_subject_train']}",
        f"- test subjects: {split_meta['n_subject_test']}",
        f"- subject overlap: {split_meta['subject_overlap_count']}",
        "",
        "## Metrics",
        "",
        "| method | RMSE | CRPS | Cov90 | Width90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['method']} | {row['rmse']:.4f} | {row['mean_crps']:.4f} | "
            f"{row['coverage_90']:.3f} | {row['mean_width_90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```powershell",
            "python experiments/uci/compare_parkinsons_grouped_split.py "
            f"--max_total_train {args.max_total_train} --max_test_anchors {args.max_test_anchors} "
            f"--max_train_anchors {args.max_train_anchors} --n {args.n} "
            f"--lmjgp_steps {args.lmjgp_steps} --n_outer {args.n_outer} "
            f"--n_m_steps {args.n_m_steps} --MC_num {args.MC_num} "
            f"--xgb_boots {args.xgb_boots} --out_dir {args.out_dir}",
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parkinsons subject-grouped method comparison.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--max_total_train", type=int, default=200)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--neighbor_pool", type=str, default="train", choices=("base", "train"))
    p.add_argument("--standardize_x", type=int, default=0)
    p.add_argument("--standardize_y", type=int, default=0)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_scale", type=float, default=0.0)
    p.add_argument("--lambda_anchor_pred", type=float, default=1.0)
    p.add_argument("--anchor_pred_var_floor", type=float, default=0.05)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--row_normalize_W", type=int, default=1)
    p.add_argument("--normalize_smoothness", type=int, default=1)
    p.add_argument("--track_best_W", type=int, default=1)
    p.add_argument("--coeff_gp_ridge", type=float, default=1e-4)
    p.add_argument("--coeff_gp_lengthscale", type=float, default=None)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--out_dir", type=str, default="experiments/uci/parkinsons_grouped_split_200")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = set(DEFAULT_METHODS.split(","))
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)
    print(f"writing artifacts to {out_dir}", flush=True)

    split = _load_parkinsons_grouped_split(seed=args.seed, train_subject_frac=args.train_subject_frac)
    X_train_full, y_train_full, train_idx = _subsample(
        split["X_train"], split["y_train"], args.max_total_train, args.seed + 500
    )
    X_test_sub, y_test_sub, test_idx = _subsample(
        split["X_test"], split["y_test"], args.max_test_anchors, args.seed + 1000
    )
    subject_train_sub = split["subject_train"][train_idx]
    subject_test_sub = split["subject_test"][test_idx]
    subject_overlap = np.intersect1d(np.unique(subject_train_sub), np.unique(subject_test_sub))

    X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
        X_train_full,
        y_train_full,
        max_train_anchors=args.max_train_anchors,
        seed=args.seed + 2000,
    )
    (
        X_train_s,
        X_base_s,
        X_anchor_s,
        X_test_s,
        y_train_s,
        y_base_s,
        y_anchor_s,
        y_test_s,
        _,
        y_scaler,
    ) = _apply_scaling(
        X_train_full,
        X_base,
        X_anchor,
        X_test_sub,
        y_train_full,
        y_base,
        y_anchor,
        y_test_sub,
        standardize_x=bool(args.standardize_x),
        standardize_y=bool(args.standardize_y),
    )

    Q = int(args.Q if args.Q is not None else split["K"])
    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(y_train_s).float().to(device)
    X_base_t = torch.from_numpy(X_base_s).float().to(device)
    y_base_t = torch.from_numpy(y_base_s).float().to(device)
    X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
    y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    if args.neighbor_pool == "train":
        X_pool_t, y_pool_t = X_train_t, y_train_t
    else:
        X_pool_t, y_pool_t = X_base_t, y_base_t

    V_np, V_info = build_V_basis(
        X_train_s,
        y_train_s,
        n_pls=args.n_pls,
        n_pca=args.n_pca,
        n_random=args.n_random,
        seed=args.seed,
    )
    W_pls_np = compute_pls_W0(X_train_s, y_train_s, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)
    cem_cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        lambda_anchor_pred=args.lambda_anchor_pred,
        anchor_pred_var_floor=args.anchor_pred_var_floor,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=True,
    )

    summary = {
        "_meta": sanitize_for_json(vars(args)),
        "dataset": DATASET,
        "Q": int(Q),
        "D": int(X_train_s.shape[1]),
        "split": {
            **split["split_meta"],
            **split_info,
            "n_train_pool_rows": int(split["X_train"].shape[0]),
            "n_test_pool_rows": int(split["X_test"].shape[0]),
            "n_train_rows": int(X_train_s.shape[0]),
            "n_test_rows": int(X_test_s.shape[0]),
            "subsample_subject_overlap_count": int(len(subject_overlap)),
            "subsample_train_subject_count": int(len(np.unique(subject_train_sub))),
            "subsample_test_subject_count": int(len(np.unique(subject_test_sub))),
            "train_indices_within_group_pool": train_idx.tolist(),
            "test_indices_within_group_pool": test_idx.tolist(),
        },
        "scaling": {"standardize_x": bool(args.standardize_x), "standardize_y": bool(args.standardize_y)},
        "V_basis": V_info,
        "metrics": {},
        "diagnostics": {},
    }
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    def record(method: str, mu_s: np.ndarray, sig_s: np.ndarray, wall_sec: float, diag: dict) -> None:
        metrics = _pack_metrics_maybe_original_y(y_scaler, y_test_s, mu_s, sig_s, wall_sec)
        row = {"dataset": DATASET, "seed": int(args.seed), "method": method}
        row.update(_as_metric_dict(metrics))
        metric_rows.append(row)
        summary["metrics"][method] = row
        summary["diagnostics"][method] = sanitize_for_json(diag)
        y_orig, _ = _invert_y(y_scaler, y_test_s)
        mu_orig, sig_orig = _invert_y(y_scaler, mu_s, sig_s)
        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "dataset": DATASET,
                    "seed": int(args.seed),
                    "method": method,
                    "test_order": int(i),
                    "test_index_within_group_pool": int(idx),
                    "subject": int(subject_test_sub[i]),
                    "y_true": float(y_orig[i]),
                    "mu": float(mu_orig[i]),
                    "sigma": float(max(sig_orig[i], 1e-12)),
                }
            )
        print(
            f"  {method}: rmse={row['rmse']:.4f} crps={row['mean_crps']:.4f} "
            f"cov90={row['coverage_90']:.3f}",
            flush=True,
        )

    print(
        "subject split: "
        f"train_subjects={summary['split']['n_subject_train']} "
        f"test_subjects={summary['split']['n_subject_test']} "
        f"overlap={summary['split']['subject_overlap_count']} "
        f"subsample_overlap={summary['split']['subsample_subject_overlap_count']}",
        flush=True,
    )

    for method in methods:
        print(f"\nrunning {method}", flush=True)
        t0 = time.perf_counter()
        if method == "xgboost":
            mu, sig, wall, diag = _run_xgboost_bootstrap(
                X_train=X_train_s,
                y_train=y_train_s,
                X_test=X_test_s,
                y_test=y_test_s,
                n_bootstrap=args.xgb_boots,
                seed=args.seed,
            )
        elif method == "lmjgp_supervised":
            mu, sig, wall, diag = _run_lmjgp_supervised(
                X_anchor_t=X_anchor_t,
                y_anchor_t=y_anchor_t,
                X_test_t=X_test_t,
                y_test_np=y_test_s,
                X_base_t=X_base_t,
                y_base_t=y_base_t,
                X_pool_t=X_pool_t,
                y_pool_t=y_pool_t,
                X_inducing_t=X_train_t,
                Q=Q,
                m1=args.m1,
                m2=args.m2,
                n=args.n,
                steps=args.lmjgp_steps,
                lr=args.lr,
                MC_num=args.MC_num,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                grad_check=False,
                device=device,
            )
        elif method == "lmjgp_residual_ridge":
            mu, sig, wall, diag = _run_lmjgp_residual_ridge(
                X_train_s=X_train_s,
                y_train_s=y_train_s,
                X_base_s=X_base_s,
                y_base_s=y_base_s,
                X_anchor_s=X_anchor_s,
                y_anchor_s=y_anchor_s,
                X_test_s=X_test_s,
                y_test_s=y_test_s,
                X_train_t=X_train_t,
                X_base_t=X_base_t,
                X_anchor_t=X_anchor_t,
                X_test_t=X_test_t,
                X_pool_t=X_pool_t,
                Q=Q,
                m1=args.m1,
                m2=args.m2,
                n=args.n,
                steps=args.lmjgp_steps,
                lr=args.lr,
                MC_num=args.MC_num,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                ridge_alpha=args.ridge_alpha,
                device=device,
            )
        elif method == "cem_em_supervised":
            mu, sig, wall, diag = _run_cem_em_supervised(
                X_anchor_t=X_anchor_t,
                y_anchor_t=y_anchor_t,
                X_test_t=X_test_t,
                y_test_np=y_test_s,
                X_base_t=X_base_t,
                y_base_t=y_base_t,
                X_pool_t=X_pool_t,
                y_pool_t=y_pool_t,
                V_t=V_t,
                W_pls_t=W_pls_t,
                Q=Q,
                m1=args.m1,
                n=args.n,
                cfg=cem_cfg,
                init_C_scale=args.init_C_scale,
                coeff_scale=args.coeff_scale,
                row_normalize_W=bool(args.row_normalize_W),
                MC_num=args.MC_num,
                coeff_gp_ridge=args.coeff_gp_ridge,
                coeff_gp_lengthscale=args.coeff_gp_lengthscale,
                seed=args.seed + 3000,
                device=device,
            )
        else:
            raise AssertionError(method)
        diag = dict(diag)
        diag["total_wall_sec"] = float(time.perf_counter() - t0)
        record(method, mu, sig, wall, diag)

    _write_csv(out_dir / "metrics.csv", metric_rows, ["dataset", "seed", "method", *RESULT_ROW_KEYS])
    _write_csv(
        out_dir / "predictions.csv",
        pred_rows,
        ["dataset", "seed", "method", "test_order", "test_index_within_group_pool", "subject", "y_true", "mu", "sigma"],
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    _write_readme(out_dir, args, metric_rows, summary["split"])
    print(f"\nSaved artifacts -> {out_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()
