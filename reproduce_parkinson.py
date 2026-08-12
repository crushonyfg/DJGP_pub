"""Reproduce the Parkinsons train=1000/test=200 comparison from the paper.

Run from the repository root after installing ``requirements.txt``::

    python reproduce_parkinson.py

The three subject-disjoint splits are committed under ``reproduction_data``.
No download, tuning, GPU, or command-line configuration is required.  The
script evaluates five baselines (DGP is intentionally excluded) and DJGP with
one SIR-initialized member, then writes per-seed and aggregate CSV tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Set deterministic CPU execution before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from scipy.stats import norm
from sklearn.decomposition import PCA

from djgp.baselines.ngboost_regression import run_ngboost_regression
from djgp.model import DJGP
from jumpgp import JumpGP


SEEDS = (0, 1, 2)
DATA_SHA256 = {
    0: "62eecad3d96d6f5b375fb4e8b6a7315776710b8cbd6b2c6d7794ea4d37e48eca",
    1: "611fe98520d95040049791c17fe14f8ea4fe6a6e5d05f4e0fd905c9e2b4338dd",
    2: "2ffeabecc519ac08ad921d04b4e75b2e85e9cd0f1a18d1bc2db74d54bfb2c3ab",
}
METHODS = ("jgp", "jgp_pca", "jgp_sir", "ngboost", "bart", "djgp_sir")
Z90 = float(norm.ppf(0.95))

# Reference aggregate values rounded to the agreed two decimal places.
# Checking these makes environment drift visible instead of silently
# publishing a materially different result.
REFERENCE_2DP = {
    "jgp": (10.62, 7.49, 0.35),
    "jgp_pca": (9.96, 6.67, 0.43),
    "jgp_sir": (9.35, 6.35, 0.43),
    "ngboost": (9.69, 6.65, 0.44),
    "bart": (10.25, 7.06, 0.35),
    "djgp_sir": (7.53, 4.59, 0.95),
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _data_path(seed: int) -> Path:
    return REPO / "reproduction_data" / (
        f"uci_parkinsons_telemonitoring_train1000_seed{seed}.npz"
    )


def _load_split(seed: int) -> dict[str, np.ndarray | float]:
    path = _data_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"missing committed reproduction split: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256[seed]:
        raise RuntimeError(f"checksum mismatch for {path.name}: {digest}")
    data = np.load(path, allow_pickle=False)
    X_train = np.asarray(data["X_train"], np.float64)
    X_test = np.asarray(data["X_test"], np.float64)
    y_train_std = np.asarray(data["y_train_std"], np.float64).ravel()
    y_test = np.asarray(data["y_test"], np.float64).ravel()
    y_mean = float(np.asarray(data["y_mean"]).ravel()[0])
    y_scale = float(np.asarray(data["y_scale"]).ravel()[0])
    if X_train.shape != (1000, 18) or X_test.shape != (200, 18):
        raise RuntimeError(
            f"unexpected split shape: train={X_train.shape}, test={X_test.shape}"
        )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_std": y_train_std,
        "y_train": y_train_std * y_scale + y_mean,
        "y_test": y_test,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def _metrics(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, elapsed: float) -> dict:
    y = np.asarray(y, np.float64).ravel()
    mu = np.asarray(mu, np.float64).ravel()
    sigma = np.maximum(np.asarray(sigma, np.float64).ravel(), 1e-12)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    # Float32 matches the public benchmark's Gaussian-CRPS implementation.
    yt = torch.as_tensor(y, dtype=torch.float32)
    mt = torch.as_tensor(mu, dtype=torch.float32)
    st = torch.as_tensor(sigma, dtype=torch.float32)
    zt = (yt - mt) / st
    crps = st * (
        zt * (2.0 * (0.5 * (1.0 + torch.erf(zt / math.sqrt(2.0)))) - 1.0)
        + 2.0 * torch.exp(-0.5 * zt**2) / math.sqrt(2.0 * math.pi)
        - 1.0 / math.sqrt(math.pi)
    )
    return {
        "rmse": rmse,
        "crps": float(crps.mean().item()),
        "cov90": float(np.mean(np.abs((y - mu) / sigma) <= Z90)),
        "wall_sec": float(elapsed),
    }


def _sir_directions(X: np.ndarray, y: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """SIR directions used by the JGP-SIR baseline (dense, deterministic solver)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64).ravel()
    n, p = X.shape
    mean = X.mean(0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(X, rowvar=False))
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    inv_half = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    whitened = (X - mean) @ inv_half
    whitened = whitened[np.argsort(y, kind="stable")]
    between = np.zeros((p, p), dtype=np.float64)
    for sl in np.array_split(np.arange(n), 10):
        if sl.size:
            slice_mean = whitened[sl].mean(0)
            between += (sl.size / n) * np.outer(slice_mean, slice_mean)
    _, directions = np.linalg.eigh(between)
    return mean, np.real(inv_half @ directions[:, ::-1][:, :q])


def _run_jumpgp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    started = time.perf_counter()
    model = JumpGP(X_train, y_train, X_test, L=1, M=25, mode="CEM", bVerbose=False)
    model.fit()
    mu = np.asarray(model.jump_results["mu"], np.float64).ravel()
    var = np.asarray(model.jump_results["sig2"], np.float64).ravel()
    var = np.where(np.isfinite(var) & (var >= 0.0), var, 1e-12)
    return _metrics(y_test, mu, np.sqrt(np.maximum(var, 1e-12)), time.perf_counter() - started)


def _run_ngboost(data: dict, seed: int) -> dict:
    started = time.perf_counter()
    _, _, _, details = run_ngboost_regression(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        n_estimators=500,
        learning_rate=0.01,
        random_state=seed,
    )
    return _metrics(
        data["y_test"], details["mu"], details["sigma"], time.perf_counter() - started
    )


def _run_bart(data: dict, seed: int) -> dict:
    from bartz.BART import gbart

    started = time.perf_counter()
    model = gbart(
        data["X_train"],
        data["y_train"],
        x_test=data["X_test"],
        ntree=200,
        ndpost=200,
        nskip=100,
        seed=seed,
    )
    draws = np.asarray(model.yhat_test)
    mu = draws.mean(0)
    epistemic = draws.var(0)
    residual_sd = np.asarray(model.sigma).ravel()[-200:]
    sigma = np.sqrt(epistemic + float(np.mean(residual_sd**2)))
    return _metrics(data["y_test"], mu, sigma, time.perf_counter() - started)


def _run_djgp_sir(data: dict) -> dict:
    """Run one DJGP member with the frozen SIR-only paper configuration."""
    started = time.perf_counter()
    model = DJGP(
        q=5,
        n_neighbors=25,
        steps=200,
        beta_kl_R=0.02,
        w_signal_var=1.0,
        w_family="free",
        gate_mode="intercept",
        K_members=1,
        topk=1,
        combiner="robust",
        noise_mode="data_driven",
        m_inducing=10,
        n_inducing_R=150,
        standardize_x=False,
        standardize_y=False,
        n_val=150,
        lr=0.01,
        seed=0,
        device="cpu",
        uniform_outlier=False,
        init_methods=("sir",),
    )
    # The committed split already contains train-only standardized X and y.
    model.fit(data["X_train"], data["y_train_std"], data["X_test"])
    mu_std, sigma_std = model.predict()
    mu = mu_std * data["y_scale"] + data["y_mean"]
    sigma = sigma_std * abs(data["y_scale"])
    # DJGP's original reporting path uses the equivalent float64 SciPy formula.
    y = np.asarray(data["y_test"], np.float64)
    z = (y - mu) / np.maximum(sigma, 1e-8)
    crps = float(
        np.mean(
            sigma
            * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        )
    )
    return {
        "rmse": float(np.sqrt(np.mean((mu - y) ** 2))),
        "crps": crps,
        "cov90": float(np.mean(np.abs(z) <= Z90)),
        "wall_sec": float(time.perf_counter() - started),
    }


def _run_seed(seed: int) -> list[dict]:
    data = _load_split(seed)
    rows: list[dict] = []

    def run(method: str, function, *args) -> None:
        # Isolate each method from random numbers consumed by earlier methods.
        _seed_everything(seed)
        result = function(*args)
        row = {"seed": seed, "method": method, **result}
        rows.append(row)
        print(
            f"  {method:10s} RMSE={row['rmse']:.3f} CRPS={row['crps']:.3f} "
            f"Cov90={row['cov90']:.3f} ({row['wall_sec']:.1f}s)",
            flush=True,
        )

    print(f"[seed {seed}]", flush=True)
    run(
        "jgp",
        _run_jumpgp,
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
    )
    pca = PCA(n_components=5)
    run(
        "jgp_pca",
        _run_jumpgp,
        pca.fit_transform(data["X_train"]),
        data["y_train"],
        pca.transform(data["X_test"]),
        data["y_test"],
    )
    sir_mean, sir_directions = _sir_directions(data["X_train"], data["y_train"], 5)
    run(
        "jgp_sir",
        _run_jumpgp,
        (data["X_train"] - sir_mean) @ sir_directions,
        data["y_train"],
        (data["X_test"] - sir_mean) @ sir_directions,
        data["y_test"],
    )
    run("ngboost", _run_ngboost, data, seed)
    run("bart", _run_bart, data, seed)
    run("djgp_sir", _run_djgp_sir, data)
    return rows


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, int | str | float] = {"method": method, "n_seeds": len(selected)}
        for metric in ("rmse", "crps", "cov90", "wall_sec"):
            values = np.asarray([row[metric] for row in selected], np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(item)
    return output


def _check_reference(aggregate: list[dict]) -> None:
    mismatches = []
    for row in aggregate:
        actual = tuple(
            float(
                Decimal(str(row[f"{metric}_mean"])).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            )
            for metric in ("rmse", "crps", "cov90")
        )
        expected = REFERENCE_2DP[row["method"]]
        if actual != expected:
            mismatches.append(f"{row['method']}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError("reproduction check failed:\n  " + "\n  ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()

    per_seed_path = args.output_dir / "parkinson_results_per_seed.csv"
    aggregate_path = args.output_dir / "parkinson_results.csv"
    rows: list[dict] = []
    for seed in SEEDS:
        rows.extend(_run_seed(seed))
        _write_csv(
            per_seed_path,
            rows,
            ("seed", "method", "rmse", "crps", "cov90", "wall_sec"),
        )
    aggregate = _aggregate(rows)
    _write_csv(
        aggregate_path,
        aggregate,
        (
            "method",
            "n_seeds",
            "rmse_mean",
            "rmse_std",
            "crps_mean",
            "crps_std",
            "cov90_mean",
            "cov90_std",
            "wall_sec_mean",
            "wall_sec_std",
        ),
    )
    _check_reference(aggregate)

    print("\nmethod       RMSE (mean±sd)   CRPS (mean±sd)   Cov90 (mean±sd)")
    for row in aggregate:
        print(
            f"{row['method']:10s}   {row['rmse_mean']:.3f}±{row['rmse_std']:.3f}      "
            f"{row['crps_mean']:.3f}±{row['crps_std']:.3f}      "
            f"{row['cov90_mean']:.3f}±{row['cov90_std']:.3f}"
        )
    print(f"\nWrote {aggregate_path}")
    print(f"Wrote {per_seed_path}")
    print("Reproduction check: PASS")


if __name__ == "__main__":
    main()
