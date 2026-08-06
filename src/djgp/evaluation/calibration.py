"""Marginal calibration: Gaussian predictive interval coverage and mean width."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gaussian_pi_calibration_numpy(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.90, 0.95),
) -> dict[str, float]:
    """
    Symmetric Normal(mu, sigma^2) PIs: for level 1-alpha, use z = Phi^{-1}(1-alpha/2).

    Returns keys ``coverage_{90|95}`` and ``mean_width_{90|95}`` (percent tags match nominal levels).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError("y, mu, sigma must have the same length after ravel.")
    sigma = np.maximum(sigma, 1e-12)
    out: dict[str, float] = {}
    for lvl in levels:
        alpha = 1.0 - lvl
        z = float(norm.ppf(1.0 - alpha / 2.0))
        lo = mu - z * sigma
        hi = mu + z * sigma
        cov = float(np.mean((y >= lo) & (y <= hi)))
        mw = float(np.mean(hi - lo))
        tag = str(int(round(lvl * 100)))
        out[f"coverage_{tag}"] = cov
        out[f"mean_width_{tag}"] = mw
    return out


def pack_gaussian_uq_list(
    rmse: float,
    mean_crps: float,
    elapsed_sec: float,
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
) -> list[float]:
    """
    Flat list for pickle export::

        [rmse, mean_crps, wall_sec, coverage_90, mean_width_90, coverage_95, mean_width_95]
    """
    cal = gaussian_pi_calibration_numpy(y_true, mu_pred, sigma_pred)
    return [
        float(rmse),
        float(mean_crps),
        float(elapsed_sec),
        cal["coverage_90"],
        cal["mean_width_90"],
        cal["coverage_95"],
        cal["mean_width_95"],
    ]


def gaussian_uq_metrics_dict(
    rmse: float,
    mean_crps: float,
    elapsed_sec: float,
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
) -> dict[str, float]:
    """Return the standard Gaussian UQ metrics as a named dictionary."""
    values = pack_gaussian_uq_list(rmse, mean_crps, elapsed_sec, y_true, mu_pred, sigma_pred)
    return {key: float(values[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def filtered_gaussian_uq_metrics_dict(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    *,
    elapsed_sec: float,
    mean_crps_fn,
    sigma_max: float = 1e6,
    filter_rule: str = "drop_nonfinite_or_sigma_gt_threshold",
) -> dict[str, float | int | str]:
    """
    Compute Gaussian UQ metrics after dropping only numerically invalid points.

    This is intended for formal experiment runners that also report the raw
    metrics.  It does not replace raw metrics; it adds explicit diagnostics for
    reproducible extreme-variance blow-ups.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    sigma = np.asarray(sigma_pred, dtype=np.float64).ravel()
    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError("y, mu, sigma must have the same length after ravel.")
    finite = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    keep = finite & (sigma <= float(sigma_max))
    if not np.any(keep):
        raise ValueError("All predictive points were filtered before metric computation.")
    y_keep = y[keep]
    mu_keep = mu[keep]
    sigma_keep = np.maximum(sigma[keep], 1e-12)
    rmse = float(np.sqrt(np.mean((mu_keep - y_keep) ** 2)))
    mean_crps = float(mean_crps_fn(y_keep, mu_keep, sigma_keep))
    out: dict[str, float | int | str] = gaussian_uq_metrics_dict(
        rmse,
        mean_crps,
        elapsed_sec,
        y_keep,
        mu_keep,
        sigma_keep,
    )
    out.update(
        {
            "metric_filter_rule": filter_rule,
            "sigma_max_for_metrics": float(sigma_max),
            "n_metric_points": int(keep.sum()),
            "n_dropped_metric_points": int((~keep).sum()),
            "max_sigma_before_filter": float(np.nanmax(sigma)),
            "max_sigma_after_filter": float(np.nanmax(sigma_keep)),
        }
    )
    return out


def nan_gaussian_uq_tail(rmse: float, mean_crps: float, elapsed_sec: float) -> list[float]:
    """When predictive sigma is unavailable (e.g. legacy JumpGP API)."""
    nan = float("nan")
    return [float(rmse), float(mean_crps), float(elapsed_sec), nan, nan, nan, nan]


RESULT_ROW_KEYS = (
    "rmse",
    "mean_crps",
    "wall_sec",
    "coverage_90",
    "mean_width_90",
    "coverage_95",
    "mean_width_95",
)
