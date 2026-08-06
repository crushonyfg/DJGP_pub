"""NGBoost (Natural Gradient Boosting) probabilistic regression baseline.

NGBoost (Duan et al., ICML 2020) fits separate boosted ensembles for the
location and (log) scale of a Normal predictive distribution, giving a
*heteroscedastic* per-point (mu, sigma) -- unlike a bootstrap ensemble with a
single global aleatoric term. This makes it the standard tree/boosting UQ
baseline that is directly comparable (CRPS, cov90, width90) with the Gaussian
predictive of the GP-family methods.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore
except ImportError as e:  # pragma: no cover
    NGBRegressor = None
    _NGB_IMPORT_ERROR = e
else:
    _NGB_IMPORT_ERROR = None

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _mean_gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form Gaussian CRPS, identical formula to djgp.evaluation.crps
    (CRPS = sigma * [z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi)]). numpy/scipy only,
    so this runner has no torch/djgp dependency and runs in a standalone env."""
    from scipy.special import ndtr  # standard normal CDF

    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    cdf = ndtr(z)
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return float(np.mean(sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)))


def run_ngboost_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_estimators: int = 500,
    learning_rate: float = 0.01,
    minibatch_frac: float = 1.0,
    random_state: int = 0,
    ngb_params: dict[str, Any] | None = None,
) -> tuple[float, float, float, dict[str, np.ndarray]]:
    """
    Fit an NGBoost Normal regressor and return Gaussian predictive metrics.

    Predictive distribution per test point: Normal(mu(x), sigma(x)) with both
    parameters output by the model (heteroscedastic).

    Returns
    -------
    rmse : float
    mean_crps : float          (same Gaussian-CRPS estimator as the other baselines)
    elapsed_sec : float
    details : dict with 'mu', 'sigma'
    """
    if _NGB_IMPORT_ERROR is not None:
        raise ImportError(
            "ngboost is required for this baseline. Install with: pip install ngboost"
        ) from _NGB_IMPORT_ERROR

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64).ravel()

    t0 = time.perf_counter()

    params: dict[str, Any] = {
        "Dist": Normal,
        "Score": LogScore,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "minibatch_frac": minibatch_frac,
        "verbose": False,
        "random_state": random_state,
    }
    if ngb_params:
        params.update(ngb_params)

    model = NGBRegressor(**params)
    model.fit(X_train, y_train)

    dist = model.pred_dist(X_test)
    # Normal distn exposes loc/scale via .params; fall back to attributes.
    try:
        mu_test = np.asarray(dist.params["loc"], dtype=np.float64).ravel()
        sigma_test = np.asarray(dist.params["scale"], dtype=np.float64).ravel()
    except (AttributeError, KeyError, TypeError):  # pragma: no cover
        mu_test = np.asarray(dist.loc, dtype=np.float64).ravel()
        sigma_test = np.asarray(dist.scale, dtype=np.float64).ravel()

    sigma_test = np.maximum(sigma_test, 1e-8)

    rmse = float(np.sqrt(np.mean((mu_test - y_test) ** 2)))
    mean_crps = _mean_gaussian_crps(y_test, mu_test, sigma_test)

    elapsed = time.perf_counter() - t0
    details = {"mu": mu_test, "sigma": sigma_test}
    return rmse, mean_crps, elapsed, details
