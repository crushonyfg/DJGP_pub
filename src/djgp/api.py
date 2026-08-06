"""DJGP — Deep Jump Gaussian Process: public API.

Minimal interface for applying DJGP to a custom regression dataset.

Example
-------
>>> from djgp.api import fit_predict
>>> result = fit_predict(X_train, y_train, X_test, y_test=y_test)
>>> print(result["rmse"], result["cov90"])
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from djgp.projections.structured_metric_lmjgp import (
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp,
    train_analytic_structured_metric_lmjgp,
)

__all__ = ["fit_predict"]

_K = 6      # number of ensemble members trained
_Z90 = 1.6449


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    Q: int = 5,
    n_neighbors: int = 25,
    steps: int = 300,
    noise_mode: str = "data_driven",
    n_ensemble: int = 4,
    seed: int = 0,
    device: str | None = None,
    y_test: np.ndarray | None = None,
    auto_tune: bool = False,
) -> dict:
    """Train DJGP on (X_train, y_train) and predict on X_test.

    Parameters
    ----------
    X_train, y_train : training inputs [N, D] and targets [N]
    X_test           : test inputs [T, D]
    Q                : projection dimension (2 for sharp-boundary data, 5 for smoother)
    n_neighbors      : local neighbourhood size k
    steps            : training iterations (~300 is a good default)
    noise_mode       : ``"data_driven"`` (recommended; heteroscedastic per-anchor noise)
                       or ``"fixed"`` (homoscedastic prior)
    n_ensemble       : top-n ensemble members to mix (selected automatically)
    seed             : random seed
    device           : ``"cpu"`` / ``"cuda"``; auto-detected if ``None``
    y_test           : if provided, RMSE / CRPS / cov90 are computed and returned
    auto_tune        : if ``True``, run a short sweep over Q ∈ {2, 5} and
                       noise_mode ∈ {fixed, data_driven} and pick the best

    Returns
    -------
    dict with keys:
        ``mu``, ``sigma``              : predictive mean / std, shape [T]
        ``Q``, ``noise_mode``          : hyperparameters used
        ``rmse``, ``crps``, ``cov90``  : evaluation metrics (only if y_test provided)
    """
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Xtr = np.asarray(X_train, np.float64)
    ytr = np.asarray(y_train, np.float64).ravel()
    Xte = np.asarray(X_test, np.float64)

    if auto_tune:
        Q, noise_mode = _auto_tune(Xtr, ytr, n_neighbors=n_neighbors, seed=seed, dev=dev)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    ym = float(ytr.mean())
    ys = max(float(ytr.std()), 1e-8)
    ytr_s = (ytr - ym) / ys

    Xtr_t = torch.as_tensor(Xtr_s, device=dev, dtype=torch.float64)
    Xte_t = torch.as_tensor(Xte_s, device=dev, dtype=torch.float64)
    ytr_t = torch.as_tensor(ytr_s, device=dev, dtype=torch.float64)

    k = min(n_neighbors, len(ytr) - 1)
    Xn_t, yn_t = _neighbors(Xte_t, Xtr_t, ytr_t, k)

    inits = [_init_W(Xtr_s, ytr_s, Q, m, seed + 513) for m in ("pls", "pca", "pca_scaled")]
    for r in range(_K - 3):
        inits.append(_init_W(Xtr_s, ytr_s, Q, "random", seed + 1000 + r))

    mus, sigs = [], []
    for ki, W0 in enumerate(inits):
        mu_s, sig_s = _train_member(Xte_t, Xn_t, yn_t, W0, Q, steps, noise_mode,
                                    seed + 211 + 17 * ki, dev)
        mus.append(mu_s * ys + ym)
        sigs.append(sig_s * ys)

    mus = np.stack(mus)   # [K, T]
    sigs = np.stack(sigs) # [K, T]

    # rank members by agreement with the ensemble mean (unsupervised)
    grand = mus.mean(0)
    order = np.argsort(np.sqrt(((mus - grand[None]) ** 2).mean(1)))
    sel = order[:n_ensemble]

    mu_mix = mus[sel].mean(0)
    var_mix = np.clip((sigs[sel] ** 2 + mus[sel] ** 2).mean(0) - mu_mix ** 2, 0, None)
    sig_mix = np.sqrt(var_mix)

    out: dict = {"mu": mu_mix, "sigma": sig_mix, "Q": Q, "noise_mode": noise_mode}
    if y_test is not None:
        yt = np.asarray(y_test, np.float64).ravel()
        out["rmse"] = float(np.sqrt(np.mean((mu_mix - yt) ** 2)))
        z = (yt - mu_mix) / np.maximum(sig_mix, 1e-8)
        crps_vec = sig_mix * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        out["crps"] = float(np.mean(crps_vec))
        out["cov90"] = float(np.mean(np.abs(z) <= _Z90))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _neighbors(Xa, Xtr, ytr, k):
    d = torch.cdist(Xa.double(), Xtr.double())
    idx = d.topk(k, largest=False).indices
    return Xtr[idx], ytr[idx]


def _init_W(X, y, Q, method, seed):
    D = X.shape[1]
    rng = np.random.RandomState(seed)
    W = None
    if method == "pls":
        try:
            nc = min(Q, X.shape[1] - 1, X.shape[0] - 1, 50)
            W = PLSRegression(n_components=nc).fit(X, y).x_rotations_.T
        except Exception:
            method = "pca"
    if W is None and method == "pca":
        nc = min(Q, X.shape[1], X.shape[0])
        W = PCA(n_components=nc).fit(X).components_
    elif W is None and method == "pca_scaled":
        yw = np.abs(y - y.mean()) + 1e-8
        nc = min(Q, X.shape[1], X.shape[0])
        W = PCA(n_components=nc).fit(X * yw[:, None]).components_
    elif W is None:  # random
        W = rng.randn(Q, D)
    if W.shape[0] < Q:
        W = np.vstack([W, rng.randn(Q - W.shape[0], D)])
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return (W / np.where(norms > 1e-8, norms, 1.0)).astype(np.float64)


def _train_member(Xte_t, Xn_t, yn_t, W0_np, Q, steps, noise_mode, seed, dev):
    kw: dict = {}
    if noise_mode == "data_driven":
        kw = dict(obs_noise_prior_mode="data_driven", obs_noise_data_driven_frac=1.0,
                  obs_noise_prior_std=0.3, obs_noise_prior_weight=3.0,
                  obs_noise_max=1.5, init_noise_at_prior=True)
    W0 = torch.as_tensor(W0_np, device=dev, dtype=torch.float64)
    C, mu0 = build_fixed_local_inducing_from_w0(Xte_t, Xn_t, yn_t, W0, m_inducing=10)
    cfg = AnalyticStructuredMetricLMJGPConfig(
        steps=steps, lr=0.01, batch_anchors=Xte_t.shape[0],
        trace_target=float(Q), trace_normalize=True,
        eval_interval=steps, restore_state="final",
        rho_init=0.97, rho_prior_strength=10.0,
        min_rho_mean=0.9, min_rho_penalty_weight=50.0,
        seed=seed, **kw)
    res = train_analytic_structured_metric_lmjgp(
        Xte_t, Xn_t, yn_t, C, init_W=W0, init_mu_u=mu0, config=cfg)
    mu, var, _ = predict_analytic_structured_metric_lmjgp(
        Xte_t, res, trace_target=float(Q), trace_normalize=True)
    return mu.detach().cpu().numpy(), var.detach().sqrt().cpu().numpy()


def _auto_tune(Xtr, ytr, *, n_neighbors, seed, dev):
    rng = np.random.RandomState(seed + 17)
    val_n = max(20, len(Xtr) // 10)
    vi = rng.choice(len(Xtr), val_n, replace=False)
    ti = np.setdiff1d(np.arange(len(Xtr)), vi)
    Xv, yv = Xtr[vi], ytr[vi]
    best = (float("inf"), 5, "data_driven")
    for q in (2, 5):
        for nm in ("fixed", "data_driven"):
            try:
                r = fit_predict(Xtr[ti], ytr[ti], Xv, Q=q,
                                n_neighbors=min(n_neighbors, len(ti) - 1),
                                steps=50, noise_mode=nm, n_ensemble=1,
                                seed=seed, device=str(dev))
                sc = float(np.sqrt(np.mean((r["mu"] - yv) ** 2)))
            except Exception:
                sc = float("inf")
            if sc < best[0]:
                best = (sc, q, nm)
    return best[1], best[2]
