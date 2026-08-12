"""User-facing DJGP regressor: transductive fit + post-training.

DJGP is TRANSDUCTIVE -- the local models are trained around the test anchors, so the
test inputs are part of fitting:

    from djgp.model import DJGP

    model = DJGP()                               # defaults = the benchmark pipeline
    model.fit(X_train, y_train, X_test)          # trains members AT the test anchors
    mu, std = model.predict()                    # combined prediction (original y scale)

Post-training (new data and/or new test anchors; train/test both optional):

    # 1) freeze the learned W field, train only the per-anchor local layers
    mu, std = model.update(X_test_new=X_new, mode="freeze_w")
    # 2) warm-start the W field from the previous posterior and fine-tune everything
    mu, std = model.update(X_train_new=Xtr2, y_train_new=ytr2,
                           X_test_new=X_new, mode="finetune_w")
    # 3) full retrain on pooled old+new data (fresh inits)
    mu, std = model.update(X_train_new=Xtr2, y_train_new=ytr2,
                           X_test_new=X_new, mode="retrain")

Optional single-dataset tuning (one random holdout split of the TRAINING data --
NOT cross-validation):

    model = DJGP()
    model.tune(X_train, y_train, params=("q", "shrink"))
    model.fit(X_train, y_train, X_test)

Inducing points: ``m_inducing`` sets the per-anchor local inducing count;
``n_inducing_R`` sets the global W-field inducing set -- an int, "auto"
(all anchors when <=250 test points, else 150), or a user-provided ``[M, D]``
array of locations in the original X space.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from djgp.projections.lowrank_factorization import compute_pls_W0
from djgp.projections.structured_metric_lmjgp import (
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    free_w_moments_from_result,
    predict_analytic_structured_metric_lmjgp,
    structured_metric_eta_moments_from_R,
    structured_metric_w_moments_from_R,
    train_analytic_structured_metric_lmjgp,
)

_RHO_NEAR1 = dict(rho_init=0.97, rho_prior_strength=10.0, min_rho_mean=0.9,
                  min_rho_penalty_weight=50.0)
_NOISE_DATA_DRIVEN = dict(obs_noise_prior_mode="data_driven", obs_noise_data_driven_frac=1.0,
                          obs_noise_prior_std=0.3, obs_noise_prior_weight=3.0,
                          obs_noise_max=1.5, init_noise_at_prior=True)

DEFAULT_TUNE_GRIDS = {
    "q": (3, 5, 7),
    "shrink": ((0.02, 1.0), (1.0, 0.1), (5.0, 0.02)),   # (beta_kl_R, w_signal_var)
    "steps": (200, 300, 400),
    "n_neighbors": (25, 35),
    "topk": (1, 2, 4),
    "combiner": ("mixture", "robust"),
}
UPDATE_MODES = ("freeze_w", "finetune_w", "retrain")


def _neighbor_stack(X_query, X_train, y_train, *, n_neighbors):
    """Return deterministic Euclidean k-nearest-neighbour stacks."""
    with torch.no_grad():
        distances = torch.cdist(X_query, X_train)
        indices = torch.topk(
            distances,
            k=min(int(n_neighbors), X_train.shape[0]),
            largest=False,
            dim=1,
        ).indices
    return X_train[indices].contiguous(), y_train[indices].contiguous(), indices.contiguous()


def _gauss_crps(y, mu, sigma):
    sigma = np.maximum(np.asarray(sigma, np.float64), 1e-12)
    z = (np.asarray(y, np.float64) - np.asarray(mu, np.float64)) / sigma
    from scipy.stats import norm
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


def _sir_W0(X, y, q, n_slices=10):
    """Deterministic sliced-inverse-regression directions (rows normalized)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64).ravel()
    n, p = X.shape
    covariance = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    inv_half = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    whitened = (X - X.mean(0)) @ inv_half
    whitened = whitened[np.argsort(y, kind="stable")]
    between = np.zeros((p, p), dtype=np.float64)
    for sl in np.array_split(np.arange(n), int(n_slices)):
        if sl.size:
            mean = whitened[sl].mean(0)
            between += (sl.size / n) * np.outer(mean, mean)
    _, directions = np.linalg.eigh(between)
    W = np.real(inv_half @ directions[:, ::-1][:, : int(q)]).T
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)


class DJGP:
    """Deep Jump Gaussian Process regressor.

    Parameters (all optional; defaults = the benchmark pipeline)
    ----------
    q : projection dimension (default 5)
    n_neighbors : local neighbourhood size (default 25)
    steps : training iterations per projection initialisation (default 300)
    beta_kl_R, w_signal_var : strength of the Gaussian-process prior on the
        projection matrices (defaults (1.0, 0.1); larger beta_kl_R / smaller
        w_signal_var enforce smoother projections)
    w_family : parameterisation of the projection matrices, "free" (default)
        or "structured"
    gate_mode : local boundary parameterisation, "intercept" (default) or "spatial"
    K_members / topk / combiner : number of projection initialisations (8), how many
        are combined at prediction (4), and the combination rule ("robust" default
        or "mixture")
    noise_mode : "data_driven" (default) or "fixed"
    uniform_outlier : outlier head -- True (default) = flat uniform density 1/u_j,
        False = broad-Gaussian outlier
    init_methods : ordered projection initializations selected from
        ``("pls", "pca", "sir", "random")``. Set ``("sir",)`` with
        ``K_members=1`` for a single deterministic SIR-initialized member.
    m_inducing : LOCAL inducing points per anchor (default 10)
    n_inducing_R : GLOBAL W-field inducing -- int, "auto" (default), or an
        [M, D] array of custom locations (original X scale)
    standardize_x, standardize_y : bool (default True)
    n_val : training anchors held out internally for member ranking (default 150)
    lr, seed, device, verbose
    """

    def __init__(self, *, q=5, n_neighbors=25, steps=300, beta_kl_R=1.0,
                 w_signal_var=0.1, w_family="free", gate_mode="intercept",
                 K_members=8, topk=4, combiner="robust", noise_mode="data_driven",
                 m_inducing=10, n_inducing_R="auto", standardize_x=True,
                 standardize_y=True, n_val=150, lr=0.01, seed=0, device=None,
                 uniform_outlier=True, init_methods=("pls", "pca", "sir"),
                 verbose=False):
        self.q = int(q)
        self.n_neighbors = int(n_neighbors)
        self.steps = int(steps)
        self.beta_kl_R = float(beta_kl_R)
        self.w_signal_var = float(w_signal_var)
        self.w_family = str(w_family)
        self.gate_mode = str(gate_mode)
        self.K_members = int(K_members)
        self.topk = int(topk)
        self.combiner = str(combiner)
        self.noise_mode = str(noise_mode)
        self.m_inducing = int(m_inducing)
        self.n_inducing_R = n_inducing_R
        self.standardize_x = bool(standardize_x)
        self.standardize_y = bool(standardize_y)
        self.n_val = int(n_val)
        self.lr = float(lr)
        self.uniform_outlier = bool(uniform_outlier)
        self.init_methods = tuple(str(method) for method in init_methods)
        if not self.init_methods:
            raise ValueError("init_methods must contain at least one initialization")
        unknown_inits = set(self.init_methods) - {"pls", "pca", "sir", "random"}
        if unknown_inits:
            raise ValueError(f"unknown initialization methods: {sorted(unknown_inits)}")
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = bool(verbose)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X_train, y_train, X_test):
        """Transductive fit: train the member ensemble AT the test anchors.

        Stores the training pool, scalers, projection inits and the per-member
        trained states (used later by ``predict`` / ``update``).
        """
        X = np.asarray(X_train, np.float64)
        y = np.asarray(y_train, np.float64).ravel()
        self._X_raw, self._y_raw = X.copy(), y.copy()
        if self.standardize_x:
            self._x_mean, self._x_std = X.mean(0), np.maximum(X.std(0), 1e-8)
        else:
            self._x_mean, self._x_std = np.zeros(X.shape[1]), np.ones(X.shape[1])
        if self.standardize_y:
            self._y_mean, self._y_std = float(y.mean()), float(max(y.std(), 1e-8))
        else:
            self._y_mean, self._y_std = 0.0, 1.0
        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std
        self._Xtr = torch.as_tensor(Xs, device=self.device, dtype=torch.float64)
        self._ytr = torch.as_tensor(ys, device=self.device, dtype=torch.float64)
        self._init_W0s = self._make_inits(Xs, ys)
        rng = np.random.RandomState(1234 + self.seed)
        self._val_idx = rng.choice(Xs.shape[0], size=min(self.n_val, Xs.shape[0]),
                                   replace=False)
        self._fitted = True
        self._X_test_raw = np.asarray(X_test, np.float64)
        anchors = self._std_x(self._X_test_raw)
        self._members = self._train_members(anchors)
        self._cache_prediction()
        return self

    train = fit  # alias

    # -------------------------------------------------------------- predict
    def predict(self, X_test=None, return_std=True):
        """Combined prediction on the ORIGINAL y scale.

        ``X_test=None`` returns the cached prediction for the anchors given to
        ``fit``/``update``. Passing new anchors trains the members fresh at those
        anchors with the current parameters (equivalent to ``update(X_test_new=...,
        mode="retrain")`` without new training data; use ``update`` with
        ``mode="freeze_w"`` to REUSE the learned W field instead).
        """
        if not self._fitted:
            raise RuntimeError("call fit(X_train, y_train, X_test) first")
        if X_test is not None:
            self._X_test_raw = np.asarray(X_test, np.float64)
            self._members = self._train_members(self._std_x(self._X_test_raw))
            self._cache_prediction()
        return (self._mu, self._sd) if return_std else self._mu

    # ---------------------------------------------------------------- update
    def update(self, X_train_new=None, y_train_new=None, X_test_new=None, *,
               mode="freeze_w", steps=None, return_std=True):
        """Post-training with new training data and/or new test anchors.

        Modes
        -----
        "freeze_w"   : keep each member's learned W field FIXED (warm-started at the
                       new anchors' inducing set from the previous posterior, then
                       frozen); only the per-anchor local layers train. Cheapest;
                       the append-only regime.
        "finetune_w" : warm-start the W field from the previous posterior and train
                       everything normally (W fine-tunes, local layers train).
        "retrain"    : discard the learned state; refit scalers/inits on the pooled
                       old+new training data and train from scratch.

        New training data (if given) joins the neighbour/inducing pool in all modes.
        ``X_test_new=None`` reuses the previous anchors. ``steps`` overrides
        ``self.steps`` for this update (e.g. fewer steps for freeze_w).
        Returns (mu, std) for the active anchors on the original y scale.
        """
        if not self._fitted:
            raise RuntimeError("call fit(X_train, y_train, X_test) first")
        if mode not in UPDATE_MODES:
            raise ValueError(f"mode must be one of {UPDATE_MODES}")
        if (X_train_new is None) != (y_train_new is None):
            raise ValueError("X_train_new and y_train_new must be given together")
        if X_train_new is not None:
            Xn = np.asarray(X_train_new, np.float64)
            yn = np.asarray(y_train_new, np.float64).ravel()
            self._X_raw = np.concatenate([self._X_raw, Xn], 0)
            self._y_raw = np.concatenate([self._y_raw, yn], 0)
        if X_test_new is not None:
            self._X_test_raw = np.asarray(X_test_new, np.float64)

        if mode == "retrain":
            saved_steps = self.steps
            if steps is not None:
                self.steps = int(steps)
            try:
                self.fit(self._X_raw, self._y_raw, self._X_test_raw)
            finally:
                self.steps = saved_steps
            return (self._mu, self._sd) if return_std else self._mu

        # freeze_w / finetune_w: keep scalers + inits + W posteriors, extend the pool
        if X_train_new is not None:
            Xs = self._std_x(self._X_raw)
            ys = (self._y_raw - self._y_mean) / self._y_std
            self._Xtr = torch.as_tensor(Xs, device=self.device, dtype=torch.float64)
            self._ytr = torch.as_tensor(ys, device=self.device, dtype=torch.float64)
            rng = np.random.RandomState(1234 + self.seed)
            self._val_idx = rng.choice(Xs.shape[0], size=min(self.n_val, Xs.shape[0]),
                                       replace=False)
        anchors = self._std_x(self._X_test_raw)
        self._members = self._train_members(
            anchors, warm_from=self._members, freeze_w=(mode == "freeze_w"),
            steps=steps)
        self._cache_prediction()
        return (self._mu, self._sd) if return_std else self._mu

    # ----------------------------------------------------------------- tune
    def tune(self, X, y, *, params=("q", "shrink", "steps", "n_neighbors", "topk_combiner"),
             grids=None, val_fraction=0.2, metric="crps"):
        """SINGLE-DATASET holdout tuning (NOT cross-validation).

        A random ``val_fraction`` of (X, y) is held out as pseudo-test anchors; the
        benchmark coordinate descent (q -> shrink -> steps -> n_neighbors, then the
        topk x combiner re-aggregation) runs on the remainder, scored by holdout
        ``metric`` ("crps" or "rmse"). Only the groups in ``params`` are tuned;
        ``grids`` overrides DEFAULT_TUNE_GRIDS per group. Chosen values are written
        onto the instance (call ``fit`` afterwards). Returns the chosen dict.
        """
        g = dict(DEFAULT_TUNE_GRIDS)
        g.update(grids or {})
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64).ravel()
        rng = np.random.RandomState(777 + self.seed)
        n = X.shape[0]
        val_idx = rng.choice(n, size=max(1, int(round(val_fraction * n))), replace=False)
        mask = np.ones(n, bool)
        mask[val_idx] = False
        Xf, yf, Xv, yv = X[mask], y[mask], X[val_idx], y[val_idx]
        cache: dict[tuple, dict] = {}

        def score_point(qv, shrink, steps, nbr):
            key = (qv, tuple(shrink), steps, nbr)
            if key in cache:
                return cache[key]
            saved = (self.q, self.beta_kl_R, self.w_signal_var, self.steps, self.n_neighbors)
            self.q, (self.beta_kl_R, self.w_signal_var) = qv, shrink
            self.steps, self.n_neighbors = steps, nbr
            try:
                self.fit(Xf, yf, Xv)
                M, S, vs = self._M_members, self._S_members, self._valscores
                order = np.argsort(vs)
                table = {}
                for tk in g["topk"]:
                    for cb in g["combiner"]:
                        mu_s, sd_s = self._combine(M[order[:tk]], S[order[:tk]], cb)
                        mu = mu_s * self._y_std + self._y_mean
                        sd = sd_s * abs(self._y_std)
                        table[(tk, cb)] = (float(np.sqrt(np.mean((mu - yv) ** 2)))
                                           if metric == "rmse"
                                           else float(np.mean(_gauss_crps(yv, mu, sd))))
            finally:
                (self.q, self.beta_kl_R, self.w_signal_var, self.steps,
                 self.n_neighbors) = saved
            cache[key] = table
            if self.verbose:
                print(f"  [tune] q={qv} shrink={shrink} steps={steps} nbr={nbr} "
                      f"best={min(table.values()):.4f}", flush=True)
            return table

        def best(qv, shrink, steps, nbr):
            return min(score_point(qv, shrink, steps, nbr).values())

        qv, shrink = self.q, (self.beta_kl_R, self.w_signal_var)
        steps, nbr = self.steps, self.n_neighbors
        if "q" in params:
            qv = min(g["q"], key=lambda v: best(v, shrink, steps, nbr))
        if "shrink" in params:
            shrink = tuple(min(g["shrink"], key=lambda v: best(qv, tuple(v), steps, nbr)))
        if "steps" in params:
            steps = min(g["steps"], key=lambda v: best(qv, shrink, v, nbr))
        if "n_neighbors" in params:
            nbr = min(g["n_neighbors"], key=lambda v: best(qv, shrink, steps, v))
        table = score_point(qv, shrink, steps, nbr)
        (tk, cb) = (min(table, key=table.get) if "topk_combiner" in params
                    else (self.topk, self.combiner))
        self.q, (self.beta_kl_R, self.w_signal_var) = qv, shrink
        self.steps, self.n_neighbors, self.topk, self.combiner = steps, nbr, int(tk), cb
        self._fitted = False   # tune leaves no fitted state; call fit() next
        return {"q": qv, "beta_kl_R": shrink[0], "w_signal_var": shrink[1],
                "steps": steps, "n_neighbors": nbr, "topk": int(tk), "combiner": cb,
                "holdout_" + metric: table[(tk, cb)],
                "note": "single holdout split (not CV); score is in-dataset"}

    # ------------------------------------------------------------- internals
    def _std_x(self, X):
        return torch.as_tensor((np.asarray(X, np.float64) - self._x_mean) / self._x_std,
                               device=self.device, dtype=torch.float64)

    def _make_inits(self, Xs, ys):
        inits, rng_id = [], 0
        for method in self.init_methods:
            try:
                if method == "pls":
                    W = compute_pls_W0(Xs, ys, Q=self.q).astype(np.float64)
                elif method == "pca":
                    from sklearn.decomposition import PCA
                    W = PCA(n_components=self.q).fit(Xs).components_.astype(np.float64)
                elif method == "sir":
                    W = _sir_W0(Xs, ys, self.q)
                elif method == "random":
                    rr = np.random.RandomState(20260000 + self.seed + rng_id)
                    W = rr.randn(self.q, Xs.shape[1]).astype(np.float64)
                    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
                    rng_id += 1
                inits.append(W)
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                if len(self.init_methods) == 1:
                    raise
        while len(inits) < self.K_members:
            rr = np.random.RandomState(20260000 + self.seed + rng_id)
            W = rr.randn(self.q, Xs.shape[1]).astype(np.float64)
            inits.append(W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8))
            rng_id += 1
        return inits[: self.K_members]

    def _global_inducing(self, anchors):
        """Resolve n_inducing_R -> (config int or None, explicit locations or None)."""
        nir = self.n_inducing_R
        if isinstance(nir, (np.ndarray, list, tuple)) and not isinstance(nir, str):
            return None, self._std_x(np.asarray(nir, np.float64))
        T = int(anchors.shape[0])
        if nir == "auto":
            nir = None if T <= 250 else 150
        if nir is not None and 0 < int(nir) < T:
            return int(nir), None
        return None, None

    def _config(self, T, ki, *, steps=None, freeze_w=False, n_inducing_R=None):
        extra = dict(_RHO_NEAR1)
        if self.noise_mode == "data_driven":
            extra.update(_NOISE_DATA_DRIVEN)
        if n_inducing_R is not None:
            extra["n_inducing_R"] = int(n_inducing_R)
        return AnalyticStructuredMetricLMJGPConfig(
            steps=int(steps if steps is not None else self.steps), lr=self.lr,
            batch_anchors=T, trace_target=float(self.q), trace_normalize=True,
            eval_interval=10 ** 9, restore_state="final",
            gate_mode=self.gate_mode, w_family=self.w_family,
            w_signal_var=self.w_signal_var, beta_kl_R=self.beta_kl_R,
            freeze_w_field=bool(freeze_w), uniform_outlier=self.uniform_outlier,
            seed=self.seed + 211 + 17 * ki, **extra)

    def _warm_kwargs(self, member, Z_new):
        """Warm-start kwargs for the trainer: previous W-field posterior read out at
        the new inducing set ``Z_new`` (free: deviation V; structured: U + eta R)."""
        res, W0 = member["result"], member["W0"]
        if getattr(res, "V_mu", None) is not None:
            W_mu, W_cov = free_w_moments_from_result(res, Z_new)
            dev = W_mu - torch.as_tensor(W0, device=W_mu.device, dtype=W_mu.dtype).unsqueeze(0)
            var = torch.diagonal(W_cov, dim1=-2, dim2=-1).clamp_min(1e-12)
            return {"init_V_mu": dev, "init_V_log_std": (0.5 * var.log()).clamp(-8.0, 2.0)}
        eta_mu, eta_var = structured_metric_eta_moments_from_R(
            Z_new, res.inducing_X, res.R_mu, res.R_log_std,
            lengthscale=float(res.w_lengthscale), signal_var=float(res.w_signal_var),
            include_conditional_residual=bool(res.include_conditional_residual),
            jitter=1e-5, eta_mean_offset=getattr(res, "eta_mean_offset", None))
        return {"init_U": res.U, "init_R_mu": eta_mu,
                "init_R_log_std": (0.5 * eta_var.clamp_min(1e-12).log()).clamp(-8.0, 2.0)}

    def _train_members(self, anchors, *, warm_from=None, freeze_w=False, steps=None):
        """Train the K members at ``anchors``; store per-member (mu, sd) in std-y
        units, the val-CRPS ranking scores, and the trained states."""
        T = int(anchors.shape[0])
        nir_int, nir_locs = self._global_inducing(anchors)
        Z_new = nir_locs if nir_locs is not None else (
            anchors if nir_int is None else None)
        Xn, yn, _ = _neighbor_stack(anchors, self._Xtr, self._ytr,
                                    n_neighbors=self.n_neighbors)
        if yn.dim() == 3:
            yn = yn.squeeze(-1)
        Xval = self._Xtr[self._val_idx]
        Xn_v, yn_v, _ = _neighbor_stack(Xval, self._Xtr, self._ytr,
                                        n_neighbors=self.n_neighbors + 1)
        Xn_v, yn_v = Xn_v[:, 1:, :], yn_v[:, 1:]
        if yn_v.dim() == 3:
            yn_v = yn_v.squeeze(-1)
        yval = self._ytr[self._val_idx].cpu().numpy()
        members, mus, sds, vscores = [], [], [], []
        t0 = time.perf_counter()
        for ki, W0_np in enumerate(self._init_W0s):
            W0 = torch.as_tensor(W0_np, device=self.device, dtype=torch.float64)
            C, mu_u0 = build_fixed_local_inducing_from_w0(
                anchors, Xn, yn, W0, m_inducing=self.m_inducing)
            warm = {}
            if warm_from is not None:
                if Z_new is None:
                    # trainer would subsample inducing; materialize the same set here
                    from djgp.projections.structured_metric_lmjgp import _select_inducing_X
                    gix = _select_inducing_X(anchors, nir_int)
                else:
                    gix = Z_new
                warm = self._warm_kwargs(warm_from[ki], gix)
                cfg = self._config(T, ki, steps=steps, freeze_w=freeze_w)
            else:
                gix = nir_locs
                cfg = self._config(T, ki, steps=steps, freeze_w=freeze_w,
                                   n_inducing_R=nir_int)
            res = train_analytic_structured_metric_lmjgp(
                anchors, Xn, yn, C, init_W=W0, init_mu_u=mu_u0,
                global_inducing_X=gix, config=cfg, **warm)
            mu_a, var_a, _ = predict_analytic_structured_metric_lmjgp(
                anchors, res, trace_target=float(self.q), trace_normalize=True)
            mus.append(mu_a.detach().cpu().numpy())
            sds.append(var_a.detach().clamp_min(1e-12).sqrt().cpu().numpy())
            vscores.append(self._val_crps(res, Xval, Xn_v, yn_v, yval))
            members.append({"W0": W0_np, "result": res})
        if self.verbose:
            print(f"  [djgp] {len(members)} members x {T} anchors in "
                  f"{time.perf_counter() - t0:.1f}s"
                  + (" (freeze_w)" if freeze_w else ""), flush=True)
        self._M_members = np.stack(mus)
        self._S_members = np.stack(sds)
        self._valscores = np.asarray(vscores)
        return members

    def _cache_prediction(self):
        order = np.argsort(self._valscores)[: self.topk]
        mu_s, sd_s = self._combine(self._M_members[order], self._S_members[order],
                                   self.combiner)
        self._mu = mu_s * self._y_std + self._y_mean
        self._sd = sd_s * abs(self._y_std)

    def _val_crps(self, res, Xval, Xn_v, yn_v, yval):
        """Member ranking score: local-GP-in-projected-space CRPS at val anchors."""
        jitter = 1e-5
        if getattr(res, "V_mu", None) is not None:
            W_mu, _ = free_w_moments_from_result(res, Xval, jitter=jitter)
        else:
            W_mu, _, _, _, _ = structured_metric_w_moments_from_R(
                res.U, Xval, res.inducing_X, res.R_mu, res.R_log_std,
                lengthscale=float(res.w_lengthscale), signal_var=float(res.w_signal_var),
                include_conditional_residual=bool(res.include_conditional_residual),
                trace_target=float(self.q), trace_normalize=True, jitter=jitter)
        za = torch.einsum("vqd,vd->vq", W_mu, Xval)
        zn = torch.einsum("vqd,vnd->vnq", W_mu, Xn_v)
        d2 = (zn.unsqueeze(2) - zn.unsqueeze(1)).pow(2).sum(-1)
        l2 = d2.median().clamp_min(1e-3)
        nv = float(res.log_noise_var.exp().median().item())
        m = zn.shape[1]
        eye = torch.eye(m, device=zn.device, dtype=zn.dtype)
        Knn = torch.exp(-0.5 * d2 / l2) + (nv + jitter) * eye
        kan = torch.exp(-0.5 * (za.unsqueeze(1) - zn).pow(2).sum(-1) / l2)
        Kinv = torch.linalg.solve(Knn, eye.expand_as(Knn))
        mu = torch.einsum("vi,vij,vj->v", kan, Kinv, yn_v)
        kKk = torch.einsum("vi,vij,vj->v", kan, Kinv, kan)
        var = (1.0 - kKk).clamp_min(1e-6) + nv
        return float(np.mean(_gauss_crps(yval, mu.cpu().numpy(),
                                         var.sqrt().cpu().numpy())))

    @staticmethod
    def _combine(M, S, combiner):
        """Combine member (mu, sd) [k,T] -> (mu, sd) [T]."""
        if combiner == "mixture":
            mu = M.mean(0)
            var = (S ** 2 + M ** 2).mean(0) - mu ** 2
            return mu, np.sqrt(np.maximum(var, 1e-12))
        mu = np.median(M, axis=0)
        between = 1.4826 * np.median(np.abs(M - mu[None]), axis=0)
        sd = np.sqrt(np.median(S ** 2, axis=0) + between ** 2)
        return mu, sd


__all__ = ["DJGP", "DEFAULT_TUNE_GRIDS", "UPDATE_MODES"]
