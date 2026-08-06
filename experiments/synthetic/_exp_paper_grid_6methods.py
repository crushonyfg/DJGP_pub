"""6-method synthetic benchmark: DKL, DeepGP, JGP(raw), JGP-PCA, JGP-SIR, DJGP.

Datasets / configurations (train=1000, test=200)
------------------------------------------------
  * L2 (2-D phantom latent) lifted to ambient D=20.
  * LH (heteroscedastic-jump latent) lifted to (D, latent z) in
    {(20,4),(30,5),(50,7)}.
Each configuration is produced by five expansions of the latent:
  rff, poly, ae, randproj, manifold.

Methods
-------
  dkl       : Deep Kernel Learning + Sparse Variational GP (djgp.baselines)
  dgp       : Deep Gaussian Process (gpytorch; skipped if unavailable)
  jgp       : JumpGP on the raw full-D observed features
  jgp_pca   : PCA -> JumpGP
  jgp_sir   : SIR -> JumpGP
  djgp      : our analytic structured-metric LMJGP ensemble (no CEM).

DJGP selection (no disagreement, no CEM)
----------------------------------------
For every configuration we first run a tuning phase on a few EXTRA seeds
(disjoint from the reporting seeds).  Per tuning point we train K=8 analytic
members from inits {pls, pca, pca_scaled, sir} + random Stiefel and rank members
by held-out TRAINING-anchor validation CRPS.  Hyperparameters are tuned by
importance via coordinate descent (NOT a full grid): first Q
(reduced/projection dim, --q_offsets), then steps, then n_neighbors, each held
at the others' current best.  Only then is the init combination -- topk in
{1,2,4} and combiner in {mixture, robust}, both free post-hoc re-aggregations --
selected at the winning hyperparameters.  The obs-noise is tied to the local
Rice estimate (noise_mode=data_driven) for all families; rho_mode=near1.

Two gate variants are run (--gates), tuned independently:
  * intercept : rho is a per-anchor intercept only.
  * spatial   : JumpGP-style linear boundary gate sigmoid(nu0 + w^T W(x-xa)),
                its expectations integrated by Gauss-Hermite quadrature.
For each gate we report the FULL top1/top2/top4 breakdown (at the gate's
selected steps/n_neighbors/Q/combiner) as methods ``djgp_{int,spa}_top{1,2,4}``
-- so the accuracy-vs-UQ trade-off across topk is visible, not just the
auto-selected k.

Pass --resume to skip (setting, seed, method) rows already status=ok in --out.

Standardization is always applied (train-only StandardScaler on X, train
mean/std on y) for DJGP and the baselines.

Outputs (under --out's directory):
  metrics.csv         : one row PER (config, method, seed) — never only aggregates.
  aggregate.csv       : mean +/- std over the reporting seeds per (config, method).
  selection/<cfg>.json: DJGP tuning record per config — selected
                        (steps, n_neighbors, Q, topk, combiner), the full grid
                        scores, the init recipe, and the tuning seeds, so a run
                        is reproducible.  metrics.csv also stores the selected
                        djgp_{topk,combiner,steps,n_neighbors,q,selected_inits}.

Metrics per (config, method, seed): rmse, crps, cov90, width90, wall_sec.

How to run (conda env ``jumpGP``)
---------------------------------
    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP_

    # Smoke test: one config, tiny budgets.
    python experiments/synthetic/_exp_paper_grid_6methods.py \
        --settings l2_d20_rff --seeds 0 --tune_seeds 101 \
        --steps_grid 30 --nbr_grid 25 --dkl_epochs 5 --dgp_epochs 20 \
        --out experiments/synthetic/_exp_paper_grid_6methods/smoke.csv

    # Full run (all 20 configs, 5 reporting seeds, 2 tuning seeds).
    python experiments/synthetic/_exp_paper_grid_6methods.py \
        --settings all --seeds 0,1,2,3,4 --tune_seeds 101,102 \
        --out experiments/synthetic/_exp_paper_grid_6methods/metrics.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import gaussian_pi_calibration_numpy  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from jumpgp import JumpGP  # noqa: E402

from djgp.projections.structured_metric_lmjgp import (  # noqa: E402
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp,
    train_analytic_structured_metric_lmjgp,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    LH_HARD_PRESET_NAMES,
    SETTING_PRESETS,
    USER_GRID_PRESET_NAMES,
    _generate_paper_data,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    _neighbor_stack,
    _projection_W0,
    _set_seed,
)
from experiments.synthetic._exp_u_ensemble_5seed import (  # noqa: E402
    _crps_vec,
    _mixture_crps,
    _mixture_pi,
    _noise_kwargs,
    _rho_kwargs,
    _val_subspace_crps,
)

try:
    from shared.deepgp import DeepGP, train_model as _dgp_train
    from torch.utils.data import DataLoader, TensorDataset
    import gpytorch  # noqa: F401
    _DGP_OK = True
except Exception:  # noqa: BLE001
    _DGP_OK = False

_Z = 1.6448536269514722  # Phi^{-1}(0.95): half-width multiplier for 90% PI

ROW_KEYS = [
    "setting", "family", "expansion", "ambient_D", "latent_z", "seed", "method",
    "rmse", "crps", "cov90", "width90", "wall_sec", "status", "error",
    "djgp_gate", "djgp_topk", "djgp_combiner", "djgp_steps", "djgp_n_neighbors",
    "djgp_q", "djgp_selected_inits", "djgp_wsv", "djgp_beta",
]


# ---------------------------------------------------------------------------
# Metric helpers (Gaussian / mixture predictive)
# ---------------------------------------------------------------------------

def _gauss_metrics(y, mu, sigma, elapsed):
    y = np.asarray(y, np.float64).ravel()
    mu = np.asarray(mu, np.float64).ravel()
    sigma = np.maximum(np.asarray(sigma, np.float64).ravel(), 1e-12)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(mean_gaussian_crps_torch(
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(mu, dtype=torch.float32),
        torch.as_tensor(sigma, dtype=torch.float32)).item())
    cal = gaussian_pi_calibration_numpy(y, mu, sigma)
    return {"rmse": rmse, "crps": crps, "cov90": cal["coverage_90"],
            "width90": cal["mean_width_90"], "wall_sec": float(elapsed)}


def _mixture_eval(y, M, S, sel):
    """Equal-weight Gaussian-mixture metrics over selected members. M,S: [K,T]."""
    Ms, Ss = M[sel], S[sel]
    mu = Ms.mean(0)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    crps = float(np.mean(_mixture_crps(y, Ms, Ss)))
    lo, hi = _mixture_pi(Ms, Ss, 0.05), _mixture_pi(Ms, Ss, 0.95)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return {"rmse": rmse, "crps": crps, "cov90": cov, "width90": width}


def _robust_eval(y, M, S, sel):
    """Median-location, MAD-spread robust combiner. M,S: [K,T]."""
    Ms, Ss = M[sel], S[sel]
    mu = np.median(Ms, axis=0)
    between = 1.4826 * np.median(np.abs(Ms - mu[None, :]), axis=0)
    within = np.sqrt(np.median(Ss ** 2, axis=0))
    sigma = np.sqrt(within ** 2 + between ** 2)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    crps = float(np.mean(_crps_vec(y, mu, sigma)))
    z = (y - mu) / np.maximum(sigma, 1e-8)
    cov = float(np.mean(np.abs(z) <= _Z))
    width = float(np.mean(2.0 * _Z * sigma))
    return {"rmse": rmse, "crps": crps, "cov90": cov, "width90": width}


def _combiner_eval(combiner, y, M, S, sel):
    return _mixture_eval(y, M, S, sel) if combiner == "mixture" else _robust_eval(y, M, S, sel)


# ---------------------------------------------------------------------------
# DJGP analytic members (no CEM)
# ---------------------------------------------------------------------------

def _robust_sir_directions(X, y, Q, H=10):
    """SIR directions via dense symmetric eigendecomposition (real, ARPACK-free).

    Mirrors djgp.sir.SIR's math but replaces scipy.sparse.linalg.eigs (which
    fails to converge or returns complex vectors for small Q / near-singular
    Sigma) with numpy.linalg.eigh.  Returns (mean_x [D], directions [D, Q])."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64).reshape(-1)
    n, p = X.shape
    mean_x = X.mean(0)
    Sigma = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(Sigma)
    w = np.maximum(w, 1e-10)
    inv_half = (V * (1.0 / np.sqrt(w))) @ V.T            # Sigma^{-1/2}
    Z = (X - mean_x) @ inv_half
    Zs = Z[np.argsort(y, kind="stable")]
    M = np.zeros((p, p))
    for sl in np.array_split(np.arange(n), int(H)):
        if sl.size == 0:
            continue
        m = Zs[sl].mean(0)
        M += (sl.size / n) * np.outer(m, m)
    _, vecs = np.linalg.eigh(M)                          # ascending, real
    Vk = vecs[:, ::-1][:, :int(Q)]                       # top-Q directions
    return mean_x, np.real(inv_half @ Vk)                # [D, Q]


def _sir_W0(X, y, Q):
    _, dirs = _robust_sir_directions(X, y, int(Q))
    W = dirs.T  # [Q, D]
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)


def _build_inits(Xtr, ytr_s, Q, K, seed):
    """K projection inits: pls, pca, pca_scaled, sir, then random Stiefel.

    Returns (inits, labels) where labels name each init for reproducibility; the
    whole recipe is deterministic given ``seed`` and ``Q``.
    """
    inits, labels = [], []
    for mth in ("pls", "pca", "pca_scaled"):
        try:
            inits.append(_projection_W0(Xtr, ytr_s, Q, mth, seed=seed + 513))
            labels.append(mth)
        except Exception:  # noqa: BLE001
            pass
    try:
        inits.append(_sir_W0(Xtr, ytr_s, Q))
        labels.append("sir")
    except Exception:  # noqa: BLE001
        pass  # SIR can fail to converge; random fill covers it
    r = 0
    while len(inits) < K:
        rr = np.random.RandomState(seed + 1000 + r)
        W = rr.randn(Q, Xtr.shape[1]).astype(np.float64)
        W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        inits.append(W)
        labels.append(f"rand{r}")
        r += 1
    return inits[:K], labels[:K]


def train_djgp_members(setting, seed, *, K, steps, n_neighbors, n_val,
                       noise_mode, rho_mode, device, q_override=None,
                       gate_mode="intercept", reproject=False, w_family="structured",
                       w_signal_var=1.0, beta_kl_R=0.02):
    """Train K analytic members; return per-member test mu/sd (raw y units),
    val-CRPS rank score, true test y, init labels, and wall time.

    Standardization (always on): X via train-only StandardScaler, y via
    train mean/std; predictions are mapped back to raw y units.  ``q_override``
    sets the projection/reduced dimension Q (defaults to the preset's true Q).
    No CEM / sampleW head.
    """
    extra_cfg = {**_rho_kwargs(rho_mode),
                 **_noise_kwargs(noise_mode, 1.0, 0.3, 3.0), "gate_mode": str(gate_mode),
                 "w_family": str(w_family), "w_signal_var": float(w_signal_var),
                 "beta_kl_R": float(beta_kl_R)}
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    sx = StandardScaler()
    Xtr = sx.fit_transform(data["X_train"]).astype(np.float64)
    Xte = sx.transform(data["X_test"]).astype(np.float64)
    ytr = np.asarray(data["y_train"], np.float64).reshape(-1)
    yte = np.asarray(data["y_test"], np.float64).reshape(-1)
    ym, ys = float(ytr.mean()), float(ytr.std() if ytr.std() > 1e-8 else 1.0)
    ytr_s = (ytr - ym) / ys
    Q = int(q_override) if q_override is not None else int(cfg["Q"])
    Xtr_t = torch.as_tensor(Xtr, device=device, dtype=torch.float64)
    Xte_t = torch.as_tensor(Xte, device=device, dtype=torch.float64)
    ytr_s_t = torch.as_tensor(ytr_s, device=device, dtype=torch.float64)
    T = Xte_t.shape[0]

    rng = np.random.RandomState(1234 + seed)
    val_idx = rng.choice(Xtr.shape[0], size=min(n_val, Xtr.shape[0]), replace=False)
    Xval_t = Xtr_t[val_idx]
    Xn_val, yn_val, _ = _neighbor_stack(Xval_t, Xtr_t, ytr_s_t, n_neighbors=n_neighbors + 1)
    Xn_val, yn_val = Xn_val[:, 1:, :], yn_val[:, 1:]
    if yn_val.dim() == 3:
        yn_val = yn_val.squeeze(-1)
    yval_true = ytr_s[val_idx]

    Xn, yn, _ = _neighbor_stack(Xte_t, Xtr_t, ytr_s_t, n_neighbors=n_neighbors)
    if yn.dim() == 3:
        yn = yn.squeeze(-1)

    inits, init_types = _build_inits(Xtr, ytr_s, Q, K, seed)
    an_mu, an_sd, valscore = [], [], []
    t0 = time.perf_counter()
    for ki, W0_np in enumerate(inits):
        W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
        built = build_fixed_local_inducing_from_w0(
            Xte_t, Xn, yn, W0, m_inducing=10, return_raw=reproject)
        if reproject:
            C, mu_u0, X_raw = built
        else:
            C, mu_u0 = built
            X_raw = None
        result = train_analytic_structured_metric_lmjgp(
            Xte_t, Xn, yn, C, init_W=W0, init_mu_u=mu_u0,
            X_inducing_raw=X_raw,
            config=AnalyticStructuredMetricLMJGPConfig(
                steps=steps, lr=0.01, batch_anchors=T,
                trace_target=float(Q), trace_normalize=True,
                eval_interval=5, restore_state="final",
                reproject_local_inducing=bool(reproject),
                # Benchmark reproduction uses the Gaussian outlier (the frozen
                # docs/benchmark_configs/*.json were tuned under it). The user-facing
                # djgp.model.DJGP defaults to the paper's uniform 1/u_j outlier.
                uniform_outlier=False,
                seed=seed + 211 + 17 * ki, **extra_cfg))
        mu_a, var_a, _ = predict_analytic_structured_metric_lmjgp(
            Xte_t, result, trace_target=float(Q), trace_normalize=True)
        an_mu.append(mu_a.detach().cpu().numpy() * ys + ym)
        an_sd.append(var_a.detach().sqrt().cpu().numpy() * abs(ys))
        noise_med = float(result.log_noise_var.exp().median().item())
        vmu, vsig = _val_subspace_crps(result, Xval_t, Xn_val, yn_val, noise_med, Q, float(Q))
        valscore.append(float(np.mean(_crps_vec(yval_true, vmu, vsig))))
    wall = time.perf_counter() - t0
    return {"an_mu": np.stack(an_mu), "an_sd": np.stack(an_sd),
            "valscore": np.asarray(valscore), "yte": yte, "wall": wall,
            "init_types": init_types, "Q": int(Q)}


def _djgp_row(members, topk, combiner):
    """Aggregate top-k members (ranked by val-CRPS) under the given combiner.

    Also reports which init types won (the selected top-k members)."""
    order = np.argsort(members["valscore"])  # lower CRPS = better
    sel = order[:int(topk)]
    out = _combiner_eval(combiner, members["yte"], members["an_mu"], members["an_sd"], sel)
    out["selected_inits"] = ",".join(members["init_types"][i] for i in sel)
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _prep_xy(setting, seed, device):
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    sc = StandardScaler()
    Xtr = sc.fit_transform(data["X_train"])
    Xte = sc.transform(data["X_test"])
    return Xtr, data["y_train"], Xte, data["y_test"], cfg


def _jumpgp_predict(Xtr, ytr, Xte, yte, n_neighbors):
    t0 = time.perf_counter()
    jgp = JumpGP(np.asarray(Xtr, np.float64), np.asarray(ytr, np.float64).reshape(-1),
                 np.asarray(Xte, np.float64), L=1, M=int(n_neighbors),
                 mode="CEM", bVerbose=False)
    jgp.fit()
    elapsed = time.perf_counter() - t0
    mu = np.asarray(jgp.jump_results["mu"], np.float64).reshape(-1)
    sig2 = np.asarray(jgp.jump_results["sig2"], np.float64).reshape(-1)
    sig2 = np.where(np.isfinite(sig2) & (sig2 >= 0.0), sig2, 1e-12)
    return _gauss_metrics(yte, mu, np.sqrt(np.maximum(sig2, 1e-12)), elapsed)


def _run_dgp(Xtr, ytr, Xte, yte, device, epochs):
    if not _DGP_OK:
        raise RuntimeError("gpytorch not available")
    from argparse import Namespace
    from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
    t0 = time.perf_counter()
    ym, ys = float(np.mean(ytr)), float(max(np.std(ytr), 1e-8))
    to32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    loader = DataLoader(TensorDataset(to32(Xtr), to32((ytr - ym) / ys)), batch_size=128, shuffle=True)
    te_loader = DataLoader(TensorDataset(to32(Xte), to32((yte - ym) / ys)), batch_size=128)
    model = DeepGP(to32(Xtr).shape, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, to32(Xtr).shape[0]))
    args = Namespace(num_epochs=int(epochs), patience=30, clip_grad=1.0, print_freq=10 ** 9, eval=True)
    _dgp_train(model, loader, te_loader, opt, mll, args)
    model.eval()
    with torch.no_grad():
        pm, pv, _ = model.predict(te_loader)
    mu = pm.mean(0).cpu().numpy() * ys + ym
    sigma = pv.mean(0).clamp_min(1e-12).sqrt().cpu().numpy() * ys
    return _gauss_metrics(yte, mu, sigma, time.perf_counter() - t0)


def _run_jgp_pca(Xtr, ytr, Xte, yte, Q, n_neighbors):
    pca = PCA(n_components=int(Q))
    return _jumpgp_predict(pca.fit_transform(Xtr), ytr, pca.transform(Xte), yte, n_neighbors)


def _run_jgp_sir(Xtr, ytr, Xte, yte, Q, n_neighbors):
    mean_x, P = _robust_sir_directions(Xtr, ytr, int(Q))  # [D, Q], real
    Ztr = (np.asarray(Xtr, np.float64) - mean_x) @ P
    Zte = (np.asarray(Xte, np.float64) - mean_x) @ P
    return _jumpgp_predict(Ztr, ytr, Zte, yte, n_neighbors)


def _run_ngboost(Xtr, ytr, Xte, yte, seed, n_estimators=500, learning_rate=0.01):
    """NGBoost Normal regressor on the ambient features (heteroscedastic mu/sigma)."""
    from djgp.baselines.ngboost_regression import run_ngboost_regression  # noqa: PLC0415
    t0 = time.perf_counter()
    _, _, _, det = run_ngboost_regression(
        np.asarray(Xtr, np.float64), np.asarray(ytr, np.float64),
        np.asarray(Xte, np.float64), np.asarray(yte, np.float64),
        n_estimators=int(n_estimators), learning_rate=float(learning_rate),
        random_state=int(seed))
    return _gauss_metrics(yte, det["mu"], det["sigma"], time.perf_counter() - t0)


def _run_bart(Xtr, ytr, Xte, yte, seed, ntree=200, ndpost=200, nskip=100):
    """BART (bartz, JAX on CPU) on the ambient features. Predictive Normal with
    mu = posterior mean of f, sigma = sqrt(Var[f] (epistemic) + E[sigma^2] (aleatoric))."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep JAX off the GPU (torch owns it)
    from bartz.BART import gbart  # noqa: PLC0415
    t0 = time.perf_counter()
    m = gbart(np.asarray(Xtr, np.float64), np.asarray(ytr, np.float64).ravel(),
              x_test=np.asarray(Xte, np.float64),
              ntree=int(ntree), ndpost=int(ndpost), nskip=int(nskip), seed=int(seed))
    yhat = np.asarray(m.yhat_test)              # [ndpost, n_test] posterior draws of f
    mu = yhat.mean(0)
    epistemic = yhat.var(0)
    sig = np.asarray(m.sigma).ravel()[-int(ndpost):]   # posterior residual-std draws
    aleatoric = float(np.mean(sig ** 2))
    sigma = np.sqrt(epistemic + aleatoric)
    return _gauss_metrics(yte, mu, sigma, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# DJGP tuning over extra seeds
# ---------------------------------------------------------------------------

def tune_djgp(setting, family, tune_seeds, *, K, steps_grid, nbr_grid, q_grid, n_val, device,
              gate_mode="intercept", reproject=False, w_family="structured",
              shrink_grid=((0.02, 1.0),)):
    """Staged (greedy) selection over the extra tuning seeds.

    Hyperparameters are tuned by importance via coordinate descent rather than a
    full Cartesian grid:
      1. Q (reduced/projection dim) at (steps0, nbr0, shrink0),
      2. shrink = (beta_kl_R, w_signal_var) PAIR at (Q*, steps0, nbr0)  [only when
         len(shrink_grid) > 1; the free-W trust-region strength -- effective KL rent
         is beta/sigma_w^2 and the row budget D*sigma_w^2 grows with ambient D],
      3. steps at (Q*, shrink*, nbr0),
      4. n_neighbors at (Q*, shrink*, steps*),
    each holding the others at their current best; defaults are Q=Q_true,
    steps=middle of the grid, n_neighbors=the preset's ``n``.  Only THEN is the
    init combination (topk x combiner) chosen at the winners -- these are
    free post-hoc re-aggregations.  noise_mode / rho_mode are fixed per family.
    Each (steps, nbr, Q, shrink) point is trained once (cached) and scored by its
    BEST achievable mean CRPS over (topk, combiner)."""
    noise_mode = "data_driven"  # tie obs-noise to the local Rice estimate (all families)
    rho_mode = "near1"
    topks, combiners = (1, 2, 4), ("mixture", "robust")
    cache: dict[tuple, dict] = {}
    shrink_grid = tuple((float(b), float(w)) for b, w in shrink_grid)

    def score_point(steps, nbr, q, shrink):
        """Mean CRPS over tune seeds for every (topk, combiner) at this point."""
        beta, wsv = shrink
        key = (int(steps), int(nbr), int(q), float(beta), float(wsv))
        if key in cache:
            return cache[key]
        per: dict[tuple, list[float]] = collections.defaultdict(list)
        for sd in tune_seeds:
            members = train_djgp_members(
                setting, sd, K=K, steps=steps, n_neighbors=nbr, n_val=n_val,
                noise_mode=noise_mode, rho_mode=rho_mode, device=device, q_override=q,
                gate_mode=gate_mode, reproject=reproject, w_family=w_family,
                w_signal_var=wsv, beta_kl_R=beta)
            for tk in topks:
                for cb in combiners:
                    per[(tk, cb)].append(_djgp_row(members, tk, cb)["crps"])
        cache[key] = {tc: float(np.mean(v)) for tc, v in per.items()}
        return cache[key]

    def best_crps(steps, nbr, q, shrink):
        return min(score_point(steps, nbr, q, shrink).values())

    q_true = int(SETTING_PRESETS[setting]["Q"])
    q0 = q_true if q_true in q_grid else q_grid[0]
    steps0 = steps_grid[len(steps_grid) // 2]
    n_preset = int(SETTING_PRESETS[setting]["n"])
    nbr0 = n_preset if n_preset in nbr_grid else nbr_grid[0]
    shrink0 = shrink_grid[0]

    # coordinate descent by importance: 1) Q  2) shrink pair  3) steps  4) n_neighbors
    bq = min(q_grid, key=lambda q: best_crps(steps0, nbr0, q, shrink0))
    bshr = (shrink0 if len(shrink_grid) == 1
            else min(shrink_grid, key=lambda p: best_crps(steps0, nbr0, bq, p)))
    bs = min(steps_grid, key=lambda s: best_crps(s, nbr0, bq, bshr))
    bn = min(nbr_grid, key=lambda n: best_crps(bs, n, bq, bshr))
    # then the init combination (topk x combiner) at the winning hyperparameters
    pt = score_point(bs, bn, bq, bshr)
    (btk, bcb) = min(pt, key=pt.get)

    grid = [{"steps": s, "n_neighbors": n, "q": q, "beta_kl_R": b, "w_signal_var": w,
             "best_topk": int(min(d, key=d.get)[0]),
             "best_combiner": min(d, key=d.get)[1],
             "best_mean_crps": float(min(d.values())),
             "by_aggregation": {f"top{tk}_{cb}": c for (tk, cb), c in d.items()}}
            for (s, n, q, b, w), d in cache.items()]
    grid.sort(key=lambda r: r["best_mean_crps"])
    return {"steps": int(bs), "n_neighbors": int(bn), "q": int(bq),
            "beta_kl_R": float(bshr[0]), "w_signal_var": float(bshr[1]),
            "topk": int(btk), "combiner": bcb,
            "noise_mode": noise_mode, "rho_mode": rho_mode, "gate_mode": str(gate_mode),
            "tune_crps": float(pt[(btk, bcb)]),
            "stage_order": ["Q", "shrink(beta,wsv)", "steps", "n_neighbors",
                            "init_combo(topk,combiner)"],
            "defaults": {"q0": int(q0), "steps0": int(steps0), "nbr0": int(nbr0),
                         "shrink0": list(shrink0)},
            "n_points_trained": len(cache),
            "grid": grid}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _meta(setting):
    cfg = SETTING_PRESETS[setting]
    return {"family": cfg["family"], "expansion": cfg["expansion"],
            "ambient_D": int(cfg["H"]), "latent_z": int(cfg["d"])}


def main():
    p = argparse.ArgumentParser(description="6-method synthetic benchmark (L2/LH grid).")
    p.add_argument("--settings", default="all",
                   help="'all' = L2 + standard (easy) LH; 'all_hard' = L2 + lh_hard "
                        "(genuine jump regression, the default LH for our runs); "
                        "or a comma list of preset names")
    p.add_argument("--methods", default="dgp,jgp,jgp_pca,jgp_sir,djgp")
    p.add_argument("--seeds", default="0,1,2,3,4", help="reporting seeds")
    p.add_argument("--tune_seeds", default="101,102", help="extra seeds for DJGP selection")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--steps_grid", default="200,300,400")
    p.add_argument("--nbr_grid", default="25,35")
    p.add_argument("--q_offsets", default="0,1",
                   help="DJGP reduced-dim Q grid as offsets from the true latent dim "
                        "(clamped >=1), e.g. '-1,0,1'")
    p.add_argument("--q_candidates", default="",
                   help="FIXED Q candidate set, e.g. '3,5,7', identical for every config. "
                        "When set, OVERRIDES --q_offsets and does NOT use the true latent dim "
                        "(avoids leaking oracle dimension info). This is the correct protocol.")
    p.add_argument("--n_val", type=int, default=150)
    p.add_argument("--gates", default="intercept,spatial",
                   help="DJGP gate variants to run: intercept (rho only) and/or spatial "
                        "(linear Gauss-Hermite boundary gate). Each emits top1/2/4 rows.")
    p.add_argument("--shrink_grid", default="0.02:1.0",
                   help="(beta_kl_R : w_signal_var) PAIR candidates for the DJGP "
                        "coordinate descent, comma-separated, e.g. "
                        "'0.02:1.0,1.0:0.1,5.0:0.02'. Controls the free-W trust region "
                        "(effective KL rent = beta/sigma_w^2). Single pair = fixed.")
    p.add_argument("--dgp_epochs", type=int, default=300)
    p.add_argument("--out", default="experiments/synthetic/_exp_paper_grid_6methods/metrics.csv")
    p.add_argument("--resume", action="store_true",
                   help="skip (setting,seed,method) rows already status=ok in --out")
    p.add_argument("--reproject", action="store_true",
                   help="DJGP: re-project local inducing inputs by the current metric W each "
                        "step/predict (default OFF = frozen W0 placement). Re-tunes with the "
                        "flag on, so steps/n_neighbors/Q/init are selected for the reprojected model.")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.settings == "all":
        settings = USER_GRID_PRESET_NAMES
    elif a.settings == "all_hard":
        # Standard grid but with the (easy) LH family replaced by lh_hard -- the
        # genuine jump-regression regime. This is the default "LH" for our runs.
        settings = [n for n in USER_GRID_PRESET_NAMES
                    if SETTING_PRESETS[n].get("family") != "LH"] + list(LH_HARD_PRESET_NAMES)
    else:
        settings = [s for s in a.settings.split(",") if s]
    methods = [m for m in a.methods.split(",") if m]
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    tune_seeds = [int(s) for s in a.tune_seeds.split(",") if s != ""]
    steps_grid = [int(s) for s in a.steps_grid.split(",") if s != ""]
    nbr_grid = [int(s) for s in a.nbr_grid.split(",") if s != ""]
    q_offsets = [int(s) for s in a.q_offsets.split(",") if s != ""]
    shrink_grid = [tuple(float(x) for x in s.split(":")) for s in a.shrink_grid.split(",") if s != ""]
    gates = [g for g in a.gates.split(",") if g]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sel_dir = out.parent / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple] = set()
    if a.resume and out.exists():
        with out.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(r["setting"], str(r["seed"]), r["method"]) for r in rows if r.get("status") == "ok"}
        print(f"[resume] loaded {len(rows)} rows, {len(done)} ok -> will skip those", flush=True)

    def is_done(setting, sd, method):
        return (setting, str(sd), method) in done

    def flush():
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ROW_KEYS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ROW_KEYS})

    gate_abbr = {"intercept": "int", "spatial": "spa", "free": "free"}

    def _gate_cfg(gate):
        """--gates entry -> (gate_mode, w_family). 'free' = intercept gate + the free
        entrywise-W family (W = W0 + per-entry-GP deviation); rows djgp_free_top{k}."""
        return ("intercept", "free") if gate == "free" else (gate, "structured")

    djgp_topks = (1, 2, 4)
    t_start = time.perf_counter()
    for setting in settings:
        meta = _meta(setting)

        # --- DJGP tuning, per gate variant ---
        sel_by_gate: dict[str, dict] = {}
        if "djgp" in methods:
            q_true = int(SETTING_PRESETS[setting]["Q"])  # metadata only (recorded, not used to center grid when q_candidates set)
            if a.q_candidates:
                # Fixed candidate set (e.g. 3,5,7) -- config-agnostic, no oracle leak.
                q_grid = sorted({max(1, int(q)) for q in a.q_candidates.split(",") if q != ""})
            else:
                q_grid = sorted({max(1, q_true + off) for off in q_offsets})
            for gate in gates:
                ga = gate_abbr.get(gate, gate)
                # skip tuning if every reporting row for this gate is already done
                if all(is_done(setting, sd, f"djgp_{ga}_top{tk}") for sd in seeds for tk in djgp_topks):
                    print(f"[tune djgp/{gate}] {setting}: all rows resumed, skip", flush=True)
                    continue
                print(f"[tune djgp/{gate}] {setting}  q_grid={q_grid}", flush=True)
                try:
                    gm, wf = _gate_cfg(gate)
                    sel = tune_djgp(setting, meta["family"], tune_seeds, K=a.K,
                                    steps_grid=steps_grid, nbr_grid=nbr_grid, q_grid=q_grid,
                                    n_val=a.n_val, device=device, gate_mode=gm,
                                    reproject=a.reproject, w_family=wf, shrink_grid=shrink_grid)
                    sel_by_gate[gate] = sel
                    print(f"  [{gate}] steps={sel['steps']} nbr={sel['n_neighbors']} Q={sel['q']} "
                          f"shrink=({sel['beta_kl_R']},{sel['w_signal_var']}) "
                          f"top{sel['topk']} {sel['combiner']} (tune_crps={sel['tune_crps']:.4f})",
                          flush=True)
                    rec = {"setting": setting, **meta, "q_true": q_true, "gate_mode": gate,
                           "tune_seeds": tune_seeds, "K": a.K, "noise_mode": sel["noise_mode"],
                           "rho_mode": sel["rho_mode"],
                           "init_recipe": ["pls", "pca", "pca_scaled", "sir", "rand0..."],
                           "selected": {k: sel[k] for k in
                                        ("steps", "n_neighbors", "q", "beta_kl_R",
                                         "w_signal_var", "topk", "combiner", "tune_crps")},
                           "grid": sel["grid"]}
                    (sel_dir / f"{setting}__{gate}.json").write_text(json.dumps(rec, indent=2),
                                                                     encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"  djgp/{gate} tuning ERROR: {type(e).__name__}: {e}", flush=True)

        for sd in seeds:
            base = {"setting": setting, **meta, "seed": sd}
            print(f"[{setting} seed={sd}]", flush=True)

            # --- baselines ---
            need_baseline = any(m != "djgp" and not is_done(setting, sd, m) for m in methods)
            if need_baseline:
                try:
                    Xtr, ytr, Xte, yte, cfg = _prep_xy(setting, sd, device)
                except Exception as e:  # noqa: BLE001
                    print(f"  data ERROR: {e}", flush=True)
                    continue
                Q, nbr_cfg = int(cfg["Q"]), int(cfg["n"])
                runners = {
                    "dgp": lambda: _run_dgp(Xtr, ytr, Xte, yte, device, a.dgp_epochs),
                    "jgp": lambda: _jumpgp_predict(Xtr, ytr, Xte, yte, nbr_cfg),
                    "jgp_pca": lambda: _run_jgp_pca(Xtr, ytr, Xte, yte, Q, nbr_cfg),
                    "jgp_sir": lambda: _run_jgp_sir(Xtr, ytr, Xte, yte, Q, nbr_cfg),
                    "ngboost": lambda: _run_ngboost(Xtr, ytr, Xte, yte, sd),
                    "bart": lambda: _run_bart(Xtr, ytr, Xte, yte, sd),
                }
                for m in methods:
                    if m == "djgp" or is_done(setting, sd, m):
                        continue
                    row = dict(base, method=m)
                    try:
                        row.update(runners[m]())
                        row["status"] = "ok"
                        print(f"  {m:8s} rmse={row['rmse']:.3f} crps={row['crps']:.3f} "
                              f"cov90={row['cov90']:.3f} w90={row['width90']:.3f} "
                              f"t={row['wall_sec']:.1f}s", flush=True)
                    except Exception as e:  # noqa: BLE001
                        row["status"] = "error"
                        row["error"] = f"{type(e).__name__}: {e}"
                        print(f"  {m:8s} ERROR: {row['error']}", flush=True)
                    rows.append(row)
                    flush()

            # --- djgp: each gate variant, top1/2/4 breakdown (members trained once) ---
            if "djgp" in methods:
                for gate, sel in sel_by_gate.items():
                    ga = gate_abbr.get(gate, gate)
                    if all(is_done(setting, sd, f"djgp_{ga}_top{tk}") for tk in djgp_topks):
                        continue
                    try:
                        gm, wf = _gate_cfg(gate)
                        members = train_djgp_members(
                            setting, sd, K=a.K, steps=sel["steps"],
                            n_neighbors=sel["n_neighbors"], n_val=a.n_val,
                            noise_mode=sel["noise_mode"], rho_mode=sel["rho_mode"],
                            device=device, q_override=sel["q"], gate_mode=gm,
                            reproject=a.reproject, w_family=wf,
                            w_signal_var=sel.get("w_signal_var", 1.0),
                            beta_kl_R=sel.get("beta_kl_R", 0.02))
                        for tk in djgp_topks:
                            method = f"djgp_{ga}_top{tk}"
                            if is_done(setting, sd, method):
                                continue
                            m = _djgp_row(members, tk, sel["combiner"])
                            sel_inits = m.pop("selected_inits", "")
                            row = dict(base, method=method, **m)
                            row["wall_sec"] = float(members["wall"])
                            row.update({"djgp_gate": gate, "djgp_topk": tk,
                                        "djgp_combiner": sel["combiner"], "djgp_steps": sel["steps"],
                                        "djgp_n_neighbors": sel["n_neighbors"], "djgp_q": sel["q"],
                                        "djgp_selected_inits": sel_inits,
                                        "djgp_wsv": sel.get("w_signal_var", 1.0),
                                        "djgp_beta": sel.get("beta_kl_R", 0.02), "status": "ok"})
                            rows.append(row)
                            print(f"  {method:14s} rmse={row['rmse']:.3f} crps={row['crps']:.3f} "
                                  f"cov90={row['cov90']:.3f} w90={row['width90']:.3f} "
                                  f"inits=[{sel_inits}]", flush=True)
                            flush()
                    except Exception as e:  # noqa: BLE001
                        for tk in djgp_topks:
                            method = f"djgp_{ga}_top{tk}"
                            if not is_done(setting, sd, method):
                                rows.append(dict(base, method=method, status="error",
                                                 error=f"{type(e).__name__}: {e}", djgp_gate=gate))
                        print(f"  djgp/{gate} ERROR: {type(e).__name__}: {e}", flush=True)
                        flush()

    _write_aggregate(rows, out.parent / "aggregate.csv")
    print(f"\nDone in {time.perf_counter() - t_start:.0f}s -> {out}", flush=True)
    print(f"Aggregate -> {out.parent / 'aggregate.csv'}", flush=True)
    _print_aggregate(rows)


def _write_aggregate(rows, path):
    """Mean +/- std over reporting seeds per (setting, method)."""
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["setting"], r["method"])
        meta[key] = {k: r.get(k, "") for k in ("family", "expansion", "ambient_D", "latent_z")}
        for k in ("rmse", "crps", "cov90", "width90", "wall_sec"):
            agg[key][k].append(float(r[k]))
    keys = ["setting", "method", "family", "expansion", "ambient_D", "latent_z", "n_seeds",
            "rmse_mean", "rmse_std", "crps_mean", "crps_std", "cov90_mean", "cov90_std",
            "width90_mean", "width90_std", "wall_sec_mean", "wall_sec_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for (setting, method) in sorted(agg):
            g = agg[(setting, method)]
            row = {"setting": setting, "method": method, **meta[(setting, method)],
                   "n_seeds": len(g["rmse"])}
            for k in ("rmse", "crps", "cov90", "width90", "wall_sec"):
                v = np.array(g[k])
                row[f"{k}_mean"] = float(v.mean())
                row[f"{k}_std"] = float(v.std())
            w.writerow(row)


def _print_aggregate(rows):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["setting"], r["method"])
        for k in ("rmse", "crps", "cov90", "width90", "wall_sec"):
            agg[key][k].append(float(r[k]))
    print(f"\n{'setting':22s} {'method':8s}  {'rmse':>14} {'crps':>14} {'cov90':>12} "
          f"{'width90':>14} {'time(s)':>12}")
    for (setting, method) in sorted(agg):
        g = agg[(setting, method)]
        def ms(k):  # noqa: E306
            v = np.array(g[k]); return f"{v.mean():.3f}±{v.std():.3f}"
        print(f"{setting:22s} {method:8s}  {ms('rmse'):>14} {ms('crps'):>14} "
              f"{ms('cov90'):>12} {ms('width90'):>14} {ms('wall_sec'):>12}")


if __name__ == "__main__":
    main()
