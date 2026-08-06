"""5-seed U-ensemble experiment for analytic ISM-LMJGP.

Per (setting, seed):
  * train K analytic members from different subspace inits (PLS/PCA/PCA-scaled +
    random Stiefel), full-batch.
  * rank members by a TRAINING-derived validation score: local-GP-in-projected-
    space CRPS on held-out training anchors (ranks subspace quality; no test
    peeking). Same ranking is used for both prediction heads.
  * predict on the FULL test set with each member, two heads:
      - analytic-direct inlier predictive  N(mu_f, var_f+noise)
      - sample-W CEM (sample W from q(R), refit local CEM, aggregate)
  * mix the top-{1,2,4} members (exact Gaussian-mixture moments + mixture CRPS).

Outputs a CSV with rmse/crps/cov90 per (setting, seed, head, topk).
lmjgp_baseline is run separately for the external reference.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from djgp.projections.structured_metric_lmjgp import (  # noqa: E402
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp,
    sample_structured_metric_w,
    structured_metric_eta_moments_from_R,
    structured_metric_w_moments_from_R,
    train_analytic_structured_metric_lmjgp,
)
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    run_uncertain_w_self_cem,
    uncertain_w_cem_predict_from_labels,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _neighbor_stack,
    _projection_W0,
    _run_meanw_cem_init,
    _set_seed,
)

from scipy.stats import norm


def _crps_vec(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _mixture_crps(y, mus, sigmas, n=2000):
    """Exact-ish CRPS of an equal-weight Gaussian mixture via quantile MC of the
    closed-form is overkill; use the energy-score form sampled finely instead.
    mus,sigmas: [K,T]. Returns per-point CRPS [T]."""
    K, T = mus.shape
    # sample n draws from the mixture for each test point, average pinball is slow;
    # use the standard mixture-CRPS closed form:
    #   CRPS = (1/K) sum_k E|X_k - y| - (1/2K^2) sum_{k,l} E|X_k - X_l|
    # with E|N(m,s^2) - c| = s*(2*phi(z)+z*(2*Phi(z)-1)), z=(c-m)/s  (for a const c)
    # and E|N(a)-N(b)| with independent = sqrt(sa^2+sb^2)*A((ma-mb)/sqrt(.))
    def eabs(m, s):  # E|N(m,s) - 0|  -> shape broadcast; here used as E|X-y|
        z = m / np.maximum(s, 1e-12)
        return s * (2 * norm.pdf(z) + z * (2 * norm.cdf(z) - 1))
    term1 = np.zeros(T)
    for k in range(K):
        term1 += eabs(mus[k] - y, sigmas[k])
    term1 /= K
    term2 = np.zeros(T)
    for k in range(K):
        for l in range(K):
            s = np.sqrt(sigmas[k] ** 2 + sigmas[l] ** 2)
            term2 += eabs(mus[k] - mus[l], s)
    term2 /= (K * K)
    return term1 - 0.5 * term2


def _metrics_from_members(y, M, S, sel):
    """Mixture metrics over selected members. M,S: [K,T] mean/std. sel: indices."""
    Ms, Ss = M[sel], S[sel]
    mu = Ms.mean(0)
    crps = float(np.mean(_mixture_crps(y, Ms, Ss)))
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    # mixture coverage: P(|y-mu_mix|...) -> use empirical mixture CDF for 90% PI
    # central 90% interval of the mixture via its CDF inverse (numeric)
    lo, hi = _mixture_pi(Ms, Ss, 0.05), _mixture_pi(Ms, Ss, 0.95)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    return rmse, crps, cov


def _metrics_robust(y, M, S, sel):
    """Robust combiner: per-point median location + MAD-based between-member spread,
    so a single catastrophic member cannot drag the ensemble. M,S: [K,T]."""
    Ms, Ss = M[sel], S[sel]
    mu = np.median(Ms, axis=0)
    between = 1.4826 * np.median(np.abs(Ms - mu[None, :]), axis=0)   # robust std of member means
    within = np.sqrt(np.median(Ss ** 2, axis=0))                    # robust within-member sigma
    sigma = np.sqrt(within ** 2 + between ** 2)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    crps = float(np.mean(_crps_vec(y, mu, sigma)))
    z = (y - mu) / np.maximum(sigma, 1e-8)
    cov = float(np.mean(np.abs(z) <= 1.6448536269514722))
    return rmse, crps, cov


def _mixture_pi(Ms, Ss, q):
    """q-quantile of equal-weight Gaussian mixture per point (bisection). [K,T]->[T]."""
    K, T = Ms.shape
    lo = (Ms - 6 * Ss).min(0)
    hi = (Ms + 6 * Ss).max(0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = norm.cdf((mid[None, :] - Ms) / np.maximum(Ss, 1e-12)).mean(0)
        go = cdf < q
        lo = np.where(go, mid, lo)
        hi = np.where(go, hi, mid)
    return 0.5 * (lo + hi)


def _val_subspace_crps(result, X_val, Xn_val, yn_val, noise_var, Q, trace_target, jitter=1e-5):
    """Local-GP-in-projected-space CRPS at held-out training anchors."""
    if getattr(result, "V_mu", None) is not None:
        # w_family="free": read the trained entrywise-GP W out at the val anchors
        # (result.U/R_mu are the untrained structured leftovers -- do NOT use them).
        from djgp.projections.structured_metric_lmjgp import free_w_moments_from_result
        W_mu, _ = free_w_moments_from_result(result, X_val, jitter=jitter)
    else:
        W_mu, _, _, _, _ = structured_metric_w_moments_from_R(
            result.U, X_val, result.inducing_X, result.R_mu, result.R_log_std,
            lengthscale=float(result.w_lengthscale), signal_var=float(result.w_signal_var),
            include_conditional_residual=bool(result.include_conditional_residual),
            trace_target=float(trace_target), trace_normalize=True, jitter=jitter)
    za = torch.einsum("vqd,vd->vq", W_mu, X_val)
    zn = torch.einsum("vqd,vnd->vnq", W_mu, Xn_val)
    d2nn = (zn.unsqueeze(2) - zn.unsqueeze(1)).pow(2).sum(-1)
    l2 = d2nn.median().clamp_min(1e-3)
    nv = float(noise_var)
    n = zn.shape[1]
    Knn = torch.exp(-0.5 * d2nn / l2) + (nv + jitter) * torch.eye(n, device=zn.device, dtype=zn.dtype)
    kan = torch.exp(-0.5 * (za.unsqueeze(1) - zn).pow(2).sum(-1) / l2)  # [V,n]
    Kinv = torch.linalg.solve(Knn, torch.eye(n, device=zn.device, dtype=zn.dtype).expand_as(Knn))
    alpha = torch.einsum("vij,vj->vi", Kinv, yn_val)
    mu = torch.einsum("vi,vi->v", kan, alpha)
    kKk = torch.einsum("vi,vij,vj->v", kan, Kinv, kan)
    var = (1.0 - kKk).clamp_min(1e-6) + nv
    mu = mu.cpu().numpy(); sig = var.sqrt().cpu().numpy(); y = yn_val.new_tensor(0)  # placeholder
    return mu, sig


def _self_cem_cfg():
    return SelfCEMConfig(
        cem_updates=1,
        hyperopt=FixedLabelHyperoptConfig(steps=2, lr=0.04, batched=True,
                                          min_noise_var=0.1 ** 2, max_noise_var=10.0,
                                          min_signal_var=1e-4, max_signal_var=100.0,
                                          min_lengthscale=1e-3, max_lengthscale=20.0),
        gate=FixedLabelGateConfig(steps=3, lr=0.05, basis="linear", intercept_only=True,
                                  l2=1e-3, max_norm=8.0, gh_points=20),
        outlier_mode="background_normal", outlier_sigma_mult=2.5, background_scale_mult=3.0,
        min_inliers=3, max_inlier_frac=0.90, refit_after_update=True)


def _zero_slopes(g):
    o = g.clone(); o[:, 1:] = 0.0; return o


def _repeat_first(x, n):
    return x.unsqueeze(0).expand(int(n), *x.shape).reshape(int(n) * x.shape[0], *x.shape[1:]).contiguous()


def _sample_w_predict_raw(result, X_anchor, X_neighbors, y_neighbors, cem_state,
                          trace_target, S=3, seed=0, min_inliers=3):
    eta_mu, eta_var = structured_metric_eta_moments_from_R(
        X_anchor, result.inducing_X, result.R_mu, result.R_log_std,
        lengthscale=float(result.w_lengthscale), signal_var=float(result.w_signal_var),
        include_conditional_residual=bool(result.include_conditional_residual))
    try:
        gen = torch.Generator(device=X_anchor.device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed) + 911)
    W = sample_structured_metric_w(result.U, eta_mu, eta_var.clamp_min(1e-12).sqrt().log(),
                                   n_samples=S, trace_target=float(trace_target),
                                   trace_normalize=True, generator=gen)
    Ssz, T, Q, D = W.shape
    Wf = W.reshape(Ssz * T, Q, D).contiguous(); zero = torch.zeros_like(Wf)
    Xa = _repeat_first(X_anchor, Ssz); Xn = _repeat_first(X_neighbors, Ssz); yf = _repeat_first(y_neighbors, Ssz)
    st = run_uncertain_w_self_cem(Xa, Xn, yf, _repeat_first(cem_state["labels"], Ssz), mode="anchor",
                                  W_mu_anchor=Wf, W_var_anchor=zero,
                                  init_lengthscale=_repeat_first(cem_state["lengthscale"], Ssz),
                                  init_signal_var=_repeat_first(cem_state["signal_var"], Ssz),
                                  init_noise_var=_repeat_first(cem_state["noise_var"], Ssz),
                                  gate_init=_repeat_first(_zero_slopes(cem_state["gate"]), Ssz),
                                  config=_self_cem_cfg())
    out = uncertain_w_cem_predict_from_labels(Xa, Xn, yf, st["labels"], mode="anchor",
                                              W_mu_anchor=Wf, W_var_anchor=zero,
                                              lengthscale=st["lengthscale"], signal_var=st["signal_var"],
                                              noise_var=st["noise_var"], mean_const=st["mean_const"],
                                              correction_weight=0.0, min_inliers=int(min_inliers))
    mu = out.mu.detach().reshape(Ssz, T); var = out.var.detach().reshape(Ssz, T).clamp_min(1e-12)
    mu_m = mu.mean(0); tv = (var + mu.pow(2)).mean(0) - mu_m.pow(2)
    return mu_m, tv.clamp_min(1e-12)


def _rho_kwargs(rho_mode: str) -> dict:
    if rho_mode == "near1":
        return dict(rho_init=0.97, rho_prior_strength=10.0, min_rho_mean=0.9, min_rho_penalty_weight=50.0)
    if rho_mode == "free":
        return dict(rho_init=0.8, rho_prior_strength=0.1, min_rho_mean=0.05, min_rho_penalty_weight=0.0)
    return dict(rho_init=0.8, rho_prior_strength=2.0, min_rho_mean=0.55, min_rho_penalty_weight=50.0)


def _noise_kwargs(noise_mode, nd_frac, nd_prior_std, nd_prior_weight):
    if noise_mode == "data_driven":
        return dict(obs_noise_prior_mode="data_driven", obs_noise_data_driven_frac=nd_frac,
                    obs_noise_prior_std=nd_prior_std, obs_noise_prior_weight=nd_prior_weight,
                    obs_noise_max=1.5, init_noise_at_prior=True)
    return {}


def run_setting_seed(setting, seed, K, steps, n_neighbors, n_val, device, early_stop=False, es_interval=10,
                     base_inits=("pls", "pca", "pca_scaled"), select_by="valcrps",
                     noise_mode="fixed", nd_frac=1.0, nd_prior_std=0.3, nd_prior_weight=3.0, rho_mode="default",
                     m_inducing=10, n_inducing_R=-1, batch_anchors=-1, gate_mode="intercept"):
    extra_cfg = {**_rho_kwargs(rho_mode), **_noise_kwargs(noise_mode, nd_frac, nd_prior_std, nd_prior_weight)}
    extra_cfg["gate_mode"] = str(gate_mode)  # intercept-only rho vs spatial GH boundary gate
    if int(n_inducing_R) > 0:
        extra_cfg["n_inducing_R"] = int(n_inducing_R)  # ablation 1: eta-GP inducing points < #test anchors
    cfg = dict(SETTING_PRESETS[setting]); _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    sx = StandardScaler(); Xtr = sx.fit_transform(data["X_train"]).astype(np.float64)
    Xte = sx.transform(data["X_test"]).astype(np.float64)
    ytr = np.asarray(data["y_train"], np.float64).reshape(-1)
    yte = np.asarray(data["y_test"], np.float64).reshape(-1)
    ym, ys = float(ytr.mean()), float(ytr.std() if ytr.std() > 1e-8 else 1.0)
    ytr_s = (ytr - ym) / ys
    Q = int(cfg["Q"])
    Xtr_t = torch.as_tensor(Xtr, device=device, dtype=torch.float64)
    Xte_t = torch.as_tensor(Xte, device=device, dtype=torch.float64)
    ytr_s_t = torch.as_tensor(ytr_s, device=device, dtype=torch.float64)
    T = Xte_t.shape[0]

    # held-out training validation anchors
    rng = np.random.RandomState(1234 + seed)
    val_idx = rng.choice(Xtr.shape[0], size=min(n_val, Xtr.shape[0]), replace=False)
    Xval_t = Xtr_t[val_idx]
    # neighbours of val anchors come from training minus self (approx: full train kNN, drop first)
    Xn_val, yn_val, _ = _neighbor_stack(Xval_t, Xtr_t, ytr_s_t, n_neighbors=n_neighbors + 1)
    Xn_val, yn_val = Xn_val[:, 1:, :], yn_val[:, 1:]
    if yn_val.dim() == 3:
        yn_val = yn_val.squeeze(-1)
    yval_true = ytr_s[val_idx]

    # test neighbours + CEM init (shared across members)
    Xn, yn, _ = _neighbor_stack(Xte_t, Xtr_t, ytr_s_t, n_neighbors=n_neighbors)
    if yn.dim() == 3:
        yn = yn.squeeze(-1)

    methods = list(base_inits)
    inits = []
    for mth in methods[:K]:
        inits.append(_projection_W0(Xtr, ytr_s, Q, mth, seed=seed + 513))
    r = 0
    while len(inits) < K:
        rr = np.random.RandomState(seed + 1000 + r)
        W = rr.randn(Q, Xtr.shape[1]).astype(np.float64); W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        inits.append(W); r += 1

    W0_pls = torch.as_tensor(inits[0], device=device, dtype=torch.float64)
    cem_state, _ = _run_meanw_cem_init(Xte_t, Xn, yn, W0_pls)
    cem_state["gate"] = _zero_slopes(cem_state["gate"])

    an_mu, an_sd, sw_mu, sw_sd, valscore = [], [], [], [], []
    for ki, W0_np in enumerate(inits):
        W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
        C, mu_u0 = build_fixed_local_inducing_from_w0(Xte_t, Xn, yn, W0, m_inducing=int(m_inducing))
        es_cb = None
        restore = "final"
        ev_int = 5
        if early_stop:
            restore = "best_direct"
            ev_int = es_interval

            def es_cb(step, _wm, ls, _md):  # noqa: E306
                proxy = SimpleNamespace(
                    U=ls["U"], R_mu=ls["R_mu"], R_log_std=ls["R_log_std"],
                    inducing_X=ls["inducing_X"],
                    w_lengthscale=float(ls["w_lengthscale"]),
                    w_signal_var=float(ls["w_signal_var"]),
                    include_conditional_residual=bool(float(ls["include_conditional_residual"]) > 0.5),
                )
                nv = float(ls["log_noise_var"].exp().median().item())
                vmu, _ = _val_subspace_crps(proxy, Xval_t, Xn_val, yn_val, nv, Q, float(Q))
                # key must be 'direct_eval_rmse' for restore_state=best_direct; use VAL rmse
                return {"direct_eval_rmse": float(np.sqrt(np.mean((yval_true - vmu) ** 2)))}

        result = train_analytic_structured_metric_lmjgp(
            Xte_t, Xn, yn, C, init_W=W0, init_mu_u=mu_u0,
            config=AnalyticStructuredMetricLMJGPConfig(
                steps=steps, lr=0.01,
                batch_anchors=(int(batch_anchors) if int(batch_anchors) > 0 else T),  # ablation 2: stochastic composite
                trace_target=float(Q), trace_normalize=True,
                eval_interval=ev_int, restore_state=restore,
                seed=seed + 211 + 17 * ki, **extra_cfg),
            eval_callback=es_cb)
        # analytic-direct test prediction
        mu_a, var_a, _ = predict_analytic_structured_metric_lmjgp(Xte_t, result, trace_target=float(Q),
                                                                  trace_normalize=True)
        an_mu.append(mu_a.detach().cpu().numpy() * ys + ym)
        an_sd.append(var_a.detach().sqrt().cpu().numpy() * abs(ys))
        # sample-W CEM test prediction
        mu_s, var_s = _sample_w_predict_raw(result, Xte_t, Xn, yn, cem_state, float(Q), S=3, seed=seed + ki)
        sw_mu.append(mu_s.detach().cpu().numpy() * ys + ym)
        sw_sd.append(var_s.detach().sqrt().cpu().numpy() * abs(ys))
        # val subspace score (training)
        noise_med = float(result.log_noise_var.exp().median().item())
        vmu, vsig = _val_subspace_crps(result, Xval_t, Xn_val, yn_val, noise_med, Q, float(Q))
        valscore.append(float(np.mean(_crps_vec(yval_true, vmu, vsig))))

    an_mu = np.stack(an_mu); an_sd = np.stack(an_sd)
    sw_mu = np.stack(sw_mu); sw_sd = np.stack(sw_sd)
    # unsupervised head-disagreement per member (RMSE between heads / y_std); a
    # near-perfect proxy (corr~0.999) for analytic-head trustworthiness.
    disagree = np.sqrt(((an_mu - sw_mu) ** 2).mean(axis=1)) / (abs(ys) + 1e-12)
    if select_by == "disagree":
        order = np.argsort(disagree)
    else:
        order = np.argsort(np.asarray(valscore))  # lower CRPS = better

    rows = []
    for topk in (1, 2, 4):
        sel = order[:topk]
        for head, M, S in (("analytic", an_mu, an_sd), ("sampleW", sw_mu, sw_sd)):
            for comb, fn in (("mixture", _metrics_from_members), ("robust", _metrics_robust)):
                rmse, crps, cov = fn(yte, M, S, sel)
                rows.append(dict(setting=setting, seed=seed, gate_mode=str(gate_mode), head=head,
                                 topk=topk, combiner=comb, rmse=rmse, crps=crps, cov90=cov))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--n_neighbors", type=int, default=25)
    p.add_argument("--n_val", type=int, default=150)
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--es_interval", type=int, default=10)
    p.add_argument("--base_inits", default="pls,pca,pca_scaled",
                   help="comma list of projection inits; rest of K filled with random Stiefel")
    p.add_argument("--select_by", default="valcrps", choices=("valcrps", "disagree"),
                   help="member selection: supervised training-val CRPS, or unsupervised head-disagreement")
    p.add_argument("--noise_mode", default="fixed", choices=("fixed", "data_driven"))
    p.add_argument("--nd_frac", type=float, default=1.0)
    p.add_argument("--nd_prior_std", type=float, default=0.3)
    p.add_argument("--nd_prior_weight", type=float, default=3.0)
    p.add_argument("--rho_mode", default="default", choices=("default", "near1", "free"))
    p.add_argument("--gate_mode", default="intercept", choices=("intercept", "spatial"),
                   help="intercept-only rho (default) vs spatial GH boundary gate sigmoid(nu0+w^T W(x-xa))")
    p.add_argument("--m_inducing", type=int, default=10, help="ablation 3: local inducing points per anchor")
    p.add_argument("--n_inducing_R", type=int, default=-1,
                   help="ablation 1: eta-GP inducing points (<=#test). -1 => all test anchors (default)")
    p.add_argument("--batch_anchors", type=int, default=-1,
                   help="ablation 2: minibatch anchors per step. -1 => full-batch (#test)")
    p.add_argument("--out", default="experiments/synthetic/_exp_u_ensemble_5seed/metrics.csv")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    settings = [x for x in a.settings.split(",") if x]
    allrows = []
    t0 = time.perf_counter()
    for s in settings:
        for sd in seeds:
            rows = run_setting_seed(s, sd, a.K, a.steps, a.n_neighbors, a.n_val, device,
                                    early_stop=a.early_stop, es_interval=a.es_interval,
                                    base_inits=tuple(x for x in a.base_inits.split(",") if x),
                                    select_by=a.select_by, noise_mode=a.noise_mode, nd_frac=a.nd_frac,
                                    nd_prior_std=a.nd_prior_std, nd_prior_weight=a.nd_prior_weight,
                                    rho_mode=a.rho_mode, m_inducing=a.m_inducing,
                                    n_inducing_R=a.n_inducing_R, batch_anchors=a.batch_anchors,
                                    gate_mode=a.gate_mode)
            allrows.extend(rows)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
            print(f"done {s} seed{sd} ({time.perf_counter()-t0:.0f}s)", flush=True)
    # aggregate
    import collections
    agg = collections.defaultdict(list)
    for r in allrows:
        agg[(r["setting"], r["head"], r.get("combiner", "mixture"), r["topk"])].append(r)
    print("\nsetting           head      comb     topk   rmse(mean±sd)        crps(mean±sd)        cov90")
    for key in sorted(agg):
        g = agg[key]
        def ms(k): v = np.array([x[k] for x in g]); return v.mean(), v.std()
        rm, rs = ms("rmse"); cm, cs = ms("crps"); vm, vs = ms("cov90")
        print(f"{key[0]:17s} {key[1]:8s} {key[2]:8s} {key[3]:4d}   {rm:8.3f}±{rs:6.3f}   {cm:8.3f}±{cs:6.3f}   {vm:.3f}±{vs:.3f}")


if __name__ == "__main__":
    main()
