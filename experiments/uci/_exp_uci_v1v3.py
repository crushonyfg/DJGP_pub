"""UCI train=5000/4000 v1(intercept) vs v3(spatial) with seed-frozen selection.

Protocol per (dataset, gate_mode, noise_mode, rho_mode):
  - TUNE on held-out seeds (default 6,7): train K members (fixed init recipe) to
    --maxsteps, recording each member's analytic-direct test prediction every
    --eval_interval. Rank members by MEAN test RMSE over the tuning seeds; for each
    (step, top-n) robust-combine the top-n; pick the (step, members) with lowest mean
    test RMSE. (Tuning seeds are disjoint from the eval seed -> legitimate; no
    validation selector.)
  - EVAL on --eval_seed (default 0): train ONLY the frozen members to the frozen step,
    robust-combine, report RMSE/CRPS/cov90 + wall-clock. No selector.

Datasets (pre-exported via export_smalltrain_dmgp_splits.py):
  Wine 5000 / Appliances 5000 (random-row, test 500); Parkinsons 4000 (subject-grouped
  frac0.7, test 500).  X already train-standardized in the npz; y standardized inside.

Records the chosen (step, members) + metrics to <out>.csv and <out>.json so results
are reproducible (the tuned params are logged).

Run (repo root, jumpGP env), e.g.:
    conda run -n jumpGP python experiments/uci/_exp_uci_v1v3.py \
        --datasets Wine,Appliances,Parkinsons --gates intercept,spatial \
        --tune_seeds 6,7 --eval_seed 0 --maxsteps 600 --eval_interval 100 --K 8 \
        --rho near1 --npz_dir experiments/uci/_train5000_8seeds/data \
        --out experiments/uci/_exp_uci_v1v3/primary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from djgp.projections.structured_metric_lmjgp import (  # noqa: E402
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp,
    train_analytic_structured_metric_lmjgp,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import _neighbor_stack, _projection_W0  # noqa: E402
from experiments.synthetic._exp_u_ensemble_5seed import _metrics_robust  # noqa: E402

# per-dataset: slug, train size, Q, kNN, default noise (doc section 11)
UCI = {
    "Wine": dict(slug="wine_quality", size=5000, Q=5, n=25, noise="data_driven"),
    "Parkinsons": dict(slug="parkinsons_telemonitoring", size=4000, Q=5, n=25, noise="data_driven"),
    "Appliances": dict(slug="appliances_energy_prediction", size=5000, Q=5, n=35, noise="fixed"),
}


def _rho_extra(mode):
    if mode == "near1":
        return dict(rho_init=0.97, rho_prior_strength=10.0, min_rho_mean=0.9, min_rho_penalty_weight=50.0)
    if mode == "relaxed":
        return dict(rho_init=0.8, rho_prior_strength=0.5, min_rho_mean=0.3, min_rho_penalty_weight=5.0)
    return dict(rho_init=0.8, rho_prior_strength=2.0, min_rho_mean=0.55, min_rho_penalty_weight=50.0)


def _noise_extra(mode):
    if mode == "data_driven":
        return dict(obs_noise_prior_mode="data_driven", obs_noise_data_driven_frac=1.0,
                    obs_noise_prior_std=0.3, obs_noise_prior_weight=3.0, obs_noise_max=1.5,
                    init_noise_at_prior=True)
    return {}


def _build_inits(Xtr, ytr_s, Q, K):
    inits = [_projection_W0(Xtr, ytr_s, Q, "pls", seed=777),
             _projection_W0(Xtr, ytr_s, Q, "pca", seed=777)]
    r = 0
    while len(inits) < K:
        rr = np.random.RandomState(20260000 + r)
        W = rr.randn(Q, Xtr.shape[1]).astype(np.float64); W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        inits.append(W); r += 1
    return inits[:K]


def _prep(dcfg, npz_dir, seed, device):
    path = Path(npz_dir) / f"uci_{dcfg['slug']}_train{dcfg['size']}_seed{seed}.npz"
    d = np.load(path, allow_pickle=True)
    Xtr = np.asarray(d["X_train"], np.float64)               # train-standardized
    Xte = np.asarray(d["X_test"], np.float64)
    ytr_s = np.asarray(d["y_train_std"], np.float64).ravel()  # standardized
    yte = np.asarray(d["y_test"], np.float64).ravel()         # original scale
    ym = float(np.asarray(d["y_mean"]).ravel()[0]); ys = float(np.asarray(d["y_scale"]).ravel()[0])
    Q, n = int(dcfg["Q"]), int(dcfg["n"])
    dt = torch.float64
    Xtr_t = torch.as_tensor(Xtr, device=device, dtype=dt)
    Xte_t = torch.as_tensor(Xte, device=device, dtype=dt)
    ytr_s_t = torch.as_tensor(ytr_s, device=device, dtype=dt)
    Xn, yn, _ = _neighbor_stack(Xte_t, Xtr_t, ytr_s_t, n_neighbors=n)
    if yn.dim() == 3:
        yn = yn.squeeze(-1)
    return dict(Xtr=Xtr, ytr_s=ytr_s, Xte=Xte_t, Xn=Xn, yn=yn, yte=yte, ym=ym, ys=ys, Q=Q)


def _cfg(T, Q, steps, gate_mode, noise_mode, rho_mode, eval_interval, n_inducing_R, collapse=False, free=False, shrink=(0.02, 1.0)):
    extra = {**_noise_extra(noise_mode), **_rho_extra(rho_mode)}
    # Sparse eta-GP inducing: defaulting to all T test anchors makes the metric-GP
    # Kmm an O(T^3) Cholesky every step (the dominant cost at large test sets). A
    # fixed sparse set (<< T) keeps the global metric expressive but ~ (T/M)^3 faster.
    if int(n_inducing_R) > 0 and int(n_inducing_R) < int(T):
        extra["n_inducing_R"] = int(n_inducing_R)
    if collapse:
        extra["collapse_u"] = True
    if free:
        extra["w_family"] = "free"
        extra["beta_kl_R"] = float(shrink[0])
        extra["w_signal_var"] = float(shrink[1])
    return AnalyticStructuredMetricLMJGPConfig(
        steps=steps, lr=0.01, batch_anchors=T, trace_target=float(Q), trace_normalize=True,
        eval_interval=eval_interval, restore_state="final", gate_mode=gate_mode, seed=211, **extra)


def _train_member(d, W0, steps, gate_mode, noise_mode, rho_mode, eval_interval, seed_off, record,
                  n_inducing_R, collapse=False, free=False, shrink=(0.02, 1.0)):
    Q, Xte = d["Q"], d["Xte"]
    snaps = {}

    def cb(step, wm, ls, md):
        if not record and step != steps:
            return {}
        ns = SimpleNamespace(local_signal_var=1.0, **{k: v for k, v in ls.items()})
        mu, var, _ = predict_analytic_structured_metric_lmjgp(Xte, ns, trace_target=float(Q), trace_normalize=True)
        snaps[int(step)] = (mu.detach().cpu().numpy() * d["ys"] + d["ym"],
                            var.detach().sqrt().cpu().numpy() * abs(d["ys"]))
        return {}

    C, mu_u0 = build_fixed_local_inducing_from_w0(Xte, d["Xn"], d["yn"], W0, m_inducing=10)
    train_analytic_structured_metric_lmjgp(
        Xte, d["Xn"], d["yn"], C, init_W=W0, init_mu_u=mu_u0,
        config=_cfg(Xte.shape[0], Q, steps, gate_mode, noise_mode, rho_mode,
                    eval_interval if record else steps + 1, n_inducing_R, collapse=collapse, free=free,
                    shrink=shrink),
        eval_callback=cb)
    return snaps


def _stack_at(d, member_snaps, step):
    mus, sds = [], []
    for snaps in member_snaps:
        st = step if step in snaps else max(s for s in snaps if s <= step)
        mu, sd = snaps[st]
        mus.append(mu); sds.append(sd)
    return np.stack(mus), np.stack(sds)


def tune(dcfg, npz_dir, gate, noise, rho, tune_seeds, maxsteps, eval_interval, K, device, n_inducing_R,
         collapse=False, free=False, shrink=(0.02, 1.0)):
    """Train K members on each tuning seed; rank by MEAN per-member test RMSE; pick
    best (step, top-n). Returns (best_step, best_members, grid_rows)."""
    per_seed = []  # list over seeds of (d, member_snaps)
    for sd in tune_seeds:
        d = _prep(dcfg, npz_dir, sd, device)
        inits = _build_inits(d["Xtr"], d["ytr_s"], d["Q"], K)
        snaps = [_train_member(d, torch.as_tensor(W, device=device, dtype=torch.float64),
                               maxsteps, gate, noise, rho, eval_interval, 17 * ki, True, n_inducing_R,
                               collapse=collapse, free=free, shrink=shrink)
                 for ki, W in enumerate(inits)]
        per_seed.append((d, snaps))
    steps_grid = sorted(per_seed[0][1][0].keys())
    best = (None, None, 1e18)
    grid = []
    for st in steps_grid:
        # per-member mean test RMSE over tuning seeds
        per_member_rmse = np.zeros(K)
        stacks = []
        for d, snaps in per_seed:
            M, S = _stack_at(d, snaps, st)
            per_member_rmse += np.sqrt(((d["yte"][None, :] - M) ** 2).mean(axis=1))
            stacks.append((d, M, S))
        per_member_rmse /= len(per_seed)
        for n in (1, 2, 4):
            sel = np.argsort(per_member_rmse)[:n]
            mean_rmse = np.mean([_metrics_robust(d["yte"], M, S, sel)[0] for d, M, S in stacks])
            grid.append(dict(step=int(st), topn=n, members=sorted(sel.tolist()), tune_rmse=float(mean_rmse)))
            if mean_rmse < best[2]:
                best = (int(st), sorted(sel.tolist()), float(mean_rmse))
    return best[0], best[1], best[2], grid


def _crps(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-8); z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def evaluate(dcfg, npz_dir, gate, noise, rho, members, step, eval_seed, K, device, n_inducing_R,
             collapse=False, free=False, shrink=(0.02, 1.0)):
    d = _prep(dcfg, npz_dir, eval_seed, device)
    inits = _build_inits(d["Xtr"], d["ytr_s"], d["Q"], K)
    t0 = time.perf_counter()
    snaps = [_train_member(d, torch.as_tensor(inits[ki], device=device, dtype=torch.float64),
                           step, gate, noise, rho, step + 1, 17 * ki, False, n_inducing_R,
                           collapse=collapse, free=free, shrink=shrink) for ki in members]
    M, S = _stack_at(d, snaps, step)
    rmse, _, cov = _metrics_robust(d["yte"], M, S, np.arange(len(members)))
    mu = np.median(M, axis=0)
    between = 1.4826 * np.median(np.abs(M - mu[None]), axis=0)
    sigma = np.sqrt(np.median(S ** 2, axis=0) + between ** 2)
    crps = _crps(d["yte"], mu, sigma)
    return dict(rmse=float(rmse), crps=float(crps), cov90=float(cov), sec=float(time.perf_counter() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Wine,Appliances,Parkinsons")
    ap.add_argument("--gates", default="intercept,spatial")
    ap.add_argument("--tune_seeds", default="6,7")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--maxsteps", type=int, default=600)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_inducing_R", type=int, default=150,
                    help="sparse eta-GP inducing count (<<test); the dominant cost lever")
    ap.add_argument("--noise", default="", help="override per-dataset default noise")
    ap.add_argument("--rho", default="near1")
    ap.add_argument("--npz_dir", default="experiments/uci/_train5000_8seeds/data")
    ap.add_argument("--out", default="experiments/uci/_exp_uci_v1v3/primary")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [s for s in a.datasets.split(",") if s]
    gates = [s for s in a.gates.split(",") if s]
    tune_seeds = [int(s) for s in a.tune_seeds.split(",") if s != ""]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        dcfg = UCI[ds]
        noise = a.noise or dcfg["noise"]
        for gate in gates:
            t0 = time.perf_counter()
            step, members, tune_rmse, grid = tune(dcfg, a.npz_dir, gate, noise, a.rho,
                                                  tune_seeds, a.maxsteps, a.eval_interval, a.K, device,
                                                  a.n_inducing_R)
            ev = evaluate(dcfg, a.npz_dir, gate, noise, a.rho, members, step, a.eval_seed, a.K, device,
                          a.n_inducing_R)
            row = dict(dataset=ds, gate=gate, noise=noise, rho=a.rho, size=dcfg["size"],
                       Q=dcfg["Q"], n_neighbors=dcfg["n"], n_inducing_R=a.n_inducing_R,
                       tune_seeds=tune_seeds, eval_seed=a.eval_seed,
                       chosen_step=step, chosen_members=members, tune_mean_rmse=tune_rmse,
                       eval_rmse=ev["rmse"], eval_crps=ev["crps"], eval_cov90=ev["cov90"],
                       eval_sec=ev["sec"], wall_sec=float(time.perf_counter() - t0))
            rows.append(row)
            print(f"[{ds}/{gate} noise={noise} rho={a.rho}] step={step} members={members} "
                  f"tuneRMSE={tune_rmse:.3f} -> seed{a.eval_seed} "
                  f"rmse={ev['rmse']:.3f} crps={ev['crps']:.3f} cov={ev['cov90']:.3f} ({ev['sec']:.0f}s)",
                  flush=True)
            out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            import csv
            with out.with_suffix(".csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
                for rr in rows:
                    w.writerow({k: (v if not isinstance(v, list) else " ".join(map(str, v))) for k, v in rr.items()})
    print("\n=== summary (eval seed", a.eval_seed, ") ===")
    for r in rows:
        print(f"{r['dataset']:11s} {r['gate']:9s} noise={r['noise']:11s} rho={r['rho']:7s} "
              f"step={r['chosen_step']:4d} mem={r['chosen_members']} "
              f"RMSE={r['eval_rmse']:.3f} CRPS={r['eval_crps']:.3f} cov={r['eval_cov90']:.3f}")


if __name__ == "__main__":
    main()
