"""Reproduce the DJGP synthetic benchmark (two latent families L2 and LH).

Settings (20): the L2 (K=2) and LH (K=4,5,7) latent families, each lifted to an observed
dimension D by one of five dimensionality-expansion techniques
(rff, poly, ae, randproj, manifold). train=1000 / test=200, generated on the fly (seeded).

Methods
-------
  djgp      : Deep Jump Gaussian Process. Hyperparameters are loaded from the frozen
              configuration in docs/benchmark_configs/<setting>.json -- no tuning here.
  jgp       : Jump GP on the observed features
  jgp_pca   : Jump GP on a PCA projection
  jgp_sir   : Jump GP on a sliced-inverse-regression projection
  ngboost   : NGBoost Normal regressor
  bart      : Bayesian Additive Regression Trees
  dgp       : Deep Gaussian Process (skipped if gpytorch is unavailable)

Metrics: RMSE / CRPS / cov90 (90% prediction-interval coverage), per seed + mean over seeds.

Usage
-----
Quick check (one setting, seed 0):
    python run_synthetic.py --settings l2_d20_rff --seeds 0
Full reproduction (all 20 settings, all methods, 25 seeds; long):
    python run_synthetic.py
Select subsets:
    python run_synthetic.py --settings lh_d30z5_rff,lh_d30z5_poly --methods djgp,jgp
Results -> --out (default results/synthetic/metrics.csv) + aggregate.csv beside it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.synthetic._exp_paper_grid_6methods import (  # noqa: E402
    ROW_KEYS,
    _djgp_row,
    _jumpgp_predict,
    _meta,
    _prep_xy,
    _run_bart,
    _run_dgp,
    _run_jgp_pca,
    _run_jgp_sir,
    _run_ngboost,
    _write_aggregate,
    train_djgp_members,
)

CFG_DIR = REPO / "docs" / "benchmark_configs"
FAMILIES = ("l2_d20", "lh_d20z4", "lh_d30z5", "lh_d50z7")
EXPANSIONS = ("rff", "poly", "ae", "randproj", "manifold")
ALL_SETTINGS = [f"{fam}_{exp}" for fam in FAMILIES for exp in EXPANSIONS]
METHODS = ("djgp", "jgp", "jgp_pca", "jgp_sir", "ngboost", "bart", "dgp")


def run_djgp_from_json(setting: str, seed: int, device) -> dict:
    """Evaluate the frozen DJGP configuration from docs/benchmark_configs (no tuning)."""
    cfg = json.loads((CFG_DIR / f"{setting}.json").read_text())
    sel = cfg["selected_params"]
    members = train_djgp_members(
        setting, seed, K=8, steps=int(sel["steps"]), n_neighbors=int(sel["n_neighbors"]),
        n_val=150, noise_mode="data_driven", rho_mode="near1", device=device,
        q_override=int(sel["q"]), gate_mode="intercept", w_family="free",
        w_signal_var=float(sel["w_signal_var"]), beta_kl_R=float(sel["beta_kl_R"]))
    row = _djgp_row(members, int(sel["topk"]), sel["combiner"])
    row.pop("selected_inits", None)
    row["wall_sec"] = float(members["wall"])
    row.update({"djgp_gate": "free", "djgp_topk": sel["topk"], "djgp_combiner": sel["combiner"],
                "djgp_steps": sel["steps"], "djgp_n_neighbors": sel["n_neighbors"],
                "djgp_q": sel["q"], "djgp_wsv": sel["w_signal_var"], "djgp_beta": sel["beta_kl_R"]})
    return row


def main():
    ap = argparse.ArgumentParser(description="DJGP synthetic benchmark (frozen configs)")
    ap.add_argument("--settings", default=",".join(ALL_SETTINGS))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--dgp_epochs", type=int, default=300)
    ap.add_argument("--out", default="results/synthetic/metrics.csv")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = [s for s in a.settings.split(",") if s in ALL_SETTINGS]
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    methods = [m for m in a.methods.split(",") if m in METHODS]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def flush():
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ROW_KEYS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ROW_KEYS})

    t0 = time.perf_counter()
    for setting in settings:
        meta = _meta(setting)
        for sd in seeds:
            base = {"setting": setting, **meta, "seed": sd}
            print(f"[{setting} seed={sd}]", flush=True)
            need_baseline = any(m != "djgp" for m in methods)
            if need_baseline:
                Xtr, ytr, Xte, yte, cfg = _prep_xy(setting, sd, device)
                Q, nbr = int(cfg["Q"]), int(cfg["n"])
            runners = {
                "djgp": lambda: run_djgp_from_json(setting, sd, device),
                "jgp": lambda: _jumpgp_predict(Xtr, ytr, Xte, yte, nbr),
                "jgp_pca": lambda: _run_jgp_pca(Xtr, ytr, Xte, yte, Q, nbr),
                "jgp_sir": lambda: _run_jgp_sir(Xtr, ytr, Xte, yte, Q, nbr),
                "ngboost": lambda: _run_ngboost(Xtr, ytr, Xte, yte, sd),
                "bart": lambda: _run_bart(Xtr, ytr, Xte, yte, sd),
                "dgp": lambda: _run_dgp(Xtr, ytr, Xte, yte, device, a.dgp_epochs),
            }
            for m in methods:
                row = dict(base, method=("djgp" if m == "djgp" else m))
                try:
                    row.update(runners[m]())
                    row["status"] = "ok"
                    print(f"  {row['method']:11s} rmse={row['rmse']:.3f} crps={row['crps']:.3f} "
                          f"cov90={row['cov90']:.3f} t={row['wall_sec']:.1f}s", flush=True)
                except Exception as e:  # noqa: BLE001
                    row.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
                    print(f"  {row['method']:11s} ERROR: {row['error']}", flush=True)
                rows.append(row)
                flush()
    _write_aggregate(rows, out.parent / "aggregate.csv")
    print(f"\nDone in {time.perf_counter() - t0:.0f}s -> {out}")
    print(f"Aggregate -> {out.parent / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
