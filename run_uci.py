"""Reproduce the DJGP UCI benchmark at the reportable sizes.

Datasets x sizes (6 configs):
  * Wine Quality       train=1000/test=200 and train=5000/test=500 (random-row)
  * Appliances Energy  train=1000/test=200 and train=5000/test=500 (random-row)
  * Parkinsons Telemon train=1000/test=200 and train=4000/test=500 (subject-grouped)

The train/test splits are derived data (not committed). Generate them once with
``python scripts/prepare_uci_data.py`` before running this script.

Methods
-------
  djgp      : Deep Jump Gaussian Process. Hyperparameters are loaded from the frozen
              configuration in docs/benchmark_configs/uci_*.json -- no tuning here.
  jgp       : Jump GP on the (train-standardized) features
  jgp_pca   : Jump GP on a PCA projection
  jgp_sir   : Jump GP on a sliced-inverse-regression projection
  ngboost   : NGBoost Normal regressor
  bart      : Bayesian Additive Regression Trees
  dgp       : Deep Gaussian Process (skipped if gpytorch is unavailable)

Metrics: RMSE / CRPS / cov90, per seed + mean over seeds.

Usage
-----
Quick check:
    python run_uci.py --datasets Wine --sizes 1000 --seeds 0
Full reproduction (all 6 configs, all methods, 25 seeds; long):
    python run_uci.py
Results -> --out (default results/uci/metrics.csv) + aggregate print.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.uci._exp_uci_free_ensemble as fe  # noqa: E402  (installs the SIR init pool)
import experiments.uci._exp_uci_v1v3 as engine  # noqa: E402
from experiments.synthetic._exp_paper_grid_6methods import (  # noqa: E402
    _jumpgp_predict,
    _run_bart,
    _run_dgp,
    _run_jgp_pca,
    _run_jgp_sir,
    _run_ngboost,
)

CFG_DIR = REPO / "docs" / "benchmark_configs"
NPZ = {"1000": "experiments/uci/_train1000_8seeds/data",
       "5000": "experiments/uci/_train5000_8seeds/data"}
TRAIN_SIZE = {("Wine", "1000"): 1000, ("Wine", "5000"): 5000,
              ("Appliances", "1000"): 1000, ("Appliances", "5000"): 5000,
              ("Parkinsons", "1000"): 1000, ("Parkinsons", "5000"): 4000}
METHODS = ("djgp", "jgp", "jgp_pca", "jgp_sir", "ngboost", "bart", "dgp")
KEYS = ["dataset", "train_size", "seed", "method", "rmse", "crps", "cov90", "wall_sec",
        "status", "error"]


def _load_npz(ds: str, regime: str, seed: int):
    dcfg = engine.UCI[ds]
    size = TRAIN_SIZE[(ds, regime)]
    d = np.load(Path(NPZ[regime]) / f"uci_{dcfg['slug']}_train{size}_seed{seed}.npz",
                allow_pickle=True)
    Xtr = np.asarray(d["X_train"], np.float64)
    Xte = np.asarray(d["X_test"], np.float64)
    ym = float(np.asarray(d["y_mean"]).ravel()[0])
    ys = float(np.asarray(d["y_scale"]).ravel()[0])
    ytr = np.asarray(d["y_train_std"], np.float64).ravel() * ys + ym
    yte = np.asarray(d["y_test"], np.float64).ravel()
    return Xtr, ytr, Xte, yte


def run_djgp_from_json(ds: str, regime: str, seed: int, device) -> dict:
    """Evaluate the frozen DJGP configuration (no tuning)."""
    size = TRAIN_SIZE[(ds, regime)]
    cfg = json.loads((CFG_DIR / f"uci_{ds.lower()}_train{size}.json").read_text())
    sel = cfg["selected_params"]
    dcfg = dict(engine.UCI[ds])
    dcfg["size"] = size
    ev = engine.evaluate(dcfg, NPZ[regime], "intercept", dcfg["noise"], "near1",
                         sel["members"], int(sel["step"]), seed, 8, device, 150,
                         free=True,
                         shrink=(float(sel["beta_kl_R"]), float(sel["w_signal_var"])))
    return {"rmse": ev["rmse"], "crps": ev["crps"], "cov90": ev["cov90"],
            "wall_sec": ev["sec"]}


def main():
    ap = argparse.ArgumentParser(description="DJGP UCI benchmark (frozen configs)")
    ap.add_argument("--datasets", default="Wine,Appliances,Parkinsons")
    ap.add_argument("--sizes", default="1000,5000", help="training regimes: 1000 and/or 5000")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--dgp_epochs", type=int, default=300)
    ap.add_argument("--out", default="results/uci/metrics.csv")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [d for d in a.datasets.split(",") if d in engine.UCI]
    regimes = [s for s in a.sizes.split(",") if s in ("1000", "5000")]
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    methods = [m for m in a.methods.split(",") if m in METHODS]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Preflight: the .npz splits are derived data, not committed. Point the user at the
    # one-time generation script if they are missing.
    for regime in regimes:
        for ds in datasets:
            size = TRAIN_SIZE[(ds, regime)]
            slug = engine.UCI[ds]["slug"]
            need = seeds and not (Path(NPZ[regime]) / f"uci_{slug}_train{size}_seed{seeds[0]}.npz").exists()
            if need:
                sys.exit(
                    f"UCI split not found: {NPZ[regime]}/uci_{slug}_train{size}_seed{seeds[0]}.npz\n"
                    "These splits are derived data and are NOT committed. Generate them once:\n"
                    f"    python scripts/prepare_uci_data.py --seeds {max(seeds) + 1}\n"
                    "then re-run this script.")
    rows: list[dict] = []

    def flush():
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=KEYS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in KEYS})

    t0 = time.perf_counter()
    for ds in datasets:
        Q, nbr = int(engine.UCI[ds]["Q"]), int(engine.UCI[ds]["n"])
        for regime in regimes:
            size = TRAIN_SIZE[(ds, regime)]
            for sd in seeds:
                base = {"dataset": ds, "train_size": size, "seed": sd}
                print(f"[{ds} train{size} seed={sd}]", flush=True)
                need_baseline = any(m != "djgp" for m in methods)
                if need_baseline:
                    Xtr, ytr, Xte, yte = _load_npz(ds, regime, sd)
                runners = {
                    "djgp": lambda: run_djgp_from_json(ds, regime, sd, device),
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
                        r = runners[m]()
                        row.update({k: r[k] for k in ("rmse", "crps", "cov90", "wall_sec")})
                        row["status"] = "ok"
                        print(f"  {row['method']:11s} rmse={row['rmse']:.3f} "
                              f"crps={row['crps']:.3f} cov90={row['cov90']:.3f} "
                              f"t={row['wall_sec']:.1f}s", flush=True)
                    except Exception as e:  # noqa: BLE001
                        row.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
                        print(f"  {row['method']:11s} ERROR: {row['error']}", flush=True)
                    rows.append(row)
                    flush()
    # aggregate print
    print(f"\nDone in {time.perf_counter() - t0:.0f}s -> {out}\n")
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("status") == "ok":
            for k in ("rmse", "crps", "cov90"):
                agg[(r["dataset"], r["train_size"], r["method"])][k].append(float(r[k]))
    print(f"{'dataset':11s} {'size':>5} {'method':12s} {'rmse':>10} {'crps':>10} {'cov90':>7}")
    for (ds, size, m), g in sorted(agg.items()):
        print(f"{ds:11s} {size:>5} {m:12s} {np.mean(g['rmse']):>10.3f} "
              f"{np.mean(g['crps']):>10.3f} {np.mean(g['cov90']):>7.3f}")


if __name__ == "__main__":
    main()
