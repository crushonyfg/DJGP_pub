"""free-W (entrywise-GP W, no collapse) init-ensemble on UCI: tune member RANKING on 2
held-out seeds, then report top1 / top2 / top4 robust-combines on the 5 eval seeds.

Motivated by the LH finding that different init basins specialize (pls/pca -> hard bin,
sir -> dual bin) under free-W training: does combining basins pay on real data?

Pool: K=8 inits = pls, pca, SIR, 5 random-Stiefel (SIR added to the standard UCI pool
per the LH basin finding; indices 0=pls 1=pca 2=sir 3..7=rand). Per topn in {1,2,4} the
(step, member-set) with the best mean tune-RMSE on seeds 5,6 is frozen, then evaluated
on seeds 0-4 (same robust combiner / metrics as the v1 benchmark).

Run (shard by dataset x regime):
  .venv/bin/python experiments/uci/_exp_uci_free_ensemble.py --regime 1000 \
      --datasets Wine --device cuda:0 --out_tag w1000
Aggregate: .venv/bin/python experiments/uci/_exp_uci_free_ensemble.py --report
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.uci._exp_uci_v1v3 as engine  # noqa: E402
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import _projection_W0  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "_exp_uci_free_ensemble"
NPZ = {"5000": "experiments/uci/_train5000_8seeds/data",
       "1000": "experiments/uci/_train1000_8seeds/data"}
TOPNS = (1, 2, 4)
INIT_LABELS = ["pls", "pca", "sir"] + [f"rand{r}" for r in range(5)]


def _build_inits_with_sir(Xtr, ytr_s, Q, K):
    inits = [_projection_W0(Xtr, ytr_s, Q, "pls", seed=777),
             _projection_W0(Xtr, ytr_s, Q, "pca", seed=777)]
    try:
        from experiments.synthetic._exp_paper_grid_6methods import _sir_W0
        inits.append(_sir_W0(Xtr, ytr_s, Q))
    except Exception:  # noqa: BLE001  (SIR can fail; random fill covers)
        pass
    r = 0
    while len(inits) < K:
        rr = np.random.RandomState(20260000 + r)
        W = rr.randn(Q, Xtr.shape[1]).astype(np.float64)
        W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        inits.append(W)
        r += 1
    return inits[:K]


engine._build_inits = _build_inits_with_sir  # tune()/evaluate() resolve it via module global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["5000", "1000"], default="1000")
    ap.add_argument("--datasets", default="Wine,Appliances,Parkinsons")
    ap.add_argument("--eval_seeds", default="0,1,2,3,4")
    ap.add_argument("--tune_seeds", default="5,6")
    ap.add_argument("--maxsteps", type=int, default=600)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_inducing_R", type=int, default=150)
    ap.add_argument("--shrink_grid", default="0.02:1.0",
                    help="(beta_kl_R : w_signal_var) pair candidates, comma-separated, "
                         "e.g. '0.02:1.0,1.0:0.1,5.0:0.02'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate", default="intercept", choices=["intercept", "spatial"])
    ap.add_argument("--out_tag", default="main")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        _report()
        return
    device = torch.device(a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    eval_seeds = [int(s) for s in a.eval_seeds.split(",") if s != ""]
    tune_seeds = [int(s) for s in a.tune_seeds.split(",") if s != ""]
    datasets = [ds for ds in a.datasets.split(",") if ds in engine.UCI]
    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / f"rows_{a.out_tag}.csv"
    choice_json = OUT_DIR / f"choice_{a.out_tag}.json"
    choices = []
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["regime", "dataset", "variant", "seed", "step",
                                           "members", "rmse", "crps", "cov90", "sec"])
        w.writeheader()
        for ds in datasets:
            dcfg = dict(engine.UCI[ds])
            if a.regime == "1000":
                dcfg["size"] = 1000
            npz_dir = NPZ[a.regime]
            noise = dcfg["noise"]
            t0 = time.perf_counter()
            # Tune once per (beta_kl_R, w_signal_var) shrink pair; the per-topn best
            # (shrink, step, member-set) is then chosen across ALL pairs by tune RMSE.
            shrink_pairs = [tuple(float(x) for x in s.split(":"))
                            for s in a.shrink_grid.split(",") if s]
            grid = []
            for pair in shrink_pairs:
                _st, _mem, _rm, g = engine.tune(
                    dcfg, npz_dir, a.gate, noise, "near1", tune_seeds,
                    a.maxsteps, a.eval_interval, a.K, device, a.n_inducing_R,
                    free=True, shrink=pair)
                for row in g:
                    row["shrink"] = pair
                grid.extend(g)
                print(f"[{ds}] tune shrink={pair} done ({time.perf_counter()-t0:.0f}s)", flush=True)
            for topn in TOPNS:
                rows_g = [g for g in grid if g["topn"] == topn]
                best = min(rows_g, key=lambda g: g["tune_rmse"])
                labels = [INIT_LABELS[i] for i in best["members"]]
                choices.append(dict(regime=a.regime, dataset=ds, topn=topn,
                                    step=best["step"], members=best["members"],
                                    shrink=list(best["shrink"]),
                                    labels=labels, tune_rmse=best["tune_rmse"]))
                choice_json.write_text(json.dumps(choices, indent=2), encoding="utf-8")
                print(f"[{ds}] top{topn}: step={best['step']} members={best['members']} "
                      f"({','.join(labels)}) shrink={best['shrink']} "
                      f"tuneRMSE={best['tune_rmse']:.3f}", flush=True)
                for sd in eval_seeds:
                    ev = engine.evaluate(dcfg, npz_dir, a.gate, noise, "near1",
                                         best["members"], best["step"], sd, a.K, device,
                                         a.n_inducing_R, free=True, shrink=best["shrink"])
                    w.writerow(dict(regime=a.regime, dataset=ds, variant=f"free_top{topn}",
                                    seed=sd, step=best["step"],
                                    members=" ".join(map(str, best["members"])),
                                    rmse=ev["rmse"], crps=ev["crps"], cov90=ev["cov90"],
                                    sec=ev["sec"]))
                    fh.flush()
                    print(f"  [{ds}/top{topn} seed {sd}] rmse={ev['rmse']:.3f} "
                          f"crps={ev['crps']:.3f} cov={ev['cov90']:.3f}", flush=True)
    print("wrote", out_csv)


def _report():
    rows = []
    for f in sorted(OUT_DIR.glob("rows_*.csv")):
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        print("no rows in", OUT_DIR)
        return
    choices = []
    for f in sorted(OUT_DIR.glob("choice_*.json")):
        choices.extend(json.loads(f.read_text()))
    for regime in ("1000", "5000"):
        rr = [r for r in rows if r["regime"] == regime]
        if not rr:
            continue
        print(f"\n=== regime train{regime} ===")
        for ds in ("Wine", "Appliances", "Parkinsons"):
            dr = [r for r in rr if r["dataset"] == ds]
            if not dr:
                continue
            print(f"\n  {ds}")
            print(f"  {'variant':10s} {'step':>5} {'members':>16} "
                  f"{'RMSE':>14} {'CRPS':>14} {'cov90':>14}")
            for topn in TOPNS:
                vr = [r for r in dr if r["variant"] == f"free_top{topn}"]
                if not vr:
                    continue
                ch = next((c for c in choices if c["dataset"] == ds and c["regime"] == regime
                           and c["topn"] == topn), None)
                lab = ",".join(ch["labels"]) if ch else vr[0]["members"]
                rm = [float(r["rmse"]) for r in vr]
                cr = [float(r["crps"]) for r in vr]
                cv = [float(r["cov90"]) for r in vr]
                print(f"  free_top{topn:<2d} {vr[0]['step']:>5} {lab:>16} "
                      f"{np.mean(rm):>7.3f}±{np.std(rm):<6.3f} "
                      f"{np.mean(cr):>7.3f}±{np.std(cr):<6.3f} "
                      f"{np.mean(cv):>7.3f}±{np.std(cv):<6.3f}")
    print()


if __name__ == "__main__":
    main()
