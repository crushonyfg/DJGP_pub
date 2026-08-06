"""
UCI benchmark: by default runs DKL-SVGP + XGBoost bootstrap + LMJGP (neighborhood VI).

Optional (off by default): JumpGP variants (--jgp_raw, --jgp_pca, --jgp_sir), GPyTorch DeepGP
(--deepgp). Use --no_lmjgp to skip LMJGP.

Each method returns a list::

    [rmse, mean_crps, wall_sec, coverage_90, mean_width_90, coverage_95, mean_width_95]

Gaussian predictive intervals use Normal(mu, sigma^2) marginals (same as CRPS). JumpGP / DeepGP
entries append NaNs for coverage/width when test-set mu/sigma are not exposed by the runner.

How to run (from repository root; conda env: jumpGP):

  conda activate jumpGP
  cd <path-to-DJGP-repo>

  # Default: LMJGP + DKL + XGB (10 seeds, Wine Quality)
  python experiments/uci/compare_uci_baselines.py --num_exp 10

  # Multiple UCI datasets (comma-separated)
  python experiments/uci/compare_uci_baselines.py --num_exp 10 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"

  # Add JumpGP on raw / PCA / SIR (enable any subset)
  python experiments/uci/compare_uci_baselines.py --num_exp 10 --jgp_raw --jgp_pca --jgp_sir

  # Add GPyTorch DeepGP (DeepGP_test)
  python experiments/uci/compare_uci_baselines.py --num_exp 10 --deepgp

  # DKL + XGB only (smoke test; skip LMJGP)
  python experiments/uci/compare_uci_baselines.py --num_exp 1 --no_lmjgp

  # Custom bootstrap count, DKL epochs, and output pickle path
  python experiments/uci/compare_uci_baselines.py --num_exp 10 --xgb_boots 30 --dkl_epochs 80 --out uci_results.pkl

Agents: when you change this file's CLI or default behavior, update this docstring so these
commands stay accurate.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.uci.uci_data import load_uci_split, sir_reduction  # noqa: E402
from jumpgp import JumpGP  # noqa: E402
from shared.jumpgp_runner import compute_metrics, find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import nan_gaussian_uq_tail, pack_gaussian_uq_list  # noqa: E402
from djgp.variational import predict_vi, train_vi  # noqa: E402

try:
    from shared.deepgp import DeepGP, train_model  # noqa: E402
    from gpytorch.mlls import DeepApproximateMLL, VariationalELBO  # noqa: E402
except ImportError:
    DeepGP = None


def _run_jgp_raw(X_train_np, y_train_np, X_test_np, y_test_np):
    t0 = time.time()
    JGP = JumpGP(X_train_np, y_train_np, X_test_np, L=1, M=20, mode="CEM", bVerbose=False)
    JGP.fit()
    rmse, mean_crps = JGP.metrics(y_test_np)
    return nan_gaussian_uq_tail(float(rmse), float(mean_crps), time.time() - t0)


def _run_jgp_pca(X_train_np, y_train_np, X_test_np, y_test_np, K):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_np)
    Xte = scaler.transform(X_test_np)
    pca = PCA(n_components=K)
    X_train_pca = pca.fit_transform(Xtr)
    X_test_pca = pca.transform(Xte)
    return _run_jgp_raw(X_train_pca, y_train_np, X_test_pca, y_test_np)


def _run_jgp_sir(X_train_np, y_train_np, X_test_np, y_test_np, H, K):
    X_train_sir, sir_vecs, scaler = sir_reduction(X_train_np, y_train_np, H=H, K=K)
    X_test_sir = scaler.transform(X_test_np) @ sir_vecs
    return _run_jgp_raw(X_train_sir, y_train_np, X_test_sir, y_test_np)


def _run_djgp_block(
    X_train,
    Y_train,
    X_test,
    Y_test,
    args_ns: Namespace,
    device: torch.device,
):
    """Same hyperparameters as `new_dataset.py` DJGP / LMJGP block."""
    N_test, D = X_test.shape
    T = N_test
    Q = args_ns.Q
    m1 = args_ns.m1
    m2 = args_ns.m2
    n = args_ns.n
    num_steps = args_ns.num_steps
    MC_num = args_ns.MC_num

    t0 = time.time()
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions = []
    for i in range(T):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})

    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    u_params = []
    for _ in range(T):
        u_params.append(
            {
                "U_logit": torch.zeros(1, device=device, requires_grad=True),
                "mu_u": torch.randn(m1, device=device, requires_grad=True),
                "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
                "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
                "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
                "omega": torch.randn(Q + 1, device=device, requires_grad=True),
            }
        )
    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std
    hyperparams = {
        "Z": Z,
        "X_test": X_test,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }

    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=args_ns.lr,
        num_steps=num_steps,
        log_interval=50,
    )
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=MC_num)
    sigmas = torch.sqrt(var_pred)
    rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)
    elapsed = time.time() - t0
    y_np = Y_test.detach().cpu().numpy()
    mu_np = mu_pred.detach().cpu().numpy()
    sig_np = sigmas.detach().cpu().numpy()
    return pack_gaussian_uq_list(
        float(rmse.item()),
        float(mean_crps.item()),
        elapsed,
        y_np,
        mu_np,
        sig_np,
    )


def _run_deepgp(X_train, Y_train, X_test, Y_test, K, device):
    if DeepGP is None:
        return None
    batch_size, hidden_dim, lr = 128, 10, 0.01
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)
    model = DeepGP(X_train.shape, hidden_dim).to(device)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_train.shape[-2]))
    args = Namespace(
        epochs=100,
        num_epochs=300,
        batch_size=128,
        hidden_dim=10,
        lr=0.01,
        print_freq=10,
        eval=True,
        patience=30,
        clip_grad=1.0,
    )
    t0 = time.time()
    metrics = train_model(model, train_loader, test_loader, optimizer, mll, args)
    metrics["run_time"] = time.time() - t0
    return nan_gaussian_uq_tail(float(metrics["rmse"]), float(metrics["crps"]), float(metrics["run_time"]))


def parse_args():
    p = argparse.ArgumentParser(
        description="Default: DKL + XGB bootstrap + LMJGP. Enable JGP / DeepGP with flags below."
    )
    p.add_argument("--num_exp", type=int, default=10)
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality",
        help="Comma-separated subset of: Wine Quality, Parkinsons Telemonitoring, Appliances Energy Prediction",
    )
    p.add_argument(
        "--jgp_raw",
        action="store_true",
        help="Run JumpGP on raw features",
    )
    p.add_argument(
        "--jgp_pca",
        action="store_true",
        help="Run JumpGP on PCA-reduced standardized features",
    )
    p.add_argument(
        "--jgp_sir",
        action="store_true",
        help="Run JumpGP on SIR-reduced features",
    )
    p.add_argument(
        "--deepgp",
        action="store_true",
        help="Run GPyTorch DeepGP (Variational Deep GP from DeepGP_test)",
    )
    p.add_argument(
        "--no_lmjgp",
        action="store_true",
        help="Skip LMJGP (neighborhood VI / train_vi + predict_vi)",
    )
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_res: dict = {}
    for dataset_name in dataset_list:
        total_res[dataset_name] = {}
        for seed in range(args.num_exp):
            X_train_np, X_test_np, y_train_np, y_test_np, K = load_uci_split(dataset_name, seed)
            res = {}

            if args.jgp_raw:
                res["jgp_raw"] = _run_jgp_raw(X_train_np, y_train_np, X_test_np, y_test_np)
            if args.jgp_pca:
                res["jgp_pca"] = _run_jgp_pca(X_train_np, y_train_np, X_test_np, y_test_np, K)
            if args.jgp_sir:
                res["jgp_sir"] = _run_jgp_sir(X_train_np, y_train_np, X_test_np, y_test_np, H=10, K=K)

            scaler_x = StandardScaler()
            X_tr_s = scaler_x.fit_transform(X_train_np)
            X_te_s = scaler_x.transform(X_test_np)

            need_torch_data = (not args.no_lmjgp) or args.deepgp
            if need_torch_data:
                X_train = torch.from_numpy(X_train_np).float().to(device)
                Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
                X_test = torch.from_numpy(X_test_np).float().to(device)
                Y_test = torch.from_numpy(y_test_np).float().squeeze(-1).to(device)

            if not args.no_lmjgp:
                djgp_args = Namespace(
                    Q=K,
                    m1=2,
                    m2=40,
                    n=20,
                    num_steps=500,
                    MC_num=3,
                    lr=0.01,
                )
                res["lmjgp_djgp"] = _run_djgp_block(X_train, Y_train, X_test, Y_test, djgp_args, device)

            if args.deepgp:
                dg = _run_deepgp(X_train, Y_train, X_test, Y_test, K, device)
                if dg is not None:
                    res["djgp_deepgp"] = dg
                elif DeepGP is None:
                    print("Warning: --deepgp requested but DeepGP_test / gpytorch import failed; skipped.")

            total_res[dataset_name][seed] = res
            print(f"{dataset_name} seed={seed} -> { {k: tuple(v[:7]) for k, v in res.items()} }")

    out_path = args.out or f"uci_baseline_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(total_res, f)
    print(f"Saved results to {out_path}")
    return total_res


if __name__ == "__main__":
    main()
