import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import argparse
import time

import torch
import numpy as np
import math

from shared.utils1 import jumpgp_ld_wrapper
# from VI_utils_gpu_acc_UZ_mean import *
# from VI_utils_gpu_acc_UZ import *
# from check_VI_utils_gpu_acc_UZ_qumodified_cor import *
# from minibatch_check_VI_utils import *
from shared.jumpgp_runner import *
# from new_minibatch_try import train_vi_minibatch
# from check_VI_utils_gpu_acc_UZ_qumodified_cor import *
from djgp.minibatch import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test DeepJumpGP model')
    parser.add_argument('--folder_name', type=str, required=True, 
                        help='Folder name containing dataset.pkl')
    parser.add_argument('--Q', type=int, default=2,
                        help='Latent dimensionality')
    parser.add_argument('--m1', type=int, default=5,
                        help='Number of inducing points per region')
    parser.add_argument('--m2', type=int, default=20,
                        help='Number of global inducing points')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of neighbors per region')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of batch size')
    parser.add_argument('--num_steps', type=int, default=200,
                        help='Number of VI training steps')
    parser.add_argument('--MC_num', type=int, default=5,
                        help='Number of Monte Carlo samples for prediction')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--use_batch', type=bool, default=False, help='Use batch training')
    return parser.parse_args()

def initialize_model(X_train, Y_train, X_test, Y_test, args, device):
    """
    Initialize all model parameters for DJGP.
    
    Args:
        X_train: Training input features
        Y_train: Training targets
        X_test: Test input features
        Y_test: Test targets
        args: Arguments containing model hyperparameters
        device: Device to place tensors on
        
    Returns:
        regions: List of region dictionaries
        V_params: Dictionary of V parameters
        u_params: List of u parameters
        hyperparams: Dictionary of hyperparameters
    """
    T, D = X_test.shape
    Q  = args.Q
    m1 = args.m1
    m2 = args.m2
    n  = args.n

    # Build neighborhoods
    neighborhoods = find_neighborhoods(
        X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n
    )
    regions = []
    for i in range(T):
        X_nb = neighborhoods[i]['X_neighbors'].to(device)   # (n, D)
        y_nb = neighborhoods[i]['y_neighbors'].to(device)   # (n,)
        regions.append({
            'X': X_nb,
            'y': y_nb,
            'C': torch.randn(m1, Q, device=device)          # random init
        })

    # Initialize V_params
    V_params = {
        'mu_V':    torch.randn(m2, Q, D, device=device, requires_grad=True),
        'sigma_V': torch.rand( m2, Q, D, device=device, requires_grad=True),
    }

    # Initialize u_params
    u_params = []
    for _ in range(T):
        u_params.append({
            'U_logit':     torch.zeros(1, device=device, requires_grad=True),
            'mu_u':        torch.randn(m1, device=device, requires_grad=True),
            'Sigma_u':     torch.eye(m1, device=device, requires_grad=True),
            'sigma_noise': torch.tensor(0.5, device=device, requires_grad=True),
            'sigma_k':torch.tensor(0.5, device=device, requires_grad=True),
            'omega':       torch.randn(Q+1, device=device, requires_grad=True),
        })

    # Initialize hyperparams
    X_train_mean = X_train.mean(dim=0)
    X_train_std  = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std

    hyperparams = {
        'Z':             Z,                      # (m2, D)
        'X_test':        X_test,                 # (T, D)
        'lengthscales': torch.rand(Q, device=device, requires_grad=True),
        'var_w':         torch.tensor(1.0, device=device, requires_grad=True),
    }
    
    return regions, V_params, u_params, hyperparams

def construct_regions(X_test, X_train, Y_train, M, device):
    neighborhoods = find_neighborhoods(
        X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=M
    )
    regions = []
    T = X_test.shape[0]
    for i in range(T):
        X_nb = neighborhoods[i]['X_neighbors'].to(device)  # (n, D)
        y_nb = neighborhoods[i]['y_neighbors'].to(device)  # (n,)
        regions.append({
            'X': X_nb,
            'y': y_nb
        })
    return regions

def main():
    args = parse_args()

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load data onto device
    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"].to(device)   # (N_train, D)
    Y_train = dataset["Y_train"].to(device)
    X_test  = dataset["X_test"].to(device)    # (N_test, D)
    Y_test  = dataset["Y_test"].to(device)

    X_val = X_test
    Y_val = Y_test

    # 3. Unpack dimensions & args
    N_test, D = X_test.shape
    T         = N_test
    Q         = args.Q
    m1        = args.m1
    m2        = args.m2
    n         = args.n
    num_steps = args.num_steps
    MC_num    = args.MC_num

    # 4. Build neighborhoods (on CPU) then move to GPU
    neighborhoods = find_neighborhoods(
        X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n
    )
    regions = []
    start_time = time.time()
    for i in range(T):
        X_nb = neighborhoods[i]['X_neighbors'].to(device)  # (n, D)
        y_nb = neighborhoods[i]['y_neighbors'].to(device)  # (n,)
        regions.append({
            'X': X_nb,
            'y': y_nb,
            'C': torch.randn(m1, Q, device=device)       # random init
        })

    regions_val = construct_regions(X_val, X_train, Y_train, n, device)
    regions_data = (regions_val, X_val,Y_val)

    # 5. Initialize V_params on GPU
    V_params = {
        'mu_V':    torch.randn(m2, Q, D, device=device, requires_grad=True),
        'sigma_V': torch.rand( m2, Q, D, device=device, requires_grad=True),
    }

    # 6. Initialize u_params on GPU
    u_params = []
    for _ in range(T):
        u_params.append({
            'U_logit':    torch.zeros(1, device=device, requires_grad=True),
            'mu_u':       torch.randn(m1, device=device, requires_grad=True),
            'Sigma_u':    torch.eye(m1, device=device, requires_grad=True),
            'sigma_noise':torch.tensor(0.5, device=device, requires_grad=True),
            'sigma_k':torch.tensor(0.5, device=device, requires_grad=True),
            'omega':      torch.randn(Q+1, device=device, requires_grad=True),
        })

    # 7. Initialize hyperparams on GPU
    X_train_mean = X_train.mean(dim=0)
    X_train_std  = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std

    hyperparams = {
        'Z':            Z,                      # (m2, D)
        'X_test':       X_test,                 # (T, D)
        'lengthscales':torch.rand(Q, device=device, requires_grad=True),
        'var_w':        torch.tensor(1.0, device=device, requires_grad=True),
    }

    print("Everything set!")

    # 8. Compute ELBO, backprop, train, predict
    # L = compute_ELBO(regions, V_params, u_params, hyperparams)
    # print("ELBO L =", L.item())
    # L.backward()
    # print("Gradients OK")

    if not args.use_batch:
        V_params, u_params, hyperparams = train_vi(
            regions=regions,
            V_params=V_params,
            u_params=u_params,
            hyperparams=hyperparams,
            lr=args.lr,
            num_steps=num_steps,
            log_interval=50,
            regions_data=regions_data
        )
    else: 
        # V_params, u_params, hyperparams = train_vi_minibatch(
        #     regions=regions,
        #     V_params=V_params,
        #     u_params=u_params,
        #     hyperparams=hyperparams,
        #     lr=args.lr,
        #     num_steps=num_steps,
        #     log_interval=50
        # )
        V_params, u_params, hyperparams = train_vi_minibatch(
        regions, V_params, u_params, hyperparams,
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=num_steps,
        log_interval=50,
        regions_data=regions_data
        )
    print("train OK")

    mu_pred, var_pred = predict_vi(
        regions, V_params, hyperparams, M=MC_num
    )
    print("Prediction OK")
    # print("mu_pred:", mu_pred.shape)
    # print("var_pred:", var_pred.shape)

    # 9. Compute metrics & runtime
    run_time = time.time() - start_time
    # rmse, q25, q50, q75 = compute_metrics(mu_pred, var_pred, Y_test)
    # print("Results [rmse, 25%, 50%, 75%, runtime]:")
    # print([rmse, q25, q50, q75, run_time])

    # return [rmse, q25, q50, q75, run_time]
    sigmas = torch.sqrt(var_pred)
    rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)
    print(f"Results [rmse, mean crps, runtime]:{[rmse, mean_crps]}")
    # return [rmse, q25, q50, q75, run_time]
    return [rmse, mean_crps, run_time]

if __name__ == "__main__":
    main()
