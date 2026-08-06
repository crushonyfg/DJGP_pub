# python JumpGP_test.py --folder_name "2025_04_01_23" --M 100 --device cpu
# python JumpGP_test.py --folder_name "2025_04_01_23" --M 100 --device cpu --use_sir --sir_H 10 --sir_K 2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
# 将 JumpGP_code_py 所在的目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import torch
import time
import math
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
import argparse
from shared.utils1 import jumpgp_ld_wrapper
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test JumpGP model')
    parser.add_argument('--folder_name', type=str, required=True, 
                      help='Folder name containing dataset.pkl')
    parser.add_argument('--M', type=int, default=100, 
                      help='Number of nearest neighbors')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use (cpu/cuda)')
    parser.add_argument('--use_sir', action='store_true',
                      help='Whether to use SIR dimension reduction')
    parser.add_argument('--sir_H', type=int, default=10,
                      help='Number of slices for SIR')
    parser.add_argument('--sir_K', type=int, default=2,
                      help='Number of components to keep in SIR')
    parser.add_argument('--use_cv', action='store_true',
                        help='Whether to cross-validate sir_K')
    return parser.parse_args()

def apply_sir_reduction(X_train, Y_train, X_test, args):
    """Apply SIR dimension reduction to the data"""
    # 转换为numpy数组进行SIR处理
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    
    # 初始化并训练SIR模型
    # sir = SIR(H=args.sir_H, K=args.sir_K, estdim=True, threshold=0.5)
    sir = SIR(H=args.sir_H, K=args.sir_K)
    sir.fit(X_train_np, Y_train_np)
    
    # 获取变换矩阵并应用降维
    transform_matrix = sir.transform()
    # 确保变换矩阵是实数
    transform_matrix = np.real(transform_matrix)
    
    X_train_reduced = np.matmul(X_train_np, transform_matrix)
    X_test_reduced = np.matmul(X_test_np, transform_matrix)
    
    # 转回PyTorch张量，确保使用正确的数据类型
    return (torch.from_numpy(X_train_reduced).to(X_train.device).to(torch.float64), 
            torch.from_numpy(X_test_reduced).to(X_test.device).to(torch.float64))

def load_dataset(folder_name):
    """Load dataset from pickle file"""
    dataset_path = os.path.join(folder_name, 'dataset.pkl')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

import numpy as np
from scipy.linalg import fractional_matrix_power, eigh
from scipy.sparse.linalg import eigs

class SIR:
    def __init__(self, H, K=None, estdim=False, threshold=0.9, Cn=1):
        """
        Sliced Inverse Regression (SIR) for dimensionality reduction.

        Parameters
        ----------
        H : int
            Number of slices.
        K : int or None
            Target dimension when estdim=False. If estdim=True, K is determined automatically.
        estdim : bool
            Whether to automatically estimate the dimension using cumulative explained variance.
        threshold : float
            Variance threshold for automatic dimension selection (e.g., 0.9 for 90%).
        Cn : float
            Placeholder for other selection criteria (unused with cumulative variance).
        """
        self.H = H
        self.K = K
        self.estdim = estdim
        self.threshold = threshold
        self.Cn = Cn

    def fit(self, X, Y):
        """
        Fit the SIR model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictor data.
        Y : array-like, shape (n_samples,) or (n_samples, 1)
            Response data.

        Returns
        -------
        V : array, shape (n_features, K)
            Leading eigenvectors (projection directions).
        K : int
            Chosen number of dimensions.
        M : array, shape (n_features, n_features)
            SIR covariance matrix.
        D : array, shape (K,)
            Leading eigenvalues.
        """
        # Store and center X
        self.X = X
        self.Y = Y
        self.mean_x = np.mean(X, axis=0)
        self.sigma_x = np.cov(X, rowvar=False)

        # Standardize X
        Z = (X - self.mean_x) @ fractional_matrix_power(self.sigma_x, -0.5)
        n, p = Z.shape

        # Ensure Y shape
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[0] != n or Y.shape[1] != 1:
            raise ValueError("X and Y must have matching samples, and Y must be 1D.")

        # Weights
        W = np.ones((n, 1)) / n

        # Determine slice sizes
        base_count = n // self.H
        extra = n % self.H
        counts = np.array([base_count + (1 if i < extra else 0) for i in range(self.H)])
        cum_counts = np.cumsum(counts)

        # Sort by Y
        data = np.hstack((Z, Y, W))
        data = data[np.argsort(data[:, p], axis=0)]
        z, y, w = data[:, :p], data[:, p:p+1], data[:, p+1:p+2]

        # Compute slice means
        muh = np.zeros((self.H, p))
        wh  = np.zeros((self.H, 1))
        slice_idx = 0
        for i in range(n):
            if i >= cum_counts[slice_idx]:
                slice_idx += 1
            muh[slice_idx] += z[i]
            wh[slice_idx]  += w[i]
        muh = muh / (wh * n)

        # Build SIR matrix M
        M = np.zeros((p, p))
        for h in range(self.H):
            M += wh[h] * np.outer(muh[h], muh[h])
        self.M = M

        # Eigen decomposition and dimension selection
        if self.estdim:
            # Full-spectrum decomposition
            eigvals, eigvecs = eigh(M)
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # Cumulative explained variance
            cumvar = np.cumsum(eigvals) / np.sum(eigvals)
            K_auto = int(np.searchsorted(cumvar, self.threshold) + 1)
            self.K = K_auto
            self.D = eigvals[:self.K]
            self.V = eigvecs[:, :self.K]
        else:
            if self.K is None:
                raise ValueError("K must be specified when estdim=False.")
            # Compute top-K eigenpairs
            D, V = eigs(A=M, k=self.K, which='LM')
            # eigs can return complex values; take real part
            self.D = np.real(D)
            self.V = np.real(V)

        return self.V, self.K, self.M, self.D

    def transform(self):
        """
        Project original data onto the SIR directions.

        Returns
        -------
        hatbeta : array, shape (n_features, K)
            Projection directions in original X-space.
        """
        # Map back through the standardization
        hatbeta = fractional_matrix_power(self.sigma_x, -0.5) @ self.V
        return hatbeta



def find_neighborhoods(X_test, X_train, Y_train, M):
    """Find M nearest neighbors for each test point"""
    T = X_test.shape[0]
    dists = torch.cdist(X_test, X_train)
    _, indices = torch.sort(dists, dim=1)
    indices = indices[:, :M]
    
    neighborhoods = []
    for t in range(T):
        idx = indices[t]
        neighborhoods.append({
            "X_neighbors": X_train[idx],
            "y_neighbors": Y_train[idx],
            "indices": idx
        })
    return neighborhoods

# def compute_metrics(predictions, sigmas, Y_test):
#     """Compute RMSE and NLPD metrics"""
#     # 计算RMSE
#     rmse = torch.sqrt(torch.mean((predictions - Y_test)**2))
    
#     # 计算NLPD
#     nlpd = 0.5 * torch.log(2 * math.pi * sigmas**2) + \
#            0.5 * ((Y_test - predictions)**2) / (sigmas**2)
    
#     # 计算四分位数
#     q75 = torch.quantile(nlpd, 0.75)
#     q25 = torch.quantile(nlpd, 0.25)
#     q50 = torch.quantile(nlpd, 0.50)
    
#     return rmse, q25, q50, q75

def compute_metrics(predictions, sigmas, Y_test):
    """Compute RMSE and mean CRPS for Gaussian predictive distributions."""
    # 计算 RMSE
    predictions = torch.as_tensor(predictions)
    device = predictions.device

    # 统一转换为 tensor 并放到同一设备
    sigmas = torch.as_tensor(sigmas, device=device)
    Y_test = torch.as_tensor(Y_test, device=device)

    rmse = torch.sqrt(torch.mean((predictions - Y_test)**2))
    
    # 计算 CRPS
    # z = (y - μ) / σ
    z = (Y_test - predictions) / sigmas
    # 标准正态 CDF 和 PDF
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    # CRPS 闭式公式
    crps = sigmas * ( z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi) )
    
    # 平均 CRPS
    mean_crps = torch.mean(crps)
    
    return rmse, mean_crps

def evaluate_jumpgp(X_test, Y_test, neighborhoods, device, std=None):
    """Evaluate JumpGP model on test data"""
    T = X_test.shape[0]
    jump_gp_results = []
    jump_gp_res_sig = []
    
    # 记录开始时间
    start_time = time.time()
    
    for t in tqdm(range(T)):
        neigh = neighborhoods[t]
        X_neigh = neigh["X_neighbors"]
        y_neigh = neigh["y_neighbors"]
        x_t_test = X_test[t]
        
        mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
            X_neigh, 
            y_neigh.view(-1, 1), 
            x_t_test.view(1, -1), 
            mode="CEM", 
            flag=False, 
            device=device
        )
        
        jump_gp_results.append(mu_t)
        jump_gp_res_sig.append(torch.sqrt(sig2_t))
    
    # 计算运行时间
    run_time = time.time() - start_time
    
    # 转换预测结果为tensor
    predictions = torch.tensor([x.detach().item() for x in jump_gp_results])
    sigmas = torch.tensor([s.detach().item() for s in jump_gp_res_sig])

    if std is not None:
        predictions = predictions * std[1] + std[0]
        sigmas = sigmas * std[1]
    
    # 计算评估指标
    # rmse, q25, q50, q75 = compute_metrics(predictions, sigmas, Y_test)
    rmse, mean_crps = compute_metrics(predictions, sigmas, Y_test)
    
    # return [rmse, q25, q50, q75, run_time]
    return [rmse, mean_crps, run_time]

def main():
    # 解析命令行参数
    args = parse_args()
    device = torch.device(args.device)
    
    # 加载数据集
    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]
    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    # 如果使用SIR，先进行降维
    if args.use_sir:
        print("Applying SIR dimension reduction...")
        X_train, X_test = apply_sir_reduction(X_train, Y_train, X_test, args)
        print(f"Data dimensionality reduced to {X_train.shape[1]}")
    
    # 找到邻域
    neighborhoods = find_neighborhoods(X_test, X_train, Y_train, args.M)
    
    # 评估模型并获取结果
    result = evaluate_jumpgp(X_test, Y_test, neighborhoods, device)
    
    # 打印结果
    # print(f"RMSE: {result[0]:.4f}")
    # print(f"NLPD Q25: {result[1]:.4f}")
    # print(f"NLPD Q50: {result[2]:.4f}")
    # print(f"NLPD Q75: {result[3]:.4f}")
    # print(f"Runtime: {result[4]:.2f} seconds")
    print(f"RMSE: {result[0]:.4f}")
    print(f"CRPS: {result[1]:.4f}")
    print(f"Runtime: {result[2]:.2f} seconds")
    
    return result

if __name__ == "__main__":
    main()