# python data_generate.py --N 1000 --T 200 --D 15 --caseno 5 --device cuda
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import math
import numpy as np
import argparse
import pickle
from datetime import datetime
from skimage import io, color, transform

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data for high-dimensional GP')
    parser.add_argument('--N', type=int, default=5000, help='Number of training points')
    parser.add_argument('--T', type=int, default=100, help='Number of test points')
    parser.add_argument('--D', type=int, default=10, help='Dimension of observation space')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimension of latent space')
    parser.add_argument('--Q', type=int, default=2, help='Number of rows in A(x) matrix')
    parser.add_argument('--caseno', type=int, default=4, help='Case number for boundary generation (4,5,6 for phantom boundary)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    return parser.parse_args()

def generate_Y_from_image(z, caseno, noise_std=0.1):
    """
    根据图像边界生成目标值 Y。
    参数：
      z: 2D 隐变量，形状 (num, 2)
      caseno: 4,5,6 对应不同边界图像
      noise_std: 噪声标准差
    返回：
      Y: 目标值，形状 (num,)
    """
    # 读取边界图像（请修改为你实际的文件路径）
    image_file = {4: 'bound3.png',
                  5: 'bound1.png',
                  6: 'bound4.png'}[caseno]
    I = io.imread(image_file)
    # 转为灰度，并二值化：True 表示边界内部
    bw = color.rgb2gray(I) > 0.5
    # 调整尺寸：将图像尺寸调整为 (G, G)，这里 G 与希望的网格分辨率有关
    G = 100
    bw = transform.resize(bw, (G, G), anti_aliasing=False)
    # 将 bw 转为二值图
    bw = bw > 0.5
    # 构造网格：这里假设隐变量 z 的范围约在 [-3, 3]（可根据数据调整）
    gx = np.linspace(-3, 3, G)
    gy = np.linspace(-3, 3, G)
    # 注意：bw 的行对应 y 方向，列对应 x 方向
    Y_out = torch.empty(z.shape[0], device=z.device, dtype=z.dtype)
    z_np = z.cpu().numpy()  # 转为 numpy 便于索引
    for i in range(z.shape[0]):
        z0, z1 = z_np[i]
        # 找到最近邻网格索引
        idx0 = np.argmin(np.abs(gx - z0))
        idx1 = np.argmin(np.abs(gy - z1))
        # 如果该点在边界内部，则赋予高均值，否则低均值
        if bw[idx1, idx0]:
            Y_out[i] = 5.0 + noise_std * torch.randn(1, device=z.device)
        else:
            Y_out[i] = -5.0 + noise_std * torch.randn(1, device=z.device)
            # Y_out[i] = 0 + noise_std * torch.randn(1, device=z.device)
    return Y_out



def generate_Y(z, noise_std=0.1, caseno=None):
    """
    如果 caseno 为 4、5 或 6，则使用图像边界；否则，使用简单规则（如 z[0]+z[1]>0）。
    """
    if caseno in [4, 5, 6]:
        return generate_Y_from_image(z, caseno, noise_std)
    else:
        Y = torch.empty(z.shape[0], device=z.device, dtype=z.dtype)
        for i in range(z.shape[0]):
            if z[i, 0] + z[i, 1] > 0:
                Y[i] = 5.0 + noise_std * torch.randn(1, device=z.device)
            else:
                Y[i] = -5.0 + noise_std * torch.randn(1, device=z.device)
                # Y[i] = 0 + noise_std * torch.randn(1, device=z.device)
        return Y
    
def generate_Y_from_image_rff(
    z: torch.Tensor,
    caseno: int,
    noise_std: float = 1,
    kernel_var: float = 1.0,
    lengthscale: float = 1.0,
    num_features: int = 500
) -> torch.Tensor:
    """
    Generate target values Y for given 2D latent points z using
    a Gaussian Process approximated by Random Fourier Features (RFF),
    with a piecewise-constant mean determined by an image boundary.

    Args:
        z (torch.Tensor): 2D latent variables, shape (N, 2).
        caseno (int): Which boundary image to use: 4 -> 'bound3.png',
                      5 -> 'bound1.png', 6 -> 'bound4.png'.
        noise_std (float): Standard deviation of additive observation noise.
        kernel_var (float): Variance (σ_k^2) of the RBF kernel.
        lengthscale (float): Lengthscale (ℓ) of the RBF kernel.
        num_features (int): Number of random Fourier features (D).

    Returns:
        torch.Tensor: Generated Y values, shape (N,).
    """
    device = z.device
    N, d = z.shape

    # 1. Load and binarize boundary image
    image_map = {4: 'bound3.png', 5: 'bound1.png', 6: 'bound4.png'}
    img = io.imread(image_map[caseno])
    gray = color.rgb2gray(img)
    binary = gray > 0.5
    G = 100
    binary = transform.resize(binary, (G, G), anti_aliasing=False) > 0.5
    binary_t = torch.from_numpy(binary.astype(np.uint8)).to(device)

    # 2. Build piecewise constant mean m(z): +5 inside, -5 outside
    # gx = torch.linspace(-3.0, 3.0, G, device=device)
    # gy = torch.linspace(-3.0, 3.0, G, device=device)
    gx = torch.linspace(-2.0, 2.0, G, device=device)
    gy = torch.linspace(-2.0, 2.0, G, device=device)
    ix = torch.argmin((z[:,0].unsqueeze(1) - gx).abs(), dim=1)
    iy = torch.argmin((z[:,1].unsqueeze(1) - gy).abs(), dim=1)
    inside = binary_t[iy, ix].bool()
    m = torch.where(inside, torch.tensor(5.0, device=device),
                          torch.tensor(-5.0, device=device))  # (N,)

    # 3. Random Fourier Feature mapping for RBF kernel
    Omega = torch.randn(num_features, d, device=device) / lengthscale  # (D,2)
    phases = torch.rand(num_features, device=device) * 2 * torch.pi      # (D,)
    projection = z @ Omega.t() + phases                                 # (N,D)
    scale = torch.tensor(2 * kernel_var / num_features, device=device)
    Phi = torch.sqrt(scale) * torch.cos(projection)                     # (N,D)

    # 4. Sample GP in feature space
    w = torch.randn(num_features, device=device)  # (D,)
    f_rff = Phi @ w                                # (N,)

    # 5. Add mean and observation noise
    Y = m + f_rff + noise_std * torch.randn(N, device=device)
    return Y

import torch
import numpy as np
from typing import List, Dict, Union

Region = Dict[str, Union[str, float, List[float]]]
# A Region is a dict with keys:
#   - "type":   "ball" or "hypercube"
#   - if "ball":       "center": List[float], "radius": float,    "mean": float
#   - if "hypercube":  "mins":   List[float], "maxs": List[float], "mean": float

def compute_piecewise_mean(
    z: torch.Tensor,
    regions: List[Region],
    default_mean: float = 0.0
) -> torch.Tensor:
    """
    For each row z[i] in (N, D), assign m[i] = regions[j]["mean"] for
    the first region j that contains z[i], else default_mean.
    """
    device = z.device
    N, D = z.shape
    m = torch.full((N,), default_mean, device=device)
    for reg in regions:
        mean_val = torch.tensor(reg["mean"], device=device, dtype=z.dtype)
        if reg["type"] == "ball":
            center = torch.tensor(reg["center"], device=device, dtype=z.dtype)
            radius = float(reg["radius"])
            # compute squared distance
            d2 = torch.sum((z - center.unsqueeze(0))**2, dim=1)
            mask = (d2 <= radius**2)
        elif reg["type"] == "hypercube":
            mins = torch.tensor(reg["mins"], device=device, dtype=z.dtype)
            maxs = torch.tensor(reg["maxs"], device=device, dtype=z.dtype)
            mask = ((z >= mins.unsqueeze(0)) & (z <= maxs.unsqueeze(0))).all(dim=1)
        else:
            raise ValueError(f"Unknown region type {reg['type']!r}")
        # only assign where not already assigned
        unassigned = (m == default_mean)
        assign_mask = mask & unassigned
        m[assign_mask] = mean_val
    return m


def generate_Y_from_regions_rff(
    z: torch.Tensor,
    regions: List[Region],
    default_mean: float = -5.0,
    noise_std: float = 0.1,
    kernel_var: float = 1.0,
    lengthscale: float = 1.0,
    num_features: int = 500
) -> torch.Tensor:
    """
    Generate Y for latent points z (N×D) by
      1) computing a piecewise-constant mean via `regions`,
      2) drawing a GP sample around that mean using RFF.

    Args:
      z            : (N, D) latent inputs.
      regions      : list of region dicts (ball or hypercube) with a "mean".
      default_mean : value for points in no region.
      noise_std    : observational noise σ_n.
      kernel_var   : GP kernel variance σ_k^2.
      lengthscale  : GP kernel lengthscale ℓ.
      num_features : number of RFF features D.

    Returns:
      Y : (N,) samples.
    """
    device = z.device
    N, D = z.shape

    # 1) piecewise constant mean
    m = compute_piecewise_mean(z, regions, default_mean)  # (N,)

    # 2) random Fourier features for RBF kernel
    Omega = torch.randn(num_features, D, device=device) / lengthscale  # (D_rff, D)
    phases = torch.rand(num_features, device=device) * 2 * torch.pi     # (D_rff,)
    proj = z @ Omega.t() + phases                                       # (N, D_rff)
    scale = torch.tensor(2 * kernel_var / num_features, device=device)
    Phi = torch.sqrt(scale) * torch.cos(proj)                           # (N, D_rff)

    # 3) sample the GP in feature space
    w = torch.randn(num_features, device=device)  # (D_rff,)
    f = Phi @ w                                   # (N,)

    # 4) add mean + noise
    Y = m + f + noise_std * torch.randn(N, device=device)
    return Y

def normalize_A(A):
    """
    对A矩阵进行正则化
    Input: A shape is (N, Q, D)
    Output: 正则化后的A
    """
    N, Q, D = A.shape
    A_normalized = A.clone()
    
    for i in range(N):
        # 方法1：Frobenius范数归一化
        # A_i = A[i]  # shape: (Q, D)
        # frob_norm = torch.norm(A_i, p='fro')
        # A_normalized[i] = A_i / frob_norm
        
        # 方法2：行归一化
        # for q in range(Q):
        #     row_norm = torch.norm(A[i, q], p=2)
        #     A_normalized[i, q] = A[i, q] / row_norm
        
        # 方法3：正交化 (使用QR分解)
        A_i = A[i]  # shape: (Q, D)
        Z, R = torch.linalg.qr(A_i.T)  # 对转置进行QR分解
        A_normalized[i] = Z[:, :Q].T  # 取前Q列的转置，确保是正交矩阵
        
    return A_normalized

# def generate_data(args):
#     """
#     Generate synthetic data based on input arguments.
#     """
#     device = torch.device(args.device)
#     torch.set_default_dtype(torch.float32)

#     # Generate input X_train and X_test
#     X_train = torch.randn(args.N, args.D, device=device)
#     X_test = torch.randn(args.T, args.D, device=device)
#     X_all = torch.cat([X_train, X_test], dim=0)
#     N_all = X_all.shape[0]

#     # Sample GP hyperparameters for A(x)
#     # gamma_dist = torch.distributions.Gamma(2.0, 1.0)
#     gamma_dist = torch.distributions.Gamma(4.0, 2.0)
#     sigma_a_A = gamma_dist.sample().to(device)
#     sigma_q_A = gamma_dist.sample((1,)).to(device)

#     # Compute GP covariance matrix K_A
#     diff = X_all.unsqueeze(1) - X_all.unsqueeze(0)
#     sqdist = torch.sum(diff**2, dim=2)
#     K_A = sigma_a_A**2 * torch.exp(-0.5 * sqdist / (sigma_q_A**2))
#     K_A = K_A + 1e-6 * torch.eye(N_all, device=device, dtype=K_A.dtype)

#     # Generate A for each (q, d)
#     A_all = torch.empty(N_all, args.Q, args.D, device=device)
#     mean_zero = torch.zeros(N_all, device=device)
#     for q in range(args.Q):
#         for d in range(args.D):
#             mvn = torch.distributions.MultivariateNormal(mean_zero, covariance_matrix=K_A)
#             A_all[:, q, d] = mvn.sample()

#     # 对A进行正则化
#     A_all = normalize_A(A_all)

#     # Split A into train and test
#     A_train = A_all[:args.N]
#     A_test = A_all[args.N:]

#     # Compute latent variables z(x)
#     z_train = torch.bmm(A_train, X_train.unsqueeze(-1)).squeeze(-1)
#     z_test = torch.bmm(A_test, X_test.unsqueeze(-1)).squeeze(-1)

#     # Generate target variables
#     # Y_train = generate_Y(z_train, noise_std=0.1, caseno=args.caseno)
#     # Y_test = generate_Y(z_test, noise_std=0.1, caseno=args.caseno)
#     # Y_train = generate_Y(z_train, noise_std=1, caseno=args.caseno)
#     # Y_test = generate_Y(z_test, noise_std=1, caseno=args.caseno)
#     Y_train = generate_Y_from_image_rff(z_train, noise_std=1, caseno=args.caseno)
#     Y_test = generate_Y_from_image_rff(z_test, noise_std=1, caseno=args.caseno)

#     return {
#         "X_train": X_train, "Y_train": Y_train,
#         "X_test": X_test, "Y_test": Y_test,
#         "hyperparams": {"sigma_a_A": sigma_a_A.item(), "sigma_q_A": sigma_q_A.item()}
#     }
regions_5d = [
    # 1) 原点小球：半径 1，mean = -5
    {"type": "ball", "center": [0.0]*5, "radius": 1.0, "mean": -5.0},
    # 2) 原点球壳：半径 1 到 2 之间，mean = +5
    {"type": "ball", "center": [0.0]*5, "radius": 2.0, "mean": +5.0},
    # 3) 坐标轴中间的小超立方体：[-0.5,0.5]^5，mean = 0
    {"type": "hypercube", "mins": [-0.5]*5, "maxs": [0.5]*5, "mean": 0.0},
    # 4) 右上角大超立方体：所有坐标 ∈ [1,2]，mean = +3
    {"type": "hypercube", "mins": [1.0]*5, "maxs": [2.0]*5, "mean": +3.0},
    # 5) 另一个偏移球体：中心在 (2,-2,0,0,1)，半径 0.8，mean = -3
    {"type": "ball", "center": [2.0,-2.0,0.0,0.0,1.0], "radius": 0.8, "mean": -3.0},
]

regions_4d = [
    # 1) 全局大球：中心原点，半径 2，mean = +2
    {"type": "ball",       "center": [0.0]*4,  "radius": 2.0, "mean": +5.0},
    # 2) 小球（嵌套）：中心原点，半径 1，mean = -2
    {"type": "ball",       "center": [0.0]*4,  "radius": 1.0, "mean": -5.0},
    # 3) 边角超立方：坐标 ∈ [-2,-1]^4，mean = +4
    {"type": "hypercube",  "mins": [-2.0]*4,   "maxs": [-1.0]*4, "mean": +4.0},
    # 4) 中间超立方：坐标 ∈ [-0.5,0.5]×[1,1.5]×…，mean = 0
    {"type": "hypercube",  "mins": [-0.5,1.0,-0.5,-0.5],
                           "maxs": [ 0.5,1.5, 0.5, 0.5], "mean": 0.0},
]

regions_3d = [
    # 1) 大球：center=(0,0,0), radius=3, mean=+1
    {"type": "ball",       "center": [0,0,0],   "radius": 3.0, "mean": +5.0},
    # 2) 中球：center=(0,0,0), radius=2, mean=-1
    {"type": "ball",       "center": [0,0,0],   "radius": 2.0, "mean": -5.0},
    # 3) 小球：center=(0,0,0), radius=1, mean=+5
    {"type": "ball",       "center": [0,0,0],   "radius": 1.0, "mean": +0.0},
    # 4) 某角落超立方：x∈[1,2],y∈[-2,-1],z∈[0,1]，mean=0
    {"type": "hypercube",  "mins": [1.0,-2.0,0.0],
                           "maxs": [2.0,-1.0,1.0], "mean": 4.0},
    # 5) 另一区域：x^2+y^2<1 且 z>1 （可通过两个条件组合实现），mean=+3
    #    —— 如果你需要更复杂的交集/并集，可自行在代码中添加逻辑。
]

def generate_data(args):
    """
    Generate synthetic data based on input arguments.
    """
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)

    # 1) Generate X_train, X_test, and stack
    X_train = torch.randn(args.N, args.D, device=device)
    X_test  = torch.randn(args.T, args.D, device=device)
    X_all   = torch.cat([X_train, X_test], dim=0)
    N_all   = X_all.shape[0]

    # 2) Sample GP hyperparameters and build K_A (unchanged) …
    gamma_dist = torch.distributions.Gamma(4.0, 2.0)
    sigma_a_A  = gamma_dist.sample().to(device)
    sigma_q_A  = gamma_dist.sample((1,)).to(device)

    diff   = X_all.unsqueeze(1) - X_all.unsqueeze(0)
    sqdist = torch.sum(diff**2, dim=2)
    K_A    = sigma_a_A**2 * torch.exp(-0.5 * sqdist / (sigma_q_A**2))
    K_A   += 1e-6 * torch.eye(N_all, device=device)

    # 3) Sample A_all and normalize (unchanged) …
    A_all = torch.empty(N_all, args.Q, args.D, device=device)
    mean0 = torch.zeros(N_all, device=device)
    for q in range(args.Q):
        for d in range(args.D):
            mvn = torch.distributions.MultivariateNormal(mean0, covariance_matrix=K_A)
            A_all[:, q, d] = mvn.sample()
    A_all = normalize_A(A_all)

    # 4) Split A into train/test
    A_train = A_all[:args.N]
    A_test  = A_all[args.N:]

    # 5) Compute latent z for train and test
    z_train = torch.bmm(A_train, X_train.unsqueeze(-1)).squeeze(-1)
    z_test  = torch.bmm(A_test,  X_test.unsqueeze(-1)).squeeze(-1)

    # ────────────────────────────────────────────────────────────────────
    # 6) Jointly sample Y_all in one go, then split
    z_all = torch.cat([z_train, z_test], dim=0) 
    if z_all.shape[1] == 2:
        Y_all = generate_Y_from_image_rff(
            z_all,
            caseno    = args.caseno,
            noise_std = 1.0,
            kernel_var = 1.0,
            lengthscale = 1.0,
            num_features = 500
        )
    else: 
        if z_all.shape[1] == 5:
            regions = regions_5d
        elif z_all.shape[1] == 4:
            regions = regions_4d
        elif z_all.shape[1] == 3:
            regions = regions_3d
        else:
            raise ValueError("Unsupported dimension")

        Y_all = generate_Y_from_regions_rff(
            z_all,
            regions       = regions,
            default_mean  = -5.0,
            noise_std     = 0.1,
            kernel_var    = 1.0,
            lengthscale   = 1.0,
            num_features  = 500
        )

    
    Y_train = Y_all[:args.N]
    Y_test  = Y_all[args.N:]
    # ────────────────────────────────────────────────────────────────────

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_test":  X_test,  "Y_test":  Y_test,
        "hyperparams": {
            "sigma_a_A": sigma_a_A.item(),
            "sigma_q_A": sigma_q_A.item()
        }
    }


import torch
import math

def random_fourier_features(X, M, sigma):
    """
    X: (N, D)
    M: 目标特征维度
    sigma: RBF 核宽度参数
    返回 Z: (N, M)
    """
    N, D = X.shape
    # W ~ N(0, I/σ²), b ~ Uniform(0, 2π)
    W = torch.randn(D, M, device=X.device) / sigma
    b = 2 * math.pi * torch.rand(M, device=X.device)
    Z = math.sqrt(2.0 / M) * torch.cos(X @ W + b)
    return Z

def generate_data1(args):
    """
    利用随机傅里叶特征（RFF）将低维 X 非线性映射到高维 Z，再生成 Y。
    新增 args.M 和 args.sigma 两个字段。
    """
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)

    # 1. 生成原始输入 X
    X_train = torch.randn(args.N, args.D, device=device)
    X_test  = torch.randn(args.T, args.D, device=device)
    X_all   = torch.cat([X_train, X_test], dim=0)  # (N+T, D)

    # 2. 随机傅里叶特征映射到高维 Z_all
    #    M: 高维特征数，sigma: 控制核宽度
    noise_std = 3
    z_all = random_fourier_features(X_all, M=args.Q, sigma=noise_std)  # (N+T, M)
    

    # z_all = torch.cat([z_train, z_test], dim=0) 
    if z_all.shape[1] == 2:
        Y_all = generate_Y_from_image_rff(
            z_all,
            caseno    = args.caseno,
            noise_std = noise_std,
            kernel_var = 1.0,
            lengthscale = 1.0,
            num_features = 500
        )
    else: 
        if z_all.shape[1] == 5:
            regions = regions_5d
        elif z_all.shape[1] == 4:
            regions = regions_4d
        elif z_all.shape[1] == 3:
            regions = regions_3d
        else:
            raise ValueError("Unsupported dimension")

        Y_all = generate_Y_from_regions_rff(
            z_all,
            regions       = regions,
            default_mean  = -5.0,
            # noise_std     = 0.1,
            # noise_std     = 1,
            noise_std     = noise_std,
            kernel_var    = 1.0,
            lengthscale   = 1.0,
            num_features  = 500
        )

    
    Y_train = Y_all[:args.N]
    Y_test  = Y_all[args.N:]
    # print(Y_train.shape)

    # 3. 切分 Z 为训练/测试
    Z_train = z_all[:args.N]     # (N, M)
    Z_test  = z_all[args.N:]     # (T, M)

    # 4. 根据高维特征 Z 生成目标 Y
    # Y_train = generate_Y(Z_train, noise_std=1.0, caseno=args.caseno)
    # Y_test  = generate_Y(Z_test,  noise_std=1.0, caseno=args.caseno)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_test":  X_test,  "Y_test":  Y_test,
        "hyperparams": {"M": 2, "sigma": 1}
    }


def save_dataset(data, args, folder_name=None):
    """
    Save the generated dataset and arguments to files.
    Args:
        data: Dictionary containing the dataset
        args: Namespace object containing command line arguments
        folder_name: Optional folder name for saving
    """
    if folder_name is None:
        folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M")
    os.makedirs(folder_name, exist_ok=True)
    
    # 将 args 转换为字典并添加到 data 中
    args_dict = vars(args)
    data['args'] = args_dict
    
    # 保存数据集
    with open(os.path.join(folder_name, 'dataset.pkl'), 'wb') as f:
        pickle.dump(data, f)

    return folder_name

def main():
    args = parse_args()
    data = generate_data1(args)
    
    # Print shapes and hyperparameters
    print("X_train shape:", data["X_train"].shape)
    print("Y_train shape:", data["Y_train"].shape)
    print("X_test shape:", data["X_test"].shape)
    print("Y_test shape:", data["Y_test"].shape)
    # print("\nSampled A-GP hyperparameters:")
    # print(" sigma_a_A =", data["hyperparams"]["sigma_a_A"])
    # print(" sigma_q_A =", data["hyperparams"]["sigma_q_A"])
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f" {k} = {v}")
    
    # Save the dataset with arguments
    folder_name = save_dataset(data, args)
    return folder_name

if __name__ == "__main__":
    main()