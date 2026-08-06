import torch
import math
import matplotlib.pyplot as plt
import numpy as np  # 用于转换显示
from scipy.stats import gamma
import torch
import pyro
import pyro.infer.mcmc as mcmc
from pyro.infer.mcmc import NUTS, MCMC
import pyro.distributions as dist
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from JumpGaussianProcess.JumpGP_LD import JumpGP_LD

from tqdm import tqdm

# 超参数先验参数（InvGamma 参数）
# HMC+gibbs_sampling
alpha_a = 2.0
beta_a = 1.0
alpha_q = 2.0
beta_q = 1.0

# =============================================================================
# 0. 定义 jumpgp_ld_wrapper，用于包装 JumpGP_LD 的输入输出
# =============================================================================
def jumpgp_ld_wrapper(
    zeta_t_torch,
    y_t_torch,
    xt_torch,
    mode,
    flag,
    device=torch.device("cpu"),
    logtheta=None,
):
    """
    Wrapper for JumpGP_LD:
      - Converts input torch.Tensors (zeta: (M, Q), y: (M,1), xt: (Nt,d))
        to numpy arrays.
      - Calls JumpGP_LD.
      - Reconstructs the returned model dictionary, converting numeric items to torch.Tensors.
    
    Returns:
      mu_t: torch.Tensor (e.g. shape (1,1))
      sig2_t: torch.Tensor
      model: dict containing keys 'w', 'ms', 'logtheta', 'r' as torch.Tensors
      h: auxiliary output (保持不变)
    """
    dtype = zeta_t_torch.dtype if torch.is_tensor(zeta_t_torch) else torch.float64
    zeta_np = np.asarray(zeta_t_torch.detach().cpu().numpy(), dtype=np.float64)
    y_np = np.asarray(y_t_torch.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 1)
    xt_np = np.asarray(xt_torch.detach().cpu().numpy(), dtype=np.float64)
    if zeta_np.ndim != 2:
        zeta_np = zeta_np.reshape(zeta_np.shape[0], -1)
    if xt_np.ndim == 1:
        xt_np = xt_np.reshape(1, -1)
    logtheta_np = None
    if logtheta is not None:
        if torch.is_tensor(logtheta):
            logtheta_np = logtheta.detach().cpu().numpy()
        else:
            logtheta_np = np.asarray(logtheta)
        logtheta_np = np.asarray(logtheta_np, dtype=np.float64).reshape(-1)
    
    # 调用原 JumpGP_LD 函数（不修改）
    if logtheta_np is None:
        mu_t_np, sig2_t_np, model_np, h = JumpGP_LD(zeta_np, y_np, xt_np, mode, flag)
    else:
        mu_t_np, sig2_t_np, model_np, h = JumpGP_LD(zeta_np, y_np, xt_np, mode, flag, logtheta_np)
    
    mu_t = torch.as_tensor(mu_t_np, dtype=dtype, device=device)
    sig2_t = torch.as_tensor(sig2_t_np, dtype=dtype, device=device)
    
    keys_to_keep = ['w', 'ms', 'logtheta', 'r', 'gamma', 'nw', 'sigma', 'RR', 'nll']
    model_torch = {}
    for key in keys_to_keep:
        if key in model_np:
            val = model_np[key]
            if isinstance(val, np.ndarray):
                if val.dtype == np.bool_:
                    model_torch[key] = torch.from_numpy(val).to(device=device)
                else:
                    model_torch[key] = torch.as_tensor(val, dtype=dtype, device=device)
            elif isinstance(val, (int, float)):
                model_torch[key] = torch.tensor(val, dtype=dtype, device=device)
            else:
                model_torch[key] = val
    return mu_t, sig2_t, model_torch, h


# 定义 RBF 核（GP 映射），torch 版本
def compute_K_A_torch(X, sigma_a, sigma_q_value):
    T = X.shape[0]
    X1 = X.unsqueeze(1)  # (T, 1, D)
    X2 = X.unsqueeze(0)  # (1, T, D)
    dists = torch.sum((X1 - X2)**2, dim=2)  # (T, T)
    return sigma_a**2 * torch.exp(-dists / (2 * sigma_q_value**2))

@torch.jit.script
def transform_torch(A_t, x):
    return A_t @ x

# 计算局部 GP 核矩阵，利用 zeta 和 logtheta
@torch.jit.script
def compute_jumpGP_kernel_torch(zeta, logtheta):
    logtheta = logtheta.to(torch.float64)
    zeta = zeta.to(torch.float64)
    M, Q = zeta.shape
    ell = torch.exp(logtheta[:Q])
    s_f = torch.exp(logtheta[Q])
    sigma_n = torch.exp(logtheta[Q+1])
    zeta_scaled = zeta / ell
    diff = zeta_scaled.unsqueeze(1) - zeta_scaled.unsqueeze(0)
    dists_sq = torch.sum(diff**2, dim=2)
    K = s_f**2 * torch.exp(-0.5 * dists_sq)

    # ✅ 添加 Jitter 以确保 K 是正定的
    jitter = 1e-6 * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
    K = K + jitter
    # K = K + (sigma_n**2) * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
    return K

@torch.jit.script
def log_normal_pdf_torch(x, mean, var):
    return -0.5 * torch.log(2 * math.pi * var) - 0.5 * ((x - mean)**2 / var)

@torch.jit.script
def log_multivariate_normal_pdf_cholesky(x, mean, cov):
    cov = cov.to(torch.float64)
    diff = x - mean
    sign, logdet = torch.slogdet(cov)
    if sign.item() <= 0:
        cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
        sign, logdet = torch.slogdet(cov)
    return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))

@torch.jit.script
def log_multivariate_normal_pdf_torch(x, mean, cov):
    #alternative to log_multivariate_normal_pdf_torch
    cov = cov.to(torch.float64)
    diff = (x - mean).unsqueeze(-1)
    # 进行 Cholesky 分解
    L = torch.linalg.cholesky(cov)
    # 解线性方程组：L y = diff
    sol = torch.cholesky_solve(diff, L)
    quad = (diff.transpose(-1, -2) @ sol).squeeze()
    # 计算对数行列式：log(det(cov)) = 2 * sum(log(diag(L)))
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    D = x.shape[0]
    return -0.5 * (quad + logdet + D * math.log(2 * math.pi))


def transform_torch(A_t, x):
    return A_t @ x


def log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q):
    T, Q, D = A.shape
    logp = 0.0
    for q in range(Q):
        K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
        for d in range(D):
            A_qd = A[:, q, d]
            idx = [i for i in range(T) if i != t]
            idx_tensor = torch.tensor(idx, dtype=torch.long, device=A.device)
            A_minus = A_qd[idx_tensor]
            k_tt = K[t, t]
            k_t_other = K[t, idx_tensor]
            K_minus = K[idx_tensor][:, idx_tensor]
            sol = torch.linalg.solve(K_minus, A_minus)
            cond_mean = k_t_other @ sol
            cond_var = k_tt - k_t_other @ torch.linalg.solve(K_minus, k_t_other)
            cond_var = torch.clamp(cond_var, min=1e-6)
            a_val = A_t_candidate[q, d]
            logp = logp + log_normal_pdf_torch(a_val, cond_mean, cond_var)
    return logp


def log_prior_A_batch_t_torch_optimized(A_batch, t_batch, A, X_test, sigma_a, sigma_q, K_cache_A=None):
    """
    计算一批候选时间步 t_batch 的先验对数概率：
      A_batch: shape (B, Q, D)，候选更新的 A 值
      t_batch: list 或 tensor，长度 B，为待更新的时间步索引
      A: 原始完整的 A，形状 (T, Q, D)（条件部分为 A[-t_batch]）
      对于每个 q 和 d，我们认为 A[:, q, d] 服从 N(0, K)，其中 K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
      对于候选时刻集合 I = t_batch，其余时刻为 J，条件分布为：
          μ = K_{I,J} K_{J,J}^{-1} A[J]
          Σ = K_{I,I} - K_{I,J} K_{J,J}^{-1} K_{J,I}
      然后利用多元高斯对数密度公式计算 log p(candidate | μ, Σ)
    """
    T, Q, D = A.shape
    B = A_batch.shape[0]
    total_logp = 0.0
    # 确保 t_batch 为 list
    if not isinstance(t_batch, list):
        t_batch = t_batch.tolist()
    full_idx = set(range(T))
    I = t_batch  # 待更新索引
    J = sorted(list(full_idx - set(I)))  # 条件部分的索引
    # 将 J 转为 tensor
    J_tensor = torch.tensor(J, dtype=torch.long, device=A.device)
    
    for q in range(Q):
        # 对于每个 q，计算核矩阵
        # 假设 compute_K_A_torch 已经实现，这里调用它
        K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])  # shape (T, T)
        # 取出候选部分
        I_tensor = torch.tensor(I, dtype=torch.long, device=A.device)
        K_IJ = K[I_tensor][:, J_tensor]     # shape (B, T-B)
        K_II = K[I_tensor][:, I_tensor]       # shape (B, B)
        K_JJ = K[J_tensor][:, J_tensor]       # shape ((T-B) x (T-B))
        # 对于每个 d
        for d in range(D):
            y = A[:, q, d]  # shape (T,)
            candidate = A_batch[:, q, d]  # shape (B,)
            y_J = y[J_tensor]  # shape (T-B,)
            # 求解 v = K_JJ^{-1} y_J
            v = torch.linalg.solve(K_JJ, y_J)
            mu = K_IJ @ v  # shape (B,)
            # 计算条件协方差
            temp = torch.linalg.solve(K_JJ, K_IJ.t())  # shape ((T-B) x B)
            Sigma = K_II - K_IJ @ temp  # shape (B, B)
            # 保证对称性
            Sigma = (Sigma + Sigma.t()) / 2.0
            # 计算多元高斯 log 密度
            logpdf = log_multivariate_normal_pdf_torch(candidate, mu, Sigma)
            total_logp = total_logp + logpdf
    return total_logp
# def log_prior_A_batch_t_torch_optimized(A_batch, t_batch, A, X_test, sigma_a, sigma_q, K_cache_A=None):
#     """
#     计算一批候选时间步 t_batch 的先验对数概率：
#       A_batch: shape (B, Q, D)，候选更新的 A 值
#       t_batch: list 或 tensor，长度 B，为待更新的时间步索引
#       A: 原始完整的 A，形状 (T, Q, D)（条件部分为 A[-t_batch]）
#       对于每个 q 和 d，我们认为 A[:, q, d] 服从 N(0, K)，其中 K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
#       对于候选时刻集合 I = t_batch，其余时刻为 J，条件分布为：
#           μ = K_{I,J} K_{J,J}^{-1} A[J]
#           Σ = K_{I,I} - K_{I,J} K_{J,J}^{-1} K_{J,I}
#       然后利用多元高斯对数密度公式计算 log p(candidate | μ, Σ)
#     """
#     T, Q, D = A.shape
#     B = A_batch.shape[0]
#     total_logp = 0.0
#     # 确保 t_batch 为 list
#     if not isinstance(t_batch, list):
#         t_batch = t_batch.tolist()
#     full_idx = set(range(T))
#     I = t_batch  # 待更新索引
#     J = sorted(list(full_idx - set(I)))  # 条件部分的索引
#     # 将 J 转为 tensor
#     I_tensor = torch.tensor(I, dtype=torch.long, device=A.device)
#     J_tensor = torch.tensor(J, dtype=torch.long, device=A.device)
    
#     for q in range(Q):
#         # 对于每个 q，计算核矩阵
#         # 假设 compute_K_A_torch 已经实现，这里调用它
#         if K_cache_A is None:
#             K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])  # shape (T, T)
#         else:
#             K = K_cache_A[q]
#         # 取出候选部分
#         K_IJ = K[I_tensor][:, J_tensor]     # shape (B, T-B)
#         K_II = K[I_tensor][:, I_tensor]       # shape (B, B)
#         K_JJ = K[J_tensor][:, J_tensor]       # shape ((T-B) x (T-B))

#         # 计算条件协方差
#         temp = torch.linalg.solve(K_JJ, K_IJ.t())  # shape ((T-B) x B)
#         Sigma = K_II - K_IJ @ temp  # shape (B, B)
#         # 保证对称性
#         Sigma = (Sigma + Sigma.t()) / 2.0
#         inv_Sigma = torch.linalg.inv(Sigma)
#         logdet_Sigma = torch.logdet(Sigma)
#         # 对于每个输出维度 d，对应的 A[:, q, d] 服从高斯分布
#         # 这里可以向量化地对所有 d 分别计算条件均值
#         # 令 Y = A[:, q, :], shape (T, D)
#         Y = A[:, q, :]                     # shape (T, D)
#         # 取出条件部分 Y_J = Y[J, :], shape (T-B, D)
#         Y_J = Y[J_tensor, :]               # shape (T-B, D)
#         # 对每个 d，求解 v = K_JJ^{-1} y_J  (这里可以用 batched solve)
#         # 令 v = torch.linalg.solve(K_JJ, Y_J) 结果形状 (T-B, D)
#         v = torch.linalg.solve(K_JJ, Y_J)    # shape (T-B, D)
#         # 条件均值 μ = K_IJ @ v, 形状 (B, D)
#         mu = K_IJ @ v                      # shape (B, D)
        
#         for d in range(D):
#             candidate = A_batch[:, q, d]   # shape (B,)
#             # 取出对应的均值： mu_d, shape (B,)
#             mu_d = mu[:, d]
#             diff = candidate - mu_d        # shape (B,)
#             # 计算二次型: diff^T Σ^{-1} diff
#             quadratic = diff @ (inv_Sigma @ diff)
#             # 多元正态对数密度（维度 B）：
#             logpdf = -0.5 * (quadratic + logdet_Sigma + B * math.log(2 * math.pi))
#             total_logp = total_logp + logpdf
#     return total_logp


def log_prior_A_batch_t_torch_optimized_vectorized(A_batch, t_batch, A, X_test, sigma_a, sigma_q, K_cache_A=None):
    """
    向量化计算一批候选时间步 t_batch 的先验对数概率。
    
    参数：
      A_batch: 候选更新的 A 值，形状 (B, Q, D)，其中 B 为候选数（batch size）
      t_batch: list 或 tensor，长度 B，表示待更新的时间步索引
      A: 原始完整的 A，形状 (T, Q, D)
      X_test: 用于计算核矩阵的测试数据
      sigma_a: 标量
      sigma_q: 长度为 Q 的张量或列表，对应每个潜在维度的参数
      K_cache_A: 可选，如果提供则为一个包含 Q 个核矩阵的字典（或列表），每个形状为 (T, T)
    
    返回：
      一个标量，表示对所有 q 和 d 的先验对数概率之和。
    """
    T, Q, D = A.shape
    B = A_batch.shape[0]
    # 确保 t_batch 为 list
    if not isinstance(t_batch, list):
        t_batch = t_batch.tolist()
    full_idx = set(range(T))
    I = t_batch             # 待更新索引集合，长度 B
    J = sorted(list(full_idx - set(I)))   # 剩余部分，长度 T-B
    I_tensor = torch.tensor(I, dtype=torch.long, device=A.device)   # shape (B,)
    J_tensor = torch.tensor(J, dtype=torch.long, device=A.device)   # shape (T-B,)
    
    # 对于每个 q，计算核矩阵 K (T x T)，并堆叠成形状 (Q, T, T)
    if K_cache_A is None:
        K_list = [compute_K_A_torch(X_test, sigma_a, sigma_q[q]) for q in range(Q)]
    else:
        K_list = [K_cache_A[q] for q in range(Q)]
    K_all = torch.stack(K_list, dim=0)   # shape (Q, T, T)
    
    # 提取子矩阵：对于所有 q 同时计算
    # K_IJ: (Q, B, T-B)
    K_IJ = K_all[:, I_tensor, :][:, :, J_tensor]
    # K_II: (Q, B, B)
    K_II = K_all[:, I_tensor, :][:, :, I_tensor]
    # K_JJ: (Q, T-B, T-B)
    K_JJ = K_all[:, J_tensor, :][:, :, J_tensor]
    
    # 将 A 重排列成 (Q, T, D)
    A_all = A.permute(1, 0, 2)   # shape (Q, T, D)
    # 对于每个 q，取出条件部分 A[q, J, :], 得到 (Q, T-B, D)
    A_all_J = A_all[:, J_tensor, :]   # shape (Q, T-B, D)
    # 对于每个 q，解线性系统 K_JJ v = A_all_J 以获得 v，形状 (Q, T-B, D)
    v = torch.linalg.solve(K_JJ, A_all_J)
    # 计算条件均值：mu = K_IJ @ v，形状 (Q, B, D)
    mu = torch.bmm(K_IJ, v)
    
    # 计算条件协方差：Sigma = K_II - K_IJ K_JJ^{-1} K_{J,I}
    # 首先，解线性系统：temp = solve(K_JJ, K_IJ^T) 得到 (Q, T-B, B)
    temp = torch.linalg.solve(K_JJ, K_IJ.transpose(1, 2))
    Sigma = K_II - torch.bmm(K_IJ, temp)  # (Q, B, B)
    # 保证对称性
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))
    # 计算条件协方差的逆和对数行列式（batched over q）
    inv_Sigma = torch.linalg.inv(Sigma)   # shape (Q, B, B)
    logdet_Sigma = torch.logdet(Sigma)      # shape (Q,)
    
    # 将候选 A_batch 调整为 (Q, B, D)
    candidate_all = A_batch.permute(1, 0, 2)  # (Q, B, D)
    # 计算 diff = candidate - mu, 形状 (Q, B, D)
    diff = candidate_all - mu
    # 计算每个 q 对每个 d 的二次项：
    # 先计算 prod = inv_Sigma @ diff，形状 (Q, B, D)
    prod = torch.bmm(inv_Sigma, diff)
    # 对 B 维度求内积：quadratic = sum(diff * prod, dim=1)，形状 (Q, D)
    quadratic = (diff * prod).sum(dim=1)
    
    # 对于每个 q 和每个 d，计算对数密度：
    # logpdf = -0.5*(quadratic + logdet_Sigma[q] + B*log(2pi))
    logpdf = -0.5 * (quadratic + logdet_Sigma.unsqueeze(1) + B * math.log(2 * math.pi))
    # 最后对所有 q 和 d 求和
    total_logp = logpdf.sum()
    return total_logp


def log_prior_A_t_torch_optimized(A_t_candidate, t, A, X_test, sigma_a, sigma_q, K_cache_A=None):
    """
    优化后的 log_prior_A_t_torch：
      - 如果传入 K_cache_A（一个字典或直接一个矩阵），则直接使用预计算的 K 矩阵，
        否则计算 compute_K_A_torch。
    """
    T, Q, D = A.shape
    logp = 0.0
    # 若没有缓存，则为每个 q 计算一次，并构建缓存字典
    if K_cache_A is None:
        K_cache_A = {q: compute_K_A_torch(X_test, sigma_a, sigma_q[q]) for q in range(Q)}
    # 对于每个潜在维度 q，向量化计算所有 d
    for q in range(Q):
        K = K_cache_A[q]  # (T, T)
        A_q = A[:, q, :]  # shape (T, D)
        # 选择除 t 外的所有索引
        mask = torch.ones(T, dtype=torch.bool, device=A.device)
        mask[t] = False
        A_minus = A_q[mask, :]  # shape (T-1, D)
        k_tt = K[t, t]         # 标量
        k_t_other = K[t, mask]   # shape (T-1,)
        K_minus = K[mask][:, mask]  # shape (T-1, T-1)
        
        sol = torch.linalg.solve(K_minus, A_minus)  # (T-1, D)
        cond_mean = k_t_other @ sol  # (D,)
        cond_var = k_tt - k_t_other @ torch.linalg.solve(K_minus, k_t_other)
        cond_var = torch.clamp(cond_var, min=1e-6)
        
        diff = A_t_candidate[q, :] - cond_mean  # (D,)
        logp += -0.5 * (D * torch.log(2 * math.pi * cond_var) + torch.sum(diff**2) / cond_var)
    return logp


@torch.jit.script
def compute_jumpGP_kernel_torch_script(zeta, logtheta):
    # 这里可以将 compute_jumpGP_kernel_torch 的逻辑用 TorchScript 实现
    # 计算局部的cov matrix
    M, Q = zeta.size()
    ell = torch.exp(logtheta[:Q])
    s_f = torch.exp(logtheta[Q])
    sigma_n = torch.exp(logtheta[Q+1])
    zeta_scaled = zeta / ell
    diff = zeta_scaled.unsqueeze(1) - zeta_scaled.unsqueeze(0)
    dists_sq = torch.sum(diff * diff, dim=2)
    K = s_f * s_f * torch.exp(-0.5 * dists_sq)
    jitter = 1e-6 * torch.eye(M, dtype=zeta.dtype, device=zeta.device)
    return K + jitter



def log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results):
    neigh = neighborhoods[t]
    X_neighbors = neigh["X_neighbors"]
    result = jump_gp_results[t]
    model = result["jumpGP_model"]
    r = model["r"].flatten()
    r = r.to(torch.float64)
    f_t = result["f_t"]
    w = model["w"]
    ms = model["ms"]
    logtheta = model["logtheta"]
    M = X_neighbors.shape[0]
    zeta = torch.stack([transform_torch(A_t_candidate, X_neighbors[j]) for j in range(M)], dim=0)
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + torch.dot(w[1:], zeta[j])
        p = torch.sigmoid(g)
        p = torch.clamp(p, 1e-10, 1-1e-10)
        if r[j]:
            loglik_r = loglik_r + torch.log(p)
        else:
            loglik_r = loglik_r + torch.log(1 - p)
    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
    return loglik_r + loglik_f


def log_likelihood_A_t_torch_optimized(A_t_candidate, t, neighborhoods, jump_gp_results, K_cache=None):
    """
    优化后的 log_likelihood_A_t_torch：
      - 先根据候选 A_t 计算当前邻域的 zeta，
      - 如果传入了 K_cache，则直接使用预计算的局部核矩阵 K_jump，否则重新计算。
    """
    neigh = neighborhoods[t]
    X_neighbors = neigh["X_neighbors"]
    result = jump_gp_results[t]
    model = result["jumpGP_model"]
    r = model["r"].flatten().to(torch.float64)
    f_t = result["f_t"]
    w = model["w"]
    ms = model["ms"]
    logtheta = model["logtheta"]
    M = X_neighbors.shape[0]
    # 计算候选的局部表示 zeta（依赖于 A_t_candidate）
    zeta = torch.stack([transform_torch(A_t_candidate, X_neighbors[j]) for j in range(M)], dim=0)
    if K_cache is None:
        K_jump = compute_jumpGP_kernel_torch_script(zeta, logtheta)
    else:
        K_jump = K_cache  # 直接使用缓存
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
    
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + torch.dot(w[1:], zeta[j])
        p = torch.sigmoid(g)
        p = torch.clamp(p, 1e-10, 1-1e-10)
        loglik_r += torch.log(p) if r[j] else torch.log(1 - p)
    return loglik_r + loglik_f

def log_likelihood_A_batch_t_torch_optimized(A_batch, t_batch, neighborhoods, jump_gp_results, K_cache=None):
    total_loglik = 0.0
    B = A_batch.shape[0]
    for i, t in enumerate(t_batch):
        candidate = A_batch[i]  # shape (Q, D)
        # 调用原来的优化版 likelihood 计算函数（对单个 t）
        loglik = log_likelihood_A_t_torch_optimized(candidate, t, neighborhoods, jump_gp_results, K_cache=K_cache)
        total_loglik += loglik
    return total_loglik


# def log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q):
#     lp_prior = log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q)
#     lp_likelihood = log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results)
#     return lp_prior + lp_likelihood

def log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=None, K_cache_A=None):
    lp_prior = log_prior_A_t_torch_optimized(A_t_candidate, t, A, X_test, sigma_a, sigma_q, K_cache_A=K_cache_A)
    lp_likelihood = log_likelihood_A_t_torch_optimized(A_t_candidate, t, neighborhoods, jump_gp_results, K_cache=K_cache)
    return lp_prior + lp_likelihood

def log_prob_A_batch_t_torch(A_batch, t_batch, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=None, K_cache_A=None):
    lp_prior = log_prior_A_batch_t_torch_optimized_vectorized(A_batch, t_batch, A, X_test, sigma_a, sigma_q, K_cache_A=K_cache_A)
    lp_likelihood = log_likelihood_A_batch_t_torch_optimized(A_batch, t_batch, neighborhoods, jump_gp_results, K_cache=K_cache)
    return lp_prior + lp_likelihood

def sample_A_t_HMC(initial_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
                   step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50, K_cache=None, K_cache_A=None):
    def potential_fn(params):
        A_t = params["A_t"]
        return -log_prob_A_t_torch(A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=K_cache, K_cache_A=K_cache_A)
    
    init_params = {"A_t": initial_A_t.clone()}
    nuts_kernel = NUTS(potential_fn=potential_fn, target_accept_prob=0.8,
                       step_size=step_size, max_tree_depth=num_steps)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    mcmc.run()
    samples = mcmc.get_samples()
    new_A_t = samples["A_t"][0].clone()
    return new_A_t

def optimize_A_t(A_t_init, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=100, lr=0.01, K_cache=None, K_cache_A=None):
    A_t_opt = A_t_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([A_t_opt], lr=lr)
    
    def closure():
        optimizer.zero_grad()
        loss = -log_prob_A_t_torch(A_t_opt, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=K_cache, K_cache_A=K_cache_A)
        loss.backward()
        return loss

    for i in range(num_steps):
        optimizer.step(closure)
    
    return A_t_opt.detach()

def optimize_A_batch_adam(A_batch_init, t_batch, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=100, lr=0.01, K_cache=None, K_cache_A=None):
    A_batch_opt = A_batch_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([A_batch_opt], lr=lr)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = -log_prob_A_batch_t_torch(A_batch_opt, t_batch, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=K_cache, K_cache_A=K_cache_A)
        loss.backward()
        optimizer.step()
    return A_batch_opt.detach()

def optimize_A_batch_adamw(A_batch_init, t_batch, A, X_test, neighborhoods, jump_gp_results, 
                           sigma_a, sigma_q, num_steps=100, lr=0.01, clip_norm=1.0, 
                           K_cache=None, K_cache_A=None):
    """
    使用 AdamW 对 A_batch 进行优化更新。
    
    参数：
      A_batch_init: 初始候选 A 值，形状 (B, Q, D)
      t_batch: 待更新的时间步索引（list 或 tensor，长度 B）
      A: 原始完整的 A，形状 (T, Q, D)
      X_test: 用于计算核矩阵的测试数据，形状 (T, D)
      neighborhoods, jump_gp_results: 相关的结构数据
      sigma_a, sigma_q: 超参数，用于计算核矩阵
      num_steps: 总迭代步数
      lr: 初始学习率
      clip_norm: 梯度裁剪的范数上限
      K_cache, K_cache_A: 可选缓存
      
    返回：
      优化后的 A_batch，形状 (B, Q, D)
    """
    A_batch_opt = A_batch_init.clone().detach().requires_grad_(True)
    # 使用 AdamW，并设置一定的 weight decay（可根据需要调整）
    optimizer = torch.optim.AdamW([A_batch_opt], lr=lr, weight_decay=1e-2)
    # 使用 StepLR，每20步将学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = -log_prob_A_batch_t_torch(A_batch_opt, t_batch, A, X_test, neighborhoods, 
                                          jump_gp_results, sigma_a, sigma_q, 
                                          K_cache=K_cache, K_cache_A=K_cache_A)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_([A_batch_opt], clip_norm)
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:
            print(f"Step {i}: loss = {loss.item()}, current lr = {scheduler.get_last_lr()[0]}")
    return A_batch_opt.detach()



def optimize_A_batch(A_batch_init, t_batch, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=100, lr=0.01, K_cache=None, K_cache_A=None):
    A_batch_opt = A_batch_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([A_batch_opt], lr=lr)
    def closure():
        optimizer.zero_grad()
        loss = -log_prob_A_batch_t_torch(A_batch_opt, t_batch, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=K_cache, K_cache_A=K_cache_A)
        loss.backward()
        return loss
    for i in range(num_steps):
        optimizer.step(closure)
    return A_batch_opt.detach()


def gibbs_update_A_with_pyro(A, X_test, neighborhoods, jump_gp_results, K_cache_A, sigma_a, sigma_q,
                             step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50, sample_type=2):
    T, Q, D = A.shape
    A_new = A.clone()
    
    for t in range(T):
        initial_A_t = A_new[t].clone()
        K_jump_t = jump_gp_results[t]["K_jump"]
        # 预先计算当前测试点的 zeta_t 与 K_jump（使用当前 A[t] 和局部超参数）
        
        # 调用采样或优化函数时，传入预计算的 K_jump_t
        if sample_type == 1:
            new_A_t = sample_A_t_HMC(initial_A_t, t, A_new, X_test, neighborhoods, jump_gp_results,sigma_a, sigma_q, step_size, num_steps, num_samples, warmup_steps, K_cache=K_jump_t, K_cache_A=K_cache_A)
        elif sample_type == 2:
            new_A_t = optimize_A_t(initial_A_t, t, A_new, X_test, neighborhoods, jump_gp_results,sigma_a, sigma_q, num_steps=100, lr=0.01, K_cache=K_jump_t, K_cache_A=K_cache_A)
        else:
            print("Error with A sampling type")

        A_new[t] = new_A_t.clone()

        X_neighbors = neighborhoods[t]["X_neighbors"]
        zeta_t = torch.stack([transform_torch(A_new[t], x) for x in X_neighbors], dim=0)
        jump_gp_results[t]["zeta_t"] = zeta_t  # 同步更新 zeta_t
        jump_gp_results[t]["x_t_test"] = transform_torch(A_new[t], X_test[t])
        logtheta = jump_gp_results[t]["jumpGP_model"]["logtheta"]
        K_jump_t = compute_jumpGP_kernel_torch(zeta_t, logtheta)
        jump_gp_results[t]["K_jump"] = K_jump_t.clone()  # 缓存计算好的 K_jump
    
    print("Gibbs update for all A_t completed with cached K_jump.")
    return A_new

def gibbs_update_A_with_pyro_batch(A, X_test, neighborhoods, jump_gp_results, K_cache_A, sigma_a, sigma_q,
                                   batch_size=4, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50, sample_type=2):
    """
    将 A (T x Q x D) 分块更新：每次更新 batch_size 个时间步 t，
    更新的目标为 P(A[t_batch] | A[-t_batch], X, Y, σ, f)。
    
    数学上，对于固定 q 和 d，我们有：
    
        A_{I} | A_{J} ~ N( μ, Σ )
        
    其中 I 为候选 t_batch, J 为其余时间步，
    
        μ = K_{I,J} K_{J,J}^{-1} A[J]  
        Σ = K_{I,I} - K_{I,J} K_{J,J}^{-1} K_{J,I}
    
    使用 LBFGS 对候选 A_batch 进行优化，目标函数为 log_prob_A_batch_t_torch。
    """
    T, Q, D = A.shape
    A_new = A.clone()
    
    # 依次对时间轴做批量更新
    for start in tqdm(range(0, T, batch_size)):
        end = min(start + batch_size, T)
        t_batch = list(range(start, end))
        # 取出待更新的候选块，形状 (B, Q, D)
        A_batch_init = A_new[t_batch].clone()
        if sample_type == 1:
            # 如果需要 HMC 采样，可实现 sample_A_batch_HMC（此处用优化代替）
            A_batch_new = sample_A_batch_HMC(A_batch_init, t_batch, A_new, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=100, lr=0.01, K_cache=None, K_cache_A=K_cache_A)
        elif sample_type == 2:
            A_batch_new = optimize_A_batch_adam(A_batch_init, t_batch, A_new, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=100, lr=step_size, K_cache=None, K_cache_A=K_cache_A)
        else:
            raise ValueError("Error with A sampling type")
        # 更新 A_new 对应的时间步
        A_new[t_batch] = A_batch_new.clone()
        # 对每个更新的 t，重新计算邻域表示和 jumpGP 核矩阵
        for i, t in enumerate(t_batch):
            X_neighbors = neighborhoods[t]["X_neighbors"]
            zeta_t = torch.stack([transform_torch(A_new[t], x) for x in X_neighbors], dim=0)
            jump_gp_results[t]["zeta_t"] = zeta_t
            jump_gp_results[t]["x_t_test"] = transform_torch(A_new[t], X_test[t])
            logtheta = jump_gp_results[t]["jumpGP_model"]["logtheta"]
            K_jump_t_new = compute_jumpGP_kernel_torch(zeta_t, logtheta)
            jump_gp_results[t]["K_jump"] = K_jump_t_new.clone()
    print("Batched Gibbs update completed.")
    return A_new

@torch.jit.script
def compute_K_A_torch(X, sigma_a, sigma_q_value):
    # X: tensor of shape (T, D)
    T = X.shape[0]
    X1 = X.unsqueeze(1)
    X2 = X.unsqueeze(0)
    dists = torch.sum((X1 - X2)**2, dim=2)
    return sigma_a**2 * torch.exp(-dists / (2 * sigma_q_value**2))

def potential_fn_theta(theta, A, X_test):
    """
    theta: tensor of shape (Q+1,), where theta[0] = log(sigma_a), theta[1:] = log(sigma_q)
    A: tensor of shape (T, Q, D)
    X_test: tensor of shape (T, D)
    
    返回负的 log 后验（潜在 GP 超参数的目标函数，不含常数项）
    """
    T, Q, D = A.shape
    sigma_a = torch.exp(theta[0])
    sigma_q = torch.exp(theta[1:])  # 长度 Q
    
    log_lik = 0.0
    for q in range(Q):
        K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
        # 加上 jitter
        jitter = 1e-6 * torch.eye(T, device=X_test.device, dtype=X_test.dtype)
        K = K + jitter
        sign, logdet = torch.slogdet(K)
        if sign.item() <= 0:
            K = K + 1e-6 * torch.eye(T, device=X_test.device, dtype=X_test.dtype)
            sign, logdet = torch.slogdet(K)
        invK = torch.inverse(K)
        for d in range(D):
            vec = A[:, q, d]
            quad = vec @ (invK @ vec)
            log_lik = log_lik - 0.5 * (logdet + quad)
    # log prior for theta
    # 对于 sigma_a: log p(theta_a) = - (alpha_a+1)*theta_a - beta_a/exp(theta_a) + theta_a
    log_prior = -alpha_a * theta[0] - beta_a / torch.exp(theta[0])
    for q in range(Q):
        log_prior = log_prior - (alpha_q + 1) * theta[1+q] - beta_q / torch.exp(theta[1+q]) + theta[1+q]
    log_post = log_lik + log_prior
    return -log_post  # potential function

def sample_theta_HMC(A, X_test, initial_theta, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    利用 Pyro 的 NUTS 对 theta 进行采样
    initial_theta: tensor of shape (Q+1,)
    A: tensor of shape (T, Q, D)
    X_test: tensor of shape (T, D)
    返回更新后的 theta（tensor of shape (Q+1,)）
    """
    def potential_fn(params):
        theta = params["theta"]
        return potential_fn_theta(theta, A, X_test)
    
    init_params = {"theta": initial_theta.clone()}
    nuts_kernel = mcmc.NUTS(potential_fn=potential_fn, target_accept_prob=0.8,
                            step_size=step_size, max_tree_depth=num_steps)
    mcmc_run = mcmc.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    mcmc_run.run()
    samples = mcmc_run.get_samples()
    new_theta = samples["theta"][0].clone()
    return new_theta

def gibbs_update_hyperparams(A, X_test, initial_theta, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    对 GP 超参数 theta = [log(sigma_a), log(sigma_q[0]), ..., log(sigma_q[Q-1])] 进行采样更新。
    A: (T, Q, D)
    X_test: (T, D)
    initial_theta: tensor of shape (Q+1,)
    返回更新后的 theta 以及 sigma_a, sigma_q
    """
    new_theta = sample_theta_HMC(A, X_test, initial_theta, step_size, num_steps, num_samples, warmup_steps)
    sigma_a_new = torch.exp(new_theta[0])
    sigma_q_new = torch.exp(new_theta[1:])
    return new_theta, sigma_a_new, sigma_q_new


# 定义局部目标函数，用于采样 φₜ = [w, logtheta] 的联合更新
# def potential_local_single(phi, jump_result):
#     """
#     phi: tensor of shape (2Q+3,), 其中 phi[:1+Q] 对应 w, phi[1+Q:] 对应 logtheta.
#     t: 测试点索引
#     jump_gp_results: 列表，每个元素为一个字典，其中 "jumpGP_model" 包含:
#          - "r": tensor of shape (M, 1), 布尔型
#          - "f_t": tensor of shape (M,)
#          - "zeta_t": tensor of shape (M, Q)
#          - "ms": scalar, 即 m_t
#     返回：潜在函数值 (标量)，即负的目标 log 概率（不含常数项）。
#     """
#     # 从 phi 中拆分出 w 和 logtheta
#     Q = jump_result["zeta_t"].shape[1]
#     M = jump_result["zeta_t"].shape[0]
#     w = phi[:1+Q]
#     logtheta = phi[1+Q:]
    
#     model = jump_result["jumpGP_model"]
#     # 取 r, f_t, zeta_t, ms
#     r = model["r"].flatten().to(torch.float64)  # shape (M,)
#     f_t = jump_result["f_t"].to(torch.float64)  # shape (M,)
#     zeta = jump_result["zeta_t"].to(torch.float64)  # shape (M, Q)
#     ms = model["ms"]
    
#     # 计算 membership likelihood
#     loglik_r = 0.0
#     for j in range(M):
#         g = w[0] + torch.dot(w[1:], zeta[j])
#         if not torch.isfinite(g):
#             print("Non-finite g detected:", g)
#         g = torch.clamp(g, -100, 100)
#         p = torch.sigmoid(g)
#         p = torch.clamp(p, 1e-10, 1-1e-10)
#         if r[j]:
#             loglik_r = loglik_r + torch.log(p)
#         else:
#             loglik_r = loglik_r + torch.log(1 - p)
    
#     # 计算 f_t 的似然

#     K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)
#     mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    
#     # def log_multivariate_normal_pdf_torch(x, mean, cov):
#     #     cov = cov.to(torch.float64)
#     #     diff = x - mean
#     #     sign, logdet = torch.slogdet(cov)
#     #     if sign.item() <= 0:
#     #         cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
#     #         sign, logdet = torch.slogdet(cov)
#     #     return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))
    
#     loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)

#     y_t = jump_result["y_neigh"]
#     sigma_n = torch.exp(logtheta[-1])  # 噪声标准差
#     sigma_n = torch.clamp(sigma_n, min=1e-1)

#     loglik_y = -0.5 * torch.sum((y_t - f_t)**2) / sigma_n**2 - M * torch.log(sigma_n) - 0.5 * M * math.log(2 * math.pi)
#     # print("try to find loglikey:", torch.sum((y_t - f_t)**2), torch.log(sigma_n))

    
#     # 定义先验项：
#     # 对 w: 假设 p(w) ~ N(0, I)，则 log p(w) = -0.5 * ||w||^2 (忽略常数)
#     log_prior_w = -0.5 * torch.sum(w**2)
#     # 对 logtheta: 假设 p(logtheta) ~ N(0, I)
#     log_prior_logtheta = -0.5 * torch.sum(logtheta**2)
    
#     log_prior = log_prior_w + log_prior_logtheta
    
#     # 目标 log 后验为： loglik_r + loglik_f + log_prior
#     log_post = loglik_r + loglik_f + loglik_y + log_prior
#     if not torch.isfinite(log_post):
#         return torch.tensor(1e6, dtype=torch.float64)
#     # print(loglik_r, loglik_f, loglik_y, log_prior)
#     return -log_post  # 返回负的 log 后验，作为 potential 函数

def potential_local_single(phi, jump_result, U=0.1):
    # 从 phi 中拆分出 w 和 logtheta
    Q = jump_result["zeta_t"].shape[1]
    M = jump_result["zeta_t"].shape[0]
    w = phi[:1+Q]
    logtheta = phi[1+Q:]
    
    model = jump_result["jumpGP_model"]
    r = model["r"].flatten().to(torch.float64)
    f_t = jump_result["f_t"].to(torch.float64)
    zeta = jump_result["zeta_t"].to(torch.float64)
    ms = model["ms"]
    
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + torch.dot(w[1:], zeta[j])
        if not torch.isfinite(g):
            print("Non-finite g detected:", g)
        g = torch.clamp(g, -100, 100)
        p = torch.sigmoid(g)
        p = torch.clamp(p, 1e-10, 1-1e-10)
        if r[j]:
            loglik_r = loglik_r + torch.log(p)
        else:
            loglik_r = loglik_r + torch.log(1 - p)
    
    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
    
    y_t = jump_result["y_neigh"]
    sigma_n = torch.exp(logtheta[-1])
    sigma_n = torch.clamp(sigma_n, min=1e-1)
    
    loglik_y = -0.5 * torch.sum((y_t - f_t)**2) / sigma_n**2 - M * torch.log(sigma_n) - 0.5 * M * math.log(2 * math.pi)
    # loglik_y = 0.0
    # for j in range(M):
    #     if r[j] >= 0.5:
    #         diff = y_t[j] - f_t[j]
    #         loglik_y += -0.5 * ((diff**2)/(sigma_n**2) + math.log(2 * math.pi * sigma_n**2))
    #     else:
    #         loglik_y += math.log(U)

    log_prior_w = -0.5 * torch.sum(w**2)
    log_prior_logtheta = -0.5 * torch.sum(logtheta**2)
    log_prior = log_prior_w + log_prior_logtheta
    
    log_post = loglik_r + loglik_f + loglik_y + log_prior
    if not torch.isfinite(log_post):
        # 修改这里，返回的张量依赖于 log_post，从而保持计算图
        return log_post * 0.0 + 1e6
    return -log_post



def update_r_for_test_point(jump_result, U=0.1):
    """
    对单个测试点 t 的邻域 membership 指示变量 r 进行更新。
    
    输入 jump_result 是一个字典，其中包含：
      - "y_neigh": tensor of shape (M,) —— 观测响应
      - "f_t": tensor of shape (M,) —— 局部 GP 潜变量
      - "zeta_t": tensor of shape (M, Q) —— 经 A_t 变换后的邻域特征
      - "jumpGP_model": 字典，至少包含：
             "w": tensor of shape (1+Q,), 局部 membership 参数
             "logtheta": tensor of shape (Q+2,), 局部 GP 核超参数（其中最后一个元素用于噪声标准差）
             "r": tensor of shape (M, 1) —— 当前的 membership 指示变量（将被更新）
    U: 离群点似然常数
    返回更新后的 jump_result，其中 jumpGP_model["r"] 已更新为新的指示变量。
    """
    y = jump_result["y_neigh"]   # (M,)
    f = jump_result["f_t"]       # (M,)
    zeta = jump_result["zeta_t"] # (M, Q)
    model = jump_result["jumpGP_model"]
    w = model["w"]               # (1+Q,)
    logtheta = model["logtheta"] # (Q+2,)
    # 根据要求，噪声标准差 sigma = exp(logtheta[-1])
    sigma = torch.exp(logtheta[-1])
    sigma2 = sigma**2
    M = y.shape[0]
    new_r = torch.zeros(M, dtype=torch.bool, device=y.device)
    for j in range(M):
        # 计算 g = w[0] + dot(w[1:], zeta[j])
        g = w[0] + torch.dot(w[1:], zeta[j])
        g = torch.clamp(g, -100, 100)
        p_mem = torch.sigmoid(g)
        p_mem = torch.clamp(p_mem, 1e-10, 1-1e-10)
        # 计算正态似然 L1 = N(y_j | f_j, sigma^2)
        L1 = (1.0 / torch.sqrt(2 * math.pi * sigma2)) * torch.exp(-0.5 * ((y[j] - f[j])**2 / sigma2))
        numerator = p_mem * L1
        denominator = numerator + (1 - p_mem) * U
        # p_z = numerator / (denominator + 1e-10)
        p_z = torch.clamp(numerator / (denominator + 1e-10), 1e-10, 1-1e-10)
        new_r[j] = (torch.rand(1, device=y.device) < p_z).bool().item()
    new_r = new_r.unsqueeze(1)  # 转换为 (M,1)
    model["r"] = new_r.clone()
    jump_result["jumpGP_model"] = model
    return jump_result

def sample_f_t_torch(jump_result):
    """
    对单个测试点 t 的 f^t 进行逐坐标 Gibbs 更新：
      对每个邻域点 j，固定其他 f 的值，计算 f_j 的后验分布，然后从中采样更新 f_j。
    
    输入 jump_result 为字典，必须包含：
      - "jumpGP_model": 字典，其中 "ms" 为 m_t, "logtheta" 为局部 GP 超参数（形状 (Q+2,)），
           其中最后一个元素用于噪声标准差 sigma_t = exp(logtheta[-1])。
           另外，"r" 为布尔张量，形状 (M,1)。
      - "zeta_t": tensor of shape (M, Q) —— 经过 A_t 变换后的邻域输入
      - "y_neigh": tensor of shape (M,) —— 邻域响应
      - "f_t": tensor of shape (M,) —— 当前 f^t 值
    返回更新后的 f^t (tensor of shape (M,))
    """
    model = jump_result["jumpGP_model"]
    m_t = model["ms"]  # scalar, GP 均值
    logtheta = model["logtheta"]  # tensor of shape (Q+2,)
    sigma_t = torch.exp(logtheta[-1])
    sigma2 = sigma_t**2

    # r: (M,1) 转换为 1d float mask (1 if True, 0 if False)
    r = model["r"].flatten().to(torch.float64).float()  # shape (M,)
    y = jump_result["y_neigh"].to(torch.float64)  # (M,)
    zeta = jump_result["zeta_t"].to(torch.float64)  # (M, Q)
    f_current = jump_result["f_t"].to(torch.float64)  # (M,)

    # 打印初始 f_current 状态
    if torch.any(torch.isnan(f_current)):
        print("Warning: f_current contains NaN:", f_current)

    M = zeta.shape[0]
    
    # 计算协方差矩阵 C（先验协方差）
    C = compute_jumpGP_kernel_torch(zeta, logtheta)
    if torch.any(torch.isnan(C)):
        print("Warning: Covariance matrix C contains NaN.")
    
    # 为了便于计算逐坐标条件分布，我们对每个 j 分块计算：
    f_new = f_current.clone()
    for j in range(M):
        # 令 I = {0,...,M-1} \ {j}
        idx = [i for i in range(M) if i != j]
        idx_tensor = torch.tensor(idx, dtype=torch.long, device=zeta.device)
        
        # 从 C 中提取相关块
        C_jj = C[j, j]  # 标量
        C_jI = C[j, idx_tensor]  # (M-1,)
        C_II = C[idx_tensor][:, idx_tensor]  # (M-1, M-1)
        
        # 调试打印：检查当前 j 下相关矩阵是否存在 NaN 或极端值
        if torch.isnan(C_jj):
            print(f"Warning: C[{j},{j}] is NaN.")
        if torch.any(torch.isnan(C_jI)):
            print(f"Warning: C[{j}, I] contains NaN, indices:", idx)
        if torch.any(torch.isnan(C_II)):
            print(f"Warning: Submatrix C_II for index {j} contains NaN.")
        
        # 先验条件分布： f_j | f_{-j} ~ N(m_prior, v_prior)
        f_minus = f_new[idx_tensor]
        try:
            invC_II = torch.inverse(C_II)
        except RuntimeError as e:
            print(f"Error in inverting C_II for index {j}: {e}")
            invC_II = torch.pinverse(C_II)
        
        m_prior = m_t + (C_jI @ (invC_II @ (f_minus - m_t * torch.ones_like(f_minus))))
        v_prior = C_jj - (C_jI @ (invC_II @ C_jI.T))
        
        # 调试打印 m_prior 和 v_prior
        if torch.isnan(m_prior):
            print(f"Warning: m_prior is NaN at index {j}. f_minus: {f_minus}, m_t: {m_t}")
        if torch.isnan(v_prior) or v_prior <= 0:
            print(f"Warning: v_prior is invalid at index {j}. v_prior: {v_prior}, C_jj: {C_jj}, C_jI: {C_jI}")
            v_prior = torch.clamp(v_prior, min=1e-6)
        
        # 如果 r[j]==1，则引入观测似然
        if r[j] > 0.5:
            v_post = 1.0 / (1.0 / v_prior + 1.0 / sigma2)
            m_post = v_post * (m_prior / v_prior + y[j] / sigma2)
        else:
            v_post = v_prior
            m_post = m_prior
        
        # 调试打印 m_post 和 v_post
        if torch.isnan(m_post) or torch.isnan(v_post):
            print(f"Warning: m_post or v_post is NaN at index {j}. m_post: {m_post}, v_post: {v_post}")
        
        # 采样 f_j ~ N(m_post, v_post)
        try:
            sample = m_post + torch.sqrt(v_post) * torch.randn(1, device=zeta.device, dtype=zeta.dtype)
        except Exception as e:
            print(f"Error sampling at index {j}: {e}. m_post: {m_post}, sqrt(v_post): {torch.sqrt(v_post)}")
            sample = m_post  # 暂时赋值 m_post
        if torch.isnan(sample):
            print(f"Warning: Sampled f[{j}] is NaN.")
        f_new[j] = sample
    return f_new



def sample_local_hyperparams_single(initial_phi, jump_result, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    利用 Pyro 的 NUTS 对单个测试点的局部超参数 φₜ = [w, logtheta] 进行采样更新。
    initial_phi: tensor of shape (2Q+3,)
    jump_result: 对应测试点的 jump_gp_results 元素
    返回更新后的 φₜ
    """
    def potential_fn(params):
        phi = params["phi"]
        return potential_local_single(phi, jump_result)
    
    init_params = {"phi": initial_phi.clone()}
    # init_params = {"phi": initial_phi.clone().detach().requires_grad_(True)}

    pyro.clear_param_store()
    kernel = mcmc.NUTS(potential_fn=potential_fn, target_accept_prob=0.8,
                       step_size=step_size, max_tree_depth=num_steps)
    mcmc_run = mcmc.MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    mcmc_run.run()
    samples = mcmc_run.get_samples()
    new_phi = samples["phi"][0].clone()
    return new_phi

# -------------------------
# 2. 对单个测试点局部更新（更新局部超参数、membership r 和 f_t）
# -------------------------
def update_one_test_point(jump_result, U=0.1, local_step_size=0.01, local_num_steps=10, local_num_samples=1, local_warmup_steps=50):
    """
    对单个测试点进行局部更新：
      1. 更新局部超参数 φₜ = [w, logtheta]
      2. 更新 membership 指示变量 r（调用 update_r_for_test_point）
      3. 更新 f^t（逐坐标 Gibbs 更新）
    """
    Q_val = jump_result["zeta_t"].shape[1]
    initial_phi = torch.cat((jump_result["jumpGP_model"]["w"].flatten(), 
                               jump_result["jumpGP_model"]["logtheta"].flatten()), dim=0)
    new_phi = sample_local_hyperparams_single(initial_phi, jump_result, 
                                               step_size=local_step_size, 
                                               num_steps=local_num_steps, 
                                               num_samples=local_num_samples, 
                                               warmup_steps=local_warmup_steps)
    new_w = new_phi[:1+Q_val]
    new_logtheta = new_phi[1+Q_val:]
    jump_result["jumpGP_model"]["w"] = new_w.clone()
    jump_result["jumpGP_model"]["logtheta"] = new_logtheta.clone()
    print(f"Test point {jump_result['test_index']}: Local hyperparameters updated.")

    # 计算新的 K_jump
    new_K_jump = compute_jumpGP_kernel_torch(jump_result["zeta_t"], new_logtheta)
    jump_result["K_jump"] = new_K_jump.clone()

    
    # 更新 membership 指示 r（调用已定义的 update_r_for_test_point）
    jump_result = update_r_for_test_point(jump_result, U)
    print(f"Test point {jump_result['test_index']}: Membership indicators updated.")
    
    # 更新 f^t（调用之前定义的逐坐标 Gibbs 更新 sample_f_t_torch）
    new_f = sample_f_t_torch(jump_result)
    jump_result["f_t"] = new_f.clone()
    print(f"Test point {jump_result['test_index']}: f_t updated.")

    # 更新 ms (从后验分布采样)
    K_jump = jump_result["K_jump"]
    inv_K_jump = torch.inverse(K_jump)
    ones_vec = torch.ones(len(new_f), device=new_f.device, dtype=new_f.dtype)
    
    # 计算后验均值和方差
    mu_post = (ones_vec @ inv_K_jump @ new_f) / (ones_vec @ inv_K_jump @ ones_vec)
    sigma_post = torch.sqrt(1.0 / (ones_vec @ inv_K_jump @ ones_vec))
    
    # 采样新的 ms
    new_ms = torch.normal(mu_post, sigma_post)
    jump_result["jumpGP_model"]["ms"] = new_ms.clone()
    print(f"Test point {jump_result['test_index']}: ms updated by posterior sampling.")
    
    return jump_result


def update_one_wrapper(args):
    """
    包装器函数，用于 multiprocessing.Pool.map 调用 update_one_test_point。
    参数 args 是一个元组，格式为：
      (index, jump_gp_results, U, local_step_size, local_num_steps, local_num_samples, local_warmup_steps)
    """
    index, jump_gp_results, U, local_step_size, local_num_steps, local_num_samples, local_warmup_steps = args
    return update_one_test_point(jump_gp_results[index], U, local_step_size, local_num_steps, local_num_samples, local_warmup_steps)

def parallel_update_all_test_points(jump_gp_results, U=0.1, local_step_size=0.01, local_num_steps=10, 
                                    local_num_samples=1, local_warmup_steps=50, num_workers=4):
    """
    并行更新所有测试点对应的 jump_gp_results：对每个测试点，依次更新局部超参数、membership r 和 f，
    使用 multiprocessing.Pool 进行并行化。
    """
    # 构造参数列表，每个元素对应一个测试点的更新参数
    args_list = [(i, jump_gp_results, U, local_step_size, local_num_steps, local_num_samples, local_warmup_steps)
                 for i in range(len(jump_gp_results))]
    with Pool(processes=num_workers) as pool:
        results = pool.map(update_one_wrapper, args_list)
    return results

# -------------------------
# 4. 外层 Gibbs 采样（包括全局 A 更新、局部更新以及 GP 超参数更新）
# -------------------------
def gibbs_sampling(A, X_test, neighborhoods, jump_gp_results, K_cache_A, sigma_a, sigma_q,
                   num_iterations=10, 
                   pyro_update_params=None, local_params=None, hyper_update_params=None, U=0.1, num_workers=4):
    Q = A.shape[1]
    for it in range(num_iterations):
        print(f"\nGibbs iteration {it+1}/{num_iterations}")
        # 1. 更新全局 A（假设 gibbs_update_A_with_pyro 已定义）
        A = gibbs_update_A_with_pyro(A, X_test, neighborhoods, jump_gp_results, K_cache_A, sigma_a, sigma_q,
                                     **(pyro_update_params or {}))
        print(f"Iteration {it+1}: Global A updated.")

        # 1.1. 同步更新 jump_gp_results 中的 A_t, zeta_t 和 jumpGP_model 内的 K_jump
        T = A.shape[0]
        for t in range(T):
            # 更新当前测试点对应的全局映射 A_t
            jump_gp_results[t]["A_t"] = A[t].clone()
            # 利用更新后的 A_t 重新计算邻域中每个点的局部表示 zeta_t
            X_neighbors = neighborhoods[t]["X_neighbors"]
            zeta_t = torch.stack([transform_torch(A[t], x) for x in X_neighbors], dim=0)
            jump_gp_results[t]["zeta_t"] = zeta_t
            # 同时更新测试点经过 A_t 变换后的表示
            jump_gp_results[t]["x_t_test"] = transform_torch(A[t], X_test[t])
            # 利用新的 zeta_t 与局部超参数 logtheta 更新局部核矩阵 K_jump
            logtheta = jump_gp_results[t]["jumpGP_model"]["logtheta"]
            K_jump = compute_jumpGP_kernel_torch(zeta_t, logtheta)
            jump_gp_results[t]["K_jump"] = K_jump
        
        # 2. 并行更新所有测试点的局部参数
        jump_gp_results = parallel_update_all_test_points(jump_gp_results, U, **(local_params or {}), num_workers=num_workers)
        print(f"Iteration {it+1}: Local parameters updated for all test points.")
        
        # 3. 更新 GP 超参数 sigma_a, sigma_q（假设 gibbs_update_hyperparams 已定义）
        Q_val = sigma_q.shape[0]
        initial_theta = torch.zeros(Q_val+1, device=X_test.device)
        initial_theta[0] = torch.log(sigma_a)
        initial_theta[1:] = torch.log(sigma_q)
        new_theta, sigma_a_new, sigma_q_new = gibbs_update_hyperparams(A, X_test, initial_theta,
                                                                       **(hyper_update_params or {}))
        sigma_a = sigma_a_new
        sigma_q = sigma_q_new
        print(f"Iteration {it+1}: Global GP hyperparameters updated: sigma_a = {sigma_a.item():.4f}, sigma_q = {sigma_q}")

        K_cache_A = {q: compute_K_A_torch(X_test, sigma_a, sigma_q[q]) for q in range(len(sigma_q))}
    return A, jump_gp_results, sigma_a, sigma_q, K_cache_A


import torch
import math

def local_log_likelihood(A_t, t, neighborhoods, jump_gp_results, y_key="y_neighbors"):
    """
    计算测试点 t 的局部对数似然，包含：
      1. GP 部分对 f_t 的似然
      2. Membership 部分
      3. 观测 y 的似然
    参数：
      A_t: 全局映射参数，形状 (Q, D) 对应 t
      t: 测试点索引
      neighborhoods: 包含每个 t 的邻域数据字典，必须含 "X_neighbors" 和 (可选) y 观测，键为 y_key
      jump_gp_results: 列表，每个元素是一个字典，包含对应 t 的局部参数，如 f_t, jumpGP_model, 其中：
          - jumpGP_model 包含 w, logtheta, ms, 等；
          - f_t: 局部 GP 的当前函数值，形状 (M,)
      y_key: 用于从 neighborhoods 中获取观测 y 的键，默认 "y_neighbors"
    返回：
      一个标量，表示局部对数似然（非标准化）
    """
    # 从 neighborhoods 获取邻域数据
    neigh = neighborhoods[t]
    X_neighbors = neigh["X_neigh"]  # shape (M, D)
    if y_key in neigh:
        y_obs = neigh[y_key]  # shape (M,)
    else:
        raise ValueError("观测数据 y 未包含在 neighborhoods 中，请检查键名。")
    
    M = X_neighbors.shape[0]
    
    # 从 jump_gp_results 中获取局部参数
    result = jump_gp_results[t]
    model = result["jumpGP_model"]
    f_t = result["f_t"]  # shape (M,)
    w = model["w"]       # shape (1+Q,)
    ms = model["ms"]     # 标量
    logtheta = model["logtheta"]  # shape (Q+?); 假定最后一项为 log(sigma_n)
    sigma_n = torch.exp(logtheta[-1])
    
    # 计算局部变换：对于每个邻域点 x，计算 zeta = A_t @ x
    # 假设 transform_torch 定义为：transform_torch(A, x) = A @ x
    # 注意：A_t 的形状为 (Q, D)，x 的形状 (D,), 则 zeta 的形状 (Q,)
    zeta = torch.stack([transform_torch(A_t, X_neighbors[j]) for j in range(M)], dim=0)  # shape (M, Q)
    
    # 计算局部 GP 核矩阵 K_jump
    # 这里采用 RBF 核，假设 compute_jumpGP_kernel_torch 已定义：
    # K_jump = s_f^2 * exp(-0.5 * ||(zeta_i - zeta_j)/ell||^2) + jitter
    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)  # shape (M, M)
    
    # 1. GP 部分对 f_t 的似然（多元正态对数密度）
    mean_vec = ms * torch.ones(M, device=A_t.device, dtype=A_t.dtype)
    # 计算多元正态 log-density
    # 这里使用简单实现，注意数值稳定性
    diff = f_t - mean_vec
    sign, logdetK = torch.slogdet(K_jump)
    invK = torch.inverse(K_jump)
    quad_form = diff @ (invK @ diff)
    loglik_f = -0.5 * (quad_form + logdetK + M * math.log(2 * math.pi))
    
    # 2. Membership 部分
    loglik_r = 0.0
    for j in range(M):
        g_j = w[0] + torch.dot(w[1:], zeta[j])
        p_j = torch.sigmoid(g_j)
        # 假设 r 已经在模型中给出，取 r_j = 1 或 0
        r_j = model["r"][j]
        # 使用数值稳定的写法
        loglik_r += r_j * torch.log(p_j + 1e-10) + (1 - r_j) * torch.log(1 - p_j + 1e-10)
    
    # 3. 观测 y 的似然
    # 注意：对每个邻域点，y ~ N(f_t, sigma_n^2)
    residual = y_obs - f_t
    loglik_y = -0.5 * torch.sum((residual ** 2) / (sigma_n ** 2) + torch.log(2 * math.pi * sigma_n ** 2))
    
    total_local_loglik = loglik_f + loglik_r + loglik_y
    return total_local_loglik

# 注意：下面的 transform_torch 与 compute_jumpGP_kernel_torch 函数需要你在 utils1.py 中定义
# 例如：
# def transform_torch(A, x):
#     return A @ x
# 
# def compute_jumpGP_kernel_torch(zeta, logtheta):
#     # 使用 RBF 核：假定 ell = exp(logtheta[:Q]), s_f = exp(logtheta[Q])
#     # 这里简单实现：
#     M = zeta.shape[0]
#     Q = zeta.shape[1]
#     ell = torch.exp(logtheta[:Q])
#     s_f = torch.exp(logtheta[Q])
#     diff = zeta.unsqueeze(1) - zeta.unsqueeze(0)  # (M, M, Q)
#     dists_sq = torch.sum((diff / ell)**2, dim=2)
#     K = s_f**2 * torch.exp(-0.5 * dists_sq)
#     jitter = 1e-6 * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
#     return K + jitter



# def overall_logposterior(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, U=0.1):
#     """
#     计算整体后验分布的对数（非标准化）：
#       - A: 全局映射参数，形状 (T, Q, D)
#       - X_test: 测试数据，用于构造全局 GP 先验核，形状 (T, D)
#       - neighborhoods: 字典或列表，包含每个 t 的邻域数据（用于局部 GP 部分）
#       - jump_gp_results: 列表，每个元素包含局部跳跃 GP 相关参数：f_t, r, w, logtheta, ms, 等
#       - sigma_a: 标量，全局 GP 超参数
#       - sigma_q: 长度为 Q 的张量或列表，全局 GP 超参数
#     返回：整体后验的 log likelihood（对数后验值，忽略常数项）
#     """
#     T = A.shape[0]
#     total_logpost = 0.0
    
#     # 1. 对每个 t，累计局部部分和全局 A 条件先验
#     for t in range(T):
#         # 局部部分：调用 log_prob_A_t_torch（内部已经包含了局部 GP likelihood 与 membership likelihood，
#         # 以及全局 A 的条件先验 p(A_t|A_{-t},X,σ)）
#         logp_t = log_prob_A_t_torch(A[t], t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
#         total_logpost += logp_t

#         # 如果你的数据观测 y_t 的似然在局部更新中没有包含，可以另外加上观测项
#         # 例如：
#         # y_t = neighborhoods[t]["y_neighbors"]  # 或其他数据对应字段
#         # f_t = jump_result["f_t"]
#         # sigma_n = ... (从 jump_gp_results[t]["jumpGP_model"]["logtheta"] 中提取噪声标准差)
#         # logp_y = -0.5 * torch.sum((y_t - jump_gp_results[t]["f_t"])**2) / sigma_n**2 - len(y_t)*torch.log(sigma_n) - 0.5 * len(y_t)*math.log(2*math.pi)
#         # total_logpost += logp_y

#         y_obs = jump_gp_results[t]["y_neigh"]
#         f_t = jump_gp_results[t]["f_t"]
#         model = jump_gp_results[t]["jumpGP_model"]
#         r = model["r"].flatten().to(torch.float64)
#         logtheta = jump_gp_results[t]["jumpGP_model"]["logtheta"]
#         sigma_n = torch.exp(logtheta[-1])
#         loglik_y = 0.0
#         for j in range(len(y_obs)):
#             if r[j] >= 0.5:
#                 # r == 1: y_{tj} ~ N(f_{tj}, sigma_n^2)
#                 # log likelihood: -0.5 * ((y - f)^2/sigma_n^2 + log(2*pi*sigma_n^2))
#                 loglik_y += -0.5 * ((y_obs[j] - f_t[j])**2 / (sigma_n**2) + math.log(2 * math.pi * sigma_n**2))
#             else:
#                 # r == 0: 观测 y 直接等于常数 U，对数似然为 log(U)
#                 loglik_y += math.log(U)

#         total_logpost += loglik_y

#     # 2. 全局 A 的 GP 先验：对每个 q 和 d，假设 A(:,q,d) ~ N(0, K_A)
#     T, Q, D = A.shape
#     for q in range(Q):
#         # 计算核矩阵 K_A，全局核
#         K_A = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
#         # 加 jitter 保证数值稳定
#         jitter = 1e-6 * torch.eye(T, device=A.device, dtype=A.dtype)
#         K_A = K_A + jitter
#         invK_A = torch.inverse(K_A)
#         sign, logdetK_A = torch.slogdet(K_A)
#         for d in range(D):
#             A_qd = A[:, q, d]  # (T,)
#             quad = A_qd @ (invK_A @ A_qd)
#             total_logpost += -0.5 * (quad + logdetK_A + T * math.log(2*math.pi))
    
#     # 3. 超参数先验：对 sigma_a 与 sigma_q (用 theta = log(sigma))
#     theta_a = torch.log(torch.tensor(sigma_a))
#     log_prior_theta_a = -alpha_a * theta_a - beta_a / torch.exp(theta_a)
#     total_logpost += log_prior_theta_a

#     for q in range(Q):
#         theta_q = torch.log(torch.tensor(sigma_q[q]))
#         # 注意：此处先验写法依照 potential_fn_theta 中的形式
#         log_prior_theta_q = - (alpha_q + 1) * theta_q - beta_q / torch.exp(theta_q) + theta_q
#         total_logpost += log_prior_theta_q

#     # 4. 局部超参数先验：对每个 t 的 w 和 logtheta（假设标准正态先验）
#     for t in range(T):
#         model = jump_gp_results[t]["jumpGP_model"]
#         w = model["w"]
#         logtheta = model["logtheta"]
#         total_logpost += -0.5 * torch.sum(w**2)
#         total_logpost += -0.5 * torch.sum(logtheta**2)
#         # 对 ms 如果有先验，也可以加上（此处假设 ms 先验为常数）

#     return -total_logpost

import torch
import math

def overall_logposterior(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, U=0.1):
    r"""
    Compute the overall (unnormalized) log posterior of the global mapping parameters A.
    
    Let:
      - \(A \in \mathbb{R}^{T \times Q \times D}\) be the global mapping parameters.
      - For each test point \(t = 1,\dots,T\), let its local data be:
          - \(X_{t,\text{neighbors}} \in \mathbb{R}^{M_t \times D}\)
          - Observations: \(y_t \in \mathbb{R}^{M_t}\)
          - Local GP latent function values: \(f_t \in \mathbb{R}^{M_t}\)
          - Membership indicators: \(r_t \in \{0,1\}^{M_t}\)
          - Local hyperparameters: \(w \in \mathbb{R}^{1+Q}\), \(\log\theta \in \mathbb{R}^{L}\) (with the last element \(\log \sigma_n\)), and a mean \(m_t\).
      - The local representation is given by 
            \(\zeta_t(j) = A_t \, x_{tj}\) for each neighbor \(j\).
    
    The overall log posterior (up to additive constants) is defined as the sum of:
    
    1. **Local likelihood** (for each \(t\)):  
       \[
       \log L_{\text{local}}^{(t)} = \underbrace{\sum_{j=1}^{M_t} \Bigl[r_{tj}\log\sigma(g_{tj}) + (1-r_{tj})\log(1-\sigma(g_{tj}))\Bigr]}_{\text{Membership likelihood}}
       \]
       with 
       \[
       g_{tj} = w_0 + \bm{w}_{1:}^\top \zeta_{tj},
       \]
       plus
       \[
       \log L_{f}^{(t)} = -\frac{1}{2}(f_t - m_t\mathbf{1})^\top K_{\text{jump}}^{-1}(f_t - m_t\mathbf{1}) -\frac{1}{2}\log|K_{\text{jump}}| - \frac{M_t}{2}\log(2\pi),
       \]
       where \(K_{\text{jump}} = K_{\text{jump}}(\zeta_t,\log\theta)\), and
       \[
       \log L_{y}^{(t)} = \sum_{j=1}^{M_t} \left\{
         \begin{array}{ll}
           -\frac{1}{2}\left(\frac{(y_{tj}-f_{tj})^2}{\sigma_n^2}+\log(2\pi\sigma_n^2)\right), & \text{if } r_{tj}=1,\\[1mm]
           \log U, & \text{if } r_{tj}=0.
         \end{array}
       \right.
       \]
       Then, for each \(t\):
       \[
       \log L_{\text{local}}^{(t)} = \log L_{r}^{(t)} + \log L_{f}^{(t)} + \log L_{y}^{(t)}.
       \]
       
    2. **Global prior for A** (for each latent dimension \(q\) and output dimension \(d\)):
       Assume 
       \[
       A(:,q,d) \sim \mathcal{N}(0, K_A),
       \]
       where
       \[
       K_A(t,t') = \sigma_a^2\exp\Bigl(-\frac{1}{2\sigma_q(q)^2}\|X_t-X_{t'}\|^2\Bigr).
       \]
       Then,
       \[
       \log p(A(:,q,d)) = -\frac{1}{2}A(:,q,d)^\top K_A^{-1}A(:,q,d) -\frac{1}{2}\log|K_A| - \frac{T}{2}\log(2\pi).
       \]
       
    3. **Hyperparameter priors:**
       For global hyperparameters, letting \(\theta_a=\log\sigma_a\) and for each \(q\), \(\theta_q = \log\sigma_q(q)\):
       \[
       \log p(\theta_a) = -\alpha_a\,\theta_a - \frac{\beta_a}{\exp(\theta_a)},
       \]
       \[
       \log p(\theta_q) = -(\alpha_q+1)\theta_q - \frac{\beta_q}{\exp(\theta_q)} + \theta_q.
       \]
       
    4. **Local hyperparameter priors:**  
       Assume standard normal priors for \(w\) and \(\log\theta\):
       \[
       \log p(w) = -\frac{1}{2}\|w\|^2, \quad \log p(\log\theta) = -\frac{1}{2}\|\log\theta\|^2.
       \]
    
    The overall log posterior is then:
    \[
    \begin{aligned}
    \log p(\text{all}) \propto\; & \sum_{t=1}^T \log L_{\text{local}}^{(t)} \\
      & + \sum_{q=1}^{Q}\sum_{d=1}^{D} \log p(A(:,q,d)) \\
      & + \log p(\theta_a) + \sum_{q=1}^{Q} \log p(\theta_q) \\
      & + \sum_{t=1}^{T}\Bigl[\log p(w_t) + \log p(\log\theta_t)\Bigr].
    \end{aligned}
    \]
    
    This function returns the negative of the above sum so that minimization corresponds to finding the posterior mode.
    """
    total_logpost = 0.0
    T = A.shape[0]
    
    # --- 1. Local likelihood for each test point t ---
    for t in range(T):
        # (a) Recompute local representation:
        #   \(\zeta_t(j) = A[t] \, x_{tj}\) for each neighbor j.
        X_neighbors = neighborhoods[t]["X_neighbors"]  # shape (M_t, D)
        M_t = X_neighbors.shape[0]
        # Compute zeta_t (shape: M_t x Q)
        zeta_t = torch.stack([transform_torch(A[t], X_neighbors[j]) for j in range(M_t)], dim=0)
        
        # Update jump_gp_results with the current zeta_t (if needed)
        jump_gp_results[t]["zeta_t"] = zeta_t
        
        # (b) Retrieve local parameters from jump_gp_results[t]:
        # y_t: observations; f_t: latent function values;
        # model contains: r (membership), w, logtheta, ms.
        y_t = jump_gp_results[t]["y_neigh"]      # shape (M_t,)
        f_t = jump_gp_results[t]["f_t"]            # shape (M_t,)
        model = jump_gp_results[t]["jumpGP_model"]
        r = model["r"].flatten().to(torch.float64)  # shape (M_t,)
        w = model["w"].to(torch.float64)            # shape (1+Q,)
        logtheta = model["logtheta"].to(torch.float64)  # shape (L,), with last element = log(σ_n)
        ms = model["ms"]  # scalar
        
        # (c) Membership likelihood:
        # For each neighbor j, compute:
        #   g_j = w_0 + w_{1:}^T ζ_{tj}, then
        #   log p(r_j | ζ_j, w) = r_j log σ(g_j) + (1 - r_j) log (1 - σ(g_j))
        loglik_r = 0.0
        for j in range(M_t):
            g = w[0] + torch.dot(w[1:], zeta_t[j])
            g = torch.clamp(g, -100, 100)
            p = torch.sigmoid(g)
            p = torch.clamp(p, 1e-10, 1-1e-10)
            if r[j] >= 0.5:
                loglik_r += torch.log(p)
            else:
                loglik_r += torch.log(1 - p)
        # (d) GP likelihood for f_t:
        # \( \log L_f^{(t)} = -\frac{1}{2}(f_t - ms\,\mathbf{1})^\top K_{\text{jump}}^{-1}(f_t - ms\,\mathbf{1})
        #                - \frac{1}{2}\log|K_{\text{jump}}| - \frac{M_t}{2}\log(2\pi) \)
        K_jump = compute_jumpGP_kernel_torch(zeta_t, logtheta)
        mean_vec = ms * torch.ones(M_t, device=K_jump.device, dtype=K_jump.dtype)
        loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
        
        # (e) Observation likelihood for y_t given f_t and r_t:
        # For each neighbor j:
        #   if r_j = 1:  \(\log L_y^{(j)} = -\frac{1}{2}\left(\frac{(y_j-f_j)^2}{\sigma_n^2} + \log(2\pi\sigma_n^2)\right)\)
        #   if r_j = 0:  \(\log L_y^{(j)} = \log U\)
        sigma_n = torch.exp(logtheta[-1])
        sigma_n = torch.clamp(sigma_n, min=1e-1)
        loglik_y = 0.0
        for j in range(M_t):
            if r[j] >= 0.5:
                diff = y_t[j] - f_t[j]
                loglik_y += -0.5 * ((diff**2)/(sigma_n**2) + math.log(2 * math.pi * sigma_n**2))
            else:
                loglik_y += math.log(U)
        
        # Sum the three local parts:
        local_loglik = loglik_r + loglik_f + loglik_y
        total_logpost += local_loglik
        
    # --- 2. Global prior for A ---
    # For each latent dimension q and output dimension d:
    # \(\log p(A(:,q,d)) = -\frac{1}{2} A(:,q,d)^\top K_A^{-1} A(:,q,d) - \frac{1}{2}\log|K_A| - \frac{T}{2}\log(2\pi)\)
    T, Q, D = A.shape
    for q in range(Q):
        # Compute global kernel K_A for dimension q:
        K_A = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
        # Add jitter for stability:
        jitter = 1e-6 * torch.eye(T, device=A.device, dtype=A.dtype)
        K_A = K_A + jitter
        invK_A = torch.inverse(K_A)
        sign, logdetK_A = torch.slogdet(K_A)
        for d in range(D):
            A_qd = A[:, q, d]  # vector of length T
            quad = A_qd @ (invK_A @ A_qd)
            total_logpost += -0.5 * (quad + logdetK_A + T * math.log(2 * math.pi))
    
    # --- 3. Hyperparameter priors for sigma_a and sigma_q ---
    # Let \(\theta_a = \log\sigma_a\):
    # \(\log p(\theta_a) = -\alpha_a\,\theta_a - \frac{\beta_a}{\exp(\theta_a)}\)
    # theta_a = torch.log(torch.tensor(sigma_a, device=A.device, dtype=A.dtype))
    theta_a = torch.log(torch.tensor(sigma_a, device=A.device, dtype=A.dtype).clone().detach())

    log_prior_theta_a = -alpha_a * theta_a - beta_a / torch.exp(theta_a)
    total_logpost += log_prior_theta_a
    # For each q, let \(\theta_q = \log\sigma_q(q)\):
    # \(\log p(\theta_q) = -(\alpha_q+1)\theta_q - \frac{\beta_q}{\exp(\theta_q)} + \theta_q\)
    for q in range(Q):
        # theta_q = torch.log(torch.tensor(sigma_q[q], device=A.device, dtype=A.dtype))
        theta_q = torch.log(torch.as_tensor(sigma_q[q], device=A.device, dtype=A.dtype).clone().detach())
        log_prior_theta_q = - (alpha_q + 1) * theta_q - beta_q / torch.exp(theta_q) + theta_q
        total_logpost += log_prior_theta_q
        
    # --- 4. Local hyperparameter priors (for each t, for w and logtheta) ---
    # Assume standard normal priors:
    # \(\log p(w) = -\frac{1}{2}\|w\|^2\) and \(\log p(\log\theta) = -\frac{1}{2}\|\log\theta\|^2\)
    for t in range(T):
        model = jump_gp_results[t]["jumpGP_model"]
        w = model["w"]
        logtheta = model["logtheta"]
        total_logpost += -0.5 * torch.sum(w**2)
        total_logpost += -0.5 * torch.sum(logtheta**2)
        # If ms has a prior, it can be added here (assumed constant here).
        
    # Return negative overall log posterior (for minimization)
    return -total_logpost



import torch
import math

def log_prob_A_joint_t_torch(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q):
    """
    计算整个 A 的对数后验（不含归一化常数），其中 A 的形状为 (T, Q, D)。
    后验包括两部分：
      1. 局部数据部分：对于每个 t，使用函数 log_prob_A_t_torch（这里仍然使用原有的函数）
      2. 全局 GP 先验部分：对每个 q, d，假设 A(:,q,d) ~ N(0, K_A)
    """
    T, Q, D = A.shape
    total_logp = 0.0
    
    # 1. 累计所有 t 的局部数据似然部分
    for t in range(T):
        # 此处 log_prob_A_t_torch 的实现假设 A[t] 是候选更新，且条件部分利用的是 A（即 A 的其它部分）；
        # 但在全体更新时，我们直接将 A[t] 看作全部参数
        total_logp += log_prob_A_t_torch(A[t], t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
    
    # 2. 全局 A 的 GP 先验部分：对每个 q, d
    for q in range(Q):
        # 计算全局核矩阵 K_A：形状 (T, T)
        K_A = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
        # 加上 jitter 保证数值稳定
        jitter = 1e-6 * torch.eye(T, device=A.device, dtype=A.dtype)
        K_A = K_A + jitter
        invK_A = torch.inverse(K_A)
        sign, logdetK_A = torch.slogdet(K_A)
        for d in range(D):
            A_qd = A[:, q, d]  # shape (T,)
            quad = A_qd @ (invK_A @ A_qd)
            total_logp += -0.5 * (quad + logdetK_A + T * math.log(2*math.pi))
    
    return total_logp

def optimize_A_joint_adam(A_init, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, 
                            num_steps=100, lr=0.01):
    """
    利用 Adam 优化器对整个 A 的后验分布求 mode，
    A_init: 初始 A，形状 (T, Q, D)
    返回优化后的 A（posterior mode）。
    """
    A_opt = A_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([A_opt], lr=lr)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        # 负对数后验（要最小化负对数 posterior）
        loss = -log_prob_A_joint_t_torch(A_opt, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    
    return A_opt.detach()

# 示例调用：
# 假设 A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q 已经定义
# A_mode = optimize_A_joint_adam(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=200, lr=0.005)
