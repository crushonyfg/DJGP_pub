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

# 超参数先验参数（InvGamma 参数）
alpha_a = 2.0
beta_a = 1.0
alpha_q = 2.0
beta_q = 1.0

# =============================================================================
# 0. 定义 jumpgp_ld_wrapper，用于包装 JumpGP_LD 的输入输出
# =============================================================================
def jumpgp_ld_wrapper(zeta_t_torch, y_t_torch, xt_torch, mode, flag, device=torch.device("cpu")):
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
    zeta_np = zeta_t_torch.detach().cpu().numpy()
    y_np = y_t_torch.detach().cpu().numpy()
    xt_np = xt_torch.detach().cpu().numpy()
    
    # 调用原 JumpGP_LD 函数（不修改）
    mu_t_np, sig2_t_np, model_np, h = JumpGP_LD(zeta_np, y_np, xt_np, mode, flag)
    
    mu_t = torch.from_numpy(mu_t_np).to(device)
    sig2_t = torch.from_numpy(sig2_t_np).to(device)
    
    keys_to_keep = ['w', 'ms', 'logtheta', 'r']
    model_torch = {}
    for key in keys_to_keep:
        if key in model_np:
            val = model_np[key]
            if isinstance(val, np.ndarray):
                model_torch[key] = torch.from_numpy(val).to(device)
            elif isinstance(val, (int, float)):
                model_torch[key] = torch.tensor(val, dtype=torch.float64, device=device)
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

def transform_torch(A_t, x):
    return A_t @ x

# 计算局部 GP 核矩阵，利用 zeta 和 logtheta
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


def log_normal_pdf_torch(x, mean, var):
    return -0.5 * torch.log(2 * math.pi * var) - 0.5 * ((x - mean)**2 / var)

def log_multivariate_normal_pdf_cholesky(x, mean, cov):
    cov = cov.to(torch.float64)
    diff = x - mean
    sign, logdet = torch.slogdet(cov)
    if sign.item() <= 0:
        cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
        sign, logdet = torch.slogdet(cov)
    return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))

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


# def log_prior_A_t_torch_optimized(A_t_candidate, t, A, X_test, sigma_a, sigma_q, K_cache=None):
#     """
#     优化后的 log_prior_A_t_torch，采用缓存和向量化计算。
#     如果 K_cache 不为 None，则期望是一个字典，key 为 q，value 为 compute_K_A_torch(X_test, sigma_a, sigma_q[q]) 计算得到的矩阵。
#     """
#     T, Q, D = A.shape
#     logp = 0.0
#     # 如果没有传入缓存，则为每个 q 计算一次，并放入缓存
#     if K_cache is None:
#         K_cache = {q: compute_K_A_torch(X_test, sigma_a, sigma_q[q]) for q in range(Q)}
    
#     # 对每个潜在维度 q，向量化计算所有 d
#     for q in range(Q):
#         K = K_cache[q]  # (T, T)，已在 GPU 上（确保 X_test 已在 device 上）
#         A_q = A[:, q, :]  # shape (T, D)
#         # 选择除 t 外的所有索引
#         mask = torch.ones(T, dtype=torch.bool, device=A.device)
#         mask[t] = False
#         A_minus = A_q[mask, :]  # shape (T-1, D)
#         k_tt = K[t, t]         # 标量
#         k_t_other = K[t, mask]   # shape (T-1,)
#         K_minus = K[mask][:, mask]  # shape (T-1, T-1)
        
#         # 采用向量化方式求解 K_minus * sol = A_minus, sol: (T-1, D)
#         sol = torch.linalg.solve(K_minus, A_minus)  # (T-1, D)
#         cond_mean = k_t_other @ sol  # (D,)
#         # 注意：条件方差对于每个 d 是一样的，因为 k_tt 与 k_t_other 都是标量/vector
#         cond_var = k_tt - k_t_other @ torch.linalg.solve(K_minus, k_t_other)
#         cond_var = torch.clamp(cond_var, min=1e-6)
        
#         # 计算 log_normal_pdf 向量化：对于 d=0,...,D-1
#         diff = A_t_candidate[q, :] - cond_mean  # (D,)
#         # log_pdf = -0.5 * ( log(2*pi*cond_var)*D + sum((diff^2)/cond_var) )
#         logp += -0.5 * (D * torch.log(2 * math.pi * cond_var) + torch.sum(diff**2) / cond_var)
#     return logp

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


# def log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q):
#     lp_prior = log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q)
#     lp_likelihood = log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results)
#     return lp_prior + lp_likelihood

def log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, K_cache=None, K_cache_A=None):
    lp_prior = log_prior_A_t_torch_optimized(A_t_candidate, t, A, X_test, sigma_a, sigma_q, K_cache_A=K_cache_A)
    lp_likelihood = log_likelihood_A_t_torch_optimized(A_t_candidate, t, neighborhoods, jump_gp_results, K_cache=K_cache)
    return lp_prior + lp_likelihood



# def sample_A_t_HMC(initial_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
#                    step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
#     """
#     利用 Pyro 的 NUTS（HMC 的变种，支持自动步长调节）对索引 t 对应的 A_t 进行采样更新。
    
#     参数:
#       initial_A_t: 用于初始化 HMC 采样的 A[t]（深拷贝，避免修改原数据）
#       t: 当前更新的索引
#       A: 当前完整的参数张量，包含已经更新的部分（保证 Gibbs 更新）
#       X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q:
#           模型中所需的其他变量（确保 log_prob_A_t_torch 可用且可微）
#       step_size: NUTS 的初始步长（会在 warmup 阶段自动调整）
#       num_steps: NUTS 的最大树深度（对应每次采样的步数）
#       num_samples: 采样的样本数
#       warmup_steps: 预热步数
#     返回:
#       new_A_t: 采样得到的新的 A[t]（深拷贝，防止后续修改）
#     """
#     def potential_fn(params):
#         # 从参数字典中提取当前待更新的 A_t
#         A_t = params["A_t"]
#         # 计算负的 log 概率（保证 log_prob_A_t_torch 对 A_t 可微分）
#         return -log_prob_A_t_torch(A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
    
#     # 用传入的初始 A_t 作为初始值（使用 clone() 防止引用原始张量）
#     init_params = {"A_t": initial_A_t.clone()}
    
#     # 使用 NUTS 内核，自动调整步长。通过 step_size 和 max_tree_depth 控制采样过程。
#     nuts_kernel = NUTS(
#         potential_fn=potential_fn,
#         target_accept_prob=0.8,   # 目标接受率，可根据需要调整
#         step_size=step_size,      # 初始步长
#         max_tree_depth=num_steps  # 最大树深度，控制每次采样的步数
#     )
    
#     # 构造 MCMC 对象，传入初始参数（以字典形式），指定采样数量和预热步数
#     mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    
#     # 开始采样
#     mcmc.run()
#     samples = mcmc.get_samples()
    
#     # 提取采样得到的第一个样本作为更新后的 A_t，并使用 clone() 确保数据独立
#     new_A_t = samples["A_t"][0].clone()
#     return new_A_t

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


# def optimize_A_t(A_t_init, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, 
#                  num_steps=100, lr=0.01):
#     """
#     利用 LBFGS 优化器对 A[t] 进行 MAP 估计
#     参数:
#       A_t_init: 当前测试点 t 对应的初始 A[t] (tensor, shape (Q, D))
#       其他参数: 与 log_prob_A_t_torch 函数所需的参数相同
#       num_steps: 优化迭代次数
#       lr: 学习率
#     返回:
#       A_t_mode: 优化得到的后验众数（MAP 估计）
#     """
#     # 将 A_t 初始化为可优化参数
#     A_t_opt = A_t_init.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.LBFGS([A_t_opt], lr=lr)
    
#     def closure():
#         optimizer.zero_grad()
#         # 负的 log posterior 就是我们的损失函数
#         loss = -log_prob_A_t_torch(A_t_opt, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
#         loss.backward()
#         return loss

#     for i in range(num_steps):
#         optimizer.step(closure)
    
#     # 返回优化后的 A_t，即 MAP 估计
#     return A_t_opt.detach()

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



# def gibbs_update_A_with_pyro(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
#                              step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50, sample_type=2):
#     """
#     对参数 A 中的每个 t 进行 Gibbs 更新，每次更新时都使用最新的 A_new，从而保证 Gibbs 效果。
    
#     参数:
#       A: 原始参数张量，形状为 [T, Q, D]
#       其他参数: 与模型相关的变量
#       step_size, num_steps, num_samples, warmup_steps: 控制 NUTS 采样的参数
#       sample_type: 1(Hybrid)/2(mode)
#     返回:
#       A_new: Gibbs 更新后的 A
#     """
#     T, Q, D = A.shape
#     # 对 A 进行深拷贝，防止直接修改原始 A
#     A_new = A.clone()
    
#     # 对每个 t 进行依次更新，更新时传入当前最新的 A_new
#     for t in range(T):
#         # 当前 A_new[t] 作为初始值进行采样（使用 clone() 确保独立性）
#         initial_A_t = A_new[t].clone()
#         # 采样更新 A[t]，此处传入 A_new 保证使用最新的 Gibbs 更新结果
#         if sample_type==1:
#             new_A_t = sample_A_t_HMC(initial_A_t, t, A_new, X_test, neighborhoods, jump_gp_results,
#                                  sigma_a, sigma_q, step_size, num_steps, num_samples, warmup_steps)
#         elif sample_type==2:
#             new_A_t = optimize_A_t(initial_A_t, t, A_new, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, num_steps=100, lr=0.01)
#         else:
#             print("Error with A sampling type")


#         # 将更新后的 A[t] 存入 A_new（clone() 确保赋值的是一个独立的张量）
#         A_new[t] = new_A_t.clone()
#     print("Pyro NUTS (HMC) update performed for all A_t.")
#     return A_new

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
def potential_local_single(phi, jump_result):
    """
    phi: tensor of shape (2Q+3,), 其中 phi[:1+Q] 对应 w, phi[1+Q:] 对应 logtheta.
    t: 测试点索引
    jump_gp_results: 列表，每个元素为一个字典，其中 "jumpGP_model" 包含:
         - "r": tensor of shape (M, 1), 布尔型
         - "f_t": tensor of shape (M,)
         - "zeta_t": tensor of shape (M, Q)
         - "ms": scalar, 即 m_t
    返回：潜在函数值 (标量)，即负的目标 log 概率（不含常数项）。
    """
    # 从 phi 中拆分出 w 和 logtheta
    Q = jump_result["zeta_t"].shape[1]
    M = jump_result["zeta_t"].shape[0]
    w = phi[:1+Q]
    logtheta = phi[1+Q:]
    
    model = jump_result["jumpGP_model"]
    # 取 r, f_t, zeta_t, ms
    r = model["r"].flatten().to(torch.float64)  # shape (M,)
    f_t = jump_result["f_t"].to(torch.float64)  # shape (M,)
    zeta = jump_result["zeta_t"].to(torch.float64)  # shape (M, Q)
    ms = model["ms"]
    
    # 计算 membership likelihood
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
    
    # 计算 f_t 的似然

    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    
    # def log_multivariate_normal_pdf_torch(x, mean, cov):
    #     cov = cov.to(torch.float64)
    #     diff = x - mean
    #     sign, logdet = torch.slogdet(cov)
    #     if sign.item() <= 0:
    #         cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
    #         sign, logdet = torch.slogdet(cov)
    #     return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))
    
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)

    y_t = jump_result["y_neigh"]
    sigma_n = torch.exp(logtheta[-1])  # 噪声标准差
    sigma_n = torch.clamp(sigma_n, min=1e-3)

    loglik_y = -0.5 * torch.sum((y_t - f_t)**2) / sigma_n**2 - M * torch.log(sigma_n) - 0.5 * M * math.log(2 * math.pi)

    
    # 定义先验项：
    # 对 w: 假设 p(w) ~ N(0, I)，则 log p(w) = -0.5 * ||w||^2 (忽略常数)
    log_prior_w = -0.5 * torch.sum(w**2)
    # 对 logtheta: 假设 p(logtheta) ~ N(0, I)
    log_prior_logtheta = -0.5 * torch.sum(logtheta**2)
    
    log_prior = log_prior_w + log_prior_logtheta
    
    # 目标 log 后验为： loglik_r + loglik_f + log_prior
    log_post = loglik_r + loglik_f + loglik_y + log_prior
    if not torch.isfinite(log_post):
        return torch.tensor(1e6, dtype=torch.float64)
    return -log_post  # 返回负的 log 后验，作为 potential 函数


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

# def sample_f_t_torch(jump_result):
#     """
#     对单个测试点 t 的 f^t 进行逐坐标 Gibbs 更新：
#       对每个邻域点 j，固定其他 f 的值，计算 f_j 的后验分布，然后从中采样更新 f_j。
    
#     输入 jump_result 为字典，必须包含：
#       - "jumpGP_model": 字典，其中 "ms" 为 m_t, "logtheta" 为局部 GP 超参数（形状 (Q+2,)），
#            其中最后一个元素用于噪声标准差 sigma_t = exp(logtheta[-1]).
#            另外，"r" 为布尔张量，形状 (M,1)。
#       - "zeta_t": tensor of shape (M, Q) —— 经过 A_t 变换后的邻域输入
#       - "y_neigh": tensor of shape (M,) —— 邻域响应
#       - "f_t": tensor of shape (M,) —— 当前 f^t 值
#     返回更新后的 f^t (tensor of shape (M,))
#     """
#     model = jump_result["jumpGP_model"]
#     m_t = model["ms"]  # scalar, GP 均值
#     logtheta = model["logtheta"]  # tensor of shape (Q+2,)
#     sigma_t = torch.exp(logtheta[-1])
#     sigma2 = sigma_t**2
#     # r: (M,1) 转换为 1d float mask (1 if True, 0 if False)
#     r = model["r"].flatten().to(torch.float64).float()  # shape (M,)
#     y = jump_result["y_neigh"].to(torch.float64)  # (M,)
#     zeta = jump_result["zeta_t"].to(torch.float64)  # (M, Q)
#     f_current = jump_result["f_t"].to(torch.float64)  # (M,)
#     M = zeta.shape[0]
    
#     # 计算协方差矩阵 C（先验协方差）：
#     # C = compute_jumpGP_kernel_torch(zeta, logtheta)
#     # 此处假设 compute_jumpGP_kernel_torch 已定义，返回 (M, M) 的核矩阵
#     C = compute_jumpGP_kernel_torch(zeta, logtheta)
    
#     # 为了便于计算逐坐标条件分布，我们对每个 j 分块计算：
#     f_new = f_current.clone()
#     for j in range(M):
#         # 令 I = {0,...,M-1} \ {j}
#         idx = [i for i in range(M) if i != j]
#         idx_tensor = torch.tensor(idx, dtype=torch.long, device=zeta.device)
#         # 从 C 中提取相关块：
#         C_jj = C[j, j]  # 标量
#         C_jI = C[j, idx_tensor]  # (M-1,)
#         C_II = C[idx_tensor][:, idx_tensor]  # (M-1, M-1)
#         # 先验条件分布：
#         # f_j | f_{-j} ~ N(m_prior, v_prior)
#         # 其中 m_prior = m_t + C_jI * inv(C_II) * (f_current[idx] - m_t)
#         f_minus = f_new[idx_tensor]
#         invC_II = torch.inverse(C_II)
#         m_prior = m_t + (C_jI @ (invC_II @ (f_minus - m_t * torch.ones_like(f_minus))))
#         v_prior = C_jj - (C_jI @ (invC_II @ C_jI.T))
#         # 如果 r[j]==1，则引入观测似然：
#         if r[j] > 0.5:
#             v_post = 1.0 / (1.0 / v_prior + 1.0 / sigma2)
#             m_post = v_post * (m_prior / v_prior + y[j] / sigma2)
#         else:
#             v_post = v_prior
#             m_post = m_prior
#         # 采样 f_j ~ N(m_post, v_post)
#         f_new[j] = m_post + torch.sqrt(v_post) * torch.randn(1, device=zeta.device, dtype=zeta.dtype)
#     return f_new

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
