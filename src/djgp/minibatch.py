import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
# 将 JumpGP_code_py 所在的目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import math
import torch.nn.functional as F
from shared.utils1 import jumpgp_ld_wrapper
from tqdm import tqdm

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 预生成 Gauss–Hermite 节点和权重
_GH_POINTS = 20
_nodes_np, _weights_np = np.polynomial.hermite.hermgauss(_GH_POINTS)
_nodes   = torch.from_numpy(_nodes_np).to(device)
_weights = torch.from_numpy(_weights_np).to(device)
_factor  = 1.0 / math.sqrt(math.pi)
_LOG_2PI = math.log(2 * math.pi)

# ===== 基础函数 =====
def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

# ===== KL(q(V)||p(V)) =====
def diagnose_cholesky_failure(Kq, ell, d2, var_w, eps=1e-6):
    # 1) 最小/最大特征值
    eigs = torch.linalg.eigvalsh(Kq)
    min_eig = eigs.min().item()
    max_eig = eigs.max().item()
    print(f"[Diagnose] ell={ell:.3e}, var_w={var_w:.3e}, min_eig={min_eig:.3e}, max_eig={max_eig:.3e}")
    # 2) 检查 d2 是否有负值
    min_d2 = d2.min().item()
    print(f"[Diagnose] min squared distance (d2) = {min_d2:.3e}")
    # 3) 检查是否有重复行（Z 中有重点时 d2 会出现整行 0）
    zero_rows = (d2.abs().sum(dim=1) < eps).sum().item()
    print(f"[Diagnose] number of rows in d2 that sum≈0 (possible duplicates) = {zero_rows}")
    # 4) 建议 ell 的合理区间（可根据你数据的尺度调整）
    if ell < 1e-3:
        print("  -> ell 很小，导致 exp(-d2/ell^2) 在非对角线上都是 0，矩阵退化。")
    if ell > 1e3:
        print("  -> ell 很大，导致 kernel≈常数核，秩退化。")
    # 5) 最后加一个小 jitter 让它能跑下去
    n = Kq.size(-1)
    I = torch.eye(n, dtype=Kq.dtype, device=Kq.device)
    jitter = max(eps - min_eig, eps)
    print(f"[Diagnose] 应用 jitter = {jitter:.3e} 到对角线。")
    return torch.linalg.cholesky(Kq + jitter * I)

def kl_qp(Z: torch.Tensor,
          mu: torch.Tensor,
          sigma: torch.Tensor,
          lengthscales: torch.Tensor,
          var_w: torch.Tensor) -> torch.Tensor:
    m2, Q, D = mu.shape
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    kl = torch.tensor(0.0, device=device)
    for q in range(Q):
        ell = lengthscales[q] if lengthscales.ndim > 0 else lengthscales
        wvar = var_w[q] if var_w.ndim > 0 else var_w
        Kq = wvar * torch.exp(-0.5 * d2 / (ell**2))

        # 20250509
        # jitter = 1e-6
        I = torch.eye(Kq.size(-1), dtype=Kq.dtype, device=Kq.device)
        Kq = Kq + jitter * I
        Kq = 0.5*(Kq + Kq.T)

        # L = torch.linalg.cholesky(Kq)
        try:
            L = torch.linalg.cholesky(Kq)
        except RuntimeError as err:
            if 'positive-definite' in str(err):
                # 诊断并 fallback 到带自适应 jitter 的 Cholesky
                L = diagnose_cholesky_failure(Kq, ell, d2, wvar)
            else:
                # 其他错误继续抛
                raise
        # L = safe_cholesky(Kq)   

        Kq_inv = torch.cholesky_inverse(L)
        diag_inv = torch.diagonal(Kq_inv)
        logdet_Kq = 2.0 * torch.log(torch.diagonal(L)).sum()
        for d in range(D):
            mu_qd = mu[:, q, d]
            s2 = sigma[:, q, d].pow(2)
            trace_term = (diag_inv * s2).sum()
            quad_term = mu_qd @ (Kq_inv @ mu_qd)
            logdet_S = torch.log(s2 + 1e-12).sum()
            kl += 0.5 * (trace_term + quad_term - m2 + logdet_Kq - logdet_S)
    return kl

# ===== q(W) posterior =====
import torch

def safe_cholesky(K, init_jitter=1e-6, max_tries=5, jitter_multiplier=10.0):
    """
    对称矩阵 K 添加 jitter 做 Cholesky。
    若失败，则指数级放大 jitter 并重试 max_tries 次。
    """
    n = K.size(-1)
    jitter = init_jitter
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter * I)
        except RuntimeError as e:
            # 如果是正定失败，放大 jitter 再试
            if 'not positive-definite' in str(e):
                jitter *= jitter_multiplier
            else:
                # 其他错误，直接抛
                raise
    # 最后一次尝试，若仍失败，就让它抛出
    return torch.linalg.cholesky(K + jitter * I)

def diagnose_Kzz_and_compute_jitter(Kzz, l, var_w, ZZ2, eps=1e-6):
    # 1) 强制对称
    Kzz = 0.5 * (Kzz + Kzz.transpose(-2, -1))
    # 2) 特征值
    eigs = torch.linalg.eigvalsh(Kzz)
    λ_min = eigs.min().item()
    λ_max = eigs.max().item()
    print(f"[DiagnoseKzz] l={l:.3e}, var_w={var_w:.3e}, min_eig={λ_min:.3e}, max_eig={λ_max:.3e}")
    # 3) 检查 ZZ2 是否有负值（理论上不应该）
    min_ZZ2 = ZZ2.min().item()
    print(f"[DiagnoseKzz] min(ZZ2) = {min_ZZ2:.3e}")
    # 4) 检查是否有重复点（ZZ2 行和列都是 0）
    dup_rows = (ZZ2.abs().sum(dim=1) < eps).sum().item()
    print(f"[DiagnoseKzz] possible duplicate rows in Z: {dup_rows}")
    # 5) 自适应 jitter：保证 λ_min + jitter ≥ eps
    jitter = max(eps - λ_min, eps)
    print(f"[DiagnoseKzz] applying jitter = {jitter:.3e}")
    return jitter

jitter = 1e-5
def qW_from_qV(X: torch.Tensor,
               Z: torch.Tensor,
               mu_V: torch.Tensor,
               sigma_V: torch.Tensor,
               lengthscales: torch.Tensor,
               var_w: torch.Tensor):
    T, D = X.shape
    m2, Q, _ = mu_V.shape
    ZZ2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    XZ2 = (X.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    mu_W    = torch.empty((T, Q, D), device=device)
    sigma_W = torch.empty((T, Q, D), device=device)
    for q in range(Q):
        l = lengthscales[q]
        Kzz = var_w * torch.exp(-0.5 * ZZ2 / (l**2))

        # 20250509
        # L = safe_cholesky(Kzz)
        # jitter = 1e-6
        # I = torch.eye(Kzz.size(-1), dtype=Kzz.dtype, device=Kzz.device)
        # Kzz = Kzz + jitter * I

        Kxz = var_w * torch.exp(-0.5 * XZ2 / (l**2))

        I = torch.eye(Kzz.size(-1), dtype=Kzz.dtype, device=Kzz.device)
        Kzz = Kzz + jitter * I
        Kzz = 0.5*(Kzz + Kzz.T)
        # L = torch.linalg.cholesky(Kzz)
        try:
            L = torch.linalg.cholesky(Kzz)
        except RuntimeError as err:
            if 'positive-definite' in str(err):
                # 诊断并算出需要的 jitter
                j = diagnose_Kzz_and_compute_jitter(Kzz, l, var_w, ZZ2)
                I = torch.eye(Kzz.size(-1), dtype=Kzz.dtype, device=Kzz.device)
                L = torch.linalg.cholesky(Kzz + j*I)
            else:
                raise


        Kzz_inv = torch.cholesky_inverse(L)
        A = Kxz @ Kzz_inv
        var_prior = var_w
        diag_cross = (A * Kxz).sum(dim=1)
        U = Kzz_inv @ Kxz.t()
        for d in range(D):
            mu_W[:, q, d] = A @ mu_V[:, q, d]
            s2 = sigma_V[:, q, d].pow(2).unsqueeze(1)
            diag3 = (U.pow(2) * s2).sum(dim=0)
            sigma_W[:, q, d] = torch.sqrt(var_prior - diag_cross + diag3 + 1e-12)
    cov_W = torch.diag_embed(sigma_W.pow(2))
    return mu_W, cov_W

# ===== 批量计算 Kfu =====
def expected_Kfu(mu_W, cov_W, X, C, sigma_f):
    # mu_W [T,Q,D], cov_W diag->[T,Q,D], X [T,n,D], C [T,m1,Q]
    T, n, D = X.shape
    m1, Q = C.shape[1], C.shape[2]
    Sigma = cov_W  # diag embed already taken
    s = torch.einsum('tnd,tqde,tne->tnq', X, Sigma, X)
    den = torch.sqrt(s + 1.0)
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)
    diff = mu_proj.unsqueeze(2) - C.unsqueeze(1)
    exp_term = torch.exp(-0.5 * diff**2 / (s.unsqueeze(2)+1.0))
    num = exp_term.prod(dim=-1)
    den_prod = den.prod(dim=-1, keepdim=True)
    return sigma_f.view(-1,1,1) * num/den_prod

# ===== 批量计算 KufKfu =====
def expected_KufKfu(mu_W, cov_W, X, C, sigma_f):
    """
    Compute E_q[K_{u,f} K_{f,u}] for all regions and data points in batch.
    Args:
      mu_W:    [T,Q,D]
      cov_W:   [T,Q,D,D] diagonal covariance
      X:       [T,n,D]
      C:       [T,m1,Q]
      sigma_f: [T]
    Returns:
      Tensor of shape [T,n,m1,m1]
    """
    # Shapes
    T, n, D = X.shape
    m1, Q = C.shape[1], C.shape[2]
    # 1) s[t,i,q] = x_{t,i}^T Sigma_{t,q} x_{t,i}
    s = torch.einsum('tnd,tqde,tne->tnq', X, cov_W, X)         # [T,n,Q]
    # 2) mu_proj[t,i,q] = mu_{t,q}^T x_{t,i}
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)           # [T,n,Q]
    # 3) midpoints of inducing locations
    mid = 0.5 * (C.unsqueeze(2) + C.unsqueeze(1))             # [T,m1,m1,Q]
    # 4) diff for each t,i,l,l',q
    diff = mu_proj.unsqueeze(2).unsqueeze(3) - mid.unsqueeze(1) # [T,n,m1,m1,Q]
    # 5) exponent term
    denom = s + 1.0                                           # [T,n,Q]
    exp_term = torch.exp(-0.5 * diff**2 / denom.unsqueeze(2).unsqueeze(2)) # [T,n,m1,m1,Q]
    num = exp_term.prod(dim=-1)                              # [T,n,m1,m1]
    # 6) prior cross-kernel
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)    # [T,m1,m1]
    prior = torch.exp(-0.25 * d2)                           # [T,m1,m1]
    # 7) denominator prod over q: prod_q sqrt(s+1)
    den_prod = torch.prod(torch.sqrt(denom), dim=-1)        # [T,n]
    # 8) combine
    return (sigma_f.view(T,1,1,1)**2) * prior.unsqueeze(1) * (num / den_prod.unsqueeze(2).unsqueeze(3))

# ===== 批量 Logistic 期望 =====
def expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X):
    T, n, D = X.shape
    Q = mu_W.shape[1]
    # 1) projection of means: [T,n,Q]
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)
    # 2) mean of z: omega_0 + sum_q omega_q * mu_proj
    mu_z = omega[:, 0].unsqueeze(1) + (omega[:, 1:].unsqueeze(1) * mu_proj).sum(dim=2)
    # 3) variance term s: [T,n,Q]
    s = torch.einsum('tnd,tqde,tne->tnq', X, cov_W, X)
    # 4) tau^2: sum_q omega_q^2 * s
    tau2 = (omega[:, 1:].unsqueeze(1).pow(2) * s).sum(dim=2)
    tau_z = torch.sqrt(tau2 + 1e-12)
    # 5) compute z at GH nodes: [T,n,P]
    z = mu_z.unsqueeze(2) + math.sqrt(2) * tau_z.unsqueeze(2) * _nodes.view(1,1,-1)
    # use F.logsigmoid
    log_sig = F.logsigmoid(z)
    log_one_minus = F.logsigmoid(-z)
    # 6) weighted sum
    E1 = _factor * (_weights.view(1,1,-1) * log_sig).sum(dim=-1)
    E2 = _factor * (_weights.view(1,1,-1) * log_one_minus).sum(dim=-1)
    return E1, E2


import torch
import math

_LOG_2PI = math.log(2 * math.pi)

def kl_q_u_batch(Z: torch.Tensor,
                 mu: torch.Tensor,
                 Sigma: torch.Tensor,
                 u_var: torch.Tensor) -> torch.Tensor:
    """
    Batched KL divergence KL( q(u) || p(u) ) for T regions.

    p(u) = N(0, K) with K_ij = u_var * exp(-0.5 * ||Z_i - Z_j||^2)
    q(u) = N(mu, Sigma)

    Args:
      Z:     [T, m1, Q]    各 region 的诱导点位置
      mu:    [T, m1]       各 region q(u) 的均值
      Sigma: [T, m1, m1]   各 region q(u) 的协方差
      u_var: [T] 或 scalar 各 region 先验方差

    Returns:
      KL:    [T]           每个 region 的 KL(q||p)
    """
    T, m1, Q = Z.shape
    device = Z.device

    # 1) pairwise squared dists
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(dim=-1)  # [T, m1, m1]

    # 2) build prior cov K
    u_var_T = u_var.view(-1, 1, 1)
    K = u_var_T * torch.exp(-0.5 * d2)                        # [T, m1, m1]

    # 3) cholesky and inverse
    L = torch.linalg.cholesky(K)                              # [T, m1, m1]
    K_inv = torch.cholesky_inverse(L)                         # [T, m1, m1]
    logdet_K = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=1)  # [T]

    # 4) trace term tr(K⁻¹ Σ)
    trace_term = torch.einsum('tij,tji->t', K_inv, Sigma)      # [T]

    # 5) quadratic term μᵀ K⁻¹ μ
    Kinv_mu = torch.einsum('tij,tj->ti', K_inv, mu)           # [T, m1]
    quad_term = (mu * Kinv_mu).sum(dim=1)                      # [T]

    # 6) logdet Σ
    m1 = Sigma.size(-1)  # 提取每个 region 的维度
    I = torch.eye(m1, dtype=Sigma.dtype, device=Sigma.device)    # [m1, m1]
    Sigma = 0.5*(Sigma + Sigma.transpose(-1,-2))
    Sigma = Sigma + 1e-6*I

    sign, logdet_Sigma = torch.linalg.slogdet(Sigma)          # ([T], [T])
    assert (sign > 0).all(), "发现 Σ 非正定 或 奇异！"

    # 7) combine
    kl = 0.5 * (trace_term + quad_term - m1 + logdet_K - logdet_Sigma)
    # if kl.sum()<0:
    #     print("trace_term:", trace_term)    # ≥0
    #     print("quad_term:", quad_term)      # ≥0
    #     print("logdet_K:", logdet_K)
    #     print("logdet_Sigma:", logdet_Sigma)
    #     print("raw KL:", kl.sum())

    return kl  # [T]


import torch.nn as nn
import torch.nn.functional as F

class SigmaUParam(nn.Module):
    def __init__(self, T: int, m1: int, eps: float = 1e-6):
        super().__init__()
        self.T = T
        self.m1 = m1
        self.eps = eps
        # raw lower-triangular parameters, including diagonal
        self.raw_L = nn.Parameter(torch.randn(T, m1, m1))
        # mask for strict lower-triangular extraction
        tril = torch.tril(torch.ones(m1, m1))
        self.register_buffer('tril_mask', tril)

    def forward(self) -> torch.Tensor:
        # 1) take lower-triangular part
        L = self.raw_L * self.tril_mask       # [T, m1, m1]
        # 2) ensure diagonal positive via softplus
        raw_diag = torch.diagonal(self.raw_L, dim1=-2, dim2=-1)  # [T, m1]
        pos_diag = F.softplus(raw_diag) + self.eps                # [T, m1]
        # zero out old diag and insert positive diag
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) \
              + torch.diag_embed(pos_diag)
        # 3) construct Sigma_u = L * L^T (guaranteed PD)
        Sigma_u = L @ L.transpose(-2, -1)       # [T, m1, m1]
        return Sigma_u

def compute_ELBO(regions, V_params, u_params, hyperparams, ell=3, debug_threshold=1e7):
    """
    Batched ELBO = ∑_t [ E_q[log p(y_t|u_t,W_t)] + E_q[log p(U_t|ω,W_t)] ]
                  - KL(q(V)||p(V)) - ∑_t KL(q(u_t)||p(u_t))
    增加 debug：当 ELBO > debug_threshold 时，打印各项诊断信息。
    """
    T = len(regions)
    device = regions[0]['X'].device

    # --- stack data ---
    X       = torch.stack([r['X'] for r in regions], dim=0)       # [T,n,D]
    y       = torch.stack([r['y'] for r in regions], dim=0)       # [T,n]
    C       = torch.stack([r['C'] for r in regions], dim=0)       # [T,m1,Q]

    # u 相关
    U_logit = torch.stack([u['U_logit']     for u in u_params], dim=0).view(-1,1)  # [T,1]
    U       = torch.sigmoid(U_logit)                                            # [T,1]
    sigma_noise = torch.stack([u['sigma_noise'] for u in u_params], dim=0).view(-1)  # [T]
    sigma_k     = torch.stack([u['sigma_k']     for u in u_params], dim=0).view(-1)  # [T]
    mu_u        = torch.stack([u['mu_u']        for u in u_params], dim=0)         # [T,m1]
    # 
    eps = 1e-6
    # 1) 原始 stacking
    raw = torch.stack([u['Sigma_u'] for u in u_params], dim=0)  # [T, m1, m1]
    # 2) 提取下三角（含对角），对角做 softplus 保正
    L = torch.tril(raw)  # [T, m1, m1]
    diag = torch.diagonal(L, dim1=-2, dim2=-1)                 # [T, m1]
    pos_diag = F.softplus(diag) + eps                          # [T, m1]
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag) 
    # 3) 构造真正用来计算 KL 的正定 Sigma_u
    Sigma_u = L @ L.transpose(-2, -1)                          # [T, m1, m1]

    omega       = torch.stack([u['omega']       for u in u_params], dim=0)         # [T,Q+1]

    # hyperparams
    Z            = hyperparams['Z']            # [m2,D]
    m2 = Z.shape[0]
    X_test       = hyperparams['X_test']       # [T,D]
    lengthscales = hyperparams['lengthscales'] # [Q]
    var_w        = hyperparams['var_w']        # scalar

    # V 相关变分
    mu_V    = V_params['mu_V']    # [m2,Q,D]
    sigma_V = V_params['sigma_V'] # [m2,Q,D]

    # === 1) KL(q(V)||p(V)) ===
    KL_V = kl_qp(Z, mu_V, sigma_V, lengthscales, var_w)

    # === 2) q(W) posterior & 相关项 ===
    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)
    Kfu     = expected_Kfu(mu_W, cov_W, X, C, sigma_k)     # [T,n,m1]
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)   # [T,m1,m1]

    # === 3) KL(q(u)||p(u)) ===
    KL_u = kl_q_u_batch(C, mu_u, Sigma_u, torch.tensor(1.0, device=device)).sum()

    # === 4) 数据依赖项 E_q[log p(y|…)] + E_q[log p(U|…)] ===
    # build prior Kuu
    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)  # [T,m1,m1]
    Kuu = torch.exp(-0.5 * d2)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)

    V1   = sigma_k.view(-1,1)**2 - torch.einsum('tij,tnji->tn', Kuu_inv, KufKfu)
    v    = torch.einsum('tij,tj->ti', Kuu_inv, mu_u)   # [T,m1]
    E_fu = torch.einsum('tni,ti->tn', Kfu, v)          # [T,n]
    T3   = torch.einsum('tij,tnjk,tkm,tmi->tn',
                       Kuu_inv, KufKfu, Kuu_inv, Sigma_u)  # [T,n]
    quad = (y.pow(2) - 2*y*E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1,1)**2)

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T1 = -0.5 * _LOG_2PI - 0.5*torch.log(sigma_noise.view(-1,1)**2) + elog_sig - quad
    T2 = torch.log(U) + elog_one_minus

    region_elbo = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum(-1)  # [T]
    data_term   = region_elbo.sum()

    # === 5) 总 ELBO ===
    elbo = data_term - KL_V - KL_u

    # --- debug: 如果 ELBO 过大，打印诊断信息 ---
    if elbo.item()/T > debug_threshold:
        print("\n====== ELBO DEBUG ======")
        print(f"ELBO = {elbo.item():.3e}  (threshold {debug_threshold:.3e})")
        parts = {
            'data_term' : data_term.item(),
            'KL_V'      : KL_V.item(),
            'KL_u'      : KL_u.item(),
        }
        total_parts = sum(parts.values())
        for name, val in parts.items():
            print(f"  {name:10s} = {val:.3e}  ({100*val/total_parts:.1f}%)")
        # 打印一些超参 & 变分参数的极值
        print("\n-- hyperparams extremes --")
        print(f"lengthscales: min={lengthscales.min().item():.3e}, max={lengthscales.max().item():.3e}")
        print(f"var_w       : {var_w.item() if hasattr(var_w,'item') else var_w:.3e}")
        # V_params
        print("\n-- V_params extremes --")
        print(f"mu_V    : min={mu_V.min().item():.3e}, max={mu_V.max().item():.3e}")
        print(f"sigma_V : min={sigma_V.min().item():.3e}, max={sigma_V.max().item():.3e}")
        # u_params
        u_noises = torch.stack([u['sigma_noise'] for u in u_params])
        print("\n-- u_params extremes --")
        print(f"sigma_noise: min={u_noises.min().item():.3e}, max={u_noises.max().item():.3e}")
        # 建议检查是否有异常值
        print("=========================\n")

    # return elbo
    elbo_region = (data_term - KL_u) / T
    elbo_V      = KL_V / m2
    elbo_norm   = elbo_region - elbo_V
    # return elbo_norm
    return elbo/T

def train_vi(regions,
             V_params,
             u_params,
             hyperparams,
             lr=1e-2,
             num_steps=400,
             log_interval=20,
             lr_factor=0.5,
             lr_patience=20,
             early_stop_patience=40,
             min_lr=1e-4,
             elbo_tol=1e-3,
             regions_data=None):
    """
    增强版 train_vi，带 LR decay、early stopping，
    且仅在每 log_interval 步并且 ELBO 提升 > elbo_tol 时保存快照。
    """
    # 1. 准备所有可训练张量
    params = [V_params['mu_V'], V_params['sigma_V']]
    for u in u_params:
        params += [u['U_logit'], u['mu_u'], u['Sigma_u'],
                   u['sigma_noise'], u['omega'], u['sigma_k']]
    params += [hyperparams['lengthscales'],
               hyperparams['var_w'],
               hyperparams['Z']]

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor,
        patience=lr_patience, min_lr=min_lr)

    best_elbo = -float('inf')
    steps_no_improve = 0

    # 只有在日志点并且 ELBO 提升明显时才 clone
    best_V = best_u = best_hyp = None

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        elbo = compute_ELBO(regions, V_params, u_params, hyperparams)
        (-elbo).backward()
        optimizer.step()

        # clamp 保证正数
        with torch.no_grad():
            V_params['sigma_V'].clamp_(min=1e-1)
            hyperparams['var_w'].clamp_(min=1e-1)
            hyperparams['lengthscales'].clamp_(min=1e-6)
            for u in u_params:
                u['sigma_noise'].clamp_(min=1e-1)
                u['sigma_k'].clamp_(min=1e-1)

        val = elbo.item()
        scheduler.step(val)

        if step % 20 == 1:
            with torch.no_grad():
                if regions_data is not None:
                    regions_val, X_val, Y_val = regions_data
                    mu_p, var_p = predict_vi(regions_val, V_params, hyperparams, X_test=X_val)
                rmse = torch.sqrt(torch.mean((mu_p - Y_val)**2))
                print(f"RMSE on validation set: {rmse:.4f}")

        # 日志点才检查 best & early stop
        if step % log_interval == 0 or step == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[Step {step}/{num_steps}] ELBO={val:.4f}, LR={lr_now:.2e}")

            # 如果提升超过阈值，就保存快照
            if val > best_elbo + elbo_tol:
                best_elbo = val
                steps_no_improve = 0

                # 深拷贝当前参数
                best_V = {
                    'mu_V':    V_params['mu_V'].detach().clone().requires_grad_(True),
                    'sigma_V': V_params['sigma_V'].detach().clone().requires_grad_(True),
                }
                best_u = []
                for u in u_params:
                    best_u.append({
                        'U_logit':    u['U_logit'].detach().clone().requires_grad_(True),
                        'mu_u':       u['mu_u'].detach().clone().requires_grad_(True),
                        'Sigma_u':    u['Sigma_u'].detach().clone().requires_grad_(True),
                        'sigma_noise':u['sigma_noise'].detach().clone().requires_grad_(True),
                        'sigma_k':u['sigma_k'].detach().clone().requires_grad_(True),
                        'omega':      u['omega'].detach().clone().requires_grad_(True),
                    })
                best_hyp = {
                    'Z':             hyperparams['Z'].detach().clone().requires_grad_(True),
                    'lengthscales':  hyperparams['lengthscales'].detach().clone().requires_grad_(True),
                    'var_w':         hyperparams['var_w'].detach().clone().requires_grad_(True),
                    'X_test':        hyperparams['X_test'],  # 不优化的张量直接引用
                }
            else:
                steps_no_improve += log_interval

            # 早停判断
            if steps_no_improve >= early_stop_patience:
                print(f"Early stopping at step {step}, best ELBO={best_elbo:.4f}")
                break
            if val >= 1e7:
                print(f"Early stopping at step {step}, ELBO is too big")
                break

    # 如果从未更新过 best，返回最后一次
    if best_V is None:
        return V_params, u_params, hyperparams
    else:
        return best_V, best_u, best_hyp



def predict_vi(regions,
               V_params,
               hyperparams,
               M=3,
               X_test=None):
    """
    Predict test outputs using M samples from variational W posterior, with NaN handling.
    Returns:
      mu_pred: [T] tensor
      var_pred: [T] tensor
    """
    T = len(regions)
    if X_test is None:
        X_test = hyperparams['X_test']      # [T,D]
    mu_V = V_params['mu_V']             # [m2,Q,D]
    sigma_V = V_params['sigma_V']       # [m2,Q,D]
    # 1) q(W) posterior
    mu_W, cov_W = qW_from_qV(
        X_test,
        hyperparams['Z'],
        mu_V,
        sigma_V,
        hyperparams['lengthscales'],
        hyperparams['var_w']
    )  # mu_W [T,Q,D], cov_W [T,Q,D,D]
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))  # [T,Q,D]
    # 2) Sample W: [M,T,Q,D]
    eps = torch.randn((M, T, mu_W.shape[1], mu_W.shape[2]), device=device)
    W_samples = mu_W.unsqueeze(0) + eps * sigma_W.unsqueeze(0)

    norms = torch.norm(W_samples, p=2, dim=-1, keepdim=True)  # [M,T,Q,1]
    W_samples = W_samples / norms
    
    # 3) Allocate predictions
    mu_s = torch.zeros((M, T), device=device)
    var_s = torch.zeros((M, T), device=device)
    # 4) Loop over samples and regions
    for m in range(M):
        Wm = W_samples[m]  # [T,Q,D]
        for j, r in tqdm(enumerate(regions)):
            Xj = r['X']  # [n_j,D]
            yj = r['y']  # [n_j]
            Wj = Wm[j]   # [Q,D]
            # Project training and test
            Xn = Xj @ Wj.T               # [n_j,Q]
            yn = yj.view(-1,1)           # [n_j,1]
            # xt = (hyperparams['X_test'][j].view(1,-1) @ Wj.T)  # [1,Q]
            xt = (X_test[j].view(1,-1) @ Wj.T)  # [1,Q]
            # GP prediction
            # GP prediction
            mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
                Xn, yn, xt,
                mode="CEM", flag=False, device=device
            )
            mu_s[m, j] = mu_t.view(-1)
            var_s[m, j]= sig2_t.view(-1)
    # 5) Aggregate and handle NaNs
    mu_pred = mu_s.mean(dim=0)
    # var_pred = var_s.mean(dim=0)
    # mu_pred = torch.nan_to_num(mu_pred, nan=0.0)
    # var_pred = torch.nan_to_num(var_pred, nan=1e-6)
    # mu_pred = mu_s.mean(dim=0)

    # 这里用广播计算 (\mu_i - mu_pred)^2
    var_pred = var_s.mean(dim=0) + ((mu_s - mu_pred) ** 2).mean(dim=0)

    # 数值稳定性处理
    mu_pred = torch.nan_to_num(mu_pred, nan=0.0)
    var_pred = torch.nan_to_num(var_pred, nan=1e-6)

    return mu_pred, var_pred

import torch


import torch
import math
import numpy as np

# 预生成 Gauss–Hermite 节点和权重
_GH_POINTS = 20
_nodes_np, _weights_np = np.polynomial.hermite.hermgauss(_GH_POINTS)
_nodes   = torch.from_numpy(_nodes_np).to(device)  # shape [P]
_weights = torch.from_numpy(_weights_np).to(device)
_factor  = 1.0 / math.sqrt(math.pi)

def expected_sigmoid_gh(omega, mu_W, cov_W, X):
    """
    Compute E_q(W)[ sigmoid( ωᵀ [1, W x] ) ] via Gauss–Hermite.
    - omega:   [T, Q+1]
    - mu_W:    [T, Q, D]
    - cov_W:   [T, Q, D, D]  (diagonal covariance)
    - X:       [T, n, D]
    Returns:
    - pi:      [T, n]
    """
    T, n, D = X.shape
    Q = mu_W.shape[1]
    # 1) μ_proj[t,i,q] = μ_{W,t,q}ᵀ x_{t,i}
    mu_proj = torch.einsum('tqd,tnd->tnq', mu_W, X)  # [T,n,Q]
    # 2) μ_z = ω₀ + Σ_q ω_{q+1} μ_proj
    mu_z = omega[:, 0].unsqueeze(1) + (omega[:, 1:].unsqueeze(1) * mu_proj).sum(dim=2)  # [T,n]
    # 3) τ²[t,i,q] = ω_{q+1}² · (xᵀ Σ_{W,t,q} x)
    s = torch.einsum('tnd,tqde,tne->tnq', X, cov_W, X)  # [T,n,Q]
    tau2 = (omega[:, 1:].unsqueeze(1).pow(2) * s).sum(dim=2)  # [T,n]
    tau_z = torch.sqrt(tau2 + 1e-12)                          # [T,n]
    # 4) z values at GH nodes: [T,n,P]
    z = mu_z.unsqueeze(2) + math.sqrt(2.0) * tau_z.unsqueeze(2) * _nodes.view(1,1,-1)
    # 5) sigmoid and integrate
    sig = torch.sigmoid(z)                                    # [T,n,P]
    pi = _factor * ( _weights.view(1,1,-1) * sig ).sum(dim=-1)  # [T,n]
    return pi

def predict_vi_analytic(regions, V_params, u_params, hyperparams):
    """
    Analytic VI prediction (no MC) for each region's test point.
    Returns:
      mu_y:   [T] predictive mean of y
      var_y:  [T] predictive variance of y
    """
    device = hyperparams['Z'].device
    T = len(regions)

    # 1. 堆叠 region-specific 参数
    C           = torch.stack([r['C']            for r in regions], dim=0)  # [T,m1,Q]
    mu_u        = torch.stack([u['mu_u']        for u in u_params], dim=0)  # [T,m1]
    Sigma_u     = torch.stack([u['Sigma_u']     for u in u_params], dim=0)  # [T,m1,m1]
    sigma_noise = torch.stack([u['sigma_noise'] for u in u_params], dim=0)  # [T]
    # Uconst      = torch.tensor([r['U']           for r in regions], device=device)  # [T]

    # 2. Unpack global variational params
    Z            = hyperparams['Z']            # [m2,D]
    lengthscales = hyperparams['lengthscales'] # [Q]
    var_w        = hyperparams['var_w']        # [] or [Q]
    X_test       = hyperparams['X_test']       # [T,D]
    mu_V         = V_params['mu_V']            # [m2,Q,D]
    sigma_V      = V_params['sigma_V']         # [m2,Q,D]

    # 3. Compute q(W) posterior at X_test
    mu_W, cov_W = qW_from_qV(
        X_test, Z, mu_V, sigma_V, lengthscales, var_w
    )  # mu_W: [T,Q,D], cov_W: [T,Q,D,D]

    # 4. Compute m_W = E_q[k_fu] and S_W = E_q[k_uf k_fu]
    Xn = X_test.unsqueeze(1)  # [T,1,D]
    m_W = expected_Kfu(mu_W, cov_W, Xn, C, sigma_noise).squeeze(1)    # [T,m1]
    S_W = expected_KufKfu(mu_W, cov_W, Xn, C, sigma_noise).squeeze(1) # [T,m1,m1]

    # 5. Build K_uu and its inverse
    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)  # [T,m1,m1]
    I = torch.eye(m1, device=device).unsqueeze(0)         # [1,m1,m1]
    Kuu = torch.exp(-0.5 * d2) + sigma_noise.view(-1,1,1)*I  # [T,m1,m1]
    Luu = torch.linalg.cholesky(Kuu)                       # [T,m1,m1]
    Kuu_inv = torch.cholesky_inverse(Luu)                  # [T,m1,m1]

    # 6. Compute a = Kuu⁻¹ μ_u
    a = torch.einsum('tij,tj->ti', Kuu_inv, mu_u)          # [T,m1]

    # 7. Predictive f* mean and variance
    mu_f    = (m_W * a).sum(dim=1)                        # [T]
    V1      = sigma_noise**2 - torch.einsum('tij,tij->t', Kuu_inv, S_W)  # [T]
    T3      = torch.einsum('tij,tjk,tkm,tmi->t', Kuu_inv, S_W, Kuu_inv, Sigma_u)  # [T]
    V_W     = torch.einsum('ti,tij,tj->t', a, (S_W - m_W.unsqueeze(2)*m_W.unsqueeze(1)), a)  # [T]
    sigma_f2 = V1 + T3 + V_W                               # [T]

    # 8. Compute mixture weight π = E_q[sigmoid(ωᵀ[1,Wx*])]
    omega = torch.stack([u['omega'] for u in u_params], dim=0)  # [T, Q+1]
    pi    = expected_sigmoid_gh(omega, mu_W, cov_W, Xn).squeeze(1)  # [T]

    # 9. Predictive y* moments
    # mu_y = pi * mu_f + (1 - pi) * Uconst              # [T]
    U = torch.sigmoid(torch.stack([u['U_logit'] for u in u_params],0))  # [T]
    mu_y = pi * mu_f + (1 - pi) * U 
    E_f2 = mu_f.pow(2) + sigma_f2                     # [T]
    # E_y2 = pi * E_f2 + (1 - pi) * Uconst.pow(2)       # [T]
    E_y2 = pi * E_f2 + (1 - pi) * U**2
    var_y = E_y2 - mu_y.pow(2)                        # [T]

    return mu_y, var_y 

import torch
import torch.nn.functional as F

def compute_ELBO_minibatch(regions_batch, 
                          region_indices, 
                          V_params, 
                          u_params, 
                          hyperparams, 
                          total_regions,
                          ell=3, 
                          debug_threshold=1e7):
    """
    Minibatch ELBO computation.
    
    Args:
        regions_batch: 当前 minibatch 的 regions 列表 (长度 M)
        region_indices: 当前 batch 对应的 region 索引 (长度 M)
        V_params: 全局变分参数
        u_params: 全部 T 个 regions 的参数，但只使用 batch 中的
        hyperparams: 超参数
        total_regions: 总的 region 数量 T
        
    Returns:
        elbo: 当前 minibatch 的 ELBO，已经按比例缩放
    """
    M = len(regions_batch)  # minibatch size
    T = total_regions       # total number of regions
    device = regions_batch[0]['X'].device

    # --- 从 minibatch 中提取数据 ---
    X = torch.stack([r['X'] for r in regions_batch], dim=0)  # [M,n,D]
    y = torch.stack([r['y'] for r in regions_batch], dim=0)  # [M,n]
    C = torch.stack([r['C'] for r in regions_batch], dim=0)  # [M,m1,Q]

    # 从 u_params 中提取对应 minibatch 的参数
    batch_u_params = [u_params[i] for i in region_indices]
    
    U_logit = torch.stack([u['U_logit'] for u in batch_u_params], dim=0).view(-1,1)  # [M,1]
    U = torch.sigmoid(U_logit)                                                        # [M,1]
    sigma_noise = torch.stack([u['sigma_noise'] for u in batch_u_params], dim=0).view(-1)  # [M]
    sigma_k = torch.stack([u['sigma_k'] for u in batch_u_params], dim=0).view(-1)          # [M]
    mu_u = torch.stack([u['mu_u'] for u in batch_u_params], dim=0)                         # [M,m1]
    omega = torch.stack([u['omega'] for u in batch_u_params], dim=0)                       # [M,Q+1]

    # 处理 Sigma_u (确保正定性)
    eps = 1e-6
    raw = torch.stack([u['Sigma_u'] for u in batch_u_params], dim=0)  # [M, m1, m1]
    L = torch.tril(raw)  # [M, m1, m1]
    diag = torch.diagonal(L, dim1=-2, dim2=-1)                       # [M, m1]
    pos_diag = F.softplus(diag) + eps                                # [M, m1]
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag) 
    Sigma_u = L @ L.transpose(-2, -1)                                # [M, m1, m1]

    # hyperparams (全局)
    Z = hyperparams['Z']                    # [m2,D]
    m2 = Z.shape[0]
    lengthscales = hyperparams['lengthscales']  # [Q]
    var_w = hyperparams['var_w']                # scalar

    # V 相关变分参数 (全局)
    mu_V = V_params['mu_V']      # [m2,Q,D]
    sigma_V = V_params['sigma_V'] # [m2,Q,D]

    # === 1) KL(q(V)||p(V)) - 全局项，不依赖于 minibatch ===
    KL_V = kl_qp(Z, mu_V, sigma_V, lengthscales, var_w)

    # === 2) 为 minibatch 的测试点计算 q(W) posterior ===
    # 从 hyperparams 中提取对应的测试点
    X_test_batch = hyperparams['X_test'][region_indices]  # [M,D]
    
    mu_W, cov_W = qW_from_qV(X_test_batch, Z, mu_V, sigma_V, lengthscales, var_w)
    Kfu = expected_Kfu(mu_W, cov_W, X, C, sigma_k)       # [M,n,m1]
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k) # [M,m1,m1]

    # === 3) KL(q(u)||p(u)) - 仅对 minibatch ===
    KL_u_batch = kl_q_u_batch(C, mu_u, Sigma_u, torch.tensor(1.0, device=device)).sum()

    # === 4) 数据依赖项 - 仅对 minibatch ===
    # build prior Kuu
    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)  # [M,m1,m1]
    Kuu = torch.exp(-0.5 * d2)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)

    V1 = sigma_k.view(-1,1)**2 - torch.einsum('tij,tnji->tn', Kuu_inv, KufKfu)
    v = torch.einsum('tij,tj->ti', Kuu_inv, mu_u)   # [M,m1]
    E_fu = torch.einsum('tni,ti->tn', Kfu, v)       # [M,n]
    T3 = torch.einsum('tij,tnjk,tkm,tmi->tn',
                     Kuu_inv, KufKfu, Kuu_inv, Sigma_u)  # [M,n]
    quad = (y.pow(2) - 2*y*E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1,1)**2)

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T1 = -0.5 * _LOG_2PI - 0.5*torch.log(sigma_noise.view(-1,1)**2) + elog_sig - quad
    T2 = torch.log(U) + elog_one_minus

    region_elbo_batch = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum(-1)  # [M]
    data_term_batch = region_elbo_batch.sum()

    # === 5) 计算 minibatch ELBO ===
    # 数据项和 KL_u 按 minibatch 计算
    # KL_V 是全局的，需要按比例分配
    elbo_batch = data_term_batch - KL_u_batch - (KL_V * M / T)

    # --- debug: 如果 ELBO 过大，打印诊断信息 ---
    if elbo_batch.item() / M > debug_threshold:
        print(f"\n====== MINIBATCH ELBO DEBUG (batch_size={M}) ======")
        print(f"Batch ELBO = {elbo_batch.item():.3e}")
        print(f"Data term = {data_term_batch.item():.3e}")
        print(f"KL_u_batch = {KL_u_batch.item():.3e}")
        print(f"KL_V (scaled) = {(KL_V * M / T).item():.3e}")
        print("===============================================\n")

    return elbo_batch / M  # 返回平均的 ELBO


def create_minibatch_iterator(regions, u_params, batch_size, shuffle=True):
    """
    创建 minibatch 迭代器
    
    Args:
        regions: 所有 regions 的列表
        u_params: 所有 u_params 的列表
        batch_size: minibatch 大小
        shuffle: 是否随机打乱
        
    Yields:
        (regions_batch, region_indices): minibatch 的 regions 和对应的索引
    """
    T = len(regions)
    indices = torch.randperm(T) if shuffle else torch.arange(T)
    
    for i in range(0, T, batch_size):
        end_idx = min(i + batch_size, T)
        batch_indices = indices[i:end_idx]
        
        regions_batch = [regions[idx] for idx in batch_indices]
        yield regions_batch, batch_indices.tolist()


def train_vi_minibatch(regions,
                      V_params,
                      u_params,
                      hyperparams,
                      batch_size=None,
                      lr=1e-2,
                      num_steps=400,
                      log_interval=20,
                      lr_factor=0.5,
                      lr_patience=20,
                      early_stop_patience=40,
                      min_lr=1e-4,
                      elbo_tol=1e-3,
                      regions_data=None):
    """
    支持 minibatch 的变分训练
    
    Args:
        batch_size: minibatch 大小，如果为 None 则使用全部数据
    """
    T = len(regions)
    training_log = []
    
    # 如果 batch_size 为 None 或者 >= T，则使用全部数据
    if batch_size is None or batch_size >= T:
        print("Using full batch training")
        return train_vi(regions, V_params, u_params, hyperparams,
                       lr=lr, num_steps=num_steps, log_interval=log_interval,
                       lr_factor=lr_factor, lr_patience=lr_patience,
                       early_stop_patience=early_stop_patience, min_lr=min_lr,
                       elbo_tol=elbo_tol, regions_data=regions_data)
    
    print(f"Using minibatch training with batch_size={batch_size}")
    
    # 1. 准备所有可训练张量
    params = [V_params['mu_V'], V_params['sigma_V']]
    for u in u_params:
        params += [u['U_logit'], u['mu_u'], u['Sigma_u'],
                   u['sigma_noise'], u['omega'], u['sigma_k']]
    params += [hyperparams['lengthscales'],
               hyperparams['var_w'],
               hyperparams['Z']]

    optimizer = torch.optim.Adam(params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=lr_factor,
    #     patience=lr_patience, min_lr=min_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,         # 第一次余弦周期长度（epoch）
        T_mult=2,       # 每次重启后周期按此倍数增长
        eta_min=1e-5
    )

    best_elbo = -float('inf')
    steps_no_improve = 0
    best_V = best_u = best_hyp = None

    best_val_rmse = float('inf')
    best_val_V = best_val_u = best_val_hyp = None
    
    # 计算每个 epoch 的 batch 数量
    num_batches_per_epoch = (T + batch_size - 1) // batch_size
    
    from tqdm import tqdm
    for step in tqdm(range(1, num_steps + 1)):
        epoch_elbo = 0.0
        num_batches = 0
        
        # 一个 epoch 内处理所有 minibatches
        for regions_batch, batch_indices in create_minibatch_iterator(
            regions, u_params, batch_size, shuffle=True):
            
            optimizer.zero_grad()
            
            elbo_batch = compute_ELBO_minibatch(
                regions_batch, batch_indices, V_params, u_params, 
                hyperparams, T)
            
            
            (-elbo_batch).backward()
            optimizer.step()
            

            # clamp 保证正数
            with torch.no_grad():
                V_params['sigma_V'].clamp_(min=1e-1)
                hyperparams['var_w'].clamp_(min=1e-1)
                hyperparams['lengthscales'].clamp_(min=1e-6)
                for u in u_params:
                    u['sigma_noise'].clamp_(min=1e-1)
                    u['sigma_k'].clamp_(min=1e-1)
            
            epoch_elbo += elbo_batch.item()
            num_batches += 1
        
        # 计算 epoch 平均 ELBO
        avg_elbo = epoch_elbo / num_batches
        # scheduler.step(avg_elbo)
        scheduler.step()

        if step % 10 == 1:
            with torch.no_grad():
                if regions_data is not None:
                    regions_val, X_val, Y_val = regions_data
                    print(type(hyperparams))
                    mu_p, var_p = predict_vi(regions_val, V_params, hyperparams, X_test=X_val)
                    print(type(mu_p), type(var_p))  
                    rmse = torch.sqrt(torch.mean((mu_p - Y_val)**2))
                    # print(f"RMSE on validation set: {rmse:.4f}, ELBO={avg_elbo:.4f}")
                    # training_log.append((avg_elbo, rmse))

                    from properscoring import crps_gaussian
                    crps = crps_gaussian(Y_val.cpu().numpy(), mu_p.cpu().numpy(), np.sqrt(var_p.cpu().numpy()))
                    mean_crps = crps.mean()

                    print(f"RMSE on validation set: {rmse:.4f}, CRPS={mean_crps:.4f}, ELBO={avg_elbo:.4f}")
                    training_log.append((avg_elbo, rmse.item(), mean_crps))

                    if rmse.item() < best_val_rmse:
                        best_val_rmse = rmse.item()
                        best_val_V = {
                            'mu_V': V_params['mu_V'].detach().clone().requires_grad_(True),
                            'sigma_V': V_params['sigma_V'].detach().clone().requires_grad_(True),
                        }
                        best_val_u = []
                        for u in u_params:
                            best_val_u.append({
                                'U_logit': u['U_logit'].detach().clone().requires_grad_(True),
                                'mu_u': u['mu_u'].detach().clone().requires_grad_(True),
                                'Sigma_u': u['Sigma_u'].detach().clone().requires_grad_(True),
                                'sigma_noise': u['sigma_noise'].detach().clone().requires_grad_(True),
                                'sigma_k': u['sigma_k'].detach().clone().requires_grad_(True),
                                'omega': u['omega'].detach().clone().requires_grad_(True),
                            })
                        best_val_hyp = {
                            'Z': hyperparams['Z'].detach().clone().requires_grad_(True),
                            'lengthscales': hyperparams['lengthscales'].detach().clone().requires_grad_(True),
                            'var_w': hyperparams['var_w'].detach().clone().requires_grad_(True),
                            'X_test': hyperparams['X_test'],
                        }

        # 日志点才检查 best & early stop
        if step % log_interval == 0 or step == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[Step {step}/{num_steps}] Avg ELBO={avg_elbo:.4f}, LR={lr_now:.2e}")

            # 如果提升超过阈值，就保存快照
            if avg_elbo > best_elbo + elbo_tol:
                best_elbo = avg_elbo
                steps_no_improve = 0

                # 深拷贝当前参数
                best_V = {
                    'mu_V': V_params['mu_V'].detach().clone().requires_grad_(True),
                    'sigma_V': V_params['sigma_V'].detach().clone().requires_grad_(True),
                }
                best_u = []
                for u in u_params:
                    best_u.append({
                        'U_logit': u['U_logit'].detach().clone().requires_grad_(True),
                        'mu_u': u['mu_u'].detach().clone().requires_grad_(True),
                        'Sigma_u': u['Sigma_u'].detach().clone().requires_grad_(True),
                        'sigma_noise': u['sigma_noise'].detach().clone().requires_grad_(True),
                        'sigma_k': u['sigma_k'].detach().clone().requires_grad_(True),
                        'omega': u['omega'].detach().clone().requires_grad_(True),
                    })
                best_hyp = {
                    'Z': hyperparams['Z'].detach().clone().requires_grad_(True),
                    'lengthscales': hyperparams['lengthscales'].detach().clone().requires_grad_(True),
                    'var_w': hyperparams['var_w'].detach().clone().requires_grad_(True),
                    'X_test': hyperparams['X_test'],
                }
            else:
                steps_no_improve += log_interval

            # 早停判断
            if steps_no_improve >= early_stop_patience:
                print(f"Early stopping at step {step}, best ELBO={best_elbo:.4f}")
                break
            if avg_elbo >= 1e7:
                print(f"Early stopping at step {step}, ELBO is too big")
                break

    # 如果从未更新过 best，返回最后一次
    if best_V is None:
        return V_params, u_params, hyperparams, training_log, (best_val_V, best_val_u, best_val_hyp)
    else:
        return best_V, best_u, best_hyp, training_log, (best_val_V, best_val_u, best_val_hyp)


# ===== 示例 =====
if __name__=="__main__":
    # 构造示例数据
    T, n, m1, m2, Q, D = 50, 100, 3, 4, 2, 3
    regions, u_params = [], []
    for _ in range(T):
        regions.append({
            'X': torch.randn(n,D,device=device),
            'y': torch.randn(n,device=device),
            # 'U': 1.0,
            # 'U': torch.tensor(0.5,device=device,requires_grad=True)
            'C': torch.randn(m1,Q,device=device)
        })
        u_params.append({
            'U_logit': torch.zeros(1, device=device, requires_grad=True),
            'mu_u': torch.randn(m1,device=device,requires_grad=True),
            'Sigma_u':torch.eye(m1,device=device,requires_grad=True),
            'sigma_noise':torch.tensor(0.5,device=device,requires_grad=True),
            'sigma_k':torch.tensor(0.5,device=device,requires_grad=True),
            'omega':torch.randn(Q+1,device=device,requires_grad=True)
        })
    V_params = {
        'mu_V':torch.randn(m2,Q,D,device=device,requires_grad=True),
        'sigma_V':torch.rand(m2,Q,D,device=device,requires_grad=True)
    }
    hyperparams = {
        # 'Z': torch.randn(m2,D,device=device),
        'Z': torch.randn(m2,D,device=device,requires_grad=True),
        'X_test': torch.randn(T,D,device=device),
        'lengthscales': torch.rand(Q,device=device,requires_grad=True),
        'var_w': torch.tensor(1.0,device=device,requires_grad=True)
    }
    # V_params, u_params, hyperparams = train_vi(regions, V_params, u_params, hyperparams,
    #                                           lr=1e-3, num_steps=10, log_interval=5)
    
    V_params_trained, u_params_trained, hyperparams_trained = train_vi_minibatch(
        regions, V_params, u_params, hyperparams,
        batch_size=32,
        lr=1e-3,
        num_steps=10,
        log_interval=10
    )
    
    # mu_p, var_p = predict_vi(regions, V_params, hyperparams, M=10)
    mu_p, var_p = predict_vi_analytic(regions, V_params, u_params, hyperparams_trained)
    print("mu_pred=", mu_p)
    print("var_pred=", var_p)
