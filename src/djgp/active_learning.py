# active_djgp_acquisition.py
# -----------------------------------------------------------
# DJGP Active Learning acquisition:
#   A(x_i) = E_{W|x_i}[ bias^2 + var ] + lambda * Var_{W|x_i}[ mu_pred ]
#
# 复用你已有的:
# - qW_from_qV  (从 q(V) 推出 q(W|x))
# - JumpGP_LD, calculate_bias_and_variance（在投影空间上直接用 JGP 的实现）
# -----------------------------------------------------------

import os
import sys
import math
import numpy as np
import torch
from tqdm import tqdm

# ==== 依据你的项目结构，把需要的模块加入路径（按需修改）====
# 例如：当前文件与这些模块在同一层或相邻层级
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)
sys.path.append(os.path.dirname(this_dir))

# ---- 你已有的函数/模块（确保可导入） ----
from djgp.variational import qW_from_qV  # q(W|x) 后验
from jumpgp.linear import JumpGP_LD
from djgp.acquisition_metrics import calculate_bias_and_variance

# -----------------------------------------------------------
# 采样 W|x_i
# -----------------------------------------------------------
def sample_W_for_point(i, V_params, hyperparams, M=8, normalize_W=True):
    """
    从 q(W|x_i) 采样 M 个 W。
    复用你已有的 qW_from_qV，将协方差当作对角项来采样。

    Args:
      i:               第 i 个候选/测试点索引
      V_params:        dict 里至少包含:
                        - 'mu_V':    [m2,Q,D] (torch)
                        - 'sigma_V': [m2,Q,D] (torch)
      hyperparams:     dict 里至少包含:
                        - 'Z':            [m2,D] (torch)
                        - 'X_test':       [T,D] (torch)
                        - 'lengthscales': [Q]   (torch)
                        - 'var_w':        [] or [Q] (torch)
      M:               采样个数
      normalize_W:     是否做行向量 L2 归一化（和你 predict_vi 一致）

    Returns:
      W_samples: [M,Q,D] (torch)
    """
    device = hyperparams['Z'].device
    x_i = hyperparams['X_test'][i].view(1, -1)  # [1,D]

    # q(W|x_i)
    mu_W_i, cov_W_i = qW_from_qV(
        X=x_i,
        Z=hyperparams['Z'],
        mu_V=V_params['mu_V'],
        sigma_V=V_params['sigma_V'],
        lengthscales=hyperparams['lengthscales'],
        var_w=hyperparams['var_w']
    )  # mu_W_i: [1,Q,D], cov_W_i: [1,Q,D,D] (对角为主)

    mu = mu_W_i[0]                                        # [Q,D]
    std = torch.sqrt(cov_W_i[0].diagonal(dim1=1, dim2=2)) # [Q,D]

    eps = torch.randn((M, *mu.shape), device=device)      # [M,Q,D]
    W_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # [M,Q,D]

    if normalize_W:
        norms = torch.norm(W_samples, p=2, dim=-1, keepdim=True)  # [M,Q,1]
        W_samples = W_samples / (norms + 1e-12)

    return W_samples


# -----------------------------------------------------------
# 单点 acquisition: 直接用 JumpGP_LD + calculate_bias_and_variance
# -----------------------------------------------------------
def acquisition_one_x_with_JGP(
    i,
    regions,
    V_params,
    hyperparams,
    jgp_args=None,
    M=8,
    lambda_=0.3
):
    """
    对第 i 个候选点计算:
      A(x_i) = E_W[ bias^2 + var ] + lambda * Var_W[ mu_pred ]

    - 邻域来自 regions[i]['X'], regions[i]['y'] (torch 张量，原空间)
    - 在每个 W 样本下，把邻域与 x_i 投影到 Z，再
      直接调用 JumpGP_LD / calculate_bias_and_variance（与 ActiveJGP 一致）

    Args:
      i:            候选索引
      regions:      list(dict) with keys { 'X': [n,D] torch, 'y': [n] or [n,1] torch }
      V_params:     {'mu_V': [m2,Q,D], 'sigma_V':[m2,Q,D]} (torch)
      hyperparams:  {'Z':[m2,D], 'X_test':[T,D], 'lengthscales':[Q], 'var_w':scalar or [Q]}
      jgp_args:     传给 JumpGP_LD / calculate_bias_and_variance 的附加参数（如 logtheta）
      M:            W 样本数
      lambda_:      A2 的权重

    Returns:
      A:    float，acquisition 值
      part: dict，包含 A1/A2 的分量
    """
    # 邻域（原空间，torch）
    X_nb = regions[i]['X']            # [n,D]
    y_nb = regions[i]['y']            # [n] or [n,1]
    if y_nb.ndim == 1:
        y_nb = y_nb.view(-1, 1)
    x_i  = hyperparams['X_test'][i]   # [D]

    # 采样 W|x_i
    W_samps = sample_W_for_point(i, V_params, hyperparams, M=M, normalize_W=True)

    mspe_list = []
    mu_list   = []

    # 对每个 W 样本，投影后直接套用你的 JGP 实现
    for m in range(M):
        Wm = W_samps[m]                       # [Q,D] (torch)

        Z_nb = (X_nb @ Wm.t()).cpu().numpy()  # [n,Q] -> numpy
        z_i  = (x_i  @ Wm.t()).cpu().numpy().reshape(1, -1)  # [1,Q]
        y_nb_np = y_nb.cpu().numpy()          # [n,1]

        if jgp_args is not None and len(jgp_args) > 0 and jgp_args[0] is not None:
            pred_i, _, tmp_model, _ = JumpGP_LD(Z_nb, y_nb_np, z_i, 'CEM', 0, jgp_args[0])
            b2, v2, _, _             = calculate_bias_and_variance(tmp_model, z_i, jgp_args[0])
        else:
            pred_i, _, tmp_model, _ = JumpGP_LD(Z_nb, y_nb_np, z_i, 'CEM', 0)
            b2, v2, _, _             = calculate_bias_and_variance(tmp_model, z_i)

        # 你的 ActiveJGP 定义：MSPE = bias^2 + var
        b2 = np.asarray(b2).ravel()[0]
        v2 = np.asarray(v2).ravel()[0]
        pred_i = np.asarray(pred_i).ravel()[0]
        mspe_list.append(float(b2 + v2))
        mu_list.append(float(pred_i))

    mspe_arr = np.asarray(mspe_list, dtype=float)  # [M]
    mu_arr   = np.asarray(mu_list,   dtype=float)  # [M]

    A1 = float(mspe_arr.mean())
    A2 = float(mu_arr.var(ddof=1)) if M > 1 else 0.0
    A  = A1 + lambda_ * A2
    return A, {'A1': A1, 'A2': A2}


# -----------------------------------------------------------
# 批量计算所有候选的 acquisition
# -----------------------------------------------------------
@torch.no_grad()
def compute_acquisition_all_with_JGP(
    regions,
    V_params,
    hyperparams,
    jgp_args=None,
    M=8,
    lambda_=0.3
):
    """
    对 X_test（=候选池）中所有点依次计算 acquisition。

    Returns:
      A_all:  (T,) numpy array
      parts:  list of dict per i (包含 A1/A2)
    """
    T = hyperparams['X_test'].shape[0]
    A = np.zeros(T, dtype=float)
    parts = []
    for i in tqdm(range(T)):
        a_i, comp = acquisition_one_x_with_JGP(
            i, regions, V_params, hyperparams,
            jgp_args=jgp_args, M=M, lambda_=lambda_
        )
        A[i] = a_i
        parts.append(comp)
    return A, parts


# -----------------------------------------------------------
# Top-K 选择（辅助）
# -----------------------------------------------------------
def select_topk_candidates(A, K=1):
    """
    从 acquisition 数组 A 里选出前 K 个索引（不排序稳定性要求时更快）。
    """
    K = int(K)
    if K <= 0:
        return np.array([], dtype=int)
    if K >= A.shape[0]:
        return np.arange(A.shape[0], dtype=int)
    idx = np.argpartition(-A, K-1)[:K]
    # 可选：按分数再排个序
    idx = idx[np.argsort(-A[idx])]
    return idx

# djgp_pipeline.py
# 封装 DJGP 训练 + 预测 (+ 可选评估)
# 输入：X_train, Y_train, X_test, （可选）Y_test
# 输出：mu_pred, var_pred，以及（若提供 Y_test）rmse, crps
# 同时返回后续 AL 需要的 regions, V_params, hyperparams, jgp_args

import time
from dataclasses import dataclass, asdict
import torch
import numpy as np

# 你的项目依赖（路径保持与原脚本一致）
from shared.utils1 import jumpgp_ld_wrapper  # 仅被 predict_vi 等间接使用
from shared.jumpgp_runner import *
from djgp.minibatch import train_vi_minibatch
from djgp.variational import *

# ---------------------------
# 参数容器（也可用 dict 传入）
# ---------------------------
@dataclass
class DJGPArgs:
    Q: int = 2                  # latent 维度
    m1: int = 5                 # 每个 region 的诱导点
    m2: int = 20                # 全局诱导点（投影 GP）
    n: int = 100                # 每个 region 的邻域大小
    num_steps: int = 200        # VI 训练步数
    MC_num: int = 5             # 预测时的 W 采样次数
    lr: float = 1e-2            # 学习率
    batch_size: int = 64        # batch 训练的 batch size
    use_batch: bool = False     # 是否启用 minibatch 训练

# ---------------------------
# 内部：构建 regions & 初始化参数
# ---------------------------
def _build_regions_and_params(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test:  torch.Tensor,
    args: DJGPArgs,
    device: torch.device
):
    """
    返回：regions, V_params, u_params, hyperparams
    """
    T, D = X_test.shape
    Q, m1, m2, n = args.Q, args.m1, args.m2, args.n

    # 1) 邻域（在 CPU 上构建、随后拷到 device）
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
            'C': torch.randn(m1, Q, device=device)          # region 内诱导点初始
        })

    # 2) 投影 GP 的变分参数（V_params）
    V_params = {
        'mu_V':    torch.randn(m2, Q, D, device=device, requires_grad=True),
        'sigma_V': torch.rand( m2, Q, D, device=device, requires_grad=True),
    }

    # 3) 每个 region 的 u_params
    u_params = []
    for _ in range(T):
        u_params.append({
            'U_logit':     torch.zeros(1, device=device, requires_grad=True),
            'mu_u':        torch.randn(m1, device=device, requires_grad=True),
            'Sigma_u':     torch.eye(m1, device=device, requires_grad=True),
            'sigma_noise': torch.tensor(0.5, device=device, requires_grad=True),
            'sigma_k':     torch.tensor(0.5, device=device, requires_grad=True),
            'omega':       torch.randn(Q+1, device=device, requires_grad=True),
        })

    # 4) 超参数（hyperparams）
    X_train_mean = X_train.mean(dim=0)
    X_train_std  = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * (X_train_std + 1e-12)

    hyperparams = {
        'Z':            Z,                                        # (m2, D)
        'X_test':       X_test,                                   # (T, D)
        'lengthscales': torch.rand(args.Q, device=device, requires_grad=True),
        'var_w':        torch.tensor(1.0, device=device, requires_grad=True),
    }

    return regions, V_params, u_params, hyperparams

# ---------------------------
# 对外：一键训练 + 预测 + （可选）指标
# ---------------------------
def djgp_fit_predict(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test:  torch.Tensor,
    Y_test:  torch.Tensor | None = None,
    args: "DJGPArgs | dict | None" = None,
    device: torch.device | None = None,
    need_predict: bool=False
):
    """
    封装 DJGP 训练/预测流程。

    Parameters
    ----------
    X_train : (N_train, D) torch.float32/float64
    Y_train : (N_train,) 或 (N_train,1)
    X_test  : (T, D)
    Y_test  : (T,) 或 (T,1)；可选，若提供则计算 RMSE/CRPS
    args    : DJGPArgs 或 dict（字段同 DJGPArgs）
    device  : torch.device，可选（默认自动用 cuda / cpu）

    Returns
    -------
    result : dict
        {
          'mu_pred':     (T,) torch.Tensor,
          'var_pred':    (T,) torch.Tensor,
          'rmse':        float or None,     # 只有提供 Y_test 时才非 None
          'crps':        float or None,     # 同上
          'regions':     list[dict],
          'V_params':    dict,
          'hyperparams': dict,
          'jgp_args':    list,              # 预留给 JGP 的额外参数（比如 logtheta），默认空
          'elapsed_sec': float,
          'args':        dict-like,
          'device':      str
        }
    """
    # 0) 设备/参数
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args is None:
        args = DJGPArgs()
    elif isinstance(args, dict):
        args = DJGPArgs(**args)

    # 1) 数据到 device，整理形状
    X_train = X_train.to(device)
    X_test  = X_test.to(device)
    Y_train = Y_train.to(device).view(-1)
    if Y_test is not None:
        Y_test = Y_test.to(device).view(-1)

    # 2) 初始化
    regions, V_params, u_params, hyperparams = _build_regions_and_params(
        X_train, Y_train, X_test, args, device
    )

    # 3) 训练
    start_time = time.time()
    if not args.use_batch:
        V_params, u_params, hyperparams = train_vi(
            regions=regions,
            V_params=V_params,
            u_params=u_params,
            hyperparams=hyperparams,
            lr=args.lr,
            num_steps=args.num_steps,
            log_interval=50
        )
    else:
        # 如果你实现了 mini-batch 版本，请按你的返回值组织
        hyperparams_trained = train_vi_minibatch(
            regions, V_params, u_params, hyperparams,
            batch_size=args.batch_size,
            lr=args.lr,
            num_steps=args.num_steps,
            log_interval=50
        )
        hyperparams = hyperparams_trained

    # 4) 预测（MC over W）
    if need_predict:
        mu_pred, var_pred = predict_vi(
            regions, V_params, hyperparams, M=args.MC_num
        )

        # 5) （可选）指标
        rmse, mean_crps = None, None
        if Y_test is not None:
            sigmas = torch.sqrt(var_pred.clamp_min(1e-12))
            rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)
            print(f"[DJGP] done. RMSE={rmse:.4f}  CRPS={mean_crps:.4f}  time={time.time()-start_time:.1f}s")
        else:
            print(f"[DJGP] done. time={time.time()-start_time:.1f}s")
    else:
        mu_pred, var_pred, rmse, mean_crps = None, None, None, None

    # 6) 后续 AL 需要的 jgp_args（按你 ActiveJGP 风格，默认空列表）
    jgp_args = []  # 如果你有 logtheta 等 JGP 额外参数，可在此填入

    # 7) 返回
    return {
        'mu_pred':     mu_pred.detach() if mu_pred is not None else None,
        'var_pred':    var_pred.detach() if var_pred is not None else None,
        'rmse':        float(rmse) if rmse is not None else None,
        'crps':        float(mean_crps) if mean_crps is not None else None,
        'regions':     regions,       # 后续 AL 需要
        'V_params':    V_params,      # 后续 AL 需要
        'hyperparams': hyperparams,   # 后续 AL 需要
        'jgp_args':    jgp_args,      # 按需外传给 ActiveJGP 风格接口
        'elapsed_sec': time.time() - start_time,
        'args':        asdict(args),
        'device':      str(device)
    }

# -----------------------------------------------------------
# 可选：命令行/示例（需要你已有的 regions/V_params/hyperparams）
# -----------------------------------------------------------
if __name__ == "__main__":
    # ========= 合成数据 =========
    import torch
    torch.manual_seed(42)
    np.random.seed(42)

    N_train = 50
    N_cand  = 100
    D       = 4

    # 训练集 + 候选点（test/candidate）
    X_train = torch.rand(N_train, D)
    X_cand  = torch.rand(N_cand,  D)

    # 合成目标: 平滑项 + sigmoid 跳变 + 轻噪声
    # 既有全局平滑，也有沿维度3的“决策面式”sigmoid 变化
    def synth_y(X):
        # X: [N,4] in [0,1]
        t = (
            torch.sin(2 * math.pi * X[:, 0]) +
            0.5 * torch.cos(2 * math.pi * X[:, 1]) +
            0.3 * torch.sin(2 * math.pi * X[:, 3])
        )
        # 决策面：sigmoid( a*(x2 - 0.5) + b*(x0 - x1) )
        logits = 8.0 * (X[:, 2] - 0.5) + 5.0 * (X[:, 0] - X[:, 1])
        jump   = torch.sigmoid(logits) * 1.5    # 连续但陡峭
        y      = t + jump
        return y

    y_train = synth_y(X_train) + 0.05 * torch.randn(N_train)  # 加轻噪声

    # ========= 训练 + 在候选上预测 =========
    # 注意：这里把 X_cand 当成 X_test 传入，以便后续做 acquisition
    # DJGPArgs 参数可以按需调整
    args = DJGPArgs(
        Q=2, m1=6, m2=20, n=50,
        num_steps=80, MC_num=6, lr=1e-2,
        use_batch=False
    )

    print("\n[Stage] Training DJGP and predicting on candidates ...")
    res = djgp_fit_predict(
        X_train=X_train, Y_train=y_train,
        X_test=X_cand,   Y_test=None,   # 此处不评估 RMSE/CRPS
        args=args
    )

    mu_pred     = res['mu_pred']
    var_pred    = res['var_pred']
    regions     = res['regions']
    V_params    = res['V_params']
    hyperparams = res['hyperparams']
    jgp_args    = res['jgp_args']

    # ========= 计算 acquisition =========
    print("\n[Stage] Computing acquisition for all candidates ...")
    A, parts = compute_acquisition_all_with_JGP(
        regions=regions,
        V_params=V_params,
        hyperparams=hyperparams,
        jgp_args=jgp_args,
        M=args.MC_num,
        lambda_=0.3
    )

    # 基本检查
    if not np.all(np.isfinite(A)):
        bad = np.where(~np.isfinite(A))[0]
        print(f"[WARN] Non-finite acquisition at indices: {bad.tolist()}")
    print(f"[Acq] A mean={A.mean():.4f} std={A.std():.4f} min={A.min():.4f} max={A.max():.4f}")

    # 选 Top-K
    K = 5
    top_idx = select_topk_candidates(A, K=K)
    print(f"\nTop-{K} candidate indices (best -> worst): {top_idx.tolist()}")
    print("Top-K scores:", A[top_idx])

    # 展示对应的候选点坐标（便于核查）
    Xc_np = X_cand.numpy()
    for rank, i in enumerate(top_idx):
        print(f"  #{rank+1}  idx={int(i)}  x={Xc_np[int(i)]}  A={A[int(i)]:.4f}")

    # 可选：也看一下预测分布的 sanity
    vp = var_pred.detach().cpu().numpy()
    mp = mu_pred.detach().cpu().numpy()
    print(f"\n[Pred] mu_pred mean={mp.mean():.4f}, std={mp.std():.4f}")
    print(f"[Pred] var_pred mean={vp.mean():.4f}, std={vp.std():.4f}, min={vp.min():.4e}, max={vp.max():.4e}")

    print("\n[Done] Smoke test finished.")

