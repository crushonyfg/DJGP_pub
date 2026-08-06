import torch
import math

def se_kernel(x1, x2, theta1=9.0, theta2=200.0):
    """
    平方指数核：
      K[i,j] = θ1 * exp(-||x1[i] - x2[j]||^2 / θ2)
    """
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)      # [n1, n2, d]
    dist2 = diff.pow(2).sum(-1)                   # [n1, n2]
    return theta1 * torch.exp(-dist2 / theta2)

def compute_region(x, r):
    """
    根据 d+1 个 g_j 分区函数，为每个点分配 region 索引
    输入:
      x: [m, d]
      r: 长度 d 的 ±1 Tensor
    返回:
      region: [m] 整数索引
    """
    m, d = x.shape
    # g0
    g0 = x.pow(2).sum(1) - 0.4**2                 # [m]
    signs = [(g0 >= 0).long()]

    # g1..gd
    for j in range(d):
        gj = x.pow(2).sum(1) \
             - x[:, j].pow(2) \
             + (x[:, j] + r[j]*0.5).pow(2) \
             - 0.3**2
        signs.append((gj >= 0).long())

    region = torch.zeros(m, dtype=torch.long, device=x.device)
    for j, s in enumerate(signs):
        region += (s << j)
    return region

def safe_cholesky(K, initial_jitter=1e-6, max_tries=5):
    """
    带自适应抖动的 Cholesky 分解
    """
    jitter = initial_jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=K.device))
        except RuntimeError:
            jitter *= 10
    # 最后一次尝试，不再捕获异常
    return torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=K.device))

def generate_data(d, N, Nt,
                  theta1=9.0, theta2=200.0,
                  noise_var=4.0,
                  test_tol=0.05,
                  device='cpu',
                  mu_scale=13.5,
                  mu_mode='escalating',
                  return_regions=False,
                  region_mode='hyperplane',
                  num_regions=None,
                  min_gap=6.0):
    """
    生成训练/测试数据：
      d  -- 输入维度
      N  -- 训练集大小
      Nt -- 测试集大小
      region_mode -- 'hyperplane' (default): 2^(d+1) sign-combination regions (original);
                     'voronoi': ~num_regions convex regions (nearest of num_regions random
                     centres). A handful of contiguous regions, closer to real data, while
                     keeping the large escalating jump. Default keeps original behaviour.
      num_regions -- number of Voronoi centres (region_mode='voronoi'); default d+1.
    返回:
      x_train [N,d], y_train [N],
      x_test  [Nt,d], y_test  [Nt]
    """
    device = torch.device(device)
    # torch.manual_seed(0)

    # 随机切分参数 r_j ∈ {+1,-1}
    r = torch.randint(0, 2, (d,), device=device).float() * 2 - 1  # ±1

    # Voronoi centres (shared train/test) when region_mode='voronoi'.
    centers = None
    if str(region_mode) == 'voronoi':
        n_reg = int(num_regions) if num_regions else (d + 1)
        centers = (torch.rand(n_reg, d, device=device) - 0.5)

    def _region(x):
        if centers is not None:
            return torch.cdist(x, centers).argmin(dim=1)
        return compute_region(x, r)

    # 1) 生成训练点
    x_train = (torch.rand(N, d, device=device) - 0.5)
    region_train = _region(x_train)                # [N]
    num_regions = int(centers.shape[0]) if centers is not None else 2 ** (d + 1)

    # 2) 构造测试点（在赋均值之前采样，使 'random_mingap' 能只对实际出现的 region 赋可分均值）
    #    hyperplane: 至少一个 g_j 的 |g_j(x)| <= test_tol；voronoi: 到最近两中心距离差 <= test_tol
    x_test_parts = []
    n_have = 0
    while n_have < Nt:
        cand = (torch.rand(Nt, d, device=device) - 0.5)
        if centers is not None:
            two = torch.cdist(cand, centers).topk(2, dim=1, largest=False).values  # [Nt,2]
            mask = (two[:, 1] - two[:, 0]) <= test_tol
        else:
            # JGP-style NORMALIZED boundary score d(z) = min_j |f_j| / ||grad f_j||_2
            # (approximate Euclidean distance to a boundary). Gradients are analytic:
            #   f_0 = ||z||^2 - 0.4^2                 -> grad = 2 z
            #   f_j = ||z||^2 - z_j^2 + (z_j+r_j.5)^2 - 0.3^2
            #        -> grad = 2 z, with the j-th entry replaced by 2 (z_j + r_j*0.5)
            zz = cand.pow(2).sum(1)
            scores = [(zz - 0.4**2).abs() / (2.0 * cand.norm(dim=1)).clamp_min(1e-12)]
            for j in range(d):
                fj = zz - cand[:, j].pow(2) + (cand[:, j] + r[j]*0.5).pow(2) - 0.3**2
                g = 2.0 * cand.clone()
                g[:, j] = 2.0 * (cand[:, j] + r[j]*0.5)
                scores.append(fj.abs() / g.norm(dim=1).clamp_min(1e-12))
            mask = (torch.stack(scores, dim=1) <= test_tol).any(dim=1)
        selected = cand[mask]
        x_test_parts.append(selected)
        n_have += int(selected.shape[0])
    x_test = torch.cat(x_test_parts, dim=0)[:Nt]
    region_test = _region(x_test)

    # 3) 为每个 region 选均值 μ_k
    signs = (torch.rand(num_regions, device=device) < 0.5).float() * 2 - 1
    if str(mu_mode) == 'random_mingap':
        # 仅对实际出现的 region 赋等间隔(间隔=min_gap)、再 shuffle 的均值：任意两区可分
        # (gap>=min_gap)，range 适中(=(#used-1)*min_gap)，不像 escalating 爆到几百。
        used = torch.unique(torch.cat([region_train, region_test]))
        nu = int(used.numel())
        grid = (torch.arange(nu, device=device, dtype=torch.float32) - (nu - 1) / 2.0) * float(min_gap)
        perm = torch.randperm(nu, device=device)
        mu = torch.zeros(num_regions, device=device)
        mu[used] = grid[perm]
    elif str(mu_mode) == 'random':
        # region means iid N(0, mu_scale^2): regions OVERLAP in y, jump is the hard part.
        mu = float(mu_scale) * torch.randn(num_regions, device=device)
    else:
        # original 'escalating' offsets (hugely separated, easy classify+lookup)
        mu = signs * (float(mu_scale) * torch.arange(num_regions, device=device).float())

    # 4) 在各 region 内生成 y_train
    y_train = torch.zeros(N, device=device)
    for k in region_train.unique():
        idx = (region_train == k).nonzero(as_tuple=False).squeeze(1)
        Xk = x_train[idx]                         # [n_k, d]
        K = se_kernel(Xk, Xk, theta1, theta2)
        K += noise_var * torch.eye(len(idx), device=device)  # 噪声方差 + 抖动
        L = safe_cholesky(K)
        eps = torch.randn(len(idx), device=device)
        f = L @ eps
        y_train[idx] = mu[k] + f

    # 5) 在测试点做无噪声 GP 采样
    y_test = torch.zeros(Nt, device=device)
    for k in region_test.unique():
        idx = (region_test == k).nonzero(as_tuple=False).squeeze(1)
        Xk = x_test[idx]
        K = se_kernel(Xk, Xk, theta1, theta2)
        L = safe_cholesky(K)
        eps = torch.randn(len(idx), device=device)
        f = L @ eps
        y_test[idx] = mu[k] + f

    if return_regions:
        # region_train/region_test are the true LH region indices (same-region
        # neighbours are the true inliers for the local jump GP / gate).
        return x_train, y_train, x_test, y_test, region_train, region_test
    return x_train, y_train, x_test, y_test

def generate_robust_data(d, N, Nt,
                         theta1=9.0, theta2=0.3,
                         noise_var=1.0,
                         outlier_frac=0.15,
                         outlier_scale=10.0,
                         device='cpu',
                         return_regions=False):
    """Robust-regression data: ONE smooth GP function f over the whole input (no regions/
    jumps), inlier observations y=f+N(0,noise_var); a fraction `outlier_frac` of TRAIN points
    are corrupted with heavy symmetric noise N(0,(outlier_scale*sqrt(noise_var))^2). TEST is
    clean (y=f+inlier noise). This is the natural home for DJGP's inlier-GP + outlier-gate
    (a local robust GP): the local likelihood routes high-residual outliers out so the inlier
    GP is not dragged, while global GPs (DKL/DGP) get pulled by the outliers.

    region_train/region_test return the OUTLIER MASK (1=outlier, 0=inlier; test all 0) so the
    same diagnostics (base = inlier-neighbour fraction) apply.
    """
    device = torch.device(device)
    M = N + Nt
    X = (torch.rand(M, d, device=device) - 0.5)
    K = se_kernel(X, X, theta1, theta2)
    L = safe_cholesky(K)
    f = L @ torch.randn(M, device=device)            # single smooth function over all points
    f_tr, f_te = f[:N], f[N:]
    sig = float(noise_var) ** 0.5
    y_train = f_tr + sig * torch.randn(N, device=device)
    is_out = torch.zeros(N, dtype=torch.long, device=device)
    n_out = int(round(float(outlier_frac) * N))
    if n_out > 0:
        out_idx = torch.randperm(N, device=device)[:n_out]
        y_train[out_idx] = f_tr[out_idx] + (float(outlier_scale) * sig) * torch.randn(n_out, device=device)
        is_out[out_idx] = 1
    y_test = f_te + sig * torch.randn(Nt, device=device)   # clean test (inlier noise only)
    x_train, x_test = X[:N], X[N:]
    if return_regions:
        region_test = torch.zeros(Nt, dtype=torch.long, device=device)
        return x_train, y_train, x_test, y_test, is_out, region_test
    return x_train, y_train, x_test, y_test


# === 测试示例 ===
if __name__ == '__main__':
    d, N, Nt = 3, 500, 200
    xtr, ytr, xte, yte = generate_data(d, N, Nt)
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
