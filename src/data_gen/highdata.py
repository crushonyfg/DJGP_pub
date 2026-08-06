import torch
import argparse
from data_gen.highdata_utils import generate_data
from data_gen.synthetic import save_dataset  

def sample_rff_params(d, D_rff, lengthscale, device):
    """
    采样一组 RFF 参数 Omega 和 phases
    Omega: [D_rff, d], phases: [D_rff]
    """
    Omega  = torch.randn(D_rff, d, device=device) / lengthscale
    phases = 2 * torch.pi * torch.rand(D_rff, device=device)
    return Omega, phases

def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cudnn 设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")

def make_rff_features(x, Omega, phases, kernel_var):
    """
    x: [N, d]
    Omega: [D_rff, d]
    phases: [D_rff]
    返回 Phi: [N, D_rff]
    """
    # x @ Omega.T -> [N, D_rff]
    proj  = x @ Omega.t() + phases.unsqueeze(0)
    scale = torch.sqrt(torch.tensor(2.0 * kernel_var / Omega.shape[0], device=x.device))
    return scale * torch.cos(proj)

def autoencoder_transform(X, X_test, hidden_dim):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = 64
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----- 2. 定义自编码器 -----
    # class Autoencoder(nn.Module):
    #     def __init__(self, input_dim=5, hidden_dim=30):
    #         super().__init__()
    #         # 编码器：5 -> 30
    #         self.encoder = nn.Sequential(
    #             nn.Linear(input_dim, hidden_dim),
    #             nn.ReLU(inplace=True),
    #             # nn.Linear(64, hidden_dim),
    #             # nn.ReLU(inplace=True),
    #         )
    #         # 解码器：30 -> 5
    #         self.decoder = nn.Sequential(
    #             nn.Linear(hidden_dim, input_dim),
    #             # nn.ReLU(inplace=True),
    #             # nn.Linear(64, input_dim),
    #             # 如果数据无负值限制，最后可以不用激活；若需 [0,1] 范围可加 Sigmoid
    #         )

    #     def forward(self, x):
    #         z = self.encoder(x)
    #         x_recon = self.decoder(z)
    #         return x_recon
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, input_dim=5, hidden_dim=30):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(64, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(64, input_dim),
                # 最后一层一般不用激活，或根据数据需求加 Sigmoid / Tanh
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    # ----- 3. 实例化模型、损失和优化器 -----
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = X.device
    input_dim = X.shape[1]
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # ----- 4. 训练 -----
    n_epochs = 100
    for epoch in range(1, n_epochs+1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs}  Loss: {epoch_loss:.6f}")

    # ----- 5. 用 encoder 提取高维特征 -----
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        Z = model.encoder(X)   # Z.shape == (n_samples, 30)
        X_test = X_test.to(device)
        Z_test = model.encoder(X_test)   # Z.shape == (n_samples, 30)


    # 现在 Z 就是维度为 30 的合成表示
    print("Encoded feature shape:", Z.shape)
    return Z, Z_test, model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--d',        type=int,   default=3,   help='原始输入维度')
    p.add_argument('--N',        type=int,   default=500, help='训练集大小')
    p.add_argument('--Nt',       type=int,   default=200, help='测试集大小')
    p.add_argument('--H',        type=int,   default=20,  help='RFF 映射到的高维度')
    p.add_argument('--lengthscale', type=float, default=0.5, help='RFF lengthscale')
    p.add_argument('--kernel_var',  type=float, default=9, help='RFF kernel variance')
    p.add_argument('--device',   type=str,   default='cpu')
    p.add_argument('--ins_dim', type=bool, default=True, help='是否将输入维度扩展为高维')
    p.add_argument('--methods', type=str, default="rff", help='which method to use to generate the high-dimensional data')
    p.add_argument('--noise_var', type=float, default=4, help='noise variance')
    p.add_argument('--noise_std', type=float, default=2, help='noise std for phantom')
    p.add_argument('--caseno', type=int, default=0, help='case number')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    return p.parse_args()

def polynomial_transform(X5):
    from sklearn.preprocessing import PolynomialFeatures

    X5_np = X5.detach().cpu().numpy()

    # 2) 用 sklearn 生成多项式特征
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly_np = poly.fit_transform(X5_np)            # -> 
    X_poly = torch.from_numpy(X_poly_np).to(X5.device)
    return X_poly

def generate_data_phantom(N, Nt, device, caseno=4, noise_std=2):
    from data_gen.synthetic import generate_Y_from_image
    import numpy as np
    z = np.random.uniform(-0.5, 0.5, (N+Nt, 2))
    z = torch.from_numpy(z).float().to(device)
    y = generate_Y_from_image(z, caseno, noise_std=noise_std)
    x_train = z[:N]
    x_test = z[N:]
    y_train = y[:N]
    y_test = y[N:]
    # x_train = torch.from_numpy(x_train).to(device)
    # x_test = torch.from_numpy(x_test).to(device)
    # y_train = torch.from_numpy(y_train).to(device)
    # y_test = torch.from_numpy(y_test).to(device)
    return x_train, y_train, x_test, y_test


def _phantom_region_labels(z, caseno):
    """Binary region label (0=outside, 1=inside boundary) for latent points z [n,2],
    using the same phantom boundary image + [-3,3] grid as generate_Y_from_image."""
    from skimage import io, color, transform
    import numpy as np
    image_file = {4: 'bound3.png', 5: 'bound1.png', 6: 'bound4.png'}[caseno]
    bw = color.rgb2gray(io.imread(image_file)) > 0.5
    G = 100
    bw = transform.resize(bw, (G, G), anti_aliasing=False) > 0.5
    gx = np.linspace(-3, 3, G)
    gy = np.linspace(-3, 3, G)
    z_np = z.detach().cpu().numpy()
    idx0 = np.argmin(np.abs(gx[None, :] - z_np[:, 0:1]), axis=1)
    idx1 = np.argmin(np.abs(gy[None, :] - z_np[:, 1:2]), axis=1)
    inside = bw[idx1, idx0].astype(np.int64)
    return torch.as_tensor(inside, device=z.device)


def _safe_cholesky(K, jitter=1e-6, max_tries=6):
    eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter * eye)
        except RuntimeError:
            jitter *= 10.0
    return torch.linalg.cholesky(K + jitter * eye)


def _gp_sample_region_joint(z, region, is_train, means, theta1, theta2, noise_var):
    """One shared GP draw per region over ALL its points (train+test jointly), so the
    region's train and test targets come from the SAME latent function f. y = mean + f,
    with iid observation noise (var=noise_var) added to TRAIN points only (test noiseless)."""
    y = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for k in torch.unique(region):
        idx = (region == k).nonzero(as_tuple=False).squeeze(1)
        Zk = z[idx]
        d2 = (Zk.unsqueeze(1) - Zk.unsqueeze(0)).pow(2).sum(-1)
        K = theta1 * torch.exp(-d2 / theta2)
        L = _safe_cholesky(K)
        f = L @ torch.randn(len(idx), device=z.device, dtype=z.dtype)
        yk = float(means[int(k)]) + f
        tr = is_train[idx]
        noise = (float(noise_var) ** 0.5) * torch.randn(len(idx), device=z.device, dtype=z.dtype)
        yk = yk + noise * tr.to(z.dtype)
        y[idx] = yk
    return y


def generate_data_phantom_gp(N, Nt, device, caseno=6, means=(0.0, 27.0),
                             noise_var=4.0, theta1=9.0, theta2=200.0):
    """L2_gp: phantom two-region latent, but each region is a GP draw (not a flat level).
    y = means[region] + f + noise, f ~ GP(0, theta1*exp(-||z-z'||^2/theta2)).
    means[0]=outside, means[1]=inside boundary. Train and test share ONE GP realization
    per region (jointly sampled); iid noise (var=noise_var) on train targets only, test
    is the noiseless latent -> test is predictable from train within a region."""
    import numpy as np
    z = torch.from_numpy(np.random.uniform(-0.5, 0.5, (N + Nt, 2))).float().to(device)
    region = _phantom_region_labels(z, caseno)
    is_train = torch.zeros(N + Nt, dtype=torch.bool, device=z.device)
    is_train[:N] = True
    y = _gp_sample_region_joint(z, region, is_train, means, theta1, theta2, noise_var)
    return z[:N], y[:N], z[N:], y[N:]

def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    # 1) 生成原始数据
    if args.caseno == 0:
        x_train, y_train, x_test, y_test = generate_data(
            d   = args.d,
            N   = args.N,
            Nt  = args.Nt,
            device = args.device,
            noise_var = args.noise_var
        )
    else:
        x_train, y_train, x_test, y_test = generate_data_phantom(args.N, args.Nt, args.device, args.caseno, args.noise_std)
    print(f"原始 X_train: {x_train.shape}, X_test: {x_test.shape}")

    if args.ins_dim:    
            # 2) 采样 RFF 参数
        if args.methods == "rff":
            Omega, phases = sample_rff_params(
                d        = args.d,
                D_rff    = args.H,
                lengthscale = args.lengthscale,
                device   = device
            )

            # 3) 映射到高维特征空间
            Phi_train = make_rff_features(x_train, Omega, phases, args.kernel_var)
            Phi_test  = make_rff_features(x_test,  Omega, phases, args.kernel_var)
            print(f"RFF 映射后 Phi_train: {Phi_train.shape}, Phi_test: {Phi_test.shape}")

            data = {
            'X_train': Phi_train,
            'Y_train'    : y_train,
            'X_test' : Phi_test,
            'Y_test'     : y_test,
            'Omega'      : Omega,
            'phases'     : phases,
        }

        elif args.methods == "random projection":
            Omega = torch.randn(args.H-args.d, args.d, device=device)
            projected_train = x_train @ Omega.t()
            projected_test = x_test @ Omega.t()
            Phi_train = torch.cat([x_train, projected_train], dim=1)
            Phi_test = torch.cat([x_test, projected_test], dim=1)
            print(f"After random projection with original data, Phi_train: {Phi_train.shape}, Phi_test: {Phi_test.shape}")
            print(f"Original dim: {x_train.shape[1]}, Projected dim: {projected_train.shape[1]}, Total dim: {Phi_train.shape[1]}")

            data = {
                'X_train': Phi_train,
                'Y_train'    : y_train,
                'X_test' : Phi_test,
                'Y_test'     : y_test,
                'Omega'      : Omega,
            }

        elif args.methods == "polynomial":
            x_train = x_train
            x_test = x_test
            Phi_train = polynomial_transform(x_train)
            Phi_test = polynomial_transform(x_test)

            if Phi_train.shape[1] > args.H:
                Phi_train = Phi_train[:, :args.H]
                Phi_test = Phi_test[:, :args.H] 

            print(f"After polynomial transformation, x_train: {Phi_train.shape}, x_test: {Phi_test.shape}")

            data = {
                'X_train': Phi_train,
                'Y_train'    : y_train,
                'X_test' : Phi_test,
                'Y_test'     : y_test,
            }   

        elif args.methods == "autoencoder":
            Z_train, Z_test, model = autoencoder_transform(x_train, x_test, args.H)
            data = {
                'X_train': Z_train,
                'Y_train'    : y_train,
                'X_test' : Z_test,
                'Y_test'     : y_test,
            }
    else:
        data = {
            'X_train': x_train,
            'Y_train'    : y_train,
            'X_test' : x_test,
            'Y_test'     : y_test,
        }
    folder_name = save_dataset(data, args)
    return folder_name

if __name__ == '__main__':
    main()
