# AE.py
# A lightweight, flexible AutoEncoder trainer for dimensionality reduction.
# Usage:
#   from AE import AE_dim
#   Z_train, Z_test, model = AE_dim(X_train, X_test, latent_dim=5)
#
# Notes:
# - X_train, X_test are torch.FloatTensor of shapes [N, D] and [M, D].
# - y tensors are not required for AE training (unsupervised).
# - You can specify encoder_layers (e.g., [64, 32, 5]); decoder is auto-mirrored.

'''
from data_gen.autoencoder import AE_dim

Z_train, Z_test, model = AE_dim(
    X_train, X_test,
    latent_dim=5,                 # or直接给 encoder_layers=[64,32,5]
    encoder_layers=None,          # 若为 None,默认用 [64,32,latent_dim]
    epochs=100,
    patience=10,
    batch_size=128,
    val_ratio=0.1,
    lr=1e-3,
    weight_decay=1e-5,
    dropout=0.0,
    use_tqdm=True,                # 关掉进度条:False
    seed=42
)


'''

import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# optional tqdm; fail gracefully if not installed
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# --------------------------
# Utils
# --------------------------
def _make_mlp(sizes, last_activation=None, act=nn.ReLU, dropout=0.0):
    """
    Build a simple MLP from a list of sizes, e.g. [18, 64, 32, 5].
    If last_activation is not None, apply it after the final Linear.
    Dropout is applied to hidden layers only.
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = (i == len(sizes) - 2)
        if not is_last:
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        elif last_activation is not None:
            layers.append(last_activation())
    return nn.Sequential(*layers)

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# --------------------------
# Model
# --------------------------
class AutoEncoder(nn.Module):
    """
    Symmetric AutoEncoder. Provide encoder_layers like [64, 32, 5] to build:
    encoder: in_dim -> 64 -> 32 -> 5
    decoder: 5 -> 32 -> 64 -> in_dim
    """
    def __init__(self, in_dim: int, encoder_layers, act=nn.ReLU, dropout: float = 0.0):
        super().__init__()
        assert isinstance(in_dim, int) and in_dim > 0
        assert len(encoder_layers) >= 1, "encoder_layers must include latent_dim at the end."
        self.in_dim = in_dim
        self.latent_dim = int(encoder_layers[-1])

        enc_sizes = [in_dim] + list(encoder_layers)
        self.encoder = _make_mlp(enc_sizes, last_activation=None, act=act, dropout=dropout)

        # Mirror decoder sizes; last layer has no activation (reconstruction)
        dec_sizes = [self.latent_dim] + list(encoder_layers[-2::-1]) + [in_dim]
        self.decoder = _make_mlp(dec_sizes, last_activation=None, act=act, dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z

# --------------------------
# Public API
# --------------------------
def AE_dim(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    latent_dim: int = 5,
    encoder_layers=None,           # e.g. [64, 32, 5]; if None -> [64, 32, latent_dim]
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.0,
    use_tqdm: bool = True,
    seed: int = 42,
    device: str = None,
):
    """
    Train an AutoEncoder to reduce dimension of X from D to latent_dim (or last of encoder_layers).

    Returns:
        Z_train (torch.Tensor) : [N, latent_dim]
        Z_test  (torch.Tensor) : [M, latent_dim]
        model   (nn.Module)    : trained AutoEncoder

    Args:
        X_train, X_test: Float tensors with shape [N, D] and [M, D].
        latent_dim: used if encoder_layers is None.
        encoder_layers: list like [64, 32, latent_dim]. Decoder is mirrored.
        epochs, patience: early stopping config.
        batch_size: training batch size.
        val_ratio: fraction of X_train used as validation.
        lr, weight_decay: optimizer settings (Adam).
        dropout: dropout on hidden layers.
        use_tqdm: show progress bar if tqdm is available.
        seed: random seed.
        device: "cuda", "cpu", or None to auto-detect.
    """
    # ---- Basic checks ----
    assert isinstance(X_train, torch.Tensor) and isinstance(X_test, torch.Tensor), \
        "X_train and X_test must be torch.Tensor."
    assert X_train.dim() == 2 and X_test.dim() == 2, "Expect 2D tensors [N, D] and [M, D]."
    assert X_train.size(1) == X_test.size(1), "Train/Test feature dimensions must match."
    in_dim = X_train.size(1)

    if encoder_layers is None:
        encoder_layers = [64, 32, int(latent_dim)]
    else:
        assert int(encoder_layers[-1]) > 0, "Last element of encoder_layers must be latent_dim."
        latent_dim = int(encoder_layers[-1])

    # ---- Device & seeding ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Standardize by train stats ----
    with torch.no_grad():
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

    # ---- Split train/val ----
    val_size = int(len(X_train_norm) * float(val_ratio))
    if val_size > 0 and val_size < len(X_train_norm):
        X_trn = X_train_norm[:-val_size]
        X_val = X_train_norm[-val_size:]
        train_loader = DataLoader(TensorDataset(X_trn), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(X_train_norm), batch_size=batch_size, shuffle=True)
        val_loader = None

    # ---- Build model ----
    model = AutoEncoder(in_dim=in_dim, encoder_layers=encoder_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training helpers ----
    def run_epoch(dl, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for (xb,) in dl:
                xb = xb.to(device)
                xrec, _ = model(xb)
                loss = criterion(xrec, xb)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                bs = xb.size(0)
                total += loss.item() * bs
                n += bs
        return total / max(1, n)

    # ---- Train with early stopping ----
    best_val = math.inf
    best_state = None
    no_improve = 0

    iterator = range(1, epochs + 1)
    if use_tqdm and _HAS_TQDM:
        iterator = tqdm(iterator, desc="Training AE", leave=False)

    for _ in iterator:
        train_loss = run_epoch(train_loader, train=True)
        if val_loader is not None:
            val_loss = run_epoch(val_loader, train=False)
            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                break
        # if no val_loader, just loop epochs (no early stop)

    # ---- Encode to latent ----
    model.eval()
    with torch.no_grad():
        Z_train = model.encoder(X_train_norm.to(device)).cpu()
        Z_test = model.encoder(X_test_norm.to(device)).cpu()

    return Z_train, Z_test, model

# End of file
