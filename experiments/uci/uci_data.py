"""UCI train/test splits used by benchmarks (matches `new_dataset.py` logic)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler


@dataclass
class UciSplitMeta:
    """Optional per-split metadata filled by ``load_uci_split`` when requested."""

    parkinsons_subject_train: np.ndarray | None = None
    parkinsons_subject_test: np.ndarray | None = None


def sir_reduction(X, y, H=10, K=5):
    """Sliced inverse regression (same as `new_dataset.py`)."""
    y = np.asarray(y).ravel()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    percentiles = np.linspace(0, 100, H + 1)[1:-1]
    bin_edges = np.percentile(y, percentiles)
    y_binned = np.digitize(y, bin_edges)
    means = np.zeros((H, X.shape[1]))
    props = np.zeros(H)
    for h in range(H):
        idx = y_binned == h
        if np.any(idx):
            means[h] = np.nan_to_num(X_std[idx].mean(axis=0))
            props[h] = idx.mean()
    props = props / np.sum(props)
    m = np.zeros_like(means[0])
    V = np.zeros((X.shape[1], X.shape[1]))
    for h in range(H):
        if props[h] > 0:
            diff = means[h] - m
            V += props[h] * np.outer(diff, diff)
    V += 1e-8 * np.eye(V.shape[0])
    try:
        _, eigvecs = eigh(V)
    except np.linalg.LinAlgError:
        _, eigvecs = np.linalg.svd(V)[:2]
        eigvecs = eigvecs.T
    selected_vecs = eigvecs[:, -K:]
    X_transformed = X_std @ selected_vecs
    return X_transformed, selected_vecs, scaler


def load_uci_split(
    dataset_name: str,
    seed: int,
    *,
    meta: UciSplitMeta | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Return X_train, X_test, y_train, y_test (numpy), and K (SIR/PCA dimension).

    Split ratios match `experiments/uci/new_dataset.py`.

    For **Parkinsons Telemonitoring**, pass ``meta=UciSplitMeta()`` to record
    the true ``subject#`` ID column from ``repo.data.ids`` for
    train/test rows — useful for grouped-split / neighbor diagnostics. Model
    features in ``X_*`` still match the historical pipeline (last feature column dropped).
    """
    from ucimlrepo import fetch_ucirepo  # lazy: only UCI data loading needs the network

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if dataset_name == "Appliances Energy Prediction":
        repo = fetch_ucirepo(id=374)
        X = repo.data.features
        y = repo.data.targets
        X_np = X.iloc[:, 1:].to_numpy()
        y_np = y.to_numpy()
        indices = np.random.permutation(X_np.shape[0])
        split = int(X_np.shape[0] * 0.97)
        train_idx, test_idx = indices[:split], indices[split:]
        X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
        y_train_np, y_test_np = y_np[train_idx], y_np[test_idx]
        K = 5
    elif dataset_name == "Parkinsons Telemonitoring":
        repo = fetch_ucirepo(id=189)
        X = repo.data.features
        y = repo.data.targets
        subject_ids = repo.data.ids["subject#"].to_numpy()
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        indices = np.random.permutation(X_np.shape[0])
        split = int(X_np.shape[0] * 0.9)
        train_idx, test_idx = indices[:split], indices[split:]
        if meta is not None:
            meta.parkinsons_subject_train = np.asarray(subject_ids[train_idx]).ravel()
            meta.parkinsons_subject_test = np.asarray(subject_ids[test_idx]).ravel()
        X_train_np, X_test_np = X_np[train_idx, :-1], X_np[test_idx, :-1]
        y_train_np, y_test_np = y_np[train_idx, 0], y_np[test_idx, 0]
        K = 5
    elif dataset_name == "Wine Quality":
        repo = fetch_ucirepo(id=186)
        X = repo.data.features
        y = repo.data.targets
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        indices = np.random.permutation(X_np.shape[0])
        split = int(X_np.shape[0] * 0.9)
        train_idx, test_idx = indices[:split], indices[split:]
        X_train_np, X_test_np = X_np[train_idx, :], X_np[test_idx, :]
        y_train_np, y_test_np = y_np[train_idx, 0], y_np[test_idx, 0]
        K = 3
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}")

    return X_train_np, X_test_np, y_train_np, y_test_np, K
