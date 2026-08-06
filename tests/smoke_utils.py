from __future__ import annotations

import pickle
from pathlib import Path

import torch


def make_tiny_dataset(folder: str | Path, n_train: int = 14, n_test: int = 3, dim: int = 3) -> Path:
    """Create a small deterministic dataset.pkl compatible with existing scripts."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(20260508)
    x_train = torch.randn(n_train, dim, generator=gen, dtype=torch.float32)
    x_test = torch.randn(n_test, dim, generator=gen, dtype=torch.float32)

    def response(x: torch.Tensor) -> torch.Tensor:
        smooth = torch.sin(x[:, 0]) + 0.25 * x[:, 1] - 0.1 * x[:, 2]
        jump = torch.where(x[:, 0] + x[:, 1] > 0, 1.0, -1.0)
        return smooth + jump

    data = {
        "X_train": x_train,
        "Y_train": response(x_train),
        "X_test": x_test,
        "Y_test": response(x_test),
        "args": {
            "source": "tests.smoke_utils.make_tiny_dataset",
            "n_train": n_train,
            "n_test": n_test,
            "dim": dim,
        },
    }

    dataset_path = folder / "dataset.pkl"
    with dataset_path.open("wb") as f:
        pickle.dump(data, f)
    return dataset_path
