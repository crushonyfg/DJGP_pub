"""Export UCI small-train splits for the external DeepMahalanobisGP runner.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/uci/export_smalltrain_dmgp_splits.py ^
        --train_sizes 200,500,1000 --num_exp 5 --max_test_anchors 200 ^
        --out_dir experiments/uci/dmgp_smalltrain_5seeds_20260517

    python experiments/uci/export_smalltrain_dmgp_splits.py ^
        --datasets "Wine Quality" --train_sizes 80 --num_exp 1 ^
        --max_test_anchors 20 ^
        --out_dir experiments/uci/dmgp_smalltrain_smoke

The exported files are consumed by
``experiments/dmgp/run_deep_mahalanobis_gp_npz.py`` in the separate
``dmgp_tf1`` environment.  Wine and Appliances use the historical random-row
split; Parkinsons uses the subject-grouped split.  By default Parkinsons
features are train-only standardized to match the latest LMJGP reporting
policy, while XGBoost/DKL comparison rows remain the existing raw-feature
best-preset rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.uci.compare_inductive_projection import _subsample  # noqa: E402
from experiments.uci.compare_parkinsons_grouped_split import _load_parkinsons_grouped_split  # noqa: E402
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DATASET_SLUGS = {
    "Wine Quality": "wine_quality",
    "Parkinsons Telemonitoring": "parkinsons_telemonitoring",
    "Appliances Energy Prediction": "appliances_energy_prediction",
}

DEFAULT_DATASETS = "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"

PRESETS = {
    "Wine Quality": {
        "split_type": "random_row",
        "standardize_x": True,
        "standardize_y_for_baseline": False,
    },
    "Parkinsons Telemonitoring": {
        "split_type": "subject_grouped",
        "standardize_x": True,
        "standardize_y_for_baseline": False,
    },
    "Appliances Energy Prediction": {
        "split_type": "random_row",
        "standardize_x": True,
        "standardize_y_for_baseline": True,
    },
}


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_datasets(value: str) -> List[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(out) - set(PRESETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return out


def _load_dataset(
    dataset_name: str,
    seed: int,
    *,
    train_size: int,
    max_test_anchors: int,
    train_subject_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Dict[str, Any]]:
    preset = PRESETS[dataset_name]
    if preset["split_type"] == "subject_grouped":
        split = _load_parkinsons_grouped_split(seed=seed, train_subject_frac=train_subject_frac)
        X_train, y_train, train_idx = _subsample(
            split["X_train"],
            split["y_train"],
            train_size,
            seed + 500,
        )
        X_test, y_test, test_idx = _subsample(
            split["X_test"],
            split["y_test"],
            max_test_anchors,
            seed + 1000,
        )
        meta = {
            **split["split_meta"],
            "split_type": "subject_grouped",
            "train_indices_within_group_pool": train_idx.tolist(),
            "test_indices_within_group_pool": test_idx.tolist(),
            "n_train_pool_rows": int(split["X_train"].shape[0]),
            "n_test_pool_rows": int(split["X_test"].shape[0]),
        }
        Q = int(split["K"])
    else:
        X_train_full, X_test_full, y_train_full, y_test_full, Q = load_uci_split(dataset_name, seed)
        X_train, y_train, train_idx = _subsample(X_train_full, y_train_full, train_size, seed + 500)
        X_test, y_test, test_idx = _subsample(X_test_full, y_test_full, max_test_anchors, seed + 1000)
        meta = {
            "split_type": "random_row",
            "train_indices_within_pool": train_idx.tolist(),
            "test_indices_within_pool": test_idx.tolist(),
            "n_train_pool_rows": int(X_train_full.shape[0]),
            "n_test_pool_rows": int(X_test_full.shape[0]),
        }
    return X_train, np.asarray(y_train).reshape(-1), X_test, np.asarray(y_test).reshape(-1), int(Q), meta


def _standardize_x(X_train: np.ndarray, X_test: np.ndarray, enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    if not enabled:
        return X_train, X_test
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def _export_one(
    *,
    out_dir: Path,
    dataset_name: str,
    train_size: int,
    seed: int,
    max_test_anchors: int,
    train_subject_frac: float,
) -> Dict[str, Any]:
    preset = PRESETS[dataset_name]
    X_train, y_train, X_test, y_test, Q, split_meta = _load_dataset(
        dataset_name,
        seed,
        train_size=train_size,
        max_test_anchors=max_test_anchors,
        train_subject_frac=train_subject_frac,
    )
    X_train_s, X_test_s = _standardize_x(X_train, X_test, bool(preset["standardize_x"]))
    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train))
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        y_scale = 1.0
    y_train_std = (np.asarray(y_train, dtype=np.float64).reshape(-1) - y_mean) / y_scale

    setting = f"uci_{DATASET_SLUGS[dataset_name]}_train{train_size}"
    rel_path = Path("data") / f"{setting}_seed{seed}.npz"
    npz_path = out_dir / rel_path
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train_s.astype(np.float64),
        X_test=X_test_s.astype(np.float64),
        y_train_std=y_train_std.astype(np.float64),
        y_test=np.asarray(y_test, dtype=np.float64).reshape(-1),
        y_mean=np.asarray([y_mean], dtype=np.float64),
        y_scale=np.asarray([y_scale], dtype=np.float64),
    )
    return {
        "benchmark": "uci",
        "dataset": dataset_name,
        "setting": setting,
        "seed": int(seed),
        "train_size": int(train_size),
        "Q": int(Q),
        "D": int(X_train_s.shape[1]),
        "N_train": int(X_train_s.shape[0]),
        "N_test": int(X_test_s.shape[0]),
        "path": str(rel_path).replace("\\", "/"),
        "split_type": preset["split_type"],
        "standardize_x": bool(preset["standardize_x"]),
        "standardize_y_for_baseline": bool(preset["standardize_y_for_baseline"]),
        "y_standardized_for_dmgp": True,
        "split_meta": split_meta,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export UCI small-train splits for DeepMahalanobisGP.")
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--train_sizes", default="200,500,1000")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument(
        "--parkinsons_standardize_x",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use train-only standardized Parkinsons features for DMGP when 1.",
    )
    p.add_argument(
        "--parkinsons_split",
        choices=("subject_grouped", "random_row"),
        default="subject_grouped",
        help="Parkinsons split type. 'random_row' = plain 90/10 row split "
        "(same instance/subject CAN leak across train/test); 'subject_grouped' "
        "= subjects disjoint (default, distribution-shift).",
    )
    p.add_argument("--out_dir", default="experiments/uci/dmgp_smalltrain_5seeds_20260517")
    return p.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    PRESETS["Parkinsons Telemonitoring"]["standardize_x"] = bool(args.parkinsons_standardize_x)
    PRESETS["Parkinsons Telemonitoring"]["split_type"] = args.parkinsons_split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for train_size in _parse_ints(args.train_sizes):
        for dataset_name in _parse_datasets(args.datasets):
            for seed in range(int(args.num_exp)):
                print(f"export {dataset_name} train={train_size} seed={seed}", flush=True)
                rows.append(
                    _export_one(
                        out_dir=out_dir,
                        dataset_name=dataset_name,
                        train_size=train_size,
                        seed=seed,
                        max_test_anchors=args.max_test_anchors,
                        train_subject_frac=args.train_subject_frac,
                    )
                )

    manifest = {
        "benchmark": "uci",
        "args": vars(args),
        "presets": PRESETS,
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = [
        "# UCI DeepMahalanobisGP Exports",
        "",
        "Produced by `experiments/uci/export_smalltrain_dmgp_splits.py`.",
        "",
        f"Rows: {len(rows)}",
        "",
        "Parkinsons uses the subject-grouped split. DMGP trains on standardized y and reports metrics on the original y scale.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote manifest -> {out_dir / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
