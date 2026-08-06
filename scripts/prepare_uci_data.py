"""Generate the UCI train/test splits that ``run_uci.py`` consumes.

The benchmark reads pre-exported ``.npz`` splits from
``experiments/uci/_train1000_8seeds/data`` and ``experiments/uci/_train5000_8seeds/data``.
These are NOT committed to the repo (they are derived data); run this script once to
create them. It fetches the three UCI datasets via ``ucimlrepo`` and writes, per
(dataset, size, seed), an ``uci_<slug>_train<size>_seed<seed>.npz`` with keys
``X_train, X_test, y_train_std, y_test, y_mean, y_scale`` (train-only standardized).

Splits (matching the paper): Wine / Appliances use a random-row split; Parkinsons uses
a subject-grouped split (subjects disjoint between train and test). Seeds are
deterministic, so seed s always reproduces the same split.

  1000 regime : Wine=1000, Appliances=1000, Parkinsons=1000  -> _train1000_8seeds/data
  5000 regime : Wine=5000, Appliances=5000, Parkinsons=4000  -> _train5000_8seeds/data

Run (from the repo root, in an env with ``ucimlrepo`` installed):
    python scripts/prepare_uci_data.py                 # both regimes, seeds 0..24
    python scripts/prepare_uci_data.py --seeds 5       # a quick check (seeds 0..4)
    python scripts/prepare_uci_data.py --regimes 1000  # only the train=1000 regime
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.uci.export_smalltrain_dmgp_splits import PRESETS, _export_one  # noqa: E402

# (dataset_name, train_size) per regime, matching run_uci.TRAIN_SIZE.
REGIMES = {
    "1000": [("Wine Quality", 1000), ("Appliances Energy Prediction", 1000),
             ("Parkinsons Telemonitoring", 1000)],
    "5000": [("Wine Quality", 5000), ("Appliances Energy Prediction", 5000),
             ("Parkinsons Telemonitoring", 4000)],
}
OUT_DIR = {"1000": REPO / "experiments/uci/_train1000_8seeds",
           "5000": REPO / "experiments/uci/_train5000_8seeds"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate UCI splits for run_uci.py")
    ap.add_argument("--seeds", type=int, default=25, help="generate seeds 0..seeds-1")
    ap.add_argument("--regimes", default="1000,5000")
    ap.add_argument("--max_test_anchors", type=int, default=500)
    ap.add_argument("--train_subject_frac", type=float, default=0.8)
    a = ap.parse_args()
    # Parkinsons: subject-grouped, train-only standardized features (paper setting).
    PRESETS["Parkinsons Telemonitoring"]["standardize_x"] = True
    PRESETS["Parkinsons Telemonitoring"]["split_type"] = "subject_grouped"
    regimes = [r for r in a.regimes.split(",") if r in REGIMES]
    for regime in regimes:
        out_dir = OUT_DIR[regime]
        for dataset_name, size in REGIMES[regime]:
            for seed in range(int(a.seeds)):
                _export_one(out_dir=out_dir, dataset_name=dataset_name, train_size=int(size),
                            seed=seed, max_test_anchors=int(a.max_test_anchors),
                            train_subject_frac=float(a.train_subject_frac))
                print(f"[{regime}] {dataset_name} train={size} seed={seed} -> "
                      f"{out_dir.name}/data", flush=True)
    print("done. run_uci.py can now read the splits.")


if __name__ == "__main__":
    main()
