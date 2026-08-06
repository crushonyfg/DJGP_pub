# Deep Jump Gaussian Process (DJGP)

Reference implementation for *"Deep Jump Gaussian Processes for Surrogate Modeling of
High-Dimensional Piecewise Continuous Functions."*

DJGP is a surrogate model for piecewise continuous functions on high-dimensional domains.
For each test location it selects a small neighbourhood of training points, maps them to a
low-dimensional space through a **region-specific locally linear projection** (a Gaussian
process prior lets these projections vary smoothly across the input space), and fits a
local **Jump Gaussian Process** in the projected space to capture the discontinuities. The
projection matrices and the local Jump-GP hyperparameters are learned jointly by
variational inference. Prediction is transductive — the local models are trained around the
test locations.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt
pip install -e .             # makes src/* importable as top-level packages
```

---

## Reproducing benchmark results

### Synthetic datasets (L2, LH)

The synthetic data follow the two-stage construction in the paper: a low-dimensional latent
family (**L2**, latent dimension K=2, and **LH**, latent dimension K=4/5/7) with a piecewise
continuous response, lifted to an observed dimension D by one of five dimensionality-expansion
techniques (`rff, poly, ae, randproj, manifold`). This gives 20 settings; each is generated
on the fly (train=1000 / test=200, seeded).

Baselines: Jump GP (`jgp`), Jump GP after PCA (`jgp_pca`) or sliced inverse regression
(`jgp_sir`), Deep GP (`dgp`), NGBoost (`ngboost`), and BART (`bart`). DJGP hyperparameters
are loaded from the frozen configuration files in `docs/benchmark_configs/` — there is no
tuning phase in this script.

```bash
# Quick check — one setting, seed 0
python run_synthetic.py --settings l2_d20_rff --seeds 0

# Full reproduction (all 20 settings, all methods, 25 seeds; long)
python run_synthetic.py

# Subsets
python run_synthetic.py --settings lh_d30z5_rff,lh_d30z5_poly --methods djgp,jgp,ngboost
```

Per-seed rows go to `results/synthetic/metrics.csv`, seed-aggregated means to
`aggregate.csv` beside it (RMSE / CRPS / cov90).

### UCI datasets (Wine Quality, Parkinsons, Appliances)

Six configurations: Wine and Appliances at train=1000 and train=5000, Parkinsons at
train=1000 and train=4000 (subject-grouped). The train/test splits are derived data and are
not committed — generate them once (fetches the datasets via `ucimlrepo`):

```bash
python scripts/prepare_uci_data.py            # both regimes, seeds 0..24
```

Then run the benchmark (same methods as above; DJGP hyperparameters loaded from
`docs/benchmark_configs/uci_*.json`):

```bash
# Quick check
python run_uci.py --datasets Wine --sizes 1000 --seeds 0

# Full reproduction (6 configs, all methods, 25 seeds; long)
python run_uci.py
```

Results go to `results/uci/metrics.csv` with an aggregate table printed at the end
(RMSE / CRPS / cov90).

---

## Using DJGP on your own data

The user-facing class is `djgp.model.DJGP`, with the benchmark defaults built in. DJGP is
**transductive**: the local models are trained around the test inputs, so the test inputs are
part of fitting.

```python
import numpy as np
from djgp.model import DJGP

# X_train: [N, D] float array   y_train: [N]   X_test: [T, D]
model = DJGP()                           # defaults = the benchmark pipeline
model.fit(X_train, y_train, X_test)      # trains around the test locations
mu, std = model.predict()                # predictive mean / std, original y scale
```

**Post-training** — new training data and/or new test locations (both optional), three modes:

```python
# 1) keep the learned projections fixed; only the per-location local layers train (cheapest)
mu, std = model.update(X_test_new=X_new, mode="freeze_w")
# 2) warm-start the projections from the learned posterior and fine-tune everything
mu, std = model.update(X_train_new=Xtr2, y_train_new=ytr2, X_test_new=X_new, mode="finetune_w")
# 3) full retrain from scratch on the pooled old+new data
mu, std = model.update(X_train_new=Xtr2, y_train_new=ytr2, X_test_new=X_new, mode="retrain")
```

**Optional single-dataset tuning** (a class method; one random holdout split of your
training data — *not* cross-validation):

```python
model.tune(X_train, y_train)
model.fit(X_train, y_train, X_test)
mu, std = model.predict()
```

All constructor options (defaults = the benchmark pipeline) are documented in the class
docstring — see `help(djgp.model.DJGP)`. Frozen benchmark hyperparameters can be loaded
directly: `DJGP.from_benchmark_json("docs/benchmark_configs/lh_d30z5_rff.json")`.

---

## Package layout

```
src/
  djgp/               # DJGP model and projections
    model.py          # public DJGP class (fit / predict / update / tune)
    projections/      # core model implementation
    baselines/        # NGBoost and global sparse-GP baselines
    evaluation/       # CRPS, calibration metrics
  jumpgp/             # Jump GP
  shared/             # utilities, Deep GP, runners
  data_gen/           # synthetic data generators
experiments/          # benchmark harnesses used by the entry points below
scripts/
  prepare_uci_data.py # one-time UCI split generation
run_synthetic.py      # benchmark entry point (synthetic)
run_uci.py            # benchmark entry point (UCI)
```

---

## Smoke tests

```bash
python -m pytest tests/ -q
```
