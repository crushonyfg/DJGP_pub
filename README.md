# Deep Jump Gaussian Process (DJGP)

Minimal reference implementation for *Deep Jump Gaussian Processes for Surrogate
Modeling of High-Dimensional Piecewise Continuous Functions*.

This release contains:

- the user-facing `djgp.model.DJGP` interface and its required implementation;
- the JumpGP, JGP-PCA, JGP-SIR, NGBoost, and BART baselines required by the paper table;
- one deterministic Parkinsons reproduction (`train=1000`, `test=200`, seeds 0–2);
- the exact subject-disjoint splits used by that reproduction;
- the L2 phantom reference image used in the synthetic study.

## L2 phantom reference

![L2 phantom reference](src/JumpGaussianProcess/bound3.png)

## Environment

Python 3.11 is required. The reported reference run used Python 3.11.18 on CPU. Install
the pinned environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Reproduce the Parkinsons table

Run one command from the repository root:

```bash
python reproduce_parkinson.py
```

No dataset download, GPU, tuning, or arguments are needed. The script verifies the SHA-256
checksums of the committed splits, forces deterministic single-threaded CPU execution, runs
three seeds, and prints the aggregate metrics. The CSV files retain full numerical
precision. Small differences from the table below are expected when package versions or
hardware differ from the pinned reference environment.

Outputs:

- `results/parkinson_results.csv`: aggregate mean and sample standard deviation;
- `results/parkinson_results_per_seed.csv`: one row per seed and method.

Expected aggregate table:

| Method | RMSE | CRPS | Cov90 |
|---|---:|---:|---:|
| JGP | 10.61 ± 0.08 | 7.47 ± 0.32 | 0.35 ± 0.07 |
| JGP-PCA | 9.97 ± 1.16 | 6.68 ± 1.18 | 0.43 ± 0.08 |
| JGP-SIR | 9.32 ± 0.29 | 6.32 ± 0.57 | 0.43 ± 0.07 |
| NGBoost | 9.69 ± 1.54 | 6.65 ± 1.20 | 0.44 ± 0.09 |
| BART | 10.25 ± 0.78 | 7.06 ± 0.49 | 0.35 ± 0.02 |
| **DJGP-SIR (single member)** | **7.57 ± 1.30** | **4.60 ± 0.62** | **0.94 ± 0.07** |

DGP is deliberately excluded. DJGP uses one SIR-initialized member—there is no ensemble.

## Use DJGP on another dataset

DJGP is transductive: training occurs around the supplied test inputs.

```python
from djgp import DJGP

model = DJGP()
model.fit(X_train, y_train, X_test)
mean, standard_deviation = model.predict()
```

To run a single SIR-initialized member:

```python
model = DJGP(K_members=1, topk=1, init_methods=("sir",))
model.fit(X_train, y_train, X_test)
mean, standard_deviation = model.predict()
```

Constructor options and the `fit`, `predict`, `update`, and `tune` methods are documented in
`djgp.model.DJGP`.
