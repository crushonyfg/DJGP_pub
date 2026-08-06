# JumpGP: Jump Gaussian Process for Local Regression

JumpGP is a Python implementation of Jump Gaussian Process, which performs local regression by selecting nearest neighbors and fitting Gaussian Process models. It supports both linear (LD) and quadratic (QD) detrending methods.

## Features

- Local Gaussian Process regression with neighborhood selection
- Support for both linear and quadratic detrending
- Multiple inference modes (CEM/VEM)
- Evaluation metrics including RMSE and CRPS
- Easy-to-use interface

## Installation

```bash
git clone https://github.com/crushonyfg/JumpGaussianProcess.git
cd JumpGaussianProcess
```

## Dependencies

- numpy
- scipy
- matplotlib
- tqdm

## Quick Start

```python
import numpy as np
from jumpgp import JumpGP

# Generate sample data
x_train = np.random.rand(100, 2)  # 100 training points in 2D
y_train = np.random.rand(100)     # training targets
x_test = np.random.rand(20, 2)    # 20 test points
y_test = np.random.rand(20)       # test targets

# Initialize and fit JumpGP
model = JumpGP(x_train, y_train, x_test, 
               L=2,           # Order of detrending (1: linear, 2: quadratic)
               M=20,         # Number of nearest neighbors
               mode='VEM',   # Inference mode ('CEM' or 'VEM')
               bVerbose=False)

# Fit the model
results = model.fit()

# Evaluate predictions
rmse, mean_crps = model.metrics(y_test)
print(f"RMSE: {rmse}, Mean CRPS: {mean_crps}")
```

## API Reference

### JumpGP Class

```python
JumpGP(x, y, xt, L=1, M=20, mode='CEM', bVerbose=False)
```

Parameters:
- `x`: Training input features (N × D matrix)
- `y`: Training targets (N × 1 vector)
- `xt`: Test input features (T × D matrix)
- `L`: Order of detrending (1: linear, 2: quadratic)
- `M`: Number of nearest neighbors
- `mode`: Inference mode ('CEM' or 'VEM')
- `bVerbose`: Whether to print detailed information

### Methods

- `fit()`: Fit the model and return predictions
- `metrics(yt)`: Compute RMSE and CRPS for test targets

## Output Format

The `fit()` method returns a dictionary containing:
- `mu`: Predicted mean values
- `sig2`: Predicted variances
- `models`: Fitted local GP models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@article{park2022jump,
  title={Jump gaussian process model for estimating piecewise continuous regression functions},
  author={Park, Chiwoo},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={278},
  pages={1--37},
  year={2022}
}
```
