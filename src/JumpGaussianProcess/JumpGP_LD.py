"""Linear-boundary Jump Gaussian Process entrypoint.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
    python JumpGaussianProcess\\JumpGP_LD.py

Most project code should call this through ``jumpgp.linear.JumpGP_LD`` or
``utils1.jumpgp_ld_wrapper``.
"""

# % ***************************************************************************************
# %
# % JumpGP_LD - The function implements Jump GP with a linear decision boundary function
# %             described in the paper,
# % 
# %   Park, C. (2022) Jump Gaussian Process Model for Estimating Piecewise 
# %   Continuous Regression Functions. Journal of Machine Learning Research.
# %   23. 
# % 
# %
# % Inputs:
# %       x - training inputs
# %       y - training responses
# %       xt - test inputs
# %       mode - inference algorithm. It can be either 
# %                        'CEM' : Classification EM Algorithm
# %                        'VEM' : Variational EM Algorithm
# %                        'SEM' : Stochastic EM Algorithm 
# %       bVerbose (Internal Use for Debugging) 
# %                  0: do not visualize output
# %                  1: visualize output 
# % Outputs:
# %       mu_t - mean prediction at xt
# %       sig2_t - variance prediction at xt
# %       model - fitted JGP model
# %       h     - (internal use only) 
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from cov.covSum import covSum
from cov.covSEard import covSEard
from cov.covNoise import covNoise
from lik.loglikelihood import loglikelihood
from local_linearfit import local_linearfit
from maximize_PD import maximize_PD
from variationalEM import variationalEM
from stochasticEM import stochasticEM
from calculate_gx import calculate_gx

# Main function for JumpGP
def JumpGP_LD(x, y, xt, mode, bVerbose=False, *args):
    """
    JumpGP_LD - The function implements Jump GP with a linear decision boundary function
    
    Parameters:
    -----------
    x : array-like
        Training inputs
    y : array-like
        Training responses
    xt : array-like
        Test inputs
    mode : str
        Inference algorithm. It can be either:
        'CEM' : Classification EM Algorithm
        'VEM' : Variational EM Algorithm
        'SEM' : Stochastic EM Algorithm
    bVerbose : bool, optional
        Whether to visualize output (for debugging)
    args : tuple, optional
        Additional arguments, including logtheta if provided
    
    Returns:
    --------
    mu_t : array-like
        Mean prediction at xt
    sig2_t : array-like
        Variance prediction at xt
    model : dict
        Fitted JGP model
    h : list
        Plot handles (for internal use only)
    """
    cv = [covSum, [covSEard, covNoise]]
    d = x.shape[1]
    px = x
    pxt = xt

    # Initial estimation of the boundary B(x)
    if len(args) > 0 and args[0] is not None:
        logtheta = args[0]
    else:
        logtheta = np.zeros(d + 2)
        logtheta[-1] = -1.15

    w, _ = local_linearfit(x, y, xt[0])  # Use only the first test point for initial fit
    w = np.asarray(w, dtype=float).reshape(-1)
    nw = np.linalg.norm(w)
    if not np.isfinite(nw) or nw <= 1e-12:
        w = np.zeros(d + 1, dtype=float)
        w[0] = 1.0
        nw = 1.0
    w = w / nw

    # Fine-tune the intercept term
    w1 = np.asarray(w).ravel()
    b = np.arange(-1 + w1[0], 1 + w1[0], 0.01)
    fd = []
    min_side = 2 if x.shape[0] >= 4 else 1
    for bi in b:
        w_d = w.copy()
        w_d[0] = bi
        gx, _ = calculate_gx(px, w_d)
        r = np.asarray(gx >= 0).reshape(-1)
        if np.sum(r) < min_side or np.sum(~r) < min_side:
            fd.append(np.inf)
            continue
        y_pos = y.reshape(-1)[r]
        y_neg = y.reshape(-1)[~r]
        fd.append(np.mean(r) * np.var(y_pos, ddof=1) + np.mean(~r) * np.var(y_neg, ddof=1))
    
    if not np.isfinite(fd).any():
        fd = []
        for bi in b:
            w_d = w.copy()
            w_d[0] = bi
            gx, _ = calculate_gx(px, w_d)
            r = np.asarray(gx >= 0).reshape(-1)
            y_pos = y.reshape(-1)[r]
            y_neg = y.reshape(-1)[~r]
            var_pos = np.var(y_pos, ddof=1) if y_pos.size > 1 else 0.0
            var_neg = np.var(y_neg, ddof=1) if y_neg.size > 1 else 0.0
            fd.append(np.mean(r) * var_pos + np.mean(~r) * var_neg)
    k = int(np.nanargmin(fd))
    w[0] = b[k]
    w = nw * w

    # Select algorithm
    if mode == 'CEM':
        model = maximize_PD(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'VEM':
        model = variationalEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'SEM':
        model = stochasticEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)

    
    mu_t = model['mu_t']
    sig2_t = model['sig2_t']

    h = []
    if bVerbose:
        a = np.array([[1, -0.5], [1, 0.5]])
        b = -a @ model['w'][0:2] / model['w'][2]
        h1 = plt.plot(a, b, 'r', linewidth=3)
        gx, _ = calculate_gx(px, model['w'])
        # print("gx", gx.shape)
        gx = gx.ravel()
        h2 = plt.scatter(x[gx >= 0, 0], x[gx >= 0, 1], color='g', marker='s')
        h = [h2, h1[0]]
    
    return mu_t, sig2_t, model, h

# Example usage
if __name__ == "__main__":
    x_train = np.random.rand(100, 2)
    y_train = np.random.rand(100)
    x_test = np.random.rand(20, 2)
    mu_t, sig2_t, model, h = JumpGP_LD(x_train, y_train.reshape(-1, 1), x_test, 'VEM', 1)
    print("Successfully run the example!")
