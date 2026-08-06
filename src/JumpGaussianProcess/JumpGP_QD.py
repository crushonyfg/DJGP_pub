import numpy as np
import argparse
from dataclasses import dataclass
from scipy.optimize import minimize
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

from cov.covSum import covSum
from cov.covSEard import covSEard
from cov.covNoise import covNoise
from lik.loglikelihood import loglikelihood
from local_linearfit import local_linearfit
from maximize_PD import maximize_PD
from calculate_gx import calculate_gx
from local_qfit import local_qfit
from variationalEM import variationalEM
from stochasticEM import stochasticEM

@dataclass
class JumpGPQDModel:
    mu_t: np.ndarray
    sig2_t: np.ndarray
    w: np.ndarray

def JumpGP_QD(x, y, xt, mode, bVerbose=False, *args):
    """
    JumpGP_QD - The function implements Jump GP with a quadratic decision boundary function
    """
    d = x.shape[1]
    cv = [covSum, [covSEard, covNoise]]
    # Quadratic feature expansion
    g1, g2 = np.meshgrid(np.arange(d), np.arange(d))
    id_mask = (g1 >= g2)
    px = np.hstack([x, x[:, g1[id_mask]] * x[:, g2[id_mask]]])
    pxt = np.hstack([xt, xt[:, g1[id_mask]] * xt[:, g2[id_mask]]])

    # Initial estimation of the boundary B(x)
    logtheta = np.zeros(d + 2)
    logtheta[-1] = -1.15
    w, _ = local_qfit(x, y, xt[0])  # Use only the first test point for initial fit
    nw = np.linalg.norm(w)
    w = w / nw

    # Fine-tune the intercept term
    b = np.arange(-0.2 + w[0].item(), 0.2 + w[0].item(), 0.0005)
    fd = []
    for bi in b:
        w_d = w.copy()
        w_d[0] = bi
        gx, _ = calculate_gx(px, w_d)
        r = gx >= 0
        r1 = r.flatten()
        # directly use loglikelihood to sum
        fd.append(
            loglikelihood(logtheta, cv[0], cv[1], x[r1, :], y[r1]) +
            loglikelihood(logtheta, cv[0], cv[1], x[~r1, :], y[~r1])
        )
    k = np.argmin(fd)
    w[0] = b[k]
    w = nw * w


    # Select algorithm
    if mode == 'CEM':
        model = maximize_PD(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'VEM':
        model = variationalEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'SEM':
        model = stochasticEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    mu_t = model['mu_t']
    sig2_t = model['sig2_t']

    h = []
    if bVerbose:
        gx_grid = np.arange(0, 1.025, 0.025) - 0.5
        L = len(gx_grid)
        ptx, pty = np.meshgrid(gx_grid, gx_grid)
        allx = np.column_stack([ptx.ravel(), pty.ravel()])
        allx_quad = np.hstack([allx, allx[:, g1[id_mask]] * allx[:, g2[id_mask]]])
        gy, _ = calculate_gx(allx_quad, model['w'])
        gy = np.sign(gy)
        h1 = plt.contour(gx_grid, gx_grid, gy.reshape(L, L), levels=[1], colors='r', linewidths=3)
        gx, _ = calculate_gx(px, model['w'])
        h2 = plt.scatter(x[gx >= 0, 0], x[gx >= 0, 1], color='g', marker='s')
        h = [h2, h1]
        plt.show()
    return mu_t, sig2_t, model, h

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jump GP with Quadratic Decision Boundary")
    parser.add_argument('--mode', type=str, default='CEM', help="Inference algorithm ('CEM', 'VEM', 'SEM')")
    parser.add_argument('--verbose', type=int, default=0, help="Verbose output (0 or 1)")
    
    args = parser.parse_args()

    # Example dummy data (replace with actual)
    x = np.random.rand(100, 2)
    y = np.random.rand(100)
    xt = np.random.rand(10, 2)

    mu_t, sig2_t, model, h = JumpGP_QD(x, y.reshape(-1, 1), xt[0].reshape(1, -1), mode=args.mode, bVerbose=args.verbose)
    print(f"mu_t: {mu_t}, sig2_t: {sig2_t}")
