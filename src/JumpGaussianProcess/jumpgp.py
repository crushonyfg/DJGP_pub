import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from JumpGP_LD import JumpGP_LD
from JumpGP_QD import JumpGP_QD

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def _cem_worker(args):
    """Top-level worker function for ProcessPoolExecutor (must be picklable)."""
    t, x_nbr, y_nbr, xt_t, L, mode, bVerbose = args
    if L == 1:
        mu_t, sig2_t, model, h = JumpGP_LD(x_nbr, y_nbr, xt_t, mode, bVerbose)
    else:
        from JumpGP_QD import JumpGP_QD as _QD
        mu_t, sig2_t, model, h = _QD(x_nbr, y_nbr, xt_t, mode, bVerbose)
    return t, float(np.asarray(mu_t).ravel()[0]), float(np.asarray(sig2_t).ravel()[0]), model


def find_neighborhoods(X_test, X_train, Y_train, M):
    """Find M nearest neighbors for each test point"""
    T = X_test.shape[0]
    dists = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))
    indices = np.argsort(dists, axis=1)[:, :M]

    neighborhoods = []
    for t in range(T):
        idx = indices[t]
        neighborhoods.append({
            "X_neighbors": X_train[idx],
            "y_neighbors": Y_train[idx].reshape(-1,1),
            "indices": idx
        })
    return neighborhoods


def compute_metrics(predictions, sigmas, Y_test):
    """Compute RMSE and mean CRPS for Gaussian predictive distributions."""
    rmse = np.sqrt(np.mean((predictions - Y_test)**2))
    z = (Y_test - predictions) / sigmas
    cdf = 0.5 * (1 + scipy.special.erf(z / np.sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    crps = sigmas * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    mean_crps = np.mean(crps)
    return rmse, mean_crps


class JumpGP:
    def __init__(self, x, y, xt, L=1, M=20, mode='CEM', bVerbose=False):
        self.x = x
        self.y = y
        self.xt = xt
        self.L = L
        self.M = M
        self.mode = mode
        self.bVerbose = bVerbose
        self.neighborhoods = find_neighborhoods(xt, x, y, M)
        self.jump_results = {}

    def fit(self):
        mu, sig2 = [], []
        models = []
        for t in tqdm(range(self.xt.shape[0])):
            x = self.neighborhoods[t]["X_neighbors"]
            y = self.neighborhoods[t]["y_neighbors"]
            xt = self.xt[t].reshape(1,-1)
            if self.L == 1:
                mu_t, sig2_t, model, h = JumpGP_LD(x, y, xt, self.mode, self.bVerbose)
            else:
                mu_t, sig2_t, model, h = JumpGP_QD(x, y, xt, self.mode, self.bVerbose)
            mu.append(mu_t)
            sig2.append(sig2_t)
            models.append(model)
        mu = np.array(mu)
        sig2 = np.array(sig2)
        self.jump_results = {"mu":mu, "sig2":sig2, "models":models}
        return self.jump_results

    def metrics(self, yt):
        mu = self.jump_results["mu"]
        sig2 = self.jump_results["sig2"]
        models = self.jump_results["models"]
        mu = np.asarray(mu).ravel()
        sig2 = np.asarray(sig2).ravel()
        yt = np.asarray(yt).ravel()
        if not (mu.shape == sig2.shape == yt.shape):
            raise ValueError("Shape mismatch: mu {}, sig2 {}, yt {}".format(
                mu.shape, sig2.shape, yt.shape))
        if len(mu.shape) != 1:
            raise ValueError("Arrays should be 1-dimensional, got shape {}".format(mu.shape))
        sigmas = np.sqrt(sig2)
        rmse, mean_crps = compute_metrics(mu, sigmas, yt)
        return rmse, mean_crps


class JumpGPBatch:
    """Batch (parallelised) variant of JumpGP using ProcessPoolExecutor.

    The API is identical to JumpGP.  Under the hood every test-point
    neighbourhood is fitted independently in a separate worker process so that
    all CPU cores are used instead of running the CEM loop sequentially.

    Parameters
    ----------
    x, y, xt :
        Training inputs/outputs and test inputs (same as JumpGP).
    L : int
        Boundary degree: 1 = linear (JumpGP_LD), 2 = quadratic (JumpGP_QD).
    M : int
        Neighbourhood size (number of nearest training points per test point).
    mode : str
        CEM inference mode: 'CEM', 'VEM', or 'SEM'.
    n_jobs : int or None
        Number of worker processes.  None lets ProcessPoolExecutor pick
        (typically = number of logical CPU cores).
    bVerbose : bool
        Passed through to JumpGP_LD / JumpGP_QD.

    How to run from the repository root with conda env jumpGP:

        conda activate jumpGP
        cd <repo root>
        python JumpGaussianProcess/jumpgp.py
    """

    def __init__(self, x, y, xt, L=1, M=20, mode='CEM', n_jobs=None, bVerbose=False):
        self.x = x
        self.y = y
        self.xt = xt
        self.L = L
        self.M = M
        self.mode = mode
        self.n_jobs = n_jobs
        self.bVerbose = bVerbose
        self.neighborhoods = find_neighborhoods(xt, x, y, M)
        self.jump_results = {}

    def fit(self):
        T = self.xt.shape[0]
        tasks = []
        for t in range(T):
            nbr = self.neighborhoods[t]
            tasks.append((
                t,
                nbr["X_neighbors"],
                nbr["y_neighbors"],
                self.xt[t].reshape(1, -1),
                self.L,
                self.mode,
                self.bVerbose,
            ))

        mu = np.empty(T)
        sig2 = np.empty(T)
        models = [None] * T

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(_cem_worker, task): task[0] for task in tasks}
            for fut in tqdm(as_completed(futures), total=T):
                t, mu_t, sig2_t, model = fut.result()
                mu[t] = mu_t
                sig2[t] = sig2_t
                models[t] = model

        self.jump_results = {"mu": mu, "sig2": sig2, "models": models}
        return self.jump_results

    def metrics(self, yt):
        mu = np.asarray(self.jump_results["mu"]).ravel()
        sig2 = np.asarray(self.jump_results["sig2"]).ravel()
        yt = np.asarray(yt).ravel()
        if not (mu.shape == sig2.shape == yt.shape):
            raise ValueError("Shape mismatch: mu {}, sig2 {}, yt {}".format(
                mu.shape, sig2.shape, yt.shape))
        sigmas = np.sqrt(np.clip(sig2, 0, None))
        return compute_metrics(mu, sigmas, yt)


if __name__ == "__main__":
    x_train = np.random.rand(100, 2)
    y_train = np.random.rand(100)
    x_test = np.random.rand(20, 2)
    y_test = np.random.rand(20)
    JGP = JumpGP(x_train, y_train, x_test, L=2, M=20, mode='VEM', bVerbose=False)
    JGP.fit()
    rmse, mean_crps = JGP.metrics(y_test)
    print("RMSE: {}, Mean CRPS: {}".format(rmse, mean_crps))
    print("Successfully run the example!")
