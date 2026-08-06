import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky
from scipy.stats import norm

from calculate_gx import calculate_gx
from cov.covSum import covSum
from cov.covSEard import covSEard
from cov.covNoise import covNoise
from lik.loglikelihood import loglikelihood
from lik.loglikelihood_jgp import loglikelihood_jgp

def variationalEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose=False):
    # Turn off warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # print("w.shape", w.shape)
    nw = np.linalg.norm(w)
    w = w / nw
    n = x.shape[0]
    O = np.ones((n, 1))
    
    nIter = 100
    
    phi_xt = np.dot(np.hstack(([1], pxt[0])), w.ravel())
    w = w * np.sign(phi_xt)
    gx, phi_x = calculate_gx(px, w)
    
    # initialize q_Z
    gx_flat = gx.flatten()  # Flatten gx to 1D array
    D_g = np.diag(1 + np.exp(-0.05 * nw * gx_flat))
    D_gi = np.diag(1 / (1 + np.exp(-0.05 * nw * gx_flat)))
    
    # initialize logtheta
    r = gx >= 0
    r1 = r.flatten()
    ms = np.mean(y[r1]).item()
    
    d = x.shape[1]
    logtheta0 = np.zeros(d+2)
    logtheta0[d+1] = -1.15
    # bLearnHyp = logtheta is None
    bLearnHyp = True
    
    if bLearnHyp:
        try:
            logtheta = minimize(loglikelihood, logtheta0, args=(cv[0], cv[1], x[r1,:], y[r1]-ms), 
                            method='L-BFGS-B', options={'maxiter': nIter}).x
        except:
            logtheta = logtheta0
    
    sigma = np.exp(logtheta[-1])
    
    # stricter sigma range
    # sigma = np.clip(sigma, 1e-3, 1e3)
    # print("sigma", sigma)
    
    for k in range(10):
        # E-step: update q(fs)
        K = cv[0](cv[1], logtheta, x)
        tmp = sigma**2 * D_g + K
        # add regularization term to prevent singularity
        eps = 1e-6
        tmp += eps * np.eye(tmp.shape[0])
        # check and handle infinite/NaN values
        # tmp = np.nan_to_num(tmp, nan=1e-6, posinf=1e6, neginf=-1e6)
        try:
            L = cholesky(tmp, lower=True)
        except:
            # if Cholesky fails, use SVD
            U, s, Vt = np.linalg.svd(tmp)
            s = np.maximum(s, 1e-12)  # ensure singular values are not zero
            L = U @ np.diag(np.sqrt(s))
        LK = np.linalg.solve(L, K)
        Sigma = K - LK.T @ LK
        KO = np.linalg.solve(K, O)
        
        
        mu = Sigma @ (ms * KO + D_gi @ y.reshape(-1, 1) / sigma**2)
        # print("mu", mu.shape, y.shape)
        
        # E-step: update q(Z) or gamma = q(Z=1)
        exp_term = np.exp(-0.5 * np.diag(Sigma) / sigma**2).reshape(-1, 1)
        like = norm.pdf(y, loc=mu, scale=sigma) * exp_term
        RR = norm.pdf(2.5 * sigma, loc=0, scale=sigma)
        prior_z = (1 / (1 + np.exp(-0.05 * nw * gx))).reshape(-1, 1)
        pos_z = prior_z * like
        gamma = pos_z / (pos_z + (1 - prior_z) * RR)
        # print("gamma", gamma.shape)
        # M-step: update sigma
        gamma_flat = np.clip(gamma.flatten(), 1e-12, 1-1e-12)
        D_g = np.diag(1 / gamma_flat)
        D_gi = np.diag(gamma_flat)
        
        # Calculate sigma with correct matrix dimensions
        diff = y - mu
        outer_product = diff @ diff.T
        # Ensure D_gi is (n,n) matrix
        # D_gi = np.diag(gamma)  # Reset D_gi to correct size

        
        sigma = np.sqrt(np.trace(D_gi @ (Sigma + outer_product)) / n)
        
        # M-step: update w
        def wfun(wo):
            phi_w = np.dot(phi_x, wo)
            # print("phi_w.shape, wo.shape", phi_x.shape, wo.shape)
            return (-(gamma.T @ np.log(1 / (1 + np.exp(-phi_w))) + 
                    (1 - gamma).T @ np.log(1 - 1 / (1 + np.exp(-phi_w))))).item()
        
        w_flattened = w.ravel()
        from scipy.optimize import LinearConstraint
        lc = LinearConstraint(-np.array([1, *pxt.flatten()]), ub=0)
        w_new = minimize(wfun, w_flattened, constraints=lc, options={'disp': False}).x
        
        conv_crit = np.linalg.norm(w_new / np.linalg.norm(w_new) - w / np.linalg.norm(w))
        if conv_crit < 5e-4:
            break
            
        # print("w_new.shape, w.shape", w_new.shape, w.shape)
        w = w_new
        nw = np.linalg.norm(w)
        w = w / nw
        gx, phi_x = calculate_gx(px, w)
        
        # M-step: update logtheta
        ms = ((KO.T @ mu) / (KO.T @ O)).squeeze()
        if bLearnHyp:
            logtheta = minimize(loglikelihood_jgp, logtheta, args=(cv[0], cv[1], x, mu-ms, Sigma),
                              method='L-BFGS-B', options={'maxiter': nIter}).x
    
    K = cv[0](cv[1], logtheta, x)
    Ktt, Kt = cv[0](cv[1], logtheta, x, xt)
    L = cholesky(K, lower=True)
    Ly = np.linalg.solve(L, mu-ms)
    LK = np.linalg.solve(L, Kt)
    LLK = np.linalg.solve(L.T, LK)
    fs = LK.T @ Ly + ms
    
    model = {
        'x': x,
        'y': y,
        'xt': xt,
        'px': px,
        'sigma': sigma,
        'pxt': pxt,
        'nll': (wfun(w) + loglikelihood_jgp(logtheta, cv[0], cv[1], x, mu-ms, Sigma) + 
                0.5 * np.sum(((y - mu)**2 * gamma) / sigma**2) + 
                0.5 * np.sum(np.log(sigma**2 / gamma)) + 
                0.5 * np.trace(D_gi @ Sigma / sigma**2)),
        'gamma': gamma,
        'r': gamma >= 0.5,
        'w': w,
        'nw': nw,
        'ms': ms,
        'logtheta': logtheta,
        'cv': cv,
        'mu_t': fs,
        'sig2_t': Ktt - LK.T @ LK + LLK.T @ Sigma @ LLK
    }
    
    return model 