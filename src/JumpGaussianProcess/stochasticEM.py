# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import numpy as np
# from scipy.optimize import minimize
# from scipy.linalg import cholesky
# from scipy.stats import norm, multivariate_normal
# from calculate_gx import calculate_gx
# from cov.covSum import covSum
# from cov.covSEard import covSEard
# from cov.covNoise import covNoise
# from lik.loglikelihood import loglikelihood
# from lik.loglikelihood_stEM import loglikelihood_stEM

# def stochasticEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose=False):
#     nw = np.linalg.norm(w)
#     w = w / nw
#     nIter = 100

#     phi_xt = np.dot(np.hstack(([1], pxt[0])), w)
#     w = w * np.sign(phi_xt)
#     gx, phi_x = calculate_gx(px, w)
    
#     r = (gx >= 0).ravel()

#     # Initialize parameters like MATLAB code
#     d = x.shape[1]
#     logtheta0 = np.zeros(d+2)
#     logtheta0[d+1] = -1.15
#     bLearnHyp = logtheta is None

#     L = 100  # Number of samples
#     for k in range(10):
#         # M-Step: update ms and logtheta
#         ms = np.sum(r.T @ y) / np.sum(r)
        
#         if bLearnHyp:
#             logtheta = minimize(loglikelihood, logtheta0, args=(cv[0], cv[1], x, y-ms, r), 
#                               method='L-BFGS-B', options={'maxiter': nIter}).x
        
#         sigma = np.exp(logtheta[-1])
        
#         fs = np.zeros((x.shape[0], L))
#         # r_samples = np.zeros((x.shape[0], L))
#         r_samples = [r]
        
#         for l in range(L):
#             # E-step: sample from q(f)
#             # print("r shape:", r.shape, "r dtype:", r.dtype)
#             idx = r_samples[-1]
#             mask = idx.ravel().astype(bool)  
#             Cn = cv[0](cv[1], logtheta, x)
#             _, Cn2 = cv[0](cv[1], logtheta, x, x)
#             Css  = Cn[np.ix_(mask, mask)]
#             Cns = Cn2[:, mask]
            
#             L_chol = cholesky(Css, lower=True)
#             Ly = np.linalg.solve(L_chol, y[idx] - ms)
#             LCns = np.linalg.solve(L_chol, Cns.T)
            
#             mu = ms + LCns.T @ Ly
#             Sigma = Cn2 - LCns.T @ LCns
            
#             # Generate sample from multivariate normal
#             sample = multivariate_normal.rvs(mean=mu.ravel(), cov=Sigma)
#             fs[:, l] = sample
            
#             # E-step: sample from q(Z)
#             like = norm.pdf(y, fs[:, l].reshape(-1, 1), sigma)
#             RR = norm.pdf(2.5 * sigma, loc=0, scale=sigma)
#             prior_z = 1 / (1 + np.exp(-0.05 * nw * gx))
#             pos_z = prior_z * like / (prior_z * like + (1 - prior_z) * RR)
#             pos_z = pos_z.ravel()
#             r_samples.append(np.random.random(size=pos_z.shape) <= pos_z)
        
#         r = np.array(r_samples[1:])
#         print("r shape:", r.shape, "r dtype:", r.dtype)
        
#         # M-Step: update w
#         def wfun(wo):
#             phi_w = np.dot(phi_x, wo)
#             return -np.sum(r.T @ np.log(1 / (1 + np.exp(-phi_w))) + 
#                          (1 - r).T @ np.log(1 - 1 / (1 + np.exp(-phi_w))))

#         w_flattened = w.ravel()
#         from scipy.optimize import LinearConstraint
#         lc = LinearConstraint(-np.array([1, *pxt.flatten()]), ub=0)
#         w_new = minimize(wfun, w_flattened, constraints=lc, options={'disp': False}).x
        
#         conv_crit = np.linalg.norm(w_new / np.linalg.norm(w_new) - w / np.linalg.norm(w))
#         if conv_crit < 1e-3:
#             break
        
#         w = w_new
#         nw = np.linalg.norm(w)
#         w = w / nw

#     # Final prediction using mean of samples
#     r = np.mean(r, axis=1) >= 0.5
#     K = cv[0](cv[1], logtheta, x[r])
#     Ktt, Kt = cv[0](cv[1], logtheta, x[r], xt)
#     L = cholesky(K, lower=True)
#     Ly = np.linalg.solve(L, y[r] - ms)
#     LK = np.linalg.solve(L, Kt)
#     fs = LK.T @ Ly + ms
    
#     model = {
#         'x': x,
#         'y': y,
#         'xt': xt,
#         'px': px,
#         'pxt': pxt,
#         'nll': loglikelihood(logtheta, cv[0], cv[1], x[r], y[r]) / np.sum(r),
#         'r': r,
#         'w': w,
#         'ms': ms,
#         'logtheta': logtheta,
#         'cv': cv,
#         'mu_t': fs,
#         'sig2_t': Ktt - np.sum(LK.T**2, axis=1)
#     }
    
#     return model 
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm, multivariate_normal
from calculate_gx import calculate_gx
from cov.covSum import covSum
from cov.covSEard import covSEard
from cov.covNoise import covNoise
from lik.loglikelihood import loglikelihood
from lik.loglikelihood_stEM import loglikelihood_stEM

def stochasticEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose=False):
    """
    Stochastic EM for JumpGP mixed model, preserving y as shape (n,1).
    Inputs:
      x     : (n, d)
      y     : (n, 1)
      xt    : (m, d)
      px    : extra features for train (n, d)
      pxt   : extra features for test  (m, d)
      w     : initial classifier weights (d+1,)
      logtheta: hyperparams or None
      cv    : (cov_func, cov_args)
      bVerbose: bool
    Returns:
      model dict with keys as in MATLAB version
    """
    n, d = x.shape

    # 1) normalize w
    nw = np.linalg.norm(w)
    w = w / nw

    # 2) enforce test-point sign
    phi_xt = np.hstack(([1.0], pxt[0])) @ w
    w = w * np.sign(phi_xt)

    # 3) initial gx, phi_x
    gx, phi_x = calculate_gx(px, w)    # gx: (n,), phi_x: (n, d+1)
    r_current = (gx >= 0)              # bool mask, shape (n,)

    # 4) EM setup
    logtheta0 = np.zeros(d + 2)
    logtheta0[-1] = -1.15
    learn_hyp = (logtheta is None)
    nIter = 100
    Lsamp = 100

    for k in range(10):
        # --- M-step: update ms and (optionally) hyperparameters ---
        # r_current is (n,), y is (n,1) -> (1,1) then scalar
        ms = (r_current.astype(float) * y).sum() / np.sum(r_current)

        if learn_hyp:
            res = minimize(
                loglikelihood_stEM,
                logtheta0,
                args=(cv[0], cv[1], x, y - ms, r_current),
                method='L-BFGS-B',
                options={'maxiter': nIter}
            )
            logtheta = res.x

        sigma = np.exp(logtheta[-1])

        # precompute full covariances
        Cn_full  = cv[0](cv[1], logtheta, x)       # (n,n)
        _, Cn2_full = cv[0](cv[1], logtheta, x, x)    # (n,n)

        # storage for E-step
        fs        = np.zeros((n, Lsamp))
        r_samples = np.zeros((n, Lsamp), dtype=bool)

        # --- E-step: sample f and Z Lsamp times ---
        for l in range(Lsamp):
            mask = r_current.ravel()  # bool mask length n

            # select submatrices
            Css = Cn_full[np.ix_(mask, mask)]    # (k,k)
            Cns = Cn2_full[:, mask]             # (n,k)

            # Cholesky on Css
            L_ch = cholesky(Css, lower=True)    # (k,k)
            # y[mask] - ms has shape (k,1); flatten for solve
            Ly = solve_triangular(
                L_ch,
                (y[mask, 0] - ms),
                lower=True
            )                                    # (k,)

            LCns = solve_triangular(
                L_ch,
                Cns.T,
                lower=True
            )                                    # (k,n)

            # posterior mean & cov
            mu    = ms + (LCns.T @ Ly)          # (n,)
            Sigma = Cn2_full - (LCns.T @ LCns)  # (n,n)

            # sample latent function
            sample_f = multivariate_normal.rvs(mean=mu, cov=Sigma)
            fs[:, l] = sample_f

            # compute posterior prob for z
            like  = norm.pdf(y[:, 0], sample_f, sigma)[:, None]  # (n,1)
            RR    = norm.pdf(2.5 * sigma, 0, sigma)
            # prior = 1.0 / (1 + np.exp(-0.05 * nw * gx))[:, None]  # (n,1)
            prior = 1.0 / (1 + np.exp(-0.05 * nw * gx))

            pos_z = (prior * like) / (prior * like + (1 - prior) * RR)
            pos_z = pos_z[:, 0]  # flatten to (n,)

            # sample new z
            r_new = (np.random.rand(n) <= pos_z.ravel())
            r_samples[:, l] = r_new
            r_current = r_new

        # now r_samples is (n, Lsamp)
        r_mat = r_samples

        # --- M-step: update w via constrained optimization ---
        def wfun(wo):
            phi_w = phi_x @ wo               # (n,)
            p     = 1.0 / (1.0 + np.exp(-phi_w))  # (n,)
            # broadcast p to (n,Lsamp)
            P = p[:, None]
            return -np.sum(r_mat * np.log(P) + (1 - r_mat) * np.log(1 - P))

        # constraint: [1, pxt] @ w <= 0
        A = -np.hstack(([1.0], pxt.flatten()))
        lc = LinearConstraint(A, ub=0.0)

        w0 = w.flatten()
        res_w = minimize(wfun, w0, constraints=[lc], options={'disp': False})
        w_new = res_w.x
        # check convergence
        conv = np.linalg.norm(w_new/np.linalg.norm(w_new) - w/nw)
        w = w_new / np.linalg.norm(w_new)
        nw = np.linalg.norm(w)
        if conv < 1e-3:
            break

    # --- Final prediction on xt ---
    r_final = (r_mat.mean(axis=1) >= 0.5)

    K   = cv[0](cv[1], logtheta, x[r_final, :])
    Ktt, Kt = cv[0](cv[1], logtheta, x[r_final, :], xt)
    Lk  = cholesky(K, lower=True)
    Ly  = solve_triangular(Lk, (y[r_final, 0] - ms), lower=True)
    LK  = solve_triangular(Lk, Kt, lower=True)
    mu_t   = LK.T @ Ly + ms
    sig2_t = np.diag(Ktt) - np.sum(LK.T**2, axis=1)

    model = {
        'x': x,
        'y': y,
        'xt': xt,
        'px': px,
        'pxt': pxt,
        'nll': loglikelihood(logtheta, cv[0], cv[1], x[r_final, :], y[r_final, :]) / np.sum(r_final),
        'r': r_final,
        'w': w,
        'ms': ms,
        'logtheta': logtheta,
        'cv': cv,
        'mu_t': mu_t,
        'sig2_t': sig2_t
    }
    return model
