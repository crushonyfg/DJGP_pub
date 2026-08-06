import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.linalg import cholesky

def loglikelihood_jgp(logtheta, covfunc1, covfunc2, x, y, Sigma, nargout=1):
    """
    loglikelihood_jgp - Computes the negative log-likelihood and its partial derivatives with
    respect to the hyperparameters for Jump GP.

    Parameters:
    logtheta : numpy array
        A vector of log hyperparameters.
    covfunc1 : function
        Main covariance function (e.g., covSum).
    covfunc2 : list
        List of covariance functions to be combined.
    x : numpy array
        Training inputs, an n by D matrix.
    y : numpy array
        Target outputs, a vector of size n.
    Sigma : numpy array
        Covariance matrix for the variational approximation.
    nargout : int, optional
        Number of outputs to return (1 or 2).

    Returns:
    loglike : float
        The negative log-likelihood of the data under the Jump GP model.
    dloglike : numpy array, optional
        The partial derivatives of the log-likelihood with respect to hyperparameters.
    """

    # Compute the covariance matrix K
    K = covfunc1(covfunc2, logtheta, x)

    # Add small jitter for numerical stability
    K += 1e-6 * np.eye(K.shape[0])

    # Cholesky decomposition of the covariance matrix K
    # L = np.linalg.cholesky(K)
    try:
        L = cholesky(K)
    except:
        # if Cholesky fails, use SVD
        U, s, Vt = np.linalg.svd(K)
        s = np.maximum(s, 1e-12)  # ensure singular values are not zero
        L = U @ np.diag(np.sqrt(s))
    
    # Solve for alpha and beta
    alpha = cho_solve((L, True), y)
    beta = cho_solve((L, True), Sigma)

    # Compute the negative log-likelihood
    loglike = (0.5 * y.T @ alpha + 
              np.sum(np.log(np.diag(L))) + 
              0.5 * len(y) * np.log(2 * np.pi) + 
              0.5 * np.trace(beta)).item()

    if nargout == 2:  # If partial derivatives are requested
        dloglike = np.zeros_like(logtheta)
        
        # Precompute for convenience
        W = (cho_solve((L, True), np.eye(len(x))) - 
             np.outer(alpha, alpha) - 
             0.5 * cho_solve((L, True), beta.T))

        # Compute partial derivatives
        for i in range(len(logtheta)):
            dK_dtheta = covfunc1(covfunc2, logtheta, x, i)  # Derivative of covariance matrix w.r.t. hyperparameter i
            dloglike[i] = 0.5 * np.sum(W * dK_dtheta)

        return loglike, dloglike
    
    return loglike 