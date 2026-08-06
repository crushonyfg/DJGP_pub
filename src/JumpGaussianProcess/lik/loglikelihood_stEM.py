import numpy as np
from .loglikelihood import loglikelihood

def loglikelihood_stEM(logtheta, covfunc1, covfunc2, x, y, r, nargout=1):
    """
    loglikelihood_stEM - Computes the negative log-likelihood and its partial derivatives with
    respect to the hyperparameters for Stochastic EM.

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
    r : numpy array
        Binary matrix indicating which samples to use for each iteration.
    nargout : int, optional
        Number of outputs to return (1 or 2).

    Returns:
    loglike : float
        The negative log-likelihood of the data under the Stochastic EM model.
    dloglike : numpy array, optional
        The partial derivatives of the log-likelihood with respect to hyperparameters.
    """

    L = r.shape[1] if r.ndim > 1 else 1
    loglike = 0
    dloglike = np.zeros_like(logtheta)

    for l in range(L):
        # Get indices for current iteration
        idx = r[:, l] if r.ndim > 1 else r
        
        # Compute log-likelihood for current subset
        if nargout == 2:
            l1, dl1 = loglikelihood(logtheta, covfunc1, covfunc2, x[idx], y[idx], nargout=2)
            loglike += l1
            dloglike += dl1
        else:
            l1 = loglikelihood(logtheta, covfunc1, covfunc2, x[idx], y[idx])
            loglike += l1

    if nargout == 2:
        return loglike, dloglike
    
    return loglike 