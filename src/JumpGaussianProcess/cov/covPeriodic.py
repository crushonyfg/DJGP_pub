import numpy as np
from .sq_dist import sq_dist

def covPeriodic(logtheta, x, z=None):
    """
    Periodic covariance function. The covariance function is parameterized as:
    
    k(x^p,x^q) = sf2 * exp(-2*sin^2(pi*(x^p-x^q)/p)/ell^2)
    
    where the hyperparameters are:
    
    logtheta = [ log(ell)
                 log(p)
                 log(sf2) ]
    
    Parameters:
    -----------
    logtheta : array-like
        Hyperparameters of the covariance function
    x : array-like
        First set of input points
    z : array-like, optional
        Second set of input points. If not provided, z = x.
    
    Returns:
    --------
    K : array-like
        Covariance matrix
    """
    if z is None:
        z = x
    
    n, D = x.shape
    m = z.shape[0]
    
    # Extract hyperparameters
    ell = np.exp(logtheta[0])
    p = np.exp(logtheta[1])
    sf2 = np.exp(2 * logtheta[2])
    
    # Compute periodic distances
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i,j] = np.sum(np.sin(np.pi * (x[i] - z[j]) / p)**2)
    
    # Compute covariance matrix
    K = sf2 * np.exp(-2 * K / ell**2)
    
    return K 