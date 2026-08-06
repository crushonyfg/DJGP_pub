import numpy as np
from .sq_dist import sq_dist

def covLINone(logtheta, x, z=None):
    """
    Linear covariance function with a single length scale. The covariance function is parameterized as:
    
    k(x^p,x^q) = sf2 * (x^p * x^q)
    
    where the hyperparameters are:
    
    logtheta = [ log(sf2) ]
    
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
    sf2 = np.exp(2 * logtheta[0])
    
    # Compute covariance matrix
    K = sf2 * (x @ z.T)
    
    return K 