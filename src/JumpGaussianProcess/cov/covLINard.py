import numpy as np
from .sq_dist import sq_dist

def covLINard(logtheta, x, z=None):
    """
    Linear covariance function with Automatic Relevance Determination (ARD). The covariance function is parameterized as:
    
    k(x^p,x^q) = sf2 * sum_d( x^p_d * x^q_d / ell_d^2 )
    
    where the hyperparameters are:
    
    logtheta = [ log(ell_1)
                 log(ell_2)
                 ...
                 log(ell_D)
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
    ell = np.exp(logtheta[:D])
    sf2 = np.exp(2 * logtheta[-1])
    
    # Scale inputs
    x_scaled = x / ell
    z_scaled = z / ell
    
    # Compute covariance matrix
    K = sf2 * (x_scaled @ z_scaled.T)
    
    return K 