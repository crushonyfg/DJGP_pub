import numpy as np
from .sq_dist import sq_dist

def covRQard(logtheta, x, z=None):
    """
    Rational Quadratic covariance function with Automatic Relevance Determination (ARD). The covariance function is parameterized as:
    
    k(x^p,x^q) = sf2 * (1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha))^(-alpha)
    
    where the P matrix is diag(ell_1^2, ell_2^2, ...) and the hyperparameters are:
    
    logtheta = [ log(ell_1)
                 log(ell_2)
                 ...
                 log(ell_D)
                 log(alpha)
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
    alpha = np.exp(logtheta[D])
    sf2 = np.exp(2 * logtheta[D+1])
    
    # Scale inputs
    x_scaled = x / ell
    z_scaled = z / ell
    
    # Compute squared distances
    K = sq_dist(x_scaled.T, z_scaled.T)
    
    # Compute covariance matrix
    K = sf2 * (1 + K/(2*alpha))**(-alpha)
    
    return K 