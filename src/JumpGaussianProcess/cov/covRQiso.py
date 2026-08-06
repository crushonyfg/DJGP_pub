import numpy as np
from .sq_dist import sq_dist

def covRQiso(logtheta, x, z=None):
    """
    Rational Quadratic covariance function with isotropic distance measure. The covariance function is parameterized as:
    
    k(x^p,x^q) = sf2 * (1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha))^(-alpha)
    
    where the P matrix is ell^2 times the unit matrix and the hyperparameters are:
    
    logtheta = [ log(ell)
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
    ell = np.exp(logtheta[0])
    alpha = np.exp(logtheta[1])
    sf2 = np.exp(2 * logtheta[2])
    
    # Compute squared distances
    K = sq_dist(x.T/ell, z.T/ell)
    
    # Compute covariance matrix
    K = sf2 * (1 + K/(2*alpha))**(-alpha)
    
    return K 