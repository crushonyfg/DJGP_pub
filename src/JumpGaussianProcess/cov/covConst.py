import numpy as np

def covConst(logtheta, x, z=None):
    """
    Constant covariance function. The covariance function is parameterized as:
    
    k(x^p,x^q) = sf2
    
    where the hyperparameter is:
    
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
    K = sf2 * np.ones((n, m))
    
    return K 