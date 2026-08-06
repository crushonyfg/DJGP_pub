# % ***************************************************************************************
# %
# % local_linearfit - The function implements a locally linear fit using local data 
# %                   (x0, y0) around test location xt
# % 
# %
# % Inputs:
# %       x - local training inputs (N x d)
# %       y - local training responses (N x 1)
# %       xt - single test location (1 x d)
# % Outputs:
# %       beta - fitted parameters of a linear model (1+d dimensions)
# %       X - local linear basis matrix (N x (d+1) matrix)
# %
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************


import numpy as np

def local_linearfit(x0, y0, xt):
    """
    local_linearfit - The function implements a locally linear fit using local data 
                      (x0, y0) around test location xt
    
    Parameters:
    -----------
    x0 : array-like
        Local training inputs (N x d)
    y0 : array-like
        Local training responses (N x 1)
    xt : array-like
        Single test location (1 x d)
    
    Returns:
    --------
    beta : array-like
        Fitted parameters of a linear model (1+d dimensions)
    X : array-like
        Local linear basis matrix (N x (d+1) matrix)
    """
    N = x0.shape[0]
    
    # Ensure xt is 2D array with shape (1, d)
    if xt.ndim == 1:
        xt = xt.reshape(1, -1)
    
    # Calculate distance between x0 and xt
    d = x0 - xt  # Broadcasting will work now
    d2 = np.sum(d ** 2, axis=1)
    
    # Calculate bandwidth h
    h = np.max(np.sqrt(d2))
    
    # Kernel calculation
    Kh = np.exp(-0.5 * d2 / (h ** 2)) / (2 * np.pi * (h ** 2))
    
    # Create local linear basis matrix X
    X = np.hstack((np.ones((N, 1)), x0))
    
    # Compute X'WX and X'Wy
    XWX = X.T @ np.diag(Kh) @ X
    XWy = X.T @ np.diag(Kh) @ y0
    
    # Solve for beta with regularization
    beta = np.linalg.solve(XWX + 1e-6 * np.eye(XWX.shape[0]), XWy)
    
    return beta, X

# test code example
if __name__ == "__main__":
    x0 = np.random.rand(100, 2)  # 100 samples, 2 features
    y0 = np.random.rand(100)  # 100 response values
    xt = np.random.rand(2)  # test point (2 features)

    beta, X = local_linearfit(x0, y0, xt)
    print("beta:", beta)
