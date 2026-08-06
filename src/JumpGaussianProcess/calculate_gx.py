# % ***************************************************************************************
# % calculate_gx - This function evaluates the boundary function with basis function values px
# % and basis weights w
# % 
# % Inputs:
# %       px - basis function values of a boundary function evaluated at
# %       training inputs, excluding a constant basis
# %       w - associated parameters of the boundary function
# % Outputs:
# %       gx - boundary function values
# %       phi_x - corresponding basis matrix 
# % 
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************


import numpy as np

def calculate_gx(px, w):
    N = px.shape[0]  # number of samples
    # build basis matrix phi_x, the first column is 1, representing the bias term
    phi_x = np.column_stack((np.ones(N), px))
    
    # compute boundary function values gx
    gx = phi_x @ w
    
    return gx, phi_x #gx shape (N,), phi_x shape (N,d+1)

# example usage
if __name__ == "__main__":
    px = np.random.rand(100, 2)  # 100 samples, 2 features
    w = np.random.rand(3)        # weight vector including the bias term
    gx, phi_x = calculate_gx(px, w)
    print("gx:", gx)
