# No additional 3rd party external libraries are allowed
import numpy as np

def linear(x, W):
    # TODO
    return np.dot(W,x)       
    # raise NotImplementedError("Linear function not implemented")
    
def linear_grad(x):
    # TODO
    return x
    # raise NotImplementedError("Gradient of Linear function not implemented")

def radial(x, W):
    # TODO
    vec=np.subtract(x,W)
    mag=np.linalg.norm(vec)
    return mag**2
    # raise NotImplementedError("Radial Basis function not implemented")
    
def radial_grad(loss_grad_y, x, W):
    # TODO
    z=np.subtract(x,W)
    mag=-2*z
    return (mag*loss_grad_y).reshape(1,-1)
    # raise NotImplementedError("Gradient of Radial Basis function not implemented")