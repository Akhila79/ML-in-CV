# No additional 3rd party external libraries are allowed
import numpy as np
from numpy.linalg import norm 

def l2(y, y_hat):
    # TODO
    prod=np.dot((y-y_hat),(y-y_hat).T)
    return np.sqrt(prod)
    # raise NotImplementedError("l2 loss function not implemented")

def l2_grad(y, y_hat):
    # TODO
    prod=np.dot((y-y_hat),(y-y_hat).T)
    z= np.sqrt(prod)
    return (1/z)*(y-y_hat)
    # raise NotImplementedError("Gradiant of l2 loss function not implemented")

def cross_entropy(A, Y):
    # TODO
    # np.dot()
    raise NotImplementedError("Cross entropy loss function not implemented")
    
def cross_entropy_grad(y, y_hat):
    # TODO
    return ((1-y)/(1-y_hat))-(y/y_hat)
    raise NotImplementedError("Gradiant of Cross entropy loss function not implemented")
    