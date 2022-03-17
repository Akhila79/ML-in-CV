# No additional 3rd party external libraries are allowed
import numpy as np

def relu(x):
    # TODO
    return x*(x>0)
    # raise NotImplementedError("ReLU function not implemented")

def relu_grad(z):
     # TODO

    return 1*(z>0)
    # raise NotImplementedError("Gradient of ReLU function not implemented")

def sigmoid(x):
    # TODO
    exponen=np.exp(-x)
    func=1/(1+exponen)
    return func
    # raise NotImplementedError("Sigmoid function not implemented")
    
def sigmoid_grad(z):
    # TODO
    return z*(1-z)
    # raise NotImplementedError("Gradient of Sigmoid function not implemented")

def softmax(x):
    # TODO
    exponen=np.exp(x)
    summation= np.sum(exponen,axis=1).reshape(-1,1)
    y=exponen/summation
    return y
    # for i in range(len(x)):
    #     y.append(exponen[i]*(1/summation))

    # raise NotImplementedError("Softmax function not implemented")
    
def softmax_grad(z):
    # TODO
    return np.diag(z.flatten())-np.outer(z,z)
    # raise NotImplementedError("Gradient of Softmax function not implemented")

def tanh(x):
    # TODO
    p_exp=np.exp(x)
    n_exp=np.exp(-x)
    func=(p_exp-n_exp)/(p_exp+n_exp)
    return func
    # raise NotImplementedError("Tanh function not implemented")

def tanh_grad(z):
    # TODO
    p_exp=np.exp(z)
    n_exp=np.exp(-z)
    func=(p_exp-n_exp)/(p_exp+n_exp)
    return 1-(func**2)
    raise NotImplementedError("Gradient of Tanh function not implemented")
