from turtle import shape
import numpy as np
from pyparsing import line
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import sigmoid, sigmoid_grad
from .base import Layer


class NN1():
    def __init__(self):
        self.layers = [
            Layer(linear, sigmoid)]
        
        self.W = [np.random.randn(1,784)*np.sqrt(1/784)] 
        # self.localgrads=[]
        # self.layers=[]

    def nn1(self, x):
        #TODO
        # self.layers.append(x) #layers?? functionality??
        W=self.W # initialize to random weights
        # print(x.shape)
        z=linear(x,W[0])
        y_hat=sigmoid(z)    
        return y_hat   

    def grad(self, x, y, W):
        # TODO
        d_L_W=[]
        for w in self.W:
            d_L_W.append(np.zeros(w.shape))

        y_hat=sigmoid(linear(x,W[0]))
        l2_loss_grad=l2_grad(y,y_hat)
        d_y_hat_z=sigmoid_grad(sigmoid(linear(x,W[0])))
        # self.localgrads.append(x)
        # self.localgrads.append(l2_loss_grad)
        # self.localgrads.append(d_y_hat_z)
        # d_L_W[0]=np.prod(self.localgrads)
        d_L_W[0]=np.dot(l2_loss_grad*d_y_hat_z,x.T)
        # print('inside grad', d_L_W[0].shape)
        return d_L_W

    def emp_loss_grad(self, train_X, train_y, W, layer):
        return self.grad(train_X,train_y, W)

        # TODO
        # return emp_loss_grad_
        # raise NotImplementedError("NN1 Emperical Loss grad not implemented")
       