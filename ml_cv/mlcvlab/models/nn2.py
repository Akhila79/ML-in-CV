from turtle import shape
import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer


class NN2():
    def __init__(self):
        self.layers = [
            Layer(None, relu), 
            Layer(None, sigmoid)]
        self.W=[np.ones(shape=[500,784]),np.ones(shape=[1,500])]
        # self.W

    def nn2(self, x):
        # TODO
        # W1=[np.ones(shape=[500,784])]
        # W2=[np.ones(shape=[1,500])]

        z1=linear(x,self.W[1])
        z1_bar=relu(z1)
        z2=linear(z1_bar,self.W[2])
        y_hat=sigmoid(z2)
        return y_hat
        # raise NotImplementedError("NN2 model not implemented")

    def grad(self, x, y, W):
        # TODO
        W1=W[0]
        W2=W[1]
        print(W1.shape)
        z1=linear(x.T,W1)
        z1_bar=relu(z1)
        z2=linear(z1_bar,W2)
        y_hat=sigmoid(z2)  

        l2_loss_grad=l2_grad(y,y_hat)
        d_y_hat_z2=sigmoid_grad(sigmoid(z2))
        d_w2=np.dot(l2_loss_grad*d_y_hat_z2,z1_bar.T)
        
        # dz=np.dot(W2.T,d_w2)*relu_grad(z1_bar)
        d_z1_bar_z1=relu_grad(z1_bar)
        d_w1=np.dot(d_w2*d_z1_bar_z1,x.T)

        # d_w1=np.dot(dz,z1_bar)
        return [d_w1,d_w2]

        # self.localgrads.append(x)
        # z1=linear(x,W[1:])
        # z1_bar=sigmoid(z1)
        # d_z1_bar_z1=l2_grad()
        # z2=linear(z1_bar,W[2:])
        # y_hat=sigmoid(z2)
        # raise NotImplementedError("NN2 gradient (backpropagation) not implemented")        

    def emp_loss_grad(self, train_X, train_y, W, layer):
        # TODO
        # d_L=0
        return self.grad(train_X,train_y, W)

        # return emp_loss_grad_
        # raise NotImplementedError("NN2 Emperical Loss grad not implemented")