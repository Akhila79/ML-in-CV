# No additional 3rd party external libraries are allowed
import numpy as np


def SGD(model,  train_X, train_y, lr=0.01, R=5):
    #Updated_Weights = None
    #TODO
    # #initialize W matrix

    # W_i=np.zeros(np.shape(model.W))
    W_i=[]
    # print(len(model.W[0]))
    for w in model.W:
        
        # print(type(w))
        W_i.append(np.zeros(w.shape))

    layer_i=int(np.random.randint(0,len(model.W),1))

    for i in range(R):
        row= int(np.random.randint(0,np.shape(model.W[layer_i])[0],1))
        col=int(np.random.randint(0,np.shape(model.W[layer_i])[1],1))
        W_i[layer_i][row][col]=model.W[layer_i][row][col]


        l_grad=model.emp_loss_grad(train_X,train_y,W_i,None)
        # print(l_grad[0].shape)
        # print(l_grad[0])
        model.W=[(i-lr*j) for i,j in zip(model.W,l_grad)]
    return model.W