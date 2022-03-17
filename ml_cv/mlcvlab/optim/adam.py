# No additional 3rd party external libraries are allowed
import numpy as np

def Adam(model,  train_X, train_y, b1=0.9, b2=0.999, epsilon=1e-8, alpha=0.1):
        
    #TODO
    mr=[np.zeros(w.shape) for w in model.W]
    sr=[np.zeros(w.shape) for w in model.W]
    
    l_grad=model.emp_loss_grad(train_X,train_y,model.W,None)
    for ind,w in enumerate(model.W):
        mr[ind]=b1*mr[ind]+(1-b1)*l_grad[ind]
        sr[ind]=b2*sr[ind]+(1-b2)*l_grad[ind]**2
        model.W[ind]=model.W[ind]-alpha*mr[ind]/(np.sqrt(sr[ind])+epsilon)
    return model.W