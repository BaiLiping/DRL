import numpy as np

def softmax(x):
    z=np.exp(x)
    return z/np.sum(z)

def cross_entropy_loss(x,y):
    m=y.shape[0]
    p=softmax(x)
    log_likelihood=-np.log(p[range(m),y])
    loss=np.sum(log_likelihood)/m
    return loss

def delta_cross_entropy(x,y):
    m=y.shape[0]
    grad=softmax(x)
    grad[range(m),y]-=1
    return grad


