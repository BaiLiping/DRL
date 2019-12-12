import numpy as np
import os
from numpy.core.umath_test import inner1d

class BatchNorm(object):
    def __init__(self,fan_in, alpha=0.9):
        self.alpha=alpha
        self.epsilon=1e-9
        self.x=None
        self.norm=None
        self.out=None
        self.nbatch=0

        self.var=np.ones((1,fan_in))
        self.mean=np.zeros((1,fan_in))

        self.gamma=np.ones((1,fan_in))
        self.dgamma=np.zeros((1,fan_in))

        self.beta=np.zeros((1,fan_in))
        self.dbeta=np.zeros((1,fan_in))

        self.running_mean=np.zeros((1,fan_in))
        self.running_var=np.ones((1,fan_in))

    def __call__(self,x,eval=False):
        return self.forward(x,eval)
    def forward(self,x,eval=False):
        self.x=x
        momentum=0.9
        if eval:
            B=np.shape(x)[0]
            mean=self.running_mean/self.nbatch
            var=self.running_var*B/(B-1)/self.nbatch
            self.norm=(x-mean)/np.sqrt(var+self.epsilon)
            self.out=self.gamma*self.norm+self.beta
        else:
            self.nbatch+=1
            self.mean=np.mean(x,0)
            self.var=np.var(x,axis=0)
            self.norm=(x-self.mean)/np.sqrt(self.var+self.epsilon)
            self.out=self.gamma*self.norm+self.beta
            self.running_mean+=self.mean
            self.running_var+=self.var

        return serl.out
    def backward(self,delta):
        N=self.x.shape[0]
        dout=delta*self.gamma
        a=-0.5*(self.var+self.epsilon)**(-1.5)
        dvar=np.sum(dout*(self.x-self.mean)*a,axis=0)
        dx_=1/np.sqrt(self.var+self.epsilon)


