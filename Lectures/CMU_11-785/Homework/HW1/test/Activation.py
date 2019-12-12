import numpy as np
import os
from numpy.core.umath_tests import inner1d

class Activation(object):
    def __init__(self):
        self.state=None
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        raise NotImplemented
    def derivative(self):
        raise NotImplemented

class Identity(Activation):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        self.state=x
        return self.state
    def derivative(self):
        return 1.0
class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid,self).__init__()
    def forward(self.x):
        self.state=1.0/(1.0+np.exp(-x))
        return self.state
    def derivative(self):
        last_result=self.state
        return last_result*(1.0-last_result)
class Tanh(Activation):
    def __init__(self):
        super(Tanh,self).__init__()
    def forward(self,x):
        m=(np.exp(x))**2
        m=(m-1.0)/(m+1.0)
        self.state=m
        return self.state
    def derivative(self):
        return 1.0-(self.state)**2
class ReLU(Activation):
    def __init__(self):
        super(ReLU,self).__init__()
    def forward(self,x):
        x[x<0]=0
        self.state=x
        return x
    def derivative(self):
        x=self.state
        x[x>0]=1.0
        return x




