import numpy as np
import os
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
    def forward(self,x):
        self.state=x
        activated_state=1/(1+np.exp(-self.state))
        return activated_state
    def derivative(self):
        derivative=np.exp(-self.state)*np.power((1+np.exp(-self.state)),-2)       
        return derivative
class Tanh(Activation):
    def __init(self):
        super(Tanh,self).__init__()
    def forward(self,x):
        self.state=x
        activated_state=np.tanh(x)
        return activated_state
    def derivative(self):
        derivative=1-np.power((np.tanh(self.state)),2)
        return derivative
class ReLU(Activation):
    def __init__(self):
        super(ReLU,self).__init__()
    def forward(self,x):
        self.state=x
        activated_state=max(x,0)
        return activated_state
    def derivative(self):
        if self.state<0:
            return 0
        else:
            return 1

