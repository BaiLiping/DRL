import numpy as np
import os

class Criterion(object):
    def __init__(self):
        self.logits=None
        self.labels=None
        self.loss=None

    def __call__(self,x,y):
        return self.forward(x,y)

    def forward(self,x,y):
        raise NotImplemented
    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy,self).__init__()
        self.softmax=None
    def forward(self,x,y):
        self.logits=x
        self.labels=y
        x-=np.max(x)
        self.softmax=np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
        self.loss=-(np.sum(self.labels*np.log(self.sm),axis=1))
        return self.loss
    def derivative(self):
        derivative=self.sm-self.labels
        return derivative



