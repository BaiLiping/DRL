import numpy as np
import os
from numpy.core.umath_test import inner1d

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
    def __init(self):
        super(SoftmaxCrossEntropy,self).__init__()
        self.softmax=None
    def forward(self,x,y):
        self.logits=x
        self.labels=y
        a=np.max(x,1).reshape(-1,1)
        log_sum_exp=a+np.log(np.sum(np.exp(x-a),1).reshape(-1,1))
        log_yhat=x-log_sum_exp
        self.state(y*log_yhat,1)
    def derivative(self):
        yhat=np.exp(self.state)
        return yhat-self.labels
