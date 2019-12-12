import numpy as np
from math import tanh

class MLP(object):
    def __init__(self,layer,size,activations, iterations, learning_rate,threshold_error, momentum=None,beta=None):
        self.input_size=None
        self.output_size=None
        self.layers=layers
        self.size=size
        self.activations=activations
        self.weights=[]
        self.iterations=iterations
        self.bias=[]
        self.layer_outputs=[]
        self.model={}
        self.learning_rate=learning_rate
        self.threshold_error=threshold_error
        self.momentum=momentum
        self._velocities=[]
        self.beta=beta

    def _initialization_weights(self,momentum=False):
        weights=[]
    def _initialization_bias(self,momentum=False):
        bias=[]
    def fit(self,x,y):
        self.input_size=x[0][0].shape[0]
        self.output_size=y[0][0].shape[0]
        self.weights=self._initialization_weights()
        self.bias=self._initialization_bias()
        self.model['weights1']=self.weights[0]
        self.model['weights2']=self.weights[1]
        self.model['bias1']=self.bias[0]
        self.model['bias2']=self.bias[1]
        self._generate_momentum()
        self.model['momentum_weights1']=self._velocities[0]
        self.model['momentum_weights2']=self._velocities[1]
        self.model['momentum_bias1']=self._velocities[2][0]
        self.model['momentum_bias2']=self._velocities[2][1]

