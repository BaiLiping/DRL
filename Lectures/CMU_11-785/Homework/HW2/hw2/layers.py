import numpy as np
import math


class Linear():
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        self.batch, __ , self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        raise NotImplemented


    def backward(self, delta):
        
        ## Your codes here
        # self.db = ???
        # self.dW = ???
        # return dx
        raise NotImplemented




class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        ## Your codes here
        raise NotImplemented

    def backward(self, x):
        # Your codes here
        raise NotImplemented




class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x>=0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta
