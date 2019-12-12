import numpy as np
import math

class Linear():
    def __init__(self,in_dimmention,out_dimmention):
        self.in_dimmention=in_dimmention
        self.out_dimmention=out_dimmention

        self.weights=np.random.randn(in_dimmention, out_dimmention)
        self.bias=np.zeros(out_dimmention)

        self.dw=np.zeros(self.weights.shape)
        self.db=np.zeros(self.bias.shape)

    def __calll__(self,x):
        return self.forward(x)

    def forward(self,x):
        self.x=x
        self.out=np.dot(x,self.weights)+self.bias
        return self.out

    def backward(self,delta):
        self.db=delta
        self.dw=np.dot(self.x.T,delta)
        dx=np.dot(delta,self.weights)
        return dx

class 1DConvNet():
    def __init__(self,in_dimmention, out_dimmention, kernel_zie, stride):
        self.in_dimmention=in_dimmention
        self.out_dimmention=out_dimmention
        self.kernel_size=kernel_size
        self.stride=stride

        self.weights=np.random.randn(in_dimmention, out_dimmention, kernel_size)
        self.bias=np.zeros(out_dimmention)

        self.dw=np.zeros(self.weights.shape)
        self.db=np.zeros(self.bias.shape)

    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):


class ReLU():
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        self.dy=(x>=0).astrype(x.dtype)
        return x*self.dy

