'''
in this notebook we will train a neural network to do a simple task. This will be a classification taskL as explained in the first week of lecture, classificiation basically means to find a decision boundary over a space of real numbers. For represtation purpose we will work with a 2D exmaple. the decision boundary will be a circle. More precisely, it wil lbe te unity circle in the plan

'''

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def sample_points(n):
    radius=np.random.uniform(low=0,high=2,size=n).reshape(-1,1)
    angle=np.random.uniform(low=0,high=2*np.pi, size=n).reshape(-1,1)
    x1=radius*np.cos(angle)
    x2=radius*np.sin(angle)
    y=(radius<1).astype(int).reshape(-1)
    x=np.concatenate([x1,x2],axis=1)
    return x,y


