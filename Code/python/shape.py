import numpy as np

x=np.array([1,2,3,4])
print(x.shape)
x=x.reshape(4,1)
print(x)
print(x.shape)
x=x.reshape(1,4)
print(x)
print(x.shape)
x=x.reshape(-1)
print(x)
print(x.shape)

