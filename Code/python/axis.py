import numpy as np

a=np.array([1,2,3])
print(a)
a=a[np.newaxis,:]
print(a)
a=a[:,np.newaxis]
print(a)

a=np.random.randn(3,4)
print(a)
print(np.sum(a,axis=0))
print(np.sum(a,axis=1))

print(a.shape[0])
print(a.shape[1])

