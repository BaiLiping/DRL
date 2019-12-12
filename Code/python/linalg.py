import numpy as np
#np.linalg.norm returns one of seven different norms of a vector

a=np.arange(9)-4
b=a.reshape((3,3))

L1=np.linalg.norm(a,1)
L2=np.linalg.norm(a,2)

print(L1)
print(L2)

