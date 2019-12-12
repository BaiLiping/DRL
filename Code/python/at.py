import numpy as np
a=np.array([1,2,3,4,5])
a=a.reshape(1,5)
b=np.array([1,2,3,4,5])
b=b.reshape(5,1)
print(a@b)
