import numpy as np
import matplotlib.pyplot as plt

a=[5,7,3,4,5]
#by default there are gonna be 10 bins
his,bins=np.histogram(a)
#print(his)
#print(bins)
plt.hist(his,bins)
plt.show()
his,bins=np.histogram(a,density=True)
plt.hist(his,bins)
plt.show()
his.bins=np.histogram(a,bins=[3,4,5,6,7],density=True)
plt.hist(his,bins)
plt.show()
