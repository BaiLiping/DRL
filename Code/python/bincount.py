import numpy as np
#np.bincount returns the number of occurrences of each value in array of non-negarive ints
#can be thought of as an frequency counter
#the difficulty is how do you know the bin refers to?

#notice that python would always start from 0 for the count, therefore, there would be a leading zero term if you don't specify the starting point
a=[1,2,3,4,5]
bins=np.bincount(a)
print(bins)


