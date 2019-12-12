import numpy as np
import string

x = np.zeros(4, dtype='int')

s = string.replace(np.array2string(x), '0', '!')

print s
