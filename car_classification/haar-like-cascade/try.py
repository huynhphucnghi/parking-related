import numpy as np 

a = np.array([1, 5, 7, 3, 2, 6, 9])
idx = [6, 5, 6]
print(a[idx])
s = np.argsort(a)[::-1]
print(s)