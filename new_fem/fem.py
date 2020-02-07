import numpy as np

a = np.arange(0, 81)
a = np.reshape(a, [9, 9])
print(a)
a = np.reshape(a, [3, 3, 3, 3])
a = np.swapaxes(a, 1, 2)
a = np.reshape(a, [9, 9])
# print(a)