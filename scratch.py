import numpy as np

x = np.arange(3)[:, np.newaxis]
A = np.arange(6).reshape(3,2)

print(x)
print()
print(A)
print()
print(x*A)

print(A*x)