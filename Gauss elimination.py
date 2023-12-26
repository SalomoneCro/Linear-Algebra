import numpy as np

def egauss(A,b):
    U = A.copy()
    y = b.copy()
    r = np.zeros(len(b))
    for i in range(len(b) - 1):
        r[i + 1:] = U[i + 1:, i] / U[i , i]
        U[i + 1 :, i:] -= np.outer(r[i + 1 :], U[i , i:])
        y[i + 1 :] = y[i + 1 :] - r[i + 1 :] * y[i]

    return U, y

G = np.array([[1,4,7], 
              [2,5,8], 
              [3,6,10]], dtype=float)

J = np.array([[2,-2,1], 
              [1,1,3],
              [0,4,1]], dtype=float)

print(egauss(J, np.array([1,2,3])))

