import numpy as np

def dlu(F):
    A = F.copy()
    n = len(A[:,1])
    for i in range(n - 1):
        A[i + 1:, i] = A[i + 1:, i] / A[i,i]
        A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i, i + 1:])
        
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    return L, U


G = np.array([[1,4,7], 
              [2,5,8], 
              [3,6,10]], dtype=float)

L, U = dlu(G)
print(L,"\n", U)

