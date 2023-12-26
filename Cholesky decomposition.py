import numpy as np


def cholesky(A):
    n = len(A[0 , :])
    G = np.copy(A)
    for i in range(n):
        G[i,i] = np.sqrt(G[i,i])
        G[i , i+1:] = G[i , i+1:] / G[i,i]
        G[i+1: , i+1:] = G[i+1: , i+1:] - np.outer(G[i , i+1:],G[i , i+1:])
    G = np.triu(G, 0)
    return G


A = np.array([[4,2,6],
              [2,2,5],
              [6,5,29]], dtype=float)

B = np.array([[4,0], [0,9]], dtype=float)

#print(cholesky(A))
#print(cholesky(B))
