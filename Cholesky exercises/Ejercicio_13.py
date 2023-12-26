import numpy as np
#Descomentar para correr
from Ejercicio_12 import cholesky


def M(n):
    M = np.zeros((n,n))
    np.fill_diagonal(M, 2)
    h = - np.ones(n-1)
    K = np.diag(h, 1)
    P = np.diag(h, -1)
    M = M + K + P
    return M


#l = cholesky(M(10000))
#print(l)
