import numpy as np

def dlup(W):
    
    A = W.copy()
    n = len(A[:,1])
    Cambios = np.arange(n)
    P = np.eye(n)

    for i in range(n - 1):
        h = np.argmax(np.abs(A[i :, i])) + i
        if h != i:
            Permutacion = [h, i]
            A[[i,h], :] = A[Permutacion, :]
            P[[i,h], :] = P[Permutacion, :]

        A[i + 1:, i] = A[i + 1:, i] / A[i,i]
        A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i, i + 1:])

    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    return L, U, P


def egaussp(A):
    U = A.copy()
    b = A[1, :]
    r = np.zeros(len(b))
    P = np.eye(len(b))
    for i in range(len(b) - 1):
        h = np.argmax(np.abs(A[i :, i])) + i
        if h != i:
            Permutacion = [h, i]
            U[[i,h], :] = U[Permutacion, :]
            P[[i,h], :] = P[Permutacion, :]
        r[i + 1 :] = U[i + 1:, i] / U[i , i]
        U[i + 1 :, i:] -= np.outer(r[i + 1 :], U[i , i:])
    return U, P


J = np.array([[0,4,1], 
              [1,1,3],
              [2,-2,1]], dtype=float)

G = np.array([[1,4,7], 
              [2,5,8], 
              [3,6,10]], dtype=float)

print(egaussp(J))
L, U, Cambio = dlup(J)
print(Cambio.T @ L @ U)
