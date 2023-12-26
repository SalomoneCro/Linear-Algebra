import numpy as np

def soltrinfcol(A,h):
    b = h.copy()
    x = np.zeros(len(h))
    k = np.min(np.nonzero(b))
    for j in range(k, len(b)):
        x[j] = b[j] / A[j,j]
        b[j:] = b[j:] - x[j] * A[j: ,j] 
    return x


def soltrsupfil(Z,c):
    b = c.copy()
    k = np.max(np.nonzero(b))
    for i in reversed(range(k + 1)):
        b[i] = (b[i] - Z[i, i + 1:] @ b[i + 1:] ) / Z[i,i]
    return b

def dlu(F):
    A = F.copy()
    n = len(A[:,1])
    for i in range(n - 1):
        A[i + 1:, i] = A[i + 1:, i] / A[i,i]
        A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i, i + 1:])
        
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    return L, U

def aut_inversas(A, b0, p, tol, maxit):
    n = A.shape[0]
    B = A - (p * np.eye(n))
    error = np.ones(n)
    k = 0
    L, U = dlu(B) 

    while np.linalg.norm(error, 2) > tol and k < maxit:
        x = soltrinfcol(L, b0)
        q1 = soltrsupfil(U, x) 
        s = q1[np.argmax(np.abs(q1))]
        q1 = q1 / s
        error = q1 - b0
        b0 = q1
        k += 1

    return b0, 1 / s + p, k 

A = np.array([[-2,1,-2],
            [1,-3,2],
            [2,4,10]], dtype=float)

print(aut_inversas(A, [0.5,0.5,0.5], 5, 1e-10, 500))

w, W = np.linalg.eig(A)
print(w, '\n')
print(W)
