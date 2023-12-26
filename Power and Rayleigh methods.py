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

A = np.array([[-2,1,-2],
              [1,-3,2],
              [2,4,10]], dtype=float)


def potencias(A, b0, tol, maxit):
    c = A @ b0.copy()
    q = [c, A @ c]
    j = 0

    while j < maxit and np.linalg.norm(q[1] - q[0], 2) > tol:
        j += 1
        q[0] = q[1]
        sigma = q[1][np.argmax(np.abs(q[1]))]
        q[1] = A @ q[0] / sigma

    return sigma, q[1], j



'''
v, V, k = potencias(A, [0.5,0.5,0.5], 1e-10, 5000)
print(v, V)
w, W = np.linalg.eig(A)
print(W, '\n', w)
for i in range(3):
    print(V[i] / W[i, 2]) # veo que un autovector sea multiplo del otro
'''

def aut_rayleigh(A, b0, tol, maxit):
    q0 = b0.copy()

    n = A.shape[0]
    error = np.ones(n)
    q0 = q0 / np.linalg.norm(q0,2) #normalizo el vector

    k = 0
    while np.linalg.norm(error,2) > tol and k < maxit:
        p = np.dot(q0, A @ q0)
        q1 = np.linalg.solve(A-(p*np.eye(n)),q0)
        s = np.linalg.norm(q1, 2)
        q1 = q1 / s
        error = q1 - q0
        q0 = q1
        k += 1
    aut = p 
    return aut, q0 ,k 


#v, V, k = aut_rayleigh(d, [0.5,0.5,0.5], 1e-10, 5000)
#print(v, V, k)
v, V, k = potencias(A, [0.5,0.5,0.5], 1e-10, 5000)
print(v,'\n', V, k)
w, W = np.linalg.eig(A)
print(w,'\n', W)
'''
for i in range(3):
    print(V[i] / W[i, 1])'''