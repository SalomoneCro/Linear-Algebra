import numpy as np

S = np.array([[1,0,0,0],
              [-1,1,0,0],
              [0,-1,1,0],
              [0,0,-2,2]],dtype=float)
s = np.array([0,0,1,1],dtype=float)

Q = np.array([[1,2,-1,1],
              [0,1,0,-1],
              [0,0,-1,4],
              [0,0,0,1]], dtype=float)
p = np.array([2,-1,0,0], dtype=float)

O = np.array([[9,2,4],
              [0,-6,3],
              [0,0,5]], dtype=float)
f = np.array([18,-2,7], dtype=float)


def soltrinffil(R, c):
    x = c.copy()
    k = np.min(np.nonzero(x))
    for i in range(k, len(x)):
        x[i] = ((x[i] - R[i, :i] @ x[0:i]) / R[i, i])
    return x

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

def soltrsupcol(M,n):
    z = n.copy()
    k = np.max(np.nonzero(n))
    for j in reversed(range(k + 1)):
        z[j] = z[j] / M[j,j]
        z[:j] = z[:j] - z[j] * M[: j, j]
    return z


#print(soltrinfcol(S,s))
#print(soltrinffil(S,s))
#print(soltrsupfil(O,f))
#print(soltrsupcol(O,f))

Q = np.array([[ 1.,  4.,  7.],
              [ 0., -3., -6.],
              [ 0.,  0.,  1.]], dtype=float)

k = np.array([1,0,0], dtype=float)
#print(soltrsupcol(Q,k))

y = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=float)

