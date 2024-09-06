import numpy as np

def sol_Richardson(A, b, x0, ItMax, tol):
    x = x0
    r = b - A @ x
    w = np.linalg.eig(A)[0]
    alfa = 2 / (w[-1]+w[0])
    for i in range(ItMax):
        if np.sqrt(np.inner(r,r)) > tol:
            v = A @ r
            x = x + alfa * r
            r = r - alfa * v
        else:
            return x
    return x

A = np.array([[3,1,5],
              [1,4,2],
              [5,2,8]], dtype=float) #Con esta matriz no anda porque no es SDP

A = np.array([[2,1,0],
              [0,3,4],
              [1,4,6]], dtype=float)

b = np.array([1,1,1], dtype=float)
'''
d = sol_Richardson(A,b, [0,0,0], 1000, 1e-8)
print(d)
print(A @ d)
print(np.linalg.eig(A))
'''