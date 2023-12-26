import numpy as np
#v = Ar
#alfa = rr / vr
#x = x + alfa r
#r = r - alfa v

def sol_gradopt(A, b, x0, ItMax, tol):
    x = x0
    r = b - A @ x
    for i in range(ItMax):
        if np.inner(r,r) > tol:
            v = A @ r
            alfa = np.inner(r, r) / np.inner(v, r)
            x = x + alfa * r
            r = r - alfa * v
        else:
            return x
    return x

A = np.array([[3,1,4],
              [1,4,2],
              [4,2,8]], dtype=float)
b = np.array([1,1,1], dtype=float)

'''d = sol_gradopt(A,b, [0,0,0], 500, 1e-8)
print(d)
print(A @ d)'''
