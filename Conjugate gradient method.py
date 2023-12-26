import numpy as np

def sol_gradcon(A, b, x0, ItMax, tol):
    x = x0
    rs = b - A @ x
    p = rs
    for k in range(ItMax):
        n2rs = np.inner(rs, rs)
        if np.sqrt(np.inner(rs ,rs)) > tol:
            v = np.dot(A, p)

            alfa = n2rs / np.inner(v, p) 
            x = x + alfa * p
            rs = rs - alfa * v
            beta = np.inner(rs, rs) / n2rs
            p = rs + beta * p
        else:
            return x
    return x

A = np.array([[4,1,2],      
              [1,6,-3],
              [2,-3,20]], dtype=float)    
b = np.array([19,3,11], dtype=float)

d = sol_gradcon(A,b, [0,0,0], 500, 1e-8)
print(d)
print(A @ d)
print(np.all(np.linalg.eigvals(A) > 0)) #Veo si todos la matriz es definida positiva
