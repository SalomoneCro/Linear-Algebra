import numpy as np
from Ejercicio_10 import sol_gradcon
from Ejercicio_9 import sol_Richardson
from Ejercicio_8 import sol_gradopt

#a
A = np.array([[9,-2,0],
              [-2,4,-1],
              [0,-1,1]], dtype=float)

b = np.array([5,1,-5/6], dtype=float)

#print(sol_gradopt(A, b, [0,0,0], 500, 1e-8))
#print(sol_Richardson(A,b, [0,0,0], 500, 1e-8))
#print(sol_gradcon(A,b, [0,0,0], 500, 1e-8))

#b
A = np.array([[4,-1,0],
              [-1,4,-1],
              [0,-1,4]], dtype=float)

b = np.array([2,6,2], dtype=float)

#print(sol_gradopt(A, b, [0,0,0], 500, 1e-8))
#print(sol_Richardson(A,b, [0,0,0], 500, 1e-8))
#print(sol_gradcon(A,b, [0,0,0], 500, 1e-8))

#c

def Matriz11c(n):
    A = np.zeros((n,n))
    np.fill_diagonal(A, 4)
    h = - np.ones(n-1)
    K = np.diag(h, 1)
    P = np.diag(h, -1)
    return A + K + P

ns = [5,10,15,20]
'''
for i in ns:
    A = Matriz11c(i)
    b = np.ones(i)
    print(f'---------------- n = {i}')
    print(sol_gradopt(A, b, np.zeros(i), 500, 1e-8),'\n')
    print(sol_Richardson(A,b, np.zeros(i), 500, 1e-8), '\n')
    print(sol_gradcon(A,b, np.zeros(i), 500, 1e-8))
'''

#d
diags = [10,100,1000]

'''
 for n in diags:
    def Matriz11d(k):
        A = np.zeros((k,k))
        np.fill_diagonal(A, n)
        h = - np.ones(k-1)
        K = np.diag(h, 1)
        P = np.diag(h, -1)
        return A + K + P
    A = Matriz11d(n)
    b = np.ones(n)
    print(f'---------------- n = {n}')
    print(sol_gradopt(A, b, np.zeros(n), 500, 1e-8))
    print(sol_Richardson(A,b, np.zeros(n), 500, 1e-8))
    print(sol_gradcon(A,b, np.zeros(n), 500, 1e-8))
'''

#e

for n in diags:
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            A[i,j] = 1 / (i + j + 1) 
            b[i] = b[i] + A[i,j] 
    b = (1/3) * b
    print(f'---------------- n = {n}')
    #print(sol_gradopt(A, b, np.zeros(n), 500, 1e-8))
    #print(sol_Richardson(A,b, np.zeros(n), 500, 1e-8))
    #print(sol_gradcon(A,b, np.zeros(n), 500, 1e-8))


