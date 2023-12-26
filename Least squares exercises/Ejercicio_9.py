import numpy as np
import sys
#sys.path.append(r"C:\Users\Pedro\Desktop\LMA\Analisis_Numerico_2")
#from Practico1.Ejercicio_13 import M
from Ejercicio_7 import sol_cuadmin
import numpy as np
def tribanda(n):
    M = np.zeros((n,n))
    np.fill_diagonal(M, 2)
    h = np.ones(n-1)
    K = np.diag(h, 1)
    P = np.diag(h, -1)
    M = M - K - P
    return M[: , : -2]

def cholesky(A):
    n = len(A[0 , :])
    G = np.copy(A)
    for i in range(n):
        G[i,i] = np.sqrt(G[i,i])
        G[i , i+1:] = G[i , i+1:] / G[i,i]
        G[i+1: , i+1:] = G[i+1: , i+1:] - np.outer(G[i , i+1:],G[i , i+1:])
    G = np.triu(G, 0)
    return G


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

def sol_QR(n):
    #Creo el sistema
    A = tribanda(n)
    b = np.zeros(n)
    b[0], b[-1] = 1, 1
    #Resuelvo el sistema
    x= sol_cuadmin(A, b)[0]
    return x

def sol_ch(n): #Resuelvo el sistema (A.T@A)x = A.T@b, para que el sistema sea cuadrado
    #Creo el sistema de ec normales
    A = tribanda(n)
    b = np.zeros(n)
    b[0], b[-1] = 1, 1
    b = A.T @ b
    A = A.T @ A
    #Resuelvo el sistema
    G = cholesky(A)
    y = soltrinfcol(G.T, b)
    x = soltrsupfil(G, y)
    return x

sol_ch100 = sol_ch(100)
sol_QR100 = sol_QR(100)

print(sol_ch100)
aver = tribanda(100) @ sol_ch100
print(tribanda(100).shape)
print(len(aver))
print(aver)



'''
n = np.array([100])

for i in n:
    A = tribanda(i)
    C = A[ : , : -2]
    b = np.zeros(i)
    b[0] = 1
    b[-1] = 1
    x = np.linalg.lstsq(C,b)
    #x = sol_cuadmin(C,b)

print(b)

print(C @ x[0])
'''
#Esta bien que C @ x[0] no me de b? -> Parece que si

