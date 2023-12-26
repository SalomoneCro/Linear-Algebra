import numpy as np
import sys
sys.path.append(r"C:\Users\Pedro\Desktop\LMA\Analisis_Numerico_2")
from Practico1.Ejercicio_3 import soltrsupcol
np.set_printoptions(precision = 2, suppress = True)

def sol_egauss(A, b):
    U = A.copy()
    y = b.copy()
    r = np.zeros(len(b))
    P = np.eye(len(b))
    for i in range(len(b) - 1):
        h = np.argmax(np.abs(A[i :, i])) + i
        if h != i:
            Permutacion = [h, i]
            U[[i,h], :] = U[Permutacion, :]
            P[[i,h], :] = P[Permutacion, :]
            y[i], y[h] = y[h], y[i]
        r[i + 1 :] = U[i + 1:, i] / U[i , i]
        U[i + 1 :, i:] -= np.outer(r[i + 1 :], U[i , i:])
        y[i + 1 :] = y[i + 1 :] - r[i + 1 :] * y[i]
    return U, y, P


D = np.array([[2,10,8,8,6],
              [1,4,-2,4,-1],
              [0,2,3,2,1],
              [3,8,3,10,9],
              [1,4,1,2,1]], dtype=float)
p = np.array([52,14,12,51,15], dtype=float)
q = np.array([50,4,12,48,12], dtype=float)

U, y, P = sol_egauss(D, p) 
S, w, C = sol_egauss(D, q)

Solucion1 = soltrsupcol(U, y)
Solucion2 = soltrsupcol(S, w)

print(D @ Solucion1)
print(D @ Solucion2)