import numpy as np
from Ejercicio_10 import D
import sys
sys.path.append(r"C:\Users\Pedro\Desktop\LMA\Analisis_Numerico_2")

def dlup(W):
    
    A = W.copy()
    n = len(A[:,1])
    Cambios = np.arange(n)
    P = np.eye(n)

    Variable_ej_13 = 1

    for i in range(n - 1):
        h = np.argmax(np.abs(A[i :, i])) + i
        if h != i:
            Variable_ej_13 = Variable_ej_13 * (-1)
            Permutacion = [h, i]
            A[[i,h], :] = A[Permutacion, :]
            P[[i,h], :] = P[Permutacion, :]

        A[i + 1:, i] = A[i + 1:, i] / A[i,i]
        A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i, i + 1:])

    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    return L, U, P.T, Variable_ej_13

def det_lu(A):
    L, U, P, signo = dlup(A)
    determinante = 1
    for i in range(A.shape[0]):
        determinante = determinante * U[i,i]
    determinante = determinante * signo
    return determinante


S = np.array([[0,4,1], [1,1,3], [2,-2,1]], dtype=float)
print(D)
print(np.linalg.det(D))
print(det_lu(D))

