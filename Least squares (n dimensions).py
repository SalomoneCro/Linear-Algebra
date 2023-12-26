import numpy as np
import sys
#sys.path.append(r"C:\Users\Pedro\Desktop\LMA\Analisis_Numerico_2")
#from Practico1.Ejercicio_3 import soltrsupcol
from Ejercicio_3 import qrhholder

def soltrsupcol(M,n):
    z = n.copy()
    k = np.max(np.nonzero(n))
    for j in reversed(range(k + 1)):
        z[j] = z[j] / M[j,j]
        z[:j] = z[:j] - z[j] * M[: j, j]
    return z

#En vez de usar givens usar householder
def sol_cuadmin(A, b):
    m, n = A.shape

    #Asumimos que A tiene rango completo
    p = np.min([m,n])
    y_sol = np.zeros(n)
    # I = {1, ... , p} -> :p
    # J = {1, ... , p} \ I = {p+1, ... , n} -> p:

    Q, R = qrhholder(A)

    q = Q.T @ b

    y_sol[:p] = soltrsupcol(R[ :p, :p], q[:p])

    residuo = np.linalg.norm(q[p:])

    return y_sol, residuo






A = np.array([[1,1],
              [0.5,0],
              [0,0.5]], dtype=float)

b = np.array([1,1,1], dtype=float)

#x_sombrero = np.linalg.lstsq(A, b)
y_sombrero = sol_cuadmin(A, b)
#print(x_sombrero)
#print(y_sombrero)

A=np.array([
    [2,3,5,5],
    [5,7,6,7],
    [2,4,6,0],
    [2,5,9,8]], dtype=float)

w, W = np.linalg.eig(A)
#print(np.real(W @ np.diag(w) @ np.linalg.inv(W)))

