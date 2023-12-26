import numpy as np
from Ejercicio_7 import sol_cuadmin
import sys
sys.path.append(r"C:\Users\Pedro\Desktop\LMA\Analisis_Numerico_2")
from Practico2.Ejercicio_9 import dlup
from Practico1.Ejercicio_3 import soltrinffil, soltrsupcol


def mtrx(x):
    A=np.array([
        [1,1],
        [x, 0],
        [0, x]
    ],dtype=float)
    return A


def sol_tradicional(F, s):
    A = F.copy()
    b = s.copy()
    b = A.T @ b
    A = A.T @ A
    L , U = dlup(A)[0], dlup(A)[1]
    y = soltrinffil(L,b)
    x = soltrsupcol(U,y)
    return x

def sol_QR(F, s):
    return sol_cuadmin(F, s)



b = np.array([1,1,1], dtype=float)
for k in range(8):

    A = np.array([[1,1],
                 [1/(10 ** k),0],
                 [0,1/(10 ** k)]], dtype=float)
    
    print(sol_tradicional(A,b))
    print(sol_QR(A, b))

