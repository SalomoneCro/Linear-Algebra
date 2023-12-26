import numpy as np
from Ejercicio_9 import dlup
from Ejercicio_10 import D
import sys
sys.path.append(r"C:\Users\Pedro\Desktop\LMA\Analisis_Numerico_2")
from Practico1.Ejercicio_3 import soltrsupcol, soltrinffil


def inv_lu(A):
    n = A.shape[0]
    L, U, P = dlup(A)

    Final_Matrix = np.zeros((n,n))

    for i in range(n):
        v = np.zeros(n)
        v[i] = 1
        p = soltrinffil(L,v)
        Columnai_Final = soltrsupcol(U, p)

        Indice = np.argmax(P[i , :])

        Final_Matrix[ : , Indice] = Columnai_Final

    return Final_Matrix





print(inv_lu(D))
print(np.linalg.inv(D))


