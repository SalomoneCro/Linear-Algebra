import numpy as np
from Ejercicio_12 import cholesky
from Ejercicio_3 import soltrsupfil, soltrinfcol

def soldefpos(A, b):
    Ch = cholesky(A)
    y = soltrinfcol(Ch.T, b)
    x = soltrsupfil(Ch, y)
    return(x)

D = np.array([[4,2,6],
              [2,2,5],
              [6,5,29]], dtype=float)

s = np.array([1,2,3], dtype=float)
#print(soldefpos(D,s))