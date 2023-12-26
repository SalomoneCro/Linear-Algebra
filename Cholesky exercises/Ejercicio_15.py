import numpy as np
from Ejercicio_14 import soldefpos
from Ejercicio_13 import M
import matplotlib.pyplot as plt

n = 500
sinnombre = np.array([1/10, 1/2, 9/10], dtype=float)

fig, ax = plt.subplots(3,1)
u = np.linspace(0, 1, n)

j = 0

for i in sinnombre:
    c = n * i
    iteracion = (np.exp((-(j-c)**2) / 100) for j in range(n))

    b = np.fromiter(iteracion, float)
    x = soldefpos(M(n), b)

    ax[j].plot(u, x, label='Soluciones del sistema',color='green')
    ax[j].plot(u, b, label='Valores del vector b',color='purple')    
    ax[j].grid()
    ax[j].legend()

    j += 1
plt.show()
