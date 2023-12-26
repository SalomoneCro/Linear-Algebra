import numpy as np
import matplotlib.pyplot as plt

'''
#Si A es de rango completo
def nivel():
    #A = np.random.randn(2,2)
    A = np.array([[375,374], [752,750]], dtype=float)
    #A = A @ A.T

    r = np.linspace(0, 2 * np.pi, 100)
    cosenos = np.cos(r)
    senos = np.sin(r)

    y_0 = np.zeros(100)
    y_1 = np.zeros(100)

    for i in range(100):
        res = A @ np.array([cosenos[i], senos[i]])

        y_0[i] = res[0]
        y_1[i] = res[1]

    fig, ax = plt.subplots(1,1)
    ax.plot(cosenos, senos, label='Bola unidad')
    ax.plot(y_0, y_1, label='Transformacion')
    ax.axis("equal")
    ax.grid()
    print(A)
    plt.show()
'''
def nivel(N, niveles):
    A = np.random.randn(2,2)
    #A = np.array([[1000, 999], [999,998]], dtype=float)
    A = A @ A.T
    X = np.linspace(-4,4,N)
    Y =  np.linspace(-4,4,N)

    XX, YY = np.meshgrid(X,Y)
    ZZ = np.zeros((N,N))

    for idx in range(N):
        for idy in range(N):
            vector = np.array([XX[idx,idy], YY[idx,idy]], dtype=float)
            ZZ[idx,idy] = vector.T @ A @ vector

    fig, ax = plt.subplots(1,1)
    CS = ax.contour(XX, YY, ZZ, niveles)
    ax.clabel(CS, inline=True, fontsize=10)
    #ax.axis("equal")
    ax.grid()
    ax.set_facecolor('xkcd:dark grey')
    fig.patch.set_facecolor('xkcd:dark grey')
    plt.show()

nivel(100, [0,1,2,3,4,5,6])
#nivel()