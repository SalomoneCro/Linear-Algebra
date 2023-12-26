import numpy as np
import numpy.linalg as l
#Esta linea es para que no me aparezcan numeros en notacion cientifica
np.set_printoptions(precision = 2, suppress = True)


A=np.array([
    [2,3,5,5],
    [5,7,6,7],
    [2,4,6,0],
    [2,5,9,8]], dtype=float)

S = np.array([[2,3],
              [5,7]], dtype=float)

b = np.array([12,29,4,9], dtype=float)

def soltrsupfil(Z,c):
    b = c.copy()
    k = np.max(np.nonzero(b))
    for i in reversed(range(k + 1)):
        b[i] = (b[i] - Z[i, i + 1:] @ b[i + 1:] ) / Z[i,i]
    return b

#Funcion que utilizare en qrgivens
def givens(a, b):
    c = a / np.sqrt(a**2 + b**2)
    s = b / np.sqrt(a**2 + b**2)

    return c, s

def qrgivens(F):
    #Estas lineas es para que el algoritmo funcione con marices rectangulares
    m, n = F.shape
    Q = np.eye(m)
    A = F.copy()
    r = min(m-1, n)
    #Estos for recorren todos los elementos debajo de la diagonal de la matriz
    for j in range(r):
        for i in range(j+1, m):

            # calculo givens (Se sirua en el elemento diagonal y de ahi va recorriendo para abajo)
            c, s = givens(A[j, j], A[i, j])
            
            rot_matrix = np.array([[c, -s],
                                   [s, c]])
            
            indices = [j, i]

            #Como la multiplicacion por izquierda(derecha) por la matriz de rotacion plana solo modifica 
            #las filas(columnas), no multiplico toda la matriz, solo estas filas(columnas)
            A[indices, :] = rot_matrix.T @ A[indices, :]
            Q[:, indices] = Q[:, indices] @ rot_matrix

    return Q, A  #A es R

'''
Q, R = qrgivens(A)

print(Q)
print(R)
print(Q @ R)

bprime = Q.T @ b

x = soltrsupfil(R, bprime)

print(A @ x)
'''
'''
def hh_vector(x):
    
    #u, rho = hh_vector(x)
    #Calcula u y rho tal que Q = I - rho u u^T
    #cumple Qx = \|x\|_2 e^1
    
    n = len(x)
    rho = 0
    u = x.copy()
    u[0] = 1.

    if n == 1:
        sigma = 0
    else:
        sigma = np.sum(x[1:]**2)

    if sigma>0 or x[0]<0:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0]<=0:
            gamma = x[0] - mu
        else:
            gamma = -sigma/(x[0] + mu)

        rho = 2 * (gamma**2) / (gamma**2 + sigma)
        u = u/gamma
        u[0] = 1

    return u, rho
'''
def hh_vector(x):

    u= x.copy()
    n= len(u)
    b= max(np.abs(u)) #norma infinito

    if b == 0:
        g = 0
        u = 1 #Nada más declaro para que no me genere problemas el return

    else:
        u = u*(1/b)
        t = np.linalg.norm(u,2)
        if u[0] < 0:
            t = -t
        u[0] = u[0] + t
        g = u[0] / t
        u[1:n] = u[1:n] / u[0]
        u[0] = 1

    return u, g

def qrhholder(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    p = min(m, n)

    for j in range(p):
        # I = j:, J = j:
        u, rho = hh_vector(R[j:, j])
        w = rho * u
        R[j:, j:] = R[j:, j:] - np.outer(w, u.T @ R[j:, j:])
        Q[:, j:] = Q[:, j:] - Q[:, j:] @ np.outer(w, u)

    return Q, R

'''
Q , R = qrhholder(A)
print(Q)
print(R)
print(Q @ R)
'''



#Este givens anda pero es muy costoso
'''
def qrgivens(F):
    A = F.copy()
    n, m = F.shape

    #En esta Q ire multiplicando todas las Qij
    Q = np.eye(m)
    r = min(n,m)

    #Estos for recorren todos los elementos debago de la diagonal de la matriz
    for j in range(r-1):
        for i in range(j+1, r):
            #Este vetor es la columna j-esima a la que le quiero producir ceros
            x = A[ : , j]

            #Este sera el angulo de mi rotacion
            tita = np.arccos(x[j] / np.sqrt(x[j]**2 + x[i]**2))

            #Armo la matriz de la rotacion plana
            Qij = np.eye(m)
            Qij[j,j] = np.cos(tita)
            Qij[j,i] = -np.sin(tita)
            Qij[i,j] = np.sin(tita)
            Qij[i,i] = np.cos(tita)
            
            #Voy construyendo Q
            Q = Q @ Qij
            #Voy haciendo ceros en A
            A = Qij.T @ A

    return Q, A  #A es R
'''

#HouseHolder que no me salio

'''
def hh_vector(x):
    u= x.copy()
    n= len(u)
    b= max(np.abs(u))
    if b==0:
        g= 0
        t=1 #Nada más la declaro para q no me genere problemas el return
    else:
        u= u*(1/b)
        t= np.linalg.norm(u,2)
        
        if u[0]<0:
            t=-t
        u[0]= u[0]+t
        g= u[0]/t
        u[1:n]= u[1:n]/u[0]
        u[0]= 1
        t= b*t

    return (t,g,u)

def QB(B, u, Gamma):
    vT = Gamma * u
    vT = vT @ B
    B = B - np.outer(u,vT)
    return B
'''
'''
def qrhholder(F):
    #Estas lineas es para que el algoritmo funcione con marices rectangulares
    m, n = F.shape
    A = F.copy()
    r = min(m-1, n)

    for i in range(n-1):

        Tao_i, Gamma, u = hh_vector(A[i : , i])

        Tao_i = np.sign(A[i,i]) * Tao_i

        #A[i + 1 : , i] = A[i + 1 : , i] / A[i,i] + Tao_i

        A[i + 1 : , i] = u[i + 1 :]
        A[i: , i + 1:] = QB(A[i : , i + 1:], u, 1/Tao_i**2 )
        A[i,i] = -Tao_i

        R = np.triu(A)
        return A , R
'''