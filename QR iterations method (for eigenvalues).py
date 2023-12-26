import numpy as np

D=np.array([
    [-88,3,5],
    [159,-4,8254],
    [2,6,90]
], dtype=float)

def givens(x1,x2):
    c = 1.
    s = 0.
    ax1 = abs(x1)
    ax2 = abs(x2)
    if ax1 + ax2 > 0:
        if ax2 > ax1:
            tau = -x1/x2
            s = -np.sign(x2)/np.sqrt(1 + tau**2)
            c = tau*s
        else:
            tau = -x2/x1
            c = np.sign(x1)/np.sqrt(1 + tau**2)
            s = tau*c
    return c, s

def hh(x):
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
    return u,g

#Reducimos a una hessenberg tal que QHQ.T=A
def hess_hh(M):
    H=M.copy()
    n=H.shape[0]    
    Q=np.eye(n)
    for i in range(n-2):
        u,p=hh(H[i+1:,i])
        w=u*p
        H[i+1:,i:]=H[i+1:,i:]-np.outer(w,u.T@H[i+1:,i:])
        H[:,i+1:]= H[:,i+1:]- H[:,i+1:]@np.outer(w,u.T)
        Q[:,i+1:]= Q[:,i+1:]- Q[:,i+1:]@np.outer(w,u.T)
    
    return H,Q

#Reducimos a una hessenberg tal que QHQ.T=A
def hess_giv(M):
    H=M.copy()
    n=H.shape[0]
    Q=np.eye(n)
    for j in range(n-2): #columnas
        for i in range(j+2, n): #filas
            c, s = givens(H[j+1, j], H[i, j])
            G = np.array([
                [c, -s],
                [s, c]
            ],dtype=float)
            H[[j+1, i], j:] = G @ H[[j+1, i], j:]
            H[:, [j+1,i]] = H[:, [j+1,i]] @ G.T
            Q[:, [j+1, i]] = Q[:, [j+1, i]] @ G.T
    return H,Q

def aut_qr(M,tol,it):
    A=M.copy()
    n=A.shape[0]
    H,Q=hess_giv(A)
    g=np.zeros((n,2)) #Guardaré las rotaciones del primer for para usarlas en el 2do
    error=1
    k=0
    while error>=tol and k<it:
        for j in range(n-1):
            c,s=givens(H[j,j], H[j+1,j])
            g[j, :] = np.array([c,s])
            G=np.array([
                [c,-s],
                [s,c]
            ], dtype=float)
            H[[j,j+1],j:]=G@H[[j,j+1],j:]
        for i in range(n-1):
            c,s=g[i,:]
            G=np.array([
                [c,-s],
                [s,c]
            ], dtype=float)            
            H[:,[i,i+1]]=H[:,[i,i+1]]@(G.T)
            Q[:,[i,i+1]]=Q[:,[i,i+1]]@(G.T)
        error=np.linalg.norm(np.diag(H,-1),2)
        k+=1
    
    eig_v=np.diag(H)
    return eig_v #Q,k,H

print(aut_qr(D,10e-10,500),'\n')
print(np.linalg.eig(D)[0])
