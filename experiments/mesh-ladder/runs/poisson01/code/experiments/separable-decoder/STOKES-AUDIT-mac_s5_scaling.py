import numpy as np


def curl_mode(n, k, ell):
    h = 1.0/n
    xline = np.arange(1,n)[:,None]
    yhalf = (np.arange(n)+0.5)[None,:]
    xhalf = (np.arange(n)+0.5)[:,None]
    yline = np.arange(1,n)[None,:]
    u = np.sin(k*np.pi*xline/n) * (2*np.sin(ell*np.pi/(2*n))/h) * np.cos(ell*np.pi*yhalf/n)
    v = -(2*np.sin(k*np.pi/(2*n))/h) * np.cos(k*np.pi*xhalf/n) * np.sin(ell*np.pi*yline/n)
    return u, v


def lap(u, v, h):
    lu = np.zeros_like(u)
    lv = np.zeros_like(v)
    # u normal-x zero values and tangential-y odd ghosts.
    ug = np.pad(u, ((1,1),(1,1)), mode="constant")
    ug[1:-1,0] = -u[:,0]
    ug[1:-1,-1] = -u[:,-1]
    lu[:] = (ug[2:,1:-1] + ug[:-2,1:-1] + ug[1:-1,2:] + ug[1:-1,:-2] - 4*u)/h**2
    # v tangential-x odd ghosts and normal-y zero values.
    vg = np.pad(v, ((1,1),(1,1)), mode="constant")
    vg[0,1:-1] = -v[0,:]
    vg[-1,1:-1] = -v[-1,:]
    lv[:] = (vg[2:,1:-1] + vg[:-2,1:-1] + vg[1:-1,2:] + vg[1:-1,:-2] - 4*v)/h**2
    return lu, lv


def vec(u,v):
    return np.concatenate((u.ravel(),v.ravel()))


def clamped(n,a,b):
    h=1/n
    psi=np.zeros((n+1,n+1))
    x=np.arange(n+1)[:,None]/n
    y=np.arange(n+1)[None,:]/n
    psi[:]=np.sin(a*np.pi*x)**2*np.sin(b*np.pi*y)**2
    u=(psi[1:n,1:]-psi[1:n,:-1])/h
    v=-(psi[1:,1:n]-psi[:-1,1:n])/h
    return u,v


for n in (8,16,32,64,128,256):
    h=1/n
    rels=[]
    phis=[]
    lams=[]
    for k in range(1,9):
        for ell in range(1,9):
            u,v=curl_mode(n,k,ell)
            lu,lv=lap(u,v,h)
            lam=4/h**2*(np.sin(k*np.pi/(2*n))**2+np.sin(ell*np.pi/(2*n))**2)
            p=vec(u,v); lp=vec(lu,lv)
            rels.append(np.linalg.norm(lp+lam*p)/np.linalg.norm(lp))
            phis.append(p); lams.append(lam)
    Phi=np.column_stack(phis)
    G=np.column_stack([vec(*clamped(n,a,b)) for a,b in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]])
    G,_=np.linalg.qr(G)
    LG=np.column_stack([vec(*lap(*[x for x in clamped(n,a,b)],h)) for a,b in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]])
    # QR transformed G; obtain the same transform for LG.
    Graw=np.column_stack([vec(*clamped(n,a,b)) for a,b in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]])
    _,R=np.linalg.qr(Graw)
    LG=LG@np.linalg.inv(R)
    B=Phi.T@G
    A=Phi.T@LG
    ratio=np.linalg.norm(A+np.asarray(lams)[:,None]*B)/np.linalg.norm(A)
    print(f"n={n}: eigdef min/med/max={min(rels):.6f}/{np.median(rels):.6f}/{max(rels):.6f}; A-ratio={ratio:.6f}")
