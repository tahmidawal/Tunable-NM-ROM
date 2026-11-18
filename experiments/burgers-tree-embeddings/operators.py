"""Incumbent scaled tested weak residual, exact fine-grid sign-upwind stencil.

No continuous-flux replacement and no pointwise residual minimization. The dense
fixture qualifies arithmetic only; large-mesh integration remains a later gate.
"""
from common import jnp,np

def modes(L,M):
    k=np.arange(1,L);kx,ky=np.meshgrid(k,k,indexing='ij')
    lam=4*L**2*(np.sin(np.pi*kx/(2*L))**2+np.sin(np.pi*ky/(2*L))**2)
    ind=np.argsort(lam.ravel(),kind='stable')[:M]
    kx,ky=kx.ravel()[ind],ky.ravel()[ind]
    x=np.arange(1,L)/L;c=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
    phi=(2./L)*np.sin(np.pi*c[:,0,None]*kx)*np.sin(np.pi*c[:,1,None]*ky)
    return jnp.asarray(phi),jnp.asarray(lam.ravel()[ind]),c

def spatial(u,L):
    p=jnp.pad(u.reshape(L-1,L-1),1)
    c,xm,xp,ym,yp=p[1:-1,1:-1],p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:]
    adv=c*L*(jnp.where(c>0,c-xm,xp-c)+jnp.where(c>0,c-ym,yp-c))
    lap=L**2*(xm+xp+ym+yp-4*c)
    return adv.reshape(-1),lap.reshape(-1)

def tested(coeff,previous_tested,A,lam,G5,Pq,L,nu,dt):
    us=jnp.einsum('msr,r->ms',G5,coeff)
    c,xp,xm,yp,ym=[us[:,i] for i in range(5)]
    adv=c*L*(jnp.where(c>0,c-xm,xp-c)+jnp.where(c>0,c-ym,yp-c))
    ah=A@coeff
    return (ah-previous_tested+dt*(Pq.T@adv+nu*lam*ah))/(1+dt*nu*lam)
