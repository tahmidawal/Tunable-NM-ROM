"""Window-aligned physical integration, using separability rather than sampled 2D fits."""
from common import *

def rule(panels=64,order=24,N=512):
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    x=((np.arange(panels)[:,None]+x[None])/panels).ravel();w=np.tile(w/panels,panels)
    S=np.sin(np.pi*x[:,None]*np.arange(1,N+1)[None]);return jnp.asarray(x),jnp.asarray(w),jnp.asarray(S)

def make_geometry(side=6):
    @jax.jit
    def geometry(global_basis,cells,x,w,S):
        h=2.**(-cells[:,0]);centers=(cells[:,1:]+.5)*h[:,None]
        def axis(j):
            v=(x[None]-centers[:,j,None])/(.75*h[:,None]);polys=[jnp.ones_like(v),v]
            for n in range(2,side):polys.append(((2*n-1)*v*polys[-1]-(n-1)*polys[-2])/n)
            t=jnp.clip((jnp.abs((x[None]-centers[:,j,None])/h[:,None])-.5)/.25,0.,1.)
            window=1-(10*t**3-15*t**4+6*t**5)
            raw=jnp.stack(polys[:side],axis=-1)*(4*x*(1-x))[None,:,None]*window[:,:,None]
            gram=jnp.einsum('pna,n,pnb->pab',raw,w,raw)
            chol=jnp.linalg.cholesky(gram+1e-14*jnp.trace(gram,axis1=1,axis2=2)[:,None,None]/side*jnp.eye(side))
            trans=jax.vmap(lambda L:jax.scipy.linalg.solve_triangular(L.T,jnp.eye(side),lower=False))(chol)
            values=jnp.einsum('pna,pab->pnb',raw,trans)
            matrix=values.transpose(1,0,2).reshape(len(x),-1)
            moments=(S*w[:,None]).T@matrix
            return matrix,trans,moments.reshape(S.shape[1],len(cells),side).transpose(1,0,2)
        X,TX,MX=axis(0);Y,TY,MY=axis(1);P=len(cells)
        gx=((X*w[:,None]).T@X).reshape(P,side,P,side)
        gy=((Y*w[:,None]).T@Y).reshape(P,side,P,side)
        local=jnp.einsum('paqb,pcqd->pacqbd',gx,gy).reshape(P*side*side,P*side*side)
        N=global_basis.shape[-1];cross=jnp.einsum('gij,pia,pjb->gpab',global_basis,MX[:,:N],MY[:,:N]).reshape(len(global_basis),-1)
        root=.25*jnp.einsum('gij,hij->gh',global_basis,global_basis)
        gram=jnp.block([[root,cross],[cross.T,local]])
        transforms=jax.vmap(jnp.kron)(TX,TY)
        return gram,transforms,MX,MY
    return geometry

@jax.jit
def rhs(global_basis,reference,MX,MY):
    N=global_basis.shape[-1];root=.25*jnp.einsum('tij,gij->tg',reference[:,:N,:N],global_basis)
    # The inherited reference uses N intervals and N-1 interior sine modes.
    nref=reference.shape[-1]
    local=jnp.einsum('tij,pia,pjb->tpab',reference,MX[:,:nref],MY[:,:nref]).reshape(len(reference),-1)
    return jnp.concatenate([root,local],axis=1)

@jax.jit
def fit(gram,b,norm2,ridge=1e-8):
    c=jnp.linalg.solve(gram+ridge*jnp.eye(gram.shape[0]),b.T).T
    squared=jnp.einsum('ti,ij,tj->t',c,gram,c)-2*jnp.sum(c*b,axis=1)+norm2
    return c,jnp.sqrt(jnp.maximum(squared,0)/norm2)
