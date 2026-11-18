"""Shared smooth local span control; fixed windows and per-cell physical whitening."""
from common import *
import geometry as geo,data

def raw_features(x,cell,side):
    v=geo.local_inputs(x,cell)[...,:2]/1.5
    def poly(t):
        p=[jnp.ones_like(t),t]
        for n in range(2,side):p.append(((2*n-1)*t*p[-1]-(n-1)*p[-2])/n)
        return jnp.stack(p[:side],axis=-1)
    px,py=poly(v[:,0]),poly(v[:,1])
    return geo.window(x,cell)[:,None]*(px[:,:,None]*py[:,None,:]).reshape(len(x),side*side)

def local_blocks(x,w,cells,side):
    blocks=[];transforms=[]
    for c in cells:
        raw=raw_features(x,c,side);gram=(raw*w[:,None]).T@raw
        L=jnp.linalg.cholesky(gram+1e-14*jnp.trace(gram)/len(gram)*jnp.eye(len(gram)))
        transform=jax.scipy.linalg.solve_triangular(L.T,jnp.eye(len(gram)),lower=False)
        blocks.append(raw@transform);transforms.append(transform)
    return jnp.concatenate(blocks,axis=1),jnp.stack(transforms)

def tree(x,w,p,nodes,depth,policy):
    indicator=np.asarray(data.initial(x,p))**2
    if policy=='swept':
        speed=float(jnp.max(data.initial(x,p)))
        # Initial callable and maximum-speed bound only; no evolved reference.
        for t in np.linspace(0,.25,9):
            shifted=jnp.asarray(x)-speed*t
            inside=jnp.all((shifted>=0)&(shifted<=1),axis=-1)
            indicator=np.maximum(indicator,np.asarray(jnp.where(inside,data.initial(shifted,p),0.))**2)
    return geo.allocated(x,indicator,w,16+nodes*4,4,depth)

@jax.jit
def ridge_projection(G,Y,w):
    gram=(G*w[:,None]).T@G;rhs=(Y*w[None])@G
    def solve(ridge):
        c=jnp.linalg.solve(gram+ridge*jnp.eye(gram.shape[0]),rhs.T).T
        return c,c@G.T
    c,p=solve(1e-10);_,p8=solve(1e-8)
    return c,p,p8
