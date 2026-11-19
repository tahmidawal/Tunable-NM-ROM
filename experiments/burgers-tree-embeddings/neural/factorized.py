"""Learned coordinate features with an identifiable nonlinear child embedding.

Each child code is exactly the first half of its feature coefficients; a small
tanh network supplies the nonlinear second half. The parent network is frozen.
"""
from common import *
from neural import fields as F


def window(cell,xy):
    h=2.**(-cell[0]);center=(cell[1:]+.5)*h;q=(xy-center)/h
    t=jnp.clip((jnp.abs(q)-.5)/.25,0,1)
    w=jnp.prod(1-(10*t**3-15*t**4+6*t**5),axis=-1)
    return q,w


def feature_matrix(feature,cells,xy):
    def one(cell):
        q,w=window(cell,xy)
        v=F.mlp(feature,jnp.clip(q/.75,-1,1),True)
        return v*(F.boundary(xy)*w)[:,None]
    return jax.vmap(one)(cells).transpose(1,0,2)


def coefficients(head,z,scale):
    tail=F.mlp(head,z)-F.mlp(head,jnp.zeros_like(z))
    return jnp.concatenate([z,tail],axis=-1)*scale


def cached(params,z,xy):
    k=params['scale'].shape[0]//2;local=z[8:].reshape(-1,k)
    c=coefficients(params['head'],local,params['scale'])
    return F.root(params['root'],z[:8],xy)+jnp.einsum('pnr,nr->p',params['B'],c)


def cached_jacobian(params,z,xy):
    k=params['scale'].shape[0]//2;local=z[8:].reshape(-1,k)
    root=jax.jacfwd(F.root,argnums=1)(params['root'],z[:8],xy)
    hj=jax.vmap(jax.jacfwd(coefficients,argnums=1),in_axes=(None,0,None))(params['head'],local,params['scale'])
    children=jnp.einsum('pnr,nrk->pnk',params['B'],hj).reshape(len(xy),-1)
    return jnp.concatenate([root,children],axis=-1)


def tree(params,z,xy,cells):
    p=dict(params,B=feature_matrix(params['feature'],cells,xy))
    return cached(p,z,xy)


@jax.jit
def linear_initialization(root,root_codes,B,xy,targets,weight):
    """Offline span diagnostic/embedding initializer, never a nonlinear-model claim."""
    A=B.reshape(len(xy),-1);J=A*jnp.sqrt(weight)[:,None]
    H=J.T@J;scale=jax.lax.rsqrt(jnp.maximum(jnp.diag(H),1e-14))
    normalized=H*scale[:,None]*scale[None,:]+1e-8*jnp.eye(H.shape[0])
    prediction=jax.vmap(lambda z:F.root(root,z,xy))(root_codes)
    rhs=(targets-prediction)@(weight[:,None]*A)
    codes=(jnp.linalg.solve(normalized,(rhs*scale).T).T)*scale
    reconstructed=prediction+codes@A.T
    errors=jnp.sqrt(jnp.sum(weight*(reconstructed-targets)**2,axis=1)/jnp.sum(weight*targets**2,axis=1))
    return codes.reshape(len(targets),B.shape[1],B.shape[2]),errors
