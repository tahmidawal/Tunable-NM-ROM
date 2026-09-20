"""Heat-only physical contract around the shared channels-last 3D networks."""
import numpy as np
import jax
import jax.numpy as jnp
from . import models3d as M


def coordinates(n):
    a=jnp.arange(1,n,dtype=jnp.float64)/n
    return jnp.stack(jnp.meshgrid(a,a,a,indexing='ij'),axis=-1)*2-1


def arrays(fields,scale):
    fields=np.asarray(fields);n=fields.shape[-1]+1
    x=np.concatenate((fields[:,0,...,None]/scale,
                      np.broadcast_to(np.asarray(coordinates(n)),(len(fields),n-1,n-1,n-1,3))),axis=-1)
    y=np.moveaxis(fields[:,1:]/scale,1,-1)
    return x,y


def engine(params,spec,scale,n):
    @jax.jit
    def query(u0,p,scale):
        x=jnp.concatenate((u0[...,None]/scale,coordinates(n)),axis=-1)[None]
        y=M.apply_model(p,x,spec)[0]*scale
        return jnp.concatenate((u0[None],jnp.moveaxis(y,-1,0)),axis=0)
    return lambda u0:query(u0,params,jnp.asarray(scale))
