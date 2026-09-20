"""Explicit forcing-to-solution contract; training-only global normalization."""
import numpy as np
import jax
import jax.numpy as jnp
from . import models3d as M


def coordinates(n):
    a=jnp.arange(1,n,dtype=jnp.float64)/n
    return jnp.stack(jnp.meshgrid(a,a,a,indexing='ij'),axis=-1)*2-1


def arrays(forcing,solution,scales):
    forcing=np.asarray(forcing);n=forcing.shape[-1]+1
    x=np.concatenate((forcing[...,None]/scales['input'],
        np.broadcast_to(np.asarray(coordinates(n)),(*forcing.shape,3))),axis=-1)
    y=np.asarray(solution)[...,None]/scales['output']
    return x,y


def engine(params,spec,scales,n):
    params=jax.device_put(params);inscale=jnp.asarray(scales['input']);outscale=jnp.asarray(scales['output'])
    jax.block_until_ready((params,inscale,outscale))
    @jax.jit
    def query(forcing,p,inscale,outscale):
        x=jnp.concatenate((forcing[...,None]/inscale,coordinates(n)),axis=-1)[None]
        return M.apply_model(p,x,spec)[0,...,0]*outscale
    return lambda forcing:query(forcing,params,inscale,outscale)


def interpolation_matrix(native,n):
    """Nodal linear interpolation with the known zero boundary included."""
    positions=np.arange(1,n)*native/n;left=np.floor(positions).astype(int);fraction=positions-left
    matrix=np.zeros((n-1,native-1),dtype=np.float64)
    for row,(index,f) in enumerate(zip(left,fraction)):
        if 1<=index<native:matrix[row,index-1]+=1-f
        if 1<=index+1<native:matrix[row,index]+=f
    return matrix


def native_interpolated(params,spec,scales,n,native):
    assert n%native==0;stride=n//native
    params=jax.device_put(params);inscale=jnp.asarray(scales['input']);outscale=jnp.asarray(scales['output'])
    @jax.jit
    def query(forcing,matrix,p,inscale,outscale):
        coarse=forcing[stride-1::stride,stride-1::stride,stride-1::stride]
        x=jnp.concatenate((coarse[...,None]/inscale,coordinates(native)),axis=-1)[None]
        value=M.apply_model(p,x,spec)[0,...,0]*outscale
        return jnp.einsum('ia,jb,kc,abc->ijk',matrix,matrix,matrix,value,optimize='optimal',precision='highest')
    matrix=jnp.asarray(interpolation_matrix(native,n))
    jax.block_until_ready((params,inscale,outscale,matrix))
    return lambda forcing:query(forcing,matrix,params,inscale,outscale)
