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


def native_engine(params,spec,scale,native_intervals,evaluation_intervals):
    """Charge native-grid prediction and nodally aligned zero-wall interpolation."""
    assert evaluation_intervals%native_intervals==0
    stride=evaluation_intervals//native_intervals
    @jax.jit
    def query(u0,p,scale):
        coarse=u0[stride-1::stride,stride-1::stride,stride-1::stride]
        x=jnp.concatenate((coarse[...,None]/scale,coordinates(native_intervals)),axis=-1)[None]
        y=jnp.moveaxis(M.apply_model(p,x,spec)[0]*scale,-1,0)
        y=jnp.pad(y,((0,0),(1,1),(1,1),(1,1)))
        position=jnp.arange(1,evaluation_intervals,dtype=jnp.float64)*native_intervals/evaluation_intervals
        left=jnp.floor(position).astype(jnp.int32);weight=position-left
        for axis in (1,2,3):
            shape=[1]*4;shape[axis]=len(position);w=weight.reshape(shape)
            y=(1-w)*jnp.take(y,left,axis=axis)+w*jnp.take(y,left+1,axis=axis)
        return jnp.concatenate((u0[None],y),axis=0)
    return lambda u0:query(u0,params,jnp.asarray(scale))


def native_sensor_deeponet_engine(params,spec,scale,native_intervals,evaluation_intervals):
    """Keep the branch sensor grid fixed and evaluate the coordinate trunk directly."""
    from . import extra_models3d as E
    assert spec['kind']=='deeponet3d' and evaluation_intervals%native_intervals==0
    stride=evaluation_intervals//native_intervals
    @jax.jit
    def query(u0,p,scale):
        coarse=u0[stride-1::stride,stride-1::stride,stride-1::stride]
        coefficients=E.deeponet_coefficients(p,(coarse/scale)[None,...,None],spec)
        trunk=E.deeponet_trunk(p,coordinates(evaluation_intervals)[None],spec)
        values=jnp.einsum('bcr,bxyzr->bxyzc',coefficients,trunk,precision='highest')/jnp.sqrt(trunk.shape[-1])+p['bias']
        return jnp.concatenate((u0[None],jnp.moveaxis(values[0]*scale,-1,0)),axis=0)
    return lambda u0:query(u0,params,jnp.asarray(scale))
