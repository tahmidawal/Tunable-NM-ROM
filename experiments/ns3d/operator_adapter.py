"""Complete physical vector-velocity query for the frozen shared operator API."""
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
from operators import models3d as M


def checkpoint(path,obj):
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(jax.tree_util.tree_map(lambda a:np.asarray(a) if hasattr(a,'dtype') else a,obj),stream,protocol=5)
    temp.replace(dest)


def make_query(spec,n,projected=True):
    @jax.jit
    def query(u0,nu,params,mean,std,scale,geom):
        inputs=jnp.concatenate((u0,jnp.broadcast_to(nu,(1,n,n,n))),axis=0)
        inputs=(inputs-mean[0])/std[0]
        x=jnp.moveaxis(inputs,0,-1)[None]
        if spec['kind'] in ('deeponet3d','transolver3d'):
            axis=jnp.arange(n,dtype=jnp.float64)/n
            xyz=jnp.stack(jnp.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
            x=jnp.concatenate((x,xyz[None]),axis=-1)
        out=M.apply_model(params,x,spec)[0]
        evolved=(jnp.moveaxis(out,-1,0)*scale).reshape(5,3,n,n,n)
        if projected:
            evolved=jax.vmap(F.project_field,in_axes=(0,None))(evolved,geom)
        return jnp.concatenate((u0[None],evolved))
    return query
