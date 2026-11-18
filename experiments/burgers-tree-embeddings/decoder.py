"""Separable global and shared-local neural decoders, all in float64."""
from common import *
import geometry as geo

def init_mlp(key,sizes,last_scale=1.):
    keys=jax.random.split(key,len(sizes)-1);p=[]
    for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
        w=jax.random.normal(keys[i],(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b))
        if i==len(sizes)-2:w=w*last_scale
        p.append(dict(w=w,b=jnp.zeros(b,dtype=jnp.float64)))
    return p

def mlp(p,x):
    for layer in p[:-1]:x=jnp.tanh(x@layer['w']+layer['b'])
    return x@p[-1]['w']+p[-1]['b']

def encoding(x):
    # Fourier convention: sin/cos(2*pi*f*x_a), f=1,2,4,8, both axes.
    z=(2*jnp.pi*x[...,None]*jnp.array([1.,2.,4.,8.])).reshape(x.shape[:-1]+(8,))
    return jnp.concatenate([2*x-1,jnp.sin(z),jnp.cos(z)],axis=-1)

def init_global(key):return init_mlp(key,[18,128,128,128,128],.1)

def global_features(p,x):return geo.bc(x)[...,None]*mlp(p,encoding(x))

def init_local(key,width=64,features=16):return init_mlp(key,[3,width,width,features],.1)

def local_features(p,x,cell,halo=.25):
    return geo.window(x,cell,halo)[...,None]*mlp(p,geo.local_inputs(x,cell))

def local_matrix(p,x,cells,rank=16,halo=.25):
    if len(cells)==0:return jnp.zeros((len(x),0),dtype=jnp.float64)
    return jnp.concatenate([local_features(p,x,c,halo)[:,:rank] for c in cells],axis=1)

def init_head(key,k,r,width,linear=None):
    if linear is None:
        linear=jnp.eye(r,k,dtype=jnp.float64)
    return dict(linear=jnp.asarray(linear),net=init_mlp(key,[k,width,width,r],.01))

def head(p,z):
    return z@p['linear'].T+mlp(p['net'],z)-mlp(p['net'],jnp.zeros_like(z))

def coefficients(parent,child,z,nlocal,rank=4):
    root=head(parent,z[:16])
    if nlocal==0:return root
    rest=z[16:].reshape(nlocal,rank)
    return jnp.concatenate([root,rest.reshape(-1) if child is None else head(child,rest).reshape(-1)])

def field(G,parent,child,z,nlocal,rank=4):
    return G@coefficients(parent,child,z,nlocal,rank)

def nparameters(p):return sum(x.size for x in jax.tree_util.tree_leaves(p))
