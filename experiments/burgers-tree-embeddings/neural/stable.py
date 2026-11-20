"""Smooth latent-conditioned coordinate neural field; no online POD expansion."""
from common import *
from neural import fields as F


def encoding(xy):
    freq=jnp.asarray([1.,2.,4.,8.,16.])
    phase=2*jnp.pi*xy[...,None]*freq
    return jnp.concatenate([2*xy-1,jnp.sin(phase).reshape(xy.shape[:-1]+(10,)),
                            jnp.cos(phase).reshape(xy.shape[:-1]+(10,))],axis=-1)


def initialize(key,latent=32,width=128,depth=4):
    k1,k2=jax.random.split(key)
    layers=F.init_mlp(k1,[22]+[width]*depth+[1])
    modulation=F.init_mlp(k2,[latent,depth*width])[0]
    modulation['w']=modulation['w']*.5
    return dict(layers=layers,modulation=modulation)


def apply(p,z,xy):
    """Latents enter tanh layers, never the periodic coordinate encoding."""
    width=p['layers'][0]['b'].shape[0];depth=len(p['layers'])-1
    shifts=(z@p['modulation']['w']+p['modulation']['b']).reshape(z.shape[:-1]+(depth,width))
    h=encoding(xy)
    for i,layer in enumerate(p['layers'][:-1]):h=jnp.tanh(h@layer['w']+layer['b']+shifts[...,i,:])
    return F.boundary(xy)*(h@p['layers'][-1]['w']+p['layers'][-1]['b'])[...,0]


def initialize_codes(encoder_fields,latent):
    """Training-only PCA seeds latent coordinates, not decoder output functions."""
    center=jnp.mean(encoder_fields,axis=0);X=encoder_fields-center
    _,s,V=jnp.linalg.svd(X,full_matrices=False)
    scale=jnp.maximum(s[:latent]/jnp.sqrt(len(X)),1e-8)
    basis=V[:latent].T
    return (X@basis)/scale,dict(center=center,basis=basis,scale=scale)


def nearest_starts(field,training_fields,codes):
    distance=jnp.mean((training_fields-field[None])**2,axis=1)
    ids=jnp.argsort(distance)[:8]
    weight=1/jnp.maximum(distance[ids],1e-12);weight=weight/weight.sum()
    return codes[ids[0]],jnp.sum(codes[ids]*weight[:,None],axis=0)


def code_jacobian(p, z, xy):
    """Scalar-output reverse differentiation independently at each point.

    Avoid a full-field jacrev, whose output dimension is the spatial grid size.
    The online root code is one vector; arbitrary point batches are flattened.
    """
    assert z.ndim == 1 and xy.shape[-1] == 2
    points=xy.reshape(-1,2)
    J=jax.vmap(jax.grad(apply,argnums=1),in_axes=(None,None,0))(p,z,points)
    return J.reshape(xy.shape[:-1]+(len(z),))
