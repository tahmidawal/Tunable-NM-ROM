"""Unit-scale latent training and unconstrained global offsets; frozen physical scaling."""
from common import *
import decoder as dec

def head(params,z,mean,scale):
    return mean+scale*(z@params['linear'].T+dec.mlp(params['net'],z)+params['bias'])

def encode(params,C,mean,scale):
    y=(C-mean)/scale
    return y@params['linear']+dec.mlp(params['net'],y)

def initialize(C,norm,k,key,width=128,hidden=None):
    weight=1/norm**2;mean=jnp.sum(C*weight[:,None],axis=0)/jnp.sum(weight)
    center=C-mean;_,_,vh=jnp.linalg.svd(center/norm[:,None],full_matrices=False)
    V=vh[:k].T;initial=center@V;latentstd=jnp.maximum(jnp.std(initial,axis=0),1e-8)
    scale=jnp.std(center,axis=0);scale=jnp.maximum(scale,jnp.max(scale)*1e-3)
    k1,k2=jax.random.split(key)
    hidden=[width,width,width] if hidden is None else list(hidden)
    hp=dict(linear=V*latentstd[None]/scale[:,None],bias=jnp.zeros(C.shape[1]),
            net=dec.init_mlp(k1,[k,*hidden,C.shape[1]],.01))
    ep=dict(linear=scale[:,None]*V/latentstd[None],net=dec.init_mlp(k2,[C.shape[1],256,128,k],.01))
    return dict(head=hp,encoder=ep,codes=initial/latentstd[None]),mean,scale
