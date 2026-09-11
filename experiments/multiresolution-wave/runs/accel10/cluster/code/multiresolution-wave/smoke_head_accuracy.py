"""Bounded normalized phase-loss and parameter-gradient smoke."""
import json
import numpy as np
import head_accuracy as h
from fresh_models import head_init

rng=np.random.default_rng(691211);linear=np.linalg.qr(rng.normal(size=(64,32)))[0]*.02
p,f=head_init(h.jax.random.PRNGKey(691211),linear,np.zeros(64),.03,'mlp',width=128)
z=h.jnp.asarray(rng.normal(size=(3,32)));w=h.jnp.asarray(rng.normal(size=(3,32)))
a=z@h.jnp.asarray(linear.T)+.001;b=w@h.jnp.asarray(linear.T)-.003
batch=(z,w,a,b,h.jnp.array([.1,.2,.3]),h.jnp.array([1.,2.,3.]),h.jnp.array([.9,1.,1.1]))
k=h.jnp.diag(h.jnp.linspace(1,100,64))
def loss(p):return h.jnp.sum(h.loss_components(p,f,*batch,k))
value,gradient=h.jax.jit(h.jax.value_and_grad(loss))(p)
delta=1e-5
def shift(amount):
 q={**p,'out':{**p['out'],'w':p['out']['w'].at[0,0].add(amount)}}
 return float(loss(q))
fd=(shift(delta)-shift(-delta))/(2*delta);ad=float(gradient['out']['w'][0,0])
assert np.isfinite(float(value)) and abs(fd-ad)<1e-8*max(1,abs(ad))
components=np.asarray(h.loss_components(p,f,*batch,k))
assert np.all(components>0)
print(json.dumps(dict(components=components.tolist(),objective=float(value),gradient_finite_difference=fd,gradient_autodiff=ad,gradient_absolute_difference=abs(fd-ad)),indent=2))
