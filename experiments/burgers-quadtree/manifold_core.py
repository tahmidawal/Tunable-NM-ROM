"""Residual POD autoencoder and damped least-squares latent optimization."""
from campaign import *
def init(key,dims):
 return [(jax.random.normal(k,(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b)),jnp.zeros(b)) for k,a,b in zip(jax.random.split(key,len(dims)-1),dims[:-1],dims[1:])]
def mlp(p,x):
 for w,b in p[:-1]:x=jnp.tanh(x@w+b)
 return x@p[-1][0]+p[-1][1]
def decode(p,z):return jnp.pad(z,[(0,0)]*(z.ndim-1)+[(0,p[-1][1].shape[0]-z.shape[-1])])+.1*mlp(p,z)
def encode(p,x,latent):return x[...,:latent]+.1*mlp(p,x)
def lm(residual,z,iterations=60):
 def body(_,carry):
  z,lam=carry;r=residual(z);J=jax.jacfwd(residual)(z);g=J.T@r
  dz=jnp.linalg.solve(J.T@J+lam*jnp.eye(z.size),-g);trial=z+dz
  accept=jnp.sum(residual(trial)**2)<jnp.sum(r*r)
  return jnp.where(accept,trial,z),jnp.clip(jnp.where(accept,lam/3,lam*5),1e-12,1e8)
 z,lam=jax.lax.fori_loop(0,iterations,body,(z,jnp.array(1e-3)))
 r=residual(z);J=jax.jacfwd(residual)(z)
 return z,jnp.linalg.norm(J.T@r,ord=jnp.inf),jnp.linalg.norm(r)
