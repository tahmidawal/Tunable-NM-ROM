from campaign import jax,jnp
import local_bank as B

def encode(x):
 f=jnp.array([1.,2.,4.,8.]);return jnp.concatenate([x,jnp.sin(jnp.pi*x[:,:,None]*f).reshape(len(x),-1),jnp.cos(jnp.pi*x[:,:,None]*f).reshape(len(x),-1)],axis=1)
def apply(p,x):
 x=encode(x)
 for w,b in p[:-1]:x=jnp.tanh(x@w+b)
 return x@p[-1][0]+p[-1][1]
def features(p,points,patches,k):
 ps=jnp.asarray(patches);z=(points[None]-ps[:,None,:2])/ps[:,None,2:3]
 v=apply(p,z.reshape(-1,2))[:,:k].reshape(len(patches),len(points),k)
 windows=jax.vmap(lambda p:B.window(points,p))(ps)
 return (v*windows[:,:,None]).transpose(1,0,2).reshape(len(points),-1)
