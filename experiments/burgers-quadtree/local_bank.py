"""Physical compact supports stay intact: only coefficients are fitted globally."""
from campaign import jax,jnp,np

def children(p):
 x,y,h=p;return [(x+i*h/2,y+j*h/2,h/2) for i in range(2) for j in range(2)]
def uniform(depth):
 return [(i/2**depth,j/2**depth,1/2**depth) for i in range(2**depth) for j in range(2**depth)]
def window(xy,p,alpha=.25):
 p=jnp.asarray(p);z=(xy-p[:2])/p[2]
 d=jnp.maximum(jnp.maximum(-z,z-1),0)/alpha
 t=jnp.clip(d,0,1);s=1-10*t**3+15*t**4-6*t**5
 # Fixed factor for exact physical walls, including corner columns.
 return jnp.prod(s,axis=-1)*jnp.sin(jnp.pi*xy[...,0])*jnp.sin(jnp.pi*xy[...,1])
def init(key,width=128,k=32):
 dims=[2,width,width,width,k];keys=jax.random.split(key,4)
 return [(jax.random.normal(q,(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b)),jnp.zeros(b)) for q,a,b in zip(keys,dims[:-1],dims[1:])]
def network(params,z):
 for w,b in params[:-1]:z=jnp.tanh(z@w+b)
 w,b=params[-1];return z@w+b

def features(xy,patches,k,kind,params=None,pod=None):
 blocks=[]
 for p in patches:
  z=(xy-jnp.array(p[:2]))/p[2]
  if kind=='neural':v=network(params,2*z-1)[:,:k]
  elif kind=='poly':
   powers=sorted([(i,j) for i in range(8) for j in range(8)],key=lambda q:(sum(q),q[0]))[:k]
   v=jnp.stack([(2*z[:,0]-1)**i*(2*z[:,1]-1)**j for i,j in powers],axis=1)
  elif kind=='pod':
   from jax.scipy.ndimage import map_coordinates
   coords=((z+.25)/1.5*(pod.shape[0]-1)).T
   v=jnp.stack([map_coordinates(pod[:,:,i],coords,order=1,mode='constant',cval=0.) for i in range(k)],axis=1)
  else:raise ValueError(kind)
  blocks.append(v*window(xy,p)[:,None])
 return jnp.concatenate(blocks,axis=1)
def fit(B,U,cutoff=1e-10):
 u,s,vh=jnp.linalg.svd(B,full_matrices=False)
 keep=s>cutoff*s[0]
 coeff=vh.T@((u.T@U)*jnp.where(keep,1/jnp.maximum(s,1e-300),0)[:,None])
 return coeff,s,jnp.sum(keep)

def smoke():
 p=(.25,.25,.25);f=lambda x:window(x,p)
 pts=jnp.array([[.1875,.3],[.25,.3],[.5,.3],[.5625,.3],[0,.3],[1,.3]])
 vals=jax.vmap(f)(pts);gr=jax.vmap(jax.grad(f))(pts);hs=jax.vmap(jax.hessian(f))(pts)
 assert np.isfinite(hs).all()
 assert abs(vals[0])+abs(vals[3])<1e-14
 assert np.max(np.abs(np.asarray(gr)[[0,3]]))<1e-10
 assert np.max(np.abs(np.asarray(hs)[[0,3]]))<1e-8
 xy=jnp.array(np.random.default_rng(3).uniform(0,1,(64,2)))
 B=features(xy,[p],4,'poly');C=features(xy,[p]+children(p),4,'poly')
 c=jnp.arange(4.);assert np.allclose(B@c,C@jnp.pad(c,(0,16)),atol=1e-14)
 return dict(window_value=vals,window_gradient=gr,window_hessian=hs,parent_child_zero_transfer=True)
