from common import *
from neural import stable as S,fields as F
import data,optax
xy,w=data.grid(17,True);xy=jnp.asarray(xy);w=jnp.asarray(w)
p=S.initialize(jax.random.PRNGKey(81),latent=32,width=128,depth=4)
z=.2*jax.random.normal(jax.random.PRNGKey(82),(32,),dtype=jnp.float64)
u=S.apply(p,z,xy);J=jax.jacfwd(S.apply,argnums=1)(p,z,xy)
s=jnp.linalg.svd(jnp.sqrt(w)[:,None]*J,compute_uv=False)
assert s[-1]/s[0]>1e-6
walls=jnp.array([[0.,.2],[1.,.3],[.6,0.],[.9,1.]])
assert jnp.max(abs(S.apply(p,z,walls)))==0
batched=S.apply(p,jnp.stack([z,z+.01])[:,None,:],jnp.stack([xy,xy]))
assert jnp.max(abs(batched[0]-u))<1e-13
assert jnp.linalg.norm(S.apply(p,z+.1,xy)-u-J@jnp.full(32,.1))>1e-6
fit=F.make_fit(S.apply,60);zz,st=fit(p,z+.001,xy,u,w,jnp.linalg.norm(u*jnp.sqrt(w)))
assert st[-1]==1 and st[0]<1e-7,st
print(json.dumps(dict(zero_boundary=True,nonlinear_embedding=True,independent_directions=32,
    minimum_relative_singular=float(s[-1]/s[0]),fit_status=np.asarray(st).tolist())))
