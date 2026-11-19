from common import *
from neural import fields as F,factorized as S
import data

xy,w=data.grid(17,True);xy=jnp.asarray(xy);w=jnp.asarray(w)
keys=jax.random.split(jax.random.PRNGKey(72),3)
p=dict(root=F.init_mlp(keys[0],[10,12,12,1],True),feature=F.init_mlp(keys[1],[2,24,24,32],True),
       head=F.init_mlp(keys[2],[16,16,16,16]),scale=jnp.ones(32))
cells=jnp.array([[1,0,0],[1,1,1]]);z=.02*jax.random.normal(keys[0],(40,),dtype=jnp.float64)
zero=z.at[8:].set(0);change=float(jnp.max(abs(S.tree(p,zero,xy,cells)-F.root(p['root'],zero[:8],xy))));assert change==0
pp=dict(p,B=S.feature_matrix(p['feature'],cells,xy))
block=S.cached_jacobian(pp,z,xy);full=jax.jacfwd(lambda zz:S.cached(pp,zz,xy))(z)
derivative_error=float(jnp.max(abs(block-full)));assert derivative_error<1e-12
c=S.coefficients(p['head'],z[8:24],p['scale']);assert jnp.max(abs(c[:16]-z[8:24]))==0
q,_=jnp.linalg.qr(jnp.sqrt(w)[:,None]*block[:,:8]);j=jnp.sqrt(w)[:,None]*block[:,8:];j=j-q@(q.T@j)
s=jnp.linalg.svd(j,compute_uv=False);assert s[-1]/s[0]>1e-6
target=S.cached(pp,z,xy);norm=jnp.sqrt(jnp.sum(w*target**2))
fit=F.make_fit(S.cached,60,jacobian=S.cached_jacobian)
returned,status=fit(pp,z+.001,xy,target,w,norm)
assert status[-1]==1 and status[0]<1e-7
print(json.dumps(dict(zero_child_change=change,block_jacobian_error=derivative_error,
    useful_child_directions=32,minimum_relative_singular=float(s[-1]/s[0]),
    exact_leading_coordinates=True,fit_status=np.asarray(status).tolist())))
