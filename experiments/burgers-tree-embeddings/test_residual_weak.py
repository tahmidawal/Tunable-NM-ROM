"""Small CPU equivalence fixture, not scientific PDE trajectory admission."""
from common import *
from neural import stable as S, residual_tree as R, residual_weak as C, weak as W

L=64; Q=16; M=64
root=S.initialize(jax.random.PRNGKey(611), latent=4, width=8, depth=2)
p=R.initialize(jax.random.PRNGKey(612),root,child_k=2,features=8,width=8)
cells=jnp.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
cache=C.prepare(p,cells,L,Q,M)
z=.2*jax.random.normal(jax.random.PRNGKey(613),(12,),dtype=jnp.float64)
apply=lambda p,z,xy:R.tree(p,z,xy,cells)
gm,ga,gp,gr=W.make_operators(apply,L,Q)
cm,ca,cp,cr=C.make_operators(L,Q)
jm,ja,jr=C.make_jacobians(L,Q)
dt=1e-6;nu=.005;norm=jnp.linalg.norm(C.field(p,z,cache))/Q
old=gp(p,z,cache,dt,nu);trial=z+.001
errors=dict(mass=float(jnp.max(abs(cm(p,z,cache)-gm(p,z,cache)))),
    advection=float(jnp.max(abs(ca(p,z,cache)-ga(p,z,cache)))),
    previous=float(jnp.max(abs(cp(p,z,cache,dt,nu)-old))),
    residual=float(jnp.max(abs(cr(p,trial,old,cache,dt,nu,norm)-gr(p,trial,old,cache,dt,nu,norm)))))
for name,explicit,original in [('mass_jac',jm,gm),('advection_jac',ja,ga)]:
    errors[name]=float(jnp.max(abs(explicit(p,z,cache)-jax.jacfwd(original,argnums=1)(p,z,cache))))
errors['residual_jac']=float(jnp.max(abs(jr(p,trial,old,cache,dt,nu,norm)
    -jax.jacfwd(gr,argnums=1)(p,trial,old,cache,dt,nu,norm))))
assert max(errors.values())<1e-11,errors
zero=z.at[4:].set(0.)
assert float(jnp.max(abs(C.field(p,zero,cache)-S.apply(root,zero[:4],cache['xy']))))==0.
# A zero field exercises the c==0 branch explicitly, with nonzero child tangents.
root_zero=jax.tree.map(lambda x:x,root)
root_zero=dict(root_zero,layers=list(root_zero['layers']))
root_zero['layers'][-1]=jax.tree.map(jnp.zeros_like,root_zero['layers'][-1])
pzero=dict(p,root=root_zero)
errors['zero_branch_jac']=float(jnp.max(abs(ja(pzero,jnp.zeros_like(z),cache)
    -jax.jacfwd(ga,argnums=1)(pzero,jnp.zeros_like(z),cache))))
assert errors['zero_branch_jac']<1e-11
step=C.make_step(L,Q)
returned,status=step(p,z,cache,dt,nu)
tight,tight_status=step(p,z,cache,dt,nu,1e-10,1e-12)
assert status[-1]==1 and tight_status[-1]==1,(status,tight_status)
change=float(jnp.linalg.norm(C.field(p,returned,cache)-C.field(p,tight,cache))/jnp.linalg.norm(C.field(p,z,cache)))
assert change<1e-8,change
print(json.dumps(dict(errors=errors,status=np.asarray(status).tolist(),
    tight_status=np.asarray(tight_status).tolist(),tight_field_change=change,
    zero_child_preservation=True,cache_reuses_child_features=True)))
