import core as C
from core import np,jnp,jax,P
Q=P.load_qcore();rng=np.random.default_rng(41)
d=dict(A=jnp.asarray(rng.normal(size=(32,8))),lam=jnp.arange(1,33.),Gq=jnp.asarray(rng.normal(size=(64,8))*.01),Gs=jnp.asarray(rng.normal(size=(64,8))*.01),Psi=jnp.asarray(rng.normal(size=(64,32))*.01),hmask=jnp.ones(32))
nl,nj=Q.tested_value('point',16),Q.tested_jac('point',16)
f=C.step_factory(nl,nj,8,4,1.,'twolevel')
x=jnp.ones(8)*.01;args=((d['A']@x,d['A']@x,jnp.zeros(32)),.01,d,jnp.asarray(P.BE_CO),.005)
out=jax.jit(f['run'])(x,args,0.,1e-6,1e-6);jax.block_until_ready(out)
r,J=f['ej'](x,args);r2,J2=f['cej'](jnp.zeros(4),(args,x))
assert np.allclose(J2,J[:,:4],rtol=1e-12,atol=1e-12)
assert float(out[0][4])<=1e-6
# Build transformation, without executing bank decoder on synthetic data.
C.make_query(Q,8,4,16,1.,'twolevel')
print('SMOKE PASS gpu',jax.default_backend(),'gamma',float(out[0][4]))
