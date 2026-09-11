"""Bounded generic-K geometry and RK parity check at K40/M64."""
import json
import numpy as np
import acceleration as a
from fresh_models import head_init
from fresh_rom import weak_acceleration

rng=np.random.default_rng(691211);linear=np.linalg.qr(rng.normal(size=(64,40)))[0]*.03
p,f=head_init(a.jax.random.PRNGKey(691211),linear,np.zeros(64),.03,'mlp',width=128)
p={**p,'out':{**p['out'],'w':a.jnp.asarray(rng.normal(size=(128,64))*.001)}}
z=a.jnp.asarray(rng.normal(size=40));w=a.jnp.asarray(rng.normal(size=40));k=a.jnp.diag(a.jnp.linspace(10,1000,64));d=a.jnp.zeros((64,64))
control=a.base.rollout(p,f,z,w,k,d,.0001,kind='mlp',steps=2,stride=1)
trial=a.rollout(p,f,z,w,k,d,.0001,variant='chol_guard',steps=2,stride=1)
for name in ('z','w','outflux'):np.testing.assert_allclose(control[name],trial[name],atol=1e-10,rtol=1e-10)
assert bool(a.jnp.all(trial['completed']))
error=float(np.linalg.norm(np.asarray(control['w']-trial['w']))/np.linalg.norm(np.asarray(control['w'])))
print(json.dumps(dict(latent=40,weak_equations=64,relative_velocity_code_parity=error,maximum_backward_error=float(trial['normal_backward_error'][-1])),indent=2))
