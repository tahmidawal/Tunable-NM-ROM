"""Sub-minute shape and nonzero-residual implicit-velocity smoke."""
from pathlib import Path
import json
import numpy as np
import modal_projection as m
import iterative_replay as previous
from fresh_models import head_apply

cell=Path(__file__).resolve().parent
_,models=previous.load_models(cell/'runs/iterative05/cluster/in/dirichlet',m.base.Grid(12))
bank=models['frozen_mlp32_seed691200'];prepared=m.prepare(bank,'h1')
z=bank['fixed_codes'][2];p,f=bank['p'],bank['frozen'];W=prepared['weight']
fun=lambda x:head_apply(p,f,x,'mlp')
a=fun(z);jac=m.jax.jacfwd(fun)(z);rng=np.random.default_rng(691113)
q,_=np.linalg.qr(np.asarray(W@jac),mode='reduced');normal=rng.normal(size=64);normal-=q@(q.T@normal)
target=a+m.jnp.linalg.solve(W,m.jnp.asarray(normal*1e-5));velocity=m.jnp.asarray(rng.normal(size=64)*.01)
actual=m.jax.jit(m.implicit_velocity)(p,f,W,z,target,velocity)
def gradient(zz,t):return m.jax.grad(lambda x:.5*m.jnp.sum((W@(fun(x)-t))**2))(zz)
step=1e-5;w=actual[2]
derivative=(gradient(z+step*w,target+step*velocity)-gradient(z-step*w,target-step*velocity))/(2*step)
scale=m.jnp.linalg.norm(jac.T@W.T@W@velocity)
relative=float(m.jnp.linalg.norm(derivative)/scale)
assert relative<1e-6 and float(actual[3])>0
fit=m.project_targets(bank,prepared,m.jnp.stack([a,a]),m.jnp.stack([velocity,velocity]),.1,iterations=2)
assert fit[0].shape==(2,64) and fit[2].shape==(2,32)
print(json.dumps(dict(nonzero_normal_residual_implicit_gradient_derivative_relative=relative,
                     hessian_min_eigenvalue=float(actual[3]),backward_error=float(actual[-1]),batched_projection_shape_passed=True),indent=2))
