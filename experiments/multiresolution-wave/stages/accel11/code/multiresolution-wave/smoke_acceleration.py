"""Bounded derivative/guard/RK parity smoke. No scientific timing claims."""
import json
from pathlib import Path
import numpy as np
import acceleration as a
import iterative_replay as previous
from fresh_rom import weak_acceleration

cell=Path(__file__).resolve().parent
full,models=previous.load_models(cell/'runs/iterative05/cluster/in/dirichlet',a.base.Grid(12))
bank=models['frozen_mlp32_seed691200'];p,f=bank['p'],bank['frozen']
z=bank['fixed_codes'][2];w=a.jnp.asarray(np.random.default_rng(709111).normal(size=32))
result=a.geometry_checks(bank,z,w)
aa=a.jax.jit(lambda p,f,z,w,k,d:weak_acceleration(p,f,z,w,k,d,'mlp'))(p,f,z,w,bank['k'],bank['d'])
checks={}
for variant in ('shared_svd_r','qr_guard','chol_guard'):
    bb=a.jax.jit(lambda p,f,z,w,k,d:a.accelerated_rhs(p,f,z,w,k,d,variant))(p,f,z,w,bank['k'],bank['d'])
    np.testing.assert_allclose(aa[0],bb[0],atol=1e-9,rtol=1e-10)
    assert bool(aa[3])==bool(bb[3])
    checks[variant]=dict(acceleration_relative=float(np.linalg.norm(np.asarray(aa[0]-bb[0]))/np.linalg.norm(np.asarray(aa[0]))),rank_diagnostic=float(bb[2]),fallback=int(bb[4]))
for ratio in (1e-2,1e-7,1e-9):
    jac=a.jnp.asarray(np.diag(np.geomspace(1.,ratio,32)))
    force=a.jnp.ones(32)
    for solve in (lambda:a.qr_solve(jac,force,True),lambda:a.cholesky_solve(jac,force)):
        acc,observed,fallback=solve()
        assert bool(observed>1e-8)==bool(ratio>1e-8)
        if ratio<=1e-7:assert int(fallback)==1
result['rhs_checks']=checks;result['guard_threshold_cases_passed']=True
control=a.base.rollout(p,f,z,w,bank['k'],bank['d'],.0001,kind='mlp',steps=2,stride=1)
trial=a.rollout(p,f,z,w,bank['k'],bank['d'],.0001,variant='chol_guard',steps=2,stride=1)
for key in ('z','w','outflux'):
    np.testing.assert_allclose(control[key],trial[key],atol=1e-10,rtol=1e-10)
np.testing.assert_array_equal(control['completed'],trial['completed'])
result['rk4_two_step_parity_passed']=True
result['maximum_scaled_normal_backward_error']=float(trial['normal_backward_error'][-1])
print(json.dumps(result,indent=2))
