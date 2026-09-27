"""Actual frozen-bank small-mesh K32 pipeline smoke; no scientific training."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import heads32 as h
from fresh_models import head_init

start=time.perf_counter();cell=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=cell/'runs/dynamics02/cluster/in');args=ap.parse_args()
inputs=args.inputs/'dirichlet';cfg=json.loads((cell/'heads32-config.json').read_text())
cfg.update(end_time=.01,observation_dt=.01,diagnostic_indices=[0,1],diagnostic_fit_budgets=[2,4])
grid=h.Grid(64);bank=h.base.rebuild(inputs,grid)
rng=np.random.default_rng(7090743);linear=np.linalg.qr(rng.normal(size=(64,32)))[0]*.01
with np.load(inputs/'coordinates.npz') as f:center=f['common_center']
p,frozen=head_init(h.jax.random.PRNGKey(7090744),linear,center,.02,'mlp',width=16)
model=h.adapt_model(bank,dict(p=p,frozen=frozen,codes=np.zeros((12,32))),linear,center,'smoke_mlp32')
par=h.parameter_rows(690602,2)[1];u0,v0=map(np.asarray,h.localized_initial(grid,par))
u,v,record,aux=h.base.query('rom',.0005,u0,v0,par[5],grid,cfg,model)
assert record['completed'] and u.shape==(2,63,63) and aux['rollout']['z'].shape==(2,32)
basis=h.jnp.asarray(bank['transform'])@h.jnp.asarray(linear);offset=h.jnp.asarray(bank['transform'])@h.jnp.asarray(center)
generator=h.d.affine_generator(basis,offset,bank['k'],bank['d'],par[5])
states,_=h.d.affine_evolve(generator,h.jnp.asarray(aux['rollout']['z'][0]),h.jnp.asarray(aux['rollout']['w'][0]),.01,observations=2)
a,b=h.d.affine_coefficients(states,basis,offset)
np.testing.assert_allclose(aux['coefficients'],a,atol=1e-7)
np.testing.assert_allclose(aux['velocity_coefficients'],b,atol=2e-5)
truth=h.base.spectral_propagate(h.jnp.asarray(u0),h.jnp.asarray(v0),par[5],h.jnp.array([0.,.01]))
with tempfile.TemporaryDirectory() as temp:
    diag=h.d.fitted_diagnostics(model,[],*map(np.asarray,truth),grid,par[5],cfg,Path(temp)/'fit.npz')
    assert diag['arms'][0]['arm']=='smoke_mlp32'
print('actual_bank_k32_pipeline_smoke_passed',time.perf_counter()-start,flush=True)
