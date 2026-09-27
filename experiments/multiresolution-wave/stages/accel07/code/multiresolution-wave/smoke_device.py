"""Short actual-checkpoint check: device contract preserves the physical query."""
import json
from pathlib import Path
import time
import numpy as np
import device_replay as replay
from fresh_models import tree_from_npz

start = time.perf_counter()
cell = Path(__file__).resolve().parent
cfg = json.loads((cell/'device-replay-config.json').read_text())
cfg.update(end_time=.005, observation_dt=.005, fit_iterations=4)
prior = cell/'runs/k32heads03/cluster'
bc = 'dirichlet'; grid = replay.Grid(32, bc, bc)
bank = replay.base.rebuild(prior/'in'/bc, grid)
with np.load(prior/f'out/pilot/training_ladder_{bc}.npz') as f:
    linear, center = f['standardized_linear'][:, :32], f['center']
head = tree_from_npz(prior/f'out/pilot/head_{bc}_new_mlp32_seed691200/head.npz')
model = replay.adapt_model(bank, head, linear, center, cfg['primary_method'])
par = replay.parameter_rows(cfg['validation_seed'], 2)[0]
u0, v0 = replay.localized_initial(grid, par)
supplied = (u0, v0, replay.jnp.asarray(par[5])); replay.jax.block_until_ready(supplied)
u, v, row, aux = replay.device_query(cfg['primary_method'], .0025, supplied,
    float(par[5]), grid, cfg, replay.numerical_bank(model))
hu, hv, hostrow, hostaux = replay.base.query('rom', .0025, np.asarray(u0), np.asarray(v0), par[5], grid, cfg, model)
np.testing.assert_allclose(u, hu, atol=1e-11, rtol=1e-10)
np.testing.assert_allclose(v, hv, atol=1e-10, rtol=1e-9)
assert row['completed'] and row['output_bytes']==2*8*2*31*31
assert abs(sum(row['seconds'][k] for k in ('initialization_and_parameter_projection', 'evolution', 'dense_device_output'))-row['seconds']['complete_device_query'])<1e-12
for bc in ('dirichlet', 'absorbing'):
    grid = replay.Grid(16, bc, bc)
    u0, v0 = replay.localized_initial(grid, par)
    supplied = (u0, v0, replay.jnp.asarray(par[5])); replay.jax.block_until_ready(supplied)
    method, setting = ('dst', 0.) if bc=='dirichlet' else ('rk4', .45)
    u, v, row, _ = replay.device_query(method, setting, supplied, float(par[5]), grid, cfg)
    hu, hv, _, _ = replay.base.query(method, setting, np.asarray(u0), np.asarray(v0), par[5], grid, cfg)
    np.testing.assert_array_equal(u, hu); np.testing.assert_array_equal(v, hv)
    assert row['completed']
print(json.dumps(dict(smoke_passed=True, elapsed_seconds=time.perf_counter()-start,
    jax_backend=replay.jax.default_backend(), x64=replay.jax.config.jax_enable_x64,
    matmul_precision=replay.jax.config.jax_default_matmul_precision)), flush=True)
