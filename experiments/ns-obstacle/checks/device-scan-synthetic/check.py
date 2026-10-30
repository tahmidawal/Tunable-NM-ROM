"""Synthetic GPU reduction equivalence only; no flow data or timing claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
assert jax.default_backend() == 'gpu'
print('jax_backend=gpu', flush=True)
helper = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle-rom/experiments/ns-obstacle-rom/accuracy_v2/device_scan_objective.py')
expected = 'c8ea231421fde3706f60b1df6bc2508cf3407b6d6ae0078c2414c362a2d33e0a'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == expected
spec = importlib.util.spec_from_file_location('scan_helper', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def objective(theta, block):
    x, y, w = block
    pred = jnp.tanh(x @ theta[:8].reshape(2, 4) + theta[8:12]) @ theta[12:20].reshape(4, 2) + theta[20:22]
    squared = jnp.sum((pred-y)**2, axis=1)
    return jnp.sum(w*squared), {'weight': jnp.sum(w), 'prediction_sum': jnp.sum(w[:, None]*pred, axis=0)}

vg = jax.value_and_grad(objective, has_aux=True)
block_vg = jax.jit(vg)
scanned = jax.jit(lambda theta, blocks: module.reduce_blocks(theta, blocks, vg))

def host_sum(theta, blocks):
    results = [block_vg(theta, block) for block in blocks]
    return jax.tree_util.tree_map(lambda *xs: sum(xs), *results)

rng = np.random.default_rng(30261030)
arrays = {}
gaps = []
for count in (1, 7, 19):
    x = rng.normal(size=(count, 2)); y = rng.normal(size=(count, 2)); w = rng.uniform(.1, 1, count)
    pad = (-count) % 7
    stacked = tuple(jnp.asarray(np.pad(a, ((0,pad),) + ((0,0),)*(a.ndim-1))).reshape((-1,7)+a.shape[1:]) for a in (x,y,w))
    unpadded = [tuple(jnp.asarray(a[start:start+7]) for a in (x,y,w)) for start in range(0,count,7)]
    padded = [tuple(a[i] for a in stacked) for i in range(stacked[0].shape[0])]
    arrays[f'n{count}_x'] = x; arrays[f'n{count}_y'] = y; arrays[f'n{count}_w'] = w
    for trial in range(3):
        theta = jnp.asarray(rng.normal(size=22)*.2)
        arrays[f'n{count}_t{trial}_theta'] = np.asarray(theta)
        outputs = [host_sum(theta, unpadded), host_sum(theta,padded), scanned(theta,stacked)]
        for name, output in zip(('unpadded','padded','scan'),outputs):
            (value, metrics), gradient = output
            for key, value in dict(value=value, gradient=gradient, **metrics).items():
                arrays[f'n{count}_t{trial}_{name}_{key}'] = np.asarray(value)
        for alternative in outputs[1:]:
            for a,b in zip(jax.tree_util.tree_leaves(outputs[0]),jax.tree_util.tree_leaves(alternative)):
                gaps.append(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
out = Path(__file__).parent
# Retain raw results before numerical assertions, including on failure.
np.savez(out/'arrays.npz', **arrays)
result = dict(scope='synthetic reduction equivalence, no performance or flow accuracy claim', seed=30261030, counts=[1,7,19], block_size=7, trials=3, helper_sha256=expected, jax_version=jax.__version__, devices=[str(d) for d in jax.devices()], f64=True, matmul_precision=str(jax.config.jax_default_matmul_precision), max_absolute_gap=max(gaps), all_finite=all(np.isfinite(a).all() for a in arrays.values()), arrays_sha256=hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest())
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['all_finite'] and result['max_absolute_gap'] < 1e-11, result
print(json.dumps(result), flush=True)
