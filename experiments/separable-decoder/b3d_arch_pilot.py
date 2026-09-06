"""Run the inherited validation pilot with an actual architecture checkpoint.

The bank stays unchanged; q coordinates are mapped back by h=R^{-1}q.
Both production and independent NumPy heads are adapted. Restricted to PILOT=1
because the inherited rollout/export path assumes the old MLP checkpoint schema.
"""
import importlib
import os
from pathlib import Path
import pickle

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

import b3d_common as b3


def adapt(payload, module):
    assert payload['kind'] == 'b3d_arch_checkpoint'
    params = dict(payload['bank_params'])
    params.update(arch_trainable=payload['trainable'], arch_frozen=payload['frozen'],
                  arch_to_bank=np.linalg.solve(payload['r'], np.eye(len(payload['r']))).T)
    params = jax.tree_util.tree_map(jnp.asarray, params)

    def head(p, z):
        return module.apply(p['arch_trainable'], p['arch_frozen'], z) @ p['arch_to_bank']

    def head_np(p, z):
        return module.apply_np(p['arch_trainable'], p['arch_frozen'], np.asarray(z)) @ np.asarray(p['arch_to_bank'])

    return params, head, head_np


def main():
    assert os.environ.get('PILOT') == '1' and os.environ.get('TRAIN') == '0', 'pilot-only adapter'
    assert os.environ.get('GATES_SOFT', '0') == '0', 'production gates must remain enabled'
    path = Path(os.environ['CKPT']).resolve()
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    name = payload['model']
    assert name.startswith('b3d_arch_') and name.replace('_', '').isalnum()
    module = importlib.import_module(name)
    assert b3.sha256_file(module.__file__) == payload['source']['model_sha256'], 'model source changed'
    params, head, head_np = adapt(payload, module)
    probe = jnp.asarray(payload['codes'][:8])
    np.testing.assert_allclose(head(params, probe), head_np(params, probe), rtol=1e-11, atol=1e-11)
    cfg = dict(payload['cfg'], architecture_model=name,
               architecture_source_commit=payload['source']['commit'],
               architecture_model_sha256=payload['source']['model_sha256'])
    original_load, original_head, original_np = b3.load_pkl, b3.head, b3.head_np

    def load(checkpoint):
        assert Path(checkpoint).resolve() == path, 'unexpected checkpoint load'
        return params, payload['codes'], cfg

    try:
        b3.load_pkl, b3.head, b3.head_np = load, head, head_np
        driver = importlib.import_module('sep_b3d_tensor')
        assert driver.PILOT == 1 and driver.TRAIN == 0 and Path(driver.CKPT).resolve() == path
        driver.main()
    finally:
        b3.load_pkl, b3.head, b3.head_np = original_load, original_head, original_np


if __name__ == '__main__':
    main()
