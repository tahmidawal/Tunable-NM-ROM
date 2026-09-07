"""Check each actual new head through the inherited decoder's pilot adapter.

Read models from this tree after a merge, or their isolated experiment trees.
The test never writes to those trees. Independent NumPy finite differences use
a nonorthogonal field bank and nonzero nonlinear weights.
"""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import b3d_arch_pilot as pilot
import b3d_common as b3

SRC = Path(__file__).resolve().parent
WORKTREES = SRC.parents[2]


def setUpModule():
    print('jax_backend=' + jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu' and jax.config.x64_enabled
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'


def check_arm(arm):
    path = SRC/f'b3d_arch_{arm}.py'
    if not path.exists():
        path = WORKTREES/f'2026-09-06-b3d-{arm}'/'experiments/separable-decoder'/path.name
    spec = importlib.util.spec_from_file_location(f'b3d_integration_{arm}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(847)
    g = rng.normal(size=(17, 5)) @ np.diag([.2, .5, 1., 2., 3.])
    q, r = np.linalg.qr(g, mode='reduced')
    u, _ = np.linalg.qr(rng.normal(size=(5, 2)), mode='reduced')
    shared = dict(anchor=u, center=rng.normal(size=5), latent_scale=np.asarray([.6, 1.4]),
                  output_scale=np.asarray(.8))
    p, frozen = module.init(jax.random.PRNGKey(401), shared)
    # Activate every trainable branch; frozen anchor/scales remain untouched.
    p = jax.tree_util.tree_map(lambda a: a + jnp.asarray(rng.normal(scale=.02, size=a.shape)), p)
    payload = dict(kind='b3d_arch_checkpoint', bank_params={}, trainable=p, frozen=frozen, r=r)
    adapted, head, head_np = pilot.adapt(payload, module)
    z = np.asarray([.17, -.23])
    expected = lambda zz: q @ module.apply_np(p, frozen, zz)
    with patch.object(b3, 'head', head):
        decoder = b3.SeparableDecoder3D(adapted, 2, 5)
        actual = lambda zz: jnp.asarray(g) @ decoder.head_fn()(zz)
        np.testing.assert_allclose(actual(jnp.asarray(z)), expected(z), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(head_np(adapted, z), head(adapted, z), rtol=1e-12, atol=1e-12)
        jac = np.asarray(jax.jacfwd(actual)(jnp.asarray(z)))
        hess = np.asarray(jax.jacfwd(jax.jacfwd(actual))(jnp.asarray(z)))
    eye, eps = np.eye(2), 2e-5
    independent_jac = np.column_stack([(expected(z+eps*v)-expected(z-eps*v))/(2*eps) for v in eye])
    np.testing.assert_allclose(jac, independent_jac, rtol=2e-7, atol=2e-8)
    eps = 3e-4
    independent_hess = np.empty_like(hess)
    for i, vi in enumerate(eye):
        for j, vj in enumerate(eye):
            independent_hess[:, i, j] = (expected(z+eps*vi+eps*vj)-expected(z+eps*vi-eps*vj)
                -expected(z-eps*vi+eps*vj)+expected(z-eps*vi-eps*vj))/(4*eps**2)
    np.testing.assert_allclose(hess, independent_hess, rtol=2e-5, atol=4e-7)
    assert np.linalg.norm(hess) > 1e-6, 'nonlinear check must not be vacuous'


class ActualHeadIntegrationTests(unittest.TestCase):
    def test_anchor(self):
        check_arm('anchor')

    def test_quadratic(self):
        check_arm('quadratic')

    def test_encoder(self):
        check_arm('encoder')

    def test_mixture(self):
        check_arm('mixture')


if __name__ == '__main__':
    unittest.main()
