"""Sub-minute GPU checks against independent polynomial mathematics.

The saved JSON records component checks, not a PDE-accuracy experiment.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import unittest

import numpy as np
import jax
import jax.numpy as jnp

import b3d_arch_bench as bench
import b3d_arch_quadratic as model

RECORD = {}


def setUpModule():
    print('jax_backend=' + jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu'
    assert jax.config.x64_enabled
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'


def fixture(k=3, r=5, seed=71):
    rng = np.random.default_rng(seed)
    anchor, _ = np.linalg.qr(rng.normal(size=(r, k)), mode='reduced')
    shared = dict(anchor=anchor, center=rng.normal(size=r),
                  latent_scale=np.linspace(.4, 2., k), output_scale=np.asarray(1.7))
    p, frozen = model.init(jax.random.PRNGKey(seed), shared)
    p['linear'] = jnp.asarray(rng.normal(size=(r, k)))
    p['quadratic'] = jnp.asarray(rng.normal(size=(k * (k + 1) // 2, r)))
    return rng, shared, p, frozen


def polynomial_tensor(p, frozen):
    """Symmetric Hessian in unscaled z coordinates, constructed by explicit loops."""
    a = np.asarray(p['linear'])
    weights = np.asarray(p['quadratic'])
    scales = np.asarray(frozen['latent_scale'])
    factor = float(frozen['output_scale'])
    r, k = a.shape
    hessian = np.zeros((r, k, k))
    pair = 0
    for i in range(k):
        for j in range(i, k):
            coefficient = factor * weights[pair] / (scales[i] * scales[j])
            if i == j:
                hessian[:, i, i] = 2 * coefficient
            else:
                hessian[:, i, j] = coefficient
                hessian[:, j, i] = coefficient
            pair += 1
    return hessian


class QuadraticTests(unittest.TestCase):
    def test_output_batch_and_checkpoint_against_symmetric_polynomial(self):
        rng, _, p, frozen = fixture()
        z = rng.normal(size=(7, 3))
        h = polynomial_tensor(p, frozen)
        expected = np.asarray(p['bias']) + z @ np.asarray(p['linear']).T
        expected += .5 * np.einsum('ni,cij,nj->nc', z, h, z)
        actual = np.asarray(jax.jit(model.apply)(p, frozen, jnp.asarray(z)))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(model.apply_np(p, frozen, z), expected, rtol=2e-13, atol=2e-13)
        singles = np.stack([np.asarray(model.apply(p, frozen, row)) for row in z])
        np.testing.assert_allclose(singles, actual, rtol=2e-13, atol=2e-13)
        blocks = np.stack([z, z / 3])
        np.testing.assert_allclose(model.apply(p, frozen, blocks), model.apply_np(p, frozen, blocks), rtol=2e-13, atol=2e-13)
        payload = jax.tree_util.tree_map(np.asarray, dict(trainable=p, frozen=frozen))
        loaded = pickle.loads(pickle.dumps(payload))
        np.testing.assert_array_equal(model.apply(loaded['trainable'], loaded['frozen'], z), model.apply(p, frozen, z))
        RECORD['output_max_absolute_error'] = float(np.max(np.abs(actual - expected)))

    def test_jacobian_constant_hessian_and_finite_differences(self):
        rng, _, p, frozen = fixture()
        z = rng.normal(size=3)
        expected_h = polynomial_tensor(p, frozen)
        expected_j = np.asarray(p['linear']) + np.einsum('cij,j->ci', expected_h, z)
        jac = jax.jacfwd(model.apply, argnums=2)
        hess = jax.jacfwd(jac, argnums=2)
        actual_j = np.asarray(jac(p, frozen, z))
        actual_h = np.asarray(hess(p, frozen, z))
        np.testing.assert_allclose(actual_j, expected_j, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(actual_h, expected_h, rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(hess(p, frozen, z + 2.7), actual_h)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        finite_j = (model.apply_np(p, frozen, z + eps * direction)
                    - model.apply_np(p, frozen, z - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(finite_j, expected_j @ direction, rtol=1e-8, atol=1e-8)
        # A mixed central difference of outputs independently checks both Hessian axes.
        hstep = 1e-3
        finite_h = np.empty_like(expected_h)
        eye = np.eye(3)
        for i in range(3):
            for j in range(3):
                ei, ej = hstep * eye[i], hstep * eye[j]
                finite_h[:, i, j] = (model.apply_np(p, frozen, z + ei + ej)
                    - model.apply_np(p, frozen, z + ei - ej)
                    - model.apply_np(p, frozen, z - ei + ej)
                    + model.apply_np(p, frozen, z - ei - ej)) / (4 * hstep**2)
        np.testing.assert_allclose(finite_h, expected_h, rtol=2e-7, atol=2e-7)
        RECORD['jacobian_max_absolute_error'] = float(np.max(np.abs(actual_j - expected_j)))
        RECORD['hessian_max_absolute_error'] = float(np.max(np.abs(actual_h - expected_h)))
        RECORD['finite_difference_hessian_max_absolute_error'] = float(np.max(np.abs(finite_h - expected_h)))

    def test_each_mixed_product_once_and_diagonal_factor_two(self):
        _, _, p, frozen = fixture(k=2, r=3)
        p = dict(linear=jnp.zeros((3, 2)), bias=jnp.zeros(3), quadratic=jnp.eye(3))
        frozen = dict(latent_scale=jnp.asarray([2., 5.]), output_scale=jnp.asarray(3.))
        z = jnp.asarray([4., 15.])
        # Lexicographic features are x0^2, x0*x1, x1^2, with x=(2,3).
        np.testing.assert_allclose(model.apply(p, frozen, z), [12., 18., 27.], rtol=0, atol=0)
        expected = np.asarray([[[1.5, 0.], [0., 0.]],
                               [[0., .3], [.3, 0.]],
                               [[0., 0.], [0., .24]]])
        actual = jax.jacfwd(jax.jacfwd(model.apply, argnums=2), argnums=2)(p, frozen, z)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_zero_quadratic_common_initialization_and_parameter_count(self):
        _, shared, _, _ = fixture(k=32, r=128)
        p, frozen = model.init(jax.random.PRNGKey(200), shared)
        repeat_p, repeat_f = model.init(jax.random.PRNGKey(201), shared)
        z = np.random.default_rng(42).normal(size=(8, 32))
        np.testing.assert_array_equal(p['quadratic'], np.zeros((528, 128)))
        expected = jnp.asarray(shared['center']) + jnp.asarray(z) @ jnp.asarray(shared['anchor']).T
        np.testing.assert_array_equal(model.apply(p, frozen, z), expected)
        np.testing.assert_array_equal(jax.jacfwd(model.apply, argnums=2)(p, frozen, z[0]), shared['anchor'])
        self.assertEqual(bench.tree_hash(p), bench.tree_hash(repeat_p))
        self.assertEqual(bench.tree_hash(frozen), bench.tree_hash(repeat_f))
        count = sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(p))
        self.assertEqual(count, 128 * (1 + 32 + 32 * 33 // 2))
        self.assertEqual(count, 71808)
        RECORD['trainable_parameter_count_K32_R128'] = count
        RECORD['deterministic_initialization_across_optimizer_seeds'] = True

    def test_invalid_scales_fail_before_training(self):
        _, shared, _, _ = fixture()
        with self.assertRaises(ValueError):
            model.init(jax.random.PRNGKey(0), dict(shared, latent_scale=np.asarray([1., 0., 1.])))
        with self.assertRaises(ValueError):
            model.init(jax.random.PRNGKey(0), dict(shared, output_scale=np.asarray(np.nan)))

    def test_common_training_decreases_synthetic_polynomial_loss(self):
        rng = np.random.default_rng(22)
        orth, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        shared = dict(anchor=orth[:, :2], center=np.zeros(5),
                      latent_scale=np.asarray([.7, 1.4]), output_scale=np.asarray(.8))
        z = rng.normal(size=(32, 2))
        target = z @ shared['anchor'].T
        target += .4 * z[:, :1]**2 * orth[:, 2]
        target += .2 * (z[:, 0] * z[:, 1])[:, None] * orth[:, 3]
        floor = np.full(len(z), .01)
        train = dict(target=target, norm2=np.sum(target**2, axis=1) + floor, perpendicular2=floor)
        p, frozen, codes, info = bench.train_model(model, shared, z, train,
            seed=200, steps=80, lr=.02, batch=16)
        self.assertLess(info['global_mse_final'], .7 * info['global_mse_initial'])
        self.assertGreater(np.linalg.norm(p['quadratic']), 0.)
        checks = bench.check_model(model, p, frozen, codes)
        RECORD['synthetic_training'] = info
        RECORD['trained_model_checks'] = checks


if __name__ == '__main__':
    start = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QuadraticTests))
    here = Path(__file__).resolve().parent
    record = dict(kind='quadratic_component_checks', scientific_accuracy_result=False,
                  passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  seconds=time.time()-start, jax_backend=jax.default_backend(),
                  jax_version=jax.__version__, gpu=jax.devices()[0].device_kind,
                  x64=bool(jax.config.x64_enabled),
                  matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                  source_sha256={name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                                 for name in ('b3d_arch_quadratic.py', 'test_b3d_arch_quadratic.py', 'b3d_arch_bench.py')},
                  checks=RECORD)
    output = here / 'runs/b3d_arch_quadratic/component_tests.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print('component_record=' + str(output), flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
