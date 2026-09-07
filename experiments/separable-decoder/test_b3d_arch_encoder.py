"""Sub-minute GPU checks for the shared encoder; no PDE result claims."""
import os
from types import SimpleNamespace
import unittest

import numpy as np
import jax
import jax.numpy as jnp

import b3d_arch_baseline as baseline
import b3d_arch_bench as bench
import b3d_arch_encoder as encoder


def setUpModule():
    print('jax_backend=' + jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu'
    assert jax.config.x64_enabled
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'


def fixture(r=7, k=3, n=16):
    rng = np.random.default_rng(71)
    anchor, _ = np.linalg.qr(rng.normal(size=(r, k)), mode='reduced')
    shared = dict(anchor=anchor, center=rng.normal(size=r),
                  latent_scale=np.linspace(.6, 1.4, k), output_scale=np.asarray(.7))
    return shared, rng.normal(size=(n, r))


def nonzero_parameters(p):
    """Exercise nonlinear paths which intentionally vanish at initialization."""
    p = jax.tree_util.tree_map(lambda x: jnp.array(x), p)
    rng = np.random.default_rng(72)
    for layers in [p['decoder']['residual'], p['encoder_residual']]:
        w, b = layers[-1]
        layers[-1] = (jnp.asarray(rng.normal(scale=.03, size=w.shape)),
                      jnp.asarray(rng.normal(scale=.02, size=b.shape)))
    return p


class SharedEncoderTests(unittest.TestCase):
    def test_common_initialization_matches_exact_baseline_and_projection(self):
        shared, a = fixture(r=128, k=32)
        key = jax.random.PRNGKey(200)
        p, f = encoder.init(key, shared)
        reference, reference_f = baseline.init(key, shared)
        for actual, expected in zip(jax.tree_util.tree_leaves(p['decoder']),
                                    jax.tree_util.tree_leaves(reference)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(bench.tree_hash(f), bench.tree_hash(reference_f))
        expected = (a-shared['center']) @ shared['anchor']
        encoded = encoder.encode(p, f, a)
        np.testing.assert_allclose(encoded, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(encoder.encode(p, f, a),
                                      (jnp.asarray(a)-f['center']) @ f['anchor'])
        np.testing.assert_array_equal(encoder.apply(p, f, encoded),
                                      baseline.apply(reference, reference_f, encoded))
        diagnostic = encoder.diagnostics(p, f, encoded)
        self.assertEqual(diagnostic['decoder_parameter_count'], 41472)
        self.assertEqual(diagnostic['encoder_parameter_count'], 37152)

    def test_nonlinear_vector_batch_numpy_and_derivatives(self):
        shared, a = fixture()
        p, f = encoder.init(jax.random.PRNGKey(13), shared)
        p = nonzero_parameters(p)
        z = encoder.encode(p, f, a)
        for apply, apply_np, x in [(encoder.encode, encoder.encode_np, a),
                                   (encoder.apply, encoder.apply_np, np.asarray(z))]:
            batch = np.asarray(apply(p, f, jnp.asarray(x)))
            single = np.stack([np.asarray(apply(p, f, xx)) for xx in x])
            np.testing.assert_allclose(batch, single, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(batch, apply_np(p, f, x), rtol=1e-12, atol=1e-12)
            self.assertEqual(batch.dtype, np.float64)
            direction = np.random.default_rng(73).normal(size=x.shape[1])
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            finite = (apply_np(p, f, x[0]+eps*direction)-
                      apply_np(p, f, x[0]-eps*direction))/(2*eps)
            _, tangent = jax.jvp(lambda xx: apply(p, f, xx),
                                 (jnp.asarray(x[0]),), (jnp.asarray(direction),))
            np.testing.assert_allclose(tangent, finite, rtol=1e-6, atol=1e-8)
        bench.check_model(encoder, p, f, np.asarray(z))

    def test_reconstruction_gradient_reaches_every_network_layer(self):
        shared, a = fixture()
        p, f = encoder.init(jax.random.PRNGKey(14), shared)
        p = nonzero_parameters(p)
        target = jnp.asarray(a)

        def loss(pp):
            prediction = encoder.apply(pp, f, encoder.encode(pp, f, target))
            return jnp.mean(jnp.sum((prediction-target)**2, axis=-1))

        grad = jax.grad(loss)(p)
        for network in [grad['decoder']['residual'], grad['encoder_residual']]:
            for w, b in network:
                self.assertGreater(float(jnp.linalg.norm(w)), 1e-8)
                self.assertGreater(float(jnp.linalg.norm(b)), 1e-8)
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['linear'])), 1e-8)
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['bias'])), 1e-8)
        # Independent finite difference through the entire composition detects
        # a detached encoder even if the decoder's own gradients are correct.
        direction = jax.tree_util.tree_map(jnp.zeros_like, p)
        w, b = direction['encoder_residual'][-1]
        direction['encoder_residual'][-1] = (w.at[0, 0].set(1.), b)
        eps = 1e-5
        plus = jax.tree_util.tree_map(lambda x, d: x+eps*d, p, direction)
        minus = jax.tree_util.tree_map(lambda x, d: x-eps*d, p, direction)
        finite = float((loss(plus)-loss(minus))/(2*eps))
        self.assertAlmostEqual(float(grad['encoder_residual'][-1][0][0, 0]), finite, places=7)

    def test_common_training_reduces_error_and_updates_both_networks(self):
        rng = np.random.default_rng(74)
        orth, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        shared = dict(anchor=orth[:, :2], center=np.zeros(5),
                      latent_scale=np.ones(2), output_scale=np.asarray(1.))
        z = rng.normal(size=(24, 2))
        target = z@shared['anchor'].T + .4*z[:, 0, None]**2*orth[:, 2]
        z0 = target @ shared['anchor']
        train = dict(target=target, norm2=np.sum(target**2, axis=1)+.1,
                     perpendicular2=np.full(24, .1))
        kp, _ = jax.random.split(jax.random.PRNGKey(200))
        initial, frozen_initial = encoder.init(kp, shared)
        p, frozen, assigned, info = bench.train_model(encoder, shared, z0, train,
                                                       seed=200, steps=80, lr=.01, batch=24)
        self.assertLess(info['global_mse_final'], info['global_mse_initial'])
        self.assertEqual(info['optimized_code_count'], 0)
        self.assertEqual(info['mode'], 'encoder')
        self.assertEqual(bench.tree_hash(frozen), bench.tree_hash(frozen_initial))
        for name in ('decoder', 'encoder_residual'):
            self.assertNotEqual(bench.tree_hash(p[name]), bench.tree_hash(initial[name]))
        np.testing.assert_allclose(assigned, encoder.encode_np(p, frozen, target), rtol=1e-11, atol=1e-11)
        bench.check_model(encoder, p, frozen, assigned)

    def test_common_bench_rejects_incorrect_encoder_initialization(self):
        shared, a = fixture()
        z0 = (a-shared['center']) @ shared['anchor']
        train = dict(target=a, norm2=np.sum(a*a, axis=1)+.1,
                     perpendicular2=np.full(len(a), .1))
        bad = SimpleNamespace(NAME='bad_encoder', MODE='encoder', init=encoder.init,
                              apply=encoder.apply,
                              encode=lambda p, f, x: encoder.encode(p, f, x)+.01)
        with self.assertRaises(AssertionError):
            bench.train_model(bad, shared, z0, train, steps=2, batch=len(a))


if __name__ == '__main__':
    unittest.main()
