"""Sub-minute GPU tests, including active-routing derivative failure controls."""
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import b3d_arch_bench as bench
import b3d_arch_mixture as mixture


def setUpModule():
    print('jax_backend=' + jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu'
    assert jax.config.x64_enabled
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'


def fixture(nonzero=False):
    rng = np.random.default_rng(65)
    anchor, _ = np.linalg.qr(rng.normal(size=(5, 3)), mode='reduced')
    shared = dict(anchor=anchor, center=rng.normal(size=5),
                  latent_scale=np.asarray([.7, 1.2, .9]), output_scale=np.asarray(1.3))
    p, f = mixture.init(jax.random.PRNGKey(16), shared)
    if nonzero:
        for expert in p['experts']:
            w, b = expert[-1]
            expert[-1] = (jnp.asarray(.04 * rng.normal(size=w.shape)),
                          jnp.asarray(.2 * rng.normal(size=b.shape)))
        p['gate'] = (jnp.asarray([[1.2, -.5], [-.8, 1.1], [.6, -.9]]),
                     jnp.asarray([.2, -.1]))
    return p, f, shared


class MixtureTests(unittest.TestCase):
    def test_common_affine_initialization_and_independent_experts(self):
        p, f, shared = fixture()
        z = np.random.default_rng(72).normal(size=(7, 3))
        actual = np.asarray(mixture.apply(p, f, jnp.asarray(z)))
        expected = np.asarray(jnp.asarray(shared['center']) + jnp.asarray(z) @ jnp.asarray(shared['anchor']).T)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(jax.jacfwd(mixture.apply, argnums=2)(p, f, jnp.asarray(z[0])),
                                   shared['anchor'], rtol=1e-14, atol=1e-14)
        self.assertFalse(np.array_equal(p['experts'][0][0][0], p['experts'][1][0][0]))
        self.assertGreater(np.std(np.asarray(mixture.routing(p, f, jnp.asarray(z)))[:, 0]), 0.)
        self.assertTrue(mixture.diagnostics(p, f, z)['identical_expert_collapse'])

    def test_numpy_batch_single_and_saturated_routing(self):
        p, f, _ = fixture(nonzero=True)
        z = np.random.default_rng(11).normal(size=(9, 3))
        actual = np.asarray(mixture.apply(p, f, jnp.asarray(z)))
        np.testing.assert_allclose(actual, mixture.apply_np(p, f, z), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual, np.stack([mixture.apply(p, f, jnp.asarray(row)) for row in z]),
                                   rtol=1e-12, atol=1e-12)
        self.assertGreater(np.ptp(mixture.parts_np(p, f, z)[0][:, 0]), .5)
        w, b = p['gate']
        p['gate'] = (w * 1e4, b * 1e4)
        expected = mixture.apply_np(p, f, z)
        self.assertTrue(np.all(np.isfinite(expected)))
        np.testing.assert_allclose(mixture.apply(p, f, jnp.asarray(z)), expected, rtol=1e-12, atol=1e-12)

    def test_nonconstant_gate_jacobian_jvp_hessian_and_mutation(self):
        p, f, _ = fixture(nonzero=True)
        z = np.asarray([.21, -.13, .17])
        eye = np.eye(len(z)); eps = 2e-5
        fun_np = lambda zz: mixture.apply_np(p, f, zz)
        jac_fd = np.stack([(fun_np(z + eps*axis)-fun_np(z - eps*axis))/(2*eps) for axis in eye], axis=1)
        jac = np.asarray(jax.jacfwd(mixture.apply, argnums=2)(p, f, jnp.asarray(z)))
        np.testing.assert_allclose(jac, jac_fd, rtol=2e-8, atol=2e-9)
        direction = np.asarray([.3, -.8, .2]); direction /= np.linalg.norm(direction)
        _, jvp = jax.jvp(lambda zz: mixture.apply(p, f, zz), (jnp.asarray(z),), (jnp.asarray(direction),))
        tangent_fd = (fun_np(z+eps*direction)-fun_np(z-eps*direction))/(2*eps)
        np.testing.assert_allclose(jvp, tangent_fd, rtol=2e-8, atol=2e-9)
        hess = np.asarray(jax.jacfwd(jax.jacfwd(lambda zz: mixture.apply(p, f, zz)))(jnp.asarray(z)))
        eps2 = 2e-4
        hess_fd = np.empty_like(hess)
        for i in range(len(z)):
            for j in range(len(z)):
                di, dj = eps2*eye[i], eps2*eye[j]
                hess_fd[:, i, j] = (fun_np(z+di+dj)-fun_np(z+di-dj)
                                   -fun_np(z-di+dj)+fun_np(z-di-dj))/(4*eps2**2)
        np.testing.assert_allclose(hess, hess_fd, rtol=2e-6, atol=3e-7)
        # Removing routing derivatives preserves values but corrupts both derivatives.
        def broken(zz):
            gate = jax.lax.stop_gradient(mixture.routing(p, f, zz))
            return p['bias'] + zz @ p['linear'].T + jnp.sum(
                gate[..., :, None] * mixture.expert_outputs(p, f, zz), axis=-2)
        np.testing.assert_array_equal(broken(jnp.asarray(z)), mixture.apply(p, f, jnp.asarray(z)))
        broken_jac = np.asarray(jax.jacfwd(broken)(jnp.asarray(z)))
        broken_hess = np.asarray(jax.jacfwd(jax.jacfwd(broken))(jnp.asarray(z)))
        self.assertGreater(np.linalg.norm(broken_jac-jac_fd)/np.linalg.norm(jac_fd), .01)
        self.assertGreater(np.linalg.norm(broken_hess-hess_fd)/np.linalg.norm(hess_fd), .05)

    def test_routing_and_identical_expert_flags_are_distinct(self):
        p, f, _ = fixture(nonzero=True)
        z = np.random.default_rng(27).normal(size=(20, 3))
        p['gate'] = (jnp.zeros_like(p['gate'][0]), jnp.asarray([20., -20.]))
        info = mixture.diagnostics(p, f, z)
        self.assertTrue(info['routing_collapse'])
        self.assertFalse(info['identical_expert_collapse'])
        self.assertEqual(info['saturation_counts']['0.99'], len(z))
        np.testing.assert_allclose(info['routing_weights'].sum(axis=1), 1., atol=1e-14)
        p['gate'] = (jnp.zeros_like(p['gate'][0]), jnp.zeros(2))
        p['experts'][1] = p['experts'][0]
        info = mixture.diagnostics(p, f, z)
        self.assertFalse(info['routing_collapse'])
        self.assertTrue(info['identical_expert_collapse'])
        np.testing.assert_allclose(info['routing_entropy'], np.log(2.), atol=1e-14)

    def test_short_training_breaks_output_symmetry_and_reduces_loss(self):
        _, _, shared = fixture()
        rng = np.random.default_rng(31)
        z = .5 * rng.normal(size=(24, 3))
        curvature = np.asarray([.3, -.5, .7, .4, -.2])
        target = shared['center'] + z @ shared['anchor'].T + z[:, 0, None]**2 * curvature
        train = dict(target=target, perpendicular2=np.full(len(z), .1),
                     norm2=np.sum(target**2, axis=1) + .1)
        p, f, codes, info = bench.train_model(mixture, shared, z, train,
                                             seed=200, steps=40, lr=.003, batch=len(z))
        self.assertLess(info['global_mse_final'], info['global_mse_initial'])
        expert_outputs = np.asarray(mixture.expert_outputs(p, f, jnp.asarray(codes)))
        self.assertGreater(np.linalg.norm(expert_outputs[:, 0]-expert_outputs[:, 1]), 1e-5)
        self.assertFalse(mixture.diagnostics(p, f, codes)['identical_expert_collapse'])
        bench.check_model(mixture, p, f, codes)


if __name__ == '__main__':
    unittest.main()
