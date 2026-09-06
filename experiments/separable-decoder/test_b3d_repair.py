"""Sub-minute GPU smoke checks against independent linear algebra references."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import b3d_repair as repair
import sep_hfit as hf


class RepairMathTests(unittest.TestCase):
    def test_parameter_provenance_rounding_and_negative_controls(self):
        original = dict(seed=0, m=2, B=np.asarray([1, 2]), c=np.ones((2, 3, 3)),
                        w=np.ones((2, 3)), rho=np.ones((2, 3)), A=np.ones(2),
                        nu=np.asarray([.02, .03]), s_star=np.asarray([.8, .9]), sha256='original')
        generated = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                     for key, value in original.items()}
        generated['sha256'] = 'regenerated'
        generated['nu'][0] = np.nextafter(generated['nu'][0], np.inf)
        repair.validate_parameter_manifest(generated, original, 'original')
        generated['c'][0, 0, 0] += 1e-5
        with self.assertRaises(AssertionError):
            repair.validate_parameter_manifest(generated, original, 'original')
        generated['c'] = original['c'].copy()
        generated['nu'][0] *= 1.001
        with self.assertRaises(AssertionError):
            repair.validate_parameter_manifest(generated, original, 'original')

    def test_qr_identity_and_head_round_trip(self):
        rng = np.random.default_rng(42)
        g = rng.normal(size=(31, 5)) * np.logspace(-4, 0, 5)
        q, r = np.linalg.qr(g, mode="reduced")
        hp = hf.init_head(jax.random.PRNGKey(2), 2, 5, hidden=4, layers=1)
        qp = hf.to_q(hp, jnp.asarray(r.T))
        restored = hf.to_h(qp, jnp.asarray(np.linalg.inv(r.T)))
        z = jnp.asarray(rng.normal(size=(3, 2)))
        h = np.asarray(hf.head_apply(hp, z))
        np.testing.assert_allclose(hf.head_apply(restored, z), h, rtol=1e-11, atol=1e-11)
        target = rng.normal(size=(3, 31))
        a = target @ q
        perp = target - a @ q.T
        field2 = np.sum((h @ g.T - target) ** 2, axis=1)
        coeff2 = np.sum((np.asarray(hf.head_apply(qp, z)) - a) ** 2, axis=1) + np.sum(perp**2, axis=1)
        np.testing.assert_allclose(coeff2, field2, rtol=2e-13, atol=2e-13)

    def test_lm_matches_independent_linear_least_squares(self):
        matrix = np.asarray([[1., 0., 1.], [0., 2., -1.]])
        hp = dict(h=[(jnp.zeros((2, 3)), jnp.zeros(3))], h_lin=jnp.asarray(matrix))
        target = np.asarray([[.7, -1., 2.]])
        floor2 = np.asarray([.25])
        norm2 = np.sum(target**2, axis=1) + floor2
        solve = repair.make_fit(100, gradient_tol=1e-10)
        z, error, optimality, attempts, accepted, reason = solve(
            hp, jnp.asarray(target), jnp.asarray(floor2), jnp.asarray(norm2), jnp.zeros((1, 2)))
        expected_z = np.linalg.lstsq(matrix.T, target[0], rcond=None)[0]
        expected_error = np.sqrt((np.linalg.norm(expected_z @ matrix - target[0])**2 + .25) / norm2[0])
        np.testing.assert_allclose(z[0], expected_z, atol=1e-9)
        self.assertAlmostEqual(float(error[0]), expected_error, places=11)
        self.assertLess(float(optimality[0]), 1e-10)
        self.assertEqual(int(reason[0]), 1)
        self.assertGreater(int(accepted[0]), 0)

    def test_exact_initial_fit_and_budget_exhaustion(self):
        hp = dict(h=[(jnp.zeros((2, 2)), jnp.zeros(2))], h_lin=jnp.eye(2))
        a = jnp.asarray([[1., 2.]])
        floor = jnp.asarray([.1])
        norm = jnp.asarray([5.1])
        stationary = repair.make_fit(4)(hp, a, floor, norm, a)
        exhausted = repair.make_fit(0)(hp, a, floor, norm, jnp.zeros_like(a))
        self.assertEqual(int(stationary[-1][0]), 1)
        self.assertEqual(int(stationary[3][0]), 0)
        self.assertEqual(int(exhausted[-1][0]), 0)
        self.assertGreater(float(exhausted[2][0]), 1e-6)


if __name__ == "__main__":
    print('jax_backend=' + jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu'
    unittest.main()
