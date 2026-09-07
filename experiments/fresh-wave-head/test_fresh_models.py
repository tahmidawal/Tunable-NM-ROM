"""Independent head geometry, persistence and weak-dynamics component controls."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from fresh_models import head_init, head_apply, head_geometry, tree_to_npz, tree_from_npz, bank_init, bank_apply
from fresh_rom import weak_acceleration, physical_energy, rollout


class FreshModelTests(unittest.TestCase):
    def test_quadratic_independent_polynomial(self):
        rng = np.random.default_rng(690711)
        r, k = 9, 3
        linear, center = rng.normal(size=(r, k)), rng.normal(size=r)
        p, f = head_init(jax.random.PRNGKey(0), linear, center, .7, "quadratic")
        weights = rng.normal(size=(k*(k+1)//2, r))*.2
        p["quadratic"] = jnp.asarray(weights)
        z, w = rng.normal(size=(2, k))
        a = center+linear@z
        jac, hww = linear.copy(), np.zeros(r)
        pair = 0
        for i in range(k):
            for j in range(i, k):
                a += .7*weights[pair]*z[i]*z[j]
                jac[:, i] += .7*weights[pair]*z[j]
                jac[:, j] += .7*weights[pair]*z[i]
                hww += 1.4*weights[pair]*w[i]*w[j]
                pair += 1
        aa, bb, jj, hh = head_geometry(p, f, jnp.asarray(z), jnp.asarray(w), "quadratic")
        for actual, expected in ((aa, a), (bb, jac@w), (jj, jac), (hh, hww)):
            np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_mlp_finite_difference_and_reload(self):
        rng = np.random.default_rng(690712)
        p, f = head_init(jax.random.PRNGKey(1), rng.normal(size=(8, 3)), rng.normal(size=8), .6, "mlp", 12)
        p["out"]["w"] = jnp.asarray(rng.normal(size=(12, 8))*.2)
        z, w = jnp.asarray(rng.normal(size=(2, 3)))
        a, b, jac, curve = head_geometry(p, f, z, w, "mlp")
        eps = 2e-4
        plus, minus = head_apply(p, f, z+eps*w, "mlp"), head_apply(p, f, z-eps*w, "mlp")
        np.testing.assert_allclose(b, (plus-minus)/(2*eps), atol=1e-8)
        np.testing.assert_allclose(curve, (plus-2*a+minus)/eps**2, atol=2e-7)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)/"head.npz"
            tree_to_npz(path, {"p": p, "f": f})
            tree = tree_from_npz(path)
            for original, restored in zip(jax.tree.leaves({"p": p, "f": f}), jax.tree.leaves(tree)):
                np.testing.assert_array_equal(original, restored)
            np.testing.assert_array_equal(head_apply(p, f, z, "mlp"), head_apply(tree["p"], tree["f"], z, "mlp"))

    def test_common_initialization_and_learned_bank_wall_trace(self):
        rng = np.random.default_rng(690713)
        linear, center = rng.normal(size=(16, 4)), rng.normal(size=16)
        z = jnp.asarray(rng.normal(size=(5, 4)))
        for kind in ("mlp", "quadratic"):
            p, f = head_init(jax.random.PRNGKey(2), linear, center, .7, kind)
            np.testing.assert_allclose(head_apply(p, f, z, kind), center+np.asarray(z)@linear.T, atol=1e-14)
        p, frequency = bank_init(jax.random.PRNGKey(3), rank=16, width=32)
        xy = jnp.array([[0., .3], [1., .7], [.4, 0.], [.8, 1.]])
        self.assertLess(float(jnp.max(abs(bank_apply(p, frequency, xy, True)))), 1e-15)

    def test_curvature_formula_and_energy_derivative(self):
        alpha, z, w = .8, .7, -.9
        p = {"linear": jnp.array([[1.], [0.]]), "bias": jnp.zeros(2), "quadratic": jnp.array([[0., alpha]])}
        f = {"output_scale": jnp.array(1.)}
        stiffness, damp = jnp.diag(jnp.array([1.4, 2.3])), jnp.diag(jnp.array([.2, .4]))
        zz, ww = jnp.array([z]), jnp.array([w])
        a, power, ratio, valid = weak_acceleration(p, f, zz, ww, stiffness, damp, "quadratic")
        expected = -(4*alpha**2*z*w*w+(.2+4*alpha**2*z*z*.4)*w+1.4*z+2*alpha**2*2.3*z**3)/(1+4*alpha**2*z*z)
        np.testing.assert_allclose(a, [expected], atol=1e-13)
        without_curvature = expected+4*alpha**2*z*w*w/(1+4*alpha**2*z*z)
        self.assertGreater(abs(float(a[0])-without_curvature), .2)
        de = jax.jvp(lambda pos, vel: physical_energy(p, f, pos, vel, stiffness, "quadratic"), (zz, ww), (ww, a))[1]
        self.assertAlmostEqual(float(de+power), 0., places=12)
        self.assertTrue(bool(valid))

    def test_curved_rk4_independent_dop853(self):
        alpha = .8
        p = {"linear": jnp.array([[1.], [0.]]), "bias": jnp.zeros(2), "quadratic": jnp.array([[0., alpha]])}
        f = {"output_scale": jnp.array(1.)}
        stiffness, damp = jnp.diag(jnp.array([1.4, 2.3])), jnp.diag(jnp.array([.2, .4]))
        def reference(t, state):
            z, w = state
            a = -(4*alpha**2*z*w*w+(.2+4*alpha**2*z*z*.4)*w+1.4*z+2*alpha**2*2.3*z**3)/(1+4*alpha**2*z*z)
            return [w, a]
        ref = solve_ivp(reference, (0, .2), [.7, -.9], method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
        result = rollout(p, f, jnp.array([.7]), jnp.array([-.9]), stiffness, damp, .002, kind="quadratic", steps=100, stride=100)
        np.testing.assert_allclose([result["z"][-1, 0], result["w"][-1, 0]], ref, atol=1e-11)
        self.assertTrue(bool(result["completed"][-1]))

    def test_nonorthogonal_linear_weak_model_against_exponential(self):
        linear = np.array([[1., .2], [.1, .8], [.3, -.2], [.4, .1]])
        p, f = head_init(jax.random.PRNGKey(5), linear, np.zeros(4), 1., "quadratic")
        stiff, damp = np.diag([1., 2., 3., 4.]), np.diag([.1, .2, .3, .4])
        mass = linear.T@linear
        kr, dr = np.linalg.solve(mass, linear.T@stiff@linear), np.linalg.solve(mass, linear.T@damp@linear)
        matrix = np.block([[np.zeros((2, 2)), np.eye(2)], [-kr, -dr]])
        init = np.array([.3, -.4, .2, .5])
        ref = expm(.3*matrix)@init
        result = rollout(p, f, jnp.asarray(init[:2]), jnp.asarray(init[2:]), jnp.asarray(stiff), jnp.asarray(damp), .002, kind="quadratic", steps=150, stride=150)
        np.testing.assert_allclose(np.r_[result["z"][-1], result["w"][-1]], ref, atol=1e-11)

    def test_flux_balance_refinement(self):
        p = {"linear": jnp.array([[1.], [0.]]), "bias": jnp.zeros(2), "quadratic": jnp.array([[0., .8]])}
        f = {"output_scale": jnp.array(1.)}
        stiffness, damp = jnp.diag(jnp.array([1.4, 2.3])), jnp.diag(jnp.array([.2, .4]))
        errors = []
        initial = float(physical_energy(p, f, jnp.array([.7]), jnp.array([-.9]), stiffness, "quadratic"))
        for steps in (20, 40, 80):
            result = rollout(p, f, jnp.array([.7]), jnp.array([-.9]), stiffness, damp, .4/steps, kind="quadratic", steps=steps, stride=steps)
            final = float(physical_energy(p, f, result["z"][-1], result["w"][-1], stiffness, "quadratic"))
            errors.append(abs(final+float(result["outflux"][-1])-initial))
        self.assertGreater(min(np.log2(np.array(errors[:-1])/errors[1:])), 3.5)

    def test_rank_deficiency_and_overflow_fail_and_freeze(self):
        f = {"output_scale": jnp.array(1.)}
        stiffness, damp = jnp.eye(2), jnp.eye(2)
        p = {"linear": jnp.zeros((2, 1)), "bias": jnp.zeros(2), "quadratic": jnp.zeros((1, 2))}
        result = rollout(p, f, jnp.array([.3]), jnp.array([.2]), stiffness, damp, .01, kind="quadratic", steps=4, stride=1)
        self.assertFalse(bool(result["completed"][0]))
        self.assertFalse(bool(result["completed"][-1]))
        np.testing.assert_array_equal(result["z"], np.full((5, 1), .3))
        p["linear"] = jnp.array([[1.], [0.]])
        p["quadratic"] = jnp.array([[0., 1.]])
        result = rollout(p, f, jnp.array([1e150]), jnp.array([1e150]), stiffness, damp, .01, kind="quadratic", steps=4, stride=1)
        self.assertFalse(bool(result["completed"][0]))
        np.testing.assert_array_equal(result["z"], np.full((5, 1), 1e150))


if __name__ == "__main__":
    print({"jax_backend": jax.default_backend(), "x64": jax.config.jax_enable_x64, "matmul_precision": str(jax.config.jax_default_matmul_precision)}, flush=True)
    unittest.main()
