"""Independent controls for cross-grid transfer, physical norms and fast FOM."""
import unittest
import numpy as np
from scipy.fft import dstn
import jax
import jax.numpy as jnp
from pilot import dst2, spectral_propagate, energy_np, restrict, transform_head, metrics
from fresh_fom import Grid, energy, integrate
from fresh_models import head_init, head_apply


class PilotTests(unittest.TestCase):
    def test_dst_matches_independent_scipy_and_inverse(self):
        a = np.random.default_rng(5).normal(size=(15, 15))
        actual = np.asarray(dst2(jnp.asarray(a)))
        np.testing.assert_allclose(actual, dstn(a, type=1, norm="ortho"), atol=2e-14)
        np.testing.assert_allclose(dst2(jnp.asarray(actual)), a, atol=2e-14)

    def test_spectral_mode_physical_velocity_and_rk4(self):
        grid = Grid(16)
        xy = grid.coordinates()
        u0 = np.sin(np.pi*xy[..., 0])*np.sin(2*np.pi*xy[..., 1])
        v0 = .4*u0
        c, t = 1.1, .05
        omega = 2*c/grid.h*np.sqrt(np.sin(np.pi/32)**2+np.sin(2*np.pi/32)**2)
        u, v = map(np.asarray, spectral_propagate(jnp.asarray(u0), jnp.asarray(v0), c, jnp.array([0., t])))
        np.testing.assert_allclose(u[1], (np.cos(omega*t)+.4*np.sin(omega*t)/omega)*u0, atol=3e-14)
        np.testing.assert_allclose(v[1], (-omega*np.sin(omega*t)+.4*np.cos(omega*t))*u0, atol=3e-13)
        ur, vr = integrate(jnp.asarray(u0), jnp.asarray(v0), c, .0005, grid=grid, steps=100, stride=100)
        np.testing.assert_allclose(u[1], ur[1], atol=1e-11)
        np.testing.assert_allclose(v[1], vr[1], atol=1e-10)

    def test_independent_edge_energy_including_absorber_corners(self):
        rng = np.random.default_rng(9)
        for bc in ("dirichlet", "absorbing"):
            grid = Grid(12, bc, bc)
            u, v = rng.normal(size=(2, 3, *grid.shape))
            np.testing.assert_allclose(energy_np(u, v, grid, 1.13), energy(jnp.asarray(u), jnp.asarray(v), grid, 1.13), rtol=1e-14)

    def test_qr_head_preserves_jacobian_and_curvature(self):
        rng = np.random.default_rng(10)
        p, f = head_init(jax.random.PRNGKey(3), rng.normal(size=(9, 3)), rng.normal(size=9), .2, "mlp", width=8)
        p["out"]["w"] = jnp.asarray(rng.normal(size=(8, 9)))
        p["out"]["b"] = jnp.asarray(rng.normal(size=9))
        raw = rng.normal(size=(20, 9))
        q, rr = np.linalg.qr(raw)
        pn = transform_head(p, jnp.asarray(rr))
        old = lambda z: head_apply(p, f, z, "mlp")
        new = lambda z: head_apply(pn, f, z, "mlp")
        z, w = jnp.array([.1, -.2, .3]), jnp.array([.3, .7, -.1])
        np.testing.assert_allclose(q@new(z), raw@old(z), atol=2e-14)
        np.testing.assert_allclose(q@jax.jacfwd(new)(z), raw@jax.jacfwd(old)(z), atol=2e-14)
        curve = lambda fun: jax.jvp(lambda zz: jax.jvp(fun, (zz,), (w,))[1], (z,), (w,))[1]
        np.testing.assert_allclose(q@curve(new), raw@curve(old), atol=2e-14)

    def test_nested_restriction_aligns_coordinates(self):
        for bc in ("dirichlet", "absorbing"):
            coarse, fine = Grid(8, bc, bc), Grid(32, bc, bc)
            for axis in (0, 1):
                np.testing.assert_array_equal(restrict(fine.coordinates()[..., axis], 32, 8, bc), coarse.coordinates()[..., axis])

    def test_vanishing_relative_flags_are_separate_from_initial_error(self):
        grid = Grid(8)
        ut = np.ones((3, *grid.shape))*np.array([1., 1e-5, 0.])[:, None, None]
        vt = np.zeros_like(ut)
        m = metrics(ut+.01, vt, ut, vt, grid, 1., {"vanishing_fraction": .001})
        self.assertEqual(m["displacement"]["reference_vanishing"], [False, True, True])
        self.assertIsNone(m["displacement"]["current_relative"][-1])
        self.assertGreater(m["displacement"]["max_current_relative"], 900)
        self.assertLess(m["displacement"]["max_initial_normalized"], .011)


if __name__ == "__main__":
    print({"jax_backend": jax.default_backend(), "x64": jax.config.jax_enable_x64}, flush=True)
    unittest.main()
