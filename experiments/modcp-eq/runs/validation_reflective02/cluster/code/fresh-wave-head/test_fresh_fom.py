"""Independent small-grid assembly and analytic checks; no old wave dependencies."""
import unittest
import numpy as np
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from fresh_fom import Grid, acceleration, positive_laplacian, damping_ratio, integrate, energy, dissipation, localized_initial, provenance


def independent_matrices(grid, c):
    """Build incidence matrices from endpoint indices, without FOM stencil code."""
    axes = []
    for bc in (grid.bx, grid.by):
        n = grid.n
        if bc == "periodic":
            edges = np.zeros((n, n))
            for i in range(n):
                edges[i, i], edges[i, (i+1) % n] = -1, 1
            w = np.full(n, grid.length/n)
            b = np.zeros(n)
        else:
            edges = np.zeros((n, n+1))
            for i in range(n):
                edges[i, i], edges[i, i+1] = -1, 1
            w = np.full(n+1, grid.length/n)
            w[0] /= 2
            w[-1] /= 2
            b = np.zeros(n+1)
            if bc == "absorbing":
                b[0] = b[-1] = 1
            if bc == "dirichlet":
                edges, w, b = edges[:, 1:-1], w[1:-1], b[1:-1]
        axes.append((np.diag(w), edges.T@edges/(grid.length/n), np.diag(b)))
    hx, sx, bx = axes[0]
    hy, sy, by = axes[1]
    m = np.kron(hx, hy)
    k = c*c*(np.kron(sx, hy)+np.kron(hx, sy))
    damp = c*(np.kron(bx, hy)+np.kron(hx, by))
    return m, k, damp


class FreshFOMTests(unittest.TestCase):
    def test_backend_and_precision(self):
        self.assertEqual(jax.default_backend(), "gpu")
        self.assertTrue(jax.config.jax_enable_x64)
        self.assertEqual(str(jax.config.jax_default_matmul_precision), "highest")

    def test_independent_matrix_operator_and_energy(self):
        rng = np.random.default_rng(6041)
        for bcs in [("dirichlet",)*2, ("absorbing",)*2, ("periodic",)*2, ("absorbing", "neumann")]:
            grid = Grid(5, *bcs)
            u, v = rng.normal(size=(2, *grid.shape))
            c = 1.17
            m, k, damp = independent_matrices(grid, c)
            ref = np.linalg.solve(m, -k@u.ravel()-damp@v.ravel())
            np.testing.assert_allclose(np.asarray(acceleration(jnp.array(u), jnp.array(v), grid, c)).ravel(), ref, atol=1e-12)
            e = .5*(v.ravel()@m@v.ravel()+u.ravel()@k@u.ravel())
            self.assertAlmostEqual(float(energy(jnp.array(u), jnp.array(v), grid, c)), e, places=11)
            a = np.asarray(acceleration(jnp.array(u), jnp.array(v), grid, c)).ravel()
            derivative = v.ravel()@m@a + v.ravel()@k@u.ravel()
            self.assertAlmostEqual(derivative, -float(dissipation(jnp.array(v), grid, c)), places=10)

    def test_exponential_initial_velocity_and_mutations(self):
        grid = Grid(4, "absorbing", "absorbing")
        c, dt, steps = .91, .0004, 50
        rng = np.random.default_rng(6042)
        u0, v0 = rng.normal(size=(2, *grid.shape))
        m, k, damp = independent_matrices(grid, c)
        count = len(m)
        a = np.block([[np.zeros_like(m), np.eye(count)], [-np.linalg.solve(m, k), -np.linalg.solve(m, damp)]])
        initial = np.r_[u0.ravel(), v0.ravel()]
        reference = expm(steps*dt*a)@initial
        u, v = integrate(jnp.array(u0), jnp.array(v0), c, dt, grid=grid, steps=steps, stride=steps)
        actual = np.r_[np.asarray(u[-1]).ravel(), np.asarray(v[-1]).ravel()]
        np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
        for mutation in ("wrong_stiffness", "no_boundary_damping", "lost_initial_velocity"):
            bad_a, bad_initial = a.copy(), initial.copy()
            if mutation == "wrong_stiffness":
                bad_a[count:, :count] *= -1
            elif mutation == "no_boundary_damping":
                bad_a[count:, count:] = 0
            else:
                bad_initial[count:] = 0
            bad = expm(steps*dt*bad_a)@bad_initial
            self.assertGreater(np.linalg.norm(bad-reference)/np.linalg.norm(reference), .005, mutation)

    def test_nonzero_velocity_standing_mode_semidiscrete(self):
        grid, c = Grid(12), 1.03
        xy = grid.coordinates()
        mode = np.sin(np.pi*xy[..., 0])*np.sin(2*np.pi*xy[..., 1])
        omega = c*np.sqrt(4/grid.h**2*(np.sin(np.pi/(2*grid.n))**2+np.sin(np.pi/grid.n)**2))
        t, dt = .16, .001
        u, v = integrate(jnp.array(.4*mode), jnp.array(.7*omega*mode), c, dt, grid=grid, steps=160, stride=160)
        np.testing.assert_allclose(u[-1], (.4*np.cos(omega*t)+.7*np.sin(omega*t))*mode, atol=2e-10)
        np.testing.assert_allclose(v[-1], omega*(-.4*np.sin(omega*t)+.7*np.cos(omega*t))*mode, atol=2e-9)

    def test_initial_support_compatibility(self):
        grid = Grid(24, "absorbing", "absorbing")
        u, v = localized_initial(grid, (.48, .53, .22, .23, .9, 1.1, .4, -.2))
        for q in (np.asarray(u), np.asarray(v)):
            self.assertEqual(float(np.max(abs(q[[0, -1], :]))), 0.)
            self.assertEqual(float(np.max(abs(q[:, [0, -1]]))), 0.)

    def test_absorber_constant_and_invariant(self):
        grid = Grid(7, "absorbing", "absorbing")
        u0 = jnp.full(grid.shape, .37)
        v0 = jnp.zeros(grid.shape)
        u, v = integrate(u0, v0, .9, .001, grid=grid, steps=20, stride=10)
        np.testing.assert_array_equal(u, np.full(u.shape, .37))
        np.testing.assert_array_equal(v, np.zeros(v.shape))
        rng = np.random.default_rng(6043)
        u0, v0 = map(jnp.asarray, rng.normal(size=(2, *grid.shape)))
        u, v = integrate(u0, v0, .9, .001, grid=grid, steps=20, stride=10)
        invariant = jnp.sum(jnp.asarray(grid.mass())*(v+damping_ratio(grid, .9)*u), axis=(-2,-1))
        np.testing.assert_allclose(invariant, invariant[0], atol=1e-13, rtol=1e-13)


if __name__ == "__main__":
    print(provenance(), flush=True)
    unittest.main()
