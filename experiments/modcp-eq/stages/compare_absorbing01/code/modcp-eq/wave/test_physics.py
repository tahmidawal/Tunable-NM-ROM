"""Independent assembly and invariant tests for the new first-order CN path."""
import unittest
import numpy as np
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from physics import (Grid, cn_step, cn_matrix_action, cn_rollout, smooth_tests,
                     face_data, positive_laplacian, damping_ratio, energy,
                     spectral_propagate)
from test_fresh_fom import independent_matrices


class CNTests(unittest.TestCase):
    def test_symmetric_spd_matrix_and_cn_balance(self):
        rng = np.random.default_rng(91051)
        for bc in ('dirichlet', 'absorbing'):
            grid = Grid(5, bc, bc)
            speed, dt = 1.07, .006
            m, k, c = independent_matrices(grid, speed)
            a = m+dt*c/2+dt*dt*k/4
            np.testing.assert_allclose(a, a.T, atol=1e-15)
            self.assertGreater(np.linalg.eigvalsh(a)[0], 0.)
            u, v = rng.normal(size=(2, *grid.shape))
            np.testing.assert_allclose(np.asarray(cn_matrix_action(jnp.asarray(u), speed, dt, grid)).ravel(), a@u.ravel(), atol=1e-14)
            result = cn_step(jnp.asarray(u), jnp.asarray(v), speed, dt, grid, 1e-12, 500)
            un, vn, count, residual, valid = [np.asarray(x) for x in result]
            expected = np.linalg.solve(a, (m+dt*c/2-dt*dt*k/4)@u.ravel()+dt*m@v.ravel())
            np.testing.assert_allclose(un.ravel(), expected, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(un-u-dt*(vn+v)/2, 0., atol=1e-15)
            np.testing.assert_allclose(m@(vn-v).ravel()+dt*c@(vn+v).ravel()/2+dt*k@(un+u).ravel()/2, 0., atol=1e-10)
            olde = .5*(v.ravel()@m@v.ravel()+u.ravel()@k@u.ravel())
            newe = .5*(vn.ravel()@m@vn.ravel()+un.ravel()@k@un.ravel())
            vmid = (vn+v).ravel()/2
            self.assertAlmostEqual(newe-olde, -dt*vmid@c@vmid, places=8)
            if bc == 'absorbing':
                self.assertAlmostEqual(np.sum(m@vn.ravel()+c@un.ravel()), np.sum(m@v.ravel()+c@u.ravel()), places=9)
            self.assertTrue(valid)
            self.assertLess(residual, 1.1e-12)

    def test_weak_stiffness_and_corner_face_moments(self):
        rng = np.random.default_rng(91052)
        for bc in ('dirichlet', 'absorbing'):
            grid = Grid(12, bc, bc)
            phi, lam, modes = smooth_tests(grid, 12)
            fields = rng.normal(size=grid.shape)
            mass = grid.mass().ravel()
            ktest = np.asarray(positive_laplacian(jnp.asarray(phi.T.reshape(12, *grid.shape)), grid)).reshape(12, -1).T
            np.testing.assert_allclose(ktest, phi*lam, atol=3e-12)
            lhs = phi.T@(mass*np.asarray(positive_laplacian(jnp.asarray(fields), grid)).ravel())
            np.testing.assert_allclose(lhs, lam*(phi.T@(mass*fields.ravel())), atol=1e-12)
            if bc == 'absorbing':
                fd = face_data(grid, modes)
                face_fields = (fields[0], fields[-1], fields[:, 0], fields[:, -1])
                projected = sum(test.T@(w*f) for (_, w, test), f in zip(fd, face_fields))
                target = phi.T@(mass*np.asarray(damping_ratio(grid, 1.)).ravel()*fields.ravel())
                np.testing.assert_allclose(projected, target, atol=1e-13)
                np.testing.assert_array_equal(phi[:, 0], np.ones(len(phi)))
                self.assertEqual(lam[0], 0.)
                self.assertAlmostEqual(sum(w.sum() for _, w, _ in fd), 4.)

    def test_cn_second_order_and_direct_parity(self):
        grid = Grid(5, 'absorbing', 'absorbing')
        m, k, c = independent_matrices(grid, .95)
        count = len(m)
        a = np.block([[np.zeros_like(m), np.eye(count)], [-np.linalg.solve(m, k), -np.linalg.solve(m, c)]])
        rng = np.random.default_rng(91053)
        u0, v0 = rng.normal(size=(2, *grid.shape))
        exact = expm(.08*a)@np.r_[u0.ravel(), v0.ravel()]
        errors = []
        for steps in (10, 20, 40):
            result = cn_rollout(jnp.asarray(u0), jnp.asarray(v0), .95, .08/steps, 1e-13,
                                grid=grid, steps=steps, stride=steps)
            pred = np.r_[np.asarray(result['u'][-1]).ravel(), np.asarray(result['v'][-1]).ravel()]
            errors.append(np.linalg.norm(pred-exact))
        self.assertGreater(errors[0]/errors[1], 3.9)
        self.assertGreater(errors[1]/errors[2], 3.9)
        grid = Grid(8)
        u, v = map(jnp.asarray, rng.normal(size=(2, *grid.shape)))
        cg = cn_step(u, v, 1.04, .005, grid, 1e-13, 1000)
        direct = cn_step(u, v, 1.04, .005, grid, 1e-13, 1000, True)
        np.testing.assert_allclose(cg[0], direct[0], atol=1e-12)
        np.testing.assert_allclose(cg[1], direct[1], atol=5e-10)

    def test_spectral_reference_mode(self):
        grid = Grid(12)
        xy = grid.coordinates()
        f = np.sin(np.pi*xy[..., 0])*np.sin(2*np.pi*xy[..., 1])
        omega = 1.03*np.sqrt(4/grid.h**2*(np.sin(np.pi/(2*grid.n))**2+np.sin(np.pi/grid.n)**2))
        times = jnp.array([0., .1, 2.4])
        u, v = spectral_propagate(jnp.asarray(.4*f), jnp.asarray(.7*omega*f), 1.03, times)
        np.testing.assert_allclose(u, (.4*np.cos(omega*np.asarray(times))+.7*np.sin(omega*np.asarray(times)))[:, None, None]*f, atol=2e-14)
        np.testing.assert_allclose(v, omega*(-.4*np.sin(omega*np.asarray(times))+.7*np.cos(omega*np.asarray(times)))[:, None, None]*f, atol=2e-13)


if __name__ == '__main__':
    unittest.main()
