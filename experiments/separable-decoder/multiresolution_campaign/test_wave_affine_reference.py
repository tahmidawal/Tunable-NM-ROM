"""Independent small-system controls for affine wave propagation, CPU only.

These algebraic fixtures are independent of the JAX production wave code. They
verify nonorthogonal reduced mass, affine forcing, physical-speed scaling and
energy balance; they are not PDE performance or accuracy results.
"""
import unittest

import numpy as np
from scipy.linalg import expm


def generator(v, center, stiffness, damping, speed):
    k = v.shape[1]
    mass = v.T@v
    matrix = np.zeros((2*k+1, 2*k+1), dtype=np.float64)
    matrix[:k, k:2*k] = np.eye(k)
    matrix[k:2*k, :k] = -np.linalg.solve(mass, v.T@(speed**2*stiffness)@v)
    matrix[k:2*k, k:2*k] = -np.linalg.solve(mass, v.T@(speed*damping)@v)
    matrix[k:2*k, -1] = -np.linalg.solve(mass, v.T@(speed**2*stiffness)@center)
    return matrix


def physical(v, center, state):
    k = v.shape[1]
    return center+v@state[:k], v@state[k:2*k]


class AffineWaveReference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(790719)
        self.v = rng.normal(size=(7, 3))
        self.center = rng.normal(size=7)
        a, b = rng.normal(size=(7, 7)), rng.normal(size=(7, 7))
        self.stiffness = a.T@a+.5*np.eye(7)
        self.damping = .1*b.T@b
        self.state = np.r_[rng.normal(size=6), 1.]

    def test_offset_has_known_scalar_solution(self):
        v, center = np.ones((1, 1)), np.array([2.])
        matrix = generator(v, center, np.array([[9.]]), np.zeros((1, 1)), 1.25)
        state0 = np.array([-.7, .4, 1.])
        omega = 3.*1.25
        for t in (0., .1, .6, 1.2):
            a, b = physical(v, center, expm(t*matrix)@state0)
            exact_a = 1.3*np.cos(omega*t)+.4*np.sin(omega*t)/omega
            exact_b = -1.3*omega*np.sin(omega*t)+.4*np.cos(omega*t)
            np.testing.assert_allclose(a, [exact_a], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(b, [exact_b], rtol=1e-12, atol=1e-12)
        wrong = matrix.copy(); wrong[1, -1] = 0.
        wrong_a, _ = physical(v, center, expm(.6*wrong)@state0)
        right_a, _ = physical(v, center, expm(.6*matrix)@state0)
        self.assertGreater(np.linalg.norm(wrong_a-right_a), .1)

    def test_nonorthogonal_mass_and_basis_change(self):
        matrix = generator(self.v, self.center, self.stiffness, self.damping, .9)
        change = np.array([[2., .3, 0.], [0., .7, .1], [.2, 0., 1.1]])
        transformed = generator(self.v@change, self.center, self.stiffness, self.damping, .9)
        initial = np.r_[np.linalg.solve(change, self.state[:3]),
                        np.linalg.solve(change, self.state[3:6]), 1.]
        for t in (.0, .05, .3):
            x = physical(self.v, self.center, expm(t*matrix)@self.state)
            y = physical(self.v@change, self.center, expm(t*transformed)@initial)
            np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-12)

    def test_weak_residual_and_dissipation_identity(self):
        speed = 1.1
        matrix = generator(self.v, self.center, self.stiffness, self.damping, speed)
        a, b = physical(self.v, self.center, self.state)
        acceleration = self.v@(matrix@self.state)[3:6]
        weak_residual = self.v.T@(acceleration+speed*self.damping@b+speed**2*self.stiffness@a)
        np.testing.assert_allclose(weak_residual, 0., atol=3e-13)
        energy_derivative = b@acceleration+b@(speed**2*self.stiffness)@a
        np.testing.assert_allclose(energy_derivative, -b@(speed*self.damping)@b, atol=3e-13)
        q, r = np.linalg.qr(self.v, mode='reduced')
        tangent_solution = np.linalg.solve(r, -q.T@(speed*self.damping@b+speed**2*self.stiffness@a))
        np.testing.assert_allclose((matrix@self.state)[3:6], tangent_solution, rtol=2e-12, atol=2e-12)

    def test_reflective_energy_preservation(self):
        speed = .85
        matrix = generator(self.v, self.center, self.stiffness, np.zeros_like(self.damping), speed)
        def energy(st):
            a, b = physical(self.v, self.center, st)
            return .5*(b@b+a@(speed**2*self.stiffness)@a)
        e0 = energy(self.state)
        for t in (.1, .7, 2.4):
            np.testing.assert_allclose(energy(expm(t*matrix)@self.state), e0, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
