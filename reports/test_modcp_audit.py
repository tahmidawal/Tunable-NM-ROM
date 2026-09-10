"""Small CPU checks of independent physics norms and all-case qualification."""
import unittest
import numpy as np

from modcp_audit import wave_energy_squared, wave_metrics, summarize_rows


class AuditTests(unittest.TestCase):
    def test_absorbing_constant_nullspace_and_kinetic_mass(self):
        n = 8
        u = np.ones((n+1, n+1)) * 7
        v = np.ones_like(u) * 3
        self.assertAlmostEqual(float(wave_energy_squared(u, v, n, 'absorbing', 1.1)), 9)

    def test_absorbing_edge_energy_of_linear_field(self):
        n = 8
        x = np.linspace(0, 1, n+1)
        u = x[:, None] + 2*x[None, :]
        self.assertAlmostEqual(float(wave_energy_squared(u, np.zeros_like(u), n, 'absorbing', 2)), 20)

    def test_zero_initial_velocity_has_finite_error_normalization(self):
        n = 8
        x = np.arange(1, n)/n
        u = (np.sin(np.pi*x)[:, None]*np.sin(np.pi*x)[None, :])[None]
        v = np.zeros_like(u)
        self.assertEqual(wave_metrics(u, v, u, v, n, 'dirichlet', 1),
                         {'displacement': 0., 'velocity': 0., 'energy_state': 0.})

    def test_missing_and_failed_cases_cannot_qualify(self):
        row = dict(case=0, rep=0, seconds=.1, errors={'displacement': .001}, finite=True, completed=True)
        self.assertFalse(summarize_rows([row], [0, 1], 1, .01)['qualified'])
        row['completed'] = False
        self.assertEqual(summarize_rows([row], [0], 1, .01)['failed_cases'], 1)

    def test_velocity_failure_is_not_hidden_by_displacement(self):
        row = dict(case=0, rep=0, seconds=.1,
                   errors={'displacement': .001, 'velocity': .08, 'energy_state': .002}, finite=True, completed=True)
        self.assertFalse(summarize_rows([row], [0], 1, .05)['qualified'])

    def test_repeated_rows_rejected(self):
        row = dict(case=0, rep=0, seconds=.1, errors={'displacement': .001}, finite=True, completed=True)
        with self.assertRaises(ValueError):
            summarize_rows([row, row], [0], 2, .01)


if __name__ == '__main__':
    unittest.main()
