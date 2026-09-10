"""Small CPU checks of independent physics norms and all-case qualification."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from modcp_audit import wave_energy_squared, wave_metrics, summarize_rows, row_error
from generate_modcp_comparison import summarize_input


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

    def test_accurate_capped_rollout_is_not_called_converged(self):
        row = dict(case=0, rep=0, method='modcp', seconds=.1, errors={'displacement': .001},
                   finite=True, completed=True, stationary=False)
        result = summarize_rows([row], [0], 1, .01)
        self.assertTrue(result['qualified'])
        self.assertFalse(result['qualified_and_converged'])
        self.assertEqual(result['nonstationary_cases'], 1)

    def test_owner_diagnostic_traces_do_not_become_target_components(self):
        self.assertEqual(row_error({'errors': {'displacement': .01, 'per_time': [.01, .001], 'initial': .005}}), .01)
        self.assertTrue(np.isinf(row_error({'errors': None})))

    def test_evaluation_requires_predeclared_validation_selection(self):
        # Synthetic fixtures only: these values are never scientific results.
        document = dict(case_name='burgers2d', status='complete',
            provenance=dict(commit='test', job_id='test', gpu='test', backend='gpu', x64=True, matmul_precision='highest'),
            config=dict(evaluation_case_ids=[0], repetitions=1, targets=[.01]), selections=[],
            invocations=[dict(split='evaluation', method='modcp', configuration='unselected', intervals=8,
                              case=0, rep=0, seconds=.1, errors={'displacement': .001}, finite=True, completed=True)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'handoff.json'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'lacks validation-frozen selection'):
                summarize_input(path)

    def test_cpu_fallback_cannot_be_reported_as_scientific_gpu_run(self):
        document = dict(provenance=dict(commit='test', job_id='test', gpu='test', backend='cpu', x64=True, matmul_precision='highest'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'handoff.json'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'backend or precision'):
                summarize_input(path)


if __name__ == '__main__':
    unittest.main()
