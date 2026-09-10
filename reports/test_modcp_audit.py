"""Small CPU checks of independent physics norms and all-case qualification."""
import unittest
import json
import hashlib
import copy
from pathlib import Path
import tempfile
import numpy as np

from modcp_audit import wave_energy_squared, wave_metrics, summarize_rows, row_error, audit_field_archive, digest
from generate_modcp_comparison import summarize_input
from modcp_freeze import verify_evaluation_freeze


def synthetic_burgers_freeze(folder, document):
    """Write integrity fixtures, never experimental results."""
    document['config']['seeds'] = {'evaluation': 123}
    document['checkpoint_hashes'] = {name: name+'-synthetic-hash' for name in ('cp', 'modcp', 'film')}
    old = copy.deepcopy(document)
    old.update(status='validation_frozen', invocations=[])
    validation_path = folder/'validation_handoff.json'
    validation_path.write_text(json.dumps(old))
    proofs = {}
    for name in ('selections.json', 'selection_freeze.json'):
        path = folder/name
        path.write_text(json.dumps(document['selections']))
        proofs[name] = digest(path)
    entry = dict(handoff_sha256=digest(validation_path), selection_proof_sha256=proofs,
                 evaluation_seed=123, source_commit='test', validation_job_id='test',
                 selected_configurations=document['selections'])
    seal_path = folder/'global_validation_seal.json'
    seal_path.write_text(json.dumps(dict(schema='modcp-global-validation-seal-v1', evaluation_generated=False,
        cases={name: entry for name in ('burgers2d', 'wave_reflective', 'wave_absorbing')})))
    document['provenance']['global_validation_seal_sha256'] = digest(seal_path)


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

    def test_slow_repetition_is_retained_and_counted_within_its_case(self):
        rows = [dict(case=case, rep=rep, seconds=duration, errors={'displacement': .001},
                     finite=True, completed=True)
                for case, durations in enumerate(([1., 1., 3.], [10., 10., 10.]))
                for rep, duration in enumerate(durations)]
        result = summarize_rows(rows, [0, 1], 3, .01)
        self.assertEqual(result['timing_outlier_invocations'], 1)
        self.assertEqual(result['observed_invocations'], 6)
        self.assertEqual(result['median_seconds'], 5.5)
        self.assertTrue(result['qualified'])

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

    def test_shared_wave_truth_requires_checksum_and_matching_geometry(self):
        n = 8
        x = np.arange(1, n)/n
        truth_u = np.stack([np.sin(np.pi*x)[:, None]*np.sin(np.pi*x)[None, :]]*2)
        truth_v = np.zeros_like(truth_u)
        u, v = truth_u*1.01, truth_v.copy()
        metadata = dict(intervals=n, boundary='dirichlet', speed=1.)
        errors = wave_metrics(u, v, truth_u, truth_v, n, 'dirichlet', 1.)
        with tempfile.TemporaryDirectory() as folder:
            predicted, reference = Path(folder)/'prediction.npz', Path(folder)/'reference.npz'
            np.savez(predicted, u=u, v=v, **metadata)
            np.savez(reference, truth_u=truth_u, truth_v=truth_v, **metadata)
            audited = audit_field_archive(predicted, 'wave_reflective', errors,
                expected_shape=(2, n-1, n-1), expected_intervals=n,
                reference_path=reference, reference_sha256=digest(reference))
            self.assertEqual(audited['errors'], errors)
            self.assertEqual(audited['output_sha256']['u'], hashlib.sha256(u.tobytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                audit_field_archive(predicted, 'wave_reflective', errors,
                    reference_path=reference, reference_sha256='wrong reference')
            metadata['speed'] = 2.
            np.savez(reference, truth_u=truth_u, truth_v=truth_v, **metadata)
            with self.assertRaisesRegex(ValueError, 'metadata differ'):
                audit_field_archive(predicted, 'wave_reflective', errors,
                    reference_path=reference, reference_sha256=digest(reference))

    def test_timing_record_must_match_the_independently_audited_output(self):
        truth = np.ones((2, 3, 3), dtype=np.float64)
        field = truth*1.001
        row = dict(split='evaluation', method='modcp', configuration='frozen', intervals=2,
                   case=0, rep=0, seconds=.1, errors={'displacement': .001}, finite=True,
                   completed=True, stationary=False, field_artifact='field.npz',
                   field_artifact_kind='self_contained_full_grid',
                   field_sha256=hashlib.sha256(field.tobytes()).hexdigest())
        document = dict(case_name='burgers2d', status='complete',
            provenance=dict(commit='test', job_id='test', gpu='test', backend='gpu', x64=True, matmul_precision='highest'),
            config=dict(evaluation_case_ids=[0], repetitions=1, targets=[.01], output_times=[0, .05]),
            selections=[dict(method='modcp', configuration='frozen', intervals=2, target=.01)], invocations=[row])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'handoff.json'
            np.savez(Path(folder)/'field.npz', u=field, truth_u=truth)
            synthetic_burgers_freeze(Path(folder), document)
            path.write_text(json.dumps(document))
            self.assertEqual(summarize_input(path)['audited_paired_evaluation_invocations'], 1)
            row['field_sha256'] = 'another invocation'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'output hash differs'):
                summarize_input(path)
            row['field_sha256'] = hashlib.sha256(field.tobytes()).hexdigest()
            path.write_text(json.dumps(document))
            np.savez(Path(folder)/'field.npz', u=field[:1], truth_u=truth[:1])
            with self.assertRaisesRegex(ValueError, 'mesh/time shape differs'):
                summarize_input(path)
            document['selections'].append(dict(method='modcp', configuration='missing', intervals=2, target=.05))
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'missing a selected evaluation'):
                summarize_input(path)

    def test_evaluation_selection_is_bound_to_saved_global_freeze(self):
        document = dict(case_name='burgers2d', status='complete', config={},
                        provenance=dict(commit='test', job_id='test'),
                        selections=[dict(method='cp', configuration='frozen', intervals=8, target=.01)],
                        invocations=[dict(split='evaluation')])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'handoff.json'
            with self.assertRaisesRegex(ValueError, 'lacks the saved global'):
                verify_evaluation_freeze(path, document)
            synthetic_burgers_freeze(Path(folder), document)
            self.assertIsNotNone(verify_evaluation_freeze(path, document))
            document['selections'][0]['configuration'] = 'chosen-after-evaluation'
            with self.assertRaisesRegex(ValueError, 'changed after validation freeze'):
                verify_evaluation_freeze(path, document)


if __name__ == '__main__':
    unittest.main()
