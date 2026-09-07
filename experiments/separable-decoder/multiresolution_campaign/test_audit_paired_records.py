"""Accounting regressions motivated by previously retracted project claims."""
import unittest

from audit_paired_records import audit


def fixture():
    record = {
        'schema_version': 1, 'stage': 'development', 'reference_kind': 'refined_physical',
        'configuration_selection': 'predeclared', 'source_artifact_sha256': {'result.json': 'a'*64},
        'provenance': {'backend': 'gpu', 'x64': True, 'matmul_precision': 'highest',
                       'job_id': '123', 'gpu_identity': 'device-1', 'source_commit': 'b'*40},
        'case_ids': ['case-a', 'case-b'], 'repetitions': 3,
        'required_error_metrics': ['u', 'v'], 'output_contract_id': 'same-dense-output',
        'reference_uncertainty_bound': .0001, 'reference_uncertainty_fraction': .1,
        'targets': [.01, .001], 'invocations': [],
    }
    for case in record['case_ids']:
        for repeat in range(3):
            for method, seconds, err in [('nmrom', .01, .005), ('fom', .02, .0002)]:
                iid = f'{case}-{repeat}-{method}'
                record['invocations'].append({
                    'case_id': case, 'repeat': repeat, 'method': method,
                    'invocation_id': iid, 'metric_invocation_id': iid,
                    'job_id': '123', 'gpu_identity': 'device-1',
                    'output_contract_id': 'same-dense-output',
                    'query_includes_input_and_output': True, 'query_seconds': seconds,
                    'warmup_and_burnin_complete': True, 'device_synchronized': True,
                    'completed': True, 'numerically_valid': True, 'errors': {'u': err, 'v': err},
                })
    return record


class AccountingTests(unittest.TestCase):
    def test_unattained_accuracy_cannot_supply_speedup(self):
        result = audit(fixture())
        self.assertEqual(result['targets'][0]['qualified_complete_query_speedup'], 2.)
        self.assertIsNone(result['targets'][1]['qualified_complete_query_speedup'])
        self.assertFalse(result['targets'][0]['independent_confirmation'])

    def test_failed_case_is_retained_and_blocks_acceptance(self):
        record = fixture()
        record['invocations'][0].update(completed=False, numerically_valid=False, errors={'u': None, 'v': None})
        result = audit(record)
        self.assertEqual(result['targets'][0]['failed_or_above_target_cases']['nmrom'], 1)
        self.assertIsNone(result['targets'][0]['qualified_complete_query_speedup'])

    def test_same_grid_agreement_cannot_claim_physical_accuracy(self):
        record = fixture()
        record['reference_kind'] = 'same_grid_discrete'
        self.assertIsNone(audit(record)['targets'][0]['qualified_complete_query_speedup'])

    def test_underresolved_reference_blocks_tight_claim(self):
        record = fixture()
        record['reference_uncertainty_bound'] = .01
        self.assertIsNone(audit(record)['targets'][0]['qualified_complete_query_speedup'])

    def test_reference_uncertainty_requires_margin_at_threshold(self):
        for observed in [.00995, .0098995]:
            with self.subTest(observed=observed):
                record = fixture()
                record['invocations'][0]['errors']['u'] = observed
                result = audit(record)['targets'][0]
                self.assertTrue(result['matched_reference_accuracy'])
                self.assertTrue(result['physical_reference_budget_met'])
                self.assertEqual(result['failed_or_above_target_with_reference_margin']['nmrom'], 1)
                self.assertIsNone(result['qualified_complete_query_speedup'])

    def test_missing_reference_bound_preserves_times_without_accuracy_claim(self):
        record = fixture()
        record['reference_uncertainty_bound'] = None
        result = audit(record)
        self.assertEqual(result['raw_speedup_median_of_case_ratios'], 2.)
        self.assertIsNone(result['targets'][0]['qualified_complete_query_speedup'])

    def test_wrong_timing_or_metric_provenance_rejected(self):
        for field, value in [('job_id', '124'), ('gpu_identity', 'another-device'),
                             ('metric_invocation_id', 'separate-accuracy-run'),
                             ('output_contract_id', 'latent-only'),
                             ('query_includes_input_and_output', False),
                             ('device_synchronized', False)]:
            with self.subTest(field=field):
                record = fixture()
                record['invocations'][0][field] = value
                with self.assertRaises(ValueError):
                    audit(record)

    def test_missing_or_duplicated_case_repetition_rejected(self):
        record = fixture()
        record['invocations'].pop()
        with self.assertRaises(ValueError):
            audit(record)
        record = fixture()
        record['invocations'].append(record['invocations'][0].copy())
        with self.assertRaises(ValueError):
            audit(record)


if __name__ == '__main__':
    unittest.main()
