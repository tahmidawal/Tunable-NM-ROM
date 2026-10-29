"""Gate mechanics only: no synthetic reference admission is written."""
import copy
import unittest
import admit_finite_channel as module


class Gates(unittest.TestCase):
    def setUp(self):
        self.required = dict(name='required', role='adopted_truth', required=True,
                             operator='le', threshold=0.01, applies_to_re='all', criterion_source='fixture')
        self.row = dict(name='required', role='adopted_truth', threshold=0.01,
                        value=0.001, status='pass', evidence_sha256='a'*64)

    def test_actual_numeric_failure_cannot_claim_pass(self):
        module.check_rows([self.row], [self.required], 20)
        self.row['value'] = 0.1
        with self.assertRaises(ValueError):
            module.check_rows([self.row], [self.required], 20)

    def test_candidate_cannot_demote_required_role(self):
        self.row['role'] = 'diagnostic'
        with self.assertRaises(ValueError):
            module.check_rows([self.row], [self.required], 20)

    def test_diagnostic_failure_stays_visible(self):
        rule = dict(self.required, name='diagnostic', role='diagnostic', required=False)
        row = dict(self.row, name='diagnostic', role='diagnostic', value=0.2, status='fail')
        module.check_rows([self.row, row], [self.required, rule], 20)
        self.assertEqual(row['status'], 'fail')

    def test_unresolved_required_is_rejected(self):
        self.row['status'] = 'unresolved'
        with self.assertRaises(ValueError):
            module.check_rows([self.row], [self.required], 20)

    def test_anchor_cannot_transfer(self):
        self.required['applies_to_re'] = [20]
        with self.assertRaises(ValueError):
            module.check_rows([self.row], [self.required], 15)
        self.row.update(status='not_applicable', value=None)
        module.check_rows([self.row], [self.required], 15)

    def test_scope_types_are_exact(self):
        scope = module.expected_scope()
        altered = copy.deepcopy(scope)
        altered['domain_independence_admitted'] = 0
        self.assertFalse(module.exact_scope(altered, scope))
        self.assertTrue(module.exact_scope(scope, scope))

    def test_original_gate_is_not_admission(self):
        with self.assertRaises(ValueError):
            module.validate({'schema': 'G2', 'passed': True}, {}, {}, {})


if __name__ == '__main__':
    unittest.main()
