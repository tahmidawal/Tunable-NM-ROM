"""Global cohort guard: corrupted or incomplete proofs never open evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from seal import sha, verify_validation_bundle


class SealTests(unittest.TestCase):
    def test_complete_and_corrupted_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {'meshes': [256, 512], 'evaluation_seed': 910603}
            checkpoints = {'cp': 'one', 'modcp': 'two', 'film': 'three'}
            old = {'case_name': 'wave_reflective', 'status': 'validation_frozen',
                   'evaluation_opened': False, 'checkpoint_sha256': checkpoints,
                   'provenance': {'commit': 'abc', 'job_id': '123'}, 'config': cfg}
            (root/'handoff.json').write_text(json.dumps(old))
            proofs = {}
            for n in cfg['meshes']:
                name = f'frozen_selection_{n}.json'
                (root/name).write_text(json.dumps({'intervals': n, 'evaluation_opened_at_selection': False,
                    'training_sha256': checkpoints, 'selected_settings': [{'id': 'fixed'}], 'quadrature_sha256': {}}))
                proofs[name] = sha(root/name)
            entry = {'handoff_sha256': sha(root/'handoff.json'), 'selection_proof_sha256': proofs,
                     'source_commit': 'abc', 'validation_job_id': '123', 'evaluation_seed': 910603}
            seal = {'schema': 'modcp-global-validation-seal-v1', 'evaluation_generated': False,
                    'created_utc': 'fixed', 'cases': dict.fromkeys(['burgers2d', 'wave_reflective', 'wave_absorbing'], entry)}
            def check():
                (root/'global_seal.json').write_text(json.dumps(seal))
                return verify_validation_bundle(root, cfg, 'wave_reflective', checkpoints)
            self.assertEqual(set(check()[1]), {256, 512})
            seal['evaluation_generated'] = True
            with self.assertRaises(RuntimeError): check()
            seal['evaluation_generated'] = False
            saved = seal['cases'].pop('wave_absorbing')
            with self.assertRaises(RuntimeError): check()
            seal['cases']['wave_absorbing'] = saved
            (root/'frozen_selection_512.json').write_text('{}')
            with self.assertRaises(RuntimeError): check()


if __name__ == '__main__':
    unittest.main()
