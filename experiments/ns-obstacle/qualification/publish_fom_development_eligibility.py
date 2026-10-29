"""Publish development-only eligibility from reviewed, retained lift controls.

This is metadata publication, not native computation or comparator selection.
The existing consumer replays all six genuine retained controls before publication.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(plan_path, output):
    plan = json.loads(plan_path.read_text())
    roots = {key: Path(value).resolve() for key, value in plan['roots'].items()}
    sys.path[:0] = [str(roots['rom'] / 'fom_native'), str(roots['rom'])]
    from candidate_lift import CandidateStokesLift, configuration_sha256
    truth = roots['coordinator'] / plan['truth_admission']['path']
    assert sha(truth) == plan['truth_admission']['sha256']
    expected = ['p3g3h1', 'p3g3h2', 'p3g3h3', 'p4g4h1', 'p4g4h2', 'p4g4h3']
    assert [row['id'] for row in plan['candidates']] == expected
    if output.exists():
        raise FileExistsError('Fresh publication directory required')
    records = {}
    with tempfile.TemporaryDirectory(prefix='eligibility-review-', dir=plan_path.parent) as scratch:
        scratch = Path(scratch)
        for row in plan['candidates']:
            config_path = roots[row['config_lane']] / row['config_path']
            assert sha(config_path) == row['config_file_sha256']
            config = json.loads(config_path.read_text())
            quartet = roots[row['quartet_lane']] / row['quartet_path']
            review_path = roots['coordinator'] / row['review_path']
            assert sha(review_path) == row['review_sha256']
            assert json.loads(review_path.read_text())['passed'] is True
            meta = json.loads((quartet / 'common_stokes_lift.json').read_text())
            assert meta['mesh_sha256'] == row['mesh_sha256']
            assert meta['layout_sha256'] == row['layout_sha256']
            record = {
                'schema': 'ns-obstacle-fom-candidate-lift-eligibility-v1',
                'candidate_id': row['id'],
                'role': 'development_calibration_only',
                'native_control_reviewed': True,
                'physical_reference_admitted': False,
                'review_sha256': row['review_sha256'],
                'truth_reference_admission_sha256': sha(truth),
                'configuration_sha256': configuration_sha256(config),
                'configuration_file_sha256': sha(config_path),
                'field_rule_mode': row['field_rule_mode'],
                'directory': row['id'],
                'allowed_Re': [16.875, 23.125],
                'selection_frozen': False,
                'matched_accuracy': False,
                'bank_matching_performed': False,
            }
            for name, key in [('common_stokes_lift.json', 'manifest_sha256'),
                              ('common_stokes_lift.npz', 'coefficient_sha256'),
                              ('control.json', 'native_control_sha256'),
                              ('lift_scaling_arrays.npz', 'native_arrays_sha256')]:
                assert sha(quartet / name) == row['quartet_sha256'][name]
                record[key] = row['quartet_sha256'][name]
            raw = (json.dumps(record, indent=2) + '\n').encode()
            (scratch / row['id']).symlink_to(quartet, target_is_directory=True)
            candidate_path = scratch / (row['id'] + '.json')
            candidate_path.write_bytes(raw)
            lift = CandidateStokesLift.from_eligibility(
                candidate_path, row['mesh_sha256'], row['layout_sha256'], .2, .1,
                config, expected_truth_admission_sha256=sha(truth))
            assert all(lift.allows_Re(re) for re in (16.875, 23.125))
            assert not any(lift.allows_Re(re) for re in (15, 16, 18, 20, 22, 24, 25))
            records[row['id'] + '.json'] = raw
    output.mkdir()
    for name, raw in records.items():
        (output / name).write_bytes(raw)
    receipt = {'schema': 'ns-obstacle-fom-development-eligibility-publication-v1',
               'plan_sha256': sha(plan_path), 'publisher_sha256': sha(__file__),
               'genuine_existing_consumer_passed': True, 'fresh_native_work': False,
               'development_only': True, 'calibration_execution_released': False,
               'records': {name: hashlib.sha256(raw).hexdigest() for name, raw in records.items()}}
    (output / 'PUBLICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.plan, args.output), indent=2))
