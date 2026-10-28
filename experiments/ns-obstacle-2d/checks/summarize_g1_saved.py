"""Extract G1 observations from retained files; does not create admission gates."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(root):
    results = root / 'results'
    summary = json.loads((results / 'smoke_summary.json').read_text())
    output = {key: summary[key] for key in (
        'gate', 'checks', 'passed', 'run_exit_code', 'stage_exit_codes',
        'G2_steady_reference_admitted', 'training_released', 'nongating')}
    output['scope'] = 'saved G1 control observations; no new thresholds or reference admission'
    output['cases'] = {}
    for name in ('channel', 'manufactured', 'dfg'):
        directory = results / name
        result = json.loads((directory / 'result.json').read_text())
        certificate = json.loads((directory / 'normal_orientation.json').read_text())
        boundary = json.loads((directory / 'boundary_rule.json').read_text())
        normal_error = max(float(np.linalg.norm(
            np.asarray(row['normal']) - row['normal_orientation_factor'] * np.asarray(row['raw_normal'])
        )) for row in boundary)
        metrics = result['metrics']
        with np.load(directory / 'full_rule.npz', allow_pickle=False) as rule:
            rule_layout = {key: {'shape': list(rule[key].shape), 'dtype': str(rule[key].dtype),
                                'bytes': rule[key].nbytes} for key in rule.files}
        output['cases'][name] = {
            'result_sha256': digest(directory / 'result.json'),
            'normal_certificate_sha256': digest(directory / 'normal_orientation.json'),
            'normal_certificate_hash_matches_result': digest(directory / 'normal_orientation.json')
                == result['normal_orientation']['certificate_sha256'],
            'status': result['status'], 'provenance': result['provenance'],
            'geometry': result['geometry'],
            'normal_certificate': {key: certificate[key] for key in (
                'passed', 'factors', 'minimum_observed_absolute_witness_cosine_by_label',
                'returned_CF_point_checks', 'analytic_checks', 'constant_test_error',
                'divergence_theorem', 'closed_components')},
            'maximum_ordinary_volume_facet_moment_relative_difference': max(
                moment['relative_difference'] for label in certificate['moment_checks'].values()
                for moment in label['moments'].values()),
            'boundary_export_samples': len(boundary),
            'boundary_export_normal_equals_factor_times_raw_maximum_error': normal_error,
            'observations_not_additional_G1_gates': {
                **{key: metrics[key] for key in ('mass', 'mass_scope', 'global_boundary_flux',
                                                 'divergence', 'boundary_trace')},
                'corrected_energy': metrics['energy']['corrected'],
                'corrected_gradient_energy': metrics['gradient_energy']['corrected'],
                'corrected_momentum': [row['corrected'] for row in metrics['momentum']],
                'corrected_gradient_momentum': [row['corrected'] for row in metrics['gradient_momentum']],
            },
            'full_rule_layout': rule_layout,
        }
    output['preflight_passed'] = summary['preflight']['passed']
    output['preflight_normal_certificate_sha256'] = digest(results / 'preflight-normal-orientation.json')
    output['limits'] = [
        'DFG remains a coarse diagnostic, with force/probe/balance/discrete-divergence discrepancies retained.',
        'Manufactured inlet/outlet-only mass ratio is ill-conditioned at its nearly zero inlet; use recorded all-boundary flux scope.',
        'Manufactured trace is error against the analytic target, including FE trace approximation; not the G2 prescribed-FE trace certificate.',
        'Text mesh persistence has not established full-precision clean-process reload. G2 binary persistence control remains required.',
        'No comparisons of elapsed wall times across allocations.',
    ]
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.root), indent=2) + '\n')
