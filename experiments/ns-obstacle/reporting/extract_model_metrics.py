"""Extract scalar ROM evidence for reporting; never infer scientific admission.

Reads only explicitly supplied completed JSON records, not validation inputs or
large field archives. Array descriptors remain provenance references. Independent
result/retention review is a separate prerequisite for publishing conclusions.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

METRICS = (
    'velocity_relative', 'pressure_scaled', 'drag_relative',
    'transverse_absolute', 'divergence', 'mass', 'boundary_trace',
    'velocity_gradient_relative',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(paths):
    rows, inputs, decisions = [], [], []
    for path in paths:
        path = Path(path).resolve()
        wrapped = json.loads(path.read_text())
        record = wrapped.get('record', wrapped)
        inputs.append({'path': str(path), 'sha256': digest(path),
                       'array_file': wrapped.get('array_file'),
                       'array_sha256': wrapped.get('array_sha256')})
        if not isinstance(record, dict):
            raise ValueError(f'Expected record object: {path}')
        if 'development_projection' in record:
            for scores in record['development_projection']:
                rows.append(dict(phase='development_projection',
                    model=scores['model'], Re=scores['Re'],
                    method='weighted_projection_not_PDE_solution',
                    eligible_accuracy=None, status='representation_diagnostic',
                    record_path=str(path),
                    **{key: scores.get(key) for key in METRICS}))
        if 'methods' in record:
            phase = 'validation' if '_validation_' in path.stem else 'development'
            for method, item in record['methods'].items():
                invocation = item.get('invocation', {})
                scores = item.get('scores', {})
                rows.append(dict(phase=phase, model=record['model'], Re=record['Re'],
                    method=method, eligible_accuracy=item.get('eligible_accuracy'),
                    status=item.get('failure', item.get('status',
                        invocation.get('work', {}).get('status', 'not_recorded'))),
                    record_path=str(path),
                    **{key: scores.get(key) for key in METRICS}))
        for key in ('construction_failures', 'bank_development_gate',
                    'bank_online_gate', 'bank_gate', 'frozen_choices', 'choices'):
            if key in record:
                decisions.append({'record_path': str(path), 'key': key,
                                  'value': record[key]})
    return {'schema': 'ns-obstacle-report-scalar-evidence-v1',
            'scientific_admission_inferred': False,
            'generator_sha256': digest(Path(__file__)),
            'input_records': inputs, 'rows': rows, 'decisions': decisions,
            'normalization': {
                'velocity_relative': 'L2 velocity error / reference velocity L2 norm',
                'pressure_scaled': 'L2 pressure error / (U^2 sqrt(physical volume)); no gauge shift',
                'drag_relative': 'absolute drag coefficient error / absolute reference drag coefficient',
                'transverse_absolute': 'absolute lift coefficient error',
                'divergence': 'D times RMS divergence / U',
                'mass': 'absolute outlet-minus-inlet flux / absolute inlet flux',
                'boundary_trace': 'maximum dimensionless essential-boundary velocity mismatch',
                'velocity_gradient_relative': 'relative velocity-gradient L2 error; diagnostic only',
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.record)
    if not result['rows'] and not result['decisions']:
        raise ValueError('No model metrics or decisions in supplied records')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'scalar-evidence.json').write_text(
        json.dumps(result, indent=2, allow_nan=False)+'\n')
    columns = ['phase', 'model', 'Re', 'method', 'eligible_accuracy', 'status',
               *METRICS, 'record_path']
    with (args.output/'case-metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result['rows'])


if __name__ == '__main__':
    main()
