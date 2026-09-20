"""Coordinator's independent worst-case field checks, without solver imports.

Uses NumPy norms and the explicit report manifest. Owners separately check all
saved fields and numerical operators. Writes only the main reports directory.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reports/2026-09-20-3d-paper-inputs.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_fields(pred, truth, initial=False):
    pred = pred.reshape(len(pred), -1)
    truth = truth.reshape(len(truth), -1)
    norm = np.linalg.norm(truth, axis=1)
    if initial:
        norm = np.full(len(norm), norm[0])
    return np.linalg.norm(pred-truth, axis=1)/np.maximum(norm, 1e-300)


def main():
    output = dict(passed=True, scope='One independently recomputed worst-case field per method and mesh; no solver imports or independent retraining.',
                  inputs={str(MANIFEST.relative_to(ROOT)): sha(MANIFEST),
                          str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}, checks=[])
    for entry in json.loads(MANIFEST.read_text())['runs']:
        path = ROOT / entry['result']
        data = json.loads(path.read_text())
        output['inputs'][entry['result']] = sha(path)
        groups = defaultdict(list)
        adapter = entry['adapter']
        if adapter == 'ns_representation':
            artifact = path.parent / 'representation_fields.npz'
            with np.load(artifact) as f:
                errors = relative_fields(f['prediction'], f['truth'])
            observed = dict(mean=float(errors.mean()), median=float(np.median(errors)), worst=float(errors.max()))
            reported = {key: data['head_fit'][key] for key in observed}
            discrepancies = {key: abs(value-reported[key]) for key, value in observed.items()}
            passed = max(discrepancies.values()) < 1e-11
            output['passed'] &= passed
            output['inputs'][str(artifact.relative_to(ROOT))] = sha(artifact)
            output['checks'].append(dict(pde=entry['pde'], attempt=entry['attempt'], method='all saved neural reconstruction fields',
                                        observed=observed, reported=reported, discrepancies=discrepancies, passed=passed))
            continue
        if adapter == 'ns_trajectory':
            from scipy.signal import resample
            records_path = ROOT / entry['invocations']
            output['inputs'][entry['invocations']] = sha(records_path)
            records = json.loads(records_path.read_text())
            reference = path.parent / 'timing_references.npz'
            prediction = path.parent / 'timed_fields.npz'
            with np.load(reference) as f:
                same, fine = f['same_grid'], f['fine_grid']
            for row in records:
                groups[row['method']].append(row)
            with np.load(prediction) as fields:
                for method, group in sorted(groups.items()):
                    row = max(group, key=lambda r: max(r['same_grid_errors'][1:]))
                    case = row['case']
                    prefix = f'{method}__case{case}'
                    matches = [key for key in fields.files if key == prefix or key.startswith(prefix + '__')]
                    selected = []
                    for key in matches:
                        pred = fields[key]
                        h = hashlib.sha256(np.ascontiguousarray(pred).view(np.uint8)).hexdigest()
                        if h == row['field_sha256']:
                            selected.append(pred)
                    assert selected, (method, case)
                    pred = selected[0]
                    errors = relative_fields(pred, same[case], initial=True)
                    lifted = pred
                    for axis in (-3, -2, -1):
                        lifted = resample(lifted, fine.shape[axis], axis=axis)
                    physical = np.linalg.norm((lifted-fine[case]).reshape(len(pred),-1),axis=1)/np.linalg.norm(fine[case,0])
                    observed = dict(worst_evolved=float(errors[1:].max()), worst_all=float(errors.max()),
                                    initial_error=float(errors[0]), physical_evolved=float(physical[1:].max()))
                    reported = dict(worst_evolved=max(row['same_grid_errors'][1:]), worst_all=max(row['same_grid_errors']),
                                    initial_error=row['same_grid_errors'][0], physical_evolved=max(row['fine_grid_errors'][1:]))
                    discrepancies = {key: abs(value-reported[key]) for key,value in observed.items()}
                    passed = all(np.isfinite(v) and v < 1e-11 for v in discrepancies.values())
                    output['passed'] &= passed
                    output['checks'].append(dict(pde=entry['pde'],attempt=entry['attempt'],method=method,
                        mesh=data['config']['n'],case=case,observed=observed,reported=reported,
                        discrepancies=discrepancies,passed=passed))
            for artifact in (reference,prediction):
                output['inputs'][str(artifact.relative_to(ROOT))] = sha(artifact)
            continue
        for row in data['invocations'] + data.get('operator_invocations', []):
            groups[row.get('intervals', data['config'].get('nodes')), row['method']].append(row)
        for (mesh, method), group in sorted(groups.items()):
            def score(row):
                if adapter == 'burgers':
                    return row['worst_evolved']
                if adapter == 'heat':
                    return row['same_grid']['current_evolved']
                return row['same_grid_error']
            row = max(group, key=score)
            case = row['case']
            if adapter == 'burgers':
                prediction = path.parent / row['artifact']
                reference = path.parent / f'reference_case{case}.npz'
                with np.load(prediction) as f, np.load(reference) as ref:
                    errors = relative_fields(f['fields'], ref['fields'], initial=True)
                observed = {'worst_evolved': float(errors[1:].max()),
                            'worst_all': float(errors.max()), 'initial_error': float(errors[0])}
                reported = {key: row[key] for key in observed}
            else:
                prediction = path.parent / 'fields' / f'N{mesh}_case{case}_{method}.npz'
                reference = path.parent / 'fields' / f'N{mesh}_case{case}_reference.npz'
                with np.load(prediction) as f, np.load(reference) as ref:
                    pred, truth = f['prediction'], ref['same_grid']
                    if adapter == 'heat':
                        errors = relative_fields(pred, truth)
                        physical = relative_fields(pred, ref['physical'])
                        observed = dict(current_evolved=float(errors[1:].max()),
                                        current_all=float(errors.max()), initial_fit=float(errors[0]),
                                        physical_evolved=float(physical[1:].max()))
                        reported = {key: row['same_grid'][key] for key in observed if key != 'physical_evolved'}
                        reported['physical_evolved'] = row['physical']['current_evolved']
                    else:
                        observed = dict(same_grid_error=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),
                                        physical_error=float(np.linalg.norm(pred-ref['physical'])/np.linalg.norm(ref['physical'])))
                        reported = {key: row[key] for key in observed}
            discrepancies = {key: abs(value-reported[key]) for key, value in observed.items()}
            passed = all(np.isfinite(value) and value < 1e-11 for value in discrepancies.values())
            output['passed'] &= passed
            for artifact in (prediction, reference):
                output['inputs'][str(artifact.relative_to(ROOT))] = sha(artifact)
            output['checks'].append(dict(pde=entry['pde'], attempt=entry['attempt'], method=method, mesh=mesh,
                                        case=case, observed=observed, reported=reported,
                                        discrepancies=discrepancies, passed=passed))
    dest = ROOT / 'reports/2026-09-20-3d-field-audit.json'
    dest.write_text(json.dumps(output, indent=2)+'\n')
    print(f"Independent worst-field checks: {len(output['checks'])}; passed={output['passed']}")
    if not output['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
