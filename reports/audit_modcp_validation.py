"""Audit a frozen validation panel before the global evaluation seal exists.

Burgers fields support full independent metrics. Wave validation retains bounded
observations: check their geometry, finiteness and common reference, without
pretending their samples reproduce the saved full-grid physical error.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_modcp_comparison import summarize_input, relative
from modcp_audit import digest
from seal_modcp_validation import entry

ROOT = Path(__file__).resolve().parents[1]


def observation_audit(path, data):
    if data['case_name'] == 'burgers2d':
        return None
    observed, references = [], {}
    cfg = data['config']
    nt = int(round(cfg['end_time']/cfg['observation_dt']))+1
    boundary = 'dirichlet' if data['case_name'] == 'wave_reflective' else 'absorbing'
    for row in data['invocations']:
        if row.get('field_artifact_kind') != 'observation_grid_only_full_metrics_computed_before_subsampling':
            raise ValueError('Unexpected wave validation observation format')
        artifact = path.parent/row['field_artifact']
        with np.load(artifact, allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        n = row['intervals']
        side = n-1 if boundary == 'dirichlet' else n+1
        indices = np.unique(np.linspace(0, side-1, min(65, side), dtype=int))
        if (int(fields['intervals']) != n or str(fields['boundary']) != boundary or
                not np.array_equal(fields['active_axis_indices'], indices)):
            raise ValueError('Wave validation observation geometry differs from its row')
        for key in ('u', 'v', 'truth_u', 'truth_v'):
            values = fields[key]
            if values.dtype != np.float64 or values.shape != (nt, len(indices), len(indices)):
                raise ValueError('Wave validation observation dtype/horizon differs from its row')
            if (key.startswith('truth_') or row['finite']) and not np.isfinite(values).all():
                raise ValueError('Wave validation observation finiteness contradicts its record')
        reference = {key: hashlib.sha256(fields[key].tobytes()).hexdigest() for key in ('truth_u', 'truth_v')}
        reference['speed'] = float(fields['speed'])
        identity = row['intervals'], row['case']
        if identity in references and references[identity] != reference:
            raise ValueError('Wave validation observations use different references for one case')
        references[identity] = reference
        observed.append(dict(path=relative(artifact), sha256=digest(artifact), intervals=n,
                             case=row['case'], configuration=row['configuration']))
    return dict(scope='Saved observation integrity only; full-grid validation error cannot be recomputed.',
                passed=True, files_checked=len(observed), files=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.output.read_text()) if args.output.exists() else {
        'scope': 'Validation selection, quadrature identities, and saved-field audit before evaluation.', 'cases': {}}
    for source in args.input:
        source = source.resolve()
        data = json.loads(source.read_text())
        case, proof = entry(source)
        summary = summarize_input(source)
        observations = observation_audit(source, data)
        result['cases'][case] = dict(passed=True, seal_candidate=proof, summary=summary, observations=observations)
        # Preserve completed independent panel audits if a later panel fails.
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(case=case, passed=True,
                              full_fields=len(summary['field_audits']),
                              observation_files=observations['files_checked'] if observations else 0)), flush=True)


if __name__ == '__main__':
    main()
