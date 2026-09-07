"""Independently check Poisson saved output hashes and physical/same-grid errors."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(path):
    native = json.loads(path.read_text())
    assert native['complete']
    cfg = native['config']
    obs = cfg['observation_intervals']
    groups = defaultdict(list)
    for row in native['rows']:
        groups[row['intervals'], row['case'], row['arm'], row['tau']].append(row)
    assert len(groups) == len(cfg['intervals'])*cfg['cohort_count']*(2+len(cfg['taus']))
    worst = 0.
    for (n, case, arm, tau), rows in groups.items():
        assert len(rows) == cfg['repetitions']
        assert {row['repetition'] for row in rows} == set(range(cfg['repetitions']))
        field = np.load(path.parent/f'field_n{n}_case{case}_{arm}_tau{tau}.npz')['field']
        digest = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
        discrete = np.load(path.parent/f'field_n{n}_case{case}_dst_tauNone.npz')['field']
        physical = np.load(path.parent/f'reference_case{case}.npz')['observation']
        common = field[::n//obs, ::n//obs]
        measured = dict(physical_error=float(np.linalg.norm(common-physical)/np.linalg.norm(physical)),
                        same_grid_error=float(np.linalg.norm(field-discrete)/np.linalg.norm(discrete)))
        for row in rows:
            # Repeated fields are not all stored: hash identity is required to
            # use the preserved field as the output evidence for another call.
            assert row['field_sha256'] == digest
            for key, value in measured.items():
                np.testing.assert_allclose(value, row[key], rtol=3e-12, atol=2e-14)
                worst = max(worst, abs(value-row[key]))
            component_total = sum(row[key] for key in ('input_seconds', 'projection_init_seconds', 'solver_seconds', 'output_seconds'))
            np.testing.assert_allclose(component_total, row['total_seconds'], rtol=1e-12, atol=1e-14)
    return dict(scope='Saved output hashes, independently recomputed errors and repetition accounting only.',
                source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                job_id=native['provenance']['job_id'], source_commit=native['provenance']['commit'],
                verified_invocations=len(native['rows']), preserved_fields=len(groups),
                all_repetitions_match_preserved_output_hash=True,
                maximum_metric_disagreement=worst,
                reference_limit='Nested reference differences are empirical evidence; this audit does not certify continuum error.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
