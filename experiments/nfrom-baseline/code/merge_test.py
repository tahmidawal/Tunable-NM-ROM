"""Merge the per-job test files (amendment A5) into one test.json; refuses gaps, overlaps and mixed GPUs/selections.

  merge_test.py results/test-parts results/test.json
"""
import json, sys
from pathlib import Path
parts = sorted((json.loads(p.read_text()) for p in Path(sys.argv[1]).glob('test-*.json')), key=lambda d: d['rows'][0])
n = parts[0]['selected_total']; pos = 0
for d in parts:
    assert d['rows'][0] == pos and d['selected_total'] == n, d['rows']
    pos = min(d['rows'][1], n)
    assert len(d['results']) == pos - d['rows'][0]
    for r in d['results']:
        r['job_id'] = d['job_id']; r['node'] = d['node']
assert pos == n, (pos, n)
for k in ('gpu', 'selection_sha256', 'cohort_sha256', 'solver', 'backend', 'jax'):
    assert len({d[k] for d in parts}) == 1, k
out = dict(parts[0]); out.pop('rows'); out['job_id'] = ','.join(str(d['job_id']) for d in parts); out['node'] = ','.join(d['node'] for d in parts)
out['results'] = [r for d in parts for r in d['results']]
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
print('merged', len(out['results']), 'rows from jobs', out['job_id'], 'on', out['gpu'])
