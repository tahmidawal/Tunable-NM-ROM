"""Summarise a collected reference job (refjob.py) into checks/<attempt>-reference.json for the report.

    python ref_summary.py runs/<attempt>/archive
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
arc = Path(sys.argv[1])
R = json.loads((arc / 'output/result.json').read_text())
logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
bad = []
for c in R['cases']:
    f = arc / 'output' / f"ref_{c['ref']}_{c['cohort']}_{c['case']:03d}.npz"
    z = np.load(f)['f257']
    if hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest() != c['f257_sha256'] or z.shape != (6, 257, 257):
        bad.append(f.name)
out = dict(attempt=R['config']['attempt'], job_id=R['job_id'], commit=R['commit'], gpu=R['gpu'],
           backend_ok='jax_backend=gpu' in logs, cohorts=R['config']['cohorts'], mesh=R['config']['mesh'],
           refs=R['config']['refs'], cases=len(R['cases']), all_accepted=R.get('all_accepted'),
           complete=R.get('complete'), elapsed_seconds=R.get('elapsed_seconds'),
           worst_residual={t: max(c['max_relative_residual'] for c in R['cases'] if c['ref'] == t) for t in ('ST', 'S')},
           median_seconds={t: float(np.median([c['seconds'] for c in R['cases'] if c['ref'] == t])) for t in ('ST', 'S')},
           file_hash_mismatches=bad, cohort_sha256=R['cohort_sha256'])
(HERE / 'checks' / f"{out['attempt']}-reference.json").write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: out[k] for k in ('attempt', 'job_id', 'cases', 'all_accepted', 'worst_residual', 'file_hash_mismatches')}))
