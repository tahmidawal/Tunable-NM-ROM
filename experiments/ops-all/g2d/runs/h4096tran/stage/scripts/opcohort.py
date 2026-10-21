"""g2d: write the job's evaluation cohort in the operator contract, right after the driver's truth solve.

Copied from burgers-compare-hires cmp.py @ 07c0e3e8 lines 202-217 (the `opcohort` block) and made a function so
b2speed.py / bankknob.py can call it with their own `truth`, `physical`, `n0`: the operators are scored against
exactly the same in-job same-grid reference (`fft_tight`) as the NM-ROM and FOM arms. The dev6 inputs written here
are the target's t=0 slice (= the supplied initial field, exactly as cmp.py did).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

TIMES = np.array([0., .05, .1, .15, .2, .25])


def write(out, truth, physical, n0, cfg, L):
    cohort = Path(out) / 'opcohort'
    cohort.mkdir(exist_ok=True)
    records = []
    for c in range(len(physical)):
        target = np.asarray(truth[c])[:, None].astype(np.float64)
        path = cohort / f'burgers-dev-{c:05d}.npz'
        np.savez(path, input=np.ascontiguousarray(target[0]), target=target,
                 parameters=np.array([float(physical[c, 4])]), times=TIMES)
        records.append(dict(case_id=f'burgers-dev-{c:05d}', split='development', case_index=c,
                            seed=int(cfg['eval_seed'] if c < cfg['eval_cases'] else cfg['eval_fresh_seed']) * 100 + c,
                            path=path.name, mesh=L, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            n0=float(n0[c]),
                            generation_descriptors=dict(zip(('cx', 'cy', 'width', 'amplitude', 'nu'), map(float, physical[c])))))
    (cohort / 'index.json').write_text(json.dumps(dict(
        schema_version=1, pde='burgers', split='development', count=len(records), mesh=L, complete=True,
        derivation=f'dev6 with this job\'s own same-grid fft_tight solve at {L} intervals, in the operator contract',
        records=records), indent=1) + '\n')
    print('OPCOHORT written', len(records), flush=True)
