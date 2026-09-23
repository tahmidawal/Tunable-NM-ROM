"""Extract, from the burgers-bank-knob lane's collected result.json (gitignored there), the per-case evolved errors of
the selected arms and the lane's own timings, into lane-ref/burgers-<L>.json (with the source sha256).
    python make_lane_ref.py <L> ...
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BK = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers-bank-knob/experiments/burgers-bank-knob')
for L in sys.argv[1:]:
    run = {'256': 'bk256b', '512': 'bk512b', '1024': 'bk1024b', '2048': 'bk2048b', '4096': 'bk4096b'}[L]
    rp = BK / 'runs' / run / 'archive' / 'output' / 'result.json'
    R = json.loads(rp.read_text())
    s = json.loads((BK / 'checks' / f'bk{L}-summary.json').read_text())
    sel = s['selection']
    names = [sel['accurate']['name'], sel['fast']['name']]
    errs = {}
    for nm in names:
        rows = sorted((x for x in R['quick'] if x['name'] == nm), key=lambda x: x['case'])
        errs[nm] = [x['same_grid_evolved'] for x in rows]
    tim = {}
    for x in R['invocations']:
        tim.setdefault(x['name'], []).append(x['gpu_seconds'])
    out = dict(mesh=int(L), source=str(rp), source_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
               job_id=R.get('job_id'), gpu=R.get('gpu'), commit=R.get('commit'),
               summary=f'checks/bk{L}-summary.json', summary_sha256=hashlib.sha256((BK / 'checks' / f'bk{L}-summary.json').read_bytes()).hexdigest(),
               accurate=names[0], fast=names[1], per_case_evolved=errs,
               lane_median_gpu_ms={k: 1e3 * float(np.median(v)) for k, v in tim.items()},
               lane_selection=dict(accurate=sel['accurate'], fast=sel['fast']))
    Path(f'lane-ref/burgers-{L}.json').write_text(json.dumps(out, indent=1) + '\n')
    Path(f'lane-ref/burgers-{L}-errors.json').write_text(json.dumps(errs, indent=1) + '\n')
    Path(f'lane-ref/burgers-{L}-selection.json').write_text(json.dumps(dict(accurate=names[0], fast=names[1], source=out['summary'], source_sha256=out['summary_sha256']), indent=1) + '\n')
    print(L, names, {k: [round(100 * e, 4) for e in v] for k, v in errs.items()})
