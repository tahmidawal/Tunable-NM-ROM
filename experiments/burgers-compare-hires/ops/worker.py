"""Train the four validation-selected operator configurations at the target mesh, one after another,
each on the 256^2 panel's equal wall budget (burgers-compare-hires DESIGN.md section 4).

    python ops/worker.py <spec.json>

The spec names the arms, their configs and the per-arm wall budget. Every arm is trained by the unchanged
(micro-batching aside) ops/train.py on data/train (128 pinned cases) with early stopping / selection on
data/validation (32 pinned cases). A failed arm is recorded and the worker moves on, so one family cannot take
the others down with it. No timing and no speed number comes out of this job.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PY = os.environ.get('OPS_PY', '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python')
SPEC = json.loads(Path(sys.argv[1]).read_text())
assert os.environ.get('SLURM_JOB_ID')
out = Path('out')
out.mkdir()
records = []
for arm in SPEC['arms']:
    start = time.time()
    log = Path('logs') / f"{arm['name']}.log"
    with log.open('w') as fh:
        code = subprocess.call([PY, 'ops/train.py', '--train-index', 'data/train/index.json',
                                '--validation-index', 'data/validation/index.json', '--config', arm['config'],
                                '--out', f"out/{arm['name']}", '--wall-seconds', str(SPEC['per_arm_seconds'])] + (['--smoke'] if SPEC.get('smoke') else []),
                               stdout=fh, stderr=subprocess.STDOUT)
    rec = dict(arm=arm['name'], config=arm['config'], exit_code=code, started_unix=start, ended_unix=time.time(),
               job_id=os.environ['SLURM_JOB_ID'])
    res = out / arm['name'] / 'result.json'
    if res.exists():
        r = json.loads(res.read_text())
        rec.update(epochs=r['epochs_completed'], stop_reason=r['stop_reason'],
                   validation_mean_case_max=r['validation']['mean_case_max'],
                   validation_worst_case_max=r['validation']['worst_case_max'],
                   micro_batch_final=r.get('micro_batch_final'), best_checkpoint_sha256=r['best_checkpoint_sha256'])
    records.append(rec)
    (out / 'worker.json').write_text(json.dumps(dict(spec=SPEC, arms=records), indent=1) + '\n')
    print(json.dumps(rec), flush=True)
print('WORKER FINISHED', flush=True)
