"""Frozen-checkpoint resolution ladder in one allocation (DESIGN §A5).

No training. Rebuilds the matched cohort from the same anchors, then runs the ladder and its
timing. Kept separate from `worker_second.py` so the training path is untouched.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()
assert str(ROOT).startswith('/cluster/tufts/paralab/tawal01/no_second_20260917/')
assert os.environ.get('SLURM_JOB_ID')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
SPEC = json.loads(Path(sys.argv[1]).read_text())
(ROOT / 'out').mkdir()
records = []


def run(name, command, allow_failure=False):
    start = time.time()
    with (ROOT / 'logs' / f'{name}.log').open('w') as log:
        code = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT).wait()
    records.append(dict(task=name, command=command, exit_code=code, started_unix=start,
                        ended_unix=time.time(), job_id=os.environ['SLURM_JOB_ID']))
    (ROOT / 'out/worker.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records[-1]), flush=True)
    if code and not allow_failure:
        sys.exit(code)
    return code


run('resolution-smoke', [PY, 'code/smoke_resolution.py', '--output', 'out/resolution-smoke.json'])
run('cohort', [PY, 'code/prepare_diagnosis_cohort.py',
               '--reference-index', 'data/refinement/index.json',
               '--train-index', 'data/train/index.json', '--out', 'data/diagnosis-cohort'])
run('resolution', [PY, 'code/resolution.py', '--checkpoints', 'code/checkpoints.json',
                   '--validation-index', 'data/validation/index.json',
                   '--cohort-index', 'data/diagnosis-cohort/index.json',
                   '--out', 'out/resolution', '--repetitions', '30', '--burn-in', '20',
                   '--timing-cases', '8'])
print('WORKER FINISHED', flush=True)
