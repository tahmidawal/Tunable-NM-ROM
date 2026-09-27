"""Run a fixed, useful task queue sequentially inside one Slurm allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path.cwd()
assert str(ROOT).startswith('/cluster/tufts/paralab/tawal01/no_audit_20260914/')
assert os.environ.get('SLURM_JOB_ID')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
child, stopping = None, False


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGUSR1)


signal.signal(signal.SIGUSR1, stop)
signal.signal(signal.SIGTERM, stop)
(ROOT/'out').mkdir()
records = []
tasks = [
    ('precision', [PY, 'code/smoke.py', '--output', 'out/precision.json']),
    ('training-smoke', [PY, 'code/training_smoke.py', '--out', 'out/training-smoke']),
]
for size in ('small', 'medium', 'large'):
    tasks.append((size, [PY, 'code/train.py', '--train-index', 'data/train/index.json',
        '--validation-index', 'data/validation/index.json', '--config', f'code/configs/fno-{size}.json',
        '--out', f'out/fno-{size}', '--wall-seconds', '7200']))
for name, command in tasks:
    if stopping:
        break
    start = time.time()
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    records.append(dict(task=name, command=command, exit_code=code, started_unix=start,
                        ended_unix=time.time(), job_id=os.environ['SLURM_JOB_ID']))
    (ROOT/'out/worker.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records[-1]), flush=True)
    if code:
        sys.exit(code)
print('WORKER FINISHED', flush=True)
