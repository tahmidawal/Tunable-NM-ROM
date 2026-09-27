"""Bounded Burgers FNO queue: smokes, three capacities, one learning-rate
refinement of the best capacity, then same-job complete-query timing.

Training is deliberately not resumable. Every training run is contained in this
one allocation and is bounded by its own wall budget and by a global deadline
that always reserves time for the timing block. An interrupted run must be
described as truncated at its recorded epoch, never as automatically resumed.
"""
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
GLOBAL_SECONDS = 15300.
TIMING_RESERVE = 700.
BUDGETS = {'small': 2100., 'medium': 2700., 'large': 3900., 'refine': 2700.}
REFINEMENT_LEARNING_RATE = 3e-4
TRAIN = 'data/train/index.json'
VALIDATION = 'data/validation/index.json'
started = time.monotonic()
child, stopping = None, False


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGUSR1)


signal.signal(signal.SIGUSR1, stop)
signal.signal(signal.SIGTERM, stop)
(ROOT / 'out').mkdir()
records = []


def remaining():
    return GLOBAL_SECONDS - (time.monotonic() - started)


def run(name, command, allow_failure=False):
    global child
    start = time.time()
    with (ROOT / 'logs' / f'{name}.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    records.append(dict(task=name, command=command, exit_code=code, started_unix=start,
                        ended_unix=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                        remaining_global_seconds=remaining()))
    (ROOT / 'out/worker.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records[-1]), flush=True)
    if code and not allow_failure:
        sys.exit(code)
    return code


def train(name, config, budget):
    wall = min(budget, remaining() - TIMING_RESERVE)
    if wall < 300:
        records.append(dict(task=name, skipped='insufficient remaining budget',
                            remaining_global_seconds=remaining()))
        (ROOT / 'out/worker.json').write_text(json.dumps(records, indent=2) + '\n')
        return False
    run(name, [PY, 'code/train.py', '--train-index', TRAIN, '--validation-index', VALIDATION,
               '--config', str(config), '--out', f'out/fno-{name}', '--wall-seconds', f'{wall:.0f}'])
    return True


run('precision', [PY, 'code/smoke.py', '--output', 'out/precision.json'])
if not stopping:
    run('training-smoke', [PY, 'code/training_smoke_burgers.py', '--out', 'out/training-smoke'])
trained = []
for size in ('small', 'medium', 'large'):
    if stopping:
        break
    if train(size, f'code/configs/burgers/{size}.json', BUDGETS[size]):
        trained.append(size)

scores = {}
for size in trained:
    result = ROOT / f'out/fno-{size}/result.json'
    if result.exists():
        scores[size] = json.loads(result.read_text())['validation']['mean_case_max']
if scores and not stopping and remaining() - TIMING_RESERVE >= 1200:
    best = min(scores, key=scores.get)
    config = json.loads((ROOT / f'code/configs/burgers/{best}.json').read_text())
    config['learning_rate'] = REFINEMENT_LEARNING_RATE
    path = ROOT / 'out/refine-config.json'
    path.write_text(json.dumps(dict(config, refines_capacity=best,
                                    selected_by='validation mean case-maximum fixed-initial error'), indent=2) + '\n')
    if train('refine', 'out/refine-config.json', BUDGETS['refine']):
        trained.append('refine')
(ROOT / 'out/capacity-selection.json').write_text(json.dumps(dict(
    capacity_scores=scores, refinement_learning_rate=REFINEMENT_LEARNING_RATE,
    trained=trained, stopped_by_signal=stopping), indent=2) + '\n')

run('timing', [PY, 'code/timing.py', '--train-index', TRAIN, '--validation-index', VALIDATION,
               '--runs', 'out', '--out', 'out/timing', '--repetitions', '30', '--burn-in', '20'],
    allow_failure=True)
print('WORKER FINISHED', flush=True)
