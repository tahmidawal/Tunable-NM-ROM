"""Equal-wall-budget Burgers FNO training, matched-cohort scoring and timing.

The bounded first screen (`fno_burgers01`) showed every capacity still improving
when its 200-epoch budget ran out, so this job gives each capacity the same,
much larger wall budget and lets early stopping decide, then refines the best
capacity's learning rate. Capacities therefore complete different epoch counts;
that is the deliberate design — equal compute, not equal epochs — and each run's
epoch count and truncation flag are recorded.

Training is still deliberately not resumable. Every run is bounded inside this
one allocation, and the global deadline always reserves time for the matched
cohort evaluation and the timing block.
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
GLOBAL_SECONDS = 14400.
RESERVE = 900.
PER_CAPACITY_SECONDS = 3000.
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


def save():
    (ROOT / 'out/worker.json').write_text(json.dumps(records, indent=2) + '\n')


def run(name, command, allow_failure=False):
    global child
    start = time.time()
    with (ROOT / 'logs' / f'{name}.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    records.append(dict(task=name, command=command, exit_code=code, started_unix=start,
                        ended_unix=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                        remaining_global_seconds=remaining()))
    save()
    print(json.dumps(records[-1]), flush=True)
    if code and not allow_failure:
        sys.exit(code)
    return code


def train(name, config, budget):
    wall = min(budget, remaining() - RESERVE)
    if wall < 300:
        records.append(dict(task=name, skipped='insufficient remaining budget',
                            remaining_global_seconds=remaining()))
        save()
        return False
    run(name, [PY, 'code/train.py', '--train-index', TRAIN, '--validation-index', VALIDATION,
               '--config', str(config), '--out', f'out/fno-{name}', '--wall-seconds', f'{wall:.0f}'])
    return True


run('precision', [PY, 'code/smoke.py', '--output', 'out/precision.json'])
run('training-smoke', [PY, 'code/training_smoke_burgers.py', '--out', 'out/training-smoke'])
run('cohort', [PY, 'code/prepare_diagnosis_cohort.py',
               '--reference-index', 'data/refinement/index.json',
               '--train-index', TRAIN, '--out', 'data/diagnosis-cohort'])

trained = []
for size in ('small', 'medium', 'large'):
    if stopping:
        break
    if train(size, f'code/configs/burgers-long/{size}.json', PER_CAPACITY_SECONDS):
        trained.append(size)

scores = {}
for size in trained:
    result = ROOT / f'out/fno-{size}/result.json'
    if result.exists():
        scores[size] = json.loads(result.read_text())['validation']['mean_case_max']
if scores and not stopping and remaining() - RESERVE >= 1200:
    best = min(scores, key=scores.get)
    config = json.loads((ROOT / f'code/configs/burgers-long/{best}.json').read_text())
    config['learning_rate'] = REFINEMENT_LEARNING_RATE
    (ROOT / 'out/refine-config.json').write_text(json.dumps(
        dict(config, refines_capacity=best,
             selected_by='validation mean case-maximum fixed-initial error'), indent=2) + '\n')
    if train('refine', 'out/refine-config.json', PER_CAPACITY_SECONDS):
        trained.append('refine')
(ROOT / 'out/capacity-selection.json').write_text(json.dumps(dict(
    capacity_scores=scores, refinement_learning_rate=REFINEMENT_LEARNING_RATE,
    per_capacity_wall_seconds=PER_CAPACITY_SECONDS, trained=trained,
    stopped_by_signal=stopping,
    design='equal wall budget per capacity with early stopping; epoch counts differ by design'), indent=2) + '\n')

run('cohort-eval', [PY, 'code/evaluate_cohort.py', '--index', 'data/diagnosis-cohort/index.json',
                    '--runs', 'out', '--out', 'out/diagnosis-cohort'], allow_failure=True)
run('timing', [PY, 'code/timing.py', '--train-index', TRAIN, '--validation-index', VALIDATION,
               '--runs', 'out', '--out', 'out/timing', '--repetitions', '30', '--burn-in', '20'],
    allow_failure=True)
print('WORKER FINISHED', flush=True)
