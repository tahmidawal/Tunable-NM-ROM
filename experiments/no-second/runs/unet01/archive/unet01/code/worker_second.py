"""Spec-driven equal-wall-budget training for one operator family, in one allocation.

A copy of the parent lane's `worker_burgers_long.py` with the arm list, family,
output prefix and refinement rule read from a job spec (`specs/<name>.json`) so
that every attempt directory documents exactly what it ran. The protocol is the
parent's: every arm receives the same wall budget and early stopping decides;
the validation-selected capacity (mean case-maximum fixed-initial error over the
32 validation cases) is then retrained at the refinement learning rate; the
matched ROM/FOM diagnosis cohort is scored; the same-job complete-query timing
block runs last. Training is deliberately not resumable. Epoch counts differ by
design: equal compute, not equal epochs.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path.cwd()
assert str(ROOT).startswith('/cluster/tufts/paralab/tawal01/no_second_20260917/')
assert os.environ.get('SLURM_JOB_ID')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
SPEC = json.loads(Path(sys.argv[1]).read_text())
GLOBAL_SECONDS = float(SPEC['global_seconds'])
RESERVE = float(SPEC['reserve_seconds'])
PER_ARM_SECONDS = float(SPEC['per_arm_seconds'])
PREFIX = SPEC['prefix']
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
               '--config', str(config), '--out', f'out/{PREFIX}-{name}', '--wall-seconds', f'{wall:.0f}'])
    return True


run('precision', [PY, 'code/smoke_second.py', '--output', 'out/precision.json'])
run('training-smoke', [PY, 'code/training_smoke_second.py', '--family', SPEC['family'],
                       '--out', 'out/training-smoke'])
run('cohort', [PY, 'code/prepare_diagnosis_cohort.py',
               '--reference-index', 'data/refinement/index.json',
               '--train-index', TRAIN, '--out', 'data/diagnosis-cohort'])

trained = []
for arm in SPEC['arms']:
    if stopping:
        break
    config = arm['config']
    if 'override' in arm:
        merged = json.loads((ROOT / config).read_text())
        merged.update(arm['override'])
        config = ROOT / f"out/{arm['name']}-config.json"
        config.write_text(json.dumps(dict(merged, derived_from=arm['config']), indent=2) + '\n')
    if train(arm['name'], config, arm.get('seconds', PER_ARM_SECONDS)):
        trained.append(arm['name'])

scores = {}
for name in trained:
    result = ROOT / f'out/{PREFIX}-{name}/result.json'
    if result.exists():
        scores[name] = json.loads(result.read_text())['validation']['mean_case_max']
refine_lr = SPEC.get('refine_learning_rate')
if refine_lr and scores and not stopping and remaining() - RESERVE >= 1200:
    best = min(scores, key=scores.get)
    source = next(a for a in SPEC['arms'] if a['name'] == best)['config']
    config = json.loads((ROOT / source).read_text())
    config['learning_rate'] = refine_lr
    (ROOT / 'out/refine-config.json').write_text(json.dumps(
        dict(config, refines_capacity=best,
             selected_by='validation mean case-maximum fixed-initial error'), indent=2) + '\n')
    if train('refine', 'out/refine-config.json', PER_ARM_SECONDS):
        trained.append('refine')
(ROOT / 'out/capacity-selection.json').write_text(json.dumps(dict(
    spec=SPEC, capacity_scores=scores, refinement_learning_rate=refine_lr,
    per_arm_wall_seconds=PER_ARM_SECONDS, trained=trained, stopped_by_signal=stopping,
    design='equal wall budget per arm with early stopping; epoch counts differ by design'), indent=2) + '\n')

run('cohort-eval', [PY, 'code/evaluate_cohort.py', '--index', 'data/diagnosis-cohort/index.json',
                    '--runs', 'out', '--out', 'out/diagnosis-cohort', '--pattern', f'{PREFIX}-*'], allow_failure=True)
run('timing', [PY, 'code/timing.py', '--train-index', TRAIN, '--validation-index', VALIDATION,
               '--runs', 'out', '--out', 'out/timing', '--pattern', f'{PREFIX}-*',
               '--repetitions', '30', '--burn-in', '20'], allow_failure=True)
print('WORKER FINISHED', flush=True)
