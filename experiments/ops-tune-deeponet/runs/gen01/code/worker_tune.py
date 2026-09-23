"""Spec-driven arm runner for this lane, in one allocation.

`experiments/ops-deeponet-b2d/worker_second.py` with four changes, all from DESIGN.md:

  * every arm is `base_config` plus its own declared `override`, so what distinguishes two
    arms is one JSON object that was written before the job;
  * an arm names the data mount and prefix length it trains on (`pool`, `pool_limit`), which
    is how the ladder of section 2.3 and the pinned-protocol control of section 5.2 are run
    inside one job;
  * the parent's learning-rate `refine` step is replaced by section 5.3's composition rule
    (`compose.py`), and by section 6's schedule decision, which applies whichever of plateau
    or cosine won to the capacity and learning-rate arms that follow it;
  * no timing block: no speed number is admissible from this lane (section 7).
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path.cwd()
assert str(ROOT).startswith('/cluster/tufts/paralab/tawal01/opstune_don_20260922/')
assert os.environ.get('SLURM_JOB_ID')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
SPEC = json.loads(Path(sys.argv[1]).read_text())
GLOBAL_SECONDS = float(SPEC['global_seconds'])
RESERVE = float(SPEC['reserve_seconds'])
PREFIX = SPEC['prefix']
PDE = SPEC.get('pde', 'burgers')
BASE = json.loads((ROOT / SPEC['base_config']).read_text())
MOUNTS = SPEC['mounts']
VALIDATION = SPEC['validation_index']
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


def train(arm, extra_override=None):
    """One arm: base + its declared override (+ the schedule decision), on its declared data."""
    name, budget = arm['name'], float(arm['seconds'])
    wall = min(budget, remaining() - RESERVE)
    if wall < 300:
        records.append(dict(task=name, skipped='insufficient remaining budget',
                            remaining_global_seconds=remaining()))
        save()
        return False
    merged = dict(BASE)
    merged.update(extra_override or {})
    merged.update(arm.get('override', {}))
    config = ROOT / f'out/{name}-config.json'
    config.write_text(json.dumps(dict(merged, arm=name, knob=arm.get('knob'),
                                      derived_from=SPEC['base_config'],
                                      applied_schedule_decision=extra_override or {}), indent=2) + '\n')
    mount = MOUNTS[arm['mount']]
    command = [PY, 'code/train.py', '--train-index', mount['train_index'],
               '--validation-index', VALIDATION, '--config', str(config),
               '--out', f'out/{PREFIX}-{name}', '--wall-seconds', f'{wall:.0f}']
    if mount.get('pool'):
        command += ['--pool', mount['pool']]
        if arm.get('pool_limit'):
            command += ['--pool-limit', str(arm['pool_limit'])]
    run(name, command)
    return True


def score(name):
    result = ROOT / f'out/{PREFIX}-{name}/result.json'
    return json.loads(result.read_text())['validation']['mean_case_max'] if result.exists() else None


run('precision', [PY, 'code/smoke_second.py', '--output', 'out/precision.json'])
run('training-smoke', [PY, 'code/training_smoke_second.py', '--family', SPEC['family'], '--pde', PDE,
                       '--out', 'out/training-smoke'])
run('cohort', [PY, 'code/prepare_diagnosis_cohort.py',
               '--reference-index', SPEC['reference_index'],
               '--train-index', MOUNTS[SPEC['cohort_train_mount']]['train_index'],
               '--out', 'data/diagnosis-cohort'])

trained, decision, decided = [], {}, False
for arm in SPEC['arms']:
    if stopping:
        break
    if train(arm, decision):
        trained.append(arm['name'])
    # Section 6: once the schedule arm and its reference both exist, whichever won applies to
    # every later arm, so the capacity and learning-rate knobs are not tested against a
    # schedule already known to be worse.
    rule = SPEC.get('schedule_decision')
    if rule and not decided and rule['arm'] in trained and rule['reference'] in trained:
        challenger, reference = score(rule['arm']), score(rule['reference'])
        won = challenger is not None and reference is not None and challenger < reference
        decision, decided = (dict(rule['override']) if won else {}), True
        (ROOT / 'out/schedule-decision.json').write_text(json.dumps(dict(
            rule=rule, challenger_score=challenger, reference_score=reference,
            applied=decision, note='DESIGN section 6'), indent=2) + '\n')
        print('SCHEDULE DECISION ' + json.dumps(dict(applied=decision, challenger=challenger,
                                                     reference=reference)), flush=True)

if SPEC.get('compose') and not stopping:
    run('compose', [PY, 'code/compose.py', '--spec', sys.argv[1], '--runs', 'out',
                    '--out', 'out/composition.json'], allow_failure=True)
    path = ROOT / 'out/composition.json'
    if path.exists():
        composed = json.loads(path.read_text())
        arm = dict(SPEC['compose']['arm'])
        arm['override'] = dict(decision, **composed['override'])
        if train(arm):
            trained.append(arm['name'])

(ROOT / 'out/selection.json').write_text(json.dumps(dict(
    spec=SPEC, scores={name: score(name) for name in trained}, trained=trained,
    schedule_decision=decision, stopped_by_signal=stopping,
    selection_metric='mean over the 32 validation cases of the per-case maximum over the six '
                     'output times of the fixed-initial relative error (DESIGN section 5.1)'), indent=2) + '\n')

run('cohort-eval', [PY, 'code/evaluate_cohort.py', '--index', 'data/diagnosis-cohort/index.json',
                    '--runs', 'out', '--out', 'out/diagnosis-cohort', '--pattern', f'{PREFIX}-*'],
    allow_failure=True)
print('WORKER FINISHED', flush=True)
