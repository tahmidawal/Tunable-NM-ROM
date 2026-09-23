"""Generate the extended Burgers training split, in one allocation, and nothing else.

The cache this writes lives in the lane NAMESPACE, not in the job directory, because the
training jobs mount it after this job's directory is collected and deleted (design audit
finding 33). It is made read-only as soon as it is complete, and its `DATA.sha256` covers both
the per-case files and the packed pool, so every later mount verifies the whole thing.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path.cwd()
NAMESPACE = '/cluster/tufts/paralab/tawal01/opstune_don_20260922'
assert str(ROOT).startswith(NAMESPACE + '/')
assert os.environ.get('SLURM_JOB_ID')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
SPEC = json.loads(Path(sys.argv[1]).read_text())
CACHE = SPEC['output_cache']
assert CACHE.startswith(NAMESPACE + '/') and not Path(CACHE).exists(), CACHE
PINNED_TRAIN = SPEC['pinned_train_index']
PINNED_VALIDATION = SPEC['pinned_validation_index']
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


def run(name, command, allow_failure=False):
    global child
    start = time.time()
    with (ROOT / 'logs' / f'{name}.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    records.append(dict(task=name, command=command, exit_code=code, started_unix=start,
                        ended_unix=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                        elapsed_seconds=time.monotonic() - started))
    (ROOT / 'out/worker.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records[-1]), flush=True)
    if code and not allow_failure:
        sys.exit(code)
    return code


run('self-check', [PY, 'code/gen_more.py', 'self-check', '--output', 'out/gen-self-check.json',
                   '--pinned-index', PINNED_TRAIN])
run('generate', [PY, 'code/gen_more.py', 'generate', '--out', CACHE,
                 '--count', str(SPEC['count']), '--intervals', '256',
                 '--candidates', *SPEC['candidates'],
                 '--milestones', *[str(m) for m in SPEC['milestones']],
                 '--pinned-index', PINNED_TRAIN,
                 '--wall-budget-seconds', str(SPEC['generation_seconds']),
                 '--reserve-seconds', str(SPEC['generation_reserve_seconds'])])
run('pool', [PY, 'code/build_pool.py', '--index', f'{CACHE}/train/index.json',
             '--validation-index', PINNED_VALIDATION, '--out', f'{CACHE}/pool'])

# One manifest over the whole cache: the per-case files (written by gen_more) plus the pool.
pool_lines = subprocess.run(['sha256sum'] + sorted(str(p.relative_to(CACHE))
                                                   for p in Path(f'{CACHE}/pool').iterdir()),
                            cwd=CACHE, capture_output=True, text=True, check=True).stdout
with open(f'{CACHE}/DATA.sha256', 'a') as manifest:
    manifest.write(pool_lines)
verify = subprocess.run(['sha256sum', '-c', 'DATA.sha256', '--quiet'], cwd=CACHE,
                        capture_output=True, text=True)
lines = len(Path(f'{CACHE}/DATA.sha256').read_text().splitlines())
assert verify.returncode == 0, verify.stderr[-2000:]
print(f'cache_verified={lines}', flush=True)
subprocess.run(['chmod', '-R', 'a-w', CACHE], check=True)

index = json.loads(Path(f'{CACHE}/train/index.json').read_text())
pool = json.loads(Path(f'{CACHE}/pool/cases.json').read_text())
(ROOT / 'out/gen-summary.json').write_text(json.dumps(dict(
    cache=CACHE, manifest_lines=lines, read_only=True, cases=index['count'],
    reference_setting=index['reference_setting'], profile=index['profile'],
    pinned_comparison={k: v for k, v in index['pinned_comparison'].items() if k != 'cases_detail'},
    generation_seconds=index.get('generation_seconds'), stop_reason=index.get('stop_reason'),
    pool=dict(count=pool['count'], index_sha256=pool['index_sha256'],
              validation_index_sha256=pool['validation_index_sha256'],
              array_sha256=pool['array_sha256']),
    stopped_by_signal=stopping), indent=2) + '\n')
print('WORKER FINISHED', flush=True)
