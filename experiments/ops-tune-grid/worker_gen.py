"""Generate the extended Burgers training bank and leave a verified shared cache behind.

One job, one directory. The sbatch preamble has already copied the pinned Burgers cache into
`data/` and verified every file against the cache's own `DATA.sha256`, so `data/train`,
`data/validation` and `data/refinement` are byte-verified before anything here runs.

This worker then:

1. runs `gen_bank.py`, which reproduces a published case at the pinned setting as its gate
   and generates the extended bank into `data/trainbig` (DESIGN 3.2, 5.1);
2. writes a single `data/DATA.sha256` covering all four directories; and
3. moves the whole verified `data/` tree to `<NAMESPACE>/cache`, so later jobs in this lane
   stage from one cache with one manifest and this job's own directory can be deleted.

Nothing here trains, and no timing is recorded.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path.cwd()
NAMESPACE = '/cluster/tufts/paralab/tawal01/opstune_grid_20260922'
assert str(ROOT).startswith(NAMESPACE + '/'), ROOT
assert os.environ.get('SLURM_JOB_ID')
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
SPEC = json.loads(Path(sys.argv[1]).read_text())
CACHE = Path(NAMESPACE) / 'cache'
started = time.monotonic()
(ROOT / 'out').mkdir()
records = []


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(name, command):
    start = time.time()
    with (ROOT / 'logs' / f'{name}.log').open('w') as log:
        code = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT).wait()
    records.append(dict(task=name, command=command, exit_code=code, started_unix=start,
                        ended_unix=time.time(), job_id=os.environ['SLURM_JOB_ID']))
    (ROOT / 'out/worker.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records[-1]), flush=True)
    if code:
        sys.exit(code)


run('generate', [PY, 'code/gen_bank.py', '--out', 'data/trainbig',
                 '--pinned-cache', 'data/train', '--count', str(SPEC['count']),
                 '--wall-budget-seconds', str(SPEC['generation_seconds'])]
    + ['--prefixes'] + [str(p) for p in SPEC['prefixes']])

report = json.loads((ROOT / 'data/trainbig/generation-report.json').read_text())
assert report['complete'] and report['reproduction']['arrays_identical'], 'generation gate failed'

# One manifest over the whole cache, paths relative to the cache root, so the staging
# preamble of every later job in this lane verifies the pinned data and the new bank
# with the same command that verified the pinned cache alone.
data = ROOT / 'data'
lines = [f'{sha(p)}  {p.relative_to(data)}' for p in sorted(data.rglob('*')) if p.is_file()]
(data / 'DATA.sha256').write_text('\n'.join(lines) + '\n')
index_hashes = {p.name: sha(p) for p in sorted((data / 'trainbig').glob('index-*.json'))}
(ROOT / 'out/cache.json').write_text(json.dumps(dict(
    cache=str(CACHE), files=len(lines), bank_index_sha256=index_hashes,
    generated_count=report['generated_count'], stop_reason=report['stop_reason'],
    fidelity_summary=report['fidelity_summary'],
    seconds_per_case_median=report['seconds_per_case_median'],
    reproduction=report['reproduction'], worker_seconds=time.monotonic() - started), indent=2) + '\n')
shutil.copy(ROOT / 'data/trainbig/generation-report.json', ROOT / 'out/generation-report.json')

assert not CACHE.exists(), f'{CACHE} already exists; refusing to overwrite another job\'s cache'
data.rename(CACHE)
print(json.dumps(dict(cache=str(CACHE), files=len(lines), indices=list(index_hashes))), flush=True)
print('WORKER FINISHED', flush=True)
