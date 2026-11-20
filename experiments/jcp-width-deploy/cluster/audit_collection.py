"""Verify a pulled, terminal lane result and charge its measured allocation."""
from pathlib import Path
import hashlib
import json
import re
import sys

lane = Path(__file__).resolve().parents[1]
attempt = sys.argv[1]
assert re.fullmatch('[a-z][a-z0-9]+', attempt)
folder = lane/'runs'/attempt
files = []
for line in (folder/'RESULTS.sha256').read_text().splitlines():
    expected, rel = line.split('  ', 1)
    p = folder/rel
    assert p.resolve().is_relative_to(folder.resolve())
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    files.append(dict(path=rel, sha256=expected, verified=h.hexdigest() == expected))
assert all(f['verified'] for f in files)
row = next(line.split('|') for line in (folder/'sacct.txt').read_text().splitlines() if '.' not in line.split('|')[0])
job_id, job_name, state, exit_code, elapsed, allocation = row[:6]
assert job_name == 'ctol_jwv_'+attempt
log = (folder/'job.log').read_text()
env = json.loads((folder/'output/environment.json').read_text())
assert str(env['job_id']) == job_id
gpu_valid = bool('jax_backend=gpu' in log and env['backend'] == 'gpu' and env['x64'] and env['matmul_precision'] == 'highest')
warnings = re.findall(r'(?im)^.*(?:captured.*constant|large.*constant|out.of.memory|no space left|traceback|cuInit|killed|oom-kill).*$' ,log)
collection = dict(job_id=int(job_id), scheduler_state=state, exit_code=exit_code,
    allocated_gpu_seconds=int(elapsed), checksum_verified=True, file_count=len(files), files=files,
    remote_directory_deleted=False, gpu_log_valid=gpu_valid,
    no_environmental_warnings=not warnings, environmental_warning_lines=warnings,
    terminal_marker_present='completed=True' in log, scientific_status='terminal raw collection; numerical audit pending')
assert gpu_valid and not warnings and state == 'COMPLETED' and exit_code == '0:0' and collection['terminal_marker_present']
(folder/'collection.json').write_text(json.dumps(collection, indent=2)+'\n')
ledger_path = lane/'runs/LEDGER.json'
ledger = json.loads(ledger_path.read_text())
reservation = json.loads((folder/'reservation.json').read_text())
record = dict(job_id=int(job_id), attempt=attempt, job_name=job_name, state=state,
    ceiling_allocated_gpu_seconds=reservation['ceiling_allocated_gpu_seconds'], charged_allocated_gpu_seconds=int(elapsed),
    source_commit=env['commit'], remote='/cluster/tufts/paralab/tawal01/jcp-weak-visibility-20261118-488b0837/'+attempt,
    reviewer='head_mechanism_run', review_decision='W1 registered trajectory matrix cleared; each wider launch conditional on baseline audit')
previous = [r for r in ledger['jobs'] if r['job_id'] == int(job_id)]
assert len(previous) <= 1
if previous:
    previous[0].update(record)
else:
    ledger['jobs'].append(record)
ledger['total_charged_allocated_gpu_seconds'] = sum(r.get('charged_allocated_gpu_seconds', 0) for r in ledger['jobs'])
assert ledger['total_charged_allocated_gpu_seconds'] <= ledger['lane_ceiling_allocated_gpu_seconds']
ledger_path.write_text(json.dumps(ledger, indent=2)+'\n')
print(json.dumps({k:collection[k] for k in ['job_id','scheduler_state','allocated_gpu_seconds','file_count','gpu_log_valid','no_environmental_warnings']}))
