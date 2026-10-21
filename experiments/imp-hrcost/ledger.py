"""GPU-hour ledger (COMMON-BRIEF 7): python ledger.py add <attempt> <jobid> | reserve <attempt> <hours> | show
add: reads sacct (ElapsedRaw, AllocTRES gres) for the job and records measured GPU-hours; reservations are kept
until the job is added."""
import json, subprocess, sys
from pathlib import Path
F = Path(__file__).resolve().parent / 'runs' / 'GPU-LEDGER.json'


def load():
    return json.loads(F.read_text()) if F.exists() else dict(cap=3.0, jobs=[], reservations={}, total_gpu_hours=0.)


def save(L):
    L['total_gpu_hours'] = round(sum(j['gpu_hours'] for j in L['jobs']), 4)
    F.parent.mkdir(exist_ok=True)
    F.write_text(json.dumps(L, indent=1))


cmd = sys.argv[1]
L = load()
if cmd == 'reserve':
    L['reservations'][sys.argv[2]] = float(sys.argv[3])
    assert L['total_gpu_hours'] + sum(L['reservations'].values()) <= L['cap'] + 1e-9, 'reservation exceeds cap'
elif cmd == 'release':
    L['reservations'].pop(sys.argv[2], None)
elif cmd == 'add':
    a, jid = sys.argv[2], sys.argv[3]
    out = subprocess.check_output(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                   f'sacct -j {jid} -X -n -P -o JobID,State,ElapsedRaw,AllocTRES,NodeList'], text=True)
    row = out.strip().splitlines()[0].split('|')
    n = 1
    for t in row[3].split(','):
        if t.startswith('gres/gpu=') or (t.startswith('gres/gpu') and '=' in t and ':' not in t.split('=')[0][9:]):
            n = int(t.split('=')[1])
    gpu = next((t.split(':')[1].split('=')[0] for t in row[3].split(',') if t.startswith('gres/gpu:')), 'unknown')
    L['jobs'].append(dict(attempt=a, jobid=jid, state=row[1], elapsed_s=int(row[2]), gpus=n, gpu_type=gpu, node=row[4],
                          gpu_hours=round(int(row[2]) * n / 3600, 4)))
    L['reservations'].pop(a, None)
save(L)
print(json.dumps(L, indent=1))
