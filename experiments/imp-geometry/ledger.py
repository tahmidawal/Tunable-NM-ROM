"""Refresh runs/GPU-LEDGER.json from sacct (elapsed GPU-hours per job, cumulative total).
  python ledger.py [--add <jobid> <name> <gpu> <purpose>]"""
import json, subprocess, sys
from pathlib import Path
P = Path(__file__).resolve().parent / 'runs' / 'GPU-LEDGER.json'
L = json.loads(P.read_text())
if len(sys.argv) > 1 and sys.argv[1] == '--add':
    L['jobs'].append(dict(job=sys.argv[2], name=sys.argv[3], gpu=sys.argv[4], purpose=' '.join(sys.argv[5:]),
                          gpu_hours=None, state='SUBMITTED'))
ids = ','.join(j['job'] for j in L['jobs'])
out = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                      f'sacct -j {ids} -X -n -P -o JobID,State,ElapsedRaw,AllocTRES'], capture_output=True, text=True).stdout
for line in out.splitlines():
    jid, state, el, tres = line.split('|')
    for j in L['jobs']:
        if j['job'] == jid:
            ngpu = 1
            for t in tres.split(','):
                if t.startswith('gres/gpu=') or t.startswith('gres/gpu:'):
                    try:
                        ngpu = int(t.split('=')[-1])
                    except ValueError:
                        pass
            j['state'] = state
            j['gpu_hours'] = round(int(el) * ngpu / 3600.0, 4)
L['total_gpu_hours'] = round(sum(j['gpu_hours'] or 0 for j in L['jobs']), 4)
L['smoke_gpu_hours'] = round(sum(j['gpu_hours'] or 0 for j in L['jobs'] if 'smoke' in j['name']), 4)
P.write_text(json.dumps(L, indent=1))
print(json.dumps({k: v for k, v in L.items() if k != 'jobs'}), [(j['job'], j['state'], j['gpu_hours']) for j in L['jobs']])
