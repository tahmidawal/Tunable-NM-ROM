"""GPU-hour ledger (COMMON-BRIEF 7, A1.10, code audit 1): runs/GPU-LEDGER.json. Queued/running jobs reserve their
time limit; finished jobs count their sacct elapsed hours (one GPU each). Lane jobs (name ifs_*) found in sacct but
missing from the receipts are added as 'unreconciled' and counted conservatively.

    python cluster/ledger.py update
    python cluster/ledger.py check <hours>     # exit 1 unless spent + reserved + hours <= cap (hours finite, > 0)
"""
import json
import math
import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
LED = LANE / 'runs' / 'GPU-LEDGER.json'
CAP = 4.0
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def ssh(cmd):
    p = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login', cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'ssh failed ({p.returncode}): {p.stderr[-300:]}')
    return p.stdout


def tl_hours(s):
    """Slurm Timelimit [d-]hh:mm:ss -> hours."""
    d = 0
    if '-' in s:
        d, s = s.split('-')
    p = [int(x) for x in s.split(':')]
    while len(p) < 3:
        p = [0] + p
    return int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600


def update():
    L = json.loads(LED.read_text()) if LED.exists() else dict(cap_gpu_hours=CAP, jobs=[])
    known = {x['job_id'] for x in L['jobs']}
    for sub in sorted((LANE / 'runs').glob('*/SUBMIT.json')):
        s = json.loads(sub.read_text())
        if s['job_id'] in known:
            continue
        res = json.loads((sub.parent / 'stage' / 'RESERVATION.json').read_text())
        L['jobs'].append(dict(job=sub.parent.name, job_id=s['job_id'], gpu=res['gpu'], time_limit_hours=res['time_limit_hours']))
        known.add(s['job_id'])
    out = ssh('sacct -u tawal01 -X -n -P -S 2026-10-21 -o JobID,JobName,State,ElapsedRaw,Timelimit,NodeList,AllocTRES')
    rows = {}
    for line in out.strip().splitlines():
        f = line.split('|')
        if len(f) >= 7:
            rows[f[0]] = f
    for jid, f in rows.items():
        if f[1].startswith('ifs_') and jid not in known:
            L['jobs'].append(dict(job=f[1][4:], job_id=jid, gpu='?', time_limit_hours=tl_hours(f[4]), unreconciled=True))
            known.add(jid)
    for e in L['jobs']:
        f = rows.get(e['job_id'])
        if f is None:
            e['state'] = e.get('state', 'UNKNOWN')
            continue
        e.update(state=f[2].split()[0], elapsed_hours=round(int(f[3] or 0) / 3600, 4), node=f[5], tres=f[6], name=f[1])
    L['spent_gpu_hours'] = round(sum(x.get('elapsed_hours', 0.) for x in L['jobs'] if x.get('state') in TERMINAL), 4)
    L['reserved_gpu_hours'] = round(sum(x['time_limit_hours'] for x in L['jobs'] if x.get('state') not in TERMINAL), 4)
    LED.write_text(json.dumps(L, indent=1) + '\n')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('update', 'check') or (sys.argv[1] == 'check' and len(sys.argv) != 3):
        sys.exit('usage: ledger.py update | check <hours>')
    L = update()
    print(json.dumps(dict(spent=L['spent_gpu_hours'], reserved=L['reserved_gpu_hours'], jobs=len(L['jobs']))))
    if sys.argv[1] == 'check':
        need = float(sys.argv[2])
        if not (math.isfinite(need) and need > 0):
            sys.exit('hours must be finite and > 0')
        ok = L['spent_gpu_hours'] + L['reserved_gpu_hours'] + need <= CAP + 1e-9
        print('admit' if ok else 'REFUSE', need)
        sys.exit(0 if ok else 1)
