"""GPU-hour ledger (COMMON-BRIEF §7): python ledger.py add <jobid> <name> | show. Elapsed from sacct (GPU count x
elapsed); runs/GPU-LEDGER.json keeps every job and the cumulative total."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

F = Path(__file__).resolve().parent / 'runs' / 'GPU-LEDGER.json'


def load():
    if F.exists():
        return json.loads(F.read_text())
    return dict(cap=87, jobs=[], total_gpu_hours=0.0)


def sacct(jobid):
    out = subprocess.check_output(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                   f'sacct -j {jobid} -X -n -P -o JobID,State,ExitCode,ElapsedRaw,AllocTRES,NodeList'],
                                  text=True).strip().splitlines()[0].split('|')
    jid, state, exitc, el, tres, node = out
    steps = subprocess.check_output(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                     f'sacct -j {jobid} -n -P -o JobID,State,ExitCode,MaxRSS,ElapsedRaw'],
                                    text=True).strip().splitlines()
    ngpu = 1
    for t in tres.split(','):
        if t.startswith('gres/gpu='):
            ngpu = int(t.split('=')[1])
    gtype = [t.split(':')[1].split('=')[0] for t in tres.split(',') if t.startswith('gres/gpu:')]
    return dict(id=jid, state=state.split(' ')[0], exit_code=exitc, elapsed_s=int(el), gpus=ngpu,
                gpu_type=gtype[0] if gtype else None, node=node, gpu_hours=ngpu * int(el) / 3600.0, steps=steps)


def add(jobid, name, gpu_file=None):
    """Serialized (flock) read-modify-write with an atomic replace (job audit 2). gpu_file: the job's
    allocated_gpu.txt (uuid, name) -- the GPU type when sacct's AllocTRES carries none."""
    import fcntl
    import os
    rec = sacct(jobid)
    rec['name'] = name
    if gpu_file and Path(gpu_file).exists():
        txt = Path(gpu_file).read_text().strip()
        rec['gpu_allocated'] = txt
        if rec.get('gpu_type') is None and 'H200' in txt:
            rec['gpu_type'] = 'h200'
    F.parent.mkdir(parents=True, exist_ok=True)
    with open(F.with_suffix('.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        L = load()
        old = [j for j in L['jobs'] if j['id'] == rec['id']]                 # merge inside the lock (code audit 8)
        if 'gpu_allocated' not in rec and old and old[0].get('gpu_allocated'):
            rec['gpu_allocated'], rec['gpu_type'] = old[0]['gpu_allocated'], old[0].get('gpu_type') or rec['gpu_type']
        L['jobs'] = [j for j in L['jobs'] if j['id'] != rec['id']] + [rec]
        L['total_gpu_hours'] = round(sum(j['gpu_hours'] for j in L['jobs']), 4)
        tmp = F.with_suffix('.tmp')
        tmp.write_text(json.dumps(L, indent=1))
        os.replace(tmp, F)
    print(json.dumps({k: v for k, v in rec.items() if k != 'steps'}), 'total', L['total_gpu_hours'])


if __name__ == '__main__':
    if sys.argv[1] == 'add':
        add(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        print(json.dumps(load(), indent=1))
