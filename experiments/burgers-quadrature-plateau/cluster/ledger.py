"""GPU-hour ledger and submission guard (DESIGN A1.8): cluster/ledger.json.
  python cluster/ledger.py check <minutes>   exit 1 unless used + outstanding limits + <minutes> <= 240 GPU-min
  python cluster/ledger.py add <job> <jobid> <limit_min> <gpu> <config> <commit>
  python cluster/ledger.py refresh           sacct -> state / elapsed of every job
  python cluster/ledger.py show
"""
import json
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent / 'ledger.json'
CAP_MIN = 240.0
DONE = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED')


def load():
    return json.loads(P.read_text()) if P.exists() else dict(cap_gpu_hours=CAP_MIN / 60, jobs=[])


def refresh(d):
    ids = [j['jobid'] for j in d['jobs']]
    if not ids:
        return d
    out = subprocess.run(['ssh', 'tufts-login', f"sacct -X -n -P -j {','.join(ids)} -o JobID,State,ElapsedRaw,NodeList"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        jid, st, el, node = line.split('|')
        for j in d['jobs']:
            if j['jobid'] == jid:
                j.update(state=st.split()[0], elapsed_s=int(el or 0), node=node)
    return d


def minutes(d):
    used = sum(j.get('elapsed_s', 0) / 60 for j in d['jobs'] if j.get('state', '').startswith(DONE))
    outstanding = sum(j['limit_min'] for j in d['jobs'] if not j.get('state', '').startswith(DONE))
    return used, outstanding


def main():
    cmd = sys.argv[1]
    d = load()
    if cmd in ('check', 'refresh', 'show'):
        d = refresh(d)
        P.write_text(json.dumps(d, indent=1) + '\n')
    used, outst = minutes(d)
    if cmd == 'check':
        new = float(sys.argv[2])
        tot = used + outst + new
        print(f'ledger: used {used:.1f} + outstanding {outst:.1f} + new {new:.1f} = {tot:.1f} of {CAP_MIN:.0f} GPU-min')
        sys.exit(0 if tot <= CAP_MIN else 1)
    if cmd == 'add':
        job, jid, lim, gpu, cfg, commit = sys.argv[2:8]
        d['jobs'].append(dict(job=job, jobid=jid, limit_min=float(lim), gpu=gpu, config=cfg, commit=commit, state='PENDING'))
        P.write_text(json.dumps(d, indent=1) + '\n')
    if cmd in ('show', 'refresh', 'add'):
        for j in d['jobs']:
            print(j['job'], j['jobid'], j.get('state'), round(j.get('elapsed_s', 0) / 60, 1), j['limit_min'], j['gpu'])
        print(f'used {used:.1f} GPU-min ({used / 60:.2f} GPU-h); outstanding limits {outst:.1f}')


if __name__ == '__main__':
    main()
