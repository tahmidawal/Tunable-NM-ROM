"""jcp-regimes GPU-hour ledger (COMMON-BRIEF §6, DESIGN A2.6). runs/GPU-LEDGER.json under an exclusive file lock.
Lane-authored (pattern of jcp-qualify's ledger.py, without its lane-specific inventory rules).

    python ledger.py reserve <attempt> <hours>   refuse unless spent + open reservations + hours <= CAP; record PENDING_SUBMIT
    python ledger.py bind <attempt> <jobid>      the reservation is job <jobid> (SUBMITTED)
    python ledger.py release <attempt>           sbatch never returned a job id: drop the reservation
    python ledger.py ambiguous <attempt>         sbatch outcome unknown: keep the reservation (settle by squeue/sacct by hand)
    python ledger.py reconcile                   sacct -X: terminal jobs -> elapsed GPU-hours (one GPU per job)
    python ledger.py show
A job's reservation is its sbatch time limit; a terminal job counts its measured elapsed time instead.
"""
import fcntl
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
LED = HERE / 'runs' / 'GPU-LEDGER.json'
LOCK = HERE / 'runs' / 'GPU-LEDGER.lock'
CAP = 60.
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL',
            'DEADLINE')


@contextmanager
def locked():
    LED.parent.mkdir(exist_ok=True)
    with open(LOCK, 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            d = json.loads(LED.read_text()) if LED.exists() else dict(lane='jcp-regimes', cap_gpu_hours=CAP, jobs=[])
            yield d
            totals(d)
            tmp = LED.with_suffix('.tmp')
            tmp.write_text(json.dumps(d, indent=1) + '\n')
            tmp.replace(LED)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def totals(d):
    d['spent_gpu_hours'] = round(sum(j.get('elapsed_hours') or 0. for j in d['jobs'] if j['state'] in TERMINAL), 4)
    d['reserved_gpu_hours'] = round(sum(j['time_limit_hours'] for j in d['jobs'] if j['state'] not in TERMINAL), 4)
    d['committed_gpu_hours'] = round(d['spent_gpu_hours'] + d['reserved_gpu_hours'], 4)
    return d


def hms(s):
    s = s.strip()
    days = 0
    if '-' in s:
        dd, s = s.split('-')
        days = int(dd)
    p = [int(x) for x in s.split(':')]
    while len(p) < 3:
        p.insert(0, 0)
    return days * 24 + p[0] + p[1] / 60 + p[2] / 3600


def main():
    cmd = sys.argv[1]
    rc = 0
    with locked() as d:
        totals(d)
        if cmd == 'reserve':
            att, h = sys.argv[2], float(sys.argv[3])
            if any(j['attempt'] == att and j['state'] not in TERMINAL for j in d['jobs']):
                print(f'{att} already has a live reservation'); rc = 1
            elif d['spent_gpu_hours'] + d['reserved_gpu_hours'] + h > CAP + 1e-9:
                print(f"refused: {d['spent_gpu_hours']} + {d['reserved_gpu_hours']} + {h} > {CAP}"); rc = 1
            else:
                d['jobs'].append(dict(attempt=att, state='PENDING_SUBMIT', time_limit_hours=h, job_id=None))
        elif cmd == 'bind':
            j = [x for x in d['jobs'] if x['attempt'] == sys.argv[2] and x['state'] == 'PENDING_SUBMIT'][-1]
            j.update(job_id=sys.argv[3], state='SUBMITTED')
        elif cmd == 'release':
            d['jobs'] = [x for x in d['jobs'] if not (x['attempt'] == sys.argv[2] and x['state'] == 'PENDING_SUBMIT')]
        elif cmd == 'ambiguous':
            for x in d['jobs']:
                if x['attempt'] == sys.argv[2] and x['state'] == 'PENDING_SUBMIT':
                    x['state'] = 'AMBIGUOUS'
        elif cmd == 'reconcile':
            for j in d['jobs']:
                if j['state'] in TERMINAL or not j.get('job_id'):
                    continue
                out = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                      f"sacct -j {j['job_id']} -X -n -P -o State,Elapsed,NodeList,AllocTRES"],
                                     capture_output=True, text=True).stdout.strip().splitlines()
                if not out:
                    continue
                st, el, node, tres = out[0].split('|')
                st = st.split()[0]
                j.update(sacct_state=st, node=node, alloc=tres)
                if st in TERMINAL:
                    j.update(state=st, elapsed_hours=round(hms(el), 4))
        elif cmd != 'show':
            print('unknown command'); rc = 2
        totals(d)
        print(json.dumps({k: v for k, v in d.items() if k != 'jobs'}), f"jobs={len(d['jobs'])}")
    sys.exit(rc)


if __name__ == '__main__':
    main()
