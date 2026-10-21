"""GPU-hour ledger (COMMON-BRIEF 7): runs/GPU-LEDGER.json, updated under an exclusive file lock.
    ledger.py reserve <attempt> <limit_hours> <projected_hours>  -> exit 0 iff spent(terminal) + active limits +
                                                                    limit <= cap and projected <= limit; writes RESERVED
    ledger.py bind <attempt> <jobid>        ledger.py release <attempt>
    ledger.py settle                        (sacct: terminal jobs get their measured elapsed GPU-hours)
spent = sum of measured hours of terminal jobs; every non-terminal job (RESERVED, PENDING, RUNNING, ...) holds its
full time limit as a reservation (no double counting).
"""
import fcntl
import json
import math
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'runs' / 'GPU-LEDGER.json'
CAP = 8.0
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL',
            'DEADLINE', 'RELEASED')


def load():
    return json.loads(P.read_text()) if P.exists() else dict(cap_gpu_hours=CAP, jobs=[], spent_gpu_hours=0.0)


def totals(d):
    spent = sum(j.get('elapsed_hours', 0.0) for j in d['jobs'] if j.get('state') in TERMINAL)
    active = sum(j['limit_hours'] for j in d['jobs'] if j.get('state') not in TERMINAL)
    return spent, active


def save(d):
    d['spent_gpu_hours'], d['reserved_gpu_hours'] = (round(x, 4) for x in totals(d))
    P.write_text(json.dumps(d, indent=1) + '\n')


def main():
    P.parent.mkdir(parents=True, exist_ok=True)
    with open(P.with_suffix('.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = load()
        cmd = sys.argv[1]
        if cmd == 'reserve':
            a, lim, proj = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
            assert math.isfinite(lim) and math.isfinite(proj) and 0 < proj <= lim, ('bad hours', lim, proj)
            assert not any(j['attempt'] == a for j in d['jobs']), f'{a} already in the ledger'
            spent, active = totals(d)
            tot = spent + active + lim
            print(f'spent {spent:.3f} + active limits {active:.3f} + this limit {lim:.3f} = {tot:.3f} / cap {CAP}')
            if tot > CAP:
                sys.exit(6)
            d['jobs'].append(dict(attempt=a, job_id=None, limit_hours=lim, projected_hours=proj, state='RESERVED'))
            save(d)
        elif cmd == 'bind':
            a, jid = sys.argv[2], sys.argv[3]
            j = next(j for j in d['jobs'] if j['attempt'] == a and j['state'] == 'RESERVED')
            j.update(job_id=jid, state='PENDING')
            save(d)
        elif cmd == 'release':
            d['jobs'] = [j for j in d['jobs'] if not (j['attempt'] == sys.argv[2] and j['state'] == 'RESERVED')]
            save(d)
        elif cmd == 'settle':
            for j in d['jobs']:
                if j.get('state') in TERMINAL or not j.get('job_id'):
                    continue
                o = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                    f'sacct -j {j["job_id"]} -X -n -P -o State,ElapsedRaw,NodeList,AllocTRES'],
                                   capture_output=True, text=True).stdout.strip().splitlines()
                if not o:
                    continue
                st, el, node, tres = o[0].split('|')
                j.update(state=st.split()[0], node=node, tres=tres, elapsed_hours=round(int(el) / 3600, 4))
            save(d)
            print(json.dumps(dict(spent=d['spent_gpu_hours'], reserved=d['reserved_gpu_hours'],
                                  jobs=[(j['attempt'], j['job_id'], j.get('state'), j.get('elapsed_hours'))
                                        for j in d['jobs']])))
        else:
            raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
