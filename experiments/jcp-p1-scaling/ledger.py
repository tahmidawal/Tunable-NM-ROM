"""GPU-hour ledger with atomic time-limit reservations (DESIGN 9.1, A1.11, A2.6). runs/GPU-LEDGER.json, guarded by an
exclusive file lock (runs/GPU-LEDGER.lock) for every read-modify-write.

    python ledger.py reserve <attempt> <hours>    atomically: refuse unless spent + reserved + hours <= 20 and the attempt
                                                  has no live reservation; then record a PENDING_SUBMIT reservation
    python ledger.py bind <attempt> <jobid>       the reservation is now job <jobid> (state SUBMITTED)
    python ledger.py release <attempt>            submission failed before sbatch returned a job id
    python ledger.py reconcile                    sacct (all attempts incl. requeues, -D): terminal -> elapsed recorded
    python ledger.py show
    python ledger.py ambiguous <attempt>          sbatch outcome unknown: keep the reservation
    python ledger.py settle <attempt>             resolve an ambiguous / pending reservation by the job name p1s_<attempt>
Time limits are never raised (no scontrol update); a job needing more time is resubmitted as a new attempt.
A PENDING_SUBMIT reservation never expires by itself (a crash between reserve and bind keeps the hours reserved until
an explicit release after checking squeue). Every attempt of a job (requeues) is summed from sacct -D.
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
CAP = 20.
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE',
            'RELEASED')


@contextmanager
def locked():
    LED.parent.mkdir(exist_ok=True)
    with open(LOCK, 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            d = json.loads(LED.read_text()) if LED.exists() else dict(cap_gpu_hours=CAP, jobs=[])
            yield d
            totals(d)
            tmp = LED.with_suffix('.tmp')
            tmp.write_text(json.dumps(d, indent=1) + '\n')
            tmp.replace(LED)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def totals(d):
    d['spent_gpu_hours'] = round(sum(j.get('elapsed_hours') or 0. for j in d['jobs'] if j['state'] in TERMINAL), 4)
    d['reserved_gpu_hours'] = round(sum(j['time_limit_hours'] + (j.get('prior_elapsed_hours') or 0.)
                                        for j in d['jobs'] if j['state'] not in TERMINAL), 4)
    return d


def main():
    cmd = sys.argv[1]
    rc = 0
    with locked() as d:
        totals(d)
        if cmd == 'reserve':
            att, h = sys.argv[2], float(sys.argv[3])
            if any(j['attempt'] == att and j['state'] not in TERMINAL for j in d['jobs']):
                print('attempt already has a live reservation'); rc = 2
            elif d['spent_gpu_hours'] + d['reserved_gpu_hours'] + h > CAP:
                print(f"refused: spent {d['spent_gpu_hours']} + reserved {d['reserved_gpu_hours']} + {h} > {CAP}"); rc = 1
            else:
                d['jobs'].append(dict(attempt=att, job_id=None, time_limit_hours=h, state='PENDING_SUBMIT', gpu=None,
                                      elapsed_hours=None))
        elif cmd == 'bind':
            j = [x for x in d['jobs'] if x['attempt'] == sys.argv[2] and x['state'] == 'PENDING_SUBMIT'][-1]
            j.update(job_id=sys.argv[3], state='SUBMITTED')
        elif cmd == 'ambiguous':
            for j in d['jobs']:
                if j['attempt'] == sys.argv[2] and j['state'] == 'PENDING_SUBMIT':
                    j['state'] = 'AMBIGUOUS'
        elif cmd == 'settle':                 # find the job by its unique name; bind it, or release if it never existed
            att = sys.argv[2]
            o = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                f"sacct -n -X -P --name p1s_{att} -S now-7days -o JobID"], capture_output=True, text=True)
            ids = [x for x in o.stdout.split() if x.strip()]
            for j in d['jobs']:
                if j['attempt'] == att and j['state'] in ('AMBIGUOUS', 'PENDING_SUBMIT'):
                    if o.returncode != 0:
                        print('sacct failed: reservation kept'); rc = 3
                    elif ids:
                        j.update(job_id=ids[-1], state='SUBMITTED')
                    else:
                        j.update(state='RELEASED', elapsed_hours=0.)
        elif cmd == 'release':
            for j in d['jobs']:
                if j['attempt'] == sys.argv[2] and j['state'] == 'PENDING_SUBMIT':
                    j.update(state='RELEASED', elapsed_hours=0.)
        elif cmd == 'reconcile':
            for j in d['jobs']:
                if j['state'] in TERMINAL or not j['job_id']:
                    continue
                o = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                    f"sacct -j {j['job_id']} -X -D -n -P -o State,ElapsedRaw,AllocTRES,NodeList,TimelimitRaw"],
                                   capture_output=True, text=True).stdout.strip().splitlines()
                if not o:
                    continue
                rows = [(x.split('|') + [''] * 5)[:5] for x in o]
                st = rows[-1][0].split()[0]
                j['attempts'] = len(rows)
                j['state'], j['node'] = st, rows[-1][3]
                j['gpu'] = next((x for x in rows[-1][2].split(',') if x.startswith('gres/gpu')), None)
                j['elapsed_hours'] = round(sum(int(r[1] or 0) for r in rows) / 3600, 4)
                j['prior_elapsed_hours'] = round(sum(int(r[1] or 0) for r in rows[:-1]) / 3600, 4)   # earlier requeues
                tl = rows[-1][4]
                if tl.isdigit() and int(tl) / 60 > j['time_limit_hours'] + 1e-9:
                    j['time_limit_hours'] = int(tl) / 60          # a raised limit is charged (A2.6)
        out = {k: v for k, v in totals(d).items() if k != 'jobs'}
    print(json.dumps(out))
    sys.exit(rc)


if __name__ == '__main__':
    main()
