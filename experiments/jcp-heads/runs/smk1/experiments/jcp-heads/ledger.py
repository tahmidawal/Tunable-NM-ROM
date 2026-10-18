"""[jcp-heads, adapted from jcp-qualify @4399491eb ledger.py: same reservation mechanics; the mesh-fallback machinery is
removed (no fallback configurations in this lane); mandatory set and job-name prefix edited.]

GPU-hour ledger with atomic time-limit reservations (DESIGN 8, A1.9, A2.7). runs/GPU-LEDGER.json, guarded by an
exclusive file lock (runs/GPU-LEDGER.lock) for every read-modify-write.

    python ledger.py pending '<json>'             set the pending non-deferrable reservations {attempt: hours}
    python ledger.py reserve <attempt> <hours>    atomically: refuse unless spent + reserved + hours + pending(other) <= 35
                                                  and the attempt has no live reservation; record PENDING_SUBMIT
    python ledger.py bind <attempt> <jobid>       the reservation is now job <jobid> (SUBMITTED)
    python ledger.py release <attempt>            submission failed before sbatch returned a job id
    python ledger.py ambiguous <attempt>          sbatch outcome unknown: keep the reservation
    python ledger.py settle <attempt> [--confirmed-absent]   resolve by the job name jh_<attempt>
    python ledger.py reconcile                    sacct (all attempts incl. requeues, -D): terminal -> elapsed recorded
    python ledger.py show
Smokes (smk*) are charged like any job. Time limits are never raised. A failed / timed-out non-deferrable job returns
to the pending inventory (DESIGN A2.7).
"""
import fcntl
import json
import math
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
LED = HERE / 'runs' / 'GPU-LEDGER.json'
LOCK = HERE / 'runs' / 'GPU-LEDGER.lock'
CAP = 35.
MANDATORY = ('train', 'rule1024', 'acc1024', 'time1024', 'time256')       # DESIGN A2.7
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE',
            'RELEASED')
SSH = ['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login']


@contextmanager
def locked():
    LED.parent.mkdir(exist_ok=True)
    with open(LOCK, 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            d = json.loads(LED.read_text()) if LED.exists() else dict(cap_gpu_hours=CAP, jobs=[], pending_core={})
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


def done_or_live(d, att):
    """A mandatory job is covered when an attempt of it (att or a retry att r<n>) is live or COMPLETED."""
    pat = re.compile(rf'^{re.escape(att)}(r\d+)?$')
    return any(pat.match(j['attempt']) and (j['state'] not in TERMINAL or j['state'] == 'COMPLETED') for j in d['jobs'])


def validate(d, new, exempt=None):
    probs = []
    for k, v in new.items():
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0):
            probs.append(f'{k}: hours must be a finite number >= 0')
        if not any(re.match(rf'^{m}(r\d+)?$', k) for m in MANDATORY):
            probs.append(f'unknown pending key {k}')
    for m in MANDATORY:
        if m == exempt or done_or_live(d, m):
            continue
        if not any(re.match(rf'^{m}(r\d+)?$', k) and new[k] > 0 for k in new):
            probs.append(f'mandatory job {m} neither live/completed nor pending')
    return probs


def pend(d, att):
    """Pending reservation excluding the attempt being submitted AND every pending entry of the same job (its base name
    and its retries <base>r<n>): a retry replaces, never adds to, its job's pending reservation (code audit 1)."""
    base = re.sub(r'r\d+$', '', att)
    same = lambda k: re.sub(r'r\d+$', '', k) == base
    return round(sum(v for k, v in d.get('pending_core', {}).items() if not same(k)), 4)


def main():
    cmd = sys.argv[1]
    rc = 0
    with locked() as d:
        totals(d)
        if cmd == 'reserve':
            att, h = sys.argv[2], float(sys.argv[3])
            base = re.sub(r'r\d+$', '', att)
            exempt = base if base in MANDATORY else None
            if not (math.isfinite(h) and h > 0):
                print('refused: reservation hours must be finite and positive'); rc = 10
            elif 'pending_core' not in d or validate(d, d['pending_core'], exempt=exempt):
                print('refused: pending inventory invalid:', validate(d, d.get('pending_core', {}), exempt=exempt)); rc = 8
            elif any(j['attempt'] == att and j['state'] not in TERMINAL for j in d['jobs']):
                print('attempt already has a live reservation'); rc = 2
            elif d['spent_gpu_hours'] + d['reserved_gpu_hours'] + h + pend(d, att) > CAP:
                print(f"refused: spent {d['spent_gpu_hours']} + reserved {d['reserved_gpu_hours']} + {h} + pending "
                      f"{pend(d, att)} > {CAP}"); rc = 1
            else:
                base = re.sub(r'r\d+$', '', att)
                same = [k for k in d['pending_core'] if re.sub(r'r\d+$', '', k) == base]
                popped = sum(d['pending_core'].pop(k) for k in same) if same else None
                d['jobs'].append(dict(attempt=att, job_id=None, time_limit_hours=h, state='PENDING_SUBMIT', gpu=None,
                                      elapsed_hours=None, pending_hours=popped))
        elif cmd == 'pending':
            new = json.loads(sys.argv[2])
            probs = validate(d, new)
            if probs:
                print('refused pending inventory:', probs); rc = 9
            else:
                d['pending_core'] = new
        elif cmd == 'bind':
            j = [x for x in d['jobs'] if x['attempt'] == sys.argv[2] and x['state'] == 'PENDING_SUBMIT'][-1]
            j.update(job_id=sys.argv[3], state='SUBMITTED')
            d.get('pending_core', {}).pop(sys.argv[2], None)
        elif cmd == 'ambiguous':
            for j in d['jobs']:
                if j['attempt'] == sys.argv[2] and j['state'] == 'PENDING_SUBMIT':
                    j['state'] = 'AMBIGUOUS'
        elif cmd == 'settle':
            att = sys.argv[2]
            o = subprocess.run(SSH + [f"sacct -n -X -P --name jh_{att} -S now-7days -o JobID"], capture_output=True, text=True)
            qq = subprocess.run(SSH + [f'squeue -h -u tawal01 -n jh_{att} -o %i'], capture_output=True, text=True)
            ids = [x for x in (o.stdout + ' ' + qq.stdout).split() if x.strip()]
            for j in d['jobs']:
                if j['attempt'] == att and j['state'] in ('AMBIGUOUS', 'PENDING_SUBMIT'):
                    if o.returncode != 0 or qq.returncode != 0:
                        print('sacct/squeue failed: reservation kept'); rc = 3
                    elif ids:
                        j.update(job_id=ids[-1], state='SUBMITTED')
                        d.get('pending_core', {}).pop(att, None)
                    elif len(sys.argv) > 3 and sys.argv[3] == '--confirmed-absent':
                        j.update(state='RELEASED', elapsed_hours=0.)
                        if j.get('pending_hours') is not None:
                            d['pending_core'][att] = j['pending_hours']
                    else:
                        print('no record found: reservation kept (re-run later, or pass --confirmed-absent)'); rc = 4
        elif cmd == 'release':
            for j in d['jobs']:
                if j['attempt'] == sys.argv[2] and j['state'] == 'PENDING_SUBMIT':
                    j.update(state='RELEASED', elapsed_hours=0.)
                    if j.get('pending_hours') is not None:
                        d.setdefault('pending_core', {})[sys.argv[2]] = j['pending_hours']
        elif cmd == 'reconcile':
            for j in d['jobs']:
                if j['state'] in TERMINAL or not j['job_id']:
                    continue
                o = subprocess.run(SSH + [f"sacct -j {j['job_id']} -X -D -n -P -o State,ElapsedRaw,AllocTRES,NodeList,TimelimitRaw"],
                                   capture_output=True, text=True).stdout.strip().splitlines()
                if not o:
                    continue
                rows = [(x.split('|') + [''] * 5)[:5] for x in o]
                st = rows[-1][0].split()[0]
                j['attempts'] = len(rows)
                j['state'], j['node'] = st, rows[-1][3]
                if st in TERMINAL and st != 'COMPLETED' and j.get('pending_hours') is not None:
                    d.setdefault('pending_core', {})[j['attempt']] = j['pending_hours']   # failed mandatory: re-pend
                j['gpu'] = next((x for x in rows[-1][2].split(',') if x.startswith('gres/gpu')), None)
                j['elapsed_hours'] = round(sum(int(r[1] or 0) for r in rows) / 3600, 4)
                j['prior_elapsed_hours'] = round(sum(int(r[1] or 0) for r in rows[:-1]) / 3600, 4)
                tl = rows[-1][4]
                if tl.isdigit() and int(tl) / 60 > j['time_limit_hours'] + 1e-9:
                    j['time_limit_hours'] = int(tl) / 60
        out = {k: v for k, v in totals(d).items() if k != 'jobs'}
    print(json.dumps(out))
    sys.exit(rc)


if __name__ == '__main__':
    main()
