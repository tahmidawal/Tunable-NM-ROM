"""jcp-dims (copied from jcp-qualify ledger.py; edited: cap 55, never-dropped list, no fallback state, job prefix jd_).
GPU-hour ledger with atomic time-limit reservations (DESIGN 9.1, A1.11, A2.6). runs/GPU-LEDGER.json, guarded by an
exclusive file lock (runs/GPU-LEDGER.lock) for every read-modify-write.

    python ledger.py pending '<json>'             set the pending core reservations {attempt: hours} (A3.1, A4.1)
    python ledger.py reserve <attempt> <hours>    atomically: refuse unless spent + reserved + hours + pending <= 55 and the attempt
                                                  has no live reservation; then record a PENDING_SUBMIT reservation
    python ledger.py bind <attempt> <jobid>       the reservation is now job <jobid> (state SUBMITTED)
    python ledger.py release <attempt>            submission failed before sbatch returned a job id
    python ledger.py reconcile                    sacct (all attempts incl. requeues, -D): terminal -> elapsed recorded
    python ledger.py show
    python ledger.py ambiguous <attempt>          sbatch outcome unknown: keep the reservation
    python ledger.py settle <attempt>             resolve an ambiguous / pending reservation by the job name jd_<attempt>
Time limits are never raised (no scontrol update); a job needing more time is resubmitted as a new attempt.
A PENDING_SUBMIT reservation never expires by itself (a crash between reserve and bind keeps the hours reserved until
an explicit release after checking squeue). Every attempt of a job (requeues) is summed from sacct -D.
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
CAP = 55.
# non-deferrable jobs (DESIGN A3.1): each must be in the pending inventory until it has a live or completed reservation
MANDATORY = ('f2a', 'f2b', 'f2k', 'b2f', 'q3r4', 'q3r5', 'q3r6', 'q3r7', 'q3r8', 'm3a', 'm3b', 'k3', 't3s', 't3c')
# jcp-dims never-dropped (A1.6); rounds 5-8 are mandatory until waived ('ledger.py waive q3r<k> <reason>', e.g. no exposed
# untested candidate after the previous round), which is recorded in the ledger
# a failed / timed-out mandatory job is not COMPLETED, so the next 'pending' call must list it again (validated)
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


def validate(d, new, exempt=None):
    """jcp-dims (DESIGN 8, A1.6): every never-dropped job that has neither a live nor a COMPLETED reservation must be in
    the pending inventory with a positive projection (certificate floor: checks/resources-<job>.json if it exists)."""
    probs = [f'pending entry {k} not a finite non-negative number' for k, v in new.items()
             if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)]
    live = {j['attempt'] for j in d['jobs'] if j['state'] not in TERMINAL}
    done = {j['attempt'] for j in d['jobs'] if j['state'] == 'COMPLETED'}

    def covered(k):                                 # a mandatory job or any of its retries k r<n>
        return any(a == k or re.fullmatch(re.escape(k) + r'r\d+', a) for a in live | done)
    for k in MANDATORY:
        if covered(k) or k == exempt or k in d.get('waived', {}):
            continue
        if not (k in new and isinstance(new[k], (int, float)) and math.isfinite(new[k]) and new[k] > 0):
            probs.append(f'never-dropped job {k} missing from the pending inventory')
            continue
        c = HERE / 'checks' / f'resources-{k}.json'
        if c.exists() and new[k] + 1e-9 < json.loads(c.read_text())['time_limit_hours']:
            probs.append(f'{k}: pending {new[k]} below its certificate')
    probs += [f'pending entry already live: {k}' for k in set(new) & live]
    return probs


def base_of(att):
    """A retry '<job>r<k>' replaces its base job's pending allowance (jcp-dims)."""
    m = re.fullmatch(r'(.+?)r\d+', att)
    return m.group(1) if m and m.group(1) in MANDATORY else att


def pend(d, att):
    """Pending core reservation excluding the attempt being submitted and, for a retry, its base job (A3.1)."""
    return round(sum(v for k, v in d.get('pending_core', {}).items() if k not in (att, base_of(att))), 4)


def main():
    cmd = sys.argv[1]
    rc = 0
    with locked() as d:
        totals(d)
        if cmd == 'reserve':
            att, h = sys.argv[2], float(sys.argv[3])
            if not (math.isfinite(h) and h > 0):
                print('refused: reservation hours must be finite and positive'); rc = 10
            elif 'pending_core' not in d:
                print('refused: no pending-core inventory set (ledger.py pending; DESIGN A3.1)'); rc = 8
            elif validate(d, {k: v for k, v in d['pending_core'].items()}, exempt=att):
                print('refused: pending inventory no longer valid:', validate(d, d['pending_core'], exempt=att)); rc = 8
            elif any(j['attempt'] == att and j['state'] not in TERMINAL for j in d['jobs']):
                print('attempt already has a live reservation'); rc = 2
            elif d['spent_gpu_hours'] + d['reserved_gpu_hours'] + h + pend(d, att) > CAP:
                print(f"refused: spent {d['spent_gpu_hours']} + reserved {d['reserved_gpu_hours']} + {h} + pending core "
                      f"{pend(d, att)} > {CAP}"); rc = 1
            else:
                popped = d['pending_core'].pop(att, None)     # from now on a live reservation: never counted twice
                if base_of(att) != att:                         # a retry consumes its base job's allowance
                    popped = d['pending_core'].pop(base_of(att), popped)
                d['jobs'].append(dict(attempt=att, job_id=None, time_limit_hours=h, state='PENDING_SUBMIT', gpu=None,
                                      elapsed_hours=None, pending_hours=popped))
        elif cmd == 'pending':             # DESIGN A3.1, A4.1, A5.1: projected time limits of not-yet-submitted
            new = json.loads(sys.argv[2])                  # non-deferrable jobs and fallback allowances, one transaction
            probs = validate(d, new)
            if probs:
                print('refused pending inventory:', probs); rc = 9
            else:
                d['pending_core'] = new
        elif cmd == 'waive':                 # a mandatory job shown unnecessary (recorded with its reason, never silent)
            att, why = sys.argv[2], ' '.join(sys.argv[3:])
            if att not in MANDATORY or not why:
                print('refused: waive needs a mandatory attempt and a reason'); rc = 15
            else:
                d.setdefault('waived', {})[att] = why
                d.get('pending_core', {}).pop(att, None)
        elif cmd == 'bind':
            j = [x for x in d['jobs'] if x['attempt'] == sys.argv[2] and x['state'] == 'PENDING_SUBMIT'][-1]
            j.update(job_id=sys.argv[3], state='SUBMITTED')
            d.get('pending_core', {}).pop(sys.argv[2], None)      # now a live reservation: never counted twice
        elif cmd == 'ambiguous':
            for j in d['jobs']:
                if j['attempt'] == sys.argv[2] and j['state'] == 'PENDING_SUBMIT':
                    j['state'] = 'AMBIGUOUS'
        elif cmd == 'settle':                 # find the job by its unique name; bind it, or release if it never existed
            att = sys.argv[2]
            o = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                f"sacct -n -X -P --name jd_{att} -S now-7days -o JobID"], capture_output=True, text=True)
            ids = [x for x in o.stdout.split() if x.strip()]
            qq = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                 f'squeue -h -u tawal01 -n jd_{att} -o %i'], capture_output=True, text=True)
            ids += [x for x in qq.stdout.split() if x.strip()]
            for j in d['jobs']:
                if j['attempt'] == att and j['state'] in ('AMBIGUOUS', 'PENDING_SUBMIT'):
                    if o.returncode != 0 or qq.returncode != 0:
                        print('sacct/squeue failed: reservation kept'); rc = 3
                    elif ids:
                        j.update(job_id=ids[-1], state='SUBMITTED')
                        d.get('pending_core', {}).pop(att, None)
                    elif len(sys.argv) > 3 and sys.argv[3] == '--confirmed-absent':   # only after a manual check
                        j.update(state='RELEASED', elapsed_hours=0.)
                        if j.get('pending_hours') is not None:        # restore the pending core reservation
                            d['pending_core'][att] = j['pending_hours']
                    else:
                        print('no record found: reservation kept (re-run later, or pass --confirmed-absent)'); rc = 4
        elif cmd == 'release':
            for j in d['jobs']:
                if j['attempt'] == sys.argv[2] and j['state'] == 'PENDING_SUBMIT':
                    j.update(state='RELEASED', elapsed_hours=0.)
                    if j.get('pending_hours') is not None:            # restore the pending core reservation
                        d['pending_core'][sys.argv[2]] = j['pending_hours']
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
                if st in TERMINAL and st != 'COMPLETED' and j.get('pending_hours') is not None:
                    d.setdefault('pending_core', {})[j['attempt']] = j['pending_hours']   # failed core job: re-pend
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
