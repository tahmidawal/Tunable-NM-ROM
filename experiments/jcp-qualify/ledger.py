"""GPU-hour ledger with atomic time-limit reservations (DESIGN 9.1, A1.11, A2.6). runs/GPU-LEDGER.json, guarded by an
exclusive file lock (runs/GPU-LEDGER.lock) for every read-modify-write.

    python ledger.py pending '<json>'             set the pending core reservations {attempt: hours} (A3.1, A4.1)
    python ledger.py reserve <attempt> <hours>    atomically: refuse unless spent + reserved + hours + pending <= 35 and the attempt
                                                  has no live reservation; then record a PENDING_SUBMIT reservation
    python ledger.py bind <attempt> <jobid>       the reservation is now job <jobid> (state SUBMITTED)
    python ledger.py release <attempt>            submission failed before sbatch returned a job id
    python ledger.py reconcile                    sacct (all attempts incl. requeues, -D): terminal -> elapsed recorded
    python ledger.py show
    python ledger.py ambiguous <attempt>          sbatch outcome unknown: keep the reservation
    python ledger.py settle <attempt>             resolve an ambiguous / pending reservation by the job name jq_<attempt>
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
CAP = 35.
# non-deferrable jobs (DESIGN A3.1): each must be in the pending inventory until it has a live or completed reservation
MANDATORY = ('acc1024', 'acc4096', 'acc256', 'time1024', 'time4096', 'time256', 'order', 'q3d')
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


FROZEN_RULE = {128: 'fib4181', 256: 'gauss128', 384: 'gauss128', 512: 'gauss192'}


def staged_subjects(att):
    """Configuration keys of the subjects of a staged timing attempt (its RESERVATION runs -> staged config)."""
    out = set()
    rf = HERE / 'runs' / att / 'RESERVATION.json'
    if not rf.exists():
        return out
    for drv, cf in json.loads(rf.read_text())['runs']:
        c = json.loads((HERE / cf).read_text()) if not (HERE / 'runs' / att / 'experiments').exists() else \
            json.loads(next((HERE / 'runs' / att).rglob(Path(cf).name)).read_text())
        for sp in c.get('subjects', []):
            r = int(sp['Rp'])
            out.add(f"{sp['mesh']}|R{r}|{sp['form']}|{sp['scheme']}|{float(sp['factor']):g}" + ('|fb' if sp['rule'] != FROZEN_RULE[r] else ''))
    return out


def staged_fallback_of(att):
    """Configuration keys a staged fallback attempt runs (its config's 'fallback_of')."""
    out = set()
    rf = HERE / 'runs' / att / 'RESERVATION.json'
    if rf.exists():
        for drv, cf in json.loads(rf.read_text())['runs']:
            c = json.loads(next((HERE / 'runs' / att).rglob(Path(cf).name)).read_text())
            out |= set(c.get('fallback_of', []))
    return out


def failing_from_analysis(m):
    """The mesh's configurations whose Q2 is not 'pass' (non-fallback), from checks/analysis.json (A4.1 evidence)."""
    an = json.loads((HERE / 'checks' / 'analysis.json').read_text())['configurations']
    return sorted(k for k, v in an.items() if k.startswith(f'{m}|') and not k.endswith('|fb') and v['Q2'].get('label') != 'pass'
                  and 'P1' not in v['source'])


def validate(d, new, exempt=None):
    """A3.1 / A4.1 / A5.1 inventory rules. pending entries are numeric GPU-h only (finite, >= 0, not bool); every
    mandatory job not yet live or completed is listed (the attempt being reserved is exempt). Per mesh: while its Q2
    labels are unknown, an allowance fballow<m> >= 2; once known (ledger.py fbstate <m> '<keys>', append-only), every
    listed fallback configuration needs a fallback job fb<m>* that is pending (> 0), live or completed, and the mesh's
    timing must cover the fallback subjects: a not-yet-submitted panel is re-priced by its certificate at staging
    (cluster/stage.py checks the fallback subjects are in the config and resources.py prices them); a panel already
    submitted or completed without them requires a repeat time<m>r<k> pending (> 0), live or completed that covers them."""
    live = {j['attempt'] for j in d['jobs'] if j['state'] not in TERMINAL}
    done = {j['attempt'] for j in d['jobs'] if j['state'] == 'COMPLETED'}
    probs = [f'bad value {k}' for k, v in new.items()
             if isinstance(v, bool) or not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)]
    pat = re.compile(r'^(fballow(256|1024|4096)|fb(256|1024|4096)[a-z0-9]*|time(256|1024|4096)r[0-9]+)$')
    probs += [f'unknown pending key {k}' for k in new if k not in MANDATORY and not pat.match(k)]
    probs += [f'{k}: a job projection below 0.25 GPU-h' for k, v in new.items()
              if not k.startswith('fballow') and isinstance(v, (int, float)) and not isinstance(v, bool) and v < .25]
    for k, v in new.items():                       # a certified job is never pending below its certificate
        cert = HERE / 'checks' / f'resources-{k}.json'
        if cert.exists() and isinstance(v, (int, float)) and v < json.loads(cert.read_text())['time_limit_hours'] - 1e-9:
            probs.append(f'{k} pending below its certificate')
    probs += [f'missing mandatory {k}' for k in MANDATORY if k != exempt and k not in new and k not in live and k not in done]
    fs = d.get('fallback_state', {})
    for m in (256, 1024, 4096):
        if str(m) not in fs:
            if new.get(f'fballow{m}', 0) < 2:
                probs.append(f'fballow{m} >= 2 GPU-h required until the Q2 labels of mesh {m} are recorded (fbstate)')
            continue
        keys = fs[str(m)]
        if not keys:
            continue
        covered = set()
        for a_ in live | done | ({exempt} if exempt else set()):
            if a_.startswith(f'fb{m}') and not a_.startswith('fballow'):
                covered |= staged_fallback_of(a_)
        pend_fb = any(k.startswith(f'fb{m}') and not k.startswith('fballow') and new[k] > 0 for k in new)
        if not (set(keys) <= covered or pend_fb):
            probs.append(f'fallback configurations of mesh {m} not covered by a live/completed or pending fallback job')
        panels = [a for a in live | done if a == f'time{m}' or a.startswith(f'time{m}r')]
        if panels and not any(set(keys) <= staged_subjects(a) for a in panels):
            rep = [k for k, v in new.items() if k.startswith(f'time{m}r') and v > 0] + \
                ([exempt] if exempt and exempt.startswith(f'time{m}r') else [])
            if not rep:
                probs.append(f'time{m} ran without the fallback subjects: a repeat time{m}r<k> must be pending')
    probs += [f'pending entry already live: {k}' for k in set(new) & live]
    return probs


def pend(d, att):
    """Pending core reservation excluding the attempt being submitted (A3.1)."""
    return round(sum(v for k, v in d.get('pending_core', {}).items() if k != att), 4)


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
            elif att.startswith('time') and not set(d.get('fallback_state', {}).get(att[4:].split('r')[0], [])) <= staged_subjects(att):
                print('refused: the staged timing panel lacks recorded fallback subjects (restage it)'); rc = 12
            elif any(j['attempt'] == att and j['state'] not in TERMINAL for j in d['jobs']):
                print('attempt already has a live reservation'); rc = 2
            elif d['spent_gpu_hours'] + d['reserved_gpu_hours'] + h + pend(d, att) > CAP:
                print(f"refused: spent {d['spent_gpu_hours']} + reserved {d['reserved_gpu_hours']} + {h} + pending core "
                      f"{pend(d, att)} > {CAP}"); rc = 1
            else:
                popped = d['pending_core'].pop(att, None)     # from now on a live reservation: never counted twice
                d['jobs'].append(dict(attempt=att, job_id=None, time_limit_hours=h, state='PENDING_SUBMIT', gpu=None,
                                      elapsed_hours=None, pending_hours=popped))
        elif cmd == 'pending':             # DESIGN A3.1, A4.1, A5.1: projected time limits of not-yet-submitted
            new = json.loads(sys.argv[2])                  # non-deferrable jobs and fallback allowances, one transaction
            probs = validate(d, new)
            if probs:
                print('refused pending inventory:', probs); rc = 9
            else:
                d['pending_core'] = new
        elif cmd == 'fbstate':             # one transaction (A4.1): record a mesh's Q2 outcome AND the new pending inventory
            m, newp = sys.argv[2], json.loads(sys.argv[3])
            keys = failing_from_analysis(m)            # derived from the analysed accuracy job, never typed
            old = d.setdefault('fallback_state', {}).get(m)
            trial = dict(d, fallback_state=dict(d['fallback_state'], **{m: keys}))
            probs = validate(trial, newp)
            if old is not None and not set(old) <= set(keys):
                print('refused: fallback state is append-only'); rc = 11
            elif probs:
                print('refused:', probs); rc = 9
            else:
                d['fallback_state'][m] = keys
                d['pending_core'] = newp
                print('fallback state', m, keys)
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
                                f"sacct -n -X -P --name jq_{att} -S now-7days -o JobID"], capture_output=True, text=True)
            ids = [x for x in o.stdout.split() if x.strip()]
            qq = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                 f'squeue -h -u tawal01 -n jq_{att} -o %i'], capture_output=True, text=True)
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
