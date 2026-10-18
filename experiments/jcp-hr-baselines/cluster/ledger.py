"""Atomic GPU-ledger operations (A33.2, A34.2). Every read-check-write of runs/GPU-LEDGER.json happens under one exclusive
fcntl lock (runs/GPU-LEDGER.lock) and is written by atomic replacement.

    ledger.py reserve <attempt> <driver> <config> <gpu> <commit> <wall_hours> <stageb_reserve_hours>
    ledger.py submit-check <attempt>          # the attempt holds exactly one unbound reservation; total <= cap
    ledger.py bind <attempt> <slurm_job_id>   # binds the reservation to its one submission (no reuse)
    ledger.py reconcile <attempt> <slurm_job_id> <gpu_hours> <state>
"""
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
LED = LANE / 'runs' / 'GPU-LEDGER.json'
LOCK = LANE / 'runs' / 'GPU-LEDGER.lock'
CAP = 30.0


@contextmanager
def locked():
    with open(LOCK, 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            L = json.loads(LED.read_text())
            yield L
            tmp = LED.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(L, indent=1) + '\n')
            os.replace(tmp, LED)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def total(L):
    return sum(float(j['measured_gpu_hours']) if j.get('measured_gpu_hours') is not None else float(j['requested_hours'])
               for j in L['jobs'])


def main():
    op, a = sys.argv[1], sys.argv[2]
    with locked() as L:
        mine = [j for j in L['jobs'] if j['attempt'] == a]
        if op == 'reserve':
            driver, cfg, gpu, commit, wall, extra = sys.argv[3:9]
            assert not mine, f'attempt {a} already in the ledger (a retry needs a new attempt name)'
            assert total(L) + float(wall) + float(extra) <= CAP + 1e-9, ('cap', total(L), wall, extra)
            L['jobs'].append(dict(attempt=a, slurm=None, gpu=gpu, commit=commit, role=f'{driver} {cfg}',
                                  requested_hours=float(wall), measured_gpu_hours=None, status='reserved at staging'))
        elif op == 'submit-check':
            assert len(mine) == 1 and mine[0]['status'] == 'reserved at staging' and mine[0]['slurm'] is None, \
                f'attempt {a}: no unbound reservation ({[m.get("status") for m in mine]})'
            assert total(L) <= CAP + 1e-9, ('cap', total(L))
            mine[0]['status'] = 'submitting'
        elif op == 'bind':
            assert len(mine) == 1 and mine[0]['status'] == 'submitting' and mine[0]['slurm'] is None, mine
            mine[0].update(slurm=int(sys.argv[3]), status='submitted')
        elif op == 'reconcile':
            job, hrs, state = int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
            assert len(mine) == 1 and mine[0]['slurm'] == job, (a, job, mine)
            mine[0].update(measured_gpu_hours=hrs, status=f'{state}; collected')
        else:
            raise SystemExit(f'unknown op {op}')
        L['spent_gpu_hours'] = round(sum(float(j['measured_gpu_hours']) for j in L['jobs']
                                         if j.get('measured_gpu_hours') is not None), 4)
        print(f'{op} {a}: ledger total (measured or reserved) {total(L):.3f} GPU-h (cap {CAP})')


if __name__ == '__main__':
    main()
