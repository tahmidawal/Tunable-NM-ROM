"""GPU-hour ledger with time-limit reservations (DESIGN 9.1, A1.11, A2.6). runs/GPU-LEDGER.json.

    python ledger.py check <hours>        exit 0 iff spent + reserved + hours <= 20
    python ledger.py add <attempt> <jobid> <hours>
    python ledger.py reconcile            sacct every non-terminal job; terminal -> elapsed recorded, reservation released
    python ledger.py show
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LED = HERE / 'runs' / 'GPU-LEDGER.json'
CAP = 20.
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def load():
    if LED.exists():
        return json.loads(LED.read_text())
    return dict(cap_gpu_hours=CAP, jobs=[], spent_gpu_hours=0., reserved_gpu_hours=0.)


def totals(d):
    d['spent_gpu_hours'] = round(sum(j.get('elapsed_hours') or 0. for j in d['jobs'] if j['state'] in TERMINAL), 4)
    d['reserved_gpu_hours'] = round(sum(j['time_limit_hours'] for j in d['jobs'] if j['state'] not in TERMINAL), 4)
    return d


def save(d):
    LED.parent.mkdir(exist_ok=True)
    LED.write_text(json.dumps(totals(d), indent=1) + '\n')


def main():
    d = totals(load())
    cmd = sys.argv[1]
    if cmd == 'check':
        h = float(sys.argv[2])
        tot = d['spent_gpu_hours'] + d['reserved_gpu_hours'] + h
        print(f"spent {d['spent_gpu_hours']} + reserved {d['reserved_gpu_hours']} + new {h} = {tot:.3f} (cap {CAP})")
        sys.exit(0 if tot <= CAP else 1)
    if cmd == 'add':
        d['jobs'].append(dict(attempt=sys.argv[2], job_id=sys.argv[3], time_limit_hours=float(sys.argv[4]), state='SUBMITTED',
                              gpu=None, elapsed_hours=None))
        save(d)
    elif cmd == 'reconcile':
        for j in d['jobs']:
            if j['state'] in TERMINAL:
                continue
            o = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                f"sacct -j {j['job_id']} -X -n -P -o State,ElapsedRaw,AllocTRES,NodeList"],
                               capture_output=True, text=True).stdout.strip().splitlines()
            if not o:
                continue
            st, el, tres, node = (o[0].split('|') + ['', '', '', ''])[:4]
            st = st.split()[0]
            j['state'], j['node'] = st, node
            j['gpu'] = next((x for x in tres.split(',') if x.startswith('gres/gpu:')), None)
            j['elapsed_hours'] = round(int(el or 0) / 3600, 4)
        save(d)
    d = totals(load())
    print(json.dumps({k: v for k, v in d.items() if k != 'jobs'}), len(d['jobs']), 'jobs')


if __name__ == '__main__':
    main()
