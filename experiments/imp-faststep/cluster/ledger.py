"""GPU-hour ledger (COMMON-BRIEF 7, A1.10): runs/GPU-LEDGER.json. Running/queued jobs reserve their time limit; finished
jobs count their sacct elapsed hours (one GPU each).

    python cluster/ledger.py update            # refresh every job from sacct
    python cluster/ledger.py check <hours>     # exit 1 unless spent + reserved + <hours> <= cap
"""
import json
import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
LED = LANE / 'runs' / 'GPU-LEDGER.json'
CAP = 4.0
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def load():
    return json.loads(LED.read_text()) if LED.exists() else dict(cap_gpu_hours=CAP, jobs=[], spent_gpu_hours=0., reserved_gpu_hours=0.)


def update():
    L = load()
    for sub in sorted((LANE / 'runs').glob('*/SUBMIT.json')):
        job = sub.parent.name
        s = json.loads(sub.read_text())
        res = json.loads((sub.parent / 'stage' / 'RESERVATION.json').read_text())
        e = next((x for x in L['jobs'] if x['job'] == job), None)
        if e is None:
            e = dict(job=job, job_id=s['job_id'], gpu=res['gpu'], time_limit_hours=res['time_limit_hours'])
            L['jobs'].append(e)
        out = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                              f"sacct -j {s['job_id']} -X -n -P -o State,ElapsedRaw,NodeList,AllocTRES"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        if out:
            st, el, node, tres = out[0].split('|')
            e.update(state=st.split()[0], elapsed_hours=round(int(el or 0) / 3600, 4), node=node, tres=tres)
    L['spent_gpu_hours'] = round(sum(x.get('elapsed_hours', 0.) for x in L['jobs'] if x.get('state') in TERMINAL), 4)
    L['reserved_gpu_hours'] = round(sum(x['time_limit_hours'] for x in L['jobs'] if x.get('state') not in TERMINAL), 4)
    LED.write_text(json.dumps(L, indent=1) + '\n')
    return L


if __name__ == '__main__':
    L = update()
    print(json.dumps(dict(spent=L['spent_gpu_hours'], reserved=L['reserved_gpu_hours'], jobs=len(L['jobs']))))
    if len(sys.argv) > 2 and sys.argv[1] == 'check':
        need = float(sys.argv[2])
        ok = L['spent_gpu_hours'] + L['reserved_gpu_hours'] + need <= CAP + 1e-9
        print('admit' if ok else 'REFUSE', need)
        sys.exit(0 if ok else 1)
