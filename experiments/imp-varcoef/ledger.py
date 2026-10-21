"""GPU-hour ledger (COMMON-BRIEF §7): runs/GPU-LEDGER.json.

    python ledger.py submit <job> <jobid>      record a submission (reservation = time limit)
    python ledger.py update                    refresh every job's state and elapsed GPU-hours from sacct
    python ledger.py show
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LED = HERE / 'runs' / 'GPU-LEDGER.json'
CAP, PHASE1_CAP = 15.0, 8.0
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def load():
    if LED.exists():
        return json.loads(LED.read_text())
    return dict(lane='imp-varcoef', cap_gpu_hours=CAP, phase1_cap_gpu_hours=PHASE1_CAP, jobs=[], total_gpu_hours=0.0)


def save(d):
    d['total_gpu_hours'] = round(sum(j.get('gpu_hours') or 0.0 for j in d['jobs']), 4)
    d['committed_gpu_hours'] = round(sum((j.get('gpu_hours') or 0.0) if j.get('final') else j['time_limit_hours']
                                         for j in d['jobs']), 4)        # finished: actual; queued/running: time limit
    d['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    LED.parent.mkdir(exist_ok=True)
    tmp = LED.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=1) + '\n')
    tmp.replace(LED)                                                    # atomic replacement


def sacct(jid):
    out = subprocess.check_output(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                   f'sacct -j {jid} -X -n -P -o State,ElapsedRaw,AllocTRES,NodeList'], text=True)
    line = out.strip().splitlines()[0].split('|')
    tres = line[2]
    ngpu = 1
    for t in tres.split(','):
        if t.startswith('gres/gpu=') or t.startswith('gres/gpu:'):
            try:
                ngpu = int(t.split('=')[-1])
            except ValueError:
                pass
    return dict(state=line[0].split()[0], elapsed_s=int(line[1] or 0), ngpu=ngpu, tres=tres, node=line[3])


def main():
    d = load()
    cmd = sys.argv[1]
    if cmd == 'submit':
        job, jid = sys.argv[2], sys.argv[3]
        assert not any(j['jobid'] == jid for j in d['jobs']), 'job id already recorded'
        res = json.loads((HERE / 'runs' / job / 'RESERVATION.json').read_text())
        d['jobs'].append(dict(job=job, jobid=jid, gpu_request=res['gpu'], time_limit_hours=res['time_limit_hours'],
                              submitted_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state='PENDING',
                              gpu_hours=0.0))
    elif cmd == 'update':
        for j in d['jobs']:
            if j.get('state') in TERMINAL and j.get('final'):
                continue
            s = sacct(j['jobid'])
            j.update(state=s['state'], elapsed_s=s['elapsed_s'], ngpu=s['ngpu'], tres=s['tres'], node=s['node'],
                     gpu_hours=round(s['elapsed_s'] * s['ngpu'] / 3600.0, 4), final=s['state'] in TERMINAL)
    save(d)
    print(json.dumps(dict(total=d['total_gpu_hours'], jobs=[(j['job'], j['jobid'], j.get('state'), j.get('gpu_hours'))
                                                            for j in d['jobs']]), indent=0))


if __name__ == '__main__':
    main()
