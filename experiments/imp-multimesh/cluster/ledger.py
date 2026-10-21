"""GPU-hour ledger (COMMON-BRIEF 7): runs/GPU-LEDGER.json.
    ledger.py check <attempt> <projected_hours>   -> exit 0 iff spent + reserved + projected <= cap
    ledger.py bind <attempt> <jobid> <limit_hours>
    ledger.py settle            (sacct for every unsettled job: state, elapsed GPU-hours, gpu)
"""
import json
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'runs' / 'GPU-LEDGER.json'
CAP = 8.0


def load():
    return json.loads(P.read_text()) if P.exists() else dict(cap_gpu_hours=CAP, jobs=[], spent_gpu_hours=0.0)


def save(d):
    d['spent_gpu_hours'] = round(sum(j.get('elapsed_hours', 0.0) for j in d['jobs']), 4)
    P.write_text(json.dumps(d, indent=1) + '\n')


def main():
    d = load()
    cmd = sys.argv[1]
    if cmd == 'check':
        a, proj = sys.argv[2], float(sys.argv[3])
        reserved = sum(j['limit_hours'] for j in d['jobs'] if j.get('state') in (None, 'PENDING', 'RUNNING'))
        tot = d['spent_gpu_hours'] + reserved + proj
        print(f'spent {d["spent_gpu_hours"]:.3f} + reserved(running limits) {reserved:.3f} + projected {proj:.3f} = {tot:.3f} / cap {CAP}')
        sys.exit(0 if tot <= CAP else 6)
    if cmd == 'bind':
        a, jid, lim = sys.argv[2], sys.argv[3], float(sys.argv[4])
        d['jobs'].append(dict(attempt=a, job_id=jid, limit_hours=lim, state='PENDING'))
        save(d)
        return
    if cmd == 'settle':
        for j in d['jobs']:
            if j.get('state') in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL') and j.get('final'):
                continue
            o = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                                f'sacct -j {j["job_id"]} -X -n -P -o State,ElapsedRaw,NodeList,AllocTRES'],
                               capture_output=True, text=True).stdout.strip().splitlines()
            if not o:
                continue
            st, el, node, tres = o[0].split('|')
            st = st.split()[0]
            j.update(state=st, elapsed_hours=round(int(el) / 3600, 4), node=node, tres=tres)
            j['final'] = st in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')
        save(d)
        print(json.dumps(dict(spent=d['spent_gpu_hours'], jobs=[(j['attempt'], j['job_id'], j.get('state'),
                                                                  j.get('elapsed_hours')) for j in d['jobs']])))


if __name__ == '__main__':
    main()
