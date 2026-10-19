"""jcp-robust: E2 3D configs (DESIGN 3.3; A1.8, A1.9, A2.6, A3.3, A3.5, A3.6).

  selection  python make_configs_e2_3d.py sel <bank>   -> configs/e2q<bank>a.json (W256, W512), e2q<bank>b.json (W768, W1024)
             ladders (d3run 'ladder'): Gauss gl16 ... gl48 candidates, gl56 / gl64 / gl80 certification rungs (gl56 -> gl64 is
             the target check, gl64 -> gl80 the Q2 of the fixed gl64 H-acc-r panel); lattice lat8192 ... lat65536 candidates,
             lat131072 successor; three specs; 64 cases; ADT tight on every case, tighter on cases 0-7
  front      python make_configs_e2_3d.py front <bank> -> configs/e2f<bank>.json from checks/select3d-<bank>.json
             (d3run 'panel': A = W256|gl24|fs|CN|0.01|K1 on the job's bank; slots + fixed configurations; cases 0-7 x 3 reps)
<bank> 0 = incumbent W1024; 1..5 = retrained banks registered from inputs/banks3d/s<seed>/BANK.json (the bank file itself
stays in the cluster archive and is copied into the job by stage.py's cluster_inputs, sha256-verified).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECS = ['fs|CN|0.01|K1', 'fs|CN|0.0125|K1', 'qn|BDF2|0.025|k4']
GAUSS = ['gl16', 'gl20', 'gl24', 'gl32', 'gl40', 'gl48', 'gl56', 'gl64', 'gl80']
GAUSS_CAND = GAUSS[:6]
LAT = ['lat8192', 'lat16384', 'lat32768', 'lat65536', 'lat131072']
LAT_CAND = LAT[:4]
WIDTHS = (256, 512, 768, 1024)
FIXED = ['W256|gl24|fs|CN|0.01|K1', 'W256|gl24|fs|CN|0.0125|K1', 'W256|lat8192|qn|BDF2|0.025|k4',
         'W512|lat16384|qn|BDF2|0.025|k4']
A_REF = dict(setting='W256', rule='gl24', spec='fs|CN|0.01|K1')


def bank_of(b):
    if int(b) == 0:
        return 'W1024', None, []
    meta = json.loads((HERE / 'inputs' / 'banks3d' / f's{b}' / 'BANK.json').read_text())
    name = f'W1024s{b}'
    path = f'inputs/banks3d/s{b}/bank.pkl'
    reg = {name: dict(path=path, sha256=meta['bank_sha256'])}
    cin = [dict(src=meta['archive_path'], dst=f'experiments/jcp-robust/{path}', sha256=meta['bank_sha256'])]
    return name, reg, cin


def settings(bname, widths):
    return [dict(name=f'W{w}', bank=bname, Rp=w, a=6, rule='gl24', reduced=[], extra=[]) for w in widths]


def w_(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=1) + '\n')
    print(name)


def sel(b):
    bname, reg, cin = bank_of(b)
    for tag, widths in (('a', (256, 512)), ('b', (768, 1024))):
        blocks = []
        for wd in widths:
            blocks.append(dict(kind='ladder', setting=f'W{wd}', specs=SPECS, rungs=GAUSS, tighter_cases=list(range(8))))
            blocks.append(dict(kind='ladder', setting=f'W{wd}', specs=SPECS, rungs=LAT, tighter_cases=list(range(8))))
        c = dict(attempt=f'e2q{b}{tag}', driver='e2run3d', mode=f'e2q{b}{tag}', cohort_seed=923801, cohort_count=64,
                 trust_fraction=.05, timing=dict(burn=.1), settings=settings(bname, widths), extra_rules=GAUSS + LAT,
                 blocks=blocks, bank_seed=int(b))
        if reg:
            c['bank_registry'] = reg
            c['cluster_inputs'] = cin
        w_(f'configs/e2q{b}{tag}.json', c)


def front(b):
    bname, reg, cin = bank_of(b)
    s = json.loads((HERE / 'checks' / f'select3d-{b}.json').read_text())
    subj, seen = [], set()
    for slot in s['slots']:                      # (width, spec, family) with its realised rule; absent slots have none
        if slot['rule']:
            k = (f"W{slot['width']}", slot['rule'], slot['spec'])
            if k not in seen:
                seen.add(k)
                subj.append(dict(setting=k[0], rule=k[1], spec=k[2]))
    for f in FIXED:
        st, rl, *sp = f.split('|')
        k = (st, rl, '|'.join(sp))
        if k not in seen:
            seen.add(k)
            subj.append(dict(setting=k[0], rule=k[1], spec=k[2]))
    c = dict(attempt=f'e2f{b}', driver='e2run3d', mode=f'e2f{b}', cohort_seed=923801, cohort_count=64, trust_fraction=.05,
             timing=dict(burn=.1), settings=settings(bname, WIDTHS), extra_rules=sorted({x['rule'] for x in subj} | {'gl24'}),
             blocks=[dict(kind='panel', label=f'front3d-{b}', A=A_REF, subjects=subj, cases=8, reps=3, seed=20261019)],
             selection=dict(file=f'checks/select3d-{b}.json'), bank_seed=int(b))
    if reg:
        c['bank_registry'] = reg
        c['cluster_inputs'] = cin
    w_(f'configs/e2f{b}.json', c)


if __name__ == '__main__':
    {'sel': sel, 'front': front}[sys.argv[1]](sys.argv[2])
