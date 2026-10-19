"""jcp-robust: E2 2D configs (DESIGN 3.2, A1.8, A2.5, A2.6, A3.5, A3.6, A4.2).

  selection   python make_configs_e2.py sel <bank>          -> configs/e2s<bank>.json (1024^2, 38 cases, r = 128/256/384,
                                                              the registered candidate list with successors, target gauss384,
                                                              check gauss320 on dev6, LSPG-BE dt0 and LSPG-CN-R dt0)
  verdicts    python make_configs_e2.py ver <bank>          -> configs/e2v<bank>{a,b,c}.json from checks/select2d-<bank>.json
                                                              a: 256^2 (38), b: 4096^2 (P1 16-case subset), c: 1024^2 factors 2, 2.5
  timing      python make_configs_e2.py time <bank>         -> configs/e2t<bank>.json (dtime, 1024^2, dev6 x 3 reps, A = R384
                                                              gauss128 frozen LSPG-CN-R dt0 on the job's bank; slots (r, scheme,
                                                              factor) at the selected rule + deployed rows when they differ)
<bank> is 0 (incumbent dn256b) or a seed 1..5 whose files are inputs/banks/s<seed>/{ckpt.pkl, rotation_R512.npz, BANK.json}.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGES = [0, 128, 256, 384, 512]
RANKS = (128, 256, 384)
CANDIDATES = [('fib4181', 'fib6765'), ('fib6765', 'fib10946'), ('gauss96', 'gauss128'), ('fib10946', 'fib17711'),
              ('gauss128', 'gauss160'), ('gauss160', 'gauss192'), ('gauss192', 'gauss256')]     # ordered by m (A1.8)
DEPLOYED = {128: 'fib4181', 256: 'gauss128', 384: 'gauss128'}
SUCC = dict(CANDIDATES)
SEL_ARMS = [dict(form='LSPG', scheme='BE', factor=1), dict(form='LSPG', scheme='CNR', factor=1)]
FRONT_ARMS = [dict(form='LSPG', scheme='BE', factor=2), dict(form='LSPG', scheme='BE', factor=2.5),
              dict(form='LSPG', scheme='CNR', factor=2), dict(form='LSPG', scheme='CNR', factor=2.5)]
SUB16 = [['dev6', c] for c in range(6)] + [['val32', c] for c in range(10)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bank_cfg(b):
    if int(b) == 0:
        return None
    d = HERE / 'inputs' / 'banks' / f's{b}'
    meta = json.loads((d / 'BANK.json').read_text())
    ck, rot = f'inputs/banks/s{b}/ckpt.pkl', f'inputs/banks/s{b}/rotation_R512.npz'
    assert sha(HERE / ck) == meta['ckpt_sha256'] and sha(HERE / rot) == meta['rotation_sha256'], 'bank files differ from BANK.json'
    return dict(ckpt=ck, ckpt_sha256=meta['ckpt_sha256'], rotation=rot, rotation_sha256=meta['rotation_sha256'], seed=int(b))


def w(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=1) + '\n')
    print(name)


def sel(b):
    groups = []
    for r in RANKS:
        groups.append(dict(name=f'R{r}', Rp=r, target='gauss384', check='gauss320', check_cases='dev6', tighter_cases='dev6',
                           arms=SEL_ARMS, rules=[dict(label=c, rule=c, succ=s, level='full') for c, s in CANDIDATES]))
    c = dict(attempt=f'e2s{b}', driver='r2acc', mesh=1024, edges=EDGES, cases='all38', groups=groups)
    bc = bank_cfg(b)
    if bc:
        c['bank'] = bc
    w(f'configs/e2s{b}.json', c)


def ver(b):
    s = json.loads((HERE / 'checks' / f'select2d-{b}.json').read_text())
    bc = bank_cfg(b)

    def rules_for(r):
        out = []
        pick = s['selected'][str(r)]['rule']          # None = "none within ladder" (A1.8): no primary row
        if pick:
            out.append(dict(label='sel', rule=pick, succ=SUCC[pick], level='full'))
        if pick != DEPLOYED[r]:                       # A2.6: the fixed deployed configuration is also run
            out.append(dict(label='dep', rule=DEPLOYED[r], succ=SUCC[DEPLOYED[r]], level='full'))
        return out

    for tag, mesh, cases, arms in (('a', 256, 'all38', SEL_ARMS), ('b', 4096, SUB16, SEL_ARMS), ('c', 1024, 'all38', FRONT_ARMS)):
        groups = [dict(name=f'R{r}', Rp=r, target=None, check=None, tighter_cases='dev6', arms=arms, rules=rules_for(r))
                  for r in RANKS if rules_for(r)]
        c = dict(attempt=f'e2v{b}{tag}', driver='r2acc', mesh=mesh, edges=EDGES, cases=cases, groups=groups,
                 selection=dict(file=f'checks/select2d-{b}.json', sha256=sha(HERE / 'checks' / f'select2d-{b}.json')))
        if bc:
            c['bank'] = bc
        w(f'configs/e2v{b}{tag}.json', c)


def time_(b):
    s = json.loads((HERE / 'checks' / f'select2d-{b}.json').read_text())
    bc = bank_cfg(b)
    sp = lambda r, rule, arm: dict(mesh=1024, Rp=r, M=4 * r, policy='frozen', rule=rule, form='LSPG', scheme=arm['scheme'],
                                   factor=arm['factor'], qp=SUCC[rule], qpp=SUCC.get(SUCC[rule], 'gauss384'))
    subj = []
    for r in RANKS:
        pick = s['selected'][str(r)]['rule']
        for arm in SEL_ARMS + FRONT_ARMS:
            if pick:
                subj.append(dict(sp(r, pick, arm), slot=f'R{r}|{arm["scheme"]}|{arm["factor"]:g}', role='selected'))
            if pick != DEPLOYED[r]:
                subj.append(dict(sp(r, DEPLOYED[r], arm), slot=f'R{r}|{arm["scheme"]}|{arm["factor"]:g}', role='deployed'))
    A = sp(384, 'gauss128', SEL_ARMS[1])
    c = dict(attempt=f'e2t{b}', driver='dtime', mesh=1024, edges=EDGES, cases='dev6', reps=3, burn=.1, seed=20261019, A=A,
             subjects=subj, doubled=[A],
             selection=dict(file=f'checks/select2d-{b}.json', sha256=sha(HERE / 'checks' / f'select2d-{b}.json')))
    if bc:
        c['bank'] = bc
    w(f'configs/e2t{b}.json', c)


if __name__ == '__main__':
    {'sel': sel, 'ver': ver, 'time': time_}[sys.argv[1]](sys.argv[2])
