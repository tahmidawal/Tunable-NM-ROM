"""jcp-dims configs that depend on accuracy results (DESIGN 4, 6.5, A1.1): the 2D timing panel t2 and the 3D screening
(t3s) and confirmation (t3c) panels. Deterministic given the analysis files.

    python make_configs_late.py t2     (needs checks/analysis-2d.json)
    python make_configs_late.py t3s    (needs checks/analysis-3d.json)
    python make_configs_late.py t3c    (needs checks/analysis-3d.json after t3s: --after-screen)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import make_configs as MC
import make_configs3d as M3

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'


def t2():
    an = json.loads((HERE / 'checks' / 'analysis-2d.json').read_text())
    sub = lambda r, M, rule, form, sch, qp=None, qpp=None: dict(mesh=1024, Rp=r, M=M, policy='frozen', rule=rule, form=form,
                                                                scheme=sch, factor=1, qp=qp, qpp=qpp)
    lad = {g: MC.ladder(MC.GAUSS)[i] for i, g in enumerate(MC.GAUSS)}
    S, seen = [], set()

    def add(r, M, rule, form, sch):
        k = (r, M, rule, form, sch)
        if k in seen:
            return
        seen.add(k)
        _, qp, qpp = lad[rule]
        S.append(sub(r, M, rule, form, sch, qp, qpp))
    for r in MC.R_LIST:                                  # every LSPG-CN-R F2 cell, every rung (A3 cost per cell)
        for M in MC.m_set(r):
            for g in MC.GAUSS:
                add(r, M, g, 'LSPG', 'CNR')
    for r in MC.R_LIST:                                  # kappa cells of the other schemes at gauss256 and at their m*
        for k in MC.KAPPAS:
            for form, sch in (('GAL', 'CN'), ('LSPG', 'CN'), ('LSPG', 'BE')):
                add(r, k * r, MC.PRIMARY, form, sch)
                if sch != 'BE':
                    lid = f"K2|R{r}|M{k * r}|{form}|{sch}|1|gauss|s0|frozen"
                    m = an['mstar'].get(lid, {}).get('mstar', {}).get('m')
                    if m:
                        add(r, k * r, f'gauss{int(round(m ** .5))}', form, sch)
    A = sub(384, 1536, 'gauss128', 'LSPG', 'CNR', 'gauss160', 'gauss192')
    d = dict(attempt='t2', mesh=1024, edges=[0, 128, 256, 384, 512], cases='dev6', reps=2, burn=0.1, seed=20261019, A=A,
             subjects=S, doubled=[A], inject_identity_fault=[])
    (CFG / 't2.json').write_text(json.dumps(d, indent=1) + '\n')
    print('t2 subjects', len(S))


def cert_blocks(keys, cases):
    """Single-rung certification blocks (A1.1): every timed configuration is run, certified and anchored in-job first."""
    by = {}
    for k in keys:
        p = k.split('|')
        if p[2] == 'ad':
            by.setdefault(('ad', p[0], p[1], p[3]), set())
        else:
            by.setdefault(('ladder', p[0], p[1]), set()).add('|'.join(p[2:]))
    out = []
    for kk, specs in sorted(by.items()):
        if kk[0] == 'ladder':
            out.append(dict(kind='ladder', setting=kk[1], specs=sorted(specs), rungs=[kk[2]], cases=cases, tighter_cases=list(cases)))
        else:
            out.append(dict(kind='adaptive', setting=kk[1], forms=[f'{kk[3]}|CN'], dt=0.01, rungs=[kk[2]], cases=cases,
                            tighter_cases=list(cases)))
    return out


def settings_for(keys):
    import re
    st = {}
    for k in keys:
        s = k.split('|')[0]
        base = re.match(r'^(M2_512|W\d+)', s).group(1)
        m = re.search(r'k(\d)$', s)
        M = M3.KAPPA_M[int(m.group(1))] if m else (int(s.split('M')[-1]) if re.search(r'M\d+$', s) and s != 'M2_512' else None)
        st[s] = M3.setting(s, base, M)
    return list(st.values())


def t3s():
    an = json.loads((HERE / 'checks' / 'analysis-3d.json').read_text())
    keys = sorted(set(an['qualified_3d']) | set(M3_COMP))
    spec_of = lambda k: '|'.join(k.split('|')[2:]) + ('|prod' if k.split('|')[2] == 'ad' else '')
    subj = [dict(setting=k.split('|')[0], rule=k.split('|')[1], spec=spec_of(k)) for k in keys if k != M3_COMP[0]]
    blocks = cert_blocks(keys, [0, 1, 2, 3])
    blocks.append(dict(kind='panel', label='screen', A=dict(setting='M2_512', rule='gl32', spec='fs|BE|0.01|K1'),
                       subjects=subj + [dict(setting='M2_512', rule='gl32', spec='fs|BE|0.01|K1', double=True)],
                       cases=4, reps=2, seed=20261019))
    rules = sorted({k.split('|')[1] for k in keys})
    M3.job('t3s', settings_for(keys), blocks, rules)
    print('t3s subjects', len(subj))


def t3c():
    an = json.loads((HERE / 'checks' / 'analysis-3d.json').read_text())
    sel = an['h2q']
    keys, blocks = set(), []
    for comp, v in sel.items():
        if v.get('winner'):
            keys |= {v['winner'], v['comparator']}
    cq = an.get('h2q_Cq', {})
    for w, v in cq.items():                              # secondary qualified-vs-qualified pairs (DESIGN 6.5)
        sw = v['result'].get('winner')
        if sw:
            keys |= {sw, v['Cq']}
    blocks = cert_blocks(sorted(keys), list(range(8)))
    for comp, v in sel.items():
        if v.get('winner'):
            c, w = v['comparator'], v['winner']
            sp = lambda k: dict(setting=k.split('|')[0], rule=k.split('|')[1],
                                spec='|'.join(k.split('|')[2:]) + ('|prod' if k.split('|')[2] == 'ad' else ''))
            blocks.append(dict(kind='panel', label=f'confirm|{comp}', A=sp(c), subjects=[sp(w)], cases=8, reps=3, seed=20261020))
    for w, v in cq.items():
        sw = v['result'].get('winner')
        if sw:
            sp = lambda k: dict(setting=k.split('|')[0], rule=k.split('|')[1],
                                spec='|'.join(k.split('|')[2:]) + ('|prod' if k.split('|')[2] == 'ad' else ''))
            blocks.append(dict(kind='panel', label=f'confirm|Cq|{w}', A=sp(v['Cq']), subjects=[sp(sw)], cases=8, reps=3, seed=20261021))
    rules = sorted({k.split('|')[1] for k in keys})
    M3.job('t3c', settings_for(sorted(keys)), blocks, rules)
    print('t3c panels', sum(b['kind'] == 'panel' for b in blocks))


M3_COMP = ['M2_512|gl32|fs|BE|0.01|K1', 'W768|lat32768|fs|BE|0.01|K1', 'W1024|lat65536|fs|BE|0.01|K1']

if __name__ == '__main__':
    {'t2': t2, 't3s': t3s, 't3c': t3c}[sys.argv[1]]()
