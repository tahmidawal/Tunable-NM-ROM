"""jcp-dims job configurations and the cell manifest (DESIGN 2, A1.6, A1.7). Deterministic; writes configs/*.json and
configs/MANIFEST-cells.json and asserts the registered counts.

    python make_configs.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'

R_LIST = (128, 256, 384, 512)
KAPPAS = (1, 2, 3, 4, 8)
GAUSS = [f'gauss{p}' for p in (64, 96, 128, 160, 192, 256, 320, 384)]
GAUSS_CAND = GAUSS[:6]                       # gauss64 ... gauss256 (each has q+ and q++ on the ladder)
FIB = [2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393]
POW = [2 ** k for k in range(11, 18)]
SMOL = list(range(8, 15))
DRAWS = range(5)
PRIMARY = 'gauss256'                         # B6 primary rung (fixed m), A3 fixed-m / fixed-(M, m) rung
FAMILY_CELLS = ((128, 512), (384, 1536))
BAD2D = 'badlat121393'


def m_set(r):
    s = {k * r for k in KAPPAS} | {4 * rr for rr in R_LIST if 4 * rr >= r} | {2048}
    return sorted(s)


def kappa_of(r, M):
    return M // r if M % r == 0 and M // r in KAPPAS else None


def kappa8_only(r, M):
    """Cells that exist only as kappa = 8 (deferral unit, A1.6)."""
    return kappa_of(r, M) == 8 and M not in {4 * rr for rr in R_LIST} and M != 2048


def draw(fam, n, s):
    return f'{fam}{n}' if s == 0 else f'{fam}{n}s{s}'


def ladder(names):
    """[(rule, q+, q++)] for an ordered ladder."""
    return [(names[i], names[i + 1] if i + 1 < len(names) else None, names[i + 2] if i + 2 < len(names) else None)
            for i in range(len(names))]


def arm(form, scheme, factor=1, rho=True, q3=False, cases='all38'):
    a = dict(form=form, scheme=scheme, factor=factor, cases=cases, tighter=True, q3=q3)
    if rho:
        a['rho_only'] = True
    return a


def setting(r, M, rule, qp, arms, policy='frozen', tag=''):
    s = dict(name=f'R{r}M{M}{tag}{rule}', Rp=r, M=M, policy=policy, rule=rule, arms=arms)
    if qp:
        s['qp'] = qp
    else:
        for a in arms:
            a.pop('rho_only', None)
    return s


def cell_settings(r, M, rules, mk_arms, policy='frozen', tag=''):
    out = []
    for rule, qp, _ in ladder(rules):
        arms = mk_arms(rule)
        if arms:
            out.append(setting(r, M, rule, qp, arms, policy, tag))
    return out


def job(name, settings, mesh=1024, blocks=(), **extra):
    d = dict(attempt=name, mesh=mesh, edges=[0, 128, 256, 384, 512], settings=settings, blocks=list(blocks), **extra)
    (CFG / f'{name}.json').write_text(json.dumps(d, indent=1) + '\n')
    return d


def main():
    CFG.mkdir(exist_ok=True)
    manifest = dict(cells=[], ladders=[], counts={})
    cells = [(r, M) for r in R_LIST for M in m_set(r)]
    assert len(cells) == 26, len(cells)
    # ---------------------------------------------------------------- F2: LSPG-CN-R Gauss ladders (f2a, f2b)
    f2a, f2b = [], []
    for r, M in cells:
        st = cell_settings(r, M, GAUSS, lambda rule: [arm('LSPG', 'CNR', q3=(rule == PRIMARY and kappa_of(r, M) is not None))])
        (f2b if r == 512 else f2a).extend(st)
        manifest['ladders'].append(dict(kind='F2', r=r, M=M, kappa=kappa_of(r, M), scheme='LSPG|CNR|1', family='gauss',
                                        draw=0, rungs=GAUSS, candidates=GAUSS_CAND, deferrable_kappa8=kappa8_only(r, M)))
    n_cnr = sum(len(s['arms']) for s in f2a + f2b)
    assert n_cnr == 208, n_cnr
    # ---------------------------------------------------------------- kappa cells: GAL-CN, LSPG-CN, LSPG-BE (f2k)
    f2k, n_gal, n_cn, n_be = [], 0, 0, 0
    for r in R_LIST:
        for k in KAPPAS:
            M = k * r
            def mk(rule):
                a = [arm('GAL', 'CN', q3=(rule == PRIMARY)), arm('LSPG', 'CN', q3=(rule == PRIMARY))]
                if rule in ('gauss256', 'gauss320', 'gauss384'):
                    a.append(arm('LSPG', 'BE', q3=(rule == PRIMARY)))
                return a
            st = cell_settings(r, M, GAUSS, mk, tag='k')
            for s in st:
                n_gal += sum(a['form'] == 'GAL' for a in s['arms'])
                n_cn += sum(a['form'] == 'LSPG' and a['scheme'] == 'CN' for a in s['arms'])
                n_be += sum(a['scheme'] == 'BE' for a in s['arms'])
            f2k.extend(st)
            for sch in ('GAL|CN|1', 'LSPG|CN|1'):
                manifest['ladders'].append(dict(kind='K2', r=r, M=M, kappa=k, scheme=sch, family='gauss', draw=0,
                                                rungs=GAUSS, candidates=GAUSS_CAND, deferrable_kappa8=(k == 8)))
            manifest['ladders'].append(dict(kind='K2', r=r, M=M, kappa=k, scheme='LSPG|BE|1', family='gauss', draw=0,
                                            rungs=GAUSS[5:], candidates=[], deferrable_kappa8=(k == 8),
                                            note='Q labels at gauss256 only; no m* (A1.7)'))
    assert (n_gal, n_cn, n_be) == (160, 160, 60), (n_gal, n_cn, n_be)
    # ---------------------------------------------------------------- B6-LMF (deferrable, its own job f2l)
    lmf = cell_settings(384, 1536, GAUSS, lambda rule: [arm('LSPG', 'CNR', q3=(rule == PRIMARY))], policy='lmf', tag='lmf')
    lmf += cell_settings(384, 1536, [f'smolyak{l}' for l in SMOL], lambda rule: [arm('LSPG', 'CNR')], policy='lmf', tag='lmf')
    for fam, rungs in (('gauss', GAUSS), ('smolyak', [f'smolyak{l}' for l in SMOL])):
        manifest['ladders'].append(dict(kind='LMF', r=384, M=1536, scheme='LSPG|CNR|1', family=fam, draw=0, policy='lmf',
                                        rungs=rungs, candidates=rungs[:-2]))
    # ---------------------------------------------------------------- B3 families stage (i) (b2f)
    b2f = []
    for r, M in FAMILY_CELLS:
        fams = [('fib', [draw('fib', n, s) for n in FIB], s) for s in DRAWS]
        fams += [('sobol', [draw('sobol', n, s) for n in POW], s) for s in DRAWS]
        fams += [('halton', [draw('halton', n, s) for n in POW], s) for s in DRAWS]
        fams += [('smolyak', [f'smolyak{l}' for l in SMOL], 0)]
        for fam, rungs, s in fams:
            b2f += cell_settings(r, M, rungs, lambda rule: [arm('LSPG', 'CNR')], tag='f')
            manifest['ladders'].append(dict(kind='B3', r=r, M=M, scheme='LSPG|CNR|1', family=fam, draw=s, rungs=rungs,
                                            candidates=rungs[:-2]))
        b2f.append(setting(r, M, BAD2D, None, [arm('LSPG', 'CNR', rho=False)], tag='f'))
    fixgate = [dict(kind='fixgate', r=r, M=M, target='gauss384',
                    rules=[BAD2D, 'gauss384', 'fib121393'] + [f'fib121393s{s}' for s in range(1, 5)]
                    + [draw('sobol', POW[-1], s) for s in DRAWS] + [draw('halton', POW[-1], s) for s in DRAWS]
                    + ['smolyak14'], fixture='inputs/fixture2d.npz') for r, M in FAMILY_CELLS]
    # ---------------------------------------------------------------- stage (ii): MC (b2m)
    b2m = []
    for r, M in FAMILY_CELLS:
        for s in DRAWS:
            rungs = [f'mc{n}s{s}' for n in POW]
            b2m += cell_settings(r, M, rungs, lambda rule: [arm('LSPG', 'CNR')], tag='f')
            manifest['ladders'].append(dict(kind='B3ii', r=r, M=M, scheme='LSPG|CNR|1', family='mc', draw=s, rungs=rungs,
                                            candidates=rungs[:-2]))
    # ---------------------------------------------------------------- write
    job('f2a', f2a)
    job('f2b', f2b)
    job('f2k', f2k)
    job('f2l', lmf)
    job('b2f', b2f, blocks=fixgate)
    job('b2m', b2m, blocks=[dict(fb, rules=[f'mc{POW[-1]}s{s}' for s in DRAWS] + ['gauss384']) for fb in fixgate])
    for s in f2a + f2b + f2k + lmf + b2f + b2m:
        for a in s['arms']:
            manifest['cells'].append(dict(setting=s['name'], r=s['Rp'], M=s['M'], policy=s['policy'], rule=s['rule'],
                                          qp=s.get('qp'), form=a['form'], scheme=a['scheme'], factor=a['factor']))
    manifest['counts'] = dict(cells_2d=len(cells), lspg_cnr=n_cnr, gal_cn=n_gal, lspg_cn=n_cn, lspg_be=n_be,
                              settings=dict(f2a=len(f2a), f2b=len(f2b), f2k=len(f2k), f2l=len(lmf), b2f=len(b2f), b2m=len(b2m)))
    (CFG / 'MANIFEST-cells.json').write_text(json.dumps(manifest, indent=1) + '\n')
    smoke()
    print(json.dumps(manifest['counts']))
    for f in sorted(CFG.glob('*.json')):
        print(f.name, hashlib.sha256(f.read_bytes()).hexdigest()[:12])


def smoke():
    """smk2 (dacc) + smk2t (dtime): real-size 1024^2 smoke (DESIGN 7, 8, A1.4)."""
    dev2, dev6 = 'dev2', 'dev6'
    S_ = []
    S_.append(dict(name='R384M1536gauss128', Rp=384, M=1536, policy='frozen', rule='gauss128', qp='gauss160',
                   arms=[arm('LSPG', 'CNR', cases=dev6), arm('LSPG', 'BE', cases=dev6), arm('GAL', 'CN', cases=dev6)]))
    S_.append(dict(name='R128M512fib4181', Rp=128, M=512, policy='frozen', rule='fib4181', qp='fib6765',
                   arms=[arm('LSPG', 'CNR', cases=dev6), arm('LSPG', 'BE', cases=dev6)]))
    S_.append(dict(name='R512M4096gauss256', Rp=512, M=4096, policy='frozen', rule='gauss256', qp='gauss320',
                   arms=[arm('LSPG', 'CNR', cases=dev2, q3=True), arm('GAL', 'CN', cases=dev2)]))
    S_.append(dict(name='R512M4096gauss320', Rp=512, M=4096, policy='frozen', rule='gauss320', qp='gauss384',
                   arms=[arm('LSPG', 'CNR', cases=dev2), arm('GAL', 'CN', cases=dev2), arm('LSPG', 'CN', cases=dev2)]))
    S_.append(dict(name='R512M4096gauss384', Rp=512, M=4096, policy='frozen', rule='gauss384',
                   arms=[arm('LSPG', 'CNR', rho=False, cases=dev2), arm('GAL', 'CN', rho=False, cases=dev2)]))
    S_.append(dict(name='R128M128gauss64', Rp=128, M=128, policy='frozen', rule='gauss64', qp='gauss96',
                   arms=[arm('LSPG', 'CNR', cases=dev2, q3=True), arm('LSPG', 'CN', cases=dev2), arm('GAL', 'CN', cases=dev2)]))
    S_.append(dict(name='R384M384gauss256', Rp=384, M=384, policy='frozen', rule='gauss256', qp='gauss320',
                   arms=[arm('LSPG', 'CNR', cases=dev2), arm('GAL', 'CN', cases=dev2)]))
    fam_ladders = [[draw('fib', n, s) for n in FIB[:3]] for s in (1, 2)] + [[draw('sobol', n, s) for n in POW[:3]] for s in (1, 2)]
    fam_ladders += [[draw('halton', n, s) for n in POW[:3]] for s in (1, 2)] + [[f'mc{n}s{s}' for n in POW[:3]] for s in (0, 1)]
    fam_ladders += [[f'smolyak{l}' for l in SMOL[:3]]]
    for rungs in fam_ladders:                 # two draws x two production rungs (+ the successor for rho) per family
        for rule, qp, _ in ladder(rungs)[:2]:
            S_.append(setting(128, 512, rule, qp, [arm('LSPG', 'CNR', cases=dev2)], tag='f'))
    S_.append(setting(128, 512, 'badlat2584', None, [arm('LSPG', 'CNR', cases=dev2)], tag='f'))
    S_.append(dict(name='R384M1536lmfgauss256', Rp=384, M=1536, policy='lmf', rule='gauss256', qp='gauss320',
                   arms=[arm('LSPG', 'CNR', cases=dev2, q3=True)]))
    S_.append(dict(name='R384M1536gauss256', Rp=384, M=1536, policy='frozen', rule='gauss256', qp='gauss320',
                   arms=[arm('LSPG', 'CNR', cases=dev2)]))
    blocks = [dict(setting='R384M1536gauss128', kind='controls', cases='dev6', under_resolved='gauss32',
                   under_resolved_successor='gauss64', b0_cases='dev6'),
              dict(kind='fixgate', r=384, M=1536, target='gauss384', rules=[BAD2D, 'gauss384', 'fib121393'],
                   fixture='inputs/fixture2d.npz'),
              dict(kind='fixgate', r=128, M=512, target='gauss384', rules=[BAD2D], fixture='inputs/fixture2d.npz')]
    job('smk2', S_, blocks=blocks)
    sub = lambda r, M, rule, form, sch, qp=None, qpp=None, pol='frozen': dict(mesh=1024, Rp=r, M=M, policy=pol, rule=rule,
                                                                             form=form, scheme=sch, factor=1, qp=qp, qpp=qpp)
    A = sub(384, 1536, 'gauss128', 'LSPG', 'CNR', 'gauss160', 'gauss192')
    subjects = [sub(128, 512, 'gauss96', 'LSPG', 'CNR', 'gauss128', 'gauss160'),
                sub(128, 512, 'gauss96', 'GAL', 'CN', 'gauss128', 'gauss160'),
                sub(128, 128, 'gauss96', 'LSPG', 'CNR', 'gauss128', 'gauss160'),
                sub(512, 4096, 'gauss256', 'LSPG', 'CNR', 'gauss320', 'gauss384'),
                sub(512, 4096, 'gauss256', 'GAL', 'CN', 'gauss320', 'gauss384'),
                sub(512, 4096, 'gauss256', 'LSPG', 'CN', 'gauss320', 'gauss384'),
                sub(384, 1536, 'gauss128', 'LSPG', 'CNR', 'gauss160', 'gauss192')]
    d = dict(attempt='smk2t', mesh=1024, edges=[0, 128, 256, 384, 512], cases=dev2, reps=1, burn=0.1, seed=20261018, A=A,
             subjects=subjects, doubled=[subjects[-1]], inject_identity_fault=['1024|R128|M128|gauss96|LSPG|CNR|1'])
    (CFG / 'smk2t.json').write_text(json.dumps(d, indent=1) + '\n')


if __name__ == '__main__':
    main()
