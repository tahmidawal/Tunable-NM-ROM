"""Write every jcp-qualify job config (DESIGN 2, 5, 6, 8, A1, A2) and check the two pre-data selections against their
sources: the larger step per scheme (DESIGN 2.2, combined's E0 grid analysis) and the 3D round-1 member list
(DESIGN 2.4, front3d.py on speed3d's analysis-grid.json). A mismatch stops (exit 1).

    python make_configs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMB_E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/checks/analysis-c2grid-2d_e0.json')
EDGES = [0, 128, 256, 384, 512]
RULES = {128: ('fib4181', 'fib6765', 'fib10946'), 256: ('gauss128', 'gauss160', 'gauss192'),
         384: ('gauss128', 'gauss160', 'gauss192'), 512: ('gauss192', 'gauss256', 'gauss320')}
FALLBACK = {128: ('fib6765', 'fib10946', 'fib17711'), 256: ('gauss160', 'gauss192', 'gauss256'),
            384: ('gauss160', 'gauss192', 'gauss256'), 512: ('gauss256', 'gauss320', 'gauss384')}
SCHEMES = (('LSPG', 'BE'), ('LSPG', 'CNR'), ('GAL', 'CN'))
LARGE = {('LSPG', 'BE'): 2.5, ('LSPG', 'CNR'): 2.5, ('GAL', 'CN'): 5.}
DT0 = .005
MESHES = (256, 1024, 4096)
A_SUBJECT = dict(mesh=1024, Rp=384, rule='gauss128', form='LSPG', scheme='CNR', factor=1, qp='gauss160', qpp='gauss192')


def larger_step_check():
    """DESIGN 2.2: largest f in {2.5, 5, 10} with R384 median E0 error <= worst E0 error of R384 LSPG-BE at dt0."""
    d = json.loads(COMB_E0.read_text())['arms']
    bar = d['R384|LSPG|BE|0.005|prod']['worst_ST']
    got = {}
    for form, sc in SCHEMES:
        ok = [f for f in (2.5, 5., 10.) if d[f'R384|{form}|{sc}|{DT0 * f:.10g}|prod']['median_ST'] <= bar]
        got[(form, sc)] = max(ok)
    return got, bar


def inherited(r, form, sc, f, mesh):
    """P1-certified configurations (DESIGN 2.1)."""
    if form != 'LSPG' or f != 1:
        return False
    return (r in (128, 256, 384)) or (r == 512 and mesh == 1024)


def arms_for(r, mesh):
    out = []
    for form, sc in SCHEMES:
        for f in (1, 2, LARGE[(form, sc)]):
            if not inherited(r, form, sc, f, mesh):
                out.append(dict(form=form, scheme=sc, factor=f, cases='all38', q2_cases='sub16'))
    if mesh == 4096 and r in (128, 256):          # 22 extra cases of P1's sub16 configurations (DESIGN 2.1)
        for sc in ('BE', 'CNR'):
            out.append(dict(form='LSPG', scheme=sc, factor=1, cases=[['val32', c] for c in range(10, 32)], q2_cases=[],
                            tighter=False, q3=False, ext22=True))
    return out


def acc_config(mesh, rs=(128, 256, 384, 512)):
    sets = []
    for r in rs:
        q, qp, qpp = RULES[r]
        sets.append(dict(name=f'R{r}', Rp=r, rule=q, qp=qp, qpp=qpp, arms=arms_for(r, mesh)))
    return dict(attempt=f'acc{mesh}', driver='qacc', mesh=mesh, edges=EDGES, settings=sets)


def fallback_config(mesh, failed, attempt):
    """DESIGN 2.3 / A4.1: one fallback configuration (main rule q+, successor q++, check q+++) per failed configuration
    key '<mesh>|R<r>|<form>|<scheme>|<factor>', all 38 cases, full Q1/Q2/Q3 treatment; setting names carry 'fb'."""
    by_r = {}
    for k in failed:
        m, r, form, sc, f = k.split('|')
        assert int(m) == mesh, k
        by_r.setdefault(int(r[1:]), []).append(dict(form=form, scheme=sc, factor=float(f), cases='all38', q2_cases='sub16'))
    sets = [dict(name=f'R{r}fb', Rp=r, rule=FALLBACK[r][0], qp=FALLBACK[r][1], qpp=FALLBACK[r][2], arms=arms)
            for r, arms in sorted(by_r.items())]
    return dict(attempt=attempt, driver='qacc', mesh=mesh, edges=EDGES, settings=sets, fallback_of=sorted(failed))


def subjects(mesh):
    out = []
    for r in (128, 256, 384, 512):
        q, qp, qpp = RULES[r]
        for form, sc in SCHEMES:
            for f in (1, 2, LARGE[(form, sc)]):
                out.append(dict(mesh=mesh, Rp=r, rule=q, form=form, scheme=sc, factor=f, qp=qp, qpp=qpp))
    return out


def time_config(mesh):
    meshes = {str(mesh): dict(edges=EDGES)}
    meshes['1024'] = dict(edges=EDGES)
    return dict(attempt=f'time{mesh}', driver='qtime', meshes=meshes, cases='dev6', reps=3, burn=.1, seed=20261018,
                A=A_SUBJECT, subjects=subjects(mesh),
                doubled=[dict(A_SUBJECT, cases='dev6')] if mesh == 1024 else [])


def order_config():
    arms = [dict(form=form, scheme=sc, factors=[LARGE[(form, sc)], 2, 1, .5, .25]) for form, sc in SCHEMES]
    blk = lambda st, a: dict(setting=st, kind='order', cases='dev6', finest=.25, anchor=1 / 64, anchor2=1 / 128, arms=a)
    th = [dict(form='GAL', scheme='TH06', factors=[2, 1, .5, .25])]
    return dict(attempt='order', driver='qacc', mesh=1024, edges=EDGES,
                settings=[dict(name='R128', Rp=128, rule=RULES[128][0], qp=RULES[128][1], qpp=RULES[128][2], arms=[]),
                          dict(name='R384', Rp=384, rule=RULES[384][0], qp=RULES[384][1], qpp=RULES[384][2], arms=[])],
                blocks=[blk('R128', arms), blk('R384', arms + th)])


def smoke_acc():
    """smkA (DESIGN 8, A2.3): dev2 path tests of every (form, scheme, factor) at R128 and R512, the designated positive
    GAL fixture (R384 GAL-CN dt0 dev6), the Delta-t0 reproduction gate vs P1 (R384 BE/CNR dev6), combined-archive gates
    (R128, R512 dev6), every numerical control, and a one-case B4 path test incl. TH06."""
    sets = []
    for r in (128, 512):
        q, qp, qpp = RULES[r]
        arms = [dict(form=form, scheme=sc, factor=f, cases='dev6', q2_cases='dev6')
                for form, sc in SCHEMES for f in (1, 2, LARGE[(form, sc)])]
        sets.append(dict(name=f'R{r}', Rp=r, rule=q, qp=qp, qpp=qpp, arms=arms))
    q, qp, qpp = RULES[384]
    arms = [dict(form='GAL', scheme='CN', factor=1, cases='dev6', q2_cases='dev6'),
            dict(form='LSPG', scheme='BE', factor=1, cases='dev6', q2_cases='dev6'),
            dict(form='LSPG', scheme='CNR', factor=1, cases='dev6', q2_cases='dev6')]
    sets.append(dict(name='R384', Rp=384, rule=q, qp=qp, qpp=qpp, arms=arms))
    ob = dict(setting='R384', kind='order', cases=[['dev6', 0]], finest=.25, anchor=1 / 64, anchor2=1 / 128,
              arms=[dict(form=form, scheme=sc, factors=[LARGE[(form, sc)], 2, 1, .5, .25]) for form, sc in SCHEMES] +
                   [dict(form='GAL', scheme='TH06', factors=[2, 1, .5, .25], cases='dev6')])   # A3.4: full TH06 control
    cb = dict(setting='R384', kind='controls', cases='dev6', under_resolved='gauss32', under_resolved_successor='gauss64',
              b0_cases='dev6')
    return dict(attempt='smkA', driver='qacc', mesh=1024, edges=EDGES, settings=sets, blocks=[ob, cb])


def smoke_time():
    subs = []
    for r, (form, sc, f) in ((128, ('LSPG', 'BE', 2.5)), (128, ('GAL', 'CN', 5.)), (384, ('LSPG', 'CNR', 2)),
                             (512, ('LSPG', 'BE', 1)), (512, ('LSPG', 'CNR', 2.5)), (512, ('GAL', 'CN', 1))):
        q, qp, qpp = RULES[r]
        subs.append(dict(mesh=1024, Rp=r, rule=q, form=form, scheme=sc, factor=f, qp=qp, qpp=qpp))
    return dict(attempt='smkB', driver='qtime', meshes={'1024': dict(edges=EDGES)}, cases='dev2', reps=2, burn=.1,
                seed=20261018, A=A_SUBJECT, subjects=subs, doubled=[dict(A_SUBJECT, cases='dev2')],
                inject_identity_fault=['1024|R128|fib4181|LSPG|BE|2.5'])


def pilot_config():
    """smkB step 2 (A2.4): the time4096 + acc4096 footprint incl. every R512 rule (main, q+, q++, fallback q+++)."""
    return dict(attempt='pilot', driver='qpilot', mesh=4096, edges=EDGES, also_bank=1024,
                settings=[dict(Rp=r, rules=list(RULES[r]) + list(FALLBACK[r][2:])) for r in (128, 256, 384, 512)])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--fallback':          # python make_configs.py --fallback <mesh> '<json keys>'
        mesh, keys = int(sys.argv[2]), json.loads(sys.argv[3])
        att = f'fb{mesh}'
        (HERE / 'configs' / f'{att}.json').write_text(json.dumps(fallback_config(mesh, keys, att), indent=1) + '\n')
        print('wrote', att)
        return
    got, bar = larger_step_check()
    want = dict(LARGE)
    print('larger step', {f'{a}-{b}': v for (a, b), v in got.items()}, 'bar', bar)
    if got != want:
        print('MISMATCH with DESIGN 2.2', got, want)
        sys.exit(1)
    import front3d
    members = front3d.round1()
    if [m['key'] for m in members] != front3d.DESIGN_ROUND1:
        print('MISMATCH with DESIGN 2.4', [m['key'] for m in members])
        sys.exit(1)
    cdir = HERE / 'configs'
    cdir.mkdir(exist_ok=True)
    cfgs = {f'acc{m}': acc_config(m) for m in MESHES}
    cfgs.update({f'time{m}': time_config(m) for m in MESHES})
    cfgs['order'] = order_config()
    cfgs['smkA'] = smoke_acc()
    cfgs['smkB'] = smoke_time()
    cfgs['pilot'] = pilot_config()
    cfgs['q3d'] = front3d.q3d_config(members, 'q3d')
    cfgs['smk3'] = front3d.smoke_config()
    for k, v in cfgs.items():
        (cdir / f'{k}.json').write_text(json.dumps(v, indent=1) + '\n')
    print('wrote', sorted(cfgs))


if __name__ == '__main__':
    main()
