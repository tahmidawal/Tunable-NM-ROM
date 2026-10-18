"""Write every job config of jcp-p1-scaling (DESIGN 2, 7 and amendments) into configs/. Deterministic.

    python make_configs.py [--prefix-4096]      (--prefix-4096: the 4096^2 bank uses the prefix-retaining edges, A9.2)
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = [0, 128, 256, 384, 512]
PREFIX = [0, 128, 256, 384]
RULE = {128: 'fib4181', 256: 'gauss128', 384: 'gauss128', 512: 'gauss192'}
QP = {'fib4181': 'fib6765', 'gauss128': 'gauss160', 'gauss192': 'gauss256', 'gauss96': 'gauss128', 'gauss32': 'gauss64',
      'gauss64': 'gauss96'}
QPP = {'fib4181': 'fib10946', 'gauss128': 'gauss192', 'gauss192': 'gauss320', 'gauss96': 'gauss160'}
HIST = {'H96': 'gauss96', 'H64': 'gauss64', 'HF6765': 'fib6765', 'H640': 'gauss640'}     # P1g historical rules (A1.8)
FIX = json.loads((HERE / 'inputs' / 'fixtures.json').read_text())


def p1d(name, r, cases, policy='frozen', schemes=('BE', 'CNR')):
    rl = RULE[r]
    return dict(name=name, Rp=r, rule=rl, policy=policy, schemes=list(schemes), cases=cases, q2_cases='sub16',
                qp=QP[rl], qpp=QPP[rl])


def hist(name, rule, policy, cases='all38'):
    return dict(name=name, Rp=384, rule=rule, policy=policy, schemes=['BE'], cases=cases)   # Q1 only (A1.8, A2.5)


def acc(mesh, edges):
    c = []
    if mesh == 4096:
        c += [p1d('R128', 128, 'sub16'), p1d('R256', 256, 'sub16'), p1d('R384', 384, 'all38'),
              p1d('R384nom', 384, 'all38', 'nominal')]
    else:
        c += [p1d('R128', 128, 'all38'), p1d('R256', 256, 'all38'), p1d('R384', 384, 'all38')]
        if mesh == 256:
            c.append(p1d('R384nom', 384, 'all38', 'nominal'))
        if mesh == 1024:
            c.append(p1d('R512', 512, 'all38'))
    for nm, rl in HIST.items():
        c.append(hist(nm, rl, 'frozen'))
        if mesh != 1024:
            c.append(hist(nm + 'nom', rl, 'nominal'))
    blocks = []
    if mesh == 1024:
        blocks = [dict(config='R256', kind='p1e', failing='gauss64', successor='gauss96', cases='dev6', scheme='BE'),
                  dict(config='R384', kind='order', case=['dev6', 0], factors=[1, .5, .25], anchor=.0625, anchor2=.03125,
                       qplus='gauss160', qplusplus='gauss192'),
                  dict(config='R384', kind='vendor', cases='dev6',
                       rules=['gauss128', 'gauss32', 'gauss64', 'gauss96', 'fib6765', 'gauss640'])]
    return dict(attempt=f'acc{mesh}', driver='p1acc', mesh=mesh, edges=edges, configs=c, blocks=blocks)


def subj(mesh, r, scheme, policy='frozen'):
    rl = RULE[r]
    return dict(mesh=mesh, Rp=r, rule=rl, policy=policy, scheme=scheme, qp=QP[rl], qpp=QPP[rl])


def timing(edges4096, smoke=False):
    if smoke:
        meshes = {'256': dict(edges=FULL), '1024': dict(edges=FULL)}
        subs = [subj(256, 128, 'CNR'), subj(1024, 128, 'CNR'), subj(256, 384, 'CNR'), subj(1024, 384, 'CNR'),
                subj(1024, 512, 'CNR'), subj(256, 128, 'BE')]
        cfg = dict(attempt='smktime', driver='p1time', meshes=meshes, A=subj(1024, 384, 'CNR'), subjects=subs, cases='dev2',
                   reps=2, burn=.1, seed=20261017, inject_identity_fault=['256|R128|fib4181|frozen|BE'],
                   doubled=[dict(subj(1024, 384, 'CNR'), cases='dev2'), dict(subj(256, 128, 'CNR'), cases='dev2')],
                   components=[dict(subj(1024, 384, 'CNR'), case=['dev6', 0], reps=30),
                               dict(subj(1024, 512, 'CNR'), case=['dev6', 0], reps=30)],
                   recording_overhead=[dict(subj(1024, 128, 'CNR'), cases='dev2'), dict(subj(1024, 384, 'CNR'), cases='dev2')])
        return cfg
    meshes = {'256': dict(edges=FULL), '1024': dict(edges=FULL), '4096': dict(edges=edges4096)}
    subs = [subj(m, r, sc) for r in (128, 256, 384) for sc in ('BE', 'CNR') for m in (256, 1024, 4096)]
    subs += [subj(1024, 512, sc) for sc in ('BE', 'CNR')]
    subs += [subj(m, 384, sc, 'nominal') for m in (256, 4096) for sc in ('BE', 'CNR')]
    return dict(attempt='time', driver='p1time', meshes=meshes, A=subj(1024, 384, 'CNR'), subjects=subs, cases='dev6',
                reps=3, burn=.1, seed=20261017,
                doubled=[dict(subj(1024, 384, 'CNR'), cases='dev6'), dict(subj(256, 128, 'BE'), cases='dev6')],
                components=[dict(subj(1024, 384, 'CNR'), case=['dev6', 0], reps=30),
                            dict(subj(1024, 512, 'CNR'), case=['dev6', 0], reps=30)])


def kern2d(smoke=False):
    return dict(attempt='smkkern' if smoke else 'kern2d', driver='p1kern', mesh=1024, edges=FULL,
                fixtures=dict(file='inputs/fixtures_p1a.npz', sha256=FIX['p1a']['file_sha256']),
                ranks=[32, 128, 384, 512] if smoke else [32, 64, 128, 256, 384, 512],
                panels=['fixed'] if smoke else ['fixed', 'fixedm', 'nominal'], fixed_M=1536, fixed_rule='gauss96',
                nominal_rules={str(k): v for k, v in RULE.items()}, nu=.01, reps=30, burn=.1, seed=20261017)


def kern3d(smoke=False):
    return dict(attempt='smkkern3d' if smoke else 'kern3d', driver='p1three', mode='kern',
                fixtures=dict(file='inputs/fixtures_p1b.npz', sha256=FIX['p1b']['file_sha256']),
                ranks=[64, 1024] if smoke else [64, 128, 256, 512, 768, 1024], fixed_r=512,
                panels=['fixed'] if smoke else ['fixed', 'nominal'], fixed_rule='gl24',
                nominal_rules={'64': 'gl24', '128': 'gl24', '256': 'gl24', '512': 'lat32768', '768': 'lat32768',
                               '1024': 'lat65536'}, nu=.01, reps=30, burn=.1, seed=20261017)


def p1e3d():
    return dict(attempt='p1e3d', driver='p1three', mode='p1e', ranks=[512], cohort_seed=923801, cohort_count=64,
                cases=[0, 1, 2, 3, 4, 5], Rp=512, a=6, gtol=1e-3, trust_fraction=.05,
                chain=['lat8192', 'lat32768', 'gl56', 'gl64'])


def smoke_acc():
    c = [dict(p1d('R128', 128, 'dev2'), q2_cases='dev2'), dict(p1d('R384', 384, 'dev2'), q2_cases='dev2'),
         dict(p1d('R512', 512, 'dev2', schemes=('BE',)), q2_cases='dev2'),
         dict(hist('H640', 'gauss640', 'frozen', 'dev2'))]
    blocks = [dict(config='R384', kind='controls', cases='dev6', ur_scheme='BE', under_resolved='gauss32',
                   under_resolved_successor='gauss64', b0_scheme='CNR', b0_cases='dev6'),
              dict(config='R384', kind='order', case=['dev6', 0], factors=[1, .5, .25], anchor=.0625, anchor2=.03125,
                   qplus='gauss160', qplusplus='gauss192'),
              dict(config='R384', kind='vendor', cases='dev2', rules=['gauss128', 'gauss32', 'gauss64', 'gauss96', 'fib6765', 'gauss640']),
              dict(config='R128', kind='p1e', failing='gauss32', successor='gauss64', cases='dev2', scheme='BE')]
    return dict(attempt='smkacc', driver='p1acc', mesh=1024, edges=FULL, configs=c, blocks=blocks)


def pilot(edges4096):
    return dict(attempt='pilot', driver='pilot', meshes={'256': dict(edges=FULL), '1024': dict(edges=FULL),
                                                         '4096': dict(edges=edges4096)}, pilot_mesh=4096,
                settings=[dict(Rp=128, rule='fib4181', roles=dict(qp='fib6765')),
                          dict(Rp=256, rule='gauss128', roles=dict(qp='gauss160')),
                          dict(Rp=384, rule='gauss128', roles=dict(qp='gauss160', qpp='gauss192', hist='gauss640'))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix-4096', action='store_true')
    a = ap.parse_args()
    e4 = PREFIX if a.prefix_4096 else FULL
    out = HERE / 'configs'
    out.mkdir(exist_ok=True)
    cfgs = dict(smkacc=smoke_acc(), smktime=timing(e4, smoke=True), smkkern=kern2d(True), smkkern3d=kern3d(True),
                pilot=pilot(FULL), acc256=acc(256, FULL), acc1024=acc(1024, FULL), acc4096=acc(4096, e4),
                time=timing(e4), kern2d=kern2d(), kern3d=kern3d(), p1e3d=p1e3d())
    for k, v in cfgs.items():
        (out / f'{k}.json').write_text(json.dumps(v, indent=1) + '\n')
    man = {k: hashlib.sha256((out / f'{k}.json').read_bytes()).hexdigest() for k in cfgs}
    (out / 'CONFIGS.sha256.json').write_text(json.dumps(man, indent=1) + '\n')
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
