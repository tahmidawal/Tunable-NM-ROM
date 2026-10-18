"""Generate every job config of the lane from the pinned inputs (DESIGN 2, A1, A2): FRONT-2D v1 members per mesh and
the 3D qualified subset. A mismatch of a pinned sha256 stops the lane.

    python make_configs.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify')
FRONT_SHA = 'a51ac1a250f00d4b0006a84f2434ab3c96407b1f575f4ffff7a3eb25b7a8b77d'
EXPORT_SHA = '6cb394d149be8852260b75a585db8102791ef52e9963ab6934f67d536c374431'
NS = '/cluster/tufts/paralab/tawal01/jcp-budget'
QP = {'fib4181': ('fib6765', 'fib10946'), 'gauss128': ('gauss160', 'gauss192')}      # qualify ladders (DESIGN 2.3 of qualify)
EDGES = [0, 128, 256, 384]
B2 = dict(qinf='gauss320', qcheck='gauss640', check_cases='dev6', m_ablation=True)
THREE_D = {'W256|gl24|fs|CN|0.01|K1': ('gl32', 'gl40'), 'W256|gl24|fs|CN|0.0125|K1': ('gl32', 'gl40'),
           'W256|lat8192|qn|BDF2|0.025|k4': ('lat16384', 'lat32768'), 'W512|lat16384|qn|BDF2|0.025|k4': ('lat32768', 'lat65536')}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def members_2d():
    assert sha(Q / 'runs/FRONT-2D.json') == FRONT_SHA, 'FRONT-2D.json differs from the pin'
    F = json.loads((Q / 'runs/FRONT-2D.json').read_text())
    out = {}
    for mesh, v in F['meshes'].items():
        ms = []
        for m in v['members']:
            mf = m['manifest']
            assert mf['mode_list']['policy'] == 'frozen F_2048 prefix' and mf['M'] == 4 * mf['Rp']
            qp, qpp = QP[mf['rule']]
            ms.append(dict(name=m['config'], Rp=mf['Rp'], form=mf['form'], scheme=mf['scheme'], factor=mf['factor'],
                           rule=mf['rule'], qp=qp, qpp=qpp, diagnostic=(mf['form'] == 'GAL')))
        out[int(mesh)] = ms
    return out


def members_3d():
    assert sha(Q / 'runs/EXPORT.json') == EXPORT_SHA, 'qualify EXPORT.json differs from the pin'
    E = json.loads((Q / 'runs/EXPORT.json').read_text())['threeD']['tested']
    keys = sorted(k for k, v in E.items() if v == 'pass')
    assert keys == sorted(THREE_D), (keys, sorted(THREE_D))
    return [dict(key=k, qp=THREE_D[k][0], qpp=THREE_D[k][1]) for k in keys]


def settings_3d(extra=()):
    rules = {'W256': {'gl24', 'gl32', 'gl40', 'gl56', 'gl64', 'lat8192', 'lat16384', 'lat32768', 'lat256'},
             'W512': {'lat16384', 'lat32768', 'lat65536', 'gl56', 'gl64'}}
    base = {'W256': dict(name='W256', bank='W1024', Rp=256, a=6, rule='gl24'),
            'W512': dict(name='W512', bank='W1024', Rp=512, a=6, rule='lat16384')}
    return [dict(base[k], reduced=[], extra=sorted(rules[k] - {base[k]['rule']})) for k in ('W256', 'W512')]


def write(name, cfg):
    (HERE / 'configs').mkdir(exist_ok=True)
    (HERE / 'configs' / f'{name}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('wrote', name)


def main():
    M2 = members_2d()
    pick = lambda mesh, names: [m for m in M2[mesh] if m['name'].split('|', 1)[1] in names]
    # ---------------- smk1 (2D smoke, DESIGN 9 / A1.8 / A2.2): reference pilot + ROM at 1024^2 + 4096^2 memory pilot
    write('smk1_e0repro', dict(mode='e0repro', dim=2, cohort='dev6', cases=[0]))
    write('smk1_ref', dict(mode='fd', dim=2, cohort='dev6', cases=[0], horizon=0.5, fine=True, allow_refine=True))
    write('smk1_x1', dict(mode='x1', dim=2, cohort='dev6', cases=[0], horizon=0.5))
    sm = pick(1024, {'R384|LSPG|CNR|1', 'R128|LSPG|BE|1', 'R128|GAL|CN|2'})
    write('smk1_b1024', dict(attempt='smk1', mesh=1024, edges=EDGES, cases='dev2', members=sm,
                             b2=dict(B2, design='full', check_cases='dev2'),
                             b7=dict(horizon=0.5, q2_cases='dev2', tighter_cases='dev2'),
                             b8=dict(horizon=0.5, refs_dir=f'{NS}/smk1/output/ref_refrun_smk1_ref',
                                     manifest_from=f'{NS}/smk1/output/ref_refrun_smk1_ref/result.json', check_cases='dev2'),
                             controls=dict(Rp=384, cases='dev2', under_resolved='gauss32')))
    write('smk1_pilot4096', dict(attempt='smk1', mesh=4096, edges=EDGES, cases='dev2', ls_cases='dev2',
                                 members=pick(4096, {'R384|LSPG|CNR|1'}), pilot=dict(qinf='gauss320', qcheck='gauss640',
                                                                                    cases='dev2')))
    # ---------------- smk2 (3D smoke)
    write('smk2_ref', dict(mode='fd', dim=3, cohort='val64', cases=[0], horizon=0.5, fine=True, allow_refine=True))
    write('smk2_x1', dict(mode='x1', dim=3, cohort='val64', cases=[0], horizon=0.5))
    write('smk2_b3', dict(attempt='smk2', mode='b3', cohort_seed=923801, cohort_count=64, case_subset=[0, 1],
                          trust_fraction=0.05, settings=settings_3d(), members=members_3d()[:2] + [members_3d()[3]],
                          b2=dict(qinf='gl56', qcheck='gl64', check_cases=[0, 1]),
                          b7=dict(horizon=0.5, q2_cases=[0, 1], qpp_cases=[0], tighter_cases=[0, 1]),
                          controls=dict(cases=[0, 1], under_resolved='lat256')))
    # ---------------- production (staged only after the coordinator's release; certificates from the smokes)
    v32 = list(range(32))
    for nm, cs in (('ref2a', ('dev6', list(range(6)), 'val32', v32[:7])), ('ref2b', ('val32', v32[7:20])),
                   ('ref2c', ('val32', v32[20:]))):
        if len(cs) == 4:
            write(f'{nm}_dev6', dict(mode='fd', dim=2, cohort='dev6', cases=cs[1], horizon=0.5, fine=True, allow_refine=True))
            write(f'{nm}_val32', dict(mode='fd', dim=2, cohort='val32', cases=cs[3], horizon=0.5, fine=True, allow_refine=True))
        else:
            write(f'{nm}_val32', dict(mode='fd', dim=2, cohort=cs[0], cases=cs[1], horizon=0.5, fine=True, allow_refine=True))
    write('ps2', dict(mode='ps', dim=2, cohort='val32', cases=[0, 1, 5, 19], horizon=0.5))
    write('ref3', dict(mode='fd', dim=3, cohort='val64', cases=list(range(16)) + [43], horizon=0.5, fine=True,
                       allow_refine=True))
    write('ps3', dict(mode='ps', dim=3, cohort='val64', cases=[0, 43], horizon=0.5))
    for mesh, cases, design, b7 in ((1024, 'all38', 'full', True), (256, 'all38', 'pairwise', True),
                                    (4096, 'sub16', 'pairwise', True)):
        cfg = dict(attempt=f'b{mesh}', mesh=mesh, edges=EDGES, cases=cases, members=M2[mesh],
                   b2=dict(B2, design=design, check_cases='dev6'),
                   b8=dict(horizon=0.5, refs_dir=f'{NS}/refs', manifest_file='REFS-MANIFEST.sha256', check_cases='dev6'),
                   controls=dict(Rp=384, cases='dev6', under_resolved='gauss32'))
        cfg['b7'] = dict(horizon=0.5, q2_cases='sub16', tighter_cases='dev6')    # at 256^2 a diagnostic (feeds B8)
        write(f'b{mesh}', cfg)
    write('b3d', dict(attempt='b3d', mode='b3', cohort_seed=923801, cohort_count=64, case_subset=list(range(16)),
                      trust_fraction=0.05, settings=settings_3d(), members=members_3d(),
                      b2=dict(qinf='gl56', qcheck='gl64', check_cases=[0, 1, 2, 3]),
                      b7=dict(horizon=0.5, q2_cases=list(range(8)), qpp_cases=[0, 1, 2, 3], tighter_cases=list(range(6))),
                      controls=dict(cases=[0, 1, 2, 3], under_resolved='lat256')))
    write('ref2t1', dict(mode='fd', dim=2, cohort='dev6', cases=list(range(6)), horizon=1.0, fine=False, allow_refine=False))
    write('ref2t1v', dict(mode='fd', dim=2, cohort='val32', cases=list(range(10)) + [19], horizon=1.0, fine=False, allow_refine=False))
    write('ps2t1', dict(mode='ps', dim=2, cohort='val32', cases=[0, 19], horizon=1.0))
    write('b1024t1', dict(attempt='b1024t1', mesh=1024, edges=EDGES, cases='sub16', members=M2[1024],
                          b7=dict(horizon=1.0, q2_cases='sub16', tighter_cases='dev6')))      # B7 T = 1.0 (last deferred)


if __name__ == '__main__':
    main()
