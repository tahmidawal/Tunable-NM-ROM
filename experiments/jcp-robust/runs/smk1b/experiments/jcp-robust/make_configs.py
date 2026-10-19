"""jcp-robust: write the E1 configs (DESIGN 2, A1-A4) and their job specs. Deterministic; re-running rewrites identical
files. E2 / E5 configs are written by their own generators once the selection results exist.

    python make_configs.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGES = [0, 128, 256, 384, 512]
ARMS_2D = [dict(form='LSPG', scheme='BE', factor=1), dict(form='LSPG', scheme='BE', factor=2),
           dict(form='LSPG', scheme='BE', factor=2.5), dict(form='LSPG', scheme='CNR', factor=1),
           dict(form='GAL', scheme='CN', factor=1), dict(form='GAL', scheme='CN', factor=2),
           dict(form='GAL', scheme='CN', factor=5)]
DRAWS = list(range(16))


def sha(p):
    return hashlib.sha256((HERE / p).read_bytes()).hexdigest()


def w(name, obj):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + '\n')
    print(name)


def e1_2d(mesh, cases='all38', draws=DRAWS, arms=ARMS_2D, sobol=True, attempt=None):
    rules = [dict(label=f'd{s}', rule=f'fib4181s{s}', succ=f'fib6765s{s}', level='full') for s in draws]
    rules.append(dict(label='bad', rule='badlat4181', succ=None, level='prod'))
    g = dict(name='R128', Rp=128, target='gauss384', check='gauss320', check_cases='dev6', tighter_cases='dev6',
             arms=arms, rules=rules, permute=dict(label='d0', arm=0), identity=dict(label_s1='d1', claimed='fib4181'))
    if mesh == 1024:
        g['fixture'] = dict(path='inputs/fixture_e1_2d.npz', sha256=sha('inputs/fixture_e1_2d.npz'), rules=['bad', 'd0'])
    groups = [g]
    if sobol and mesh == 1024:
        groups.append(dict(name='R128sob', Rp=128, target='gauss384', check=None, tighter_cases='dev6',
                           arms=[arms[0]],
                           rules=[dict(label=f'sob{s}', rule=f'sobol8192s{s}', succ=None, level='prod') for s in draws]))
    return dict(attempt=attempt or f'e1m{mesh}', driver='r2acc', mesh=mesh, edges=EDGES, cases=cases, groups=groups)


def e1_3d(draws=DRAWS, cases=None, sobol=True, attempt='e13d'):
    lat = lambda n: [f'lat{n}s{s}' for s in draws]
    cfgs = [dict(name='E1-3D-a', setting='W256', spec='qn|BDF2|0.025|k4', n=8192, succ_n=16384, target='gl64', check='gl56',
                 draws=draws, bad='badlat8192', permute=True,
                 fixture=dict(path='inputs/fixture_e1_3d.npz', sha256=sha('inputs/fixture_e1_3d.npz'))),
            dict(name='E1-3D-b', setting='W512', spec='qn|BDF2|0.025|k4', n=16384, succ_n=32768, target='gl64', check='gl56',
                 draws=draws, extra=[f'sob16384s{s}' for s in draws] if sobol else [])]
    extra = sorted(set(['lat8192', 'lat16384', 'lat32768', 'gl56', 'gl64', 'badlat8192'] + lat(8192) + lat(16384) + lat(32768)
                       + (cfgs[1]['extra'])))
    c = dict(attempt=attempt, driver='e1run3d', mode='e1', cohort_seed=923801, cohort_count=64, trust_fraction=.05,
             settings=[dict(name='W256', bank='W1024', Rp=256, a=6, rule='gl24', reduced=[], extra=[]),
                       dict(name='W512', bank='W1024', Rp=512, a=6, rule='gl24', reduced=[], extra=[])],
             extra_rules=extra, configs=cfgs)
    if cases is not None:
        c['case_subset'] = cases
    return c


def main():
    for m in (256, 1024, 4096):
        w(f'configs/e1m{m}.json', e1_2d(m))
    w('configs/e13d.json', e1_3d())
    # smoke smk1 (DESIGN 6): 1024^2 LSPG BE 1 + CN-R 1, draws 0-2 + bad, dev6; 3D E1-3D-a draws 0-2 + bad, cases 0-7
    w('configs/smk1m1024.json', e1_2d(1024, cases='dev6', draws=[0, 1, 2], arms=[ARMS_2D[0], ARMS_2D[3]], sobol=False,
                                      attempt='smk1'))
    s3 = e1_3d(draws=[0, 1, 2], cases=list(range(8)), sobol=False, attempt='smk1')
    s3['configs'] = s3['configs'][:1]
    w('configs/smk13d.json', s3)


if __name__ == '__main__':
    main()
