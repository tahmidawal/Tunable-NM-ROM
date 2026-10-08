"""Write the jcp-time2 job configs (DESIGN section 3, A1-A4). python experiments/jcp-time2/make_configs.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_CONTRACT = {'mesh': 8192, 'refs': {'ST': {'dt': 0.0003125, 'ntol': 1e-11, 'ltol': 1e-09, 'accept_residual': 2e-11},
                                       'S': {'dt': 0.005, 'ntol': 1e-11, 'ltol': 1e-09, 'accept_residual': 2e-11}}}
GRID = dict(dt_factors=[.125, .25, .5, 1, 2, 5, 10], dyadic_factors=[.125, .25, .5, 1, 2], tighter_factors=[.125, .25, .5, 1, 2],
            forms=['LSPG', 'GAL'], schemes=['BE', 'CN', 'CNR', 'BDF2'], control_schemes=['TH06'],
            anchor_schemes=['BDF2', 'CN'], anchor='BDF2', old_schemes=['BE', 'CN', 'BDF2'], old_dt_factors=[.5, 1, 2, 5])
RULES = {'acc': {'main': {'kind': 'point', 'rule': 'gauss96'}, 'hq': {'kind': 'point', 'rule': 'gauss192'},
                 'old': {'kind': 'mesh', 'rule': 'lat64'}},
         'wide': {'main': {'kind': 'point', 'rule': 'gauss128'}, 'hq': {'kind': 'point', 'rule': 'gauss192'},
                  'old': {'kind': 'mesh', 'rule': 'lat64'}},
         'fast': {'main': {'kind': 'point', 'rule': 'fib1597'}, 'hq': {'kind': 'point', 'rule': 'fib6765'},
                  'old': {'kind': 'mesh', 'rule': 'lat64'}}}


def cfg(attempt, mesh, settings, cohorts, timing, subset=None):
    c = dict(attempt=attempt, mesh=mesh, settings=settings, cohorts=cohorts, rules={s: RULES[s] for s in settings},
             grid=GRID, refs='REFS_DIR', ref_contract=REF_CONTRACT, timing=timing)
    if subset:
        c['case_subset'] = subset
    return c


C = {
    'smk': cfg('smk', 256, ['fast', 'acc'], ['dev6'], dict(cohort='dev6', cases=2, reps=1, burn=.1, seed=20261008),
               subset={'dev6': [0, 2]}),
    'a1kfast': cfg('a1kfast', 1024, ['fast'], ['dev6', 'val32'], dict(cohort='dev6', cases=6, reps=3, burn=.1, seed=20261008)),
    'a1kwide': cfg('a1kwide', 1024, ['wide'], ['dev6', 'val32'], dict(cohort='dev6', cases=6, reps=3, burn=.1, seed=20261008)),
    'a1kacc': cfg('a1kacc', 1024, ['acc'], ['dev6', 'val32'], dict(cohort='dev6', cases=6, reps=3, burn=.1, seed=20261008)),
}
ROM_ARMS = [dict(setting=st_, rule=RULES[st_]['main']['rule'], form=f_, scheme=sc, dt_factor=df)
            for st_ in ('fast', 'acc') for f_ in ('LSPG', 'GAL') for sc in ('BE', 'CN', 'CNR', 'BDF2') for df in (1, 2, 5)]
C['fom1k'] = dict(attempt='fom1k', driver='fomrun', mesh=1024, cohorts=['dev6', 'val32'], schemes=['BE', 'CN', 'CNR', 'BDF2'],
                  dt_factors=[.5, 1, 2, 5], ntols=[1e-4, 1e-6, 1e-8], order_cases=[['dev6', 0], ['dev6', 2]],
                  refs='REFS_DIR', rom_arms=ROM_ARMS, timing=dict(cases=6, reps=3, burn=.1, seed=20261008))
for k, v in C.items():
    (HERE / 'configs' / f'{k}.json').write_text(json.dumps(v, indent=1) + '\n')
    print(k)
