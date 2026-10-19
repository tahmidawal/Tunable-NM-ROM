"""jcp-regimes: write the job configs (DESIGN §5, A1.6, A2.5, A3.2, A3.3). Run from the lane directory:

    python make_configs.py smk1          the real-size smoke (one cluster job, <= 2 GPU-h)
    python make_configs.py prod          the production configs (references per cohort, the ROM job, PS)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'
EDGES = [0, 128, 256, 384, 512]
SETTINGS = [('R128', 128, 'fib4181', 'fib6765', 'fib10946'), ('R256', 256, 'gauss128', 'gauss160', 'gauss192'),
            ('R384', 384, 'gauss128', 'gauss160', 'gauss192'), ('R512', 512, 'gauss192', 'gauss256', 'gauss320')]
REGIMES = ['N0', 'N0s', 'Ap', 'Am', 'C1', 'C2', 'B1', 'B2']
PILOTS = dict(C2=(9, 11), B2=(15, 0), Ap=(13,))           # A2.5 (largest s; A+ largest nu)
TIGHT_ALLOWANCE = __import__('rcohorts').TIGHT_ALLOWANCE   # A2.6 / A3.2
PS_CASE = {}                                             # filled by prod(): largest s per cohort (A3 minor)


def arms(name, cases):
    a = [dict(form='LSPG', scheme='BE', factor=1, cases=cases, q2_cases='all', tighter_cases='sub'),
         dict(form='LSPG', scheme='CNR', factor=1, cases=cases, q2_cases='all', tighter_cases='sub')]
    if name == 'R384':
        a.append(dict(form='GAL', scheme='CN', factor=1, cases=cases, q2_cases='all', tighter_cases='sub'))
    return a


def rom_config(attempt, cohorts, cases, blocks=()):
    return dict(attempt=attempt, driver='qacc', mesh=1024, edges=EDGES, cohorts=cohorts,
                settings=[dict(name=n, Rp=r, rule=m, qp=qp, qpp=qpp, arms=arms(n, cases)) for n, r, m, qp, qpp in SETTINGS],
                blocks=list(blocks))


def write(name, d):
    CFG.mkdir(exist_ok=True)
    (CFG / f'{name}.json').write_text(json.dumps(d, indent=1) + '\n')
    print('wrote', CFG / f'{name}.json')


def smk1():
    write('smk1-ref-dev6', dict(mode='fd', dim=2, cohort='dev6', cases=[0], kinds=['ST']))
    write('smk1-ref-C2', dict(mode='fd', dim=2, cohort='C2', cases=[9], kinds=['ST'], tight_allowance=0,
                              probes=dict(upwind=[9], tight_coarse=[9])))
    write('smk1-ref-B2', dict(mode='fd', dim=2, cohort='B2', cases=[15], kinds=['ST'], tight_allowance=0,
                              probes=dict(tight_coarse=[15])))
    write('smk1-ref-Ap', dict(mode='fd', dim=2, cohort='Ap', cases=[13], kinds=['ST'], tight_allowance=0,
                              probes=dict(tight_coarse=[13])))
    write('smk1-slim', dict(sources=[f'refjob_smk1-ref-{g}' for g in ('dev6', 'C2', 'B2', 'Ap')]))
    smoke = dict(N0=[0, 1], N0s=[0, 1], Ap=[0, 13], Am=[0, 1], C1=[0, 1], C2=[9, 11], B1=[0, 1], B2=[0, 15])
    cases = [['dev6', c] for c in range(6)] + [[g, c] for g in REGIMES for c in smoke[g]]
    ctrl = [['dev6', 0], ['dev6', 1], ['C2', 9], ['C2', 11], ['B2', 15], ['B2', 0]]
    blocks = [dict(setting='R384', kind='controls', cases=ctrl, under_resolved='gauss32',
                   under_resolved_successor='gauss64', b0_cases=ctrl)]
    write('smk1-rom', rom_config('smk1', ['dev6'] + REGIMES, cases, blocks))


def prod():
    import numpy as np
    import rcohorts as RG
    for g in REGIMES:
        n = len(RG.descriptors_2d(g))
        write(f'ref-{g}', dict(mode='fd', dim=2, cohort=g, cases=list(range(n)), kinds=['ST'],
                               tight_allowance=TIGHT_ALLOWANCE.get(g, 0)))
        write(f'slim-{g}', dict(sources=[f'refjob_ref-{g}']))
        if g != 'N0s':
            D = RG.descriptors_2d(g)
            key = (lambda r_: RG.nu_2d(g, r_)) if g == 'Ap' else (lambda r_: RG.desc_2d(g, r_)['s'])
            PS_CASE[g] = int(np.argmax([key(r_) for r_ in D]))
    write('rom1024', rom_config('rom1024', ['dev6', 'val32'] + REGIMES, 'all'))
    write('ps', dict(cases=PS_CASE))       # one PS job step per cohort is generated at staging (A3 minor)
    for g, c in PS_CASE.items():
        write(f'ps-{g}', dict(mode='ps', dim=2, cohort=g, cases=[c], attempt=0))


if __name__ == '__main__':
    {'smk1': smk1, 'prod': prod}[sys.argv[1]]()
