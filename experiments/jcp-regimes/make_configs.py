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


def arms(name, cases, tighter='sub'):
    a = [dict(form='LSPG', scheme='BE', factor=1, cases=cases, q2_cases='all', tighter_cases=tighter),
         dict(form='LSPG', scheme='CNR', factor=1, cases=cases, q2_cases='all', tighter_cases=tighter)]
    if name == 'R384':
        a.append(dict(form='GAL', scheme='CN', factor=1, cases=cases, q2_cases='all', tighter_cases=tighter))
    return a


FRONT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/FRONT-2D.json')
FRONT_SHA = 'a51ac1a250f00d4b0006a84f2434ab3c96407b1f575f4ffff7a3eb25b7a8b77d'


def frozen():
    """Frozen identities from FRONT-2D (qualify): checkpoint, rotation and, per setting, the main rule's manifest."""
    import hashlib
    raw = FRONT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FRONT_SHA
    d = json.loads(raw)['meshes']['1024']
    mfs = [x['manifest'] for g in ('members', 'other_qualified') for x in d[g]]
    out = dict(front_sha256=FRONT_SHA, settings={})
    for n, r, m, _, _ in SETTINGS:
        # the Delta t0 LSPG entries are P1-cited without rule_meta; the setting identity (rule, mesh, modes, fingerprint
        # of the LSPG tables) is the same for every non-fallback arm of the setting, so take all of them and require one
        metas = {json.dumps(mf['rule_meta'], sort_keys=True) for mf in mfs
                 if mf['Rp'] == r and mf['rule'] == m and not mf['fallback'] and mf.get('rule_meta')}
        assert len(metas) == 1, (n, len(metas))
        out['settings'][n] = json.loads(metas.pop())
        ids = {(mf['checkpoint_sha256'], mf['rotation_sha256']) for mf in mfs}
        assert len(ids) == 1
        out['checkpoint_sha256'], out['rotation_sha256'] = ids.pop()
    return out


def rom_config(attempt, cohorts, cases, blocks=(), tighter='sub'):
    return dict(attempt=attempt, driver='qacc', mesh=1024, edges=EDGES, cohorts=cohorts, frozen=frozen(),
                settings=[dict(name=n, Rp=r, rule=m, qp=qp, qpp=qpp, arms=arms(n, cases, tighter)) for n, r, m, qp, qpp in SETTINGS],
                blocks=list(blocks))


def arm_names():
    return sorted(f"{n}_{a['form']}_{a['scheme']}_{a['factor']:g}" for n, *_ in SETTINGS for a in arms(n, 'all'))


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
    write('smk1-fverify', dict(source='qacc_smk1-rom', arms=arm_names()))
    write('smk1-rom', rom_config('smk1', ['dev6'] + REGIMES, cases, blocks, tighter='all'))   # smoke: every case (no regime has c0-3 complete)


def prod(reduce_c2=False, calib_probes=False):
    """Production configs (A3.3, A2.2): reduce_c2 -> C2 c0-c7 for references, ROM, Q1/Q2 together; calib_probes -> a
    coarsest-level tight probe on case 0 of every regime (A1.4, if the smoke's R-calib did not agree)."""
    import numpy as np
    import rcohorts as RG
    rom_cases = []
    for g in REGIMES:
        n = len(RG.descriptors_2d(g))
        cases = list(range(8)) if (g == 'C2' and reduce_c2) else list(range(n))
        cfg = dict(mode='fd', dim=2, cohort=g, cases=cases, kinds=['ST'],
                   tight_allowance=TIGHT_ALLOWANCE.get(g, 0) if cases == list(range(n)) else 0)
        if calib_probes:
            cfg['probes'] = dict(tight_coarse=[0])
        write(f'ref-{g}', cfg)
        write(f'slim-{g}', dict(sources=[f'refjob_ref-{g}']))
        rom_cases += [[g, c] for c in cases]
        if g != 'N0s':
            D = RG.descriptors_2d(g)[cases]
            PS_CASE[g] = int(np.array(cases)[np.argmax([RG.desc_2d(g, r_)['s'] for r_ in D])])   # largest s (A3 minor)
    base = [['dev6', c] for c in range(6)] + [['val32', c] for c in range(32)]
    write('rom1024', rom_config('rom1024', ['dev6', 'val32'] + REGIMES, 'all' if not reduce_c2 else base + rom_cases))
    write('fverify-rom1024', dict(source='qacc_rom1024', arms=arm_names()))
    for g, c in PS_CASE.items():
        write(f'ps-{g}', dict(mode='ps', dim=2, cohort=g, cases=[c], attempt=0))
    return dict(reduce_c2=reduce_c2, calib_probes=calib_probes, ps_cases=dict(PS_CASE))


if __name__ == '__main__':
    {'smk1': smk1, 'prod': lambda: prod('--reduce-c2' in sys.argv, '--calib-probes' in sys.argv)}[sys.argv[1]]()
