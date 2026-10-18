"""jcp-dims 3D job configurations (DESIGN 2.3, A1.3, A1.6, A1.7, A2). Deterministic; writes configs/<attempt>.json and
appends the 3D ladders to configs/MANIFEST-cells3d.json.

    python make_configs3d.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'
QUALIFY = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify')

GL = ['gl16', 'gl20', 'gl24', 'gl32', 'gl40', 'gl48', 'gl56', 'gl64']          # candidates gl16 ... gl48
LAT = ['lat8192', 'lat16384', 'lat32768', 'lat65536', 'lat131072', 'lat262144']  # candidates lat8192 ... lat65536
SPECS = ['fs|BE|0.01|K1', 'fs|CN|0.01|K1', 'qn|BDF2|0.025|k4']
BASE = {'M2_512': dict(bank='M2', Rp=512, a=3), 'W256': dict(bank='W1024', Rp=256, a=6),
        'W512': dict(bank='W1024', Rp=512, a=6), 'W768': dict(bank='W1024', Rp=768, a=6),
        'W1024': dict(bank='W1024', Rp=1024, a=6)}
KAPPA_M = {1: 512, 2: 1027, 3: 1538, 4: 2052, 8: 4097}       # W512 lists (A1.3): kappa 1 exact 512, others complete_M
SOB = [f'sob{2 ** k}' for k in range(13, 19)]
SMOL3 = [f'smolyak{l}' for l in range(8, 14)]
CONTROL = dict(key='W512|lat8192|fs|BE|0.01|K1', qp='lat16384', qpp=None)


def setting(name, base, M=None, rule='lat16384'):
    s = dict(name=name, **BASE[base], rule=rule, reduced=[], extra=[])
    if M:
        s['M'] = M
    return s


def job(name, settings, blocks, rules, cases=64, **extra):
    d = dict(attempt=name, driver='d3run', mode=name, cohort_seed=923801, cohort_count=cases, trust_fraction=0.05,
             timing=dict(burn=0.1), settings=settings, extra_rules=sorted(set(rules)), blocks=blocks, **extra)
    (CFG / f'{name}.json').write_text(json.dumps(d, indent=1) + '\n')
    return d


def successors(rule):
    if rule.startswith('gl'):
        lad = GL + ['gl80']
    else:
        lad = LAT
    i = lad.index(rule)
    return (lad[i + 1] if i + 1 < len(lad) else None), (lad[i + 2] if i + 2 < len(lad) else None)


def main():
    CFG.mkdir(exist_ok=True)
    man = dict(ladders=[])
    std = [setting(k, k) for k in BASE]
    # ------------------------------------------------------------------ smoke
    smk_settings = std + [setting('W512M1027', 'W512', 1027), setting('W512k1', 'W512', 512), setting('W512k8', 'W512', 4097)]
    smk_blocks = [
        # heaviest 3D blocks actually compiled and executed (A1.4): W1024 (M = 4097) Gauss tail and lattice tail, all specs,
        # ADT anchors; the adaptive forms at W512 kappa = 8 (M = 4097) on the lattice tail
        dict(kind='ladder', setting='W1024', specs=SPECS, rungs=['gl48', 'gl56', 'gl64'], cases=[0], tighter_cases=[0],
             tag_prefix='pilot|'),
        dict(kind='ladder', setting='W1024', specs=SPECS, rungs=['lat131072', 'lat262144'], cases=[0], tighter_cases=[0],
             tag_prefix='pilot|'),
        dict(kind='adaptive', setting='W512k8', forms=['LSPG|CN', 'GAL|CN'], dt=0.01, rungs=['lat131072', 'lat262144'],
             cases=[0], tighter_cases=[0]),
        dict(kind='override', setting_default='W512', setting_override='W512M1027', rule='lat16384',
             fixture='inputs/fixture3d.npz'),
        dict(kind='ladder', setting='W512', specs=['fs|CN|0.01|K1'], rungs=['lat16384'], cases=[0, 1, 2, 3], tighter_cases=[0]),
        dict(kind='ladder', setting='M2_512', specs=['fs|BDF2|0.0125|K1'], rungs=['gl24'], cases=[0, 1, 2, 3], tighter_cases=[0]),
        dict(kind='ladder', setting='W512', specs=SPECS, rungs=['lat8192', 'lat16384'], cases=[0, 1], tighter_cases=[0],
             tag_prefix='lad|'),
        dict(kind='ladder', setting='M2_512', specs=['fs|BE|0.01|K1'], rungs=['gl32'], cases=[0, 1], tighter_cases=[0]),
        dict(kind='ladder', setting='W1024', specs=['fs|CN|0.01|K1'], rungs=['lat262144'], cases=[0, 1], tighter_cases=[0]),
        dict(kind='adaptive', setting='W512k1', forms=['LSPG|CN', 'GAL|CN'], dt=0.01, rungs=['lat65536', 'lat131072'],
             cases=[0, 1], tighter_cases=[0]),
        dict(kind='adaptive', setting='W512', forms=['LSPG|CN', 'GAL|CN'], dt=0.01, rungs=['lat65536'], cases=[0, 1],
             tighter_cases=[0]),
        dict(kind='rounds', members=[], control=CONTROL),
        dict(kind='fixgate', setting='W512', target='gl64', rules=['badlat262144', 'lat262144', 'sob262144s0', 'smolyak13', 'gl64'],
             fixture='inputs/fixture3d.npz'),
        dict(kind='panel', label='smoke', A=dict(setting='M2_512', rule='gl32', spec='fs|BE|0.01|K1'),
             subjects=[dict(setting='W512', rule='lat16384', spec='fs|CN|0.01|K1'),
                       dict(setting='W512', rule='lat65536', spec='ad|GAL|CN|0.01|prod'),
                       dict(setting='W1024', rule='lat262144', spec='fs|CN|0.01|K1'),
                       dict(setting='M2_512', rule='gl32', spec='fs|BE|0.01|K1', double=True)],
             cases=2, reps=1, seed=7),
        dict(kind='panel', label='smoke-uncertified', A=dict(setting='M2_512', rule='gl32', spec='fs|BE|0.01|K1'),
             subjects=[dict(setting='W256', rule='gl24', spec='fs|CN|0.01|K1')], cases=1, reps=1, seed=8)]
    rules = GL + ['gl80'] + LAT + ['badlat262144', 'sob262144s0', 'smolyak13']
    job('smk3', smk_settings, smk_blocks, rules)
    # ------------------------------------------------------------------ round 4: qualify's 16 exposed (A1.1 / 3R)
    an = json.loads((QUALIFY / 'checks' / 'analysis.json').read_text())['threeD']
    exposed = an['exposed_untested_after_last_round']
    assert len(exposed) == 16, len(exposed)
    members = []
    for k in exposed:
        rule = k.split('|')[1]
        qp, qpp = successors(rule)
        members.append(dict(key=k, qp=qp, qpp=qpp))
    job('q3r4', std, [dict(kind='rounds', members=members, control=CONTROL)], GL + ['gl80'] + LAT)
    # ------------------------------------------------------------------ m* ladders (3M)
    for att, names in (('m3a', ['M2_512', 'W256', 'W512']), ('m3b', ['W768', 'W1024'])):
        blocks = []
        for n in names:
            for fam, rungs in (('gauss', GL), ('lattice', LAT)):
                blocks.append(dict(kind='ladder', setting=n, specs=SPECS, rungs=rungs))
                for sp in SPECS:
                    man['ladders'].append(dict(kind='3M', setting=n, M='default', spec=sp, family=fam, draw=0, rungs=rungs,
                                               candidates=rungs[:-2]))
        job(att, [setting(n, n) for n in names], blocks, GL + LAT)
    # ------------------------------------------------------------------ test-space and fixed-(M, m) panels (3B, A1.6)
    k_settings = [setting(f'W512k{k}', 'W512', M) for k, M in KAPPA_M.items()]
    fix_settings = [setting(f'{w}M2052', w, 2052) for w in ('W256', 'W768', 'W1024')]
    blocks = []
    for k in KAPPA_M:
        blocks.append(dict(kind='adaptive', setting=f'W512k{k}', forms=['LSPG|CN', 'GAL|CN'], dt=0.01,
                           rungs=['lat65536', 'lat131072', 'lat262144']))
        for f in ('LSPG|CN', 'GAL|CN'):
            man['ladders'].append(dict(kind='3B-ad', setting=f'W512k{k}', M=KAPPA_M[k], form=f, family='lattice', draw=0,
                                       rungs=['lat65536', 'lat131072', 'lat262144'], candidates=['lat65536'],
                                       deferrable_kappa8=(k == 8)))
        if k in (2, 3, 8):                       # kappa 4 = the default W512 of m3a; no fs at kappa 1 (A2.1)
            blocks.append(dict(kind='ladder', setting=f'W512k{k}', specs=['fs|CN|0.01|K1'], rungs=LAT[1:]))
            man['ladders'].append(dict(kind='3B-fs', setting=f'W512k{k}', M=KAPPA_M[k], spec='fs|CN|0.01|K1',
                                       family='lattice', draw=0, rungs=LAT[1:], candidates=LAT[1:-2],
                                       deferrable_kappa8=(k == 8)))
    for s in fix_settings:
        blocks.append(dict(kind='ladder', setting=s['name'], specs=['fs|CN|0.01|K1'], rungs=LAT[1:]))
        man['ladders'].append(dict(kind='3A-fixMm', setting=s['name'], M=2052, spec='fs|CN|0.01|K1', family='lattice',
                                   draw=0, rungs=LAT[1:], candidates=LAT[1:-2]))
    job('k3', k_settings + fix_settings, blocks, LAT)
    # ------------------------------------------------------------------ families (3F) stage (i)
    blocks, rules = [], list(LAT) + ['gl64', 'badlat262144'] + SMOL3
    for s in range(1, 5):
        rungs = [f'{r}s{s}' for r in LAT]
        rules += rungs
        blocks.append(dict(kind='ladder', setting='W512', specs=['fs|CN|0.01|K1'], rungs=rungs))
        man['ladders'].append(dict(kind='3F', setting='W512', spec='fs|CN|0.01|K1', family='lattice', draw=s, rungs=rungs,
                                   candidates=rungs[:-2]))
    for s in range(5):
        rungs = [f'{r}s{s}' for r in SOB]
        rules += rungs
        blocks.append(dict(kind='ladder', setting='W512', specs=['fs|CN|0.01|K1'], rungs=rungs))
        man['ladders'].append(dict(kind='3F', setting='W512', spec='fs|CN|0.01|K1', family='sobol', draw=s, rungs=rungs,
                                   candidates=rungs[:-2]))
    blocks.append(dict(kind='ladder', setting='W512', specs=['fs|CN|0.01|K1'], rungs=SMOL3))
    man['ladders'].append(dict(kind='3F', setting='W512', spec='fs|CN|0.01|K1', family='smolyak', draw=0, rungs=SMOL3,
                               candidates=SMOL3[:-2]))
    blocks.append(dict(kind='ladder', setting='W512', specs=['fs|CN|0.01|K1'], rungs=['badlat262144']))
    fin = ['lat262144'] + [f'lat262144s{s}' for s in range(1, 5)] + [f'sob262144s{s}' for s in range(5)] + ['smolyak13']
    blocks.append(dict(kind='fixgate', setting='W512', target='gl64', rules=['badlat262144'] + fin,
                       fixture='inputs/fixture3d.npz'))
    job('b3f', [setting('W512', 'W512')], blocks, rules)
    # ------------------------------------------------------------------ stage (ii) (b3m)
    blocks, rules = [], ['gl64']
    for fam in ('mc', 'hal'):
        for s in range(5):
            rungs = [f'{fam}{2 ** k}s{s}' for k in range(13, 19)]
            rules += rungs
            blocks.append(dict(kind='ladder', setting='W512', specs=['fs|CN|0.01|K1'], rungs=rungs))
            man['ladders'].append(dict(kind='3Fii', setting='W512', spec='fs|CN|0.01|K1', family=fam, draw=s, rungs=rungs,
                                       candidates=rungs[:-2]))
    blocks.append(dict(kind='fixgate', setting='W512', target='gl64',
                       rules=[f'{f}262144s{s}' for f in ('mc', 'hal') for s in range(5)], fixture='inputs/fixture3d.npz'))
    job('b3m', [setting('W512', 'W512')], blocks, rules)
    (CFG / 'MANIFEST-cells3d.json').write_text(json.dumps(man, indent=1) + '\n')
    print('3D ladders', len(man['ladders']))


if __name__ == '__main__':
    main()
