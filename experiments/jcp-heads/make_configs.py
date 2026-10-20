"""Generate the lane's job configs (DESIGN 8). Smoke configs now; production configs from the frozen rule selection
(runs/RULES.json, written by analyze.py after `rule1024`) later.

    python make_configs.py smoke
    python make_configs.py train            (the matched training job)
    python make_configs.py rule1024         (needs the trained compact heads committed in heads/)
    (accuracy and timing configs follow the frozen rule selection runs/RULES.json; written and audited then)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = '/cluster/tufts/paralab/tawal01/jcp-heads'
EDGES = [0, 4, 8, 16, 32, 64, 128, 256, 384, 512]
LADDER = [16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320]
INC = '../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
NPZ = f'{NS}/data/sep_coeff_N256_K16_R512.npz'
KS, SEEDS = (4, 8, 16, 32), (0, 1, 2)
LIN = (4, 8, 16, 32, 64, 128, 256, 384)
DEV2 = [['dev6', 0], ['dev6', 1]]
A_SPEC = dict(mesh=1024, name='A_R384', kind='lin', Rp=384, rule='gauss128', scheme='CNR')   # P1's reference A


def g(p):
    return f'gauss{p}'


def succ(rule, n=1):
    i = LADDER.index(int(rule[5:]))
    return g(LADDER[i + n])


def write(name, cfg):
    p = HERE / 'configs' / f'{name}.json'
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1) + '\n')
    print(p)


def smoke():
    smk_head = lambda k: f'{NS}/smk1/output/htrain_smk1train/head_smk_k{k}_s0.npz'
    write('smk1train', dict(npz=NPZ, ckpt=INC, arms=[
        dict(name='identity', k=16, seed=0, identity=True),
        dict(name='smk_k4_s0', k=4, seed=0, steps=2000),
        dict(name='smk_k32_s0', k=32, seed=0, steps=2000)]))
    rep_head = {r: {f'dev6|{c}': f'inputs/repro/head_{r}_dev6{c}.npz' for c in (0, 2)} for r in ('gauss32', 'gauss64')}
    write('smk1acc', dict(mesh=1024, edges=EDGES, arms=[
        dict(name='Hinc', kind='head', head=INC, rule='gauss32', qp='gauss48', qpp='gauss64', schemes=['BE', 'CNR'],
             cases='all38', q2_cases='sub16'),
        dict(name='Hk4', kind='head', head=smk_head(4), rule='gauss32', qp='gauss48', qpp='gauss64', schemes=['BE'],
             cases=DEV2, q2_cases=DEV2),
        dict(name='Hk32', kind='head', head=smk_head(32), rule='gauss32', qp='gauss48', qpp='gauss64', schemes=['BE'],
             cases=DEV2, q2_cases=DEV2),
        dict(name='L4', kind='lin', Rp=4, rule='gauss24', qp='gauss32', qpp='gauss48', schemes=['BE'], cases=DEV2, q2_cases=DEV2),
        dict(name='L384', kind='lin', Rp=384, rule='gauss128', qp='gauss160', qpp='gauss192', schemes=['BE'], cases=DEV2,
             q2_cases=DEV2)],
        blocks=[dict(kind='controls', arm='Hinc', repro_head=rep_head, under_resolved=['gauss8', 'gauss16', 'gauss24'],
                     repro_lin=dict(Rp=384, rule='gauss128', fixtures={
                         'BE': 'inputs/repro/lin_R384_gauss128_BE_dev6.npz',
                         'CNR': 'inputs/repro/lin_R384_gauss128_CNR_dev6.npz'}))]))
    write('smk1rule', dict(mesh=1024, edges=EDGES, ladder=LADDER, selectable_max=192, population='sub16', check='dev6',
                           groups=[dict(name='Hinc', min_pass=1, arms=[dict(name='Hinc', kind='head', head=INC)]),
                                   dict(name='Hk4', min_pass=2, arms=[dict(name='Hk4', kind='head', head=smk_head(4)),
                                                                      dict(name='Hk4b', kind='head', head=smk_head(4))]),
                                   dict(name='L8', min_pass=1, arms=[dict(name='L8', kind='lin', Rp=8)])]))
    subj = [dict(mesh=1024, name='Hinc', kind='head', head=INC, rule='gauss32', qp='gauss48', qpp='gauss64', scheme='BE'),
            dict(mesh=1024, name='Hinc', kind='head', head=INC, rule='gauss32', qp='gauss48', qpp='gauss64', scheme='CNR'),
            dict(mesh=1024, name='L16', kind='lin', Rp=16, rule='gauss24', qp='gauss32', qpp='gauss48', scheme='BE'),
            dict(mesh=1024, name='L384', kind='lin', Rp=384, rule='gauss128', qp='gauss160', qpp='gauss192', scheme='BE')]
    write('smk2time', dict(meshes={'1024': dict(edges=EDGES)}, A=A_SPEC, subjects=subj,
                           cases=DEV2 + [['val32', 0]], cert_cases=DEV2, reps={'dev6': 2, 'val32': 1}, burn=0.1, seed=20261018,
                           doubled=[dict(subj[0], cases=[['dev6', 0]])], inject_identity_fault=['1024|L16|gauss24|BE']))
    write('smk2pilot', dict(mesh=4096, edges=EDGES, edges_A=[0, 128, 256, 384], rule='gauss192', qp='gauss256', qpp='gauss320',
                            A=A_SPEC, heads=[dict(name=f'P{k}s{s}', K=k) for k in KS for s in SEEDS] +
                            [dict(name='Pinc', head=INC)], lin=list(LIN)))


def head_path(K, s):
    return f'heads/head_mid_k{K}_s{s}.npz'


def train():
    arms = [dict(name='identity', k=16, seed=0, identity=True)]
    arms += [dict(name=f'mid_k{K}_s{s}', k=K, seed=s, steps=200000) for K in KS for s in SEEDS]
    arms += [dict(name='probe_k16_s0_2000', k=16, seed=0, steps=2000)]          # A1.7 sensitivity probe
    write('train', dict(npz=NPZ, ckpt=INC, arms=arms))


def rule1024():
    for K in KS:
        for s in SEEDS:
            assert (HERE / head_path(K, s)).exists(), head_path(K, s)
    groups = [dict(name=f'H{K}', min_pass=2, arms=[dict(name=f'H{K}s{s}', kind='head', head=head_path(K, s)) for s in SEEDS])
              for K in KS]
    groups += [dict(name=f'H{K}m128', min_pass=2, M=128,
                    arms=[dict(name=f'H{K}s{s}m128', kind='head', head=head_path(K, s)) for s in SEEDS]) for K in KS if 4 * K != 128]
    groups += [dict(name='Hinc', min_pass=1, arms=[dict(name='Hinc', kind='head', head=INC)])]
    groups += [dict(name=f'L{r}', min_pass=1, arms=[dict(name=f'L{r}', kind='lin', Rp=r)]) for r in LIN]
    groups += [dict(name=f'L{r}m128', min_pass=1, M=128, arms=[dict(name=f'L{r}m128', kind='lin', Rp=r)])
               for r in (4, 8, 16) ]
    write('rule1024', dict(mesh=1024, edges=EDGES, ladder=LADDER, selectable_max=192, population='sub16', check='dev6',
                           groups=groups))


def rules():
    """Frozen rule per arm group (runs/RULES.json from analyze.py rules): {group: (main, qp, qpp)}; exhausted -> None."""
    r = json.loads((HERE / 'runs' / 'RULES.json').read_text())['groups']
    return {g: (v['chosen']['main'], v['chosen']['qp'], v['chosen']['qpp']) if v['chosen'] else None for g, v in r.items()}


def arm_specs(R, m128=True):
    """Every accuracy arm (DESIGN 1, 3, A1.5): name, kind, head/Rp, M, rule roles, from the frozen rules. Groups whose
    ladder was exhausted are run at the last selectable rung (gauss192) and are labelled not quadrature-qualified."""
    fallback = ('gauss192', 'gauss256', 'gauss320')
    arms = []

    def roles(g):
        m, qp, qpp = R.get(g) or fallback
        return dict(rule=m, qp=qp, qpp=qpp, rule_group=g, rule_status='selected' if R.get(g) else 'ladder exhausted')
    for K in KS:
        for s in SEEDS:
            arms.append(dict(name=f'H{K}s{s}', kind='head', head=head_path(K, s), **roles(f'H{K}')))
    arms.append(dict(name='Hinc', kind='head', head=INC, **roles('Hinc')))
    for r in LIN:
        arms.append(dict(name=f'L{r}', kind='lin', Rp=r, **roles(f'L{r}')))
    if m128:
        for K in KS:
            if 4 * K != 128:
                for s in SEEDS:
                    arms.append(dict(name=f'H{K}s{s}m128', kind='head', head=head_path(K, s), M=128, **roles(f'H{K}m128')))
        for r in (4, 8, 16):
            arms.append(dict(name=f'L{r}m128', kind='lin', Rp=r, M=128, **roles(f'L{r}m128')))
    return arms


def acc(mesh, m128=True, K_out=()):
    R = rules()
    arms = []
    for a in arm_specs(R, m128=m128):
        if any(a['name'].startswith(f'H{K}s') for K in K_out):
            continue
        sch = ['BE'] if a['name'].endswith('m128') else ['BE', 'CNR']
        arms.append(dict(a, schemes=sch, cases='all38', q2_cases='sub16'))
    write(f'acc{mesh}', dict(mesh=mesh, edges=EDGES, arms=arms))


def time(mesh, subjects_filter, schemes, cases, reps, K_out=()):
    R = rules()
    subj = []
    for a in arm_specs(R, m128=False):
        if not subjects_filter(a) or any(a['name'].startswith(f'H{K}s') for K in K_out):
            continue
        for sc in schemes:
            subj.append(dict({k: v for k, v in a.items() if k not in ('rule_group', 'rule_status')}, mesh=mesh, scheme=sc))
    meshes = {str(mesh): dict(edges=EDGES)}
    if mesh != 1024:
        meshes['1024'] = dict(edges=[0, 128, 256, 384])
    write(f'time{mesh}', dict(meshes=meshes, A=A_SPEC, subjects=subj, cases=cases, cert_cases='dev6', reps=reps, burn=0.1,
                             seed=20261018 + mesh, doubled=[dict(subj[0], cases=[['dev6', 0]])]))


def production(which):
    if which == 'acc1024':
        acc(1024)
    elif which == 'acc256':
        acc(256, m128=False)
    elif which == 'time1024':
        time(1024, lambda a: True, ['BE', 'CNR'], 'all38', {'dev6': 3, 'val32': 1})
    elif which == 'time256':
        time(256, lambda a: a['kind'] == 'head' or a['Rp'] <= 32, ['BE'], 'dev6', {'dev6': 3})
    elif which == 'time256full':
        time(256, lambda a: True, ['BE'], 'dev6', {'dev6': 3})
    elif which == 'time4096':
        time(4096, lambda a: a['name'] in ('Hinc', 'H4s0', 'H8s0', 'H16s0', 'H32s0'), ['BE'], 'dev6', {'dev6': 3})


if __name__ == '__main__':
    if sys.argv[1] in ('smoke', 'train', 'rule1024'):
        {'smoke': smoke, 'train': train, 'rule1024': rule1024}[sys.argv[1]]()
    else:
        production(sys.argv[1])
