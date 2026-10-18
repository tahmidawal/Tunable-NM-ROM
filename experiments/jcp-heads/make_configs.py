"""Generate the lane's job configs (DESIGN 8). Smoke configs now; production configs from the frozen rule selection
(runs/RULES.json, written by analyze.py after `rule1024`) later.

    python make_configs.py smoke
    python make_configs.py production      (needs runs/RULES.json and the trained heads in heads/)
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


if __name__ == '__main__':
    {'smoke': smoke}[sys.argv[1]]()
