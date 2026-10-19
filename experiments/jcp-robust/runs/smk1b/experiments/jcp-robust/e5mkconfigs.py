"""Write the E5 job configs from the copied sources (no hand-typed numbers): configs/e5smk4.json (A100 smoke),
configs/e5prod_a100.json, configs/e5prod_h200.json (DESIGN 4, A1.11, A2.8, A2.9 smk4, A3.8).

Sources (copies in e5/, see e5/PROVENANCE.json): P1's configs/time.json (subject specs incl. Q2 roles, mesh edges, A,
seed, burn) -> P-cost; fom2d's configs/c21024.json (A, ROM specs with target_e, FOM ladder_e, edges, seed, burn) and
FROZEN-SELECTION-1024.json (pinned comparators) -> P-cross.

    python e5mkconfigs.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE / 'e5'
FAMS = (dict(Rp=128, rule='fib4181', policy='frozen', scheme='BE'),          # R128 LSPG-BE dt0 fib4181 frozen
        dict(Rp=384, rule='gauss128', policy='frozen', scheme='CNR'))        # R384 LSPG-CN-R dt0 gauss128 frozen
ROM_LABELS = ('ckpt:18f0266ae6f0|R128|fib4181|frozen|BE|0.005|LSPG|prod',
              'ckpt:18f0266ae6f0|R384|gauss128|frozen|CNR|0.005|LSPG|prod')
SUFFIXES = ('|H|union|joint', '|H|CC|same_mesh')                             # A2.8 (CC|joint not used)
STORE = '/cluster/tufts/paralab/tawal01/jcp-ref/store'


def fam_match(sp, f):
    return all(sp.get(k, 'frozen' if k == 'policy' else None) == v for k, v in f.items())


def pcost(t, meshes, cases, reps):
    subs = [s for s in t['subjects'] if int(s['mesh']) in meshes and any(fam_match(s, f) for f in FAMS)]
    assert len(subs) == 2 * len(meshes), subs
    A = t['A']
    assert (A['mesh'], A['Rp'], A['rule'], A['policy'], A['scheme']) == (1024, 384, 'gauss128', 'frozen', 'CNR')
    r128_1024 = [s for s in subs if s['mesh'] == 1024 and s['Rp'] == 128][0]
    dbl = [dict(r128_1024, cases=cases), dict(A, cases=cases)]           # C-E5.1: R128 BE 1024^2 (DESIGN 4.4) and A
    return dict(meshes={str(m): t['meshes'][str(m)] for m in meshes}, A=A, subjects=subs, cases=cases, reps=reps,
                burn=t['burn'], seed=t['seed'], doubled=dbl)


def pcross(c2, fz, fz_sha, cases, reps):
    roms = [r for r in c2['roms'] if r['label'] in ROM_LABELS]
    assert len(roms) == 2, [r['label'] for r in roms]
    roms = sorted(roms, key=lambda r: ROM_LABELS.index(r['label']))
    exp = {lab: {suf: fz['selection'][f'{lab}@1024{suf}']['selected'] for suf in SUFFIXES} for lab in ROM_LABELS}
    fkeys = sorted({k for v in exp.values() for k in v.values()})
    return dict(mesh=1024, edges=c2['edges'], cases=cases, A=c2['A'], roms=roms,
                foms={k: dict(ladder_e=c2['foms'][k]['ladder_e']) for k in fkeys},
                frozen_selection=dict(path='e5/FROZEN-SELECTION-1024.json', sha256=fz_sha,
                                      source='jcp-fom2d runs/c21024 (job 5456289, H200) output/c2time_c21024'),
                selection_suffixes=list(SUFFIXES), expected_comparators=exp, reps=reps, burn=c2['burn'], seed=c2['seed'],
                controls=dict(doubled_cases=[['dev6', 0], ['dev6', 1]]),
                label='frozen H200-selected comparators (not the best comparator per GPU)')


def main():
    t = json.loads((E5 / 'p1_time.json').read_text())
    c2 = json.loads((E5 / 'c21024.json').read_text())
    fzb = (E5 / 'FROZEN-SELECTION-1024.json').read_bytes()
    fz = json.loads(fzb)
    fz_sha = hashlib.sha256(fzb).hexdigest()
    dev5 = [['dev6', i] for i in range(5)]
    dev6 = [['dev6', i] for i in range(6)]
    x16 = dev6 + [['val32', i] for i in range(10)]
    jobs = {
        'e5smk4': dict(gpu_type='A100-80G', meshes=(256, 1024), pc_cases=dev5, x_cases=dev5, reps=2),
        'e5prod_a100': dict(gpu_type='A100-80G', meshes=(256, 1024), pc_cases='dev6', x_cases=x16, reps=3),
        'e5prod_h200': dict(gpu_type='H200', meshes=(256, 1024, 4096), pc_cases='dev6', x_cases=x16, reps=3),
    }
    (HERE / 'configs').mkdir(exist_ok=True)
    for att, j in jobs.items():
        cfg = dict(attempt=att, driver='e5time', gpu_type=j['gpu_type'], allow_4096=4096 in j['meshes'],
                   jcpref_store=STORE,
                   pcost=pcost(t, j['meshes'], j['pc_cases'], j['reps']),
                   pcross=pcross(c2, fz, fz_sha, j['x_cases'], j['reps']),
                   cross_type_meshes=[256, 1024],
                   note=('4096^2 subjects are H200-only, separately labelled, never in a cross-type intersection'
                         if 4096 in j['meshes'] else 'two-mesh P-cost panel (256^2, 1024^2)'))
        p = HERE / 'configs' / f'{att}.json'
        p.write_text(json.dumps(cfg, indent=1) + '\n')
        print(p.relative_to(HERE), hashlib.sha256(p.read_bytes()).hexdigest()[:16])


if __name__ == '__main__':
    main()
