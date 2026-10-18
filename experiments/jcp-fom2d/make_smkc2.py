"""jcp-fom2d C2 smoke config (DESIGN A1.5/A1.6): one H200 job at the largest mesh 4096^2 with the P1 reference A, one P1
target ROM (R128 fib4181 frozen LSPG-BE dt0, P1 4096^2 population includes dev6), the four FOM configurations of the
ladder smoke (their ladder errors on dev6 from the accepted smoke) and the combined-workload memory pilot (FD at L=4096,
full-field output, bank resident). Writes configs/smkc2.json and checks/smkc2-plan.json (c2analyze plan).

    python make_smkc2.py --smoke runs/smk1/archive/output/ladder_smk1
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TC = [['dev6', j] for j in range(6)]
ROM_T = 'dn256b|R128|fib4181|frozen|BE|0.005|LSPG|prod@4096|H'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', required=True)
    a = ap.parse_args()
    r = json.loads((Path(a.smoke) / 'result.json').read_text())
    assert r.get('accepted') is True, 'the ladder smoke is not accepted'
    T = json.loads((HERE / 'checks' / 'targets-p1.json').read_text())['targets'][ROM_T]
    foms = {}
    for k, v in r['configs'].items():
        foms[k] = dict(ladder_e={p['case']: p['scores']['ST']['e'] for p in v['production'] if p['case'] in [f'{c}|{j}' for c, j in TC]})
    keys = sorted(foms)
    cc = [k for k in keys if k.startswith('fd|CC')]
    rom = dict(label=T['identity'], Rp=128, rule='fib4181', policy='frozen', scheme='BE', dt=0.005, form='LSPG', level='prod',
               target_e={f'{c}|{j}': T['e'][f'{c}|{j}'] for c, j in TC})
    groups = [dict(id='smk|CC|joint', target=rom['label'], family='CC', criterion='joint', candidates=cc),
              dict(id='smk|union|joint', target=rom['label'], family='union', criterion='joint', candidates=keys, pool_by_family=True)]
    cfg = dict(attempt='smkc2', mesh=4096, edges=[0, 128, 256, 384, 512], cases=TC,
               A=dict(label='A', Rp=384, rule='gauss128', policy='frozen', scheme='CNR', dt=0.005, form='LSPG', level='prod'),
               roms=[rom], foms=foms, groups=groups, reps=2, burn=0.1, seed=4096,
               controls=dict(doubled_cases=[['dev6', 0], ['dev6', 1]], fault_subject=cc[0], identity_fault_subject=cc[0]),
               pilot_foms=['fd|UB|L4096|0.000625|t0', 'fd|CC|L4096|0.0025|t0', 'ps|N256|0.0005'])
    (HERE / 'configs' / 'smkc2.json').write_text(json.dumps(cfg, indent=1) + '\n')
    plan = dict(meshes={'4096': dict(targets={'smk': dict(rom_label=rom['label'], population='H', groups={
        'smk|CC|joint': dict(family='CC', criterion='joint'), 'smk|union|joint': dict(family='union', criterion='joint')},
        group_ids={'CC|joint': 'smk|CC|joint', 'union|joint': 'smk|union|joint'}, union_joint_group='smk|union|joint')},
        k2_universe=['smk'], k2_universe_complete=False)}, identities={rom['label']: dict(targets={'4096': 'smk'})},
        note='C2 smoke plan (mechanics only; no claim)')
    (HERE / 'checks' / 'smkc2-plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    print('foms', keys)


if __name__ == '__main__':
    main()
