"""jcp-speed3d freeze (DESIGN A1.11): after p1 and m1 are analysed (checks/analysis-pm.json), write ONE immutable
checks/FROZEN-SELECTION.json holding every validation-selected decision, and the replication config configs/r1.json
(historical test cohort 923901 = E0 test32, 32 cases; replication only, nothing is selected on it).

Frozen: the comparators, H2 winners and B* (with their confirmed verdicts), the H1.4 pairs, the H3.2 mesh-free
qualifications, the matched FOM configurations (fom-match.json) and the sha256 of every input. r1 re-runs, on 923901:
every frozen ROM configuration on the mesh path (and on the mesh-free path where H3.2 qualified it) with its ADT tight
comparator, every matched FOM, and one paired timing panel per comparator (A = comparator, B = its winner; 8 cases x 3).
Refuses to overwrite an existing freeze.

    python freeze_s3.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import make_configs as MC

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def split(key):
    p = key.split('|')
    return p[0], p[1], '|'.join(p[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    fz = HERE / 'checks' / 'FROZEN-SELECTION.json'
    assert not fz.exists(), 'the freeze is immutable; it already exists'
    ins = {n: HERE / 'checks' / n for n in ('analysis-grid.json', 'SELECTION-SCREEN.json', 'fom-match.json', 'analysis-pm.json')}
    sel = json.loads(ins['SELECTION-SCREEN.json'].read_text())
    pm = json.loads(ins['analysis-pm.json'].read_text())
    fm = json.loads(ins['fom-match.json'].read_text())
    assert 'p1' in pm and 'm1' in pm, 'freeze only after both p1 and m1 are analysed (A1.11)'
    assert pm['p1']['acceptance']['accepted_for_claims'] and pm['m1']['acceptance']['accepted_for_claims'], 'p1/m1 not accepted (A10)'
    labels = sel['labels']
    keep = set()
    for c, v in sel['comparators'].items():
        keep |= {c} | {x for x in (v['winner_label'], v['B_star_label']) if x}
    h14 = {l_ for p_ in sel['h14_pairs'] for l_ in (p_['fs2'], p_['qn'])}      # H1 selections: accuracy replication only
    mf_ok = {l_ for l_, v in pm['p1']['H32'].items() if v.get('qualified')}
    roms = []
    for l_ in sorted(keep | h14):
        s, r, sp = split(labels[l_])
        roms.append(dict(label=l_, setting=s, rule=r, spec=sp, paths=['mesh'] + (['mf'] if l_ in mf_ok else []),
                         role='H2' if l_ in keep else 'H1.4'))
    foms = sorted({m['matched'] for l_ in keep for m in fm['matches'].get(l_, {}).values() if m.get('matched')})
    confirm = []
    for c, v in sel['comparators'].items():
        w = v['winner_label']
        if w and w != c:
            confirm.append([c, w])
    frozen = dict(note='immutable validation-selected decisions (A1.11); replication on 923901 only; nothing selected on it',
                  inputs={n: sha(p) for n, p in ins.items()},
                  comparators={c: dict(key=v['key'], winner=v['winner'], winner_label=v['winner_label'], B_star=v['B_star'],
                                       B_star_label=v['B_star_label'], H2=pm['p1']['H2'][c]) for c, v in sel['comparators'].items()},
                  h14=pm['p1']['H14'], mesh_free_qualified=sorted(mf_ok),
                  rom_configs=[dict(label=x['label'], setting=x['setting'], rule=x['rule'], spec=x['spec']) for x in roms],
                  fom_configs=foms, fom_matches={l_: fm['matches'][l_] for l_ in sorted(keep) if l_ in fm['matches']})
    fbytes = (json.dumps(frozen, indent=1) + '\n').encode()
    settings = sorted({x['setting'] for x in roms})
    sets = [dict(MC.SETTINGS[s], extra=sorted({x['rule'] for x in roms if x['setting'] == s})) for s in settings]
    r1 = dict(MC.BASE, attempt='r1', mode='rep', cohort_seed=923901, cohort_count=32, settings=sets, timing=MC.TIMING,
              frozen_sha256=hashlib.sha256(fbytes).hexdigest(), rep=dict(roms=roms, foms=foms, confirm=confirm, confirm_cases=8, confirm_reps=3))
    rp = HERE / 'configs' / 'r1.json'
    rp.with_suffix('.tmp').write_text(json.dumps(r1, indent=1) + '\n')
    fz.with_suffix('.tmp').write_bytes(fbytes)
    rp.with_suffix('.tmp').replace(rp)                 # both written fully before either is in place
    fz.with_suffix('.tmp').replace(fz)
    print('frozen', fz, 'roms', [x['label'] for x in roms], 'foms', foms, 'confirm', confirm)


if __name__ == '__main__':
    main()
