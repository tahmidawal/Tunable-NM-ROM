"""jcp-speed3d selection (DESIGN H1.4, H2, H3, H4.3, A1.5, A1.7, A1.11, A1.13, A2.7; A4.5): from the screening analysis of
the grid jobs (analyze.py -> checks/analysis-grid.json) and the f1 FOM ladder, write
  checks/SELECTION-SCREEN.json   the screening selections (H2 winners W*, B* per comparator; H1.4 pairs) — validation only
  checks/rom-scores.json         E0 scores of the ROM configurations handed to the FOM matcher
  configs/p1.json                profile, mesh-free qualification, batching, confirmation panels, conv-rule sensitivity
  configs/m1.json                ROM vs FOM per mesh (L = 64, 128, 256, 384), matched FOMs from checks/fom-match.json
The freeze (checks/FROZEN-SELECTION.json) is written by freeze_s3.py only after p1 and m1 (A1.11).

    python select_s3.py --stage screen            # SELECTION-SCREEN.json + rom-scores.json + p1.json
    python select_s3.py --stage mesh              # m1.json (after fom_match.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import make_configs as MC  # noqa: E402

CONV = {'M2_512': 'gl56', 'W256': 'gl56', 'W512': 'gl56', 'W768': 'gl64', 'W1024': 'gl80'}
COMP = {'C_old': 'M2_512|gl32|fs|BE|0.01|K1', 'C_W768': 'W768|lat32768|fs|BE|0.01|K1', 'C_W1024': 'W1024|lat65536|fs|BE|0.01|K1'}


def split(key):
    p = key.split('|')
    return p[0], p[1], '|'.join(p[2:])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def screen():
    A = json.loads((HERE / 'checks' / 'analysis-grid.json').read_text())
    C = A['configs']
    labels = {}

    def lab(name, key):
        if key is None:
            return None
        for k, v in labels.items():
            if v == key:
                return k
        labels[name] = key
        return name
    sel = dict(analysis_sha256=sha(HERE / 'checks' / 'analysis-grid.json'), comparators={}, h14_pairs=[], note=(
        'screening selections on the validation cohort 923801 (common-baseline ratios); final verdicts only from the p1 '
        'confirmation panels (A1.5)'))
    for c, key in COMP.items():                         # comparators first: canonical labels (audit select-1)
        labels[c] = key
    for c, key in COMP.items():
        h = A['H2'].get(c, {})
        w = lab(f'W_{c}', h.get('winner'))
        b = lab(f'B_{c}', h.get('B_star'))
        sel['comparators'][c] = dict(key=key, winner=h.get('winner'), winner_label=w, B_star=h.get('B_star'), B_star_label=b,
                                     status=h.get('status'), S_screen=h.get('S_screen'), S_BE_screen=h.get('S_BE_screen'),
                                     winner_is_comparator=h.get('winner') == key, decomposition=h.get('decomposition'))
    for cell, v in A['H1']['H14'].items():
        if v['verdict'] in ('pass (screening)', 'fail (screening)'):
            q = lab(f'H14q_{cell}', v['qn'])
            f = lab(f'H14f_{cell}', v['fs2'])
            sel['h14_pairs'].append(dict(cell=cell, fs2=f, qn=q))
    sel['labels'] = labels
    (HERE / 'checks' / 'SELECTION-SCREEN.json').write_text(json.dumps(sel, indent=1) + '\n')
    # ROM scores for the FOM matcher: comparators, winners, B*
    roms = {}
    for c, v in sel['comparators'].items():
        for l_ in (c, v['winner_label'], v['B_star_label']):
            if l_ and l_ not in roms:
                cfg = C[labels[l_]]
                roms[l_] = dict(key=labels[l_], e=cfg['e'], eta=cfg['eta'], ranked_ids=cfg['ranked_ids'],
                                eligibility=cfg['eligibility'], selectable=cfg['selectable'])
    (HERE / 'checks' / 'rom-scores.json').write_text(json.dumps(roms, indent=1) + '\n')
    # p1
    configs = []
    for l_, key in labels.items():
        s, r, sp = split(key)
        configs.append(dict(label=l_, setting=s, rule=r, spec=sp))
    winners = [v['winner_label'] for v in sel['comparators'].values() if v['winner_label']]
    bstars = [v['B_star_label'] for v in sel['comparators'].values() if v['B_star_label']]
    fin = sorted(set(winners + bstars))
    confirm, identity = [], []
    for c, v in sel['comparators'].items():
        for l_ in (v['winner_label'], v['B_star_label']):
            if l_ and l_ == c:
                identity.append(dict(comparator=c, selected=l_, ratio=1.0, note='winner/B* is the comparator itself: ratio 1, H2 fails the bar'))
            elif l_ and [c, l_] not in confirm:
                confirm.append([c, l_])
        w_, b_ = v['winner_label'], v['B_star_label']
        if w_ and b_ and w_ != b_ and [b_, w_] not in confirm:   # A2.7(b): each decomposition factor from its own panel
            confirm.append([b_, w_])
    sel['identity_pairs'] = identity
    (HERE / 'checks' / 'SELECTION-SCREEN.json').write_text(json.dumps(sel, indent=1) + '\n')
    confirm += [[p['fs2'], p['qn']] for p in sel['h14_pairs']]
    settings = sorted({split(k)[0] for k in labels.values()})
    sets = []
    for s in settings:
        d = dict(MC.SETTINGS[s])
        d['extra'] = sorted({CONV[s], 'gl12'} | {split(k)[1] for k in labels.values() if split(k)[0] == s})
        sets.append(d)
    deployed = {s: MC.SETTINGS[s]['rule'] for s in settings}
    p1 = dict(MC.BASE, attempt='p1', mode='prof', settings=sets, timing=MC.TIMING, prof=dict(
        configs=configs, mf=sorted(set(winners + ['C_old', 'C_W1024']) & set(labels)), k5_rule='gl12',
        profile=sorted(set(winners + ['C_old', 'C_W1024']) & set(labels)),
        batch=dict(labels=sorted(set(winners + ['C_old']) & set(labels)), sizes=[1, 4, 16, 64]),
        confirm=confirm, confirm_cases=8, confirm_reps=3,
        conv={l_: CONV[split(labels[l_])[0]] for l_ in fin},
        reduced_adt=[l_ for l_ in fin if split(labels[l_])[1] != deployed[split(labels[l_])[0]]]))
    (HERE / 'configs' / 'p1.json').write_text(json.dumps(p1, indent=1) + '\n')
    print('screen:', json.dumps(sel['comparators'], indent=1))
    print('p1 configs', len(configs), 'confirm', confirm)


def mesh():
    sel = json.loads((HERE / 'checks' / 'SELECTION-SCREEN.json').read_text())
    fm = json.loads((HERE / 'checks' / 'fom-match.json').read_text())
    assert fm['K10_all_rejected'], 'K10 did not reject the synthetic FOM entries: no match may be used'
    labels = sel['labels']
    winners = [v['winner_label'] for v in sel['comparators'].values() if v['winner_label']]
    scores = json.loads((HERE / 'checks' / 'rom-scores.json').read_text())
    roms_l = sorted(set(winners + ['C_old', 'C_W1024']) & set(labels))
    roms_l = [l_ for l_ in roms_l if scores.get(l_, {}).get('selectable') is True]   # matches only for claim-admissible ROMs
    roms = [dict(label=l_, setting=split(labels[l_])[0], rule=split(labels[l_])[1], spec=split(labels[l_])[2]) for l_ in roms_l]
    matched = []
    for l_ in roms_l:
        for fam, m in fm['matches'].get(l_, {}).items():
            if m.get('matched'):
                matched.append([l_, m['matched']])
    settings = sorted({r['setting'] for r in roms})
    sets = [dict(MC.SETTINGS[s], extra=sorted({r['rule'] for r in roms if r['setting'] == s})) for s in settings]
    if 'W256' not in settings:
        sets.append(dict(MC.SETTINGS['W256']))                   # G5/K7 population (A1.9)
    m1 = dict(MC.BASE, attempt='m1', mode='mesh', settings=sets, timing=MC.TIMING, mesh=dict(
        # one job: H4.1 needs every mesh in one job (A1.9); L = 384 fits (pilot smkmesh384)
        Ls=[64, 128, 256, 384], roms=roms, same=[['UB', 0.01], ['CC', 0.01]], matched=matched,
        calib_small=[0, 1, 2, 3], calib_large=[0, 1, 2, 3], acc_cases=8),   # A1.9 / A2.5: four calibration cases at every L (A9)
        selection_sha256=sha(HERE / 'checks' / 'SELECTION-SCREEN.json'), fom_match_sha256=sha(HERE / 'checks' / 'fom-match.json'))
    (HERE / 'configs' / 'm1.json').write_text(json.dumps(m1, indent=1) + '\n')
    print('m1 roms', roms_l, 'matched', matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'mesh'], required=True)
    a = ap.parse_args()
    screen() if a.stage == 'screen' else mesh()


if __name__ == '__main__':
    main()
