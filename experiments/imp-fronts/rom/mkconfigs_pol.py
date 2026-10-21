"""imp-fronts §B5 production policy configs (one per bank; DESIGN B5, Amendments 2-5).

    python mkconfigs_pol.py <label> [--schedule S1|S2]      -> configs/pol<label>.json
<label> = 1 (K1; selected rule gauss160 from jcp-robust E2) or an arm label whose select2d output
checks/select2d-<label>.json (written by the unchanged select2d.py) gives the selected R384 rule; no certified rule ->
selected = null (uncertified start fib4181, Amendment 4). Cases: dev/val j = 0..3 of every fresh cohort + the schedule's
test cases (lowest test indices first, Amendment 1 B3): S1 FI 8, FN0s 8, FC1 8, FC2 4; S2 FI 6, FN0s 8, FC1 8, FC2 2.
"""
import json
import sys
from pathlib import Path

import bankguard as BG

HERE = Path(__file__).resolve().parent
TEST = {'S1': dict(FI=8, FN0s=8, FC1=8, FC2=4), 'S2': dict(FI=6, FN0s=8, FC1=8, FC2=2)}
K1_SELECTED = 'gauss160'          # jcp-robust E2 procedure-selected R384 rule of seed 1 (DESIGN B1)


def selected_of(label):
    if label == '1':
        return K1_SELECTED
    s = json.loads((HERE / 'checks' / f'select2d-{label}.json').read_text())['selected']['384']
    return s['rule'] if s['outcome'] == 'selected' else None


def main():
    label = sys.argv[1]
    sch = sys.argv[sys.argv.index('--schedule') + 1] if '--schedule' in sys.argv else 'S1'
    BG.verify({'dir': f'inputs/banks/s{label}'}, HERE)
    cases = [f'{c}|{j}' for c in ('FI', 'FN0s', 'FC1', 'FC2') for j in range(4 + TEST[sch][c])]
    c = dict(attempt=f'pol{label}', driver='fpolicy', mesh=1024, edges=[0, 128, 256, 384, 512], Rp=384, scheme='CNR',
             bank={'dir': f'inputs/banks/s{label}'}, bank_label=label, selected=selected_of(label), cases=cases,
             table='phaseb/fresh_cohorts.json', test_once=True, anchors=None, ceiling_hours=.4, stop_fraction=.9,
             max_rounds=10, controls={}, schedule=sch)
    p = HERE / 'configs' / f'pol{label}.json'
    p.write_text(json.dumps(c, indent=1) + '\n')
    print(p, len(cases), 'cases, start', c['selected'])


if __name__ == '__main__':
    main()
