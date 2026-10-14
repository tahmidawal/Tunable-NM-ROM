"""Budget configs from the committed frozen selection (DESIGN A1.0 step 6, A1.5, A8.3). The configs carry the sha256 of
checks/FROZEN-SELECTION.json; stage.py re-verifies it against the file at the staged commit.

    python make_budget_configs.py [--only 2d|3d]
"""
import argparse
import hashlib
import json
from pathlib import Path

import make_configs as MC

HERE = Path(__file__).resolve().parent
FINER = {'gl56': 'gl64', 'gl64': 'gl80', 'gl80': 'gl96', 'gl96': 'gl112'}      # A11 successors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=('2d', '3d'))
    a = ap.parse_args()
    fp = HERE / 'checks' / 'FROZEN-SELECTION.json'
    fz = json.loads(fp.read_text())
    fsha = hashlib.sha256(fp.read_bytes()).hexdigest()
    if a.only in (None, '2d') and '2d' in fz:
        rules = {s['name']: s for s in MC.SETTINGS2}
        sets = [dict(rules[n], arms=b['arms']) for n, b in fz['2d']['budget'].items()]
        c = MC.c2('c2bud', settings=sets)
        c.update(driver='c2bud', conv_rule='gauss320', check_rule='gauss640', successor_check='gauss768',
                 control_rule='gauss8', conv_bar=2.5e-5, check_val32=10, frozen_sha256=fsha)
        for k in ('grid', 'fom', 'timing', 'common_baseline'):
            c.pop(k, None)
        (HERE / 'configs' / 'c2bud.json').write_text(json.dumps(c, indent=1) + '\n')
        print('c2bud', len(sets), 'settings')
    if a.only in (None, '3d') and '3d_solver' in fz and '3d_rules' in fz:
        base = {s['name']: s for s in MC.SETTINGS3}
        sets = []
        for n, arms in fz['3d_solver']['budget'].items():
            r = fz['3d_rules']['settings'][n]
            if r['deployed'] is None or r['converged'] is None:
                continue
            sets.append(dict(base[n], rule=r['deployed'], conv=r['converged'], finer=FINER[r['converged']], arms=arms))
        c = MC.c3('c3bud', 'bud', sets, timing=dict(cases=0, reps=0, burn=0.1, seed=20261013))
        c.update(budget_cases=32, control_rule='lat256', frozen_sha256=fsha,
                 rule_selection_label=fz['3d_rules'].get('label', 'pre-registered'), rule_selection_posthoc=fz['3d_rules'].get('posthoc', False))
        c.pop('grid', None)
        (HERE / 'configs' / 'c3bud.json').write_text(json.dumps(c, indent=1) + '\n')
        print('c3bud', len(sets), 'settings')


if __name__ == '__main__':
    main()
