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
        # three jobs (resource certificate: <= 48 h each with the worst-case conditional successor path priced in)
        groups = dict(c2budA=['R512', 'R256'], c2budB=['R384', 'acc'], c2budC=['fast', 'R128'])
        assert sorted(sum(groups.values(), [])) == sorted(fz['2d']['budget']), 'every frozen setting in exactly one job'
        for att, names in groups.items():
            sets = [dict(rules[n], arms=fz['2d']['budget'][n]['arms']) for n in names]
            c = MC.c2(att, settings=sets)
            c.update(driver='c2bud', conv_rule='gauss320', check_rule='gauss640', successor_check='gauss768',
                     control_rule='gauss8', conv_bar=2.5e-5, check_val32=10, frozen_sha256=fsha)
            for k in ('grid', 'fom', 'timing', 'common_baseline'):
                c.pop(k, None)
            (HERE / 'configs' / f'{att}.json').write_text(json.dumps(c, indent=1) + '\n')
            print(att, names)
    if a.only in (None, '2d') and '2d' in fz:
        # A14: conditional fallback budget for R256 (frozen first fallback gauss128), same arms as R256's frozen budget
        b = fz['2d']['budget']['R256']
        fb = fz['2d']['per_scheme_setting']                   # (recorded for provenance only)
        sets = [dict(name='R256fb', Rp=256, rule='gauss128', arms=b['arms'], fallback_of=dict(setting='R256', rule=b['rule']))]
        c = MC.c2('c2fbbud', settings=sets)
        c.update(driver='c2bud', conv_rule='gauss320', check_rule='gauss640', successor_check='gauss768',
                 control_rule='gauss8', conv_bar=2.5e-5, check_val32=10, frozen_sha256=fsha)
        for k in ('grid', 'fom', 'timing', 'common_baseline'):
            c.pop(k, None)
        (HERE / 'configs' / 'c2fbbud.json').write_text(json.dumps(c, indent=1) + '\n')
        print('c2fbbud', [x['name'] for x in sets], 'fallback of R256', b['rule'])
        del fb
    if a.only in (None, '3d') and '3d_solver' in fz and '3d_rules' in fz:
        base = {s['name']: s for s in MC.SETTINGS3}
        sets = []
        for n, arms in fz['3d_solver']['budget'].items():
            r = fz['3d_rules']['settings'][n]
            if r['deployed'] is None or r['converged'] is None:
                continue
            sets.append(dict(base[n], rule=r['deployed'], conv=r['converged'], finer=FINER[r['converged']], arms=arms))
        # two jobs (resource certificate <= 48 h each): W1024 alone (r_inf Gauss 80^3, finer 96^3), the other four together
        groups = dict(c3budW1024=['W1024'], c3budA=['M2_512', 'W256', 'W512', 'W768'])
        assert sorted(sum(groups.values(), [])) == sorted(x['name'] for x in sets), 'every 3D budget setting in exactly one job'
        for att, names in groups.items():
            c = MC.c3(att, 'bud', [x for x in sets if x['name'] in names], timing=dict(cases=0, reps=0, burn=0.1, seed=20261013))
            c.update(budget_cases=32, control_rule='lat256', frozen_sha256=fsha,
                     rule_selection_label=fz['3d_rules'].get('label', 'pre-registered'),
                     rule_selection_posthoc=fz['3d_rules'].get('posthoc', False))
            c.pop('grid', None)
            (HERE / 'configs' / f'{att}.json').write_text(json.dumps(c, indent=1) + '\n')
            print(att, names)


if __name__ == '__main__':
    main()
