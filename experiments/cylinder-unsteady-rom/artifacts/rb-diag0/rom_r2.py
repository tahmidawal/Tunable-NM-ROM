"""G5 R2: projection floors on the development trajectory (DESIGN-G5 §5.4).

Usage: rom_r2.py --export <dir> --case <dev name> --out R2-FLOORS.json
Applies the predeclared floor rule (median <= 5e-3 and max <= 1e-2, velocity, basis only)
and the R3 rank rule (smallest POD r in {8,12,16,20,24} meeting it, else 24).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rom_core as rc

RANKS = [2, 4, 8, 12, 16, 20, 24]
MEDIAN, MAXIMUM = 5e-3, 1e-2


def summary(a):
    a = np.asarray(a)
    return {'median': float(np.median(a)), 'max': float(a.max()), 'min': float(a.min()), 'count': int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True); ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ex = rc.Export(a.export)
    proj = dict(np.load(Path(a.export) / f'proj_{a.case}.npz'))
    res = {'case': a.case, 'rule': {'median_max': MEDIAN, 'max_max': MAXIMUM, 'metric': 'velocity_basis_only'}, 'arms': {}}
    arms = [('frozen_bank_r4', 'bank', 4, 'frozen')] + [(f'pod_r{r}', 'pod', r, None) for r in RANKS]
    for name, kind, r, key in arms:
        arm = rc.Arm(ex, kind, r, key)
        fl = arm.floors(proj)
        rec = {k: summary(v) for k, v in fl.items()}
        rec['passes_floor'] = rec['velocity_basis_only']['median'] <= MEDIAN and rec['velocity_basis_only']['max'] <= MAXIMUM
        rec['retained_dimension'] = arm.retained; rec['raw_dimension'] = arm.raw_count
        res['arms'][name] = rec
        np.savez(Path(a.out).with_name(f'R2-series-{name}.npz'), t=proj['t'], **fl)
    res['frozen_bank_adequate'] = res['arms']['frozen_bank_r4']['passes_floor']
    passing = [r for r in [8, 12, 16, 20, 24] if res['arms'][f'pod_r{r}']['passes_floor']]
    res['R3_required'] = not res['frozen_bank_adequate']
    res['R3_rank'] = passing[0] if passing else 24
    res['R3_rank_reason'] = 'smallest POD rank meeting the floor' if passing else 'no tested POD rank meets the floor; 24 by rule'
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: (v['velocity_basis_only'], v['passes_floor']) for k, v in res['arms'].items()}, indent=1))
    print('R3 required:', res['R3_required'], 'rank', res['R3_rank'])


if __name__ == '__main__':
    main()
