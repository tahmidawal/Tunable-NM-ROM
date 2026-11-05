"""G5-A5 quadrature check: raw-basis reduced operators at two quadrature orders.

Usage: rom_qcheck.py --low <export order 12> --high <export order 14> --arms <json list> --out QCHECK.json
Raw basis = FE POD/supremizer coefficient modes (quadrature-independent) and fixed bank
functions, so no rotation/sign ambiguity enters. Criterion: relative Frobenius change
<= 1e-6 for every operator of every ranked arm (denominator: order-14 norm, floor 1e-14).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rom_core as rc

OPS = ('M', 'K', 'KL', 'B', 'BL', 'c0', 'C1', 'C2', 'f_m', 'f_k', 'f_kl', 'f_c0', 'f_b', 'f_C1', 'f_C2', 'Gp')
TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--low', required=True); ap.add_argument('--high', required=True)
    ap.add_argument('--arms', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--trained-low'); ap.add_argument('--trained-high')
    a = ap.parse_args()
    exl, exh = rc.Export(a.low), rc.Export(a.high)
    for ex, tb in ((exl, a.trained_low), (exh, a.trained_high)):
        if tb:
            z = np.load(tb); ex.add_bank('trained', z['V'], z['G'], z['outlet_V'])
    res = {'tolerance': TOL, 'arms': {}}
    for spec in json.loads(a.arms):
        al = rc.Arm(exl, spec['kind'], spec['r'], spec.get('key'), orthonormalize=False)
        ah = rc.Arm(exh, spec['kind'], spec['r'], spec.get('key'), orthonormalize=False)
        rel = {op: float(np.linalg.norm(getattr(al, op) - getattr(ah, op)) / max(np.linalg.norm(getattr(ah, op)), 1e-14))
               for op in OPS}
        res['arms'][spec['name']] = {'relative_change': rel, 'max': max(rel.values()), 'pass': max(rel.values()) <= TOL}
    res['pass'] = all(v['pass'] for v in res['arms'].values())
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: (v['max'], v['pass']) for k, v in res['arms'].items()}))


if __name__ == '__main__':
    main()
