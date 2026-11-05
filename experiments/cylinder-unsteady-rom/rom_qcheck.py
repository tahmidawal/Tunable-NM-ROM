"""G5-A5 quadrature check: raw-basis reduced operators at two quadrature orders.

Usage: rom_qcheck.py --low <export order 12> --high <export order 14> --arms <json list> --out QCHECK.json
Raw basis = FE POD/supremizer coefficient modes (quadrature-independent) and fixed bank
functions, so no rotation/sign ambiguity enters. Criteria (G5-A5 item 7): relative Frobenius change <= 1e-6 for every operator of every ranked
arm (denominator: order-14 norm, floor 1e-14); development floors (median, velocity basis
only) relative change <= 1e-3; R4 primary metrics (time-RMS e_u, e_p) relative change <= 1e-2.
Identity binding: identical FE POD/pressure/supremizer coefficient modes, identical frozen-bank
SHA, identical trained-parameter digest across the two rules; otherwise the check refuses.
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
    ap.add_argument('--case', required=True); ap.add_argument('--r4-low', required=True); ap.add_argument('--r4-high', required=True)
    ap.add_argument('--trained-proj-low'); ap.add_argument('--trained-proj-high')
    a = ap.parse_args()
    pl, ph = np.load(Path(a.low) / 'pod.npz'), np.load(Path(a.high) / 'pod.npz')
    for k in ('modes', 'pressure_modes', 'supremizers', 'lift'):
        if np.max(np.abs(pl[k] - ph[k])) > 1e-12 * max(np.max(np.abs(ph[k])), 1.0):
            raise ValueError(f'FE {k} differ between the two exports')
    jl, jh = dict(np.load(Path(a.low) / f'proj_{a.case}.npz')), dict(np.load(Path(a.high) / f'proj_{a.case}.npz'))
    if str(jl['bank_frozen_sha256']) != str(jh['bank_frozen_sha256']):
        raise ValueError('frozen bank differs')
    if a.trained_low:
        if str(np.load(a.trained_low)['params_digest']) != str(np.load(a.trained_high)['params_digest']):
            raise ValueError('trained parameters differ between the two evaluations')
        jl['uv_trained'] = np.load(a.trained_proj_low)['uv_trained']; jh['uv_trained'] = np.load(a.trained_proj_high)['uv_trained']
    exl, exh = rc.Export(a.low), rc.Export(a.high)
    for ex, tb in ((exl, a.trained_low), (exh, a.trained_high)):
        if tb:
            z = np.load(tb); ex.add_bank('trained', z['V'], z['G'], z['outlet_V'],
                                         {lab: (z[f'{lab}_V'], z[f'{lab}_G']) for lab in ('inlet', 'outlet')})
    res = {'tolerance': TOL, 'arms': {}}
    for spec in json.loads(a.arms):
        al = rc.Arm(exl, spec['kind'], spec['r'], spec.get('key'), orthonormalize=False)
        ah = rc.Arm(exh, spec['kind'], spec['r'], spec.get('key'), orthonormalize=False)
        rel = {op: float(np.linalg.norm(getattr(al, op) - getattr(ah, op)) / max(np.linalg.norm(getattr(ah, op)), 1e-14))
               for op in OPS}
        fl = rc.Arm(exl, spec['kind'], spec['r'], spec.get('key')).floors(jl)['velocity_basis_only']
        fh = rc.Arm(exh, spec['kind'], spec['r'], spec.get('key')).floors(jh)['velocity_basis_only']
        floor_rel = abs(np.median(fl) - np.median(fh)) / np.median(fh)
        r4l = json.loads(Path(a.r4_low).read_text())['arms'][spec['name']]
        r4h = json.loads(Path(a.r4_high).read_text())['arms'][spec['name']]
        metric_rel = {m: abs(r4l[m]['rms'] - r4h[m]['rms']) / r4h[m]['rms'] for m in ('e_u', 'e_p')} \
            if r4l.get('status') == r4h.get('status') == 'completed' else None
        ok = max(rel.values()) <= TOL and floor_rel <= 1e-3 and metric_rel is not None and max(metric_rel.values()) <= 1e-2
        res['arms'][spec['name']] = {'relative_change': rel, 'operator_max': max(rel.values()), 'floor_relative_change': float(floor_rel),
                                     'r4_metric_relative_change': metric_rel, 'pass': ok}
    res['pass'] = all(v['pass'] for v in res['arms'].values())
    res['consequence_if_fail'] = 'affected method ranking reported as unresolved (G5-A5 item 7)'
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: (v['operator_max'], v['floor_relative_change'], v['pass']) for k, v in res['arms'].items()}))


if __name__ == '__main__':
    main()
