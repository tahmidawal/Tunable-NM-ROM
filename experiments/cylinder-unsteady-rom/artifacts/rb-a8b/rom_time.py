"""Rebuttal push: whole-window ROM timings with spread, for the accuracy-matched cost comparison.

Workload = the full recorded window of one case (same as one fe_ladder entry): reaction forces and
probe pressures every step, reduced state every step.  Per repetition: an untimed 20-step warm-up,
then cold (operator/tensor build + start projection + rollout) and warm (rollout only) wall times;
reconstruction of the velocity on the quadrature points at every 1/400 save is timed separately.
CPU, same allocation and thread count as the FE entries it is compared with.
Usage: rom_time.py --export E --manifest M --case C --dt DT --arms JSON --out FILE --reps N
       [--trained-bank npz --trained-proj npz]
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import rom_core as rc


def main():
    ap = argparse.ArgumentParser()
    for k in ('--export', '--manifest', '--case', '--arms', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--dt', type=float, required=True); ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--trained-bank'); ap.add_argument('--trained-proj')
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    nu = 0.1 / case['Re']
    ex = rc.Export(a.export)
    proj = dict(np.load(Path(a.export) / f'proj_{a.case}.npz'))
    z = None
    if a.trained_bank:
        z = np.load(a.trained_bank)
        ex.add_bank('trained', z['V'], z['G'], z['outlet_V'], {lab: (z[f'{lab}_V'], z[f'{lab}_G']) for lab in ('inlet', 'outlet')})
        tp = dict(np.load(a.trained_proj))
        if str(tp['bank_eval_sha256']) != hashlib.sha256(Path(a.trained_bank).read_bytes()).hexdigest():
            raise ValueError('trained-bank projections are not bound to this bank evaluation')
        proj['uv_trained'] = tp['uv_trained']
    ts = proj['t']; t0 = ts[0]
    idx = np.rint((ts - t0) / a.dt).astype(int)
    if np.max(np.abs(t0 + idx * a.dt - ts)) > 1e-9:
        raise ValueError('save times are not on the rollout grid')
    n_steps = int(idx[-1])
    res = {'case': a.case, 'dt': a.dt, 'steps': n_steps, 'reps': a.reps, 'arms': {}}
    for spec in json.loads(a.arms):
        key = spec.get('key')
        if spec['kind'] == 'bank' and key == 'trained' and spec['r'] < z['V'].shape[-1]:
            key = f"trained_p{spec['r']}"; n_ = spec['r']
            if key not in ex.bank:
                ex.add_bank(key, z['V'][..., :n_], z['G'][..., :n_], z['outlet_V'][..., :n_],
                            {lab: (z[f'{lab}_V'][..., :n_], z[f'{lab}_G'][..., :n_]) for lab in ('inlet', 'outlet')})
                proj[f'uv_{key}'] = proj['uv_trained'][:, :n_]
        rows = []
        for rep in range(a.reps):
            arm = rc.Arm(ex, spec['kind'], spec['r'], key)
            a0 = arm.ref_coeffs({k: v[:1] for k, v in proj.items() if k == 't' or k in ('uu', 'pp', 'pv', 'pref') or k.startswith('uv')})[0][0]
            arm.rollout(nu, a.dt, a0, 20)
            tc = time.perf_counter()
            arm = rc.Arm(ex, spec['kind'], spec['r'], key)
            a0 = arm.ref_coeffs({k: v[:1] for k, v in proj.items() if k == 't' or k in ('uu', 'pp', 'pv', 'pref') or k.startswith('uv')})[0][0]
            tw = time.perf_counter()
            A, C, F, P, its = arm.rollout(nu, a.dt, a0, n_steps)
            warm = time.perf_counter() - tw; cold = time.perf_counter() - tc
            tr_ = time.perf_counter()
            for k in idx:
                f = arm.lV + arm.Vb @ A[k]
            recon = time.perf_counter() - tr_
            err = arm.errors(proj, A, C, idx)
            rows.append({'warm': warm, 'cold': cold, 'reconstruction': recon, 'newton_iterations_total': int(its.sum()),
                         'e_u_rms': float(np.sqrt(np.mean(err['e_u'] ** 2))),
                         'checksum': hashlib.sha256(np.ascontiguousarray(A[idx]).tobytes()).hexdigest()})
        w = np.array([r['warm'] for r in rows])
        res['arms'][spec['name']] = {'velocity_dimension': arm.d, 'pressure_dimension': arm.dp, 'reps': rows,
                                     'warm_median': float(np.median(w)), 'warm_min': float(w.min()), 'warm_max': float(w.max()),
                                     'cold_median': float(np.median([r['cold'] for r in rows])),
                                     'reconstruction_median': float(np.median([r['reconstruction'] for r in rows]))}
        print(spec['name'], res['arms'][spec['name']]['warm_median'], w.min(), w.max(), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2) + '\n')


if __name__ == '__main__':
    main()
