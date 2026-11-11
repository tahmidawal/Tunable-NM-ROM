"""Rebuttal push (PROTOCOL.md amendment A6): whole-window ROM timings with spread, all steps in one process.

Workload = the full recorded window of one case (same as one fe_ladder entry): reaction forces and
probe pressures every step, reduced state every step.  Pure NumPy on the CPU (rom_core uses no GPU),
same allocation and thread count as the FE entries it is compared with.  Per arm:
  1. untimed warm-up: operator build + 20 steps;
  2. build_seconds: a second, timed operator/tensor build + start projection (the per-basis offline cost);
  3. for every step size, `reps` timed rollouts (warm = time stepping only, what rom_r4 calls rollout_seconds);
  4. field output, timed separately because the rollout itself never forms a field:
     - output_seconds: velocity (lift + basis @ a) and pressure (pressure modes @ c) at every 1/400 save on
       every `stride`-th quadrature point (about as many points as half the P2/P1 comparator's unknowns,
       i.e. at least its number of velocity nodes), repeated for every rollout;
     - full_quadrature_output_seconds: the same on all quadrature points, once per arm.
Usage: rom_time2.py --export E --manifest M --case C --dts "dt1 dt2 ..." --arms JSON --out FILE --reps N
       [--trained-bank npz --trained-proj npz] [--stride 21]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

import rom_core as rc


def main():
    ap = argparse.ArgumentParser()
    for k in ('--export', '--manifest', '--case', '--arms', '--out', '--dts'):
        ap.add_argument(k, required=True)
    ap.add_argument('--reps', type=int, default=5); ap.add_argument('--stride', type=int, default=21)
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
    sel = np.arange(0, len(ex.w), a.stride)
    cpu = next((ln.split(':', 1)[1].strip() for ln in open('/proc/cpuinfo') if ln.startswith('model name')), '')
    res = {'case': a.case, 'reps': a.reps, 'host': platform.node(), 'cpu_model': cpu,
           'threads': os.environ.get('OMP_NUM_THREADS'), 'backend': 'numpy (CPU); no GPU is used by the rollout',
           'saves': int(len(ts)), 'quadrature_points': int(len(ex.w)), 'output_points': int(len(sel)),
           'output_values_per_save': int(3 * len(sel)), 'arms': {}}
    first = {k: v[:1] for k, v in proj.items() if k == 't' or k in ('uu', 'pp', 'pv', 'pref') or k.startswith('uv')}

    def start(arm):
        return arm.ref_coeffs(first)[0][0]

    for spec in json.loads(a.arms):
        key = spec.get('key')
        if spec['kind'] == 'bank' and key == 'trained' and spec['r'] < z['V'].shape[-1]:
            key = f"trained_p{spec['r']}"; n_ = spec['r']
            if key not in ex.bank:
                ex.add_bank(key, z['V'][..., :n_], z['G'][..., :n_], z['outlet_V'][..., :n_],
                            {lab: (z[f'{lab}_V'][..., :n_], z[f'{lab}_G'][..., :n_]) for lab in ('inlet', 'outlet')})
                proj[f'uv_{key}'] = proj['uv_trained'][:, :n_]
                first[f'uv_{key}'] = proj[f'uv_{key}'][:1]
        arm = rc.Arm(ex, spec['kind'], spec['r'], key)
        arm.rollout(nu, float(a.dts.split()[0]), start(arm), 20)
        tb = time.perf_counter()
        arm = rc.Arm(ex, spec['kind'], spec['r'], key)
        a0 = start(arm)
        build = time.perf_counter() - tb
        Vs, lVs, Ps = np.ascontiguousarray(arm.Vb[sel]), np.ascontiguousarray(arm.lV[sel]), np.ascontiguousarray(arm.Pb[sel])
        rec = {'velocity_dimension': arm.d, 'pressure_dimension': arm.dp, 'build_seconds': build, 'steps': {}}
        for dts in a.dts.split():
            dt = float(dts)
            idx = np.rint((ts - t0) / dt).astype(int)
            if np.max(np.abs(t0 + idx * dt - ts)) > 1e-9:
                raise ValueError('save times are not on the rollout grid')
            n_steps = int(idx[-1])
            warm, outp, sums = [], [], set()
            for rep in range(a.reps):
                tw = time.perf_counter()
                A, C, F, P, its = arm.rollout(nu, dt, a0, n_steps)
                warm.append(time.perf_counter() - tw)
                to = time.perf_counter()
                acc = 0.0
                for k in idx:
                    u = lVs + Vs @ A[k]; p = Ps @ C[k]
                    acc += float(u[0, 0]) + float(p[0])
                outp.append(time.perf_counter() - to)
                sums.add(hashlib.sha256(np.ascontiguousarray(A[idx]).tobytes()).hexdigest())
            err = arm.errors(proj, A, C, idx)
            w = np.array(warm); o = np.array(outp)
            rec['steps'][dts] = {'dt': dt, 'n_steps': n_steps, 'warm': warm, 'warm_median': float(np.median(w)),
                                 'warm_min': float(w.min()), 'warm_max': float(w.max()),
                                 'output': outp, 'output_median': float(np.median(o)), 'output_min': float(o.min()),
                                 'output_max': float(o.max()), 'newton_iterations_total': int(its.sum()),
                                 'e_u_rms': float(np.sqrt(np.mean(err['e_u'] ** 2))),
                                 'identical_across_reps': len(sums) == 1}
            if 'full_quadrature_output_seconds' not in rec:
                tf = time.perf_counter()
                for k in idx:
                    u = arm.lV + arm.Vb @ A[k]; p = arm.Pb @ C[k]
                rec['full_quadrature_output_seconds'] = time.perf_counter() - tf
            print(spec['name'], dts, rec['steps'][dts]['warm_median'], w.min(), w.max(), 'output', rec['steps'][dts]['output_median'], flush=True)
        res['arms'][spec['name']] = rec
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2) + '\n')


if __name__ == '__main__':
    main()
