"""Independent, restricted NumPy audit of one panel output directory (DESIGN.md section 5.9).

* regenerates every case's u0 with NumPy (ns3d_fom.initial_raw is pure NumPy; the Leray
  projection is re-implemented here with numpy.fft) and checks the saved truth's t=0;
* recomputes every saved full-field error exactly (<= 1e-9 relative to the job) and a
  sampled estimate of every other case's error (reported; loose bound 20 %);
* recomputes every timing median from the raw repetitions, the FOM rule, every speedup and
  the drift / order gate values;
* must reject a copy with one saved field perturbed by 1e-6 relative.
Usage: audit_panel.py --out <panel output dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'ns3d'))


def initial_raw(n, parameter):  # verbatim NumPy logic of ns3d_fom.initial_raw (amplitude 0.2)
    x = np.arange(n, dtype=np.float64) / n
    xyz = np.stack(np.meshgrid(x, x, x, indexing='ij'))
    c, sep, strength = parameter[:3], parameter[3], parameter[4]
    offset = np.array([1., 1., 1.]) / np.sqrt(3)
    directions = (np.array([0., 0., 1.]), np.array([1., 1., 0.]) / np.sqrt(2))
    u = np.zeros((3, n, n, n))
    for side, weight, direction in ((-1, 1., directions[0]), (1, strength, directions[1])):
        center = c + side * .5 * sep * offset
        angle = 2 * np.pi * (xyz - center[:, None, None, None])
        scalar = np.exp(2 * np.sum(np.cos(angle) - 1, axis=0))
        grad = -4 * np.pi * np.sin(angle) * scalar
        u += weight * np.moveaxis(np.cross(np.moveaxis(grad, 0, -1), direction), -1, 0)
    return 0.2 * u


def leray(u):
    n = u.shape[-1]
    kk = np.fft.fftfreq(n, d=1.0 / n)
    modes = np.stack(np.meshgrid(kk, kk, kk, indexing='ij'))
    k = 2 * np.pi * modes
    k2 = np.sum(k * k, axis=0)
    mask = np.all(np.abs(modes) < n / 3, axis=0)
    mask[0, 0, 0] = False
    uh = np.fft.fftn(u, axes=(-3, -2, -1))
    dot = np.sum(k * uh, axis=0)
    uh = (uh - k * (dot / np.where(k2 > 0, k2, 1.0))) * mask
    return np.fft.ifftn(uh, axes=(-3, -2, -1)).real


def rel(pred, ref, u0n):
    d = (pred - ref).reshape(len(pred), -1)
    return np.sqrt(np.sum(d * d, axis=1)) / u0n


def audit(out, perturb=False):
    s = json.loads((out / 'summary.json').read_text())
    fdir = out / 'fields'
    tz = np.load(fdir / 'truth.npz')
    tsamp = np.load(fdir / 'truth_samples.npy')
    sidx = np.load(fdir / 'sample_index.npy')
    params = tz['parameters']
    n = int(s['config']['n'])
    ncase = len(params)
    fails = []
    # u0 regenerated independently
    u0n = np.empty(ncase)
    u0_gap = 0.0
    tcase = list(tz['cases'])
    for c in range(ncase):
        u0 = leray(initial_raw(n, params[c]))
        u0n[c] = np.sqrt(np.sum(u0 * u0))
        u0_gap = max(u0_gap, float(np.max(np.abs(u0.reshape(3, -1)[:, sidx] - tsamp[c, 0])) / np.max(np.abs(u0))))
        if c in tcase:
            full = tz['fields'][tcase.index(c)][0]
            u0_gap = max(u0_gap, float(np.max(np.abs(full - u0)) / np.max(np.abs(u0))))
    if u0_gap > 1e-12:
        fails.append(f'u0 regeneration gap {u0_gap}')
    norm_gap = float(np.max(np.abs(u0n - tz['u0_norm']) / u0n))
    if norm_gap > 1e-12:
        fails.append(f'u0 norm gap {norm_gap}')
    arms = {}
    scale = np.sqrt(n ** 3 / len(sidx))
    for name, r in s['results'].items():
        if not r['finite']:
            continue
        z = np.load(fdir / f'{name}.npz')
        job = np.asarray(r['errors'])
        exact_gap = 0.0
        for i, c in enumerate(z['cases']):
            f = z['fields'][i].copy()
            if perturb and i == 0:
                f[3] *= 1 + 1e-6
            e = rel(f, tz['fields'][tcase.index(int(c))], u0n[c])
            exact_gap = max(exact_gap, float(np.max(np.abs(e - job[c]) / np.maximum(np.abs(job[c]), 1e-12))))
        est_gap = 0.0
        for c in range(ncase):
            est = scale * np.sqrt(np.sum((z['samples'][c] - tsamp[c]).reshape(6, -1) ** 2, axis=1)) / u0n[c]
            big = job[c, 1:] > 1e-6
            if big.any():
                est_gap = max(est_gap, float(np.max(np.abs(est[1:][big] - job[c, 1:][big]) / job[c, 1:][big])))
        worst = float(job[:, 1:].max())
        arms[name] = dict(exact_cases=[int(c) for c in z['cases']], exact_max_relative_gap=exact_gap,
                          sampled_max_relative_gap=est_gap, evolved_worst_recomputed=worst,
                          evolved_worst_matches=bool(abs(worst - r['stats']['evolved_worst']) <= 1e-15))
        if exact_gap > 1e-9:
            fails.append(f'{name}: exact error gap {exact_gap}')
        if not arms[name]['evolved_worst_matches']:
            fails.append(f'{name}: evolved worst mismatch')
    # timing, FOM rule, speedups, gates
    t = s['timing']
    tim = {}
    for name, raw in t['raw'].items():
        allms = raw['A1'] + raw['B'] + raw['A2']
        med = float(np.median(allms))
        drift = float(np.median(raw['A2']) / np.median(raw['A1']))
        order = float(np.median(raw['B']) / np.median(raw['A1'] + raw['A2']))
        tim[name] = dict(median_ms=med, drift=drift, order=order)
        a = t['arms'][name]
        if abs(med - a['median_ms']) > 1e-9 * med or abs(drift - a['drift_ratio']) > 1e-12 or \
                abs(order - a['order_ratio']) > 1e-12:
            fails.append(f'{name}: timing recomputation mismatch')
        if min(allms) <= 0 or not np.all(np.isfinite(allms)):
            fails.append(f'{name}: nonpositive or nonfinite timing')
    acc = s['results']['nmrom_accurate_head_k8']['stats']['evolved_worst']
    cands = [nm for nm, r in s['results'].items() if r['kind'] == 'cnab2' and r['finite']
             and r['stats']['evolved_worst'] <= 1.0 and r['stats']['evolved_worst'] <= acc]
    fom = min(cands, key=lambda nm: tim[nm]['median_ms']) if cands else None
    if fom != s['fom_rule']['chosen']:
        fails.append(f'FOM rule mismatch: {fom} vs {s["fom_rule"]["chosen"]}')
    sp_gap = 0.0
    if fom:
        for nm, r in s['results'].items():
            if nm in tim and 'speedup_vs_fom' in r:
                sp = tim[fom]['median_ms'] / tim[nm]['median_ms']
                sp_gap = max(sp_gap, abs(sp - r['speedup_vs_fom']) / sp)
        if sp_gap > 1e-9:
            fails.append(f'speedup mismatch {sp_gap}')
    return dict(u0_regeneration_gap=u0_gap, u0_norm_gap=norm_gap, arms=arms, timing=tim, fom=fom,
                speedup_max_relative_gap=sp_gap, failures=fails, passed=not fails,
                note='restricted audit: exact on saved cases, sampled estimates elsewhere; truth not re-solved')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    res = audit(args.out)
    ctrl = audit(args.out, perturb=True)
    res['perturbed_control_rejected'] = not ctrl['passed']
    res['perturbed_control_failures'] = ctrl['failures'][:5]
    ok = res['passed'] and res['perturbed_control_rejected']
    res['all_passed'] = ok
    (args.out / 'audit.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(passed=res['passed'], control_rejected=res['perturbed_control_rejected'],
                          failures=res['failures'][:10], fom=res['fom'],
                          sampled_max=max((a['sampled_max_relative_gap'] for a in res['arms'].values()), default=0))))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
