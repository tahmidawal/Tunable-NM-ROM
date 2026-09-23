"""Independent NumPy/SciPy audit of one panel output directory (no JAX, no lane imports).

For every arm and every case it rebuilds the initial field from the saved draw, computes the
same-grid and continuum references with scipy.fft.dstn on the FULL grid, and
  (1) recomputes the strided sub-grid errors from the saved sub-grid field (must equal the
      in-job sub-grid errors to 1e-10 absolute);
  (2) for the random-node cases, estimates the full-grid error from the saved random-node field
      and compares it with the in-job full-grid error (relative gap <= 5 % for reduced arms,
      <= 10 % for operators/FOM/controls, skipped when the reported error is < 1e-9);
  (3) for reduced arms (smooth error) requires the sub-grid error to agree with the in-job
      full-grid error within 5 % relative;
and recomputes every per-arm statistic the summary reports (worst / median over cases of the
max over all times, from the in-job per-case vectors) plus the order-effect gate.
Output: JSON on stdout.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn, idstn

out = Path(sys.argv[1]); res = json.loads((out / 'results.json').read_text()); cfg = res['config']
n = res['mesh']; times = np.asarray(cfg['times']); nu = cfg['diffusivity']
REDUCED = ('nmrom', 'linear_bank', 'pod', 'qm')
failures, checked, worst = [], 0, dict(sub_abs=0.0, reduced_full_vs_sub=0.0, reduced_rand=0.0, other_rand=0.0)
k = np.arange(1, n); a = np.arange(1, n) / n
lam = {'same': 4.0 * n * n * np.sin(np.pi * k / (2 * n)) ** 2, 'physical': (np.pi * k) ** 2}
arms = res['arms']; ncases = len(next(iter(arms.values()))['same'])
recomputed = {}; rand_est = {}
expected_cases = sum(c for _, c in cfg['cohorts'])
for ci in range(ncases):
    draw = None; refs = None
    for name, row in arms.items():
        z = np.load(out / 'fields' / f'{name}__case{ci}.npz')
        if refs is None:
            draw = z['draw']; u0 = draw[3] * np.outer(4 * a * (1 - a) * np.exp(-(a - draw[0]) ** 2 / (2 * draw[2] ** 2)),
                                                       4 * a * (1 - a) * np.exp(-(a - draw[1]) ** 2 / (2 * draw[2] ** 2)))
            coef = dstn(u0, type=1, norm='ortho'); refs = {}
            for key, l in lam.items():
                L = l[:, None] + l[None, :]
                refs[key] = np.stack([u0] + [idstn(coef * np.exp(-nu * t * L), type=1, norm='ortho') for t in times[1:]])
        assert np.array_equal(z['draw'], draw), (name, ci)
        s = int(z['stride']); sub = (slice(None), slice(s - 1, None, s), slice(s - 1, None, s))
        fam = row['family']
        for key in ('same', 'physical'):
            ref = refs[key][sub]; pred = z['sub']
            e = np.sqrt(np.sum((pred - ref) ** 2, axis=(1, 2)) / np.sum(ref ** 2, axis=(1, 2)))
            d = float(np.max(np.abs(e - np.asarray(row[key + '_sub'][ci])))); checked += 1
            worst['sub_abs'] = max(worst['sub_abs'], d)
            if not d <= 1e-10:   # NaN-safe
                failures.append(dict(arm=name, case=ci, ref=key, check='subgrid_recompute', value=d))
            full = np.asarray(row[key][ci])
            if fam in REDUCED:
                m = full > 1e-9
                if not (np.isfinite(e).all() and np.isfinite(full).all()):
                    failures.append(dict(arm=name, case=ci, ref=key, check='nonfinite', value=None))
                gap = float(np.max(np.abs(e[m] - full[m]) / full[m])) if m.any() else 0.0
                worst['reduced_full_vs_sub'] = max(worst['reduced_full_vs_sub'], gap)
                if not gap <= 0.05:
                    failures.append(dict(arm=name, case=ci, ref=key, check='reduced_full_vs_subgrid', value=gap))
            if ci < cfg['random_audit_cases'] and 'rand' not in z.files:
                failures.append(dict(arm=name, case=ci, check='random_sample_missing', value=None))
            if 'rand' in z.files:
                rand_est.setdefault(name, {}).setdefault(key, {})[ci] = None
                idx = z['rand_idx']; refr = refs[key].reshape(len(times), -1)[:, idx]
                er = np.sqrt(np.sum((z['rand'] - refr) ** 2, 1) / np.sum(refr ** 2, 1))
                rand_est[name][key][ci] = er.tolist()
                m = full > 1e-9
                gap = float(np.max(np.abs(er[m] - full[m]) / full[m])) if m.any() else 0.0
                tol = 0.05 if fam in REDUCED else 0.10
                slot = 'reduced_rand' if fam in REDUCED else 'other_rand'; worst[slot] = max(worst[slot], gap); checked += 1
                if not gap <= tol:
                    failures.append(dict(arm=name, case=ci, ref=key, check='random_nodes_vs_full', value=gap, tolerance=tol))
for name, row in arms.items():
    same = np.asarray(row['same']); ms = np.asarray(row['device_ms'])
    if same.shape != (expected_cases, len(times)) or ms.shape != (expected_cases, cfg['repetitions']) or not np.isfinite(same).all():
        failures.append(dict(arm=name, check='coverage_or_finite', value=[list(same.shape), list(ms.shape)]))
    par = max(row.get('timed_vs_warm_max_relative', [np.inf]))
    ptol = 1e-4 if row['family'] == 'operator' else 1e-9   # f32 cuDNN kernels need not be bitwise deterministic
    if not par <= ptol:
        failures.append(dict(arm=name, check='timed_output_differs_from_audited_output', value=par))
    re_ = rand_est.get(name, {}).get('same', {})
    est = np.array([max(v) for v in re_.values()]) if re_ else np.array([np.nan])
    recomputed[name] = dict(worst_all_times=float(same.max()), median_case_max=float(np.median(same.max(1))),
                            worst_evolved=float(same[:, 1:].max()), device_ms_median=float(np.median(ms)),
                            physical_worst=float(np.max(row['physical'])), random_node_estimate_worst=float(np.max(est)),
                            random_node_estimate_median=float(np.median(est)), random_node_cases=int(len(re_)), timed_vs_warm_max_relative=par)
sent = np.array([float(np.median(s['ms'])) for s in res['sentinels']]); ref = float(np.median(sent))
if len(sent) < len(arms):
    failures.append(dict(check='sentinel_coverage', value=[len(sent), len(arms)]))
cont = np.array([c['ms'] for c in res['contaminated']])
order = dict(max_relative_deviation=float(np.max(np.abs(sent / ref - 1))), tolerance=cfg['order_tolerance'],
             positive_control_min_relative_deviation=float(np.min(cont / ref - 1)) if cont.size else None)
order['passed_v1_per_block'] = bool(order['max_relative_deviation'] <= cfg['order_tolerance'])   # pre-registered v1 test (A2: noise-dominated)
# A2 gate: (a) carry-over — pooled sentinel reps after full-order blocks vs after every other block;
#          (b) re-timed real arms after the full-order phase vs their own block medians.
fom_names = set(cfg['fom_order'])
post = np.concatenate([s['ms'] for s in res['sentinels'] if s['after'] in fom_names]) if fom_names else np.array([])
rest = np.concatenate([s['ms'] for s in res['sentinels'] if s['after'] not in fom_names])
order['carryover_ratio'] = float(np.median(post) / np.median(rest)) if post.size else None
order['carryover_tolerance'] = cfg.get('carryover_tolerance')
retime = {k: dict(of=v['retime_of'], ratio=float(np.median(v['device_ms']) / np.median(arms[v['retime_of']]['device_ms'])))
          for k, v in arms.items() if v.get('retime_of')}
order['retime'] = retime; order['retime_tolerance'] = cfg.get('retime_tolerance')
if 'retime' in cfg:
    order['passed'] = bool(post.size and abs(order['carryover_ratio'] - 1) <= cfg['carryover_tolerance'] and retime
                           and all(abs(v['ratio'] - 1) <= cfg['retime_tolerance'] for v in retime.values()))
else:
    order['passed'] = order['passed_v1_per_block']
order['positive_control_fails_as_required'] = bool(cont.size) and order['positive_control_min_relative_deviation'] > cfg['order_tolerance']
if not order['passed']:
    failures.append(dict(check='order_effect_gate', value=order['max_relative_deviation']))
if not order['positive_control_fails_as_required']:
    failures.append(dict(check='order_effect_positive_control_did_not_fail', value=order['positive_control_min_relative_deviation']))
print(json.dumps(dict(passed=bool(res['complete']) and not failures and checked > 0, complete=res['complete'], checked_error_vectors=checked,
                      worst=worst, order_effect=order, recomputed=recomputed, failure_count=len(failures), failures=failures[:100],
                      note='Sub-grid errors are recomputed exactly; full-grid errors are cross-checked by a random-node estimate '
                           'for every case (random_audit_cases) and, for smooth reduced arms, by the sub-grid value; random_node_estimate_worst/median re-derive the table '
                           'statistics from those independent estimates. Full fields are not saved (0.8 GB per arm-case at 4096^2), so the full-grid values '
                           'themselves are cross-checked, not recomputed exactly.'), indent=1))
