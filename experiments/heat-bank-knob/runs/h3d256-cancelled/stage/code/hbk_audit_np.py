"""Independent NumPy/SciPy audit of a heat-bank-knob panel (no JAX, no lane imports).

For every saved case and arm: the initial field is rebuilt from the draw, the same-grid and continuum references are
recomputed with scipy.fft.dstn, and
  - the strided sub-grid error recorded in-job is recomputed exactly (|diff| <= 1e-10),
  - full fields (small meshes, first cases) reproduce the recorded FULL-grid error exactly (<= 1e-10),
  - random-node samples (first two cases) estimate the full-grid error within 5 % for reduced arms.
Controls that must be DETECTED (else the audit fails): the truth of a different case, and a perturbed recorded error.
Usage: hbk_audit_np.py <panel out dir>
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn, idstn

out = Path(sys.argv[1]); res = json.loads((out / 'results.json').read_text()); cfg = res['config']
times, nu = np.asarray(cfg['times']), cfg['diffusivity']


def references(n, draw):
    d = len(draw) - 2; a = np.arange(1, n) / n; u0 = np.asarray(draw[d + 1], dtype=np.float64)
    for ax in range(d):
        f = 4 * a * (1 - a) * np.exp(-(a - draw[ax]) ** 2 / (2 * draw[d] ** 2)); u0 = u0 * f.reshape((-1,) + (1,) * (d - 1 - ax))
    k = np.arange(1, n); coef = dstn(u0, type=1, norm='ortho'); refs = {}
    for key, l in (('same', 4 * n * n * np.sin(np.pi * k / (2 * n)) ** 2), ('physical', (np.pi * k) ** 2)):
        lam = sum(l.reshape((-1,) + (1,) * (d - 1 - ax)) for ax in range(d))
        refs[key] = np.stack([u0] + [idstn(coef * np.exp(-nu * t * lam), type=1, norm='ortho') for t in times[1:]])
    return d, refs


def rel(pred, ref, axes):
    return np.sqrt(np.sum((pred - ref) ** 2, axis=axes) / np.sum(ref ** 2, axis=axes))


def reduced(m):
    return m.startswith(('nmrom', 'lin_', 'parent'))


worst = dict(sub=0., full=0., rand=0., sub_gap=0.); checked = 0; failures = []; controls = {}; coverage_missing = []
for mesh in res['meshes']:
    n = mesh['intervals']; rows = {r['method']: r for r in mesh['rows']}; prev = None
    ctl = controls.setdefault(str(n), dict(swapped_case_detected=None, perturbed_error_detected=None))
    for case in mesh['cases']:
        ci = case['case']; z = np.load(out / f'fields_n{n}_case{ci}.npz'); draw = z['draw']; s = int(z['stride'])
        assert np.array_equal(draw, np.asarray(case['draw']))
        missing = [m for m in rows if m not in z.files]
        if missing: coverage_missing.append(dict(n=n, case=ci, missing=missing))
        d, refs = references(n, draw); sub = (slice(None),) + (slice(s - 1, None, s),) * d; axes = tuple(range(1, d + 1))
        for name in z.files:
            if name in ('draw', 'stride', 'sample_indices'): continue
            kind = 'full' if name.startswith('FULL_') else 'rand' if name.startswith('RAND_') else 'sub'
            m = name[5:] if kind != 'sub' else name; pred = z[name]
            if not np.isfinite(pred).all(): failures.append(dict(n=n, case=ci, method=m, nonfinite_field=True)); continue
            for key in ('same', 'physical'):
                if kind == 'rand':
                    ref = refs[key].reshape(len(times), -1)[:, z['sample_indices']]; err = rel(pred, ref, 1)
                    gap = float(np.max(np.abs(err - np.asarray(rows[m][key][ci])) / np.maximum(np.asarray(rows[m][key][ci]), 1e-9)))
                    if reduced(m):
                        worst['rand'] = max(worst['rand'], gap)
                        if not np.isfinite(gap) or gap > .05: failures.append(dict(n=n, case=ci, method=m, ref=key, random_sample_gap=gap))
                    checked += 1; continue
                ref = refs[key] if kind == 'full' else refs[key][sub]
                err = rel(pred, ref, axes); recorded = np.asarray(rows[m][key if kind == 'full' else key + '_sub'][ci])
                dev = float(np.max(np.abs(err - recorded))); worst[kind] = max(worst[kind], dev); checked += 1
                if kind == 'sub' and key == 'same' and reduced(m):   # diagnostic only: sub-grid vs recorded full-grid error
                    full_rec = np.asarray(rows[m]['same'][ci]); worst['sub_gap'] = max(worst['sub_gap'], float(np.max(np.abs(err - full_rec) / np.maximum(full_rec, 1e-9))))
                if not np.isfinite(dev) or dev > 1e-10: failures.append(dict(n=n, case=ci, method=m, ref=key, kind=kind, discrepancy=dev))
        # controls, once per mesh, on the first saved reduced arm: they must be detected
        if prev is not None and ctl['swapped_case_detected'] is not True:   # per mesh
            m = next(k for k in z.files if reduced(k)); _, rp = references(n, prev)
            err = rel(z[m], rp['same'][sub], axes); ctl['swapped_case_detected'] = bool(np.max(np.abs(err - np.asarray(rows[m]['same_sub'][ci]))) > 1e-10)
        if ctl['perturbed_error_detected'] is not True:
            m = next(k for k in z.files if reduced(k)); err = rel(z[m], refs['same'][sub], axes)
            ctl['perturbed_error_detected'] = bool(np.max(np.abs(err - np.asarray(rows[m]['same_sub'][ci]) * (1 + 1e-3))) > 1e-10)
        prev = draw
ctl_ok = all(c['swapped_case_detected'] and c['perturbed_error_detected'] for c in controls.values()) and len(controls) == len(res['meshes'])
passed = bool(not failures and not coverage_missing and checked > 0 and res['complete'] and ctl_ok)
print(json.dumps(dict(passed=passed, version=2, checked_error_vectors=checked, max_subgrid_abs_discrepancy=worst['sub'], max_full_abs_discrepancy=worst['full'],
                      max_reduced_random_sample_relative_gap=worst['rand'], diagnostic_max_reduced_subgrid_vs_fullgrid_relative_gap=worst['sub_gap'],
                      coverage_missing=coverage_missing[:20], controls=controls, failure_count=len(failures), failures=failures[:50],
                      definition='exact recomputation (<=1e-10) of every saved sub-grid and full-grid error with an independent SciPy DST reference; '
                                 'random-node full-grid estimate within 5% for reduced arms; swapped-case and perturbed-error controls must be detected'), indent=1))
