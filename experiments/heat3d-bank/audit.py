"""Independent NumPy/SciPy re-computation of every reported error from saved fields (no JAX, no lane imports)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn, idstn

out = Path(sys.argv[1]); res = json.loads((out / 'results.json').read_text()); cfg = res['config']
times, nu = np.asarray(cfg['times']), cfg['diffusivity']; worst = dict(full=0., restricted=0.); checked = 0; failures = []; skipped = 0
for mesh in res['meshes']:
    n = mesh['intervals']; rows = {r['method']: r for r in mesh['rows']}
    for case in mesh['cases']:
        if case.get('fields_saved', True) is False: skipped += 1; continue   # heat3d-bank: errors-only validation cases beyond the saved prefix
        z = np.load(out / f"fields_n{n}_case{case['case']}.npz"); draw = z['draw']; d = len(draw) - 2; s = int(z['stride'])
        a = np.arange(1, n) / n; u0 = np.asarray(draw[d + 1])
        for ax in range(d):
            f = 4 * a * (1 - a) * np.exp(-(a - draw[ax]) ** 2 / (2 * draw[d] ** 2)); u0 = u0 * f.reshape((-1,) + (1,) * (d - 1 - ax))
        k = np.arange(1, n); coef = dstn(u0, type=1, norm='ortho'); refs = {}
        for key, l in (('same', 4 * n * n * np.sin(np.pi * k / (2 * n)) ** 2), ('physical', (np.pi * k) ** 2)):
            lam = sum(l.reshape((-1,) + (1,) * (d - 1 - ax)) for ax in range(d))
            refs[key] = np.stack([u0] + [idstn(coef * np.exp(-nu * t * lam), type=1, norm='ortho') for t in times[1:]])
        sub = (slice(None),) + (slice(s - 1, None, s),) * d; axes = tuple(range(1, d + 1))
        for name in z.files:
            if name in ('draw', 'stride', 'sample_indices'): continue
            full = name.startswith('FULL_'); rand = name.startswith('RAND_'); method = name[5:] if (full or rand) else name; pred = z[name]
            for key in ('same', 'physical'):
                if rand:   # independently seeded random nodes: a second estimate of the FULL-grid error for reduced arms
                    ref = refs[key].reshape(len(times), -1)[:, z['sample_indices']]; err = np.sqrt(np.sum((pred - ref) ** 2, axis=1) / np.sum(ref ** 2, axis=1)); rep_ = np.asarray(rows[method][key][case['case']])
                    gap = float(np.max(np.abs(err - rep_) / np.maximum(rep_, 1e-9))); checked += 1
                    if method.startswith('nmrom') or 'BASELINE' in method:
                        worst['rand_gap'] = max(worst.get('rand_gap', 0.), gap)
                        if gap > .05: failures.append(dict(n=n, case=case['case'], method=method, reference=key, random_sample_gap=gap))
                    continue
                ref = refs[key] if full else refs[key][sub]
                err = np.sqrt(np.sum((pred - ref) ** 2, axis=axes) / np.sum(ref ** 2, axis=axes)); case_i = case['case']; checked += 1
                if full:
                    rel = float(np.max(np.abs(err - np.asarray(rows[method][key][case_i])))); worst['full'] = max(worst['full'], rel); bad = rel > 1e-10
                else:   # exact recomputation of the in-job SUB-GRID error; full-vs-sub agreement gated for smooth-error (reduced) arms only
                    rel = float(np.max(np.abs(err - np.asarray(rows[method][key + '_sub'][case_i])))); worst['restricted'] = max(worst['restricted'], rel); bad = rel > 1e-10
                    rep = np.asarray(rows[method][key][case_i]); gap = float(np.max(np.abs(err - rep) / np.maximum(rep, 1e-9)))
                    if method.startswith('nmrom') or 'BASELINE' in method:
                        worst['gap'] = max(worst.get('gap', 0.), gap); bad = bad or gap > .05
                if bad: failures.append(dict(n=n, case=case_i, method=method, reference=key, full=full, discrepancy=rel))
print(json.dumps(dict(passed=not failures and checked > 0 and res['complete'], checked_error_vectors=checked, cases_without_saved_fields=skipped, cases_with_selection_arms_only=sum(c.get('fields_saved') == 'selection' for m in res['meshes'] for c in m['cases']), max_full_field_absolute_discrepancy=worst['full'],
    max_subgrid_absolute_discrepancy=worst['restricted'], max_reduced_arm_full_vs_subgrid_relative_gap=worst.get('gap'), max_reduced_arm_full_vs_random_sample_relative_gap=worst.get('rand_gap'), gap_tolerance=.05, failures=failures[:50], failure_count=len(failures),
    note='Full-field audits recompute the reported full-grid error exactly. Sub-grid audits recompute exactly the in-job strided sub-grid error from the saved strided field and an independent SciPy reference; for reduced arms (smooth error) the full-grid and sub-grid error values must also agree within 5%. CG arms have rough error fields, so their full-grid error is audited exactly only where full fields are saved.'), indent=1))
