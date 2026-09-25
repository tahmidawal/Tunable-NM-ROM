"""Independent re-derivation of every ms, error and ratio in checks/<attempt>-summary.json from the raw result.json
(invocations and quick rows only; no code shared with audit_eqtol.py). Prints max relative disagreement."""
import json, sys
import numpy as np
att = sys.argv[1]
r = json.load(open(f'runs/{att}/archive/output/result.json'))
s = json.load(open(f'checks/{att}-summary.json'))['ladder']
ms = {}
for x in r['invocations']:
    ms.setdefault(x['name'], []).append(x['gpu_seconds'] * 1e3)
med = {k: float(np.median(v)) for k, v in ms.items()}
err = {}
for x in r['quick']:
    err.setdefault(x['name'], []).append(100 * x['same_grid_evolved'])
t1 = min(med[n] for n in med if n.split('__')[0] == 'lean_nt3e-3_l3e-3_dt005')
worst = 0.
for n, row in s['rows'].items():
    base = n[:-len('_graphs')] if n.endswith('_graphs') else n
    both = [m for m in (base, base + '_graphs') if m in med]
    mine = min(med[m] for m in both)
    cur = s['rows'][s['ladders'][str(row['R_prime'])]['current']]
    cb = cur['arm'][:-7] if cur['arm'].endswith('_graphs') else cur['arm']
    cms = min(med[m] for m in (cb, cb + '_graphs') if m in med)
    checks = dict(ms=(mine, row['median_gpu_ms']), worst=(max(err[n]), row['worst_percent']),
                  median=(float(np.median(err[n])), row['median_percent']),
                  speedup=(t1 / mine, row['speedup_vs_table1_fom']), over_current=(mine / cms, row['ms_over_current']))
    for k, (a, b) in checks.items():
        worst = max(worst, abs(a / b - 1))
    print(row['R_prime'], row['rule'], row['gtol'], f'{mine:.3f} ms', f'{t1 / mine:.3f}x', f'{mine / cms:.4f} of current')
print('Table-1 FOM ms', t1, 'summary', s['table1_fom_ms'])
print('max relative disagreement', worst)
