"""Independent NumPy re-computation of every reported family error from saved fields (no JAX import)."""
import json, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
S = json.loads((run / 'output/summary.json').read_text())
REF = np.load(run / 'output/reference.npz')['reference']
tol = 1e-10 if S['saved_field_dtype'] == 'float64' else 5e-6

def worst(F, ref):
    d = (F.astype(np.float64) - ref)[:, :, 1:-1, 1:-1].reshape(ref.shape[0], ref.shape[1], -1)
    den = np.sqrt((ref[:, 0, 1:-1, 1:-1].reshape(ref.shape[0], -1) ** 2).sum(1))
    e = np.sqrt((d ** 2).sum(2)) / den[:, None]
    return float(e[:, 1:].max()), e

rows, ok = [], True
for name, arm in S['arms'].items():
    p = run / f'output/fields_{name}.npz'
    if not p.exists():
        rows.append(dict(arm=name, audited=False, cohort=arm.get('cohort'))); continue
    w, _ = worst(np.load(p)['fields'], REF)
    agree = bool(abs(w - arm['worst_evolved']) <= tol * max(1., w))
    rows.append(dict(arm=name, audited=True, reported=arm['worst_evolved'], recomputed=w, agree=agree)); ok &= agree
timed = []
for p in sorted((run / 'output').glob('timed_*_case*.npy')):
    name, c = p.stem[len('timed_'):].rsplit('_case', 1)
    f = run / f'output/fields_{name}.npz'
    if f.exists():
        same = bool(np.array_equal(np.load(p), np.load(f)['fields'][int(c)]))
        timed.append(dict(arm=name, case=int(c), timed_output_equals_evaluated=same)); ok &= same
res = dict(run=str(run), tolerance=tol, rows=rows, timed=timed, all_agree=ok,
           log_has_gpu=('jax_backend=gpu' in ''.join(p.read_text() for p in (run / 'logs').glob('*.out'))))
(run / 'audit.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(dict(all_agree=ok, audited=sum(r['audited'] for r in rows), unaudited=[r['arm'] for r in rows if not r['audited']]), indent=1))
