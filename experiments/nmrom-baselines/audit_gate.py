"""Independent NumPy re-computation of the gate errors from the saved fields (no JAX import)."""
import json, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
S = json.loads((run / 'output/summary.json').read_text())
n = S['n']
truth = np.load(run / 'output/fields_common.npz')['truth']

def maxrel(f):
    out = []
    for sl in (slice(0, n), slice(n, 2 * n)):
        num = np.sqrt(((f[1:, sl] - truth[1:, sl]) ** 2).sum(1)); den = np.sqrt((truth[1:, sl] ** 2).sum(1))
        out.append(float((num / den).max()))
    return max(out)

rows, ok = [], True
ls = maxrel(np.load(run / 'output/fields_common.npz')['ls_lspg'])
rows.append(dict(arm='ls_lspg', reported=S['ls_lspg']['error']['max'], recomputed=ls))
for r in S['seeds']:
    f = np.load(run / f"output/fields_seed{r['seed']}.npz")['nm_lspg']
    rows.append(dict(arm=f"nm_lspg_seed{r['seed']}", reported=r['nm_lspg']['error']['max'], recomputed=maxrel(f)))
    hp = run / f"output/fields_seed{r['seed']}_hr.npz"
    if hp.exists() and r.get('hr'):
        rows.append(dict(arm=f"nm_lspg_hr_seed{r['seed']}_{r['hr'][0]['residual_basis']}x{r['hr'][0]['samples']}",
                         reported=r['hr'][0]['error']['max'], recomputed=maxrel(np.load(hp)['nm_lspg_hr'])))
for w in rows:
    w['agree'] = bool(abs(w['reported'] - w['recomputed']) <= 1e-10 * max(1., abs(w['reported'])))
    ok &= w['agree']
nm = [w['recomputed'] for w in rows if w['arm'].startswith('nm_lspg_seed')]
res = dict(run=str(run), rows=rows, all_agree=ok, recomputed_nm_median=float(np.median(nm)) if len(nm) == 3 else None,
           gate_passed_reported=S['gate']['passed'], hr_passed_reported=S['gate']['hr_passed'],
           log_has_gpu=('jax_backend=gpu' in ''.join(p.read_text() for p in (run / 'logs').glob('*.out'))))
(run / 'audit.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(res, indent=1))
