"""Block-stored bank vs the numbers already committed in runs/final01/final01 (panel A, 64^3, first cases of each cohort).
Reruns the frozen model/arms locally and compares to results.json entry-by-entry."""
import json, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
H = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H))
import core as C
if len(sys.argv) > 1: _t = int(sys.argv[1]); C.row_blocks = (lambda rows, target=_t, _f=C.row_blocks: _f(rows, target))   # force multi-block
cfg = json.loads((H / 'configs' / 'final01.json').read_text()); n = 64; d = 3
res = json.loads((H / 'runs/final01/pull/out/final01/results.json').read_text())
mesh = next(m for m in res['meshes'] if m['intervals'] == n); rows = {r['method']: r for r in mesh['rows']}
model = C.load_model(cfg['model'], H / 'inputs'); bank = C.bank_at(model, n)
print('blocks', len(bank), [b.shape for b in bank])
rtri = C.tsqr_r(bank); modes = C.mode_list(cfg['tests'], n, d); a = C.weak_matrix(bank, modes, n, d)
setup = dict(n=n, d=d, times=cfg['times'], nu=cfg['diffusivity'], modes=modes, a=a, rtri=rtri,
             mode_lam=C.mode_eigs(n, modes), directions=model['directions'])
prop = C.make_propagate(d); lam = C.eig_grid(n, d); tj = jnp.asarray(cfg['times'])
draws = np.concatenate([C.family('h3d', s, c) for s, c in cfg['cohorts']])
worst = 0.
for ci in (0, 1, 64, 65):
    u0 = C.initial_grid(n, d, draws[ci]); truth = prop(u0, lam, tj, cfg['diffusivity'])
    for arm in cfg['rom_arms']:
        st = C.make_stages(model, setup, arm['q'], {**cfg['rom_defaults'], **arm.get('opt', {})})
        e = np.asarray(C.rel_errors(st['query'](u0, bank)[0], truth)); ref = np.asarray(rows[arm['name']]['same'][ci])
        worst = max(worst, float(np.max(np.abs(e - ref))))
    lb = C.linear_bank(setup, 'field'); e = np.asarray(C.rel_errors(lb(u0, bank), truth))
    worst = max(worst, float(np.max(np.abs(e - np.asarray(rows['linear_bank_field_BASELINE']['same'][ci])))))
print('max |error difference| vs committed final01 results:', worst)
