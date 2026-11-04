"""Local CPU/GPU smoke for divfree_bank: no-slip, exact divergence, FD gradients, save/load.

Synthetic random weights only; no project data. Writes SMOKE-BANK.json beside this file.
"""
import sys, json, time
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent/'ns-obstacle-rom')]
import numpy as np
import divfree_bank as B
import jax

rng = np.random.default_rng(0)
p0 = B.initialize(seed=1)
p0['layers'][-1] = (rng.normal(size=p0['layers'][-1][0].shape)*0.3, rng.normal(size=4)*0.1)
p0['c'] = rng.normal(size=4)
bank = B.DivFreeBank(p0, {'trained': False})
g = B.CHANNEL; out = {}

# boundary points (physical): inlet, walls, cylinder; outlet for contrast
t = np.linspace(0.01, 0.99, 200)
th = np.linspace(0, 2*np.pi, 400, endpoint=False)
pts = {'inlet': np.c_[0*t, t*g.height], 'bottom': np.c_[t*2.2, 0*t], 'top': np.c_[t*2.2, 0*t+g.height],
       'cylinder': np.c_[0.2+0.05*np.cos(th), 0.2+0.05*np.sin(th)], 'outlet': np.c_[0*t+2.2, t*g.height]}
for k, x in pts.items():
    v, _ = bank.evaluate(x)
    out[f'max_abs_velocity_{k}'] = float(np.abs(v).max())
# interior: divergence and FD check
n = 20000
x = np.c_[rng.uniform(0, 2.2, n), rng.uniform(0, .41, n)]
x = x[np.linalg.norm(x-np.array(g.center), axis=1) > 0.051]
v, gr = bank.evaluate(x)
div = gr[:, 0, 0] + gr[:, 1, 1]
out['max_abs_divergence_physical'] = float(np.abs(div).max())
out['max_abs_gradient_physical'] = float(np.abs(gr).max())
h = 1e-6; errs = []
for j in range(2):
    e = np.zeros(2); e[j] = h
    vp, _ = bank.evaluate(x[:500]+e); vm, _ = bank.evaluate(x[:500]-e)
    fd = (vp-vm)/(2*h)
    errs.append(float(np.abs(fd-gr[:500, :, j, :]).max()/np.abs(gr[:500]).max()))
out['fd_relative_gradient_error'] = errs
# chi near cylinder flux check: velocity just off cylinder small, nonzero further out
for dr in (1e-6, 1e-3, 0.02):
    xr = np.c_[0.2+(0.05+dr)*np.cos(th), 0.2+(0.05+dr)*np.sin(th)]
    out[f'max_speed_at_cyl_offset_{dr}'] = float(np.abs(bank.evaluate(xr)[0]).max())
tmp = here/'_smoke_bank.npz'; sha = bank.save(tmp); b2 = B.DivFreeBank.load(tmp)
out['reload_identical'] = bool(np.array_equal(b2.evaluate(x[:100])[1], gr[:100])); tmp.unlink()
tic = time.perf_counter(); bank.evaluate(x); out['eval_seconds_%d_points' % len(x)] = time.perf_counter()-tic
out['backend'] = jax.default_backend()
ok = (max(out[f'max_abs_velocity_{k}'] for k in ('inlet', 'bottom', 'top', 'cylinder')) < 1e-12
      and out['max_abs_divergence_physical'] == 0.0 and max(errs) < 1e-6 and out['reload_identical']
      and out['max_abs_velocity_outlet'] > 1e-3)
out['passed'] = bool(ok)
print(json.dumps(out, indent=1))
(here/'SMOKE-BANK.json').write_text(json.dumps(out, indent=1)+'\n')
sys.exit(0 if ok else 1)
