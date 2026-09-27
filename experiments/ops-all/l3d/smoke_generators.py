"""Check problems.py (PyTorch generators) against the Table-1 JAX generators (deps/poisson, deps/heat).
Usage: smoke_generators.py <n> [<n> ...]  -> prints max relative gaps; exits 1 if any > 1e-12."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'deps' / 'poisson')]
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import torch
import common as PC
import poisson as PP
import problems as PB
sys.path.insert(0, str(HERE / 'deps' / 'heat'))
import importlib
HC = importlib.import_module('core')

bad = 0
for n in map(int, sys.argv[1:]):
    draws = PB.family(PB.POISSON['train_seed'], 6)
    assert np.array_equal(draws, PC.family(PB.POISSON['train_seed'], 6))
    assert np.array_equal(PB.family(PB.HEAT['train_seed'], 6), HC.family('h3d', PB.HEAT['train_seed'], 6))
    P = PB.Problem('poisson', n)
    x, y = P.batch(draws)
    lam = PC.eigenvalues(n)
    for i, p in enumerate(draws):
        f = np.asarray(PP.source(n, p)); u = np.asarray(PP.solve_dst(PP.source(n, p), lam))
        gf = np.linalg.norm(x[i].cpu().numpy() - f) / np.linalg.norm(f)
        gu = np.linalg.norm(y[i].cpu().numpy() - u) / np.linalg.norm(u)
        bad += (gf > 1e-12) + (gu > 1e-12)
        if i == 0:
            print(f'poisson n={n} source gap {gf:.2e} solution gap {gu:.2e}')
    H = PB.Problem('heat', n)
    hd = PB.family(PB.HEAT['train_seed'], 4)
    x, y = H.batch(hd)
    prop = HC.make_propagate(3)
    for i, p in enumerate(hd):
        u0 = HC.initial_grid(n, 3, p)
        full = np.asarray(prop(u0, HC.eig_grid(n, 3), jnp.asarray(PB.HEAT_TIMES), PB.HEAT_NU))
        g0 = np.linalg.norm(x[i].cpu().numpy() - full[0]) / np.linalg.norm(full[0])
        gl = max(np.linalg.norm(y[i, j].cpu().numpy() - full[j + 1]) / np.linalg.norm(full[j + 1]) for j in range(5))
        bad += (g0 > 1e-12) + (gl > 1e-12)
        if i == 0:
            print(f'heat n={n} u0 gap {g0:.2e} later gap {gl:.2e}')
    # the operator error of the exact truth is zero; of zero prediction it is one
    e = H.errors(torch.cat((x[:, None], y), 1), y, x)
    assert float(e.max()) == 0.0
print('GENERATORS', 'OK' if bad == 0 else f'FAILED {bad}')
sys.exit(1 if bad else 0)
