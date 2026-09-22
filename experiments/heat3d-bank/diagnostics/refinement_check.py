"""run.py asserts max_t rel ||phys_n[::2] - phys_{n/2}|| < reference_budget for every case. Check that bound at n=32/64/128 on
VALIDATION draws (921777) only, before the final job (NumPy/SciPy, same formulas as audit.py)."""
import sys
import numpy as np
from scipy.fft import dstn, idstn
sys.path.insert(0, '.')
draws = np.random.default_rng(921777).random((256, 5)) * np.array([.3, .3, .3, .05, .4]) + np.array([.35, .35, .35, .10, .8])
T = np.array([0., .1, .2, .3, .4, .5]); nu = .02
def phys(n, dr):
    a = np.arange(1, n) / n; u = dr[4]
    for ax in range(3): u = u * (4 * a * (1 - a) * np.exp(-(a - dr[ax]) ** 2 / (2 * dr[3] ** 2))).reshape((-1,) + (1,) * (2 - ax))
    k = (np.pi * np.arange(1, n)) ** 2; lam = k[:, None, None] + k[None, :, None] + k[None, None, :]; c = dstn(u, type=1, norm='ortho')
    return np.stack([u] + [idstn(c * np.exp(-nu * t * lam), type=1, norm='ortho') for t in T[1:]])
for n in map(int, sys.argv[1:]):
    worst = 0.
    for dr in draws[:int(64 if n >= 128 else 256)]:
        f, h = phys(n, dr)[:, 1::2, 1::2, 1::2], phys(n // 2, dr)
        worst = max(worst, float(np.max(np.sqrt(np.sum((f - h) ** 2, axis=(1, 2, 3)) / np.sum(h ** 2, axis=(1, 2, 3))))))
    print(n, 'max reference refinement', worst, flush=True)
