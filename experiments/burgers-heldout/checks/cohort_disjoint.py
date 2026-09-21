"""sel32 (this lane's selection cohort) vs every training draw and every evaluation cohort.

Same tests as reports/checks/2026-09-21-burgers-train-eval-overlap.py (on main): exact / near row
matches in the unit-scaled parameter cube and shared single scalar values (reused-stream landmine).
NumPy only; run with /home/tahmid/Dev/.venv/bin/python. Writes cohort_disjoint.json beside itself.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def params_draw(seed, count):   # copy of mr-burgers2d/engines.py:20
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count), r.uniform(.05, .20, count),
                     r.uniform(.5, 2., count), np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)


def sample_params(seed, m):     # copy of burgers2d_film.sample_params (non-z part)
    rng = np.random.default_rng(seed)
    cx = rng.uniform(.15, .85, m); cy = rng.uniform(.15, .85, m)
    w = rng.uniform(.05, .2, m); a = rng.uniform(.5, 2., m); nu = np.exp(rng.uniform(np.log(.01), np.log(.1), m))
    return np.stack([cx, cy, w, a, nu], 1)


def case_seed(split, i):
    codes = dict(calibration=0, train=1, validation=2)
    return int(np.random.SeedSequence([20260914, 22, codes[split], i]).generate_state(1, dtype=np.uint32)[0])


lo = np.array([.15, .15, .05, .5, np.log(.01)])
hi = np.array([.85, .85, .2, 2., np.log(.1)])


def scaled(P):
    Q = P.copy()
    Q[:, 4] = np.log(Q[:, 4])
    return (Q - lo) / (hi - lo)


training = {
    'head+bank 4608': np.concatenate([sample_params(0, 576), sample_params(1000, 4032)]),
    'params_draw(0,128) populations/directions': params_draw(0, 128),
}
cohorts = {
    'sel32': params_draw(20260927, 32),
    'dev6': np.concatenate([params_draw(7090702, 4), params_draw(911702, 2)]),
    'hold64': params_draw(20260916, 64),
    'table2_val32': np.stack([params_draw(case_seed('validation', i), 1)[0] for i in range(32)]),
    'sealed6': params_draw(17092026, 6),
    'bankfloor_confirm8': params_draw(20260922, 32)[:8],
    'fresh64': params_draw(20260929, 64),
}
out = dict(tests={}, cross={})
for tn, T in training.items():
    Ts = scaled(T)
    vals = set(T.ravel().tolist())
    for k, P in cohorts.items():
        d = np.sqrt(((scaled(P)[:, None, :] - Ts[None]) ** 2).sum(-1))
        shared = sum(v in vals for v in P.ravel().tolist())
        out['tests'][f'{k} vs {tn}'] = dict(min_dist=float(d.min()), exact=int((d == 0).sum()),
                                            near_1e8=int((d < 1e-8).sum()), shared_scalars=int(shared))
for (a, A), (b, B) in itertools.combinations(cohorts.items(), 2):
    d = np.sqrt(((scaled(A)[:, None] - scaled(B)[None]) ** 2).sum(-1))
    vals = set(B.ravel().tolist())
    out['cross'][f'{a} vs {b}'] = dict(min_dist=float(d.min()), shared_scalars=int(sum(v in vals for v in A.ravel().tolist())))
bad = [k for k, v in out['tests'].items() if v['exact'] or v['near_1e8'] or v['shared_scalars']]
bad += [k for k, v in out['cross'].items() if (k.startswith('sel32') or 'fresh64' in k) and (v['min_dist'] < 1e-8 or v['shared_scalars'])]
out['sel32_disjoint_from_everything'] = not [k for k in bad if 'sel32' in k]
out['fresh64_disjoint_from_everything'] = not [k for k in bad if 'fresh64' in k]
out['all_flags'] = bad
out['sel32_sha256_of_float64_bytes'] = __import__('hashlib').sha256(cohorts['sel32'].tobytes()).hexdigest()
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1) + '\n')
for k, v in out['tests'].items():
    print(k, v)
for k, v in out['cross'].items():
    if 'sel32' in k:
        print(k, v)
print('sel32 disjoint from everything:', out['sel32_disjoint_from_everything'], 'flags:', bad)
