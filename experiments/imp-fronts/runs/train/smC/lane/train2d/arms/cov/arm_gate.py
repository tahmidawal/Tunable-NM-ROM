"""imp-fronts arm (b) 'cov': the R0 replacement gate (DESIGN B1(b), B2 C-B4, Amendment 1), called by the arm's
smoothtrain.py at the production data size (256 nodes, 576 trajectories) after the training data are built.

Replaced gate (fails closed): C-B4 on the COMPLETE parameter tables of both training populations drawn by THIS
generator's sample_params (seed 0 x 576, seed 1000 x 4032): max s >= 900, every s <= 1000, min nu >= 0.0016
(s = a/nu); c_x, c_y, w, a bitwise equal to the incumbent draws; the incumbent tables (same seeds) FAIL max s >= 900
(must-fail); and the incumbent data gate kept: Newton relative residual <= 1e-8 and every state finite (Stage A here,
Stage B extraction checks its 4,608 trajectories itself).
"""
from __future__ import annotations

import numpy as np

POPULATIONS = (('seed0', 0, 576), ('seed1000', 1000, 4032))
S_MIN_MAX, S_CAP, NU_MIN = 900.0, 1000.0, 0.0016
TOL = 1e-12                                                 # relative round-off allowance on the two caps


def incumbent_table(seed, m):
    """The incumbent burgers2d_film.sample_params draw (cx, cy, w, a, nu), reproduced call for call."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = rng.uniform(0.05, 0.20, m)
    a = rng.uniform(0.5, 2.0, m)
    nu = np.exp(rng.uniform(np.log(0.01), np.log(0.1), m))
    return cx, cy, w, a, nu


def stats(a, nu):
    s = a / nu
    return dict(n=int(len(a)), max_s=float(s.max()), min_s=float(s.min()), min_nu=float(nu.min()),
                max_nu=float(nu.max()), n_s_ge_900=int((s >= S_MIN_MAX).sum()))


def c_b4(sample_params):
    out, ok = {}, True
    for name, seed, m in POPULATIONS:
        cx, cy, w, a, nu, _z = sample_params(seed=seed, m=m)
        icx, icy, iw, ia, inu = incumbent_table(seed, m)
        arm, inc = stats(a, nu), stats(ia, inu)
        same = all(np.array_equal(p, q) for p, q in ((cx, icx), (cy, icy), (w, iw), (a, ia)))
        checks = dict(max_s_ge_900=arm['max_s'] >= S_MIN_MAX, all_s_le_1000=arm['max_s'] <= S_CAP * (1 + TOL),
                      min_nu_ge_0p0016=arm['min_nu'] >= NU_MIN * (1 - TOL), cx_cy_w_a_identical=bool(same),
                      incumbent_fails_max_s_ge_900=inc['max_s'] < S_MIN_MAX)
        ok = ok and all(checks.values())
        out[name] = dict(seed=seed, m=m, arm=arm, incumbent=inc, checks=checks)
    return dict(passed=bool(ok), populations=out)


def r0_replacement(S_tr, worst, fingerprint):
    import burgers2d_film as bf
    r = c_b4(bf.sample_params)
    finite = bool(np.isfinite(np.asarray(S_tr)).all())
    resid_ok = bool(np.isfinite(worst) and worst <= 1e-8)
    passed = r['passed'] and finite and resid_ok
    return dict(arm='cov', passed=bool(passed), C_B4=r, all_states_finite=finite, newton_worst_rel_residual=float(worst),
                newton_residual_ok=resid_ok, fingerprint=fingerprint,
                summary=dict(C_B4=r['passed'], finite=finite, newton_worst=float(worst),
                             max_s={k: v['arm']['max_s'] for k, v in r['populations'].items()}),
                note='replaces R0 (r3a fingerprint) for the coverage arm (DESIGN B1(b), B2 C-B4, Amendment 1)')
