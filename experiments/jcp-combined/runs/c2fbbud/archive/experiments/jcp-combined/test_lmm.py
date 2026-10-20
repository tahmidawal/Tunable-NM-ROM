"""Manufactured implementation tests of the LMM stepper (DESIGN.md A2.10; gate G2a, must pass before any GPU run).

The production stepper (t2core.make_evolve, same code path) on a frozen small over-determined system, against an
independent oracle (SciPy Radau on the Galerkin ODE written in NumPy) and independent NumPy/SciPy step checks, plus
mutation tests that must be detected. CPU only:

    JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/test_lmm.py [--out checks/test_lmm.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2core as T2  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SEED, M, RP, NU = 20261008, 24, 6, .05
TIMES = [.05 * k for k in range(1, 6)]
ORDER_DTS = [.05 / 16, .05 / 32, .05 / 64]   # 0.25/64-type steps do not divide 0.05 (amendment A3.1)


def problem():
    r = np.random.default_rng(SEED)
    A = r.normal(size=(M, RP)) / np.sqrt(M)
    lam = np.geomspace(1., 1e3, M)
    Tt = .3 * r.normal(size=(M, RP, RP))
    c0 = .5 * r.normal(size=RP)
    return A, lam, Tt, c0


A, LAM, TT, C0 = problem()


def F_np(c):
    return np.einsum('mij,i,j->m', TT, c, c) + NU * LAM * (A @ c)


def dF_np(c):
    return np.einsum('mij,j->mi', TT + TT.transpose(0, 2, 1), c) + NU * LAM[:, None] * A


def oracle(tol=1e-13):
    P = np.linalg.pinv(A)
    sol = scipy.integrate.solve_ivp(lambda t, c: -P @ F_np(c), (0, .25), C0, method='Radau', rtol=tol, atol=tol,
                                    t_eval=TIMES, jac=lambda t, c: -P @ dF_np(c))
    assert sol.success
    return sol.y.T


def nl(c, d):
    return jnp.einsum('mij,i,j->m', d['T'], c, c)


def nlJ(c, d, Tm):
    T = d['T']
    return jnp.einsum('mij,i,j->m', T, c, c), jnp.einsum('mij,j->mi', T + T.transpose(0, 2, 1), c)


DATA = {}


def data(form):
    if form not in DATA:
        DATA[form] = T2.form_data(dict(A=jnp.asarray(A), lam=jnp.asarray(LAM)), dict(T=jnp.asarray(TT)), form)
    return DATA[form]


_EV = {}


def run(form, scheme, dt, steps=None, keep=None, mutation=None):
    key = (form, mutation)
    if key not in _EV:
        _EV[key] = jax.jit(T2.make_evolve(nl, nlJ, form, RP, np.inf, budget=600, mutation=mutation))
    gtol, tolf = (1e-12, 0.) if form == 'LSPG' else (0., 1e-13)
    sch = T2.sched(scheme, dt, gtol, tolf)
    if steps is not None:
        sch = dict(sch, steps=jnp.asarray(steps, jnp.int32), keep=jnp.asarray(keep, jnp.int32))
    W, st = _EV[key](jnp.asarray(C0), NU, 1., data(form), sch)
    return np.asarray(W), jax.tree_util.tree_map(np.asarray, st)


def err(W, ref):
    return max(np.linalg.norm(W[k + 1] - ref[k]) for k in range(5)) / np.linalg.norm(C0)


def order(form, scheme, ref, mutation=None):
    e = [err(run(form, scheme, dt, mutation=mutation)[0], ref) for dt in ORDER_DTS]
    return e, [float(np.log2(e[i] / e[i + 1])) for i in range(2)]


# independent specification (DESIGN section 2), NOT read from the production table (code audit 1, item 2)
SPEC = dict(BE=(1., -1., 0., 1., 0., 0), CN=(1., -1., 0., .5, .5, 0), CNR=(1., -1., 0., .5, .5, 2),
            BDF2=(1.5, -2., .5, 1., 0., 1), TH06=(1., -1., 0., .6, .4, 0))


def step_coeffs(scheme, k):
    a0, a1, a2, b0, b1, ns = SPEC[scheme]
    return (1., -1., 0., 1., 0.) if k < ns else (a0, a1, a2, b0, b1)


def intended_R(scheme, k, W, dt):
    a0, a1, a2, b0, b1 = step_coeffs(scheme, k)
    cn1, cn = W[k + 1], W[k]
    cm = W[k - 1] if k >= 1 else W[0]
    R = A @ (a0 * cn1 + a1 * cn + a2 * cm) + dt * (b0 * F_np(cn1) + b1 * F_np(cn))
    return R, (a0, b0), cn, cm


def gal_step_check(scheme, mutation=None, dt=.01, per_step=False):
    """max over steps 1..5 (or the list) of ||Q^T R_n(c_{n+1})|| / ||A c_n|| for the intended LMM (independent NumPy)."""
    W, _ = run('GAL', scheme, dt, steps=5, keep=1, mutation=mutation)
    Qn = np.linalg.qr(A)[0]
    v = []
    for k in range(5):
        R, _, cn, _ = intended_R(scheme, k, W, dt)
        v.append(float(np.linalg.norm(Qn.T @ R) / np.linalg.norm(A @ cn)))
    return v if per_step else float(max(v))


def lspg_step_check(scheme, mutation=None, dt=.01):
    """max over steps 1..5 of the relative difference between the stepper's c_{n+1} and a SciPy least-squares solve of
    the intended weighted residual D R_n(c), D = 1/(a0 + dt b0 nu lam), from the stepper's own history."""
    W, _ = run('LSPG', scheme, dt, steps=5, keep=1, mutation=mutation)
    v = []
    for k in range(5):
        a0, a1, a2, b0, b1 = step_coeffs(scheme, k)
        cn, cm = W[k], (W[k - 1] if k >= 1 else W[0])
        D = 1. / (a0 + dt * b0 * NU * LAM)
        fr = lambda c: D * (A @ (a0 * c + a1 * cn + a2 * cm) + dt * (b0 * F_np(c) + b1 * F_np(cn)))
        fj = lambda c: D[:, None] * (a0 * A + dt * b0 * dF_np(c))
        s = scipy.optimize.least_squares(fr, cn, jac=fj, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15)
        v.append(np.linalg.norm(W[k + 1] - s.x) / np.linalg.norm(s.x))
        if k == 0:
            resid = float(np.linalg.norm(fr(s.x)))
    return float(max(v)), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'checks/test_lmm.json'))
    a = ap.parse_args()
    ref = oracle()
    rep = dict(seed=SEED, M=M, Rp=RP, nu=NU, order_dts=ORDER_DTS, gates={}, info={})
    move = float(np.linalg.norm(ref[-1] - C0) / np.linalg.norm(C0))
    rep['info']['nonstationarity'] = move
    rep['gates']['nonstationary'] = move > .1
    ref12 = oracle(1e-12)
    odiff = max(np.linalg.norm(ref12[k] - ref[k]) for k in range(5)) / np.linalg.norm(C0)
    rep['info']['oracle_refinement'] = float(odiff)
    bands = dict(BE=(.85, 1.15), TH06=(.85, 1.15), CN=(1.85, 2.15), CNR=(1.85, 2.15), BDF2=(1.85, 2.15))
    for form in ('GAL', 'LSPG'):
        for sc, (lo, hi) in bands.items():
            e, p = order(form, sc, ref)
            rep['info'][f'order_{form}_{sc}'] = dict(errors=e, orders=p)
            if form == 'GAL':
                rep['gates'][f'order_GAL_{sc}'] = bool(lo <= p[-1] <= hi)
    emin = min(min(rep['info'][f'order_GAL_{sc}']['errors']) for sc in bands)
    rep['gates']['oracle_refined'] = bool(odiff < 1e-3 * emin)
    for sc in bands:
        g = gal_step_check(sc)
        rep['info'][f'gal_step_{sc}'] = g
        rep['gates'][f'gal_step_{sc}'] = g <= 1e-9
    for sc in ('BE', 'BDF2', 'CN'):
        d, resid = lspg_step_check(sc)
        rep['info'][f'lspg_step_{sc}'] = dict(rel_diff=d, ls_residual_step1=resid)
        if sc != 'CN':
            rep['gates'][f'lspg_step_{sc}'] = d <= 1e-7 and resid > 1e-8   # nonzero LS residual: weights matter
    # mutations: each must FAIL its named check (detected = True)
    det = {}
    g = gal_step_check('BDF2', 'a2flip')
    e = err(run('GAL', 'BDF2', ORDER_DTS[-1], mutation='a2flip')[0], ref)
    det['m1_a2flip'] = dict(step_check=g, oracle_err=e, detected=bool(g > 1e-9 and e > 10 * rep['info']['order_GAL_BDF2']['errors'][-1]))
    g = gal_step_check('CN', 'fn_lag')
    det['m2_fn_lag'] = dict(step_check=g, detected=bool(g > 1e-9))
    g = gal_step_check('BDF2', 'stale_hist')
    det['m3_stale_hist'] = dict(step_check=g, detected=bool(g > 1e-9))
    g1 = gal_step_check('CN', 'late_out', per_step=True)[0]          # the first step: c_1 checked against c_0
    Wm = run('GAL', 'CN', ORDER_DTS[-1], mutation='late_out')[0]
    Wu = run('GAL', 'CN', ORDER_DTS[-1])[0]
    eT = np.linalg.norm(Wm[5] - ref[4]) / np.linalg.norm(C0)       # the final time t = 0.25
    eTu = np.linalg.norm(Wu[5] - ref[4]) / np.linalg.norm(C0)
    det['m4_late_out'] = dict(step1_check=g1, final_time_err=eT, final_time_err_unmutated=eTu,
                              detected=bool(g1 > 1e-9 and eT > 10 * eTu))
    for sc in ('BE', 'BDF2'):
        d, _ = lspg_step_check(sc, 'd_b1')
        det[f'm5_d_b1_{sc}'] = dict(rel_diff=d, detected=bool(d > 1e-6))
    # production table equals the specification; GAL must reject a non-root exit (code audit 1, items 1-2)
    rep['gates']['production_table_matches_spec'] = all(tuple(T2.SCHEMES[k_]) == SPEC[k_] for k_ in SPEC)
    ev = jax.jit(T2.make_evolve(nl, nlJ, 'GAL', RP, np.inf, budget=5))
    _, stn = ev(jnp.asarray(C0), NU, 1., data('GAL'), T2.sched('CN', .01, 0., 1e-40))
    rep['info']['gal_unattainable_nfail'] = int(stn['nfail'])
    rep['gates']['gal_nonroot_rejected'] = int(stn['nfail']) == int(stn['steps']) > 0
    rep['mutations'] = det
    for k, v in det.items():
        rep['gates'][f'mutation_{k}_detected'] = v['detected']
    rep['all_pass'] = bool(all(rep['gates'].values()))
    import hashlib
    rep['source_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('t2core.py', 'test_lmm.py')}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float) + '\n')
    for k, v in rep['gates'].items():
        print(('PASS ' if v else 'FAIL ') + k)
    for k, v in rep['info'].items():
        print(k, v)
    print('ALL PASS' if rep['all_pass'] else 'SOME GATES FAILED')
    sys.exit(0 if rep['all_pass'] else 1)


if __name__ == '__main__':
    main()
