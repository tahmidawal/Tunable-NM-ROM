"""jcp-references: code gates and unit tests (DESIGN §7, A1.13, A2.7, A2.8, A3.4).

    G1   parity: upwind + BE of refcore reproduces the vendor FOMs (2D engines.make_fom L=256; 3D common.make_fom n=65)
    G2   FD manufactured solution (CN-R central): true order and Richardson effectivity
    G3a  PS linear exactness; G3b IC quadrature check; G3c ETDRK4 contour check
    G4   PS manufactured solution: spatial ladder, temporal self-difference ladder, projection quadrature order
    U*   unit tests: order solve edge cases, X1b synthetic first-order rejection, injected Peclet, RNG prefix
         stability + golden rows, encryption round trip
Each gate returns a dict with 'pass'. run_all(size) runs the set needed by a job ('fd' or 'ps' or 'all').
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import cohorts as CH
import refcore as RC
import pscore as PS
import verdict as V

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / 'checks' / 'rng_golden.json'


def _relmax(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# --------------------------------------------------------------------------------------------------------- G1 --

def g1_parity(d):
    if d == 2:
        L, coh = 256, CH.Cohort(2, 'dev6')
        desc = coh.D[0]
        u0f = jnp.asarray(CH.ic_2d(L, desc))
        q = CH.E2.make_fom(L, 0.005)[0]
        ref, it, rn = q(u0f, float(desc[4]), 1e-10, 1e-11)
        ref = np.asarray(ref)[:, 1:-1, 1:-1].reshape(6, -1)
    else:
        L, coh = 64, CH.Cohort(3, 'dev4')
        u0 = jnp.asarray(CH.ic_3d(L, coh.T, 0))
        ref, it, rn = CH.C3.make_fom(L + 1, 0.005, 1e-10, 1e-11)(u0, coh.nu(0))
        ref = np.asarray(ref)
    run = RC.make_solver(d, L, adv='upwind')
    r = run(coh.ic(L, 0), coh.nu(0), 0.005, 'be', 1e-10, 1e-11)
    out = np.asarray(r['out'])
    nw = np.asarray(r['nwt'])[:50]
    dev = _relmax(out, ref)
    return dict(gate='G1', d=d, L=L, max_rel_diff=dev, newton_equal=bool(np.array_equal(nw, np.asarray(it))),
                **{'pass': dev <= 1e-12})


# --------------------------------------------------------------------------------------------------------- G2 --

def g2_mms(d):
    levels = {2: [(256, 0.05 / 16), (512, 0.05 / 32), (1024, 0.05 / 64)],      # A4.1
              3: [(32, 0.05 / 4), (64, 0.05 / 8), (128, 0.05 / 16)]}[d]
    L1 = levels[0][0]
    lat, errs = [], []
    x1 = np.arange(0, L1 + 1) / L1
    Xc = [jnp.asarray(a) for a in np.meshgrid(*([x1] * d), indexing='ij')]
    exact = np.stack([np.asarray(PS.mms_exact(Xc, t)) for t in RC.TIMES])        # (6, (L1+1)^d) incl. walls
    for L, dt in levels:
        X = RC.node_coords(L, d)
        u0 = PS.mms_exact(X, 0.0)
        run = RC.make_solver(d, L, adv='central', forcing=lambda t, X: PS.mms_forcing_grid(t, X))
        r = run(u0, PS.MMS_NU, dt, 'cnr', 1e-12, 1e-12)
        out = np.asarray(r['out']).reshape((6,) + (L - 1,) * d)
        s = L // L1
        full = np.pad(out, [(0, 0)] + [(1, 1)] * d)
        sl = (slice(None),) + tuple(slice(None, None, s) for _ in range(d))
        lat.append(full[sl])
    n0 = float(np.linalg.norm(exact[0].ravel()))
    e_true = [V.tnorm(u - exact)[1:] / n0 for u in lat]
    p_true = [math.log2(e_true[1][k] / e_true[2][k]) for k in range(5)]
    d12, d23 = V.tnorm(lat[0] - lat[1])[1:], V.tnorm(lat[1] - lat[2])[1:]
    p_obs = [V.solve_order(d12[k], d23[k], 1 / levels[0][0], 1 / levels[1][0], 1 / levels[2][0]) for k in range(5)]
    eps = [d23[k] / n0 * V.rich_factor(p_obs[k], 1 / levels[1][0], 1 / levels[2][0]) for k in range(5)]
    eff = [eps[k] / e_true[2][k] for k in range(5)]
    ok = all(1.9 <= p <= 2.1 for p in p_true) and all(0.5 <= x <= 2.0 for x in eff)
    return dict(gate='G2', d=d, p_true=p_true, p_obs=p_obs, effectivity=eff, e_true_finest=e_true[2].tolist(),
                **{'pass': bool(ok)})


# -------------------------------------------------------------------------------------------------------- G3 ---

G3A_MODES = {2: ([(1, 1), (3, 2), (17, 5), (40, 40)], [1, .5, -.3, .1]),
             3: ([(1, 1, 1), (2, 3, 1), (9, 4, 7), (30, 30, 30)], [1, .5, -.2, .05])}


def g3a_linear(d, N=128, nu=0.01):
    modes, coefs = G3A_MODES[d]
    b0 = np.zeros((N - 1,) * d)
    for m, c in zip(modes, coefs):
        b0[tuple(k - 1 for k in m)] = c
    m = RC.LATTICE[d]
    S = PS.lattice_S(N, m)
    r = PS.make_ps(N, d, nonlin_on=False)(jnp.asarray(b0), nu, 0.01, S)
    L = PS.lin_eig(N, d, nu)
    # exact solution synthesised INDEPENDENTLY of pscore (direct NumPy sums over the listed modes)
    m_ = RC.LATTICE[d]
    x = np.arange(1, m_) / m_
    rels = []
    for k, t in enumerate(RC.TIMES):
        ex = 0.0
        for mo, c in zip(modes, coefs):
            lam = -nu * np.pi ** 2 * sum(q * q for q in mo)
            f = c * np.exp(lam * t)
            fac = [np.sin(np.pi * q * x) for q in mo]
            ex = ex + f * (np.multiply.outer(fac[0], fac[1]) if d == 2 else
                           np.multiply.outer(np.multiply.outer(fac[0], fac[1]), fac[2]))
        ex = np.pad(ex, 1) if d == 2 else ex.reshape(-1)
        if k >= 1:
            rels.append(float(np.linalg.norm(r['lat'][k] - ex) / np.linalg.norm(ex)))
    ok = bool(r['finite'] and np.all(np.isfinite(rels)) and max(rels) <= 1e-12)
    return dict(gate='G3a', d=d, worst_rel=float(np.max(rels)) if np.all(np.isfinite(rels)) else None,
                **{'pass': ok})


def g3b_ic_quadrature(d, N):
    if d == 2:
        desc = CH.Cohort(2, 'dev6').D[0]
        a, b = PS.ic_coefs_2d(desc, N, 256), PS.ic_coefs_2d(desc, N, 512)
    else:
        T = CH.Cohort(3, 'dev4').T
        a, b = PS.ic_coefs_3d(T, 0, N, 256), PS.ic_coefs_3d(T, 0, N, 512)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    mx = float(np.max(np.abs(a - b)))
    return dict(gate='G3b', d=d, N=N, rel_l2=rel, max_abs=mx, **{'pass': rel <= 1e-12 and mx <= 1e-13})   # A4.2


def g3c_contour(d, N, dt, nu):
    Lk = jnp.asarray(PS.lin_eig(N, d, nu))
    a, b = PS.etdrk4_coefs(Lk, dt, 32), PS.etdrk4_coefs(Lk, dt, 64)
    vals = [float(jnp.max(jnp.abs(a[k] - b[k])) / jnp.max(jnp.abs(b[k]))) for k in ('Q', 'f1', 'f2', 'f3')]
    worst = max(vals) if all(np.isfinite(vals)) else float('inf')
    return dict(gate='G3c', d=d, N=N, dt=dt, nu=nu, worst_rel=worst, **{'pass': worst <= 1e-10})   # A4.2


# -------------------------------------------------------------------------------------------------------- G4 ---

def _ps_mms(N, d, dt):
    m = RC.LATTICE[d]
    S = PS.lattice_S(N, m)
    run = PS.make_ps(N, d, forcing=PS.mms_forcing_coefs(N, d))
    return run(PS.mms_ic_coefs(N, d), PS.MMS_NU, dt, S)['lat']


def _mms_lattice_exact(d):
    m = RC.LATTICE[d]
    x = np.arange(1, m) / m
    X = [jnp.asarray(a) for a in np.meshgrid(*([x] * d), indexing='ij')]
    ex = np.stack([np.asarray(PS.mms_exact(X, t)) for t in RC.TIMES])
    if d == 2:
        return np.pad(ex, ((0, 0), (1, 1), (1, 1)))
    return ex.reshape(6, -1)


def g4_mms(d):
    ex = _mms_lattice_exact(d)
    n0 = float(np.linalg.norm(ex[0].ravel()))
    Ns = [32, 48, 64]
    lat = [_ps_mms(N, d, 1e-4) for N in Ns]
    e = [V.tnorm(u - ex)[1:] / n0 for u in lat]
    q_sp = [math.log(e[1][k] / e[2][k]) / math.log(64 / 48) for k in range(5)]
    q_sp0 = [math.log(e[0][k] / e[1][k]) / math.log(48 / 32) for k in range(5)]
    half = _ps_mms(64, d, 5e-5)
    sep = float(np.max(V.tnorm(half - lat[2])[1:] / n0) / np.min(e[2]))
    dts = [2.5e-3, 1.25e-3, 6.25e-4]
    tl = [_ps_mms(64, d, dt) for dt in dts]
    d12, d23 = V.tnorm(tl[0] - tl[1])[1:], V.tnorm(tl[1] - tl[2])[1:]
    q_t = [math.log2(d12[k] / d23[k]) if d23[k] > 0 and d12[k] > 0 else None for k in range(5)]
    ok = (all(q >= 3 for q in q_sp + q_sp0) and sep <= 1e-3 and all(q is not None and 3.5 <= q <= 4.5 for q in q_t)
          and all(np.isfinite(x).all() for x in lat + tl + [half]))
    return dict(gate='G4', d=d, spatial_err=[x.tolist() for x in e], spatial_order=q_sp, spatial_order_coarse=q_sp0,
                half_step_ratio=sep,
                temporal_order=q_t, **{'pass': bool(ok)})


def g4_projection():
    """Codex's example f = sin(2 pi x) sin^2(pi y): (2,1) coefficient by the padded-grid analysis vs exact 8/(3 pi)."""
    exact = 8.0 / (3.0 * np.pi)
    errs = []
    for M in (12, 24, 48):
        x = np.arange(1, M) / M
        X, Y = np.meshgrid(x, x, indexing='ij')
        f = jnp.asarray(np.sin(2 * np.pi * X) * np.sin(np.pi * Y) ** 2)
        b = np.asarray(PS.analysis(f, M))
        errs.append(abs(b[1, 0] - exact))
    q = [math.log2(errs[0] / errs[1]), math.log2(errs[1] / errs[2])]
    return dict(gate='G4proj', errors=errs, orders=q, **{'pass': all(3.5 <= x <= 4.5 for x in q)})


# --------------------------------------------------------------------------------------------------- units -----

def u_order_solve():
    cases = [V.solve_order(0.0, 1.0, .5, .25, .125), V.solve_order(1.0, 0.0, .5, .25, .125),
             V.solve_order(np.nan, 1.0, .5, .25, .125), V.solve_order(1e6, 1.0, 3., 1.5, 1.),
             V.solve_order(1.0, 1.0, 3., 1.5, 1.), V.solve_order(1.0, 1.0, .5, .25, .125),
             V.solve_order(0.5, 1.0, .5, .25, .125), V.solve_order(1024., 1.0, .5, .25, .125)]
    p2 = V.solve_order(5.4, 1.0, 3., 1.5, 1.)
    ok = all(c is None for c in cases) and abs(p2 - 2.0) < 1e-9 and abs(V.solve_order(4.0, 1.0, .5, .25, .125) - 2) < 1e-12
    return dict(gate='Uorder', undefined=[c for c in cases], p_refined_5p4=p2, **{'pass': bool(ok)})


def _synthetic_levels(Ls, C, rng, d=2, order=1.0):
    shape = (6, 257, 257) if d == 2 else (6, 63 ** 3)
    U = rng.standard_normal(shape)
    phi = rng.standard_normal(shape)
    U[0] = 1.0
    phi[0] = 0.0
    return [dict(L=L, lat=U + C * (1.0 / L) ** order * phi, tau=0.0, steps_ok=True, umax=1.0) for L in Ls], U[0]


def u_x1b():
    rng = np.random.default_rng(1)
    out = {}
    for name, Ls in (('dyadic', (2048, 4096, 8192)), ('refined', (4096, 8192, 12288))):
        lv, t0 = _synthetic_levels(Ls, 10.0, rng)
        j = V.judge(lv, t0, nu=0.05)
        out[name] = dict(p=j['p'], outcome=j['outcome'])
    lv, t0 = _synthetic_levels((2048, 4096, 8192), 10.0, rng, order=2.0)
    ctrl = V.judge(lv, t0, nu=0.05)
    ok = (all(v['outcome'] == 'order-fail' and all(abs(p - 1) < 1e-6 for p in v['p']) for v in out.values())
          and ctrl['outcome'] == 'accepted')
    return dict(gate='X1b', **out, second_order_control=ctrl['outcome'], **{'pass': bool(ok)})


def u_peclet():
    rng = np.random.default_rng(2)
    lv, t0 = _synthetic_levels((2048, 4096, 8192), 10.0, rng, order=2.0)
    j_ok = V.judge(lv, t0, nu=0.05)
    j_bad = V.judge(lv, t0, nu=1e-4)                  # umax h1 / nu = 1/2048/1e-4 = 4.9 > 2
    return dict(gate='Upeclet', ok=j_ok['outcome'], injected=j_bad['outcome'],
                **{'pass': j_ok['outcome'] == 'accepted' and j_bad['outcome'] == 'peclet-fail'})


def rng_rows():
    s2, s3 = CH.PUBLIC_CHECK_SEEDS[2], CH.PUBLIC_CHECK_SEEDS[3]
    out = {'2d_family': CH.rowwise_2d(s2, 3, 'family').tolist(), '2d_ood': CH.rowwise_2d(s2, 3, 'ood').tolist(),
           '3d_ood_nu': CH._draw3d_table(s3, 3, 'ood')['nu'].tolist(),
           '3d_ood_w': CH._draw3d_table(s3, 3, 'ood')['w'].tolist(), 'numpy': np.__version__}
    for kind in ('family', 'ood'):
        t = CH._draw3d_table(s3, 3, kind)
        for k in ('B', 'c', 'w', 'rho', 'A', 'nu'):
            out[f'3d_{kind}_all_{k}'] = np.asarray(t[k]).tolist()
    return out


def u_rng():
    s2, s3 = CH.PUBLIC_CHECK_SEEDS[2], CH.PUBLIC_CHECK_SEEDS[3]
    prefix = (np.array_equal(CH.rowwise_2d(s2, 10, 'family')[:4], CH.rowwise_2d(s2, 4, 'family'))
              and np.array_equal(CH.rowwise_2d(s2, 10, 'ood')[:4], CH.rowwise_2d(s2, 4, 'ood'))
              and np.array_equal(CH._draw3d_table(s3, 10, 'ood')['w'][:4], CH._draw3d_table(s3, 4, 'ood')['w']))
    ood = CH.rowwise_2d(s2, 50, 'ood')
    t3 = CH._draw3d_table(s3, 20, 'ood')
    ranges = (np.all((ood[:, 2] >= .03) & (ood[:, 2] < .05)) and np.all((ood[:, 4] >= .01) & (ood[:, 4] <= .0125893))
              and np.all((t3['w'] >= .07) & (t3['w'] < .10)) and np.all((t3['nu'] >= .01) & (t3['nu'] <= .0125893)))
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else None
    now = rng_rows()
    def same(g, n):
        g, n = np.asarray(g), np.asarray(n)
        if g.shape != n.shape:
            return False
        if np.issubdtype(n.dtype, np.integer):
            return bool(np.array_equal(g, n))
        # A7.1: values derived through exp/log may differ by an ulp across NumPy builds (2.4.4 vs 2.5.0)
        return bool(np.allclose(g, n, rtol=1e-14, atol=0.0))
    gold_ok = golden is not None and set(now) <= set(golden) and all(same(golden[k], now[k]) for k in now if k != 'numpy')
    return dict(gate='Urng', prefix_stable=bool(prefix), ranges=bool(ranges), golden_match=bool(gold_ok),
                numpy=np.__version__, **{'pass': bool(prefix and ranges and gold_ok)})


def u_transforms():
    """Mode-by-mode check of pscore's synthesis/analysis against direct O(n^2) NumPy sums (independent of the FFT path)."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for M in (6, 12, 17):
        b = rng.standard_normal(M - 1)
        j = np.arange(1, M)
        k = np.arange(1, M)
        Sd = np.sin(np.pi * np.outer(j, k) / M) @ b
        Cd = np.cos(np.pi * np.outer(j, k) / M) @ b
        Ad = (2.0 / M) * np.sin(np.pi * np.outer(k, j) / M) @ b
        Sp = np.asarray(PS.sine_synth_last(jnp.asarray(b), M))
        Cp = np.asarray(PS.cos_synth_last(jnp.asarray(b), M))
        Ap = np.asarray(PS.analysis(jnp.asarray(b), M))
        for x, y in ((Sp, Sd), (Cp, Cd), (Ap, Ad)):
            v = float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
            worst = v if not np.isfinite(v) else max(worst, v)
            if not np.isfinite(worst):
                return dict(gate='Utransforms', worst_rel=None, **{'pass': False})
    # 2D derivative synthesis and lattice synthesis against direct sums
    N, M = 6, 9
    bb = rng.standard_normal((N - 1, N - 1))
    k = np.arange(1, N)
    x = np.arange(1, M) / M
    S, Cx = np.sin(np.pi * np.outer(x, k)), np.cos(np.pi * np.outer(x, k))
    u_d = S @ bb @ S.T
    g_d = (Cx * (np.pi * k)) @ bb @ S.T + S @ bb @ (Cx * (np.pi * k)).T
    nb, _ = PS.nonlinear(jnp.asarray(bb), N, 2)
    f = u_d * g_d
    A = (2.0 / M) * np.sin(np.pi * np.outer(k, np.arange(1, M)) / M)
    nb_d = -(A @ f @ A.T)
    v = float(np.max(np.abs(np.asarray(nb) - nb_d)) / np.max(np.abs(nb_d)))
    if not np.isfinite(v):
        return dict(gate='Utransforms', worst_rel=None, **{'pass': False})
    worst = max(worst, v)
    Sl = np.asarray(PS.lattice_S(N, 8))
    lat = np.asarray(PS.synth_lattice(jnp.asarray(bb), jnp.asarray(Sl), 2))
    xl = np.arange(1, 8) / 8
    Sx = np.sin(np.pi * np.outer(xl, k))
    v = float(np.max(np.abs(lat[1:-1, 1:-1] - Sx @ bb @ Sx.T)))
    if not (np.isfinite(v) and np.isfinite(lat).all()):
        return dict(gate='Utransforms', worst_rel=None, **{'pass': False})
    worst = max(worst, v)
    ok = bool(np.isfinite(worst) and worst <= 1e-12)
    return dict(gate='Utransforms', worst_rel=worst if np.isfinite(worst) else None, **{'pass': ok})


def u_ic3d():
    """Chunked 3D IC (used above 256 intervals) equals the vendor unchunked IC."""
    T = CH.Cohort(3, 'dev4').T
    a = CH.ic_3d_chunked(128, T, 1, chunk=5000)
    b = np.asarray(CH.C3.initial_interior(129, T, 1))
    dev = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    return dict(gate='Uic3d', max_rel=dev, **{'pass': dev <= 1e-14})


def u_crypto():
    import sealio as SI
    with tempfile.TemporaryDirectory() as td:
        k = Path(td) / 'k'
        k.write_text(os.urandom(32).hex() + '\n')
        k2 = Path(td) / 'k2'
        k2.write_text(os.urandom(32).hex() + '\n')
        data = SI.npz_bytes(a=np.arange(10.0))
        blob = SI.encrypt(data, k)
        ok_round = SI.decrypt(blob, k) == data
        try:
            wrong = SI.decrypt(blob, k2)
            wrong_detected = SI.sha(wrong) != SI.sha(data)
        except Exception:
            wrong_detected = True
    return dict(gate='Ucrypto', round_trip=bool(ok_round), wrong_key_detected=bool(wrong_detected),
                **{'pass': bool(ok_round and wrong_detected)})


# ---------------------------------------------------------------------------------------------------- runner ---

def run_all(which, ps_N=None, ps_dt=None, ps_nu=0.01, dims=(2, 3), units=True):
    """which: 'fd' (G1, G2, units), 'ps' (G3, G4, units), 'units', 'all'. ps_N: {d: largest N used} for G3b/G3c."""
    res = []
    t = time.perf_counter()
    for f in ([u_order_solve, u_x1b, u_peclet, u_rng, u_crypto, u_transforms, u_ic3d] if units else []):
        res.append(f())
    if which in ('fd', 'all'):
        for d in dims:
            res.append(g1_parity(d))
            res.append(g2_mms(d))
    if which in ('ps', 'all'):
        res.append(g4_projection())
        for d in dims:
            res.append(g3a_linear(d))
            res.append(g4_mms(d))
            if ps_N:
                res.append(g3b_ic_quadrature(d, ps_N[d]))
                res.append(g3c_contour(d, ps_N[d], ps_dt[d], ps_nu))
    return dict(gates=res, all_pass=all(r['pass'] for r in res), seconds=time.perf_counter() - t)


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'units'
    if which == 'golden':
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(rng_rows(), indent=1) + '\n')
        print('wrote', GOLDEN)
    else:
        r = run_all(which)
        print(json.dumps(r, indent=1, default=float))
