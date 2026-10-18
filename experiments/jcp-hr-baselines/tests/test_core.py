"""Small-mesh unit gates (CPU): G-fd-a/b/c, N-fd, G-full, G-nnls, G-qdeim, G-pod, FOM vs refcore, training-draw
provenance. Run:  JAX_PLATFORMS=cpu python tests/test_core.py   (prints one line per gate; exits non-zero on failure)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'e0'))

import numpy as np
import scipy.linalg as sla
import scipy.optimize as so
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import hrcore as H
import rom as RM
import build as BD
import nnls as NN
import refcore as RC        # byte-identical E0 solver (via the cohorts shim)

RES = {}


def gate(name):
    def deco(f):
        def run():
            try:
                v = f()
                RES[name] = ('PASS', v)
            except Exception as e:  # noqa: BLE001
                RES[name] = ('FAIL', f'{type(e).__name__}: {e}')
                traceback.print_exc()
            print(f'{name}: {RES[name][0]} {RES[name][1]}', flush=True)
        run.__name__ = name
        return run
    return deco


def rnd_basis(n, r, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, r)))
    return jnp.asarray(q)


def smooth_state(d, L, seed=1, amp=1.0):
    fam = _fam(d)
    return jnp.asarray(fam.ic(L, seed % fam.n)) * amp, fam.nu(seed % fam.n)


_FAMS = {}


def _fam(d):
    if d not in _FAMS:
        _FAMS[d] = H.Family(d, 'val64', count=4) if d == 3 else H.Family(2, 'dev6')
    return _FAMS[d]


@gate('provenance-train-draw-2d')
def t_draw():
    sys.path.insert(0, str(HERE.parents[1] / 'experiments' / 'wave2d-rom-latent-stepping' / 'deps' / 'burgers2d-coord-rom'))
    import burgers2d_film as bf
    cx, cy, w, a, nu, _z = bf.sample_params(seed=0, m=576)
    ref = np.stack([cx, cy, w, a, nu], 1)
    D = H.train_draw_2d()
    assert np.array_equal(ref, D)
    return 'equal'


@gate('disjoint-train-cohorts')
def t_disjoint():
    return H.assert_disjoint(2) and H.assert_disjoint(3)


@gate('G-fd-a operators vs refcore')
def t_ops():
    out = []
    for d, L in ((2, 32), (3, 12)):
        u, nu = smooth_state(d, L)
        g = u.reshape(H.shape_of(d, L))
        a = H.F_flat(u, nu, d, L)
        b = (RC.adv_central(g, L) - nu * RC.lap(g, L)).reshape(-1)
        e1 = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
        idx = H.stencil_index(np.arange((L - 1) ** d), d, L)
        us = jnp.concatenate([u, jnp.zeros(1)])[jnp.asarray(idx)]
        Fs, _ = H.stencil_F(us, nu, L, d)
        e2 = float(jnp.max(jnp.abs(Fs - b)) / jnp.max(jnp.abs(b)))
        # stencil Jacobian vs jacfwd of the refcore operator
        Jr = jax.jacfwd(lambda v: (RC.adv_central(v.reshape(H.shape_of(d, L)), L)
                                   - nu * RC.lap(v.reshape(H.shape_of(d, L)), L)).reshape(-1))(u)
        dF, _ = H.stencil_dF(us, nu, L, d)
        Js = np.zeros(Jr.shape)
        n = (L - 1) ** d
        for s in range(2 * d + 1):
            ok = idx[:, s] < n
            Js[np.arange(n)[ok], idx[ok, s]] += np.asarray(dF[ok, s])
        e3 = float(np.max(np.abs(Js - np.asarray(Jr))) / np.max(np.abs(np.asarray(Jr))))
        assert e1 <= 1e-12 and e2 <= 1e-12 and e3 <= 1e-11, (d, e1, e2, e3)
        out.append((d, e1, e2, e3))
    return out


def arm_data(kind, d, L, Phi, nu_rows=None, seed=3):
    n, r = Phi.shape
    if kind == 'none':
        return BD.none_data(Phi, d, L), None
    if kind == 'ecsw':
        rng = np.random.default_rng(seed)
        nodes = np.sort(rng.choice(n, max(r + 5, n // 3), replace=False))
        w = rng.uniform(0.5, 2.0, len(nodes))
        dat, _ = BD.ecsw_data(Phi, nodes, w, d, L)
        return dict(dat, Phi=Phi), (nodes, w)
    U = rnd_basis(n, min(2 * r, n), seed=seed + 1)
    P = BD.qdeim_points(U)
    LPhi = H.lap_cols(Phi, d, L)
    dat, _ = BD.qdeim_data(Phi, LPhi, U, P, d, L)
    return dict(dat, Phi=Phi), (U, P, LPhi)


def resid_args(d, L, Phi, nu, seed=4):
    rng = np.random.default_rng(seed)
    r = Phi.shape[1]
    u0, _ = smooth_state(d, L, seed=seed)
    c = Phi.T @ u0
    cn = c + 0.01 * jnp.asarray(rng.standard_normal(r))
    cm = cn + 0.01 * jnp.asarray(rng.standard_normal(r))
    co = jnp.asarray([1.5, -2., .5, 1., 0.]) if seed % 2 else jnp.asarray([1., -1., 0., .5, .5])
    return c, cn, cm, co, 0.01


@gate('G-fd-b online Jacobians (FD + jacfwd), every arm')
def t_fd():
    out = []
    for d, L in ((2, 24), (3, 10)):
        n = (L - 1) ** d
        u0, nu = smooth_state(d, L)
        Phi = jnp.linalg.qr(jnp.concatenate([u0[:, None], rnd_basis(n, 11)], 1))[0]
        for kind in RM.ARMS:
            dat, _ = arm_data(kind, d, L, Phi)
            for seed in (4, 5):
                c, cn, cm, co, dt = resid_args(d, L, Phi, nu, seed)
                sc, rJ = RM.make_arm(kind, d, L)
                const = sc(cn, cm, nu, co, dt, dat)
                r, J = rJ(c, const, nu, co, dt, dat)
                Ja = jax.jacfwd(lambda x: rJ(x, const, nu, co, dt, dat)[0])(c)
                ej = float(jnp.max(jnp.abs(J - Ja)) / jnp.max(jnp.abs(Ja)))
                rng = np.random.default_rng(seed)
                efd = 0.
                for _ in range(5):
                    v = jnp.asarray(rng.standard_normal(len(c)))
                    v = v / jnp.linalg.norm(v)
                    h = 1e-6 * max(1., float(jnp.linalg.norm(c)))
                    fd = (rJ(c + h * v, const, nu, co, dt, dat)[0] - rJ(c - h * v, const, nu, co, dt, dat)[0]) / (2 * h)
                    efd = max(efd, float(jnp.linalg.norm(fd - J @ v) / jnp.linalg.norm(J @ v)))
                assert ej <= 1e-11 and efd <= 1e-6, (d, kind, ej, efd)
                out.append((d, kind, f'{ej:.1e}', f'{efd:.1e}'))
    return out


@gate('N-fd mutant Jacobian fails the FD check')
def t_nfd():
    out = []
    for d, L in ((2, 24), (3, 10)):
        n = (L - 1) ** d
        u0, nu = smooth_state(d, L)
        Phi = jnp.linalg.qr(jnp.concatenate([u0[:, None], rnd_basis(n, 11)], 1))[0]
        for kind in RM.ARMS:
            dat, _ = arm_data(kind, d, L, Phi)
            c, cn, cm, co, dt = resid_args(d, L, Phi, nu, 4)
            sc, rJ = RM.make_arm(kind, d, L)
            _, rJm = RM.make_arm(kind, d, L, mutation='drop_adv_jac')
            const = sc(cn, cm, nu, co, dt, dat)
            _, Jm = rJm(c, const, nu, co, dt, dat)
            v = jnp.asarray(np.random.default_rng(0).standard_normal(len(c)))
            v = v / jnp.linalg.norm(v)
            h = 1e-6 * max(1., float(jnp.linalg.norm(c)))
            fd = (rJ(c + h * v, const, nu, co, dt, dat)[0] - rJ(c - h * v, const, nu, co, dt, dat)[0]) / (2 * h)
            e = float(jnp.linalg.norm(fd - Jm @ v) / jnp.linalg.norm(fd))
            assert e > 1e-3, (d, kind, e)
            out.append((d, kind, f'{e:.1e}'))
    return out


@gate('G-fd-c ECSW rows and Q-DEIM objective vs independent assembly')
def t_fdc():
    out = []
    for d, L in ((2, 24), (3, 10)):
        n = (L - 1) ** d
        u0, nu = smooth_state(d, L)
        Phi = jnp.linalg.qr(jnp.concatenate([u0[:, None], rnd_basis(n, 11)], 1))[0]
        c, cn, cm, co, dt = resid_args(d, L, Phi, nu, 4)
        a0, a1, a2, b0, b1 = [float(x) for x in co]
        sh = H.shape_of(d, L)
        Fr = lambda u: (RC.adv_central(u.reshape(sh), L) - nu * RC.lap(u.reshape(sh), L)).reshape(-1)
        u, un, um = Phi @ c, Phi @ cn, Phi @ cm
        Rfull = a0 * u + a1 * un + a2 * um + dt * (b0 * Fr(u) + b1 * Fr(un))
        # ECSW rows
        dat, (nodes, w) = arm_data('ecsw', d, L, Phi)
        sc, rJ = RM.make_arm('ecsw', d, L)
        r, _ = rJ(c, sc(cn, cm, nu, co, dt, dat), nu, co, dt, dat)
        e1 = float(jnp.max(jnp.abs(r - jnp.sqrt(w) * Rfull[nodes])) / jnp.max(jnp.abs(Rfull)))
        # Q-DEIM: ||R_B z|| and J^T r against the assembled full-field DEIM residual
        datq, (U, P, LPhi) = arm_data('qdeim', d, L, Phi)
        scq, rJq = RM.make_arm('qdeim', d, L)
        NPf = lambda cc: RC.adv_central((Phi @ cc).reshape(sh), L).reshape(-1)[P]
        Minv = jnp.linalg.inv(U[P])
        Rd = lambda cc: (Phi @ (a0 * cc + a1 * cn + a2 * cm) - dt * nu * (LPhi @ (b0 * cc + b1 * cn))
                         + U @ (Minv @ (dt * (b0 * NPf(cc) + b1 * NPf(cn)))))
        constq = scq(cn, cm, nu, co, dt, datq)
        rq, Jq = rJq(c, constq, nu, co, dt, datq)
        e2 = abs(float(jnp.linalg.norm(rq)) - float(jnp.linalg.norm(Rd(c)))) / float(jnp.linalg.norm(Rd(c)))
        g_ass = jax.grad(lambda cc: 0.5 * jnp.sum(Rd(cc) ** 2))(c)
        e3 = float(jnp.linalg.norm(Jq.T @ rq - g_ass) / jnp.linalg.norm(g_ass))
        # near-cancellation state: minimise the assembled DEIM residual first (tight GN), then compare again
        cs = c
        for _ in range(8):
            Ja = jax.jacfwd(Rd)(cs)
            cs = cs - jnp.linalg.lstsq(Ja, Rd(cs))[0]
        rq2, Jq2 = rJq(cs, constq, nu, co, dt, datq)
        e4 = abs(float(jnp.linalg.norm(rq2)) - float(jnp.linalg.norm(Rd(cs)))) / float(jnp.linalg.norm(Rd(cs)))
        g2 = jax.grad(lambda cc: 0.5 * jnp.sum(Rd(cc) ** 2))(cs)
        e5 = float(jnp.linalg.norm(Jq2.T @ rq2 - g2)) / (float(jnp.linalg.norm(Jq2)) * float(jnp.linalg.norm(rq2)))
        assert e1 <= 1e-10 and e2 <= 1e-10 and e3 <= 1e-10 and e4 <= 1e-10 and e5 <= 1e-10, (d, e1, e2, e3, e4, e5)
        out.append((d, f'{e1:.1e}', f'{e2:.1e}', f'{e3:.1e}', f'{e4:.1e}', f'{e5:.1e}'))
    return out


@gate('FOM (hrcore) vs refcore.make_solver, BE and CN')
def t_fom():
    out = []
    for d, L in ((2, 32), (3, 12)):
        u0, nu = smooth_state(d, L)
        fom = H.make_fom(d, L)
        ref = RC.make_solver(d, L, adv='central')
        for sc_h, sc_r in (('BE', 'be'), ('CN', 'cn')):
            dt = 0.01
            a = fom(u0, nu, H.sched(sc_h, dt), 1e-12, 1e-14)['out']
            b = ref(u0, nu, dt, sc_r, 1e-12, 1e-14)['out']
            e = float(jnp.max(jnp.linalg.norm(a - b, axis=1)) / jnp.linalg.norm(b[0]))
            assert e <= 1e-10, (d, sc_h, e)
            out.append((d, sc_h, f'{e:.1e}'))
    return out


def complete_basis(n, seed=7):
    return rnd_basis(n, n, seed)


@gate('G-full complete basis reproduces the FOM (every arm, option, solver)')
def t_full():
    out = []
    for d, L in ((2, 16), (3, 8)):
        n = (L - 1) ** d
        fam = _fam(d)
        u0 = jnp.asarray(fam.ic(L, 0))
        nu = fam.nu(0)
        Phi = complete_basis(n)
        datas = {'none': BD.none_data(Phi, d, L)}
        de, _ = BD.ecsw_data(Phi, np.arange(n), np.ones(n), d, L)
        datas['ecsw'] = dict(de, Phi=Phi)
        U = jnp.eye(n)
        dq, _ = BD.qdeim_data(Phi, H.lap_cols(Phi, d, L), U, np.arange(n), d, L)
        datas['qdeim'] = dict(dq, Phi=Phi)
        fom = H.make_fom(d, L)
        opts = H.OPTIONS[d] if d == 2 else [('BE', 0.01), ('CN', 0.01), ('BDF2', 0.01), ('BDF2', 0.025)]
        refs = {o: fom(u0, nu, H.sched(*o), 1e-13, 1e-15)['out'] for o in opts}
        for kind in RM.ARMS:
            for solver in ('native', 'common'):
                q = RM.make_query(kind, d, L, solver, trust=1e9)          # one compilation per (kind, solver)
                for sc_, dt in opts:
                    o = q(u0, nu, datas[kind], RM.schedule(d, sc_, dt, 'recover'))
                    ref = refs[(sc_, dt)]
                    e = float(jnp.max(jnp.linalg.norm(o['fields'] - ref, axis=1)) / jnp.linalg.norm(ref[0]))
                    assert e <= 1e-8, (d, sc_, dt, kind, solver, e)
                    out.append(e)
    return f'{len(out)} runs, worst {max(out):.1e}'


@gate('G-nnls greedy NNLS vs scipy (objective, KKT, tau stop)')
def t_nnls():
    rng = np.random.default_rng(0)
    out = []
    for m, N in ((60, 40), (200, 120), (400, 250)):
        G = rng.standard_normal((m, N))
        b = G @ np.abs(rng.standard_normal(N)) * (rng.uniform(size=N) < 0.3) if False else rng.standard_normal(m)
        b = b + G @ np.where(rng.uniform(size=N) < 0.2, rng.uniform(size=N), 0.)
        x_s, rs = so.nnls(G, b, maxiter=50 * N)
        o = NN.greedy_nnls(G, b, taus=(), cap=N)
        x = np.zeros(N)
        x[o['support']] = o['weights']
        r_g = np.linalg.norm(G @ x - b)
        assert abs(r_g - rs) <= 1e-10 * max(rs, 1.), (m, N, r_g, rs)
        gr = G.T @ (G @ x - b)
        on = x > 0
        assert np.all(x >= 0) and np.all(np.abs(gr[on]) <= 1e-9 * np.linalg.norm(G) * np.linalg.norm(b)) \
            and np.all(gr[~on] >= -1e-9 * np.linalg.norm(G) * np.linalg.norm(b)), 'KKT'
        out.append(f'{abs(r_g - rs):.1e}')
    # tau checkpoints
    G = rng.standard_normal((300, 500))
    b = G @ np.where(rng.uniform(size=500) < 0.1, rng.uniform(size=500), 0.)
    o = NN.greedy_nnls(G, b, taus=(1e-1, 1e-2, 1e-3), cap=500)
    for t, rr in o['records'].items():
        assert rr is not None and rr['relres'] <= t and np.all(rr['weights'] >= 0)
        x = np.zeros(500)
        x[rr['support']] = rr['weights']
        assert abs(np.linalg.norm(G @ x - b) / np.linalg.norm(b) - rr['relres']) <= 1e-10
    out.append({t: o['records'][t]['size'] for t in o['records']})
    return out


@gate('G-qdeim pivots vs scipy, exact interpolation on span(U)')
def t_qdeim():
    rng = np.random.default_rng(1)
    out = []
    for n, p in ((400, 12), (900, 40)):
        U = np.linalg.qr(rng.standard_normal((n, p)))[0]
        P = BD.qdeim_points(jnp.asarray(U))
        _, _, piv = sla.qr(U.T, pivoting=True, mode='economic')
        assert np.array_equal(np.asarray(P), piv[:p]), 'pivots differ'
        f = U @ rng.standard_normal(p)
        fi = U @ np.linalg.solve(U[P], f[P])
        e = float(np.linalg.norm(fi - f) / np.linalg.norm(f))
        assert e <= 1e-10
        out.append((n, p, f'{e:.1e}', f'{1 / np.linalg.svd(U[P], compute_uv=False)[-1]:.2e}'))
    return out


@gate('G-pod orthonormality, snapshot reproduction and POD optimality')
def t_pod():
    d, L = 2, 16
    fam = H.Family(2, 'train')
    S, info = BD.snapshots(d, L, fam, list(range(20)), log=None)
    Phi, sig, dev = BD.pod(S, 60, nb=64)
    pe, C = BD.projection_errors(S, Phi, [10, 30, 60], sig, nb=64)
    assert dev <= 1e-10, dev
    rank = int(np.sum(sig > 1e-8 * sig[0]))
    Pf, sf, _ = BD.pod(S, min(rank, 200), nb=64)
    rec = float(np.max(np.linalg.norm(S - (S @ np.asarray(Pf)) @ np.asarray(Pf).T, axis=1) / np.linalg.norm(S, axis=1)))
    assert rec <= 1e-6, rec
    for r, v in pe.items():
        assert abs(v['frob_rel'] - v['pod_tail']) <= 0.01 * v['pod_tail'] + 1e-6, (r, v)
    return dict(dev=dev, rec=rec, pe={r: (round(v['frob_rel'], 6), round(v['pod_tail'], 6)) for r, v in pe.items()})


@gate('G-lmm step equations (refcore operators) and temporal order, every scheme')
def t_lmm():
    out = []
    for d, L in ((2, 16), (3, 8)):
        u0, _ = smooth_state(d, L)
        nu = 0.1                       # frozen order fixture (A3.12): case 0 IC, nu = 0.1, T = 0.25
        sh = H.shape_of(d, L)
        Fr = lambda u: (RC.adv_central(u.reshape(sh), L) - nu * RC.lap(u.reshape(sh), L)).reshape(-1)
        fom = H.make_fom(d, L)
        for sc_ in ('BE', 'CN', 'CNR', 'BDF2'):
            a0, a1, a2, b0, b1, ns = H.SCHEMES[sc_]
            dt = 0.01
            N = 25
            U = np.asarray(fom(u0, nu, H.sched(sc_, dt, save_steps=tuple(range(N + 1))), 1e-13, 1e-15)['out'])
            worst = 0.
            for k in range(1, N + 1):
                co = (1., -1., 0., 1., 0.) if k <= ns else (a0, a1, a2, b0, b1)
                um = U[max(k - 2, 0)]
                R = co[0] * U[k] + co[1] * U[k - 1] + co[2] * um + dt * (co[3] * np.asarray(Fr(jnp.asarray(U[k])))
                                                                         + co[4] * np.asarray(Fr(jnp.asarray(U[k - 1]))))
                worst = max(worst, float(np.linalg.norm(R) / np.linalg.norm(U[k - 1])))
            assert worst <= 1e-11, (d, sc_, worst)
            fin = lambda dt_: np.asarray(fom(u0, nu, H.sched(sc_, dt_, save_steps=(int(round(.25 / dt_)),)),
                                             1e-13, 1e-15)['out'][0])
            # self-convergence (A3.12): p = log2(|u(dt) - u(dt/2)| / |u(dt/2) - u(dt/4)|), final state, discrete L2
            us = [fin(x) for x in (0.01, 0.005, 0.0025, 0.00125)]
            dif = [np.linalg.norm(us[i] - us[i + 1]) for i in range(3)]
            orders = [float(np.log2(dif[i] / dif[i + 1])) for i in range(2)]
            lo, hi = (0.9, 1.1) if sc_ == 'BE' else (1.8, 2.2)
            assert all(lo <= o <= hi for o in orders), (d, sc_, orders)
            out.append((d, sc_, f'{worst:.1e}', [round(o, 3) for o in orders]))
    return out


@gate('N-dom decision table on synthetic data (A2.11)')
def t_ndom():
    import stats as ST
    rng = np.random.default_rng(0)
    n = 38
    eX = np.exp(rng.uniform(np.log(1e-3), np.log(1e-2), n))
    eps = np.full(n, 1e-6)
    sig = np.where(np.arange(n) % 2 == 0, 1., -1.)
    def rho_of(v):
        return {c: [v * (1 + 0.01 * s) for s in (1., -1., 1.)] if c % 2 == 0 else [v * (1 + 0.01 * s) for s in (-1., 1., -1.)]
                for c in range(6)}
    a = ST.pair_label(eX, 0.5 * eX, eps, rho_of(0.5))
    b = ST.pair_label(eX, eX * (1 + 0.005 * sig), eps, rho_of(1.0))
    eB = 0.5 * eX.copy()
    eB[3] = 2 * eX.max()
    c = ST.pair_label(eX, eB, eps, rho_of(0.5))
    z = ST.pair_label(eX, 0.5 * eX, eps, {0: [0.5], 1: [0.5]})
    assert a['label'].startswith('baseline dominates'), a['label']
    assert b['label'] == 'matches', (b['label'], b.get('time'))
    assert not c['label'].startswith('baseline dominates'), c['label']
    assert z['label'] == 'not assessed', z['label']
    return [a['label'], b['label'], c['label'], z['label']]


if __name__ == '__main__':
    allf = (t_draw, t_disjoint, t_ops, t_fd, t_nfd, t_fdc, t_fom, t_lmm, t_full, t_nnls, t_qdeim, t_pod, t_ndom)
    sel = sys.argv[1:]
    for f in allf:
        if not sel or any(k in f.__name__ for k in sel):
            f()
    bad = [k for k, v in RES.items() if v[0] != 'PASS']
    print('ALL PASS' if not bad else f'FAILED: {bad}')
    sys.exit(1 if bad else 0)
