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
                resf = RM.make_res(kind, d, L)
                const = sc(cn, cm, nu, co, dt, dat)
                r, J = rJ(c, const, nu, co, dt, dat)
                er = float(jnp.max(jnp.abs(resf(c, const, nu, co, dt, dat) - r)) / jnp.max(jnp.abs(r)))
                assert er <= 1e-14, (d, kind, 'make_res differs from rJ', er)
                Ja = jax.jacfwd(lambda x: resf(x, const, nu, co, dt, dat))(c)
                ej = float(jnp.max(jnp.abs(J - Ja)) / jnp.max(jnp.abs(Ja)))
                rng = np.random.default_rng(seed)
                efd = 0.
                for _ in range(5):
                    v = jnp.asarray(rng.standard_normal(len(c)))
                    v = v / jnp.linalg.norm(v)
                    h = 1e-6 * max(1., float(jnp.linalg.norm(c)))
                    fd = (rJ(c + h * v, const, nu, co, dt, dat)[0] - rJ(c - h * v, const, nu, co, dt, dat)[0]) / (2 * h)
                    efd = max(efd, float(jnp.linalg.norm(fd - J @ v) / jnp.linalg.norm(J @ v)), key=lambda x: (not np.isfinite(x), x))
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

        def parts(cc):
            return (Phi @ (a0 * cc + a1 * cn + a2 * cm), -dt * nu * (LPhi @ (b0 * cc + b1 * cn)),
                    U @ (Minv @ (dt * (b0 * NPf(cc) + b1 * NPf(cn)))))

        def sig_err(cc):                     # A2.1 cancellation-safe bounds, Sigma = sum of the assembled terms' norms
            t1, t2, t3 = parts(cc)
            Sig = float(jnp.linalg.norm(t1) + jnp.linalg.norm(t2) + jnp.linalg.norm(t3))
            rq, Jq = rJq(cc, constq, nu, co, dt, datq)
            g_ass = jax.grad(lambda x: 0.5 * jnp.sum(Rd(x) ** 2))(cc)
            return (abs(float(jnp.linalg.norm(rq)) - float(jnp.linalg.norm(Rd(cc)))) / Sig,
                    float(jnp.linalg.norm(Jq.T @ rq - g_ass)) / (Sig * float(jnp.linalg.norm(Jq))),
                    float(jnp.linalg.norm(Rd(cc))) / Sig)
        e2, e3, _ = sig_err(c)
        cs = c
        for _ in range(8):                   # the frozen cancellation candidate (A2.1)
            Ja = jax.jacfwd(Rd)(cs)
            cs = cs - jnp.linalg.lstsq(Ja, Rd(cs))[0]
        e4, e5, ratio_c = sig_err(cs)
        assert e1 <= 1e-10 and e2 <= 1e-10 and e3 <= 1e-10 and e4 <= 1e-10 and e5 <= 1e-10, (d, e1, e2, e3, e4, e5)
        out.append((d, 'cancel_ratio', f'{ratio_c:.1e}'))
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
        opts = H.OPTIONS[d]                                    # every registered option (code audit 1, item 16)
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
    # support checkpoints (A15.1, A15.6): first crossing, actual size >= target, exact residual, weights > 0;
    # drop-count banning (A15.3) with dependent (duplicated) columns still reaches the NNLS optimum
    o2 = NN.greedy_nnls(G, b, supports=(5, 12, 20), cap=60)
    for k in (5, 12, 20):
        rr = o2['records'][('s', k)]
        assert rr is not None and rr['size'] >= k and np.all(rr['weights'] > 0), (k, rr)
        x = np.zeros(500)
        x[rr['support']] = rr['weights']
        assert abs(np.linalg.norm(G @ x - b) / np.linalg.norm(b) - rr['relres']) <= 1e-10
    Gd = np.concatenate([G[:, :50], G[:, :50]], 1)
    o3 = NN.greedy_nnls(Gd, b, taus=(), cap=100)
    xs, rs_ = so.nnls(Gd, b, maxiter=10000)
    x3 = np.zeros(100)
    x3[o3['support']] = o3['weights']
    assert abs(np.linalg.norm(Gd @ x3 - b) - rs_) <= 1e-8 * max(rs_, 1.), (np.linalg.norm(Gd @ x3 - b), rs_)
    out.append(dict(supports={k: o2['records'][('s', k)]['size'] for k in (5, 12, 20)}, dup_banned=o3['banned'],
                    dup_reason=o3['reason']))
    # A20: the updating-QR path (default) equals the full-QR path (same supports at every checkpoint, residuals to
    # 1e-10 relative), including with frequent refactorisation and deletions
    G4 = rng.standard_normal((400, 900))
    b4 = G4 @ np.where(rng.uniform(size=900) < 0.05, rng.uniform(size=900), 0.) + 0.05 * rng.standard_normal(400)
    ou = NN.greedy_nnls(G4, b4, supports=(10, 40, 120), cap=150, method='update', refactor_every=16)
    of = NN.greedy_nnls(G4, b4, supports=(10, 40, 120), cap=150, method='full')
    for k in (10, 40, 120):
        a, c = ou['records'][('s', k)], of['records'][('s', k)]
        assert a is not None and c is not None
        assert np.array_equal(np.sort(a['support']), np.sort(c['support'])), k
        assert abs(a['relres'] - c['relres']) <= 1e-10 * c['relres'], (k, a['relres'], c['relres'])
    assert ou['qr_drift'] and max(ou['qr_drift']) <= 1e-10, ou['qr_drift']
    out.append(dict(update_vs_full='identical supports', drift=max(ou['qr_drift']), refactors=len(ou['qr_drift'])))
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
    rank = int(np.sum(sig > 1e-8 * sig[0]))          # A12.6: the Gram method resolves sigma only above ~sqrt(eps) sigma_1
    Pf, sf, _ = BD.pod(S, rank, nb=64)
    rec = float(np.max(np.linalg.norm(S - (S @ np.asarray(Pf)) @ np.asarray(Pf).T, axis=1) / np.linalg.norm(S, axis=1)))
    assert rec <= 1e-6, rec
    for r, v in pe.items():
        assert abs(v['frob_rel'] - v['pod_tail']) <= 0.01 * v['pod_tail'] + 1e-6, (r, v)
    return dict(dev=dev, rec=rec, pe={r: (round(v['frob_rel'], 6), round(v['pod_tail'], 6)) for r, v in pe.items()})


@gate('G-lmm step equations (refcore operators) and temporal order, every scheme')
def t_lmm():
    out = []
    for d, L in ((2, 16), (3, 8)):
        u0, _ = smooth_state(d, L, seed=0)
        nu = 0.1                       # frozen order fixture (A3.8): case 0 IC, nu = 0.1, T = 0.25
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
                worst = max(worst, float(np.linalg.norm(R) / np.linalg.norm(U[k - 1])), key=lambda x: (not np.isfinite(x), x))
            assert worst <= 1e-11, (d, sc_, worst)
            fin = lambda dt_: np.asarray(fom(u0, nu, H.sched(sc_, dt_, save_steps=(int(round(.25 / dt_)),)),
                                             1e-13, 1e-15)['out'][0])
            # self-convergence (A3.8, A4.6): p = log2(|u(dt) - u(dt/2)| / |u(dt/2) - u(dt/4)|) on the final state; the
            # finest estimator decides; if it fails, one more level (dt = 0.000625) and the next estimator decides
            lo, hi = (0.9, 1.1) if sc_ == 'BE' else (1.8, 2.2)
            dts = [0.01, 0.005, 0.0025, 0.00125]
            us = [fin(x) for x in dts]
            orders = [float(np.log2(np.linalg.norm(us[i] - us[i + 1]) / np.linalg.norm(us[i + 1] - us[i + 2]))) for i in range(2)]
            if not (lo <= orders[-1] <= hi):
                us.append(fin(0.000625))
                orders.append(float(np.log2(np.linalg.norm(us[2] - us[3]) / np.linalg.norm(us[3] - us[4]))))
            assert lo <= orders[-1] <= hi, (d, sc_, orders)
            status = 'pre-asymptotic at 0.01' if not (lo <= orders[0] <= hi) else 'asymptotic'
            out.append((d, sc_, f'{worst:.1e}', [round(o, 3) for o in orders], status))
    return out


@gate('N-dom decision table on synthetic data (A2.11, A3.3, A4.2; hand-derived labels)')
def t_ndom():
    import stats as ST
    rng = np.random.default_rng(0)
    n = 38
    tc = list(range(6))
    eX = np.exp(rng.uniform(np.log(1e-3), np.log(1e-2), n))
    eps = np.full(n, 1e-6)
    sig = np.where(np.arange(n) % 2 == 0, 1., -1.)

    def rho_of(v, cases=tc):
        return {c: ([v * 1.01, v * 0.99, v * 1.01] if c % 2 == 0 else [v * 0.99, v * 1.01, v * 0.99]) for c in cases}
    out = {}
    out['a'] = ST.pair_label(eX, 0.5 * eX, eps, rho_of(0.5), tc)['label']
    out['b'] = ST.pair_label(eX, eX * (1 + 0.005 * sig), eps, rho_of(1.0), tc)['label']
    eB = 0.5 * eX.copy()
    eB[3] = 2 * eX.max()
    out['c'] = ST.pair_label(eX, eB, eps, rho_of(0.5), tc)['label']
    e5 = np.full(n, 5e-5)
    out['d1'] = ST.pair_label(e5, e5, np.full(n, 5e-5), {c: [1., 1., 1.] for c in tc}, tc)['label']
    e2 = np.full(n, 1e-2)
    out['d2'] = ST.pair_label(e2, e2, np.full(n, 1e-6), {c: [1., 1., 1.] for c in tc}, tc)['label']
    for k in (2, 3, 4, 5):
        out[f'e{k}'] = ST.pair_label(eX, 0.5 * eX, eps, rho_of(0.5, list(range(k))), tc)['label']
    # timing estimand (A3.9): per-case medians, then the median over cases
    t = ST.time_stats({0: [.5, .5, .5], 1: [.5, .5, .5], 2: [.5], 3: [2.], 4: [2.], 5: [2.]})
    want = dict(a='baseline dominates', b='matches', d1='unresolved', d2='matches',
                e2='not assessed', e3='not assessed', e4='not assessed', e5='not assessed')
    for k, v in want.items():
        assert out[k] == v, (k, out[k], v)
    assert not out['c'].startswith('baseline dominates'), out['c']
    assert abs(t['median'] - 1.25) < 1e-12, t['median']
    return out

if __name__ == '__main__':
    allf = (t_draw, t_disjoint, t_ops, t_fd, t_nfd, t_fdc, t_fom, t_lmm, t_full, t_nnls, t_qdeim, t_pod, t_ndom)
    sel = sys.argv[1:]
    for f in allf:
        if not sel or any(k in f.__name__ for k in sel):
            f()
    bad = [k for k, v in RES.items() if v[0] != 'PASS']
    print('ALL PASS' if not bad else f'FAILED: {bad}')
    if not sel:
        import hashlib
        import json
        srcs = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in
                ('hrcore.py', 'rom.py', 'build.py', 'nnls.py', 'stats.py', 'pipeline.py', 'tests/test_core.py')}
        (HERE / 'checks').mkdir(exist_ok=True)
        (HERE / 'checks' / 'test_core.json').write_text(json.dumps(dict(all_pass=not bad, source_sha256=srcs,
            gates={k: [v[0], str(v[1])[:2000]] for k, v in RES.items()}), indent=1))
    sys.exit(1 if bad else 0)
