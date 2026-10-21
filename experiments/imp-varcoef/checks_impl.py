"""Implementation checks (DESIGN §10, A1.19, A2.19): I-C1, I-sym, I-quad, I-mms, T-loose, T-nan.

    python checks_impl.py --out checks/impl-<tag>.json [--small]

--small uses tiny sizes (local CPU sanity); the real-size smoke uses the defaults (N = 48/64, L = 128-256).
Every check returns a dict with its measured value, the bar and pass/fires.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vcoef as VC     # noqa: E402
import vspec as VS     # noqa: E402
import vfd as VF       # noqa: E402
import psjax2d         # noqa: E402
import refcore as FC   # noqa: E402


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-300))


def lat_dist(F, G):
    """Scoring-lattice relative distance: max_k>=1 |F_k - G_k| / |G_0|."""
    F, G = np.asarray(F), np.asarray(G)
    n0 = np.linalg.norm(G[0].ravel())
    return float(max(np.linalg.norm((F[k] - G[k]).ravel()) for k in range(1, 6)) / n0)


def i_c1(case, N, dt, L, dtfd):
    kp1 = jnp.asarray(VC.kappa_params(case, 1.0))
    nu = float(case['desc'][4])
    b0 = VS.ic(case['desc'], N)
    ref = np.asarray(psjax2d.make_ps_query(N, dt)(b0, nu)[0])
    etd = np.asarray(VS.make_etd(N, dt, 'nu')(b0, kp1)[0])
    out = dict(etd_vs_ps=dict(value=rel(etd, ref), bar=1e-10))
    out['etd_vs_ps']['pass'] = out['etd_vs_ps']['value'] <= 1e-10
    res = {}
    for sch in ('SBDF2', 'CNAB2'):
        F, st = VS.make_psi(N, dt, sch, 1e-10)(b0, kp1)
        st = jax.device_get(st)
        res[sch] = dict(it_max=int(st['it_max']), nst=int(st['nst']), pass_=bool(int(st['it_max']) <= 1))
    out['psi_cg_iterations_C1'] = res
    out['psi_cg_iterations_C1_pass'] = all(v['pass_'] for v in res.values())
    # CCv vs refcore.make_solver at C = 1
    u0 = VF.fd_ic(case['desc'], L)
    r1 = VF.vref_solver(L)(u0, kp1, dtfd, 'cnr', 1e-11, 1e-9)
    r2 = FC.make_solver(2, L)(u0, nu, dtfd, 'cnr', 1e-11, 1e-9)
    out['ccv_vs_refcore'] = dict(value=rel(r1['out'], r2['out']), bar=1e-10)
    out['ccv_vs_refcore']['pass'] = out['ccv_vs_refcore']['value'] <= 1e-10
    out['pass'] = bool(out['etd_vs_ps']['pass'] and out['psi_cg_iterations_C1_pass'] and out['ccv_vs_refcore']['pass'])
    return out


def i_sym(case, C, N, L, dt):
    kp = jnp.asarray(VC.kappa_params(case, C))
    rng = np.random.default_rng(0)
    D, _ = VS.make_D(N, 2 * N, kp)
    rows = []
    for _ in range(8):
        x, y = (jnp.asarray(rng.standard_normal((N - 1, N - 1))) for _ in range(2))
        a, b = float(jnp.vdot(x, D(y))), float(jnp.vdot(D(x), y))
        rows.append(dict(sym=abs(a - b) / (float(jnp.linalg.norm(x) * jnp.linalg.norm(D(y)))), neg=float(jnp.vdot(x, D(x)))))
    psi = dict(sym_max=max(r['sym'] for r in rows), neg_max=max(r['neg'] for r in rows))
    psi['pass'] = psi['sym_max'] <= 1e-12 and psi['neg_max'] < 0
    KX, KY = VF.faces(kp, L)
    rows = []
    for _ in range(8):
        x, y = (jnp.asarray(rng.standard_normal((L - 1, L - 1))) for _ in range(2))
        Dy, Dx = VF.divk(y, KX, KY, L), VF.divk(x, KX, KY, L)
        rows.append(dict(sym=abs(float(jnp.vdot(x, Dy)) - float(jnp.vdot(Dx, y))) / float(jnp.linalg.norm(x) * jnp.linalg.norm(Dy)),
                         neg=float(jnp.vdot(x, Dx))))
    fd = dict(sym_max=max(r['sym'] for r in rows), neg_max=max(r['neg'] for r in rows))
    fd['pass'] = fd['sym_max'] <= 1e-12 and fd['neg_max'] < 0
    mg = {}
    for alpha in (1.0, 1.5):
        lv, ch = VF.make_hierarchy(kp, L, alpha, dt)
        M = jax.jit(VF.vcycle(lv, ch, alpha, dt))
        rows = []
        for _ in range(8):
            x, y = (jnp.asarray(rng.standard_normal((L - 1, L - 1))) for _ in range(2))
            My, Mx = M(y), M(x)
            rows.append(dict(sym=abs(float(jnp.vdot(x, My)) - float(jnp.vdot(Mx, y))) / float(jnp.linalg.norm(x) * jnp.linalg.norm(My)),
                             pos=float(jnp.vdot(x, Mx))))
        mg[str(alpha)] = dict(sym_max=max(r['sym'] for r in rows), pos_min=min(r['pos'] for r in rows))
    mgp = all(v['sym_max'] <= 1e-10 and v['pos_min'] > 0 for v in mg.values())
    return dict(psi=psi, fd=fd, mg=mg, mg_pass=mgp, pass_=bool(psi['pass'] and fd['pass'] and mgp))


def i_quad(case, C, N, dt):
    kp = jnp.asarray(VC.kappa_params(case, C))
    b0 = VS.ic(case['desc'], N)
    F2 = np.asarray(VS.make_psi(N, dt, 'SBDF2', 1e-12)(b0, kp)[0])
    F3 = np.asarray(VS.make_psi(N, dt, 'SBDF2', 1e-12, Q=3 * N)(b0, kp)[0])
    v = lat_dist(F2, F3)
    return dict(value=v, bar=1e-8, N=N, C=C, pass_=bool(v <= 1e-8))


def i_mms(case, C, N, Ls):
    kp = VC.kappa_params(case, C)
    kpj = jnp.asarray(kp)
    # spectral: D applied to u = sin(pi x) sin(2 pi y) vs the projection of the analytic divergence (fine midpoint rule)
    b = np.zeros((N - 1, N - 1))
    b[0, 1] = 1.0
    D, _ = VS.make_D(N, 2 * N, kpj)
    Db = np.asarray(D(jnp.asarray(b)))
    Qf = 8 * N
    xq = (np.arange(Qf) + 0.5) / Qf
    kap, kx, ky = VC.kappa_grad_np(kp, xq, xq)
    sx, sy = np.sin(np.pi * xq), np.sin(2 * np.pi * xq)
    cx, cy = np.pi * np.cos(np.pi * xq), 2 * np.pi * np.cos(2 * np.pi * xq)
    u = np.outer(sx, sy)
    ux, uy = np.outer(cx, sy), np.outer(sx, cy)
    f = kap * (-5 * np.pi ** 2) * u + kx * ux + ky * uy
    k = np.arange(1, N)
    S = np.sin(np.pi * np.outer(xq, k))
    proj = 4.0 * S.T @ f @ S / Qf ** 2
    spec = dict(value=rel(Db, proj), bar=1e-8, N=N)
    spec['pass'] = spec['value'] <= 1e-8
    # FD: divk on nodal u vs analytic f at the nodes
    errs = []
    for L in Ls:
        xs = np.arange(1, L) / L
        U = np.outer(np.sin(np.pi * xs), np.sin(2 * np.pi * xs))
        KX, KY = VF.faces(kpj, L)
        Du = np.asarray(VF.divk(jnp.asarray(U), KX, KY, L))
        kap, kx, ky = VC.kappa_grad_np(kp, xs, xs)
        fa = kap * (-5 * np.pi ** 2) * U + kx * np.outer(np.pi * np.cos(np.pi * xs), np.sin(2 * np.pi * xs)) \
            + ky * np.outer(np.sin(np.pi * xs), 2 * np.pi * np.cos(2 * np.pi * xs))
        errs.append(float(np.max(np.abs(Du - fa)) / np.max(np.abs(fa))))
    orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    fd = dict(L=list(Ls), err=errs, orders=orders, pass_=bool(all(1.8 <= p <= 2.2 for p in orders)))
    return dict(spectral=spec, fd=fd, pass_=bool(spec['pass'] and fd['pass_']))


def t_loose(case, C, N, L, dt):
    kp = jnp.asarray(VC.kappa_params(case, C))
    b0 = VS.ic(case['desc'], N)
    _, st = VS.make_psi(N, dt, 'SBDF2', 1e-8, maxiter=0)(b0, kp)
    st = jax.device_get(st)
    psi = dict(res_max=float(st['res_max']), nbad=int(st['nbad']), valid=bool(int(st['nbad']) == 0))
    u0 = VF.fd_ic(case['desc'], L)
    _, st = VF.make_fdmg(L, dt, 1e-7, 'MG', maxiter=0)(u0, kp)
    st = jax.device_get(st)
    fd = dict(res_max=float(st['res_max']), nbad=int(st['nbad']), valid=bool(int(st['nbad']) == 0))
    return dict(psi=psi, fdmg=fd, fires=bool((not psi['valid']) and (not fd['valid'])))


def run_all(small=False):
    case = VC.cohort('dev')[0]
    t = time.perf_counter()
    if small:
        N, L, Ls, dt, dtfd = 16, 32, (32, 64, 128), 0.0125, 0.0125
    else:
        N, L, Ls, dt, dtfd = 48, 128, (64, 128, 256), 0.0125, 0.00625
    out = dict(case=case['id'], small=small)
    out['I-C1'] = i_c1(case, N, dt, L, dtfd)
    out['I-sym'] = i_sym(case, 100.0, N, L, dt)
    out['I-quad'] = i_quad(case, 100.0, 64 if not small else 24, dt)
    out['I-mms'] = i_mms(case, 100.0, 64 if not small else 32, Ls)
    out['T-loose'] = t_loose(case, 100.0, N, L, dt)
    out['seconds'] = time.perf_counter() - t
    out['backend'] = jax.default_backend()
    out['all_pass'] = bool(out['I-C1']['pass'] and out['I-sym']['pass_'] and out['I-quad']['pass_'] and out['I-mms']['pass_'])
    out['controls_fire'] = bool(out['T-loose']['fires'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--small', action='store_true')
    a = ap.parse_args()
    r = run_all(a.small)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))
