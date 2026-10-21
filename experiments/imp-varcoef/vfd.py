"""imp-varcoef finite-difference competitors and reference solver (DESIGN §4, §6, A1.5, A1.9, A1.12, A1.14).

Grid: mesh L (h = 1/L), interior nodes (L-1)^2 (x slowest, flat as refcore), zero Dirichlet walls.
Diffusion (conservative 5-point flux form): Dk u = [KX_{i+1/2,j}(u_{i+1,j}-u_ij) - KX_{i-1/2,j}(u_ij-u_{i-1,j})]/h^2 + (y),
face coefficients = kappa at face midpoints. Symmetric negative definite.
Advection: refcore.adv_central (u * central differences).

* FDMG: SBDF2 IMEX (first step SBDF1), A = alpha I - dt Dk (SPD), PCG preconditioned by a fixed linear V(2,2) cycle
  (damped Jacobi omega 0.8, full weighting R, P = 4 R^T, coarse operators re-discretised from kappa at coarse faces,
  dense Cholesky on L = 8) or by the constant-coefficient DST-I inverse (kbar = nu).
* CCv: E0 refcore.make_solver with lap replaced by Dk and the DST preconditioner on kbar = nu (vref_solver).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
FOM2D = HERE.parent / 'jcp-fom2d'
if str(FOM2D) not in sys.path:
    sys.path.insert(0, str(FOM2D))
import refcore as FC          # noqa: E402  (E0 refcore, unchanged)
import fomrun2 as FR          # noqa: E402  (sine_interp, fd_input)
import vcoef as VC            # noqa: E402
from vspec import pcg         # noqa: E402

UMAX_GUARD = 10.0
COARSE_L = 8
# jcp-fom2d levels t0..t3 (+X); '4' = one step looser than t3 (cheap-edge extension); 'E' = E0 tight (1e-13, 1e-12, 40 Newton),
# 'F' = tighter anchor (1e-13, 1e-13, 40) for the A1.12 qualification of CCv
CCV_TOLS = dict(FR.TOLS, **{'4': (1e-3, 1e-4), 'E': (1e-13, 1e-12), 'F': (1e-13, 1e-13)})
CCV_NEWTON = {'E': 40, 'F': 40}


# ---------------------------------------------------------------------------------------------- operators -----

def faces(kp, L):
    """kappa at x-faces (L, L-1) and y-faces (L-1, L) of mesh L (traced kp)."""
    xf = (jnp.arange(L, dtype=jnp.float64) + 0.5) / L
    xs = jnp.arange(1, L, dtype=jnp.float64) / L
    return VC.kappa_tensor(kp, xf, xs), VC.kappa_tensor(kp, xs, xf)


def divk(u, KX, KY, L):
    """u (L-1, L-1) grid -> Dk u (grid)."""
    up = jnp.pad(u, 1)
    fx = KX * (up[1:, 1:-1] - up[:-1, 1:-1])
    fy = KY * (up[1:-1, 1:] - up[1:-1, :-1])
    return L * L * ((fx[1:, :] - fx[:-1, :]) + (fy[:, 1:] - fy[:, :-1]))


def diag_divk(KX, KY, L):
    return -L * L * (KX[1:, :] + KX[:-1, :] + KY[:, 1:] + KY[:, :-1])


def restrict(r):
    """Full weighting (L-1)^2 -> (L/2-1)^2, coarse node i <-> fine node 2i (1-based)."""
    w = jnp.array([0.25, 0.5, 0.25])
    rp = jnp.pad(r, 1)
    n = (r.shape[0] + 1) // 2 - 1
    # fine indices (0-based interior) of coarse node c: 2c+1; padded index 2c+2
    idx = 2 * jnp.arange(n) + 2
    sx = sum(w[a] * rp[idx + a - 1, :] for a in range(3))          # (n, L+1)
    return sum(w[a] * sx[:, idx + a - 1] for a in range(3))


def prolong(e, nf):
    """Bilinear (L/2-1)^2 -> (L-1)^2; equals 4 R^T for the full weighting above."""
    nc = e.shape[0]
    ep = jnp.pad(e, 1)                                              # (nc+2)
    out = jnp.zeros((nf + 2, nf + 2), e.dtype)
    # coarse padded index c (0..nc+1) sits at fine padded index 2c
    out = out.at[::2, ::2].set(ep)
    out = out.at[1::2, ::2].set(0.5 * (ep[:-1, :] + ep[1:, :]))
    out = out.at[::2, 1::2].set(0.5 * (ep[:, :-1] + ep[:, 1:]))
    out = out.at[1::2, 1::2].set(0.25 * (ep[:-1, :-1] + ep[1:, :-1] + ep[:-1, 1:] + ep[1:, 1:]))
    return out[1:-1, 1:-1]


def level_sizes(L):
    out = []
    Ll = L
    while Ll >= COARSE_L:
        out.append(Ll)
        if Ll == COARSE_L:
            break
        Ll //= 2
    return out


def make_hierarchy(kp, L, alpha, cdt):
    """Levels L, L/2, ..., COARSE_L as arrays [(KX, KY, diag)] and the coarse Cholesky factor. A_l = alpha I - cdt Dk_l."""
    levels = []
    for Ll in level_sizes(L):
        KX, KY = faces(kp, Ll)
        dg = alpha - cdt * diag_divk(KX, KY, Ll)
        levels.append((KX, KY, dg))
    Lc = level_sizes(L)[-1]
    KXc, KYc, _ = levels[-1]
    n = (Lc - 1) ** 2
    Ac = jax.vmap(lambda e: (alpha * e.reshape(Lc - 1, Lc - 1) - cdt * divk(e.reshape(Lc - 1, Lc - 1), KXc, KYc, Lc)
                             ).reshape(-1))(jnp.eye(n))
    Ac = 0.5 * (Ac + Ac.T)
    chol = jnp.linalg.cholesky(Ac)
    return levels, chol


def vcycle(levels, chol, alpha, cdt, omega=0.8, nu1=2, nu2=2):
    sizes = [lv[0].shape[0] for lv in levels]          # KX has shape (L_l, L_l - 1): static

    def A(l, u):
        KX, KY, _ = levels[l]
        return alpha * u - cdt * divk(u, KX, KY, sizes[l])

    def cyc(l, r):
        KX, KY, dg = levels[l]
        if l == len(levels) - 1:
            y = jax.scipy.linalg.cho_solve((chol, True), r.reshape(-1))
            return y.reshape(r.shape)
        x = jnp.zeros_like(r)
        for _ in range(nu1):
            x = x + omega * (r - A(l, x)) / dg
        rc = restrict(r - A(l, x))
        x = x + prolong(cyc(l + 1, rc), r.shape[0])
        for _ in range(nu2):
            x = x + omega * (r - A(l, x)) / dg
        return x
    return lambda r: cyc(0, r)


def dst_pinv(L, alpha, cdt, nu):
    k1 = jnp.arange(1, L, dtype=jnp.float64)
    lam1 = 4 * L ** 2 * jnp.sin(jnp.pi * k1 / (2 * L)) ** 2
    den = alpha + cdt * nu * (lam1[:, None] + lam1[None, :])
    return lambda r: FC.dstn(FC.dstn(r) / den)


def fd_out(out, L):
    T = 256 if out == 'lattice' else int(out[1])

    def f(U):                                  # U (L-1, L-1) -> (T+1, T+1)
        if L >= T:
            s = L // T
            return jnp.pad(U, 1)[::s, ::s]
        return jnp.pad(FR.sine_interp(U, L, T), 1)
    return f


# ---------------------------------------------------------------------------------------------- FDMG ----------

def make_fdmg(L, dt, tol, prec, out='lattice', maxiter=200):
    """FDMG configuration (x0 = flat interior nodal IC; tol traced, A7.1)."""
    from vspec import cached
    base = cached(('fdmg', L, float(dt), prec, str(out), int(maxiter)), lambda: _make_fdmg(L, dt, prec, out, maxiter))
    return base.bind(tol, f'fdmg|L{L}|{dt:g}|{prec}|{tol:g}')


def _make_fdmg(L, dt, prec, out='lattice', maxiter=200):
    from vspec import Program
    assert prec in ('MG', 'DST'), prec
    keep = int(round(0.05 / dt))
    assert abs(keep * dt - 0.05) < 1e-12, dt
    synth = fd_out(out, L)
    k1 = jnp.arange(1, L, dtype=jnp.float64)
    lam1 = 4 * L ** 2 * jnp.sin(jnp.pi * k1 / (2 * L)) ** 2
    lam = lam1[:, None] + lam1[None, :]

    def setup(kp, tol):
        KX, KY = faces(kp, L)
        S_ = dict(KX=KX, KY=KY)
        for tag, alpha in (('1', 1.0), ('2', 1.5)):
            if prec == 'MG':
                lv, ch = make_hierarchy(kp, L, alpha, dt)
                S_['mg' + tag] = (lv, ch)
            else:
                S_['den' + tag] = alpha + dt * kp[0] * lam
        return S_

    def pinv(S_, tag, alpha):
        if prec == 'MG':
            lv, ch = S_['mg' + tag]
            return vcycle(lv, ch, alpha, dt)
        den = S_['den' + tag]
        return lambda r: FC.dstn(FC.dstn(r) / den)

    def core(u0, S_, tol):
        U0 = u0.reshape(L - 1, L - 1)
        KX, KY = S_['KX'], S_['KY']
        P1, P2 = pinv(S_, '1', 1.0), pinv(S_, '2', 1.5)

        def step(k, c):
            u, up, Ap, um, st = c
            Au = FC.adv_central(u, L)
            first = k == 0

            def solve(alpha, rhs, x0, P):
                return pcg(lambda x: alpha * x - dt * divk(x, KX, KY, L), rhs, x0, P, tol, maxiter)
            rhs1 = u - dt * Au
            rhs2 = 2 * u - 0.5 * up - dt * (2 * Au - Ap)
            x, it, tr = jax.lax.cond(first, lambda: solve(1.0, rhs1, u, P1),
                                     lambda: solve(1.5, rhs2, 2 * u - up, P2))
            st = dict(it_sum=st['it_sum'] + it, it_max=jnp.maximum(st['it_max'], it),
                      res_max=jnp.maximum(st['res_max'], tr),
                      it_stage=st['it_stage'].at[k].set(it), res_stage=st['res_stage'].at[k].set(tr),
                      nbad=st['nbad'] + jnp.where((tr <= 10 * tol) & jnp.isfinite(tr), 0, 1), nst=st['nst'] + 1)
            return x, u, Au, jnp.maximum(um, jnp.max(jnp.abs(x))), st

        def interval(c, i):
            c = jax.lax.fori_loop(i * keep, (i + 1) * keep, step, c)
            return c, synth(c[0])
        st0 = dict(it_sum=jnp.int32(0), it_max=jnp.int32(0), res_max=jnp.float64(0.0), nbad=jnp.int32(0),
                   nst=jnp.int32(0), it_stage=jnp.full(5 * keep, -1, jnp.int32), res_stage=jnp.full(5 * keep, jnp.nan))
        c0 = (U0, U0, jnp.zeros_like(U0), jnp.max(jnp.abs(U0)), st0)
        c, outs = jax.lax.scan(interval, c0, jnp.arange(5))
        fields = jnp.concatenate((synth(U0)[None], outs))
        return fields, dict(c[4], umax=c[3], nst_expected=jnp.int32(5 * keep))
    return Program(setup, core, f'fdmg|L{L}|{dt:g}|{prec}')


# ---------------------------------------------------------------------------------------------- CCv / ref -----

def vref_solver(L, max_newton=20, lin_maxiter=None):
    """refcore.make_solver (d = 2, central advection, CN-R) with nu * lap replaced by Dk (face kappa from kp) and the
    DST preconditioner on kbar = nu. run(u0, kp, dt, scheme, ntol, ltol) -> dict(out (6, (L-1)^2), res, nwt, umax, stages,
    units), the refcore record."""
    d = 2
    shape = (L - 1,) * d
    lin_maxiter = FC.LIN_MAXITER[d] if lin_maxiter is None else lin_maxiter

    def query(u0, kp, sched, lam):
        KX, KY = faces(kp, L)
        return solve(u0, KX, KY, kp[0], sched, lam)

    def solve(u0, KX, KY, nu, sched, lam, den_pre=None):

        def F(u):
            g = u.reshape(shape)
            return (FC.adv_central(g, L) - divk(g, KX, KY, L)).reshape(-1)
        dt, nstart, theta_m = sched['dt'], sched['nstart'], sched['theta']
        total_units = 2 * sched['steps']

        def stage(k, units, un):
            start = k < nstart
            tau = jnp.where(start, dt / 2, dt)
            th = jnp.where(start, 1.0, theta_m)
            Fn = F(un)
            const = (1 - th) * tau * Fn
            res = lambda u: u - un + th * tau * F(u) + const
            den = 1 + th * tau * nu * lam if den_pre is None else den_pre   # th * tau = dt/2 at every CN-R stage

            def pre(v):
                return FC.dstn(FC.dstn(v.reshape(shape)) / den).reshape(-1)
            scale = jnp.maximum(jnp.linalg.norm(un), 1e-300)

            def body(s):
                u, it, _ = s
                r = res(u)
                jv = lambda v: jax.jvp(res, (u,), (v,))[1]
                du, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=sched['ltol'], maxiter=lin_maxiter, M=pre)
                u = u + du
                return u, it + 1, jnp.linalg.norm(res(u))
            u, it, rn = jax.lax.while_loop(
                lambda s: (s[2] > sched['ntol'] * scale) & (s[1] < max_newton) & jnp.isfinite(s[2]), body,
                (un, jnp.int32(0), jnp.linalg.norm(res(un))))
            return u, it, rn / scale, jnp.where(start, 1, 2)

        def body(c):
            k, units, u, out, res_, nwt, umax = c
            u, it, rel, du = stage(k, units, u)
            units = units + du
            ks = jnp.minimum(k, FC.NST - 1)
            res_ = res_.at[ks].set(rel)
            nwt = nwt.at[ks].set(it)
            umax = jnp.maximum(umax, jnp.max(jnp.abs(u)))
            store = (units % sched['keep_units']) == 0
            idx = jnp.clip(units // sched['keep_units'], 0, 5)
            out = jax.lax.cond(store, lambda o: o.at[idx].set(u), lambda o: o, out)
            return k + 1, units, u, out, res_, nwt, umax

        out0 = jnp.zeros((6,) + u0.shape, jnp.float64).at[0].set(u0)
        c0 = (jnp.int32(0), jnp.int32(0), u0, out0, jnp.full(FC.NST, jnp.nan), jnp.full(FC.NST, -1, jnp.int32),
              jnp.max(jnp.abs(u0)))
        k, units, _, out, res_, nwt, umax = jax.lax.while_loop(lambda c: c[1] < total_units, body, c0)
        return dict(out=out, res=res_, nwt=nwt, umax=umax, stages=k, units=units)

    k1 = jnp.arange(1, L, dtype=jnp.float64)
    lam1 = 4 * L ** 2 * jnp.sin(jnp.pi * k1 / (2 * L)) ** 2
    lam = lam1[:, None] + lam1[None, :]
    jq = jax.jit(query)
    cache = {}

    def run(u0, kp, dt, scheme, ntol, ltol):
        sched = FC.schedule(dt, scheme, ntol, ltol)
        args = (jnp.asarray(u0, jnp.float64).reshape(-1), jnp.asarray(kp, jnp.float64), sched, lam)
        run.compile_seconds = 0.0
        if 'exe' not in cache:
            t = time.perf_counter()
            cache['exe'] = jq.lower(*args).compile()
            run.compile_seconds = time.perf_counter() - t
        return cache['exe'](*args)
    run.compile_seconds = 0.0
    run.solve = solve
    run.lam = lam
    return run


def make_ccv(L, dt, level, out='lattice'):
    """CCv FOM as a Program (A3.14): setup(kp) = face coefficients and nu; core(u0, S) = the E0 CN-R solve with variable
    kappa (Newton + DST-preconditioned BiCGStab; the DST denominator 1 + dt/2 nu Lambda_h, the same at every CN-R stage, is
    prepared in setup, A6.2) followed by the output map; query = core(u0, setup(kp)) as ONE jitted program. Tolerance level
    from CCV_TOLS."""
    from vspec import cached
    ntol, ltol = CCV_TOLS[level]
    lim = CCV_NEWTON.get(level, FR.MAX_NEWTON)
    base = cached(('ccv', L, float(dt), str(out), lim), lambda: _make_ccv(L, dt, out, lim))
    return base.bind((ntol, ltol), f'ccv|L{L}|{dt:g}|t{level}')


def _make_ccv(L, dt, out, lim):
    from vspec import Program
    run = vref_solver(L, max_newton=lim)
    fo = fd_out(out, L)
    sched0 = FC.schedule(dt, 'cnr', 1e-6, 1e-7)                # tolerances replaced by the traced p below
    lam = run.lam
    nexp = int(FC.n_stages(dt, 'cnr'))

    def setup(kp, p):
        KX, KY = faces(kp, L)
        # every CN-R stage has theta * tau = dt / 2 (four BE half steps: 1 * dt/2; CN: 1/2 * dt), so one denominator
        return dict(KX=KX, KY=KY, nu=kp[0], den=1 + 0.5 * dt * kp[0] * lam)

    def core(u0, S_, p):
        sched = dict(sched0, ntol=p[0], ltol=p[1])
        r = run.solve(jnp.asarray(u0).reshape(-1), S_['KX'], S_['KY'], S_['nu'], sched, lam, S_['den'])
        F = jax.vmap(lambda f: fo(f.reshape(L - 1, L - 1)))(r['out'])
        return F, dict(res=r['res'], nwt=r['nwt'], umax=r['umax'], stages=r['stages'], nexp=jnp.int32(nexp),
                       ntol=p[0])
    return Program(setup, core, f'ccv|L{L}|{dt:g}')


def fd_ic(desc, L):
    return FR.fd_input(desc, L)
