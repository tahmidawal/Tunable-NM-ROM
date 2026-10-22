"""imp-box sine pseudo-spectral FOM on the unit box: fixed-step ETDRK4 (F-SPEC), dyadic step-doubling ETDRK4 (F-SPEC-AD),
references and training data (DESIGN §3, §4.1-4.2, A1.2-A1.3, A2.2, A2.4, A2.6, A3.5-A3.6).

Kernels: vendor/pscore.py (E0 cross-check solver, byte-identical): `nonlinear` (3/2-padded products, right-hand side
-P[u(u_x+u_y)]), `etdrk4_coefs` (Kassam-Trefethen contour means), `lattice_S`, `composite_gl`. State convention
u = sum b[k-1, l-1] sin(k pi x) sin(l pi y). Every query is `jax.jit` of a function of the descriptor d = (cx, cy, w, a, nu)
alone; the builders see only case-independent arrays (MF-SETUP), so the IC coefficients, every nu-dependent ETDRK4
coefficient, all steps and the output synthesis are inside the timed computation.

FOM stats vector (7): [n_nonfinite, coef_cert, n_steps(accepted), n_cap_hits, defect_max, n_rejects, exhausted].
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))
import pscore as P      # noqa: E402
import geom as G        # noqa: E402
import rom as RM        # noqa: E402  (etd_coefs_direct: closed-form phi functions for the certificate)

AD_J, AD_J0, AD_MAX_ATTEMPTS = 9, 2, 8192


def ic_panels(N):
    """Pilot rule (code audit 1 of the pilot): 64 x 16 up to N = 384, 256 x 32 above."""
    return (64, 16) if N <= 384 else (256, 32)


class ICTable:
    """Separable IC quadrature (case-independent, offline): g_k = 2 int exp(-(x-c)^2/(2w^2)) sin(k pi x) dx."""

    def __init__(self, N, panels=None, pts=None):
        if panels is None:
            panels, pts = ic_panels(N)
        X, W = P.composite_gl(panels, pts)
        self.X = jnp.asarray(X)
        self.SW = jnp.asarray(2.0 * W[:, None] * np.sin(np.pi * np.outer(X, np.arange(1, N))))

    def coefs(self, d):
        cx, cy, w, a = d[0], d[1], d[2], d[3]
        gx = jnp.exp(-(self.X - cx) ** 2 / (2 * w * w)) @ self.SW
        gy = jnp.exp(-(self.X - cy) ** 2 / (2 * w * w)) @ self.SW
        return a * jnp.outer(gx, gy)


def nonlinear_M(b, N, M):
    """pscore.nonlinear (d = 2) with the padded grid size M as a parameter (copy of the pilot's `nonlinear_M`)."""
    bp = jnp.pad(b, [(0, M - N)] * 2)
    n = M - 1
    S = lambda x, ax: P.along(P.sine_synth_last, x, ax, M)
    Cc = lambda x, ax: P.along(P.cos_synth_last, x, ax, M)
    Ty = S(bp, 1)
    u = S(Ty, 0)
    g = Cc(P.kvec(n, 2, 0) * Ty, 0) + S(Cc(P.kvec(n, 2, 1) * bp, 1), 0)
    f = P.analysis(u * g, M)
    return -f[:N - 1, :N - 1]


def nonlinear_dense(N, M=None):
    """The same discretisation as pscore.nonlinear (M-point padded grid, trapezoid sine analysis) with dense transform
    matrices instead of FFTs (A11): u = S b S^T, g = (C b S^T + S b C^T), -A (u g) A^T. Few, fused kernels: the
    latency-optimal implementation at small N on a GPU at batch size 1."""
    M = 3 * N // 2 if M is None else M
    x = np.arange(1, M) / M
    k = np.arange(1, N)
    S = jnp.asarray(np.sin(np.pi * np.outer(x, k)))                       # (M-1, N-1)
    C = jnp.asarray(np.cos(np.pi * np.outer(x, k)) * (np.pi * k)[None, :])
    A = jnp.asarray((2.0 / M) * np.sin(np.pi * np.outer(k, x)))          # (N-1, M-1)

    def f(b):
        u = S @ b @ S.T
        g = C @ b @ S.T + S @ b @ C.T
        return -(A @ (u * g) @ A.T)
    return f


def nl_fun(N, M=None, xform='fft'):
    if xform == 'dense':
        return nonlinear_dense(N, M)
    if M is None or M == 3 * N // 2:
        return lambda b: P.nonlinear(b, N, 2)[0]
    return lambda b: nonlinear_M(b, N, M)


def lam_grid(N):
    k = np.arange(1, N, dtype=np.float64)
    return jnp.asarray(-np.pi ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))


def synth_fn(N, lat=G.LAT):
    """Coefficients -> values on the full (lat+1)^2 scoring lattice (walls exactly zero), x-index major."""
    S = P.lattice_S(N, lat)                                      # (lat-1, N-1) interior rows
    return lambda b: jnp.pad(S @ b @ S.T, 1).reshape(-1)


def _coefs(lam, nu, dt):
    co = P.etdrk4_coefs(nu * lam, dt)
    return tuple(co[k] for k in ('E', 'E2', 'Q', 'f1', 'f2', 'f3'))


def _coef_cert(c, lam, nu, dt):
    d = [x.real for x in RM.etd_coefs_direct(nu * lam, dt)]
    return jnp.max(jnp.stack([jnp.max(jnp.abs(a - b) / (jnp.abs(b) + 1e-300)) for a, b in zip(c, d)])), tuple(d)


def _etd(v, c, NL, nan_stage=None):
    E, E2, Q, f1, f2, f3 = c
    Nv = NL(v)
    a_ = E2 * v + Q * Nv
    Na = NL(a_)
    b_ = E2 * v + Q * Na
    if nan_stage is not None:
        b_ = jnp.where(nan_stage, jnp.nan, b_)
    Nb = NL(b_)
    c_ = E2 * a_ + Q * (2 * Nb - Nv)
    Nc = NL(c_)
    return E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3, (a_, b_, c_)


def _etd_indep(y, dc, NL):
    """Independent transcription (Cox-Matthews form) with the closed-form coefficients."""
    dE, dE2, dQ, df1, df2, df3 = dc
    k1 = NL(y)
    s2 = dE2 * y + dQ * k1
    k2 = NL(s2)
    s3 = dE2 * y + dQ * k2
    k3 = NL(s3)
    s4 = dE2 * s2 + dQ * (2 * k3 - k1)
    k4 = NL(s4)
    return dE * y + df1 * k1 + df2 * (2 * k2 + 2 * k3) + df3 * k4, (s2, s3, s4)


def _defect(x, xi, y):
    return jnp.linalg.norm(x - xi) / (jnp.linalg.norm(xi) + 1e-14 * jnp.linalg.norm(y) + 1e-300)


def _finite_all(xs):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in xs]))


# ------------------------------------------------------------------------------------------- F-SPEC (fixed) ----

def make_fom(N, nsteps, M=None, out='lattice', n_out=G.N_OUT, cert=False, nan_stage=False, perturb_E=0.0,
             ic_table=None, lat=G.LAT, xform='fft'):
    """Fixed-step query d -> (outputs (n_out+1, P) lattice or (n_out+1, N-1, N-1) coefficients, stats (7,))."""
    keep = nsteps // n_out
    assert keep * n_out == nsteps, (nsteps, n_out)
    dt = G.T_END / nsteps
    tab = ICTable(N) if ic_table is None else ic_table
    lam = lam_grid(N)
    NL = nl_fun(N, M, xform)
    syn = synth_fn(N, lat) if out == 'lattice' else (lambda b: b)

    def q(d):
        nu = d[4]
        c = _coefs(lam, nu, dt)
        if cert:
            cc, dc = _coef_cert(c, lam, nu, dt)
        else:
            cc, dc = jnp.asarray(0.0), None
        if perturb_E:
            c = (c[0] * (1.0 + perturb_E),) + c[1:]
        b0 = tab.coefs(d)

        def st(i, s):
            v, bad, dfx = s
            flag = (i == 0) if nan_stage else None
            vn, stg = _etd(v, c, NL, flag)
            bad = bad + (~_finite_all(stg + (vn,))).astype(jnp.float64)
            if cert:
                vi, stgi = _etd_indep(v, dc, NL)
                dd = _defect(vn, vi, v)
                for x_, xi_ in zip(stg, stgi):
                    dd = jnp.maximum(dd, _defect(x_, xi_, v))
                dfx = jnp.maximum(dfx, jnp.where(jnp.isfinite(dd), dd, jnp.inf))
            return vn, bad, dfx

        def block(s, k):
            s = jax.lax.fori_loop(k * keep, (k + 1) * keep, st, s)
            return s, syn(s[0])
        s, outs = jax.lax.scan(block, (b0, jnp.asarray(0.0), jnp.asarray(0.0)), jnp.arange(n_out))
        O = jnp.concatenate([syn(b0)[None], outs])
        stats = jnp.stack([s[1], cc, jnp.asarray(float(nsteps)), jnp.asarray(0.0), s[2], jnp.asarray(0.0),
                           jnp.asarray(0.0)])
        return O, stats
    return jax.jit(q)


# --------------------------------------------------------------------------------- F-SPEC-AD (step doubling) ---

def make_fom_ad(N, tol, J=AD_J, j0=AD_J0, M=None, out='lattice', n_out=G.N_OUT, cert=False, max_attempts=AD_MAX_ATTEMPTS,
                force_reject_first=False, force_inf_all=False, nan_stage=False, lat=G.LAT, xform='fft'):
    """Dyadic step-doubling ETDRK4 (DESIGN A1.3, A2.4): levels dt_j = 0.05 / 2^j, j = 0..J; coefficients j = 0..J+1
    computed in the query; time in integer ticks of dt_{J+1}; e = ||big - half|| / max(||half||, 1e-30) (coefficients);
    accept the half state iff e <= tol; coarsen after e <= tol/32 when aligned; refine on reject; a rejected trial at
    j = J terminates the run (invalid, cap-hit)."""
    assert G.T_END / n_out == 0.05
    tab = ICTable(N)
    lam = lam_grid(N)
    NL = nl_fun(N, M, xform)
    syn = synth_fn(N, lat) if out == 'lattice' else (lambda b: b)
    dts = [0.05 / 2 ** j for j in range(J + 2)]
    L = np.array([2 ** (J + 1 - j) for j in range(J + 2)], dtype=np.int64)      # tick length per level
    T_OUT = int(2 ** (J + 1))
    Lj = jnp.asarray(L)

    tolj = jnp.asarray(float(tol))              # (no executable sharing across tolerances is claimed; code audit 7)

    def q(d):
        nu = d[4]
        tol_ = tolj
        dtj = jnp.asarray(dts)
        Cs = jax.vmap(lambda h: _coefs(lam, nu, h))(dtj)                       # charged: all levels (stacked)
        if cert:
            cc_j, Ds = jax.vmap(lambda c_, h: _coef_cert(c_, lam, nu, h))(Cs, dtj)
            cc = jnp.max(cc_j)
        else:
            cc = jnp.asarray(0.0)
        b0 = tab.coefs(d)
        o0 = syn(b0)
        outs0 = jnp.zeros((n_out,) + o0.shape, o0.dtype)

        def lev(Cx, j):
            return tuple(x[j] for x in Cx)

        def trial(v, j, first):
            flag = first if nan_stage else None
            big, s1 = _etd(v, lev(Cs, j), NL, flag)
            h1, s2 = _etd(v, lev(Cs, j + 1), NL)
            half, s3 = _etd(h1, lev(Cs, j + 1), NL)
            fin = _finite_all(s1 + s2 + s3 + (big, h1, half))
            dfx = jnp.asarray(0.0)
            if cert:
                for (x, stg, y, jj) in ((big, s1, v, j), (h1, s2, v, j + 1), (half, s3, h1, j + 1)):
                    xi, stgi = _etd_indep(y, lev(Ds, jj), NL)
                    dd = _defect(x, xi, y)
                    for a_, b_ in zip(stg, stgi):
                        dd = jnp.maximum(dd, _defect(a_, b_, y))
                    dfx = jnp.maximum(dfx, jnp.where(jnp.isfinite(dd), dd, jnp.inf))
            e = jnp.linalg.norm(big - half) / jnp.maximum(jnp.linalg.norm(half), 1e-30)
            return half, e, fin, dfx

        def cond(s):
            v, tick, j, k, outs, att, nacc, nrej, bad, fail, dfx, mis = s
            return (k < n_out) & (att < max_attempts) & (~fail)

        def body(s):
            v, tick, j, k, outs, att, nacc, nrej, bad, fail, dfx, mis = s
            half, e, fin, dd = trial(v, j, att == 0)
            if force_reject_first:
                e = jnp.where(att == 0, 10.0 * tol_, e)
            if force_inf_all:                                  # MF-AD-cap (A4): every trial rejected
                e = jnp.asarray(jnp.inf)
            e = jnp.where(jnp.isfinite(e) & fin, e, jnp.inf)
            acc = e <= tol_
            capfail = (~acc) & (j == J)
            tick_n = tick + jnp.where(acc, Lj[j], 0)
            Lprev = Lj[jnp.maximum(j - 1, 0)]
            coarse = acc & (e <= tol_ / 32) & (j > 0) & (tick_n % Lprev == 0)
            jn = jnp.where(acc, jnp.where(coarse, j - 1, j), jnp.minimum(j + 1, J))
            vn = jnp.where(acc, half, v)
            hit = acc & (tick_n == (k + 1) * T_OUT)
            mis = mis + (acc & (tick_n > (k + 1) * T_OUT)).astype(jnp.float64)     # alignment assertion (must stay 0)
            outs = jax.lax.cond(hit, lambda o: o.at[k].set(syn(vn)), lambda o: o, outs)
            return (vn, tick_n, jn, k + hit.astype(jnp.int32), outs, att + 1, nacc + acc.astype(jnp.float64),
                    nrej + (~acc).astype(jnp.float64), bad + (~fin).astype(jnp.float64), fail | capfail,
                    jnp.maximum(dfx, dd), mis)

        s0 = (b0, jnp.int64(0), jnp.int32(j0), jnp.int32(0), outs0, jnp.int32(0), jnp.asarray(0.0), jnp.asarray(0.0),
              jnp.asarray(0.0), jnp.asarray(False), jnp.asarray(0.0), jnp.asarray(0.0))
        s = jax.lax.while_loop(cond, body, s0)
        v, tick, j, k, outs, att, nacc, nrej, bad, fail, dfx, mis = s
        exhausted = (k < n_out) & ~fail
        O = jnp.concatenate([o0[None], outs])
        O = jnp.where(k == n_out, O, jnp.nan)                                     # unfinished run -> non-finite
        stats = jnp.stack([bad + mis, cc, nacc, fail.astype(jnp.float64), dfx, nrej, exhausted.astype(jnp.float64)])
        return O, stats
    return jax.jit(q)


def fom_valid(fam, st):
    """Validity from FOM stats (A1.5, A2.6): finite stages; AD: no cap-hit, no attempt exhaustion."""
    st = np.asarray(st)
    ok = np.all(st[:, 0] == 0)
    if fam == 'specad':
        ok = ok and np.all(st[:, 3] == 0) and np.all(st[:, 6] == 0)
    return bool(ok)


def ref_cfl_steps(N, factor=2.0):
    """Pilot: dt ~ 0.6 / (u_max N), u_max = 2; n = 5 * 2^k >= factor * T / dt."""
    need = G.T_END * 2.0 * N / 0.6 * factor
    n = 5
    while n < need:
        n *= 2
    return n
