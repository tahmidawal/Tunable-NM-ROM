"""jcp-time2 core: a two-step linear-multistep (LMM) reduced step for the off-mesh coordinate-bank Burgers 2D ROM.

DESIGN.md (sections 2, 5 and amendment A1) is the specification. Only the time discretisation of the vendored
2D lane query (vendor/quad2d/qcore.make_linear_query, backward Euler) changes:

    R_n(c) = A (a0 c + a1 c_n + a2 c_{n-1}) + dt (b0 F(c) + b1 F(c_n)),   F(c) = N(c) + nu Lambda A c

  LSPG  minimise || D R_n(c) ||,  D = 1 / (a0 + dt b0 nu Lambda)       (BE: algebraically the deployed residual; G1a tolerances)
  GAL   solve Q^T R_n(c) = 0,     A = Q Rm (thin QR)                   (the LMM for Rm c' = -Q^T F(c))

The stepper is generic in the tested advection (nl, nlJ), so the manufactured tests (test_lmm.py) run the same code.
dt, step count, output stride, scheme coefficients, startup length and tolerances are traced (one compile per
(rule, form)). Outputs: the six output coefficient vectors (t = 0, 0.05, ..., 0.25) in a fixed (6, R') buffer,
per-trajectory statistics accumulated in the loop carry. f64 throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VQ = HERE / 'vendor' / 'quad2d'
for _p in (ROOT / 'experiments' / 'separable-decoder', ROOT / 'experiments' / 'mr-burgers2d', VQ / 'vendor', VQ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

CKPT = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'

# scheme: (a0, a1, a2, b0, b1, number of BE startup steps)
SCHEMES = dict(BE=(1., -1., 0., 1., 0., 0), CN=(1., -1., 0., .5, .5, 0), CNR=(1., -1., 0., .5, .5, 2),
               BDF2=(1.5, -2., .5, 1., 0., 1), TH06=(1., -1., 0., .6, .4, 0))
BE_CO = (1., -1., 0., 1., 0.)
REASONS = {0: 'budget', 1: 'tol', 2: 'tiny_step', 3: 'damping_limit', 4: 'stationary'}
ALT_TAU = 1e-5
MUTATIONS = (None, 'a2flip', 'fn_lag', 'stale_hist', 'late_out', 'd_b1')


def sched(scheme, dt, gtol, tolf, horizon=.25, spacing=.05):
    """Traced schedule of one rollout. LSPG: tol = tolf * scale (deployed 1e-9); GAL: tol = tolf * ||A c_n||."""
    a0, a1, a2, b0, b1, ns = SCHEMES[scheme]
    steps, keep = int(round(horizon / dt)), int(round(spacing / dt))
    assert abs(steps * dt - horizon) < 1e-12 and abs(keep * dt - spacing) < 1e-12, (scheme, dt)
    return dict(dt=jnp.asarray(dt, jnp.float64), steps=jnp.asarray(steps, jnp.int32), keep=jnp.asarray(keep, jnp.int32),
                co=jnp.asarray([a0, a1, a2, b0, b1], jnp.float64), nstart=jnp.asarray(ns, jnp.int32),
                gtol=jnp.asarray(gtol, jnp.float64), tolf=jnp.asarray(tolf, jnp.float64))


def make_lm(evalJ, budget, trust, ridge, solver='chol', clip=True):
    """hfast.make_fused_lm (K = R', q = 0) with gtol and tol passed per call (traced). Returns also the final
    stationarity ratio and residual norm of the RETURNED state (used for the per-step verification of A1.5)."""
    def solve(Hm, g):
        if solver == 'chol':
            c = jax.scipy.linalg.cho_factor(Hm, lower=True)
            return jax.scipy.linalg.cho_solve(c, -g)
        return jnp.linalg.solve(Hm, -g)

    def ratio(g, J, rn):
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * rn + 1e-300)

    def lm(w0, args, tol, gtol, lam0):
        r, J = evalJ(w0, args)
        rn = jnp.linalg.norm(r)
        g = J.T @ r
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(ratio(g, J, rn) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, g, rn, lam, it, reason, rej = s
            Hm = J.T @ J
            d = jnp.diag(Hm) + 1e-30
            step = solve(Hm + jnp.diag(lam * d), g)
            if clip:
                nz = jnp.linalg.norm(step)
                step = step * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
                ok = jnp.all(jnp.isfinite(step))
            else:
                ok = jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step) <= trust)
            wn = w + jnp.where(ok, step, 0.)
            r2, J2 = evalJ(wn, args)
            rn2 = jnp.linalg.norm(r2)
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            w3 = jnp.where(accept, wn, w)
            r3 = jnp.where(accept, r2, r)
            J3 = jnp.where(accept, J2, J)
            rn3 = jnp.where(accept, rn2, rn)
            g3 = J3.T @ r3
            gn = ratio(g3, J3, rn3)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn3 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            return (w3, r3, J3, g3, rn3, lam2, it + 1, reason, rej + (~accept).astype(jnp.int32))

        w, r, J, g, rn, lam, it, reason, rej = jax.lax.while_loop(
            lambda s: (s[6] < budget) & (s[7] == 0), body,
            (w0, r, J, g, rn, jnp.asarray(lam0, dtype=jnp.float64), jnp.int32(0), reason, jnp.int32(0)))
        return w, rn, it, reason, ratio(g, J, rn), rej, lam
    return lm


def make_evolve(nl, nlJ, form, Rp, trust, budget=600, ridge=1e-10, solver='chol', clip=True, mutation=None):
    """evolve(w0, nu, scale, data, sch) -> (W (6, R'), stats). data: A (M, R'), lam (M,), hmask (M,), and for GAL
    Qm (M, R'), Rm (R', R'), QtLA = Qm^T (lam A); plus the rule blocks nl/nlJ read. nl(c, data) -> (M,);
    nlJ(c, data, None) -> (N(c), dN/dc (M, R'))."""
    assert form in ('LSPG', 'GAL') and mutation in MUTATIONS

    def inner(w, Aw, val, hist, nu, data, co, dt):
        a0, a1, a2, b0, b1 = co[0], co[1], co[2], co[3], co[4]
        Acn, Acm, Fn = hist
        lam = data['lam']
        return a0 * Aw + a1 * Acn + a2 * Acm + dt * (b0 * (val + nu * lam * Aw) + b1 * Fn)

    def dscale(co, dt, nu, lam):
        b = co[4] if mutation == 'd_b1' else co[3]
        return co[0] + dt * b * nu * lam

    def res(w, args):
        hist, nu, data, co, dt = args
        Aw = data['A'] @ w
        R = inner(w, Aw, nl(w, data), hist, nu, data, co, dt)
        if form == 'LSPG':
            return R / dscale(co, dt, nu, data['lam'])
        return data['Qm'].T @ R

    def evalJ(w, args):
        hist, nu, data, co, dt = args
        Aw = data['A'] @ w
        val, dN = nlJ(w, data, None)
        R = inner(w, Aw, val, hist, nu, data, co, dt)
        if form == 'LSPG':
            S = 1. / dscale(co, dt, nu, data['lam'])
            if mutation == 'd_b1':  # unsimplified derivative for the wrong-weight mutant (code audit 1, item 8)
                return R * S, S[:, None] * (co[0] * data['A'] + dt * co[3] * (dN + (nu * data['lam'])[:, None] * data['A']))
            # S (a0 A + dt b0 (dN + nu lam A)) = A + dt b0 S dN, since S (a0 + dt b0 nu lam) = 1 (the vendor BE form)
            return R * S, data['A'] + (dt * co[3] * S)[:, None] * dN
        Qt = data['Qm'].T
        return Qt @ R, co[0] * data['Rm'] + dt * co[3] * (Qt @ dN + nu * data['QtLA'])

    lm = make_lm(evalJ, budget, trust, ridge, solver, clip)
    be = jnp.asarray(BE_CO, jnp.float64)

    def evolve(w0, nu, scale, data, sch):
        A, lam, hm = data['A'], data['lam'], data['hmask']
        dt = sch['dt']
        M = A.shape[0]
        Fval = lambda w: nl(w, data) + nu * lam * (A @ w)
        z5 = jnp.zeros(5, jnp.int32)
        out0 = jnp.zeros((6, Rp), jnp.float64).at[0].set(w0)
        stats0 = dict(it_sum=jnp.int32(0), it_max=jnp.int32(0), rej=jnp.int32(0), exits=z5, nfail=jnp.int32(0),
                      first_fail=jnp.int32(-1), worst_ratio=jnp.float64(0.), worst_tolratio=jnp.float64(0.),
                      alt_num=jnp.float64(0.), alt_den=jnp.float64(0.), amp=jnp.float64(0.))
        nA0 = jnp.linalg.norm(A @ w0) + 1e-300
        thr = ALT_TAU * jnp.max(jnp.abs(A @ w0))

        def body(c):
            k, wv, wprev, wprev2, lam0, dprev, runs, out, st = c
            co = jnp.where(k < sch['nstart'], be, sch['co'])
            if mutation == 'a2flip':
                co = co.at[2].multiply(-1.)
            Acn = A @ wv
            Acm = A @ (wprev2 if mutation == 'stale_hist' else wprev)
            fsrc = wprev if mutation == 'fn_lag' else wv
            Fn = jax.lax.cond(co[4] != 0., lambda: Fval(fsrc), lambda: jnp.zeros(M, jnp.float64))
            hist = (Acn, Acm, Fn)
            args = (hist, nu, data, co, dt)
            we = wv + (wv - wprev)
            wqd = jnp.where(k >= 2, 3. * wv - 3. * wprev + wprev2, we)
            cand = jnp.stack((wv, we, wqd))
            rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, args)))(cand)
            rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
            wi = cand[jnp.argmin(rs)]
            tol = sch['tolf'] * (scale if form == 'LSPG' else jnp.linalg.norm(Acn))
            w2, rn, it, reason, gn, rej, lamn = lm(wi, args, tol, sch['gtol'], lam0)
            # A1.5 verification from the returned state: stationarity ratio <= gtol or residual <= tol
            if form == 'LSPG':      # LSPG: stationarity or residual tolerance
                conv = (gn <= sch['gtol']) | (rn <= tol)
            else:                   # GAL: a root, i.e. residual tolerance only (code audit 1, item 1)
                conv = rn <= tol
            ok = jnp.isfinite(rn) & conv & (reason != 0) & (reason != 3)
            # A1.6 alternation index on the stiff half of the tests
            Aw2 = A @ w2
            dcur = hm * (Aw2 - Acn)
            pr = dcur * dprev
            use = k >= 1
            # A2.14: amplitude-qualified pairs; an alternation event counts once its mode has alternated >= 3 steps running
            qual = use & (jnp.minimum(jnp.abs(dcur), jnp.abs(dprev)) > thr)
            event = qual & (pr < 0)
            runs = jnp.where(event, runs + 1, 0)
            st = dict(it_sum=st['it_sum'] + it, it_max=jnp.maximum(st['it_max'], it), rej=st['rej'] + rej,
                      exits=st['exits'].at[reason].add(1), nfail=st['nfail'] + (~ok).astype(jnp.int32),
                      first_fail=jnp.where((st['first_fail'] < 0) & (~ok), k, st['first_fail']),
                      worst_ratio=jnp.maximum(st['worst_ratio'], gn),
                      worst_tolratio=jnp.maximum(st['worst_tolratio'], rn / (tol + 1e-300)),
                      alt_num=st['alt_num'] + jnp.sum(jnp.where(event & (runs >= 3), jnp.abs(pr), 0.)),
                      alt_den=st['alt_den'] + jnp.sum(jnp.where(qual, jnp.abs(pr), 0.)),
                      amp=jnp.maximum(st['amp'], jnp.linalg.norm(dcur) / nA0))
            kk = k + 1
            store = (kk % sch['keep']) == 0
            idx = jnp.clip(kk // sch['keep'], 0, 5)
            val = wv if mutation == 'late_out' else w2
            out = jnp.where(store, out.at[idx].set(val), out)
            return (kk, w2, wv, wprev, lamn, dcur, runs, out, st)

        c0 = (jnp.int32(0), w0, w0, w0, jnp.asarray(1e-6, jnp.float64), jnp.zeros(M, jnp.float64),
              jnp.zeros(M, jnp.int32), out0, stats0)
        k, *_rest, out, st = jax.lax.while_loop(lambda c: c[0] < sch['steps'], body, c0)
        st = dict(st, steps=k, alt_index=jnp.where(st['alt_den'] > 0, st['alt_num'] / (st['alt_den'] + 1e-300), 0.))
        return out, st

    return evolve


# ------------------------------------------------------------------ the Burgers 2D ROM (vendored qcore) ----

def load_qcore():
    import qcore as Q
    Q.CKPT = CKPT
    return Q


def form_data(base, blocks, form):
    """base: A, lam, G (qstudy.Mesh.base); blocks: rule blocks. Adds hmask, and for GAL the thin QR of A."""
    d = dict(base, **blocks)
    lam = d['lam']
    d['hmask'] = (lam > jnp.median(lam)).astype(jnp.float64)
    if form == 'GAL':
        Qm, Rm = jnp.linalg.qr(d['A'], mode='reduced')
        d.update(Qm=Qm, Rm=Rm, QtLA=Qm.T @ (lam[:, None] * d['A']))
    return jax.block_until_ready(d)


def a_conditioning(A):
    s = np.linalg.svd(np.asarray(A), compute_uv=False)
    return dict(rank=int(np.sum(s > s[0] * 1e-13)), cond=float(s[0] / s[-1]), smax=float(s[0]), smin=float(s[-1]))


def make_query(Q, kind, form, Rp, L, trust, mutation=None, budget=600):
    """End-to-end query (initial fit + LMM stepping + decode of the six output fields), as qcore.make_linear_query."""
    import engines as e
    import hops as H
    nl, nlJ = Q.tested_value(kind, L), Q.tested_jac(kind, L)
    evolve = make_evolve(nl, nlJ, form, Rp, trust, budget=budget, mutation=mutation)

    def initialize(u0, cold):
        xy, wq, Qc, Rc, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        w0 = jax.scipy.linalg.solve_triangular(Rc, Qc.T @ ui, lower=False)
        return w0, jnp.linalg.norm(ui) * jnp.sqrt(len(wq))

    def decode(W, data):
        U = H.bank_apply(data['G'], W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold, sch):
        w0, scale = initialize(u0, cold)
        W, st = evolve(w0, nu, scale, data, sch)
        return dict(fields=decode(W, data), W=W, w0=w0, stats=st)

    return jax.jit(query)
