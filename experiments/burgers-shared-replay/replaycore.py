# Copy of hcore: only an optional, guarded replay guess and its evidence are added.
"""jcp-heads core: the nonlinear-head reduced step on the frozen rotated bank (DESIGN 1, 3.1).

The head ROM is P1's two-step LSPG stepper (`p1core.make_evolve`, form LSPG) with the coefficient written as
c = h~(z) = L h_theta(z) (the head folded into the rotated basis, `bkfast.fold_head`), unknown z in R^K:

    R_n(z) = A (a0 h~(z) + a1 c_n + a2 c_{n-1}) + dt (b0 F(h~(z)) + b1 F(c_n)),   F(c) = N(c) + nu Lambda A c,
    r_n(z) = D R_n(z),  D = 1 / (a0 + dt b0 nu Lambda),       min_z ||r_n(z)||,

with the analytic Jacobian J_z = (A + dt b0 S dN(c)) dh~/dz, the advection part through the rule's tangent block
(`qcore.tested_jac('point')(c, data, Jh)`). The LM is P1's `make_lm` (= hfast.make_fused_lm at q = 0: K-only damping,
the step clipped onto the trust radius, one (r, J) evaluation per iteration) with solver 'lu' (the deployed head
query's `jnp.linalg.solve`). With scheme BE this is, operation for operation, the deployed head query
`qcore.make_head_query('point', ...)` (hfast.make_query with qcore.make_head_eval): same residual, Jacobian, predictor
(best of current / linear / quadratic extrapolation by residual norm), damping carry, tolerance 1e-9 * scale, budget,
initial fit (Gauss-48 sampling, nearest candidate code, stationary GJ LM). The reproduction gate (DESIGN 7) checks it.

Also here: the head's in-job certifier (gamma from a forward-mode autodiff Jacobian in z of the rebuilt residual, at
every step, from the saved internal z trajectory) and the head step Jacobian for conditioning (Q3).
Every large array is a jit argument. f64.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import p1core as T2

Q = T2.load_qcore()
A_ = Q.A                       # vendored arms.py (make_stationary_lm, build_cold)
NSTEP = T2.NSTEP


def head_fn(pf):
    """z -> h~(z) for folded head parameters pf (rotated basis, R' = 512)."""
    return lambda z: Q.sc.head(pf, z)


def make_head_evolve(hz, nl, nlJ, K, trust, budget=600, ridge=1e-10, solver='lu', clip=True, mutation=None, nrec=0,
                     nstep=NSTEP, seed=None, stop_on_invalid=False, force_failure_step=None):
    """evolve(z0, nu, scale, data, sch) -> (Z (6, K), stats), the text of p1core.make_evolve (form LSPG) with
    c = hz(z). data: A (M, R'), lam (M,), hmask (M,) plus the point-rule blocks. mutation in (None, 'jflip')."""
    assert mutation in (None, 'jflip')
    jh = jax.jacfwd(hz)

    def inner(c, Ac, val, hist, nu, data, co, dt):
        a0, a1, a2, b0, b1 = co[0], co[1], co[2], co[3], co[4]
        Acn, Acm, Fn = hist
        lam = data['lam']
        return a0 * Ac + a1 * Acn + a2 * Acm + dt * (b0 * (val + nu * lam * Ac) + b1 * Fn)

    def dscale(co, dt, nu, lam):
        return co[0] + dt * co[3] * nu * lam

    def res(z, args):
        hist, nu, data, co, dt = args
        c = hz(z)
        Ac = data['A'] @ c
        return inner(c, Ac, nl(c, data), hist, nu, data, co, dt) / dscale(co, dt, nu, data['lam'])

    def evalJ(z, args):
        hist, nu, data, co, dt = args
        c = hz(z)
        Jh = jh(z)                                              # (R', K)
        Ac = data['A'] @ c
        val, dNJ = nlJ(c, data, Jh)                            # N(c), dN/dc @ Jh  (M, K)
        R = inner(c, Ac, val, hist, nu, data, co, dt)
        S = 1. / dscale(co, dt, nu, data['lam'])
        sgn = -1. if mutation == 'jflip' else 1.
        return R * S, data['A'] @ Jh + sgn * (dt * co[3] * S)[:, None] * dNJ

    lm = T2.make_lm(evalJ, budget, trust, ridge, solver, clip)
    be = jnp.asarray(T2.BE_CO, jnp.float64)

    def evolve(z0, nu, scale, data, sch, initial_valid=True):
        A, lam, hm = data['A'], data['lam'], data['hmask']
        dt = sch['dt']
        M = A.shape[0]
        Fval = lambda c: nl(c, data) + nu * lam * (A @ c)
        z5 = jnp.zeros(5, jnp.int32)
        out0 = jnp.zeros((6, K), jnp.float64).at[0].set(z0)
        stats0 = dict(it_sum=jnp.int32(0), it_max=jnp.int32(0), rej=jnp.int32(0), exits=z5, nfail=jnp.int32(0),
                      first_fail=jnp.int32(-1), worst_ratio=jnp.float64(0.), worst_tolratio=jnp.float64(0.),
                      alt_num=jnp.float64(0.), alt_den=jnp.float64(0.), amp=jnp.float64(0.),
                      step_gn=jnp.full(nstep, jnp.nan), step_tolratio=jnp.full(nstep, jnp.nan),
                      step_reason=jnp.full(nstep, -1, jnp.int32), step_it=jnp.full(nstep, -1, jnp.int32),
                      replay_info=jnp.zeros((nstep, 5), jnp.float64))
        if nrec:
            stats0['states'] = jnp.full((nrec, K), jnp.nan).at[0].set(z0)
        if stop_on_invalid:
            stats0['failed_code'] = jnp.full(K, jnp.nan)
            stats0['initial_valid'] = jnp.asarray(initial_valid)
            stats0['states'] = jnp.where(initial_valid, stats0['states'], jnp.nan)
        nA0 = jnp.linalg.norm(A @ hz(z0)) + 1e-300
        thr = T2.ALT_TAU * jnp.max(jnp.abs(A @ hz(z0)))

        def body(cr):
            k, zv, zprev, zprev2, lam0, dprev, runs, out, st = cr
            co = jnp.where(k < sch['nstart'], be, sch['co'])
            cv, cp = hz(zv), hz(zprev)
            Acn = A @ cv
            Acm = A @ cp
            Fn = jax.lax.cond(co[4] != 0., lambda: Fval(cv), lambda: jnp.zeros(M, jnp.float64))
            args = ((Acn, Acm, Fn), nu, data, co, dt)
            ze = zv + (zv - zprev)
            zqd = jnp.where(k >= 2, 3. * zv - 3. * zprev + zprev2, ze)
            cand = jnp.stack((zv, ze, zqd))
            rs = jax.vmap(lambda zz: jnp.linalg.norm(res(zz, args)))(cand)
            rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
            zi = cand[jnp.argmin(rs)]
            tol = sch['tolf'] * scale
            if seed is not None:
                zi, replay_info = seed(k, zi, args)
            else:
                replay_info = jnp.zeros(5, jnp.float64)
            z2, rn, it, reason, gn, rej, lamn = lm(zi, args, tol, sch['gtol'], lam0)
            if force_failure_step is not None:
                reason = jnp.where(k == force_failure_step, jnp.int32(3), reason)
            conv = (gn <= sch['gtol']) | (rn <= tol)
            ok = jnp.isfinite(rn) & conv & (reason != 0) & (reason != 3)
            if stop_on_invalid:
                st = dict(st, failed_code=jnp.where(ok, st['failed_code'], z2))
                z2 = jnp.where(ok, z2, zv)
            Ac2 = A @ hz(z2)
            dcur = hm * (Ac2 - Acn)
            pr = dcur * dprev
            qual = (k >= 1) & (jnp.minimum(jnp.abs(dcur), jnp.abs(dprev)) > thr)
            event = qual & (pr < 0)
            runs = jnp.where(event, runs + 1, 0)
            st = dict(it_sum=st['it_sum'] + it, it_max=jnp.maximum(st['it_max'], it), rej=st['rej'] + rej,
                      exits=st['exits'].at[reason].add(1), nfail=st['nfail'] + (~ok).astype(jnp.int32),
                      first_fail=jnp.where((st['first_fail'] < 0) & (~ok), k, st['first_fail']),
                      worst_ratio=jnp.maximum(st['worst_ratio'], gn),
                      worst_tolratio=jnp.maximum(st['worst_tolratio'], rn / (tol + 1e-300)),
                      alt_num=st['alt_num'] + jnp.sum(jnp.where(event & (runs >= 3), jnp.abs(pr), 0.)),
                      alt_den=st['alt_den'] + jnp.sum(jnp.where(qual, jnp.abs(pr), 0.)),
                      amp=jnp.maximum(st['amp'], jnp.linalg.norm(dcur) / nA0),
                      step_gn=st['step_gn'].at[jnp.minimum(k, nstep - 1)].set(gn),
                      step_tolratio=st['step_tolratio'].at[jnp.minimum(k, nstep - 1)].set(rn / (tol + 1e-300)),
                      step_reason=st['step_reason'].at[jnp.minimum(k, nstep - 1)].set(reason),
                      step_it=st['step_it'].at[jnp.minimum(k, nstep - 1)].set(it),
                      replay_info=st['replay_info'].at[jnp.minimum(k, nstep - 1)].set(replay_info),
                      **({'failed_code': st['failed_code'], 'initial_valid': st['initial_valid']} if stop_on_invalid else {}),
                      **({'states': st['states'].at[jnp.minimum(k + 1, nrec - 1)].set(
                          jnp.where(ok, z2, jnp.nan) if stop_on_invalid else z2)} if nrec else {}))
            kk = k + 1
            store = (kk % sch['keep']) == 0
            idx = jnp.clip(kk // sch['keep'], 0, 5)
            out = jnp.where(store, out.at[idx].set(z2), out)
            return (kk, z2, zv, zprev, lamn, dcur, runs, out, st)

        c0 = (jnp.int32(0), z0, z0, z0, jnp.asarray(1e-6, jnp.float64), jnp.zeros(M, jnp.float64),
              jnp.zeros(M, jnp.int32), out0, stats0)
        def keep_going(c):
            if stop_on_invalid:
                return (c[0] < sch['steps']) & initial_valid & (c[-1]['nfail'] == 0)
            return c[0] < sch['steps']
        k, *_rest, out, st = jax.lax.while_loop(keep_going, body, c0)
        if stop_on_invalid:
            valid_steps = k - st['nfail']
            available = initial_valid & (jnp.arange(6) * sch['keep'] <= valid_steps)
            out = jnp.where(available[:, None], out, jnp.nan)
            st = dict(st, valid_steps=valid_steps, output_available=available)
        st = dict(st, steps=k, alt_index=jnp.where(st['alt_den'] > 0, st['alt_num'] / (st['alt_den'] + 1e-300), 0.))
        return out, st

    return evolve, res, evalJ


def make_head_query(pf, K, L, trust, mutation=None, budget=600, nrec=0, nstep=NSTEP, ic_budget=400, ic_gtol=1e-6,
                    seed=None, stop_on_invalid=False, force_failure_step=None):
    """Split head query (init / evolve / decode / fused), the head analogue of p1core.make_split_query.
    init: the deployed q = 0 initial fit (hfast.make_query.initialize): Gauss-48 sampling of the supplied field,
    nearest candidate code by the whitened distance, stationary GJ LM in z (budget ic_budget, gtol ic_gtol)."""
    import engines as e
    import hops as H
    hz = head_fn(pf)
    nl, nlJ = Q.tested_value('point', L), Q.tested_jac('point', L)
    evolve, res, evalJ = make_head_evolve(hz, nl, nlJ, K, trust, budget=budget, mutation=mutation, nrec=nrec,
                                          nstep=nstep, seed=seed, stop_on_invalid=stop_on_invalid,
                                          force_failure_step=force_failure_step)
    ic_lm = A_.make_stationary_lm(lambda z, tgt, Rm: Rm @ hz(z) - tgt, ic_budget, gtol=ic_gtol, linear='gj')

    def initialize(u0, cold):
        xy, wq, Qc, Rc, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Qc.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ yv)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        z0, icrn, icit, icreason, icgn = ic_lm(Zcand[idx][:K], (yv, Rc), 0.)
        return z0, scale, dict(ic_it=icit, ic_reason=icreason, ic_gn=icgn, ic_rn=icrn, ic_idx=idx,
                               ic_rel=icrn / (jnp.linalg.norm(yv) + 1e-300))

    def decode(Z, data):
        Hm = jax.vmap(hz)(Z)                                     # (6, R')
        U = H.bank_apply(data['G'], Hm.T).T.reshape(len(Z), L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold, sch):
        z0, scale, ic = initialize(u0, cold)
        if stop_on_invalid:
            ic_ok = ((ic['ic_reason'] == 1) | (ic['ic_reason'] == 4)) & jnp.all(jnp.isfinite(z0)) & jnp.isfinite(ic['ic_gn'])
            Z, st = evolve(z0, nu, scale, data, sch, initial_valid=ic_ok)
        else:
            Z, st = evolve(z0, nu, scale, data, sch)
        return dict(fields=decode(Z, data), W=Z, w0=z0, stats=dict(st, **ic))

    return dict(init=jax.jit(initialize), evolve=jax.jit(evolve), decode=jax.jit(decode), fused=jax.jit(query),
                nl=nl, hz=hz, res=jax.jit(res), evalJ=jax.jit(evalJ))


# ------------------------------------------------------------------ certification ----

def make_head_certifier(hz, nl):
    """Per-step stationarity certificate (DESIGN 3.1): from the saved internal trajectory z_0..z_S rebuild every step's
    LSPG residual in z (scheme coefficients incl. the BE start-up, history A c_k, A c_{k-1}, F(c_k) with c = hz(z)) and
    form its Jacobian by forward-mode autodiff in z (not the solver's analytic Jacobian). Returns gamma_k, ||r_k||."""
    be = jnp.asarray(T2.BE_CO, jnp.float64)

    def one(k, states, nu, data, sch):
        dt = sch['dt']
        co = jnp.where(k < sch['nstart'], be, sch['co'])
        ck, cm = hz(states[k]), hz(states[jnp.maximum(k - 1, 0)])
        A, lam = data['A'], data['lam']
        Fn = jnp.where(co[4] != 0., nl(ck, data) + nu * lam * (A @ ck), 0.)
        rc = T2.lspg_residual(nl, co, dt, nu, data, (A @ ck, A @ cm, Fn))
        r = lambda z: rc(hz(z))
        zn = states[k + 1]
        rv = r(zn)
        J = jax.jacfwd(r)(zn)
        g = J.T @ rv
        rn = jnp.linalg.norm(rv)
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * rn + 1e-300), rn

    def cert(states, nu, data, sch):
        S = states.shape[0] - 1
        return jax.lax.map(lambda k: one(k, states, nu, data, sch), jnp.arange(S))
    return jax.jit(cert)


def make_head_jac(hz, nl):
    """History-free head step Jacobian dr/dz (autodiff) for Q3: r(z) = D (a0 A h~(z) + dt b0 F(h~(z)))."""
    def jac(z, co, dt, nu, data):
        M = data['A'].shape[0]
        zz = jnp.zeros(M, jnp.float64)
        rc = T2.lspg_residual(nl, co, dt, nu, data, (zz, zz, zz))
        return jax.jacfwd(lambda x: rc(hz(x)))(z)
    return jax.jit(jac)


def trust_of(Z):
    """The deployed head trust radius (qcore.Model.trust_head): 0.01 max_i ||z_i - mean||, over the head's codes."""
    Z = np.asarray(Z)
    return .01 * float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))


def zsub_of(Z):
    """Candidate codes of the initial fit (qstudy: Zsub = Z[::max(1, len(Z) // 8192)])."""
    Z = np.asarray(Z)
    return Z[::max(1, len(Z) // 8192)]
