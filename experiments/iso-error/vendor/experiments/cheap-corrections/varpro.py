"""Variable projection, test-count rules and per-rung empirical quadrature.

The reduced state is the audited correction ladder's, unchanged:

    u(z, y) = G ( h_theta(z) + C_q y ),      w = (z, y) in R^{K+q},

with C_q the first q columns of the ladder's own nested residual-POD directions. Only
the solver, the test count M and the quadrature rule change here.

Three solver variants share one residual, one stopping rule and one output contract:

  joint    the audited ladder: damped Levenberg-Marquardt on all K+q unknowns at the
           q = 0 trust radius.
  varpro   y is eliminated. For fixed z the weak residual is quadratic in y (through
           the upwind advection product), so y*(z) is found by a few damped
           Gauss-Newton steps with a three-point safeguard, and the outer LM runs on
           z in R^16 only, at the q = 0 trust radius, with Kaufman's Jacobian
           J_z = dr/dz at y = y*(z). The gradient is exact at inner optimality, so the
           stationarity test is never approximated; only the Hessian model is.
  alt      block coordinate descent: rounds of (LM on z at fixed y, then GN on y).

At q = 0 every variant is special-cased to the joint path with an empty y, so all
three must produce bitwise identical output there.

The initial-condition fit is LINEAR in y for every q, so `varpro`/`alt` eliminate it
exactly by an orthogonal projection onto the complement of col(R_cold C) — the same
construction the Poisson cell already uses — rather than iterating.

Inner Jacobians are taken with `jax.jacfwd` rather than hand-derived. That costs
O(n q^2) instead of O(n q) on the dense-quadrature arms and O(m q^2) on the empirical
quadrature arms, which is negligible where it matters; correctness is worth more than
the dense-arm constant, and the empirical-quadrature arms — where the whole cost
argument lives — are unaffected. `smoke_cheap.py` checks instead that the inner solve
actually drives ||J_y^T r|| down and that variable projection and the joint solver
agree on the solved field at small q, which is what the elimination has to deliver.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import ladder as LD


# --------------------------------------------------------------- the head ----

def corrected_head(params, C, K):
    """h(w) = h_theta(w[:K]) + C w[K:]; at q = 0 this is h_theta verbatim."""
    return LD.corrected_head(params, C, K)


def head_only(params):
    return lambda z: A.sc.head(params, z)


# ----------------------------------------------------------- inner y-solve ----

def make_inner_gn(res, q, iters, damping):
    """A few damped Gauss-Newton steps on y at fixed z, safeguarded.

    res(z, y, *args) is quadratic in y, so this converges from a warm start. The
    accepted iterate is the best of {y, y + dy/2, y + dy} by residual norm, so the
    inner solve can never increase ||r||.
    """
    def inner(z, y0, args):
        def body(_, y):
            r = res(z, y, *args)
            J = jax.jacfwd(lambda yy: res(z, yy, *args))(y)
            H = J.T @ J
            d = jnp.linalg.solve(H + damping * jnp.diag(jnp.diag(H) + 1e-30), -(J.T @ r))
            d = jnp.where(jnp.all(jnp.isfinite(d)), d, 0.)
            cand = jnp.stack((y, y + .5 * d, y + d))
            rn = jax.vmap(lambda yy: jnp.linalg.norm(res(z, yy, *args)))(cand)
            rn = jnp.where(jnp.isfinite(rn), rn, jnp.inf)
            return cand[jnp.argmin(rn)]
        return jax.lax.fori_loop(0, iters, body, y0)
    return inner


def blocks(res, z, y, args):
    """The five norms the joint normalized gradient is reconstructed from."""
    r = res(z, y, *args)
    Jz = jax.jacfwd(lambda zz: res(zz, y, *args))(z)
    Jy = jax.jacfwd(lambda yy: res(z, yy, *args))(y)
    return (jnp.linalg.norm(Jz.T @ r), jnp.linalg.norm(Jz),
            jnp.linalg.norm(Jy.T @ r), jnp.linalg.norm(Jy), jnp.linalg.norm(r))


def joint_gradient(gz, nz, gy, ny, rn):
    return jnp.sqrt(gz ** 2 + gy ** 2) / (jnp.sqrt(nz ** 2 + ny ** 2) * rn + 1e-300)


# ------------------------------------------------------------ the outer LM ----

def make_varpro_lm(res, inner, budget, trust=np.inf, gtol=1e-6, linear='gj'):
    """`arms.make_stationary_lm` with y eliminated inside every evaluation.

    The inner solve is hoisted OUT of the differentiated function: y is computed
    first and then held fixed while the Jacobian in z is taken, so no forward
    tangent is ever pushed through the inner iteration.
    """
    solve = e.gj_solve if linear == 'gj' else jnp.linalg.solve

    def lm(z0, y0, args, tol):
        def probe(z):
            y = inner(z, y0, args)
            r = res(z, y, *args)
            return y, r, jnp.linalg.norm(r)

        def jac(z, y):
            return jax.jacfwd(lambda zz: res(zz, y, *args))(z)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        y, r, rn = probe(z0)
        J = jac(z0, y)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4,
                                     jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, y, r, J, rn, lam, it, reason = s
            H = J.T @ J
            g = J.T @ r
            dz = solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            yn, rn_vec, rn2 = probe(zn)
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            z2 = jnp.where(accept, zn, z)
            y2 = jnp.where(accept, yn, y)
            r2 = jnp.where(accept, rn_vec, r)
            rn2 = jnp.where(accept, rn2, rn)
            J2 = jax.lax.cond(accept, lambda: jac(zn, yn), lambda: J)
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (z2, y2, r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, y, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[6] < budget) & (s[7] == 0), body,
            (z0, y, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, y, rn, it, reason, grad(r, J)
    return lm


def make_block_lm(res, K, q, budget, trust=np.inf, gtol=1e-6, ridge=1e-10):
    """One Jacobian per iteration, but the damping and the trust radius apply to z only.

    This is the audited joint LM with the augmented normal equations split into blocks:

        ( J^T J + lam D_z + eps D_y ) [dz; dy] = - J^T r,

    where D_z zeroes the y diagonal and D_y is a fixed tiny ridge. The y block is
    therefore an exact Gauss-Newton step on the current linearisation and is never
    restricted by the trust radius, which is the audited ladder's binding constraint at
    large q; the z block keeps the retained Levenberg damping and the q = 0 trust
    radius. Cost per iteration is the joint solver's, because the two Jacobian blocks
    come from one forward-mode pass. At q = 0 it IS the joint solver.
    """
    def lm(z0, y0, args, tol):
        def ev(w):
            r = res(w[:K], w[K:], *args)
            J = jax.jacfwd(lambda ww: res(ww[:K], ww[K:], *args))(w)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        mask_z = jnp.concatenate((jnp.ones(K), jnp.zeros(q)))
        mask_y = 1. - mask_z
        w0 = jnp.concatenate((z0, y0))
        r, J, rn = ev(w0)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, rn, lam, it, reason = s
            H = J.T @ J
            d = jnp.diag(H) + 1e-30
            step = jnp.linalg.solve(H + jnp.diag(lam * d * mask_z + ridge * d * mask_y), -(J.T @ r))
            ok = jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step[:K]) <= trust)
            wn = w + jnp.where(ok, step, 0.)
            rn2 = jnp.linalg.norm(res(wn[:K], wn[K:], *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: ev(wn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, wn, w), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        w, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (w0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return w[:K], w[K:], rn, it, reason, grad(r, J)
    return lm


def make_alt_lm(res, inner, budget, rounds, trust=np.inf, gtol=1e-6, linear='gj'):
    """Block coordinate descent: rounds of (LM on z at frozen y, GN on y at frozen z)."""
    per = max(1, int(budget) // int(rounds))
    lm = A.make_stationary_lm(lambda z, y, *args: res(z, y, *args), per, trust, gtol, linear)

    def run(z0, y0, args, tol):
        z, y = z0, y0
        total = jnp.int32(0)
        reason = jnp.int32(0)
        rn = jnp.asarray(0.)
        gz = jnp.asarray(0.)
        for _ in range(int(rounds)):
            z, rn, it, reason, gz = lm(z, (y,) + tuple(args), tol)
            total = total + it
            y = inner(z, y, args)
        return z, y, rn, total, reason, gz
    return run


# ------------------------------------------------------------- the query ------

def make_query(params, C, K, q, L, dt, trust, quadrature, variant, ic_budget=400,
               step_budget=180, gtol=1e-6, linear='gj', inner_iters=3,
               inner_damping=1e-10, alt_rounds=3):
    """The complete matched query: supplied dense field on GPU -> six dense fields.

    Returns, per invocation:
      0 fields          (6, L+1, L+1)
      1 iterations      per time step
      2 residual norms  per time step
      3 exit reasons    per time step
      4 output latents  (6, K+q)
      5 ic iterations   6 ic reason        7 internal latents
      8 outer gradient per step            9 ic outer gradient
     10 ic residual    11 ||u_gauss||
     12 joint normalized gradient per step
     13 ic joint normalized gradient
     14 inner y-gradient per step (||Jy^T r||/(||Jy|| ||r||))
    """
    head = corrected_head(params, C, K)
    hz = head_only(params)
    wk = A.weak_fn(quadrature)
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))

    def res(z, y, prev, nu, data):
        return wk(jnp.concatenate((z, y)), prev, nu, data, head, L, dt)

    joint = (variant == 'joint') or q == 0

    if joint:
        # The audited path, verbatim: one LM on the whole augmented vector.
        lm = A.make_stationary_lm(
            lambda w, p, nu, data: wk(w, p, nu, data, head, L, dt), step_budget, trust, gtol, linear)
        ic = A.make_stationary_lm(lambda w, y, R: R @ head(w) - y, ic_budget, gtol=gtol, linear=linear)
    else:
        inner = make_inner_gn(res, q, inner_iters, inner_damping)
        if variant == 'varpro':
            outer = make_varpro_lm(res, inner, step_budget, trust, gtol, 'gj')
        elif variant == 'alt':
            outer = make_alt_lm(res, inner, step_budget, alt_rounds, trust, gtol, 'gj')
        elif variant == 'block':
            outer = make_block_lm(res, K, q, step_budget, trust, gtol, inner_damping)
        else:
            raise ValueError(variant)
        # The initial fit is linear in y at every q, so it is eliminated exactly.
        ic_lm = A.make_stationary_lm(lambda z, tgt, Rm, P: P(Rm @ hz(z) - tgt),
                                     ic_budget, gtol=gtol, linear='gj')

    def query(u0, nu, data, cold):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        yv = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ yv)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))

        if joint:
            w0, icrn, icit, icreason, icgn = ic(Zcand[idx], (yv, R), 0.)
            icres = R @ head(w0) - yv
            icJ = jax.jacfwd(lambda ww: R @ head(ww) - yv)(w0)
            icgj = jnp.linalg.norm(icJ.T @ icres) / (jnp.linalg.norm(icJ) * jnp.linalg.norm(icres) + 1e-300)
        else:
            RC = R @ C                                   # (R, q), the linear block
            Qr, Rr = jnp.linalg.qr(RC, mode='reduced')
            proj = lambda v: v - Qr @ (Qr.T @ v)          # exact elimination of y
            z0 = Zcand[idx][:K]
            z, icrn, icit, icreason, icgn = ic_lm(z0, (yv, R, proj), 0.)
            ycor = jax.scipy.linalg.solve_triangular(Rr, Qr.T @ (yv - R @ hz(z)), lower=False)
            w0 = jnp.concatenate((z, ycor))
            icres = R @ head(w0) - yv
            icJz = jax.jacfwd(lambda zz: R @ head(jnp.concatenate((zz, ycor))) - yv)(z)
            icgj = jnp.linalg.norm(
                jnp.concatenate((icJz.T @ icres, RC.T @ icres))) / (
                jnp.sqrt(jnp.linalg.norm(icJz) ** 2 + jnp.linalg.norm(RC) ** 2)
                * jnp.linalg.norm(icres) + 1e-300)

        if joint:
            def step(carry, _):
                wv, wprev = carry
                p = data['A'] @ head(wv)
                we = wv + (wv - wprev)
                r0 = jnp.linalg.norm(wk(wv, p, nu, data, head, L, dt))
                re = jnp.linalg.norm(wk(we, p, nu, data, head, L, dt))
                wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
                w2, rn, it, reason, gn = lm(wi, (p, nu, data), 1e-9 * scale)
                # the joint normalized gradient is the retained rule itself here
                return (w2, wv), (w2, rn, it, reason, gn, gn, jnp.asarray(0.))
        else:
            def step(carry, _):
                wv, wprev = carry
                p = data['A'] @ head(wv)
                we = wv + (wv - wprev)
                r0 = jnp.linalg.norm(wk(wv, p, nu, data, head, L, dt))
                re = jnp.linalg.norm(wk(we, p, nu, data, head, L, dt))
                wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
                zi, yi = wi[:K], wi[K:]
                z2, y2, rn, it, reason, _ = outer(zi, yi, (p, nu, data), 1e-9 * scale)
                gzn, nz, gyn, ny, rnorm = blocks(res, z2, y2, (p, nu, data))
                gj = joint_gradient(gzn, nz, gyn, ny, rnorm)
                gz = gzn / (nz * rnorm + 1e-300)
                gy = gyn / (ny * rnorm + 1e-300)
                return (jnp.concatenate((z2, y2)), wv), (
                    jnp.concatenate((z2, y2)), rn, it, reason, gz, gj, gy)

        _, (ws, rn, it, reason, gn, gj, gy) = jax.lax.scan(step, (w0, w0), None, length=steps)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = jax.vmap(lambda v: e.output_field(data['G'] @ head(v), L, L))(W)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn,
                icrn, jnp.linalg.norm(ui), gj, icgj, gy)
    return jax.jit(query)


# -------------------------------------------------- bounded NNLS quadrature ---

def bounded_nnls(G, b, target, seconds, block=None):
    """Block-greedy nonnegative least squares with a walltime bound.

    `engines.nnls_capped` refits the whole support after every single added point,
    which is why m = 512 cost 250 s. This adds `block` points per refit and stops at
    whichever of {target support, walltime} binds, recording which.
    """
    t0 = time.perf_counter()
    block = int(block or max(1, target // 64))
    support = np.zeros(0, dtype=int)
    w = np.zeros(0)
    residual = b.copy()
    refits, truncated, reason = 0, False, 'target'
    while len(support) < target:
        if time.perf_counter() - t0 > seconds:
            truncated, reason = True, 'walltime'
            break
        grad = G.T @ residual
        if support.size:
            grad[support] = -np.inf
        take = min(block, target - len(support))
        order = np.argsort(-grad)[:take]
        order = order[grad[order] > 1e-10 * max(np.linalg.norm(b), 1e-300)]
        if order.size == 0:
            reason = 'gradient'
            break
        support = np.concatenate((support, order))
        w, _ = scipy.optimize.nnls(G[:, support], b, maxiter=20 * len(support))
        refits += 1
        active = w > 1e-14
        support, w = support[active], w[active]
        residual = b - G[:, support] @ w
    return np.asarray(support, dtype=int), w, dict(
        fitter='bounded_block_greedy', block=block, target_support=int(target),
        support=int(len(support)), refits=int(refits), truncated=bool(truncated),
        stop_reason=reason, seconds=time.perf_counter() - t0)


def retained_nnls(G, b, target):
    """`engines.nnls_capped` exactly, so the audited rules are reproduced bitwise."""
    t0 = time.perf_counter()
    supp, w = e.nnls_capped(G, b, target)
    if len(supp) < target:
        rest = np.setdiff1d(np.arange(G.shape[1]), supp)
        supp = np.concatenate((supp, rest[np.argsort(-np.abs(G[:, rest]).mean(0))[:target - len(supp)]]))
    w, _ = scipy.optimize.nnls(G[:, supp], b, maxiter=10 * len(supp))
    return np.asarray(supp, dtype=int), w, dict(
        fitter='retained_nnls_capped', block=1, target_support=int(target),
        support=int(len(supp)), refits=None, truncated=False, stop_reason='target',
        seconds=time.perf_counter() - t0)


def build_operators(bank, L, M, quadrature, head=None, Wcodes=None, m=None, fitter='retained',
                    eq_seed=20259, candidate_cap=8192, fit_states=64, max_fit_rows=None,
                    eq_seconds=240.):
    """`arms.build_operators` with a selectable, walltime-bounded quadrature fitter.

    With `fitter='retained'`, `fit_states=64` and the retained seed this is the audited
    construction row for row, so the q = 0 and q = 16 rules reproduce `qlad01` bitwise.
    """
    t0 = time.perf_counter()
    D = bank.dim
    G = bank.on_grid(L)
    jax.block_until_ready(G)
    Phi, lam, mode_ids = e.modes(L, M)
    P = jnp.asarray(Phi)
    Amat = P.T @ G
    info = dict(intervals=L, interior_unknowns=(L - 1) ** 2, dim=D, M=int(M),
                quadrature=quadrature, mode_ids_count=int(len(mode_ids)))
    data = dict(A=Amat, lam=jnp.asarray(lam), G=G)
    if quadrature == 'dense':
        data['Phi'] = P
        info.update(m=None, eq_relative_fit=None, eq_fit=None)
    else:
        assert m is not None and head is not None and Wcodes is not None
        rng = np.random.default_rng(eq_seed)
        cp = np.sort(rng.choice((L - 1) ** 2, min(candidate_cap, (L - 1) ** 2), replace=False))
        states = fit_states if max_fit_rows is None else int(
            np.clip(max_fit_rows // int(M), 8, fit_states))
        fit = np.sort(rng.choice(len(Wcodes), min(states, len(Wcodes)), replace=False))
        adv = jax.jit(lambda g, w: e.spatial(g @ head(w), L)[0])
        rows, targets = [], []
        for i in fit:
            nvec = adv(G, jnp.asarray(Wcodes[i]))
            targets.append(np.asarray(P.T @ nvec))
            rows.append(Phi[cp].T * np.asarray(nvec)[cp])
        design = np.concatenate(rows)
        b = np.concatenate(targets)
        scale = np.linalg.norm(design, axis=1) + 1e-300
        design /= scale[:, None]
        b /= scale
        if fitter == 'retained':
            supp, w, finfo = retained_nnls(design, b, m)
        else:
            supp, w, finfo = bounded_nnls(design, b, m, eq_seconds)
        pos = cp[supp]
        ij = np.stack(np.unravel_index(pos, (L - 1, L - 1)), 1) + 1
        data['G5'] = bank.stencil(ij, L)
        data['Pq'] = jnp.asarray(Phi[pos] * w[:, None])
        rel = float(np.linalg.norm(design[:, supp] @ w - b) / np.linalg.norm(b))
        finfo.update(design_rows=int(design.shape[0]), fit_states=int(len(fit)),
                     candidates=int(len(cp)), relative_fit=rel)
        info.update(m=int(len(supp)), m_target=int(m), eq_seed=int(eq_seed),
                    eq_actual_candidates=int(len(cp)), eq_fit_rows=fit.tolist(),
                    eq_indices=pos.tolist(), eq_weights=w.tolist(),
                    eq_relative_fit=rel, eq_fit=finfo)
    info['setup_seconds'] = time.perf_counter() - t0
    info['array_bytes'] = int(sum(x.nbytes for x in jax.tree_util.tree_leaves(data)))
    return data, info


# -------------------------------------------------------- enriched codes ------

def enriched_codes(Zstar, rho, Rb, Ct, q):
    """w_i = (z*_i, y_i) with y_i the field-metric least-squares correction.

    Ct is field-orthonormal (C = Rb^{-1} Ct), so y_i = Ct[:, :q]^T Rb rho_i is exactly
    the best correction for the head residual rho_i on the q-rung manifold.
    """
    Zstar = np.asarray(Zstar)
    if q == 0:
        return np.concatenate((Zstar, np.zeros((len(Zstar), 0))), axis=1)
    Y = np.asarray(jnp.asarray(rho) @ Rb.T @ jnp.asarray(Ct)[:, :q])
    return np.concatenate((Zstar, Y), axis=1)
