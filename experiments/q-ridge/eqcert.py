"""Empirical-quadrature rule certification: reachable states, a GPU fitter, and rho.

`q-diag` (2026-09-16) showed that the Burgers correction ladder's evolved-times regression
is the empirical quadrature: the rule's own relative error on the advection functional,

    rho(u) = || sum_j w_j Phi(x_j) a(u)(x_j) - Phi^T a(u) || / || Phi^T a(u) || ,

rises with q at fixed m while its NNLS relative fit stays flat and is anti-correlated with
rho at the top. A rule cannot be certified by its own fit residual.

This module supplies the three pieces that fixes that, and nothing else:

  `collect_states`   the REACHABLE population: the ROM's own dense-quadrature rollouts on
                     training-family trajectories, recording the converged step solution AND
                     the intermediate Levenberg-Marquardt iterates, by unrolling a fixed
                     number of steps of the retained step formula.
  `fit_rule`         block-greedy support selection plus an EXACT active-set nonnegative
                     least-squares solve on the Gram matrix, float64 on the GPU, so m = 8192
                     is reachable inside a job. scipy's Lawson-Hanson NNLS took about 1000 s
                     at m = 2048; `smoke_eqcert.py` gates this fitter against it at small m.
  `certify`          rho on a HELD-OUT set of reachable states, reported as max, 95th
                     percentile and median, never as the fit residual.

The online residual `arms.weak_eq` is untouched: a rule is exactly the pair (nodes, weights)
it always was, so a certified rule drops into the retained query with no other change.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import varpro as VP


# ------------------------------------------------- the reachable population ---

def make_collect_query(params, C, K, q, L, dt, trust, iters=12, ic_budget=400,
                       gtol=1e-6, linear='gj', inner_damping=1e-10):
    """Roll the ROM out with DENSE quadrature and return every iterate it visits.

    The step formula is `varpro.make_block_lm`'s, verbatim, but unrolled to a FIXED number
    of iterations with `scan` so each iterate can be returned instead of only the last. The
    initializer policy, the trust radius, the damping schedule and the acceptance test are
    the retained ones, so these iterates are the states the production solver visits (its
    median is 3-5 iterations, well inside `iters`).

    Untimed: this never runs inside a timed query and its output is a fit population, not a
    result. Dense quadrature by construction, so the population cannot depend on the rule
    being fitted.

    Returns (states, converged) with states (steps, iters + 1, K + q) every visited iterate
    and converged (steps, K + q) the last iterate of each step.
    """
    head = VP.corrected_head(params, C, K)
    hz = VP.head_only(params)
    wk = A.weak_fn('dense')
    steps = int(round(.25 / dt))
    mask_z = jnp.concatenate((jnp.ones(K), jnp.zeros(q)))
    mask_y = 1. - mask_z

    def res(w, prev, nu, dat):
        return wk(w, prev, nu, dat, head, L, dt)

    def lm_unrolled(w0, args):
        def ev(w):
            r = res(w, *args)
            J = jax.jacfwd(lambda ww: res(ww, *args))(w)
            return r, J, jnp.linalg.norm(r)

        def body(state, _):
            w, r, J, rn, lam = state
            H = J.T @ J
            d = jnp.diag(H) + 1e-30
            step = jnp.linalg.solve(H + jnp.diag(lam * d * mask_z + inner_damping * d * mask_y),
                                    -(J.T @ r))
            ok = jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step[:K]) <= trust)
            wn = w + jnp.where(ok, step, 0.)
            rn2 = jnp.linalg.norm(res(wn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: ev(wn), lambda: (r, J, rn))
            w2 = jnp.where(accept, wn, w)
            lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            return (w2, r2, J2, rn2, lam2), w2

        r, J, rn = ev(w0)
        _, seen = jax.lax.scan(body, (w0, r, J, rn, jnp.asarray(1e-6)), None, length=iters)
        return jnp.concatenate((w0[None], seen))

    ic_lm = A.make_stationary_lm(lambda z, tgt, Rm, P: P(Rm @ hz(z) - tgt),
                                 ic_budget, gtol=gtol, linear='gj')
    joint_ic = A.make_stationary_lm(lambda w, y, R: R @ head(w) - y, ic_budget,
                                    gtol=gtol, linear=linear)

    def body(u0, nu, data, cold):
        # `data` is an ARGUMENT, never a closure: the bank G and the mode matrix Phi are
        # hundreds of megabytes and XLA would embed them as captured constants.
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        yv = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ yv)
        if q == 0:
            w0 = joint_ic(Zcand[idx], (yv, R), 0.)[0]
        else:
            RC = R @ C
            Qr, Rr = jnp.linalg.qr(RC, mode='reduced')
            proj = lambda v: v - Qr @ (Qr.T @ v)
            z = ic_lm(Zcand[idx][:K], (yv, R, proj), 0.)[0]
            ycor = jax.scipy.linalg.solve_triangular(Rr, Qr.T @ (yv - R @ hz(z)), lower=False)
            w0 = jnp.concatenate((z, ycor))

        def step(carry, _):
            wv, wprev = carry
            p = data['A'] @ head(wv)
            we = wv + (wv - wprev)
            r0 = jnp.linalg.norm(wk(wv, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(wk(we, p, nu, data, head, L, dt))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            seen = lm_unrolled(wi, (p, nu, data))
            return (seen[-1], wv), seen

        _, seen = jax.lax.scan(step, (w0, w0), None, length=steps)
        return seen, seen[:, -1]

    return jax.jit(body)


def static_codes(Zstar, rho, Rb, Ct, q, Zold):
    """The INCUMBENT fit population: static decoder outputs, `varpro.enriched_codes`."""
    if q == 0:
        return np.concatenate((np.asarray(Zold), np.zeros((len(Zold), 0))), axis=1)
    return VP.enriched_codes(Zstar, rho, Rb, Ct, q)


# ---------------------------------------------------------- the GPU fitter ----

def _masked_solve(H, c, mask, ridge):
    """Unconstrained least squares restricted to the free set, at fixed array shape.

    Rows and columns outside the free set are replaced by the identity, so the linear
    solve has the same shape at every active-set iteration and XLA compiles it once.
    """
    m = mask.astype(H.dtype)
    A = H * m[:, None] * m[None, :] + jnp.diag(jnp.where(mask, ridge, 1.))
    w = jnp.linalg.solve(A, c * m)
    return jnp.where(mask, w, 0.)


_solve = jax.jit(_masked_solve)


def _refine(H, c, w, mask, ridge, rounds=3):
    """Iterative refinement on the free set: the tiny Gram ridge is a solve device, not
    part of the objective, so its bias is removed rather than reported."""
    for _ in range(int(rounds)):
        w = w + _solve(H, c - H @ w, mask, ridge)
    return jnp.where(mask, w, 0.)


def _nnls_on_support(H, c, support, ridge, iters=40):
    """Exact nonnegative least squares on a fixed support, by active set.

    Solve unconstrained on the free set, drop every negative weight, repeat. This is the
    classical fast-NNLS inner loop; it terminates when no weight is negative, and each
    iteration is one dense solve of the Gram system, which is milliseconds on a GPU even at
    n = 16384. Lawson-Hanson could not reach m = 8192 in a job; this can.
    """
    free = support
    w = jnp.zeros(H.shape[0], dtype=H.dtype)
    used = 0
    for used in range(1, int(iters) + 1):
        w = _solve(H, c, free, ridge)
        w = _refine(H, c, w, free, ridge)
        neg = free & (w < 0.)
        if not bool(jnp.any(neg)):
            break
        free = free & ~neg
    return w, free, used


def gpu_nnls(design, b, target, blocks=8, ridge_rel=1e-12, inner_iters=40, seconds=None,
             **_ignored):
    """Block-greedy nonnegative least squares, solved exactly on the GPU.

    The support is grown in `blocks` passes by the residual correlation — the retained
    block-greedy rule — and the inner solve is an exact active-set NNLS on the Gram matrix
    rather than scipy's Lawson-Hanson, which is what makes m = 8192 reachable inside a job.
    Everything is float64. The Gram matrix is formed once and every pass reuses it, so the
    design matrix is touched exactly twice.
    """
    t0 = time.perf_counter()
    D = jnp.asarray(design)
    bv = jnp.asarray(b)
    n = int(D.shape[1])
    H = D.T @ D
    c = D.T @ bv
    bb = float(jnp.dot(bv, bv))
    ridge = float(ridge_rel * (jnp.mean(jnp.diag(H)) + 1e-300))
    target = int(min(target, n))
    blocks = max(1, int(blocks))
    per = max(1, target // blocks)
    chosen = np.zeros(n, dtype=bool)
    w = jnp.zeros(n, dtype=H.dtype)
    truncated, reason, passes, inner_used = False, 'target', 0, 0
    while int(chosen.sum()) < target:
        if seconds is not None and time.perf_counter() - t0 > seconds:
            truncated, reason = True, 'walltime'
            break
        g = np.array(c - H @ w, copy=True)                 # -d/dw of 0.5||Dw-b||^2
        g[chosen] = -np.inf
        take = min(per, target - int(chosen.sum()))
        order = np.argsort(-g)[:take]
        order = order[g[order] > 1e-12 * max(bb, 1e-300) ** .5]
        if order.size == 0:
            reason = 'gradient'
            break
        chosen[order] = True
        w, free, used = _nnls_on_support(H, c, jnp.asarray(chosen), ridge, inner_iters)
        chosen = np.array(free, copy=True)
        passes += 1
        inner_used = max(inner_used, used)
    wn = np.asarray(w)
    keep = wn > 0.
    support = np.flatnonzero(keep)
    wn = wn[keep]
    # Measured on the DESIGN, not on the Gram: ||D w||^2 - 2 c^T w + ||b||^2 cancels
    # catastrophically when the fit is near-exact and would report sqrt(eps) instead of eps.
    rel = float(jnp.linalg.norm(D @ w - bv) / jnp.maximum(jnp.linalg.norm(bv), 1e-300))
    obj = float(jnp.dot(w, H @ w) - 2 * jnp.dot(c, w) + bb)
    return support.astype(int), wn, dict(
        fitter='gpu_block_greedy_active_set', blocks=blocks, target_support=target,
        support=int(len(support)), passes=passes, max_active_set_iterations=int(inner_used),
        truncated=bool(truncated), stop_reason=reason, relative_fit=rel,
        gram_relative_fit=float(np.sqrt(max(obj, 0.)) / max(np.sqrt(bb), 1e-300)),
        seconds=time.perf_counter() - t0)


# ----------------------------------------------------- the rule and its rho ---

def fit_rule(bank, G, Phi, L, M, m, coefficients, candidates, fitter='gpu', blocks=8,
             inner_iters=40, final_iters=None, seconds=None, scipy_blocks=16):
    """Fit one m-point rule from a population given as BANK COEFFICIENTS.

    `coefficients` is (S, R): each row decodes to a fit state u = G c. That is the only
    interface, so the static (incumbent) and reachable populations enter identically and the
    population is the single manipulated variable.
    """
    t0 = time.perf_counter()
    P = jnp.asarray(Phi)
    cp = np.asarray(candidates)
    Pc = np.asarray(Phi)[cp]                                   # (candidates, M)
    adv = jax.jit(lambda g, c: e.spatial(g @ c, L)[0])
    rows, targets = [], []
    for c in np.asarray(coefficients):
        n = adv(G, jnp.asarray(c))
        targets.append(np.asarray(P.T @ n))
        rows.append(Pc.T * np.asarray(n)[cp])
    design = np.concatenate(rows)
    b = np.concatenate(targets)
    scale = np.linalg.norm(design, axis=1) + 1e-300
    design /= scale[:, None]
    b /= scale
    if fitter == 'gpu':
        supp, w, info = gpu_nnls(design, b, m, blocks=blocks, inner_iters=inner_iters,
                                 seconds=seconds)
    else:
        supp, w, info = (VP.retained_nnls(design, b, m) if fitter == 'retained'
                         else VP.bounded_nnls(design, b, m, seconds or 3600.,
                                              block=max(1, int(m) // scipy_blocks)))
        info.setdefault('relative_fit', float(
            np.linalg.norm(design[:, supp] @ w - b) / max(np.linalg.norm(b), 1e-300)))
    pos = cp[supp]
    ij = np.stack(np.unravel_index(pos, (L - 1, L - 1)), 1) + 1
    rule = dict(G5=bank.stencil(ij, L), Pq=jnp.asarray(np.asarray(Phi)[pos] * w[:, None]))
    info.update(m=int(len(supp)), m_target=int(m), M=int(M), fit_states=int(len(coefficients)),
                design_rows=int(design.shape[0]), candidates=int(len(cp)),
                nodes=pos.tolist(), weights=w.tolist(), total_seconds=time.perf_counter() - t0,
                eq_rule_valid=bool(not info['truncated']))
    return rule, info


def certify(G, Phi, L, rule, coefficients, chunk=32):
    """rho on a held-out population: q-diag's definition, verbatim.

    rho(u) = || sum_j w_j Phi(x_j) a(u)(x_j) - Phi^T a(u) || / || Phi^T a(u) ||

    The sampled term is formed exactly as `arms.weak_eq` forms it, from the rule's own
    five-point stencil, so this measures the operator the ROM will actually run.
    """
    P = jnp.asarray(Phi)
    G5, Pq = rule['G5'], rule['Pq']

    def one(c, g, p, g5, pq):
        full = p.T @ e.spatial(g @ c, L)[0]
        us = jnp.einsum('msr,r->ms', g5, c)
        cc, xp, xm, yp, ym = [us[:, i] for i in range(5)]
        a = cc * L * (jnp.where(cc > 0, cc - xm, xp - cc) + jnp.where(cc > 0, cc - ym, yp - cc))
        samp = pq.T @ a
        return jnp.linalg.norm(samp - full) / (jnp.linalg.norm(full) + 1e-300)

    # Every large array is an ARGUMENT; nothing is captured as a compile-time constant.
    batch = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None)))
    vals = []
    Cs = jnp.asarray(np.asarray(coefficients))
    for s in range(0, len(Cs), chunk):
        vals.append(np.asarray(batch(Cs[s:s + chunk], G, P, G5, Pq)))
    v = np.concatenate(vals)
    return dict(states=int(len(v)), rho_max=float(v.max()), rho_p95=float(np.quantile(v, .95)),
                rho_median=float(np.median(v)), rho_mean=float(v.mean()),
                rho_min=float(v.min()))
