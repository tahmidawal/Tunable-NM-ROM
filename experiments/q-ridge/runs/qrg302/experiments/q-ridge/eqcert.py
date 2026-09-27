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
  `fit_rule`         support selection by a projected-gradient solve of the full
                     nonnegative problem, then an exact drop-loop refinement on the chosen
                     support, float64 on the GPU, so m = 8192 is reachable inside a job.
                     scipy's Lawson-Hanson took about 1000 s at m = 2048;
                     `checks/fitter_bench.py` gates this fitter against it up to m = 2048.
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

def _ls_on_free(Hs, cs, free, ridge):
    """Unconstrained least squares on the free subset of a Gram block, at fixed shape.

    Rows and columns outside the free set are replaced by the identity, so every call at a
    given block width compiles once.
    """
    m = free.astype(Hs.dtype)
    A = Hs * m[:, None] * m[None, :] + jnp.diag(jnp.where(free, ridge, 1.))
    w = jnp.linalg.solve(A, cs * m)
    return jnp.where(free, w, 0.)


_ls = jax.jit(_ls_on_free)


def _fista_impl(H, c, lip, iters):
    """min_w 0.5 w^T H w - c^T w subject to w >= 0, by projected gradient with momentum.

    Used only to CHOOSE the support -- the weights it returns are refined exactly afterwards
    -- so its slow tail convergence does not enter any reported number. It is what replaces
    the greedy selection that made attempt `qrg301` cycle.
    """
    step = 1. / jnp.maximum(lip, 1e-300)

    def body(i, st):
        w, y, t = st
        wn = jnp.maximum(y - step * (H @ y - c), 0.)
        tn = (1. + jnp.sqrt(1. + 4. * t * t)) / 2.
        return (wn, wn + ((t - 1.) / tn) * (wn - w), tn)

    z = jnp.zeros(H.shape[0], dtype=H.dtype)
    w, _, _ = jax.lax.fori_loop(0, iters, body, (z, z, jnp.asarray(1.)))
    return w


_fista = jax.jit(_fista_impl, static_argnums=(3,))


def _drop_loop(Hs, cs, free0, ridge, iters=40, refine=2):
    """Exact least squares on the free set, dropping every non-positive weight, to a fixed
    point. The free set only ever shrinks, so this cannot cycle; the returned point is the
    exact minimiser over the surviving columns. Iterative refinement removes the bias of the
    tiny Gram ridge, which is a solve device and not part of the objective.
    """
    free = free0
    w = jnp.zeros(Hs.shape[0], dtype=Hs.dtype)
    used = 0
    for used in range(1, int(iters) + 1):
        w = _ls(Hs, cs, free, ridge)
        bad = free & (w <= 0.)
        if not bool(jnp.any(bad)):
            break
        free = free & ~bad
    for _ in range(int(refine)):
        w = w + _ls(Hs, cs - Hs @ w, free, ridge)
    return jnp.where(free & (w > 0.), w, 0.), free, used


def gpu_nnls(design, b, target, fista_iters=6000, drop_iters=40, reentry=3,
             ridge_rel=1e-12, seconds=None, **_ignored):
    """Nonnegative least squares with a bounded support, solved on the GPU in float64.

    Two stages, because scipy's Lawson-Hanson cannot reach m = 8192 inside a job and a
    gradient-greedy outer loop cycles on degenerate zero steps (which is exactly how attempt
    `qrg301` failed, at 97 of 256 points):

      1 a projected-gradient (FISTA) solve of the FULL nonnegative problem over every
        candidate point, used only to RANK the candidates;
      2 the `target` largest of those, refined by an exact least-squares drop loop that
        cannot cycle, followed by up to `reentry` rounds that re-fill the support with the
        best remaining positive-gradient candidates.

    Every reported weight comes from stage 2, so stage 1's slow tail convergence never
    enters a number. The relative fit is measured on the design, not on the Gram, because
    the Gram form cancels catastrophically when the fit is near-exact.
    """
    t0 = time.perf_counter()
    D = jnp.asarray(design)
    bv = jnp.asarray(b)
    n = int(D.shape[1])
    H = D.T @ D
    c = D.T @ bv
    bnorm = float(jnp.linalg.norm(bv))
    ridge = float(ridge_rel * (jnp.mean(jnp.diag(H)) + 1e-300))
    target = int(min(target, n))
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (n,), dtype=H.dtype)
    for _ in range(40):
        v = H @ v
        v = v / (jnp.linalg.norm(v) + 1e-300)
    lip = float(jnp.dot(v, H @ v) / (jnp.dot(v, v) + 1e-300))
    wf = np.asarray(_fista(H, c, lip, int(fista_iters)))
    fista_support = int((wf > 0).sum())
    order = np.argsort(-wf)[:target]
    sel = np.sort(order[wf[order] > 0.])
    if sel.size == 0:
        sel = np.sort(np.argsort(-np.asarray(c))[:target])
    rounds, used = 0, 0
    w_full = jnp.zeros(n, dtype=H.dtype)
    reason = 'target'
    for rounds in range(int(reentry) + 1):
        pad = target - len(sel)
        idx = jnp.asarray(np.concatenate((sel, np.zeros(max(pad, 0), dtype=int))))
        S = jnp.asarray(np.concatenate((np.ones(len(sel), bool),
                                        np.zeros(max(pad, 0), bool))))
        wb, _, u = _drop_loop(H[idx][:, idx], c[idx], S, ridge, drop_iters)
        used = max(used, u)
        wb = np.asarray(wb)[:len(sel)]
        keep = wb > 0.
        sel = sel[keep]
        w_full = jnp.zeros(n, dtype=H.dtype).at[jnp.asarray(sel)].set(jnp.asarray(wb[keep]))
        if len(sel) >= target or (seconds is not None
                                  and time.perf_counter() - t0 > seconds):
            break
        g = np.array(c - H @ w_full, copy=True)
        g[sel] = -np.inf
        add = np.argsort(-g)[:target - len(sel)]
        add = add[g[add] > 1e-13 * max(bnorm, 1e-300)]
        if add.size == 0:
            reason = 'gradient'
            break
        sel = np.sort(np.concatenate((sel, add)))
    else:
        reason = 'reentry_cap'
    rel = float(jnp.linalg.norm(D @ w_full - bv) / max(bnorm, 1e-300))
    truncated = bool(len(sel) < target and reason not in ('gradient',))
    info = dict(fitter='gpu_fista_select_exact_refine', target_support=target,
                support=int(len(sel)), fista_iterations=int(fista_iters),
                fista_support=fista_support, reentry_rounds=int(rounds),
                max_drop_iterations=int(used), truncated=truncated, stop_reason=reason,
                relative_fit=rel, seconds=time.perf_counter() - t0)
    print(f'    nnls m={info["support"]}/{target} rounds={rounds} drops={used} '
          f'stop={reason} fit={rel:.3e} {info["seconds"]:.1f}s', flush=True)
    return sel.astype(int), np.asarray(w_full)[sel], info


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
        supp, w, info = gpu_nnls(design, b, m, seconds=seconds)
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
