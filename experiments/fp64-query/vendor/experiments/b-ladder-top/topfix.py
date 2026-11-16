"""Three isolated fixes for the top of the Burgers correction ladder.

The reduced state, directions, weak objective, initializer policy, stopping rule and
output contract are the cheap-corrections contract verbatim:

    u(z, y) = G ( h_theta(z) + C_q y ),      w = (z, y) in R^{K+q}.

Only the linear algebra inside the block-damped Levenberg-Marquardt iteration, the
starting iterate, and the offline quadrature budget change. See `DESIGN.md` for the
equations; the short version:

  (a) cascade   the rung is started from the converged q/2 rung's trajectory, padded
                with zeros, chosen against the retained extrapolation by residual norm.
  (b) pre       the augmented Jacobian is column-equilibrated before the normal
                equations are formed. A no-op in exact arithmetic (the retained
                Marquardt scaling already uses diag(J^T J)); a floating-point
                conditioning fix, measured by kappa before and after.
  (c) damp      the y block gets its own Levenberg schedule and its own trust radius,
                instead of a frozen 1e-10 ridge and no radius at all. Because
                G C_q has orthonormal columns, ||dy|| IS the field norm of the
                correction increment, so the y radius is a physical radius.

`base` is `varpro.make_block_lm` itself, so the isolation is exact rather than a
re-implementation that happens to agree. At q = 0 the whole query is
`varpro.make_query`, so every arm is the retained joint path there.
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
import varpro as VP

corrected_head = VP.corrected_head
blocks = VP.blocks
joint_gradient = VP.joint_gradient


# ------------------------------------------------------------- the solvers ----

def make_fixed_lm(res, K, q, budget, trust, gtol=1e-6, equilibrate=False,
                  adaptive_y=False, ridge=1e-10, eps0=1e-6, eps_min=1e-12, eps_max=1e14):
    """The block-damped LM with fixes (b) and/or (c) switched on.

    With `equilibrate=False, adaptive_y=False` this is algebraically
    `varpro.make_block_lm`; `base` arms use that function itself, not this one.
    """
    mask_z = jnp.concatenate((jnp.ones(K), jnp.zeros(q)))
    mask_y = 1. - mask_z

    def normal_step(J, r, lam, eps):
        if equilibrate:
            s = jnp.linalg.norm(J, axis=0)
            s = jnp.maximum(s, 1e-16 * jnp.max(s) + 1e-300)
            Js = J / s
            H = Js.T @ Js
            g = Js.T @ r
            step = jnp.linalg.solve(H + jnp.diag(lam * mask_z + eps * mask_y), -g)
            return step / s
        H = J.T @ J
        d = jnp.diag(H) + 1e-30
        return jnp.linalg.solve(H + jnp.diag(lam * d * mask_z + eps * d * mask_y), -(J.T @ r))

    def lm(z0, y0, args, tol, dy_trust):
        def ev(w):
            r = res(w[:K], w[K:], *args)
            J = jax.jacfwd(lambda ww: res(ww[:K], ww[K:], *args))(w)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        w0 = jnp.concatenate((z0, y0))
        r, J, rn = ev(w0)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, rn, lam, eps, it, reason = s
            step = normal_step(J, r, lam, eps)
            ok = (jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step[:K]) <= trust)
                  & (jnp.linalg.norm(step[K:]) <= dy_trust))
            wn = w + jnp.where(ok, step, 0.)
            rn2 = jnp.linalg.norm(res(wn[:K], wn[K:], *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: ev(wn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            if adaptive_y:
                eps2 = jnp.where(accept, jnp.maximum(eps / 3, eps_min), jnp.minimum(eps * 10, eps_max))
            else:
                eps2 = eps
            return (jnp.where(accept, wn, w), r2, J2, rn2, lam2, eps2, it + 1, reason)

        eps_start = jnp.asarray(eps0 if adaptive_y else ridge)
        w, r, J, rn, lam, eps, it, reason = jax.lax.while_loop(
            lambda s: (s[6] < budget) & (s[7] == 0), body,
            (w0, r, J, rn, jnp.asarray(1e-6), eps_start, jnp.int32(0), reason))
        return w[:K], w[K:], rn, it, reason, grad(r, J)
    return lm


def make_base_lm(res, K, q, budget, trust, gtol=1e-6, ridge=1e-10):
    """The retained block-damped solver itself, adapted to the (.., dy_trust) call."""
    inner = VP.make_block_lm(res, K, q, budget, trust, gtol, ridge)
    return lambda z0, y0, args, tol, dy_trust: inner(z0, y0, args, tol)


ARMS = {
    'base':     dict(equilibrate=False, adaptive_y=False, cascade=False),
    'pre':      dict(equilibrate=True,  adaptive_y=False, cascade=False),
    'damp':     dict(equilibrate=False, adaptive_y=True,  cascade=False),
    'predamp':  dict(equilibrate=True,  adaptive_y=True,  cascade=False),
    'casc':     dict(equilibrate=True,  adaptive_y=True,  cascade=True),
}


# --------------------------------------------------------------- the query ----

def make_query(params, C, K, q, L, dt, trust, quadrature, arm, Rb=None, ic_budget=400,
               step_budget=180, gtol=1e-6, ic_gtol=None, linear='gj', inner_damping=1e-10,
               tau_y=.1, cascade=False):
    """The complete matched query: supplied dense field on GPU -> six dense fields.

    Returns, per invocation, exactly the `varpro.make_query` tuple:
      0 fields (6, L+1, L+1)   1 iterations/step        2 residual norms/step
      3 exit reasons/step      4 output latents         5 ic iterations
      6 ic reason              7 internal latents       8 z-block gradient/step
      9 ic outer gradient     10 ic residual           11 ||u_gauss||
     12 joint normalized gradient/step                 13 ic joint gradient
     14 y-block gradient/step

    When `cascade` is set the query takes one extra argument: the coarse rung's
    internal latents, already padded with zeros to K + q columns.
    """
    ic_gtol = gtol if ic_gtol is None else ic_gtol
    if q == 0:
        assert not cascade
        return VP.make_query(params, C, K, q, L, dt, trust, quadrature, 'joint',
                             ic_budget=ic_budget, step_budget=step_budget, gtol=gtol,
                             linear=linear, inner_iters=1, inner_damping=inner_damping)

    opts = ARMS[arm]
    head = corrected_head(params, C, K)
    hz = VP.head_only(params)
    wk = A.weak_fn(quadrature)
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))

    def res(z, y, prev, nu, data):
        return wk(jnp.concatenate((z, y)), prev, nu, data, head, L, dt)

    if arm == 'base':
        outer = make_base_lm(res, K, q, step_budget, trust, gtol, inner_damping)
    else:
        outer = make_fixed_lm(res, K, q, step_budget, trust, gtol,
                              equilibrate=opts['equilibrate'], adaptive_y=opts['adaptive_y'],
                              ridge=inner_damping)
    ic_lm = A.make_stationary_lm(lambda z, tgt, Rm, P: P(Rm @ hz(z) - tgt),
                                 ic_budget, gtol=ic_gtol, linear='gj')
    adaptive = opts['adaptive_y']
    Rbm = None if Rb is None else jnp.asarray(Rb)

    def body(u0, nu, data, cold, warm):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        yv = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ yv)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))

        RC = R @ C                                    # (R, q), the linear block
        Qr, Rr = jnp.linalg.qr(RC, mode='reduced')
        proj = lambda v: v - Qr @ (Qr.T @ v)          # exact elimination of y
        z0 = Zcand[idx][:K] if warm is None else warm[0][:K]
        z, icrn, icit, icreason, icgn = ic_lm(z0, (yv, R, proj), 0.)
        ycor = jax.scipy.linalg.solve_triangular(Rr, Qr.T @ (yv - R @ hz(z)), lower=False)
        w0 = jnp.concatenate((z, ycor))
        icres = R @ head(w0) - yv
        icJz = jax.jacfwd(lambda zz: R @ head(jnp.concatenate((zz, ycor))) - yv)(z)
        icgj = jnp.linalg.norm(
            jnp.concatenate((icJz.T @ icres, RC.T @ icres))) / (
            jnp.sqrt(jnp.linalg.norm(icJz) ** 2 + jnp.linalg.norm(RC) ** 2)
            * jnp.linalg.norm(icres) + 1e-300)

        def solve_from(wi, wv, p):
            dy = tau_y * jnp.linalg.norm(Rbm @ head(wi)) if adaptive else jnp.inf
            z2, y2, rn, it, reason, _ = outer(wi[:K], wi[K:], (p, nu, data), 1e-9 * scale, dy)
            gzn, nz, gyn, ny, rnorm = blocks(res, z2, y2, (p, nu, data))
            gj = joint_gradient(gzn, nz, gyn, ny, rnorm)
            gz = gzn / (nz * rnorm + 1e-300)
            gy = gyn / (ny * rnorm + 1e-300)
            return (jnp.concatenate((z2, y2)), wv), (
                jnp.concatenate((z2, y2)), rn, it, reason, gz, gj, gy)

        def plain(carry, _):
            # The retained two-way initializer guard, verbatim.
            wv, wprev = carry
            p = data['A'] @ head(wv)
            we = wv + (wv - wprev)
            r0 = jnp.linalg.norm(wk(wv, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(wk(we, p, nu, data, head, L, dt))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            return solve_from(wi, wv, p)

        def cascaded(carry, extra):
            # Three-way guard: the padded coarse iterate can never make the start worse.
            wv, wprev = carry
            p = data['A'] @ head(wv)
            cand = jnp.stack((wv, wv + (wv - wprev), extra))
            rs = jax.vmap(lambda c: jnp.linalg.norm(wk(c, p, nu, data, head, L, dt)))(cand)
            rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
            return solve_from(cand[jnp.argmin(rs)], wv, p)

        if warm is None:
            _, out = jax.lax.scan(plain, (w0, w0), None, length=steps)
        else:
            _, out = jax.lax.scan(cascaded, (w0, w0), warm[1:])
        ws, rn, it, reason, gn, gj, gy = out
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = jax.vmap(lambda v: e.output_field(data['G'] @ head(v), L, L))(W)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn,
                icrn, jnp.linalg.norm(ui), gj, icgj, gy)

    if cascade:
        return jax.jit(lambda u0, nu, data, cold, warm: body(u0, nu, data, cold, warm))
    return jax.jit(lambda u0, nu, data, cold: body(u0, nu, data, cold, None))


# ------------------------------------------------------ conditioning probe ----

def conditioning(params, C, K, q, L, dt, trust, quadrature, data, u0, nu, cold, step_index=25,
                 lams=(0., 1e-6)):
    """kappa of the augmented normal equations, unscaled and column-equilibrated.

    Offline diagnostic at one representative state; never inside a timed query. The
    state is reached by running the retained `base` arm for `step_index` steps and
    taking its iterate, so the probe reports the conditioning the solver actually meets.
    """
    head = corrected_head(params, C, K)
    wk = A.weak_fn(quadrature)

    def res(z, y, prev, nu_, dat):
        return wk(jnp.concatenate((z, y)), prev, nu_, dat, head, L, dt)

    query = make_query(params, C, K, q, L, dt, trust, quadrature, 'base',
                       ic_budget=400, step_budget=180, gtol=1e-6, linear='lu')
    v = query(u0, nu, data, cold)
    internal = np.asarray(v[7])
    t = int(min(step_index, len(internal) - 2))
    wv = jnp.asarray(internal[t])
    p = data['A'] @ head(wv)
    r = res(wv[:K], wv[K:], p, nu, data)
    J = jax.jacfwd(lambda ww: res(ww[:K], ww[K:], p, nu, data))(wv)
    J = np.asarray(J)
    r = np.asarray(r)
    s = np.linalg.norm(J, axis=0)
    floor = 1e-16 * s.max()
    sf = np.maximum(s, floor)
    Js = J / sf
    H = J.T @ J
    Hs = Js.T @ Js
    d = np.diag(H) + 1e-30
    mz = np.concatenate((np.ones(K), np.zeros(q)))
    my = 1. - mz
    out = dict(q=int(q), step_index=t, M=int(J.shape[0]), unknowns=int(J.shape[1]),
               jacobian_condition=float(np.linalg.cond(J)),
               column_norm_min=float(s.min()), column_norm_max=float(s.max()),
               column_norm_ratio=float(s.max() / max(s.min(), 1e-300)),
               y_column_norm_min=(float(s[K:].min()) if q else None),
               y_column_norm_max=(float(s[K:].max()) if q else None),
               residual_norm=float(np.linalg.norm(r)), lambdas={})
    for lam in lams:
        A1 = H + np.diag(lam * d * mz + 1e-10 * d * my)
        A2 = Hs + np.diag(lam * mz + 1e-10 * my)
        out['lambdas'][f'{lam:g}'] = dict(
            unscaled_condition=float(np.linalg.cond(A1)),
            equilibrated_condition=float(np.linalg.cond(A2)),
            improvement_factor=float(np.linalg.cond(A1) / max(np.linalg.cond(A2), 1e-300)))
    return out


# ------------------------------------------- empirical quadrature, unbounded --

def bounded_nnls(G, b, target, seconds, blocks_wanted=20):
    """`varpro.bounded_nnls` with the refit block size expressed as a refit count."""
    return VP.bounded_nnls(G, b, target, seconds, block=max(1, int(target) // int(blocks_wanted)))


def build_operators(bank, L, M, quadrature, head=None, Wcodes=None, m=None, fitter='bounded',
                    eq_seed=20259, candidate_cap=8192, fit_states=64, max_fit_rows=None,
                    eq_seconds=3000., blocks_wanted=20):
    """`varpro.build_operators` with a selectable refit block count.

    Identical arithmetic; only the greedy NNLS block size and the walltime budget move,
    so `fitter='retained'` still reproduces the audited rules bitwise.
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
            supp, w, finfo = VP.retained_nnls(design, b, m)
        else:
            supp, w, finfo = bounded_nnls(design, b, m, eq_seconds, blocks_wanted)
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
                    eq_relative_fit=rel, eq_fit=finfo,
                    eq_rule_valid=bool(not finfo['truncated']))
    info['setup_seconds'] = time.perf_counter() - t0
    info['array_bytes'] = int(sum(x.nbytes for x in jax.tree_util.tree_leaves(data)))
    return data, info


enriched_codes = VP.enriched_codes
