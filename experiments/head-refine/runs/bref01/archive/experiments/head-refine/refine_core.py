"""Per-query refinement of the frozen head's own weights, anchored to theta_0.

Everything except the refinement is the head-ablation arm (a) contract: the frozen
bank G, the latent dimension K, the weak objective, the test-mode family, the time
discretization, the initializer policy, the Levenberg-Marquardt stopping rule and
the output contract. The only online knob is the number n of gradient steps taken
on theta at query time.

theta is EXACTLY the head's own weights -- the MLP_h weight/bias pairs and the
linear skip W_lin -- flattened into one vector. The spatial track (B, g, out_scale)
and therefore the bank G, the empirical-quadrature rule and the cold-start candidate
table are never touched.

Two variants:

  V1 INITIAL-ONLY   refine once against the supplied initial field, on the same
                    sampled node set the initializer uses, then evolve with the
                    refined head and the unchanged solver.
  V2 PER-STEP       after each time step's LM converges on z, take n steps on theta
                    against that step's weak residual, anchored, then re-solve z.

At n = 0 the refinement block is omitted at the Python level, so no extra LM
re-solve can perturb a converged point and the query is arm (a) with theta as a
traced runtime operand instead of a compile-time constant.
"""
from __future__ import annotations

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import engines as e
import arms as A


# ------------------------------------------------------------------ theta ----

def theta_parts(params):
    """The head's own trainable weights. `hB` (latent Fourier features) is not
    present in either retained checkpoint and would change the head's function
    class, so it is rejected rather than silently frozen or refined."""
    assert 'hB' not in params, 'latent Fourier features are out of contract here'
    return dict(h=params['h'], h_lin=params['h_lin'])


def flatten_theta(params):
    """theta_0 as one flat f64 vector, with its exact inverse."""
    flat, unravel = ravel_pytree(theta_parts(params))
    return flat, unravel


def make_head_of(unravel):
    """head_of(theta, z) is `sep_common.head` verbatim for a checkpoint without
    `hB`: the same two terms in the same order, so at theta = theta_0 it is
    bit-identical to the retained head."""
    def head_of(theta, z):
        d = unravel(theta)
        return A.sc.apply_mlp(d['h'], z) + z @ d['h_lin']
    return head_of


# ------------------------------------------------------------------- Adam ----

ADAM = dict(b1=0.9, b2=0.999, eps=1e-8)


def adam_init(theta):
    return (jnp.zeros_like(theta), jnp.zeros_like(theta), jnp.zeros((), jnp.float64))


def adam_update(g, state, alpha, b1=ADAM['b1'], b2=ADAM['b2'], eps=ADAM['eps']):
    """The standard Adam step with bias correction. Moments start at zero at the
    beginning of every query, so no state survives between queries."""
    m, v, t = state
    t = t + 1.
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return alpha * mh / (jnp.sqrt(vh) + eps), (m, v, t)


# ------------------------------------------------------- Burgers 2D queries ---

def make_query(head_of, K, L, dt, trust, quadrature, variant, n,
               ic_budget=400, step_budget=180, gtol=1e-6, linear='gj'):
    """The complete matched query: supplied dense field -> six dense fields.

    variant: 'baseline' (n must be 0), 'v1' or 'v2'.
    mu and alpha are RUNTIME operands, so one compiled query serves both anchor
    weights and the calibration sweep.
    """
    assert variant in ('baseline', 'v1', 'v2')
    assert (n == 0) == (variant == 'baseline')
    wk = A.weak_fn(quadrature)
    nsub = int(round(.05 / dt))
    nblock = 5
    nsteps = nsub * nblock
    assert nsteps == int(round(.25 / dt))

    ic = A.make_stationary_lm(lambda z, y, Rm, th: Rm @ head_of(th, z) - y,
                              ic_budget, gtol=gtol, linear=linear)
    lm = A.make_stationary_lm(
        lambda z, p, nu, data, th: wk(z, p, nu, data, lambda zz: head_of(th, zz), L, dt),
        step_budget, trust, gtol, linear)

    def start(u0, cold, theta0):
        xy, w, Q, Rm, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(Zcand[idx], (y, Rm, theta0), 0.)
        return z, ui, y, Rm, (icrn, icit, icreason, icgn)

    def anchor_of(th, theta0, tn2):
        d = th - theta0
        return jnp.sum(d * d) / tn2

    def evolve_fixed(z, theta, nu, data, scale):
        """The arm (a) evolution, theta held fixed for the whole trajectory."""
        head = lambda zz: head_of(theta, zz)

        def step(carry, _):
            zc, zprev = carry
            p = data['A'] @ head(zc)
            ze = zc + (zc - zprev)
            r0 = jnp.linalg.norm(wk(zc, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(wk(ze, p, nu, data, head, L, dt))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, zc)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data, theta), 1e-9 * scale)
            return (z2, zc), (z2, rn, it, reason, gn)

        _, (zs, rn, it, reason, gn) = jax.lax.scan(step, (z, z), None, length=nsteps)
        return zs, rn, it, reason, gn

    # ------------------------------------------------------------ baseline ---
    if variant == 'baseline':
        def query(u0, nu, data, cold, theta0, mu, alpha):
            z, ui, y, Rm, icrec = start(u0, cold, theta0)
            scale = jnp.linalg.norm(ui) * jnp.sqrt(len(ui))
            zs, rn, it, reason, gn = evolve_fixed(z, theta0, nu, data, scale)
            internal = jnp.concatenate((z[None], zs))
            Z = internal[::nsub]
            head = lambda zz: head_of(theta0, zz)
            fields = jax.vmap(lambda zz: e.output_field(data['G'] @ head(zz), L, L))(Z)
            return dict(fields=fields, latents=Z, internal=internal,
                        iterations=it, residuals=rn, reasons=reason, stationarity=gn,
                        resolve_iterations=jnp.zeros(0, jnp.int32),
                        resolve_reasons=jnp.zeros(0, jnp.int32),
                        resolve_stationarity=jnp.zeros(0), resolve_residuals=jnp.zeros(0),
                        ic_iterations=icrec[1], ic_reason=icrec[2], ic_stationarity=icrec[3],
                        ic_residual=icrec[0], ic_input_norm=jnp.linalg.norm(ui),
                        refine_iterations=jnp.zeros(0, jnp.int32),
                        refine_reasons=jnp.zeros(0, jnp.int32),
                        refine_stationarity=jnp.zeros(0), refine_residuals=jnp.zeros(0),
                        refine_data_term=jnp.zeros(0), refine_anchor_term=jnp.zeros(0),
                        refine_data_grad=jnp.zeros(0), refine_anchor_grad=jnp.zeros(0),
                        drift=jnp.zeros(1), theta=theta0[None, :])
        return jax.jit(query)

    # ------------------------------------------------------------------ V1 ---
    if variant == 'v1':
        def query(u0, nu, data, cold, theta0, mu, alpha):
            z, ui, y, Rm, icrec = start(u0, cold, theta0)
            scale = jnp.linalg.norm(ui) * jnp.sqrt(len(ui))
            fit2 = jnp.sum(ui * ui)
            tn2 = jnp.sum(theta0 * theta0)

            def data_term(th, zz):
                r = Rm @ head_of(th, zz) - y
                return jnp.sum(r * r) / fit2

            def body(carry, _):
                th, st, zz = carry
                gd = jax.grad(data_term)(th, zz)
                ga = jax.grad(anchor_of)(th, theta0, tn2)
                upd, st = adam_update(gd + mu * ga, st, alpha)
                th = th - upd
                zz, rn, it, reason, gn = ic(zz, (y, Rm, th), 0.)
                return (th, st, zz), (rn, it, reason, gn, data_term(th, zz),
                                      anchor_of(th, theta0, tn2),
                                      jnp.linalg.norm(gd), jnp.linalg.norm(ga))

            (theta, _, z), tr = jax.lax.scan(body, (theta0, adam_init(theta0), z), None, length=n)
            zs, rn, it, reason, gn = evolve_fixed(z, theta, nu, data, scale)
            internal = jnp.concatenate((z[None], zs))
            Z = internal[::nsub]
            head = lambda zz: head_of(theta, zz)
            fields = jax.vmap(lambda zz: e.output_field(data['G'] @ head(zz), L, L))(Z)
            return dict(fields=fields, latents=Z, internal=internal,
                        iterations=it, residuals=rn, reasons=reason, stationarity=gn,
                        resolve_iterations=jnp.zeros(0, jnp.int32),
                        resolve_reasons=jnp.zeros(0, jnp.int32),
                        resolve_stationarity=jnp.zeros(0), resolve_residuals=jnp.zeros(0),
                        ic_iterations=icrec[1], ic_reason=icrec[2], ic_stationarity=icrec[3],
                        ic_residual=icrec[0], ic_input_norm=jnp.linalg.norm(ui),
                        refine_residuals=tr[0], refine_iterations=tr[1], refine_reasons=tr[2],
                        refine_stationarity=tr[3], refine_data_term=tr[4],
                        refine_anchor_term=tr[5], refine_data_grad=tr[6],
                        refine_anchor_grad=tr[7],
                        drift=jnp.sqrt(anchor_of(theta, theta0, tn2))[None],
                        theta=theta[None, :])
        return jax.jit(query)

    # ------------------------------------------------------------------ V2 ---
    def query(u0, nu, data, cold, theta0, mu, alpha):
        z, ui, y, Rm, icrec = start(u0, cold, theta0)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(ui))
        tn2 = jnp.sum(theta0 * theta0)
        p0 = data['A'] @ head_of(theta0, z)

        def step(carry, _):
            zc, zprev, th, st, p = carry
            head = lambda zz: head_of(th, zz)
            ze = zc + (zc - zprev)
            r0 = jnp.linalg.norm(wk(zc, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(wk(ze, p, nu, data, head, L, dt))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, zc)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data, th), 1e-9 * scale)
            pn2 = jnp.sum(p * p)

            def data_term(t2):
                r = wk(z2, p, nu, data, lambda zz: head_of(t2, zz), L, dt)
                return jnp.sum(r * r) / pn2

            def inner(c, _):
                t2, s2 = c
                gd = jax.grad(data_term)(t2)
                ga = jax.grad(anchor_of)(t2, theta0, tn2)
                upd, s2 = adam_update(gd + mu * ga, s2, alpha)
                return (t2 - upd, s2), (jnp.linalg.norm(gd), jnp.linalg.norm(ga))

            (th2, st2), gtr = jax.lax.scan(inner, (th, st), None, length=n)
            z3, rn2, it2, reason2, gn2 = lm(z2, (p, nu, data, th2), 1e-9 * scale)
            pnew = data['A'] @ head_of(th2, z3)
            rec = (z3, rn, it, reason, gn, rn2, it2, reason2, gn2,
                   data_term(th2), anchor_of(th2, theta0, tn2), gtr[0][0], gtr[1][0])
            return (z3, zc, th2, st2, pnew), rec

        def block(carry, _):
            carry, rec = jax.lax.scan(step, carry, None, length=nsub)
            return carry, (carry[2], rec)

        _, (thb, rec) = jax.lax.scan(block, (z, z, theta0, adam_init(theta0), p0),
                                     None, length=nblock)
        flat = [x.reshape((nsteps,) + x.shape[2:]) for x in rec]
        internal = jnp.concatenate((z[None], flat[0]))
        thetas = jnp.concatenate((theta0[None, :], thb))
        Z = internal[::nsub]
        fields = jax.vmap(lambda zz, th: e.output_field(data['G'] @ head_of(th, zz), L, L))(
            Z, thetas)
        drift = jax.vmap(lambda th: jnp.sqrt(anchor_of(th, theta0, tn2)))(thetas)
        return dict(fields=fields, latents=Z, internal=internal,
                    iterations=flat[2], residuals=flat[1], reasons=flat[3], stationarity=flat[4],
                    resolve_residuals=flat[5], resolve_iterations=flat[6],
                    resolve_reasons=flat[7], resolve_stationarity=flat[8],
                    ic_iterations=icrec[1], ic_reason=icrec[2], ic_stationarity=icrec[3],
                    ic_residual=icrec[0], ic_input_norm=jnp.linalg.norm(ui),
                    refine_residuals=jnp.zeros(0), refine_iterations=jnp.zeros(0, jnp.int32),
                    refine_reasons=jnp.zeros(0, jnp.int32), refine_stationarity=jnp.zeros(0),
                    refine_data_term=flat[9], refine_anchor_term=flat[10],
                    refine_data_grad=flat[11], refine_anchor_grad=flat[12],
                    drift=drift, theta=thetas)
    return jax.jit(query)


def make_reconstruction(head_of, K, budget=400, gtol=1e-6, linear='gj'):
    """Best-found reconstruction of a supplied field on the REFINED manifold.
    Offline diagnostic; never inside a timed query."""
    lm = A.make_stationary_lm(lambda z, target, G, th: G @ head_of(th, z) - target,
                              budget, gtol=gtol, linear=linear)

    def fit(starts, target, G, theta):
        out = jax.vmap(lambda z0: lm(z0, (target, G, theta), 0.))(starts)
        best = jnp.argmin(out[1])
        return out[0][best], out[1][best], out[2][best], out[3][best]
    return jax.jit(fit)


# ------------------------------------------------------- Poisson 2D query ----

def make_poisson_query(head_of, n, intervals, trust, budget, gtol, linear):
    """One matched complete query: host source in, dense nodal field out.

    The Poisson weak residual is exactly B h(z) - f_m with no time stepping and no
    quadrature approximation, so the solve IS the fit: V2 has no per-step structure
    to exploit and collapses onto V1. Only V1 is defined here. At n = 0 this is the
    pabl01 arm (a) query with theta as a traced operand.
    """
    lm = A.make_stationary_lm(lambda z, fm, Bm, th: Bm @ head_of(th, z) - fm,
                              budget, trust, gtol, linear)

    def kernel(source, S, I, J, W, B, bank, predictions, codes, theta0, mu, alpha):
        # The incumbent skinny sine-product projection, charged inside the query.
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        index = jnp.argmin(jnp.sum((predictions - fm[None, :]) ** 2, axis=1))
        z, rn, it, reason, gn = lm(codes[index], (fm, B, theta0), 0.)
        fmn2 = jnp.sum(fm * fm)
        tn2 = jnp.sum(theta0 * theta0)

        def anchor(th):
            d = th - theta0
            return jnp.sum(d * d) / tn2

        def data_term(th, zz):
            r = B @ head_of(th, zz) - fm
            return jnp.sum(r * r) / fmn2

        if n == 0:
            theta = theta0
            tr = (jnp.zeros(0), jnp.zeros(0, jnp.int32), jnp.zeros(0, jnp.int32),
                  jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
            rn2, it2, reason2, gn2 = rn, it, reason, gn
        else:
            def body(carry, _):
                th, st, zz = carry
                gd = jax.grad(data_term)(th, zz)
                ga = jax.grad(anchor)(th)
                upd, st = adam_update(gd + mu * ga, st, alpha)
                th = th - upd
                zz, r2, i2, s2, g2 = lm(zz, (fm, B, th), 0.)
                return (th, st, zz), (r2, i2, s2, g2, data_term(th, zz), anchor(th),
                                      jnp.linalg.norm(gd), jnp.linalg.norm(ga))
            (theta, _, z), tr = jax.lax.scan(body, (theta0, adam_init(theta0), z),
                                             None, length=n)
            rn2, it2, reason2, gn2 = tr[0][-1], tr[1][-1], tr[2][-1], tr[3][-1]
        field = jnp.pad((bank @ head_of(theta, z)).reshape(intervals - 1, intervals - 1), 1)
        return dict(field=field, latent=z,
                    residual=rn2, iterations=it2, reason=reason2, stationarity=gn2,
                    first_residual=rn, first_iterations=it, first_reason=reason,
                    first_stationarity=gn,
                    index=index, source_norm=jnp.linalg.norm(fm),
                    refine_residuals=tr[0], refine_iterations=tr[1], refine_reasons=tr[2],
                    refine_stationarity=tr[3], refine_data_term=tr[4],
                    refine_anchor_term=tr[5], refine_data_grad=tr[6], refine_anchor_grad=tr[7],
                    drift=jnp.sqrt(anchor(theta)), theta=theta)
    return jax.jit(kernel)
