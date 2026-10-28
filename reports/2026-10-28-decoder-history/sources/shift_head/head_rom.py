"""The paper's nonlinear head and nested corrections inside the co-moving frame.

Trial state in the moving frame:  v = G (h(z) + C_q y),  u(x, t) = v(x - c(t), t).

  G      fixed centered POD bank (3 n^3, R), the lane's bank, unchanged;
  h      the paper's head: two hidden SiLU layers plus a linear skip W_s^T z,
         R outputs (bank coefficients), trained as an auto-decoder on the centered
         bank coefficients of TRAINING snapshots only (one stored code per state);
  C_q    the leading q principal directions of the head's training misses
         a_i - h(z_i*) in coefficient space (weighted PCA; nested in q);
  delta  the frame increment, solved online exactly as in shift_rom.

Unknowns per step: (z, y, delta) in R^{k + q + 3}. The residual is the lane's
co-moving weak residual evaluated at a = h(z) + C_q y; its Jacobian is the lane's
analytic (a, delta) Jacobian chained with [dh/dz, C_q]. The solver is the lane's
fixed-sweep damped Gauss-Newton (Marquardt diagonal, Cholesky), warm started by
extrapolation. Nothing grid-sized is touched inside the time loop.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

import shift_rom as SR


# ---------------------------------------------------------------------- head --

def head_apply(p, z):
    """h(z) = W3 silu(W2 silu(W1 z + b1) + b2) + b3 + Ws^T z.  z (..., k) -> (..., R)."""
    x = jax.nn.silu(z @ p["W1"] + p["b1"])
    x = jax.nn.silu(x @ p["W2"] + p["b2"])
    return x @ p["W3"] + p["b3"] + z @ p["Ws"]


def head_jacobian(p, z):
    """dh/dz at one z, (R, k), analytic."""
    pre1 = z @ p["W1"] + p["b1"]
    x1 = jax.nn.silu(pre1)
    pre2 = x1 @ p["W2"] + p["b2"]
    ds1 = jax.nn.sigmoid(pre1) * (1 + pre1 * (1 - jax.nn.sigmoid(pre1)))
    ds2 = jax.nn.sigmoid(pre2) * (1 + pre2 * (1 - jax.nn.sigmoid(pre2)))
    # d x2 / d z = diag(ds2) W2^T diag(ds1) W1^T
    inner = (p["W1"] * ds1[None, :]) @ (p["W2"] * ds2[None, :])     # (k, width)
    return (inner @ p["W3"] + p["Ws"]).T


def init_head(key, coeffs, weights, k, width):
    """Weighted-PCA affine initialisation, nonlinear part initially zero.

    coeffs (N, R), weights (N,). Returns params and initial codes (N, k) with the
    head reproducing the rank-k weighted PCA reconstruction exactly at init."""
    coeffs = jnp.asarray(coeffs)
    w = jnp.asarray(weights)
    centre = jnp.sum(w[:, None] * coeffs, 0) / jnp.sum(w)
    centred = coeffs - centre
    _, vecs = jnp.linalg.eigh(centred.T @ (w[:, None] * centred))
    basis = vecs[:, ::-1][:, :k]                                      # (R, k)
    scores = centred @ basis
    scale = jnp.sqrt(jnp.mean(scores * scores)) / 0.25
    codes = scores / scale
    R = coeffs.shape[1]
    k1, k2 = jax.random.split(key)
    p = dict(W1=jax.random.normal(k1, (k, width)) * jnp.sqrt(2.0 / k), b1=jnp.zeros(width),
             W2=jax.random.normal(k2, (width, width)) * jnp.sqrt(2.0 / width),
             b2=jnp.zeros(width), W3=jnp.zeros((width, R)), b3=centre,
             Ws=(basis * scale).T)
    got = head_apply(p, codes)
    want = centre + scores @ basis.T
    parity = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
    if parity > 1e-10:
        raise RuntimeError(f"head initialisation parity {parity}")
    return p, codes, dict(init_parity=parity, code_scale=float(scale))


def train_head(p, codes, coeffs, weights, steps, lr, floor_lr, code_reg, log=print,
               log_every=5000):
    """Full-batch Adam on (theta, codes): mean_i w_i ||h(z_i) - a_i||^2 + reg ||z||^2."""
    coeffs = jnp.asarray(coeffs)
    w = jnp.asarray(weights)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(1000, steps // 20 + 1), steps,
                                               floor_lr)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    pair = (p, jnp.asarray(codes))
    state = opt.init(pair)

    def loss_fn(pair, coeffs, w):
        theta, z = pair
        miss = head_apply(theta, z) - coeffs
        fit = jnp.mean(w * jnp.sum(miss * miss, axis=1))
        return fit + code_reg * jnp.mean(jnp.sum(z * z, axis=1)), fit

    @jax.jit
    def chunk(pair, state, coeffs, w):
        def body(carry, _):
            pair, state = carry
            (value, fit), grad = jax.value_and_grad(loss_fn, has_aux=True)(pair, coeffs, w)
            upd, state = opt.update(grad, state, pair)
            return (optax.apply_updates(pair, upd), state), fit
        (pair, state), fits = jax.lax.scan(body, (pair, state), None, length=log_every)
        return pair, state, fits[-1]

    curve = []
    start = time.time()
    done = 0
    while done < steps:
        pair, state, fit = chunk(pair, state, coeffs, w)
        done += log_every
        curve.append((done, float(fit)))
        log(f"  head step {done}/{steps} weighted fit {float(fit):.4e} "
            f"({time.time() - start:.0f}s)")
    return pair[0], np.asarray(pair[1]), curve


def make_code_fit(k, iters, damping=1e-8):
    """Fixed-sweep Gauss-Newton for min_z ||P (h(z) - a)||, P = I - C C^T.

    Variable projection: the correction amplitudes y = C^T (a - h(z)) are
    eliminated exactly, so (z, y) is the least-squares fit of a in the (k + q)
    trial set. C may have zero columns."""

    def fit(p, z0, a, C):
        def proj(v):
            return v - C @ (C.T @ v)

        def body(z, _):
            res = proj(head_apply(p, z) - a)
            J = proj(head_jacobian(p, z))
            H = J.T @ J
            L = jnp.linalg.cholesky(H + jnp.diag(damping * jnp.diagonal(H) + 1e-300))
            return z + jax.scipy.linalg.cho_solve((L, True), -(J.T @ res)), None
        z, _ = jax.lax.scan(body, z0, None, length=iters)
        y = C.T @ (a - head_apply(p, z))
        return z, y
    return fit


def best_codes(p, codes, coeffs, iters=30, chunk=4096):
    """Refit every stored code with the head frozen (best-found codes)."""
    k = codes.shape[1]
    C0 = jnp.zeros((coeffs.shape[1], 0))
    fit = make_code_fit(k, iters)
    vfit = jax.jit(jax.vmap(lambda z, a: fit(p, z, a, C0)[0]))
    out = []
    for s in range(0, len(codes), chunk):
        zc = vfit(jnp.asarray(codes[s:s + chunk]), jnp.asarray(coeffs[s:s + chunk]))
        before = jnp.sum((head_apply(p, jnp.asarray(codes[s:s + chunk]))
                          - coeffs[s:s + chunk]) ** 2, 1)
        after = jnp.sum((head_apply(p, zc) - coeffs[s:s + chunk]) ** 2, 1)
        # keep the stored code wherever the refit did not improve it
        out.append(np.asarray(jnp.where((after <= before)[:, None], zc,
                                        jnp.asarray(codes[s:s + chunk]))))
    return np.concatenate(out)


def correction_directions(p, codes, coeffs, weights):
    """Weighted PCA of the misses a_i - h(z_i). Returns C (R, R), energies."""
    miss = np.asarray(jnp.asarray(coeffs) - head_apply(p, jnp.asarray(codes)))
    w = np.asarray(weights)
    cov = miss.T @ (w[:, None] * miss)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return np.ascontiguousarray(vecs[:, order]), vals[order]


# --------------------------------------------------------------- the query ----

def make_head_step(dt, n, R, k, q, iters, damping, solve_delta=True):
    """(prepare, coeff, residual_jac, step_solve) for one reduced step in (z, y, delta)."""
    _, prepare, residual_and_jacobian = SR.make_frozen_step(dt, n, R, iters, damping)

    def coeff(p, w, C):
        a = head_apply(p, w[:k])
        if q:
            a = a + C @ w[k:k + q]
        return a

    def residual_jac(w, a_prev, ops, p, C):
        """Residual and its Jacobian w.r.t. the solved unknowns (delta columns only
        when the frame is solved)."""
        a = coeff(p, w, C)
        res, jfull = residual_and_jacobian(jnp.concatenate((a, w[k + q:])), a_prev, ops)
        ja = jfull[:, :R]
        blocks = [ja @ head_jacobian(p, w[:k])]
        if q:
            blocks.append(ja @ C)
        if solve_delta:
            blocks.append(jfull[:, R:])
        return res, jnp.concatenate(blocks, axis=1)

    def step_solve(w0, a_prev, ops, p, C):
        def body(w, _):
            res, jac = residual_jac(w, a_prev, ops, p, C)
            hess = jac.T @ jac
            L = jnp.linalg.cholesky(hess + jnp.diag(damping * jnp.diagonal(hess) + 1e-300))
            stepv = jax.scipy.linalg.cho_solve((L, True), -(jac.T @ res))
            if not solve_delta:
                stepv = jnp.concatenate((stepv, jnp.zeros(3)))
            return w + stepv, jnp.linalg.norm(res)
        w, norms = jax.lax.scan(body, w0, None, length=iters, unroll=iters)
        return w, norms[-1]
    return prepare, coeff, residual_jac, step_solve


def make_head_run(dt, nsteps, out_every, n, R, k, q, iters=3, damping=1e-6,
                  ic_iters=12, extrapolate=True, diagnose=False, frame="free"):
    """Complete co-moving query with the head and q corrections.

    run(u0, nu, basis, A, T, lam, Dd, head_params, C, Hcand, Hnorm, Zcand)
      C       (R, q) correction directions (the nested prefix)
      Zcand   (Ncand, k) best-found training codes; Hcand = h(Zcand) (Ncand, R);
      Hnorm   ||Hcand||^2 row norms.  The initial code is the nearest candidate
              in coefficient space, then `ic_iters` variable-projection sweeps.
    frame='zero' freezes delta = 0 (must-fail control)."""
    assert nsteps % out_every == 0
    assert frame in ("free", "zero")
    solve_delta = frame == "free"
    prepare, coeff, _, step_solve = make_head_step(dt, n, R, k, q, iters, damping,
                                                   solve_delta)
    code_fit = make_code_fit(k, ic_iters)

    @jax.jit
    def run(u0, nu, basis, A, T, lam, Dd, p, C, Hcand, Hnorm, Zcand):
        ops = prepare(nu, A, T, lam, Dd)
        c0 = SR.grid_centroid(u0)
        a0 = basis.T @ SR.shift_field(u0, -c0 * n).ravel()
        idx = jnp.argmin(Hnorm - 2 * (Hcand @ a0))
        z0, y0 = code_fit(p, Zcand[idx], a0, C)
        w0 = jnp.concatenate((z0, y0, jnp.zeros(3)))

        def step(carry, _):
            w, previous, c = carry
            guess = w + (w - previous) if extrapolate else w
            if not solve_delta:
                guess = guess.at[k + q:].set(0.0)
            new, rn = step_solve(guess, coeff(p, w, C), ops, p, C)
            centre = (c + new[k + q:]) % 1.0
            return (new, w, centre), ((rn, new[k + q:]) if diagnose else None)

        def block(carry, _):
            carry, info = jax.lax.scan(step, carry, None, length=out_every)
            w, _, c = carry
            field = SR.shift_field((basis @ coeff(p, w, C)).reshape(3, n, n, n), c * n)
            return carry, (field, info)

        frame0 = SR.shift_field((basis @ coeff(p, w0, C)).reshape(3, n, n, n), c0 * n)
        _, (fields, info) = jax.lax.scan(block, (w0, w0, c0), None,
                                         length=nsteps // out_every)
        fields = jnp.concatenate((frame0[None], fields))
        if not diagnose:
            return fields
        return fields, tuple(x.reshape(nsteps, *x.shape[2:]) for x in info)
    return run


def make_head_reference(dt, nsteps, out_every, n, R, k, q, budget=60, gtol=1e-7,
                        ic_iters=12):
    """Reference driver for the parity gate: generic LM (jacfwd, adaptive damping,
    while_loop) on the same (z, y, delta) residual, same initial code."""
    from ns2d_rom import make_lm
    _, prepare, residual_and_jacobian = SR.make_frozen_step(dt, n, R, 1, 1e-6)
    code_fit = make_code_fit(k, ic_iters)

    def coeff(p, w, C):
        a = head_apply(p, w[:k])
        return a + C @ w[k:k + q] if q else a

    def residual(w, a_prev, ops, p, C):
        a = coeff(p, w, C)
        return residual_and_jacobian(jnp.concatenate((a, w[k + q:])), a_prev, ops)[0]

    lm = make_lm(residual, budget, gtol=gtol)

    @jax.jit
    def run(u0, nu, basis, A, T, lam, Dd, p, C, Hcand, Hnorm, Zcand):
        ops = prepare(nu, A, T, lam, Dd)
        c0 = SR.grid_centroid(u0)
        a0 = basis.T @ SR.shift_field(u0, -c0 * n).ravel()
        idx = jnp.argmin(Hnorm - 2 * (Hcand @ a0))
        z0, y0 = code_fit(p, Zcand[idx], a0, C)
        w0 = jnp.concatenate((z0, y0, jnp.zeros(3)))

        def step(carry, _):
            w, c = carry
            new, rn, it, reason, gn = lm(w, (coeff(p, w, C), ops, p, C), 0.0)
            return (new, (c + new[k + q:]) % 1.0), (it, reason)

        def block(carry, _):
            carry, info = jax.lax.scan(step, carry, None, length=out_every)
            w, c = carry
            return carry, (SR.shift_field((basis @ coeff(p, w, C)).reshape(3, n, n, n),
                                          c * n), info)

        frame0 = SR.shift_field((basis @ coeff(p, w0, C)).reshape(3, n, n, n), c0 * n)
        _, (fields, info) = jax.lax.scan(block, (w0, c0), None, length=nsteps // out_every)
        return jnp.concatenate((frame0[None], fields)), info
    return run


# ------------------------------------------------------------------ floors ----

def head_floor_errors(p, Zcand, Hcand, coeffs, outside_sq, u0_sq, C, ic_iters=12):
    """Oracle-shift representation floor of the (k + q) trial set, per frame.

    coeffs (cases, times, R) are G^T of the truth centred on its own centroid,
    outside_sq the squared bank residual ||v - G G^T v||^2 of each frame, u0_sq
    ||u0||^2 per case. The fit is exactly the query's initial encoder (nearest
    candidate + variable projection), so this is the floor the query starts from.
    Returns (cases, times) errors relative to ||u0||."""
    k = Zcand.shape[1]
    fit = make_code_fit(k, ic_iters)
    Hnorm = jnp.sum(Hcand * Hcand, 1)

    @jax.jit
    def one(a):
        idx = jnp.argmin(Hnorm - 2 * (Hcand @ a))
        z, y = fit(p, Zcand[idx], a, C)
        return jnp.sum((head_apply(p, z) + C @ y - a) ** 2)

    cases, times, _ = coeffs.shape
    inside = np.asarray(jax.vmap(one)(jnp.asarray(coeffs.reshape(-1, coeffs.shape[-1]))))
    inside = inside.reshape(cases, times)
    return np.sqrt((inside + outside_sq) / u0_sq[:, None])
