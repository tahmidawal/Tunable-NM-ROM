"""Round-1 solver INTERNALS for the N=256 push (2026-08-23).

PUSH-PLAN.md round 1: solver-termination repair, EQ tail control, IC encoder,
latent warm-start extrapolation.  GOVERNING RULES (PUSH-PLAN non-negotiables):

  * The residual/Jacobian DEFINITIONS, the discretization, and the tolerance
    definitions reported against are NEVER changed here.  Everything in this
    file is either (a) a different iteration schedule over the SAME objective
    (damping, trust-region schedule, restarts, warm starts), (b) offline
    training of an initial-guess encoder on TRAINING data only, or (c) an EQ
    point/weight selection variant whose resulting rule is gated exactly like
    the incumbent's (gate 0 + full EQ diagnostics, reported side by side).
  * Every repaired solver, run with its repairs DISABLED (no restarts, same
    budget), must reproduce the incumbent solver's latent (asserted in the
    drivers via an agreement check before any repaired result is recorded).
  * No test-truth enters any code path in this file's solvers/encoders.

Reason codes are the incumbent ones, unchanged:
  Poisson lm  : 0 budget, 1 rel-dec/step stall, 2 TAU reached, 3 lambda_max,
                5 nan_at_init            (ctol_tol.lm_tau_poisson)
  Burgers step: 0 budget, 1 tol, 2 stalled, 3 lambda_max, 4 tol_at_init,
                5 nan_at_init            (blat_common.lm_step_jit)
A restart consumes attempts from the SAME budget; 'reason' is the reason at
final termination, and the restart count is returned alongside so censoring
can be reported against the identical tolerance rule.
"""
from __future__ import annotations

import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

F64 = jnp.float64


# ===========================================================================
# 1. Poisson: cached-bank weak LM with argument-passed EQ tables + restarts
# ===========================================================================
def lm_tau_cached_restart(head_fn, K, budget, noise=None):
    """Incumbent damped-LM on the incumbent weak residual
        r(z) = Wl * (PhiT @ (wq * (G_q @ h(z)))) - f_m
    (identical formula to ctol_tol.lm_tau_poisson with the cached decoder),
    with the EQ tables (G_q, wq, PhiT, Wl) passed as ARGUMENTS so one compile
    serves any EQ set of the same shape, and ONE repair: on a rel-dec/step
    stall or lambda saturation, if restarts remain, resume from the best
    iterate plus a fixed precomputed perturbation (noise (n_restarts, K)),
    with lam reset to 1e-6.  The tau tolerance is tau * ||r(z0)|| at the
    ORIGINAL z0 -- the incumbent definition, untouched.

    With noise=None (0 restarts) this is algorithmically identical to
    ctol_tol.lm_tau_poisson (asserted in the driver).

    lm(z0, G_q, wq, PhiT, Wl, f_m, tau, trust_delta) ->
       (z_best, val_best, v0, n_jac, accepted, attempts, reason, restarts)."""
    n_restarts = 0 if noise is None else int(noise.shape[0])
    noise_j = None if noise is None else jnp.asarray(noise, dtype=F64)

    def r_of(z, G_q, wq, PhiT, Wl, f_m):
        return Wl * (PhiT @ (wq * (G_q @ head_fn(z)))) - f_m

    def lm(z0, G_q, wq, PhiT, Wl, f_m, tau, trust_delta):
        rJ = lambda z: (r_of(z, G_q, wq, PhiT, Wl, f_m),
                        jax.jacfwd(lambda zz: r_of(zz, G_q, wq, PhiT, Wl, f_m))(z))
        rn_fn = lambda z: jnp.linalg.norm(r_of(z, G_q, wq, PhiT, Wl, f_m))
        r0, J0 = rJ(z0)
        v0 = jnp.linalg.norm(r0)
        tol = tau * v0
        init_reason = jnp.where(~jnp.isfinite(v0), jnp.int32(5),
                                jnp.where((tau > 0) & (v0 <= tol), jnp.int32(2),
                                          jnp.int32(0)))
        # z, J, r, val, lam, att, acc, nJ, reason, z_best, val_best, rs
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), init_reason, z0, v0, jnp.int32(0))

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, acc, nJ, _, z_b, v_b, rs = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            within_trust = jnp.linalg.norm(dz) <= trust_delta
            admissible = finite & within_trust
            z_new = z + jnp.where(admissible, dz, 0.0)
            v_new = jnp.where(admissible, rn_fn(z_new), jnp.inf)
            accept = admissible & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new), lambda: (r, J))
            z = jnp.where(accept, z_new, z)
            val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            better = val < v_b
            z_b = jnp.where(better, z, z_b)
            v_b = jnp.where(better, val, v_b)
            reason = jnp.where(accept & (tau > 0) & (val <= tol), jnp.int32(2),
                               jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)),
                                         jnp.int32(1),
                                         jnp.where((~accept) & (lam >= 1e12),
                                                   jnp.int32(3), jnp.int32(0))))
            if n_restarts > 0:
                do_rs = ((reason == 1) | (reason == 3)) & (rs < n_restarts)

                def restart():
                    z_r = z_b + noise_j[jnp.minimum(rs, n_restarts - 1)]
                    r3, J3 = rJ(z_r)
                    return (z_r, J3, r3, jnp.linalg.norm(r3),
                            jnp.asarray(1e-6, F64), jnp.int32(0), rs + 1)

                def keep():
                    return (z, J2, r2, val, lam, reason, rs)

                z, J2, r2, val, lam, reason, rs = jax.lax.cond(do_rs, restart, keep)
                nJ = nJ + do_rs.astype(jnp.int32)
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason, z_b, v_b, rs)

        s = jax.lax.while_loop(cond, body, init)
        z, J, r, val, lam, att, acc, nJ, reason, z_b, v_b, rs = s
        # final reason coding against the SAME tau rule, on the best iterate
        reason = jnp.where((tau > 0) & (v_b <= tol), jnp.int32(2), reason)
        return z_b, v_b, v0, nJ, acc, att, reason, rs

    return jax.jit(lm)


# ===========================================================================
# 2. Burgers: adaptive-trust-region LM step + repaired rollouts
# ===========================================================================
def make_step_adaptive(r_w, K, noise=None, grow=2.0, shrink=0.5):
    """blat_common.lm_step_jit with TWO repairs, on the SAME residual r_w and
    the SAME absolute tolerance rule (tol_abs is passed in unchanged):
      (a) the trust radius is adaptive: delta *= grow on an accepted step
          (capped at dmax), delta *= shrink on a rejected one (floored at
          dmin), instead of the fixed global TR_DELTA clamp;
      (b) on a stall/lambda saturation, if restarts remain, resume from the
          best iterate + noise[rs] with lam and delta reset.
    Reason codes identical to lm_step_jit.  With noise=None, grow=shrink=1 and
    delta0=TR_DELTA this reproduces the incumbent kernel (asserted in-driver).

    step(z0, prev_c, nu, tol_abs, budget, delta0, dmin, dmax) ->
      (z_best, rn_best, n_jac, accepted, reason, attempts, delta_end, restarts)
    """
    n_restarts = 0 if noise is None else int(noise.shape[0])
    noise_j = None if noise is None else jnp.asarray(noise, dtype=F64)
    rn_fn = lambda z, p, nu: jnp.linalg.norm(r_w(z, p, nu))
    rJ = lambda z, p, nu: (r_w(z, p, nu), jax.jacfwd(r_w)(z, p, nu))

    def step(z0, prev_c, nu, tol_abs, budget, delta0, dmin, dmax):
        r0, J0 = rJ(z0, prev_c, nu)
        rn0 = jnp.linalg.norm(r0)
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where(rn0 <= tol_abs, jnp.int32(4), jnp.int32(0)))
        # z, r, J, rn, lam, delta, att, acc, nJ, reason, z_b, rn_b, rs
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), delta0, jnp.int32(0),
                jnp.int32(0), jnp.int32(1), init_reason, z0, rn0, jnp.int32(0))

        def cond(s):
            return (s[9] == 0) & (s[6] < budget)

        def body(s):
            z, r, J, rn, lam, delta, att, acc, nJ, _, z_b, rn_b, rs = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            within_trust = jnp.linalg.norm(dz) <= delta
            tiny = finite & (jnp.linalg.norm(dz)
                             <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
            z_new = z + jnp.where(finite & within_trust, dz, 0.0)
            rn_new = rn_fn(z_new, prev_c, nu)
            accept = finite & within_trust & jnp.isfinite(rn_new) & (rn_new < rn)
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, prev_c, nu),
                                  lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            delta = jnp.where(accept, jnp.minimum(delta * grow, dmax),
                              jnp.maximum(delta * shrink, dmin))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            better = rn < rn_b
            z_b = jnp.where(better, z, z_b)
            rn_b = jnp.where(better, rn, rn_b)
            reason = jnp.where(accept & (rn <= tol_abs), 1,
                               jnp.where((accept & (rel_dec < 1e-12)) | tiny, 2,
                                         jnp.where((~accept) & (lam >= 1e12), 3,
                                                   0))).astype(jnp.int32)
            if n_restarts > 0:
                do_rs = ((reason == 2) | (reason == 3)) & (rs < n_restarts)

                def restart():
                    z_r = z_b + noise_j[jnp.minimum(rs, n_restarts - 1)]
                    r3, J3 = rJ(z_r, prev_c, nu)
                    return (z_r, r3, J3, jnp.linalg.norm(r3),
                            jnp.asarray(1e-6, F64), delta0, jnp.int32(0), rs + 1)

                def keep():
                    return (z, r2, J2, rn, lam, delta, reason, rs)

                z, r2, J2, rn, lam, delta, reason, rs = jax.lax.cond(
                    do_rs, restart, keep)
                nJ = nJ + do_rs.astype(jnp.int32)
            return (z, r2, J2, rn, lam, delta, att + 1, acc, nJ, reason,
                    z_b, rn_b, rs)

        s = jax.lax.while_loop(cond, body, init)
        z, r, J, rn, lam, delta, att, acc, nJ, reason, z_b, rn_b, rs = s
        reason = jnp.where(rn_b <= tol_abs, jnp.int32(1), reason)
        return z_b, rn_b, nJ, acc, reason, att, delta, rs

    return jax.jit(step, static_argnums=(4,))


def make_rollout_v2(step_kind, ops=None, step_ad=None, rn_fn=None, prev_of=None,
                    num_steps=50, extrap=0.0):
    """Repaired latent rollouts over the incumbent per-step tolerance array.

    step_kind='incumbent': uses ops['step_jit'] (blat_common.lm_step_jit,
      untouched) -- combined with extrap>0 this is the pure warm-start-
      extrapolation arm.
    step_kind='adaptive' : uses step_ad from make_step_adaptive.

    Warm-start extrapolation (extrap=1.0 -> 2-step linear): the step's initial
    guess is z_n + extrap*(z_n - z_{n-1}), SAFEGUARDED by one residual
    evaluation each at z_n and the extrapolated point -- the cheaper start
    wins.  Those two evaluations are inside the timed region (real cost of
    the warm start).  The tolerance passed to the kernel is the incumbent
    us-array, unchanged.

    roll(z0, nu, us, budget, delta0, dmin, dmax) ->
      (Z, rns, nJs, reasons, restarts_per_step)"""
    if step_kind == "incumbent":
        step_fn = ops["step_jit"]
        rn_f = ops["rn"]
        prev_f = ops["prev_of"]
    else:
        step_fn = step_ad
        rn_f = rn_fn
        prev_f = prev_of

    def roll(z0, nu, us, budget, delta0, dmin, dmax):
        def body(carry, tol_abs):
            z, z_prev, prev_c, delta = carry
            if extrap > 0.0:
                z_ex = z + extrap * (z - z_prev)
                rn_a = rn_f(z, prev_c, nu)
                rn_b = rn_f(z_ex, prev_c, nu)
                z_init = jnp.where(jnp.isfinite(rn_b) & (rn_b < rn_a), z_ex, z)
            else:
                z_init = z
            if step_kind == "incumbent":
                z2, rn, nJ, acc, reason, att = step_fn(z_init, prev_c, nu,
                                                       tol_abs, budget)
                delta2 = delta
                rs = jnp.int32(0)
            else:
                z2, rn, nJ, acc, reason, att, delta2, rs = step_fn(
                    z_init, prev_c, nu, tol_abs, budget, delta, dmin, dmax)
            return (z2, z, prev_f(z2), delta2), (z2, rn, nJ, reason, rs)

        (_, _, _, _), out = jax.lax.scan(
            body, (z0, z0, prev_f(z0), delta0), us)
        return out

    return jax.jit(roll, static_argnums=(3,))


# ===========================================================================
# 3. Offline initial-guess encoders (TRAINING data only)
# ===========================================================================
def fit_code_encoder(key, X_np, Z_np, steps=8000, hidden=128, layers=2,
                     lr=1e-3, tag=""):
    """Small MLP  standardized X -> latent code  trained by Adam on TRAINING
    pairs only (X = observable features of a training snapshot, Z = its
    auto-decoder code).  Used ONLY to produce initial guesses for the online
    solves; it never sees test data.  Returns (params, apply_fn, info) where
    apply_fn(params, x) is jittable and includes the standardization."""
    X = jnp.asarray(np.asarray(X_np), dtype=F64)
    Z = jnp.asarray(np.asarray(Z_np), dtype=F64)
    mu = jnp.mean(X, axis=0)
    sd = jnp.std(X, axis=0) + 1e-8
    sizes = [X.shape[1]] + [hidden] * layers + [Z.shape[1]]
    params = {"mu": mu, "sd": sd, "w": [], "b": []}
    for i in range(len(sizes) - 1):
        key, k1 = jax.random.split(key)
        params["w"].append(jax.random.normal(k1, (sizes[i], sizes[i + 1]),
                                             dtype=F64)
                           * jnp.sqrt(2.0 / sizes[i]))
        params["b"].append(jnp.zeros((sizes[i + 1],), dtype=F64))

    def apply_fn(p, x):
        h = (x - p["mu"]) / p["sd"]
        for w, b in zip(p["w"][:-1], p["b"][:-1]):
            h = jax.nn.silu(h @ w + b)
        return h @ p["w"][-1] + p["b"][-1]

    z_ms = jnp.mean(Z * Z)
    loss = lambda p: jnp.mean((apply_fn(p, X) - Z) ** 2) / z_ms
    sched = optax.warmup_cosine_decay_schedule(0.0, lr,
                                               min(200, steps // 10 + 1),
                                               steps, lr * 1e-2)
    opt = optax.adam(sched)

    def masked(p):
        return {"w": p["w"], "b": p["b"]}

    state = opt.init(masked(params))

    @jax.jit
    def step(p, s):
        v, g = jax.value_and_grad(lambda q: loss({**p, **q}))(masked(p))
        upd, s = opt.update(g, s)
        q = optax.apply_updates(masked(p), upd)
        return {**p, **q}, s, v

    t0 = time.time()
    v = jnp.inf
    for i in range(steps):
        params, state, v = step(params, state)
    info = dict(final_rel_mse=float(v), steps=steps, hidden=hidden,
                layers=layers, seconds=time.time() - t0,
                n_pairs=int(X.shape[0]), n_features=int(X.shape[1]))
    print(f"  encoder[{tag}]: rel-MSE {float(v):.3e} on {info['n_pairs']} "
          f"training pairs [{info['seconds']:.0f}s]", flush=True)
    return params, apply_fn, info


# ===========================================================================
# 4. Batched representation-oracle LM (diagnostic ONLY -- uses truth)
# ===========================================================================
def make_oracle_lm(dec_full_fn, K, budget=200):
    """Damped LM on the full-field data misfit ||dec_full(z) - target||,
    vmapped over (z0, target) pairs.  Same damping schedule as the incumbent
    lm solvers.  DIAGNOSTIC ONLY: targets are truth fields; results feed the
    representation-oracle rung of the error ladder, never a solve path.
    solve(z0s (B,K), targets (B,n)) -> (z (B,K), rn (B,))."""
    def f(z, t):
        return dec_full_fn(z) - t

    def one(z0, t):
        rJ = lambda z: (f(z, t), jax.jacfwd(lambda zz: f(zz, t))(z))
        rn_fn = lambda z: jnp.linalg.norm(f(z, t))
        r0, J0 = rJ(z0)
        v0 = jnp.linalg.norm(r0)
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(0))

        def cond(s):
            return (s[6] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            v_new = jnp.where(finite, rn_fn(z_new), jnp.inf)
            accept = finite & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300),
                                1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new), lambda: (r, J))
            z = jnp.where(accept, z_new, z)
            val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            done = jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)),
                             jnp.int32(1),
                             jnp.where((~accept) & (lam >= 1e12), jnp.int32(1),
                                       jnp.int32(0)))
            return (z, J2, r2, val, lam, att + 1, done)

        z, J, r, val, lam, att, done = jax.lax.while_loop(cond, body, init)
        return z, val

    return jax.jit(jax.vmap(one))


def oracle_multi_init(oracle_lm, z0_sets, targets):
    """Run the batched oracle from several init sets; per-target best.
    z0_sets: list of (B,K) arrays; targets (B,n).  Returns (z (B,K), rn (B,))."""
    best_z, best_v = None, None
    for z0s in z0_sets:
        z, v = oracle_lm(jnp.asarray(z0s, dtype=F64), targets)
        if best_z is None:
            best_z, best_v = z, v
        else:
            better = v < best_v
            best_z = jnp.where(better[:, None], z, best_z)
            best_v = jnp.where(better, v, best_v)
    return best_z, best_v


# ===========================================================================
# 5. EQ tail control + per-query adaptive re-fit
# ===========================================================================
def _nnls_rows(Gn, bn, m, nnls_capped, rng, eq_rows, pad_score):
    """Reference NNLS sequence on PRE-normalized (possibly reweighted) rows:
    capped Lawson-Hanson on an eq_rows subsample -> support padding -> final
    nonnegative refit on ALL rows.  Mirrors ctol_eq._solve_nnls WITHOUT the
    internal row normalization (so deliberate row weights survive).  When
    eq_rows covers all rows, no subsample (and no rng draw) happens at all --
    that keeps the all-rows path bit-identical to blat_common.fit_eq_weights'
    sequence."""
    n_c = Gn.shape[1]
    if eq_rows >= Gn.shape[0]:
        rows = np.arange(Gn.shape[0])
    else:
        rows = rng.choice(Gn.shape[0], size=eq_rows, replace=False)
    wts, _, _ = nnls_capped(Gn[rows], bn[rows], max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]
        padded = 0
    else:
        rest = np.setdiff1d(np.arange(n_c), supp)
        score = np.abs(Gn).mean(0) if pad_score is None else pad_score
        pad = rest[np.argsort(-score[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad])
        padded = len(pad)
    wq, _, _ = nnls_capped(Gn[:, keep], bn, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    return keep, wq, padded


def eq_diag(Gn, bn, keep, wq):
    res = Gn[:, keep] @ wq - bn
    rel = np.abs(res) / (np.abs(bn) + 1e-300)
    return dict(rel_fit=float(np.linalg.norm(res) / (np.linalg.norm(bn) + 1e-300)),
                row_rel_median=float(np.median(rel)),
                row_rel_p95=float(np.quantile(rel, 0.95)),
                row_rel_max=float(np.max(rel)))


def tail_reweight_fit(G, b, m, nnls_capped, seed, cap=3e-2, rounds=3,
                      eq_rows=3072, pad_score=None, label=""):
    """EQ tail control (PUSH-PLAN round-1 lever 2): starting from the
    reference row-normalized system, iteratively UP-WEIGHT rows whose relative
    fit error exceeds `cap` (weight *= clip(rel/cap, 1, 10)) and refit
    support+weights with the reference NNLS sequence.  Row weights would be
    invisible to ctol_eq._solve_nnls (it re-normalizes rows), hence the local
    _nnls_rows.  Diagnostics are always computed on the UNWEIGHTED normalized
    rows, so they are directly comparable with the control fit's.

    Returns (keep, wq, info) with per-round diagnostics in info."""
    t0 = time.time()
    sc = np.linalg.norm(G, axis=1) + 1e-300
    Gn = G / sc[:, None]
    bn = b / sc
    wgt = np.ones(Gn.shape[0])
    rng = np.random.default_rng(seed)
    hist = []
    keep = wq = None
    for rd in range(rounds):
        keep, wq, padded = _nnls_rows(Gn * wgt[:, None], bn * wgt, m,
                                      nnls_capped, rng, eq_rows, pad_score)
        d = eq_diag(Gn, bn, keep, wq)
        d["round"] = rd
        d["n_upweighted"] = int(np.sum(wgt > 1.0))
        d["max_weight"] = float(np.max(wgt))
        hist.append(d)
        res = Gn[:, keep] @ wq - bn
        rel = np.abs(res) / (np.abs(bn) + 1e-300)
        wgt = wgt * np.clip(rel / cap, 1.0, 10.0)
    info = dict(kind="tail_reweight", cap=cap, rounds=rounds, m=int(len(keep)),
                secs=time.time() - t0, per_round=hist, **hist[-1])
    print(f"  EQ-tail {label}: rel fit {info['rel_fit']:.2e} "
          f"(p95 {info['row_rel_p95']:.1e}, max {info['row_rel_max']:.1e}) "
          f"after {rounds} rounds [{info['secs']:.0f}s]", flush=True)
    return keep, wq, info


def adq_extend_fit(G, b, new_G_rows, new_b_rows, m, nnls_capped, seed,
                   eq_rows=3072, pad_score=None):
    """Per-query adaptive quadrature (PUSH-PLAN round-1 lever 2): extend the
    training-snapshot EQ system with rows built AT THE QUERY'S CHEAP SOLUTION
    (decoder output at z*, no truth anywhere) and refit points+weights with
    the reference sequence (fresh rng from `seed`).  Returns (keep, wq,
    diagnostics-on-extended-system)."""
    Ge = np.concatenate([G, new_G_rows], axis=0)
    be = np.concatenate([b, new_b_rows])
    sc = np.linalg.norm(Ge, axis=1) + 1e-300
    Gn = Ge / sc[:, None]
    bn = be / sc
    rng = np.random.default_rng(seed)
    keep, wq, padded = _nnls_rows(Gn, bn, m, nnls_capped, rng, eq_rows,
                                  pad_score)
    d = eq_diag(Gn, bn, keep, wq)
    d["n_new_rows"] = int(new_G_rows.shape[0])
    return keep, wq, d
