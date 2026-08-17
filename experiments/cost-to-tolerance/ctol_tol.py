"""Tolerance-stopped latent solvers for the cost-to-tolerance surface.

WHY THIS FILE EXISTS
--------------------
The committed cost tables of the two source cells report ITERATIONS TO SOLVER
TERMINATION (relative-decrease / step-size / budget), so different `k` stop at
different accuracy and "cost vs k" silently conflates work with target.  This
cell instead stops on a DEPLOYABLE tolerance:

    ||r(z_j)||  <=  tau * ||r(z_0)||          tau in {1e-1, 1e-2, 1e-3}

i.e. the RELATIVE REDUCTION of the objective that is actually being minimised,
measured from the run's own initial guess.  It needs no oracle and no held-out
field, so it is computable online.

A tolerance on the true discrete residual ||A u - f|| / ||f|| is NOT usable
here: at the weak-form solution that quantity is ~2e-1 while the field error is
~8e-3 (that amplification is the entire reason the weak form exists).  The
achieved ||A u - f|| / ||f|| is reported per cell for reference, never used to
stop.

WHAT IS REUSED
--------------
* Burgers: the reference ROM is used verbatim.  `make_weak_ops` builds the
  operators; the per-step kernel is `ops["step_jit"]` (blat_common's
  `lm_step_jit`, a lax.while_loop LM).  This file only supplies the per-step
  ABSOLUTE tolerance `tau * ||r(z_n)||` -- computed inside the scan from the
  step's own warm start -- and drives the scan.  No solver logic is rewritten.
* Poisson: `lm_tau_poisson` is `followup/fu_eq.make_lm_jit` with its optional
  absolute stop `rel_tol*||f_m||` replaced by `tau*||r(z0)||`.  Everything else
  (damping schedule, acceptance test, relative-decrease / step-size stops,
  accounting, reason codes) is character-for-character the same, and
  `check_tau_agreement` asserts that at tau=0 this solver reproduces the
  reference solver's latent bit-for-bit.
* IC cold start (Burgers): `lm_tau_generic` is `followup/fu_common.lm_jit_solver`
  plus the same tau test; `check_tau_agreement` covers it too.

Reason codes (shared with the reference solvers):
  lm_tau_poisson : 0 budget, 1 converged(rel-dec/step), 2 TAU REACHED (including
                   at the initial guess), 3 lambda saturation, 5 nan_at_init
  lm_tau_generic : as above, and additionally 4 = lambda saturation after a
                   NON-FINITE step (the reference `lm_jit_solver` distinguishes
                   these two; `fu_eq.make_lm_jit` does not, and neither does
                   lm_tau_poisson)
  Burgers step   : 0 budget, 1 TAU REACHED, 2 stalled, 3 lambda_max/nan,
                   4 tol at init, 5 nan_at_init  (blat_common.lm_step_jit)
A cell is CENSORED at tau when the solver stopped for any reason other than
reaching tau (Poisson: for any test source; Burgers: at any time step or in the
cold start).  Censored cells are reported, never dropped.
"""
from __future__ import annotations

import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

F64 = jnp.float64

# reason codes that mean "the tau target was met"
GENERIC_TAU_OK = (2,)          # lm_tau_generic  (cold start)
POISSON_TAU_OK = (2,)          # lm_tau_poisson
BURGERS_TAU_OK = (1, 4)        # blat_common.lm_step_jit: 1 = tol, 4 = tol at init


# --------------------------------------------------------------------------
# generic residual-norm LM with a relative-reduction stop
# --------------------------------------------------------------------------
def lm_tau_generic(f, K, budget):
    """Jitted lm(z0, tau) minimising ||f(z)||, algorithmically identical to
    `fu_common.lm_jit_solver` / `ms_autodecoder.lm_solve` (same lam0=1e-6,
    /3 on accept, x10 on reject, clamp [1e-12, 1e12]; accept iff finite and
    strictly decreasing; stop tests only after an accepted step) with ONE
    addition: stop when ||r|| <= tau * ||r(z0)|| (tau <= 0 disables it, which
    reproduces the reference solver exactly).

    Returns (z, rn, rn0, n_jac, n_res, accepted, rejected, attempts, reason)."""
    rJ = lambda z: (f(z), jax.jacfwd(f)(z))
    rn_fn = lambda z: jnp.linalg.norm(f(z))

    def lm(z0, tau):
        r0, J0 = rJ(z0)
        rn0 = jnp.linalg.norm(r0)
        tol = tau * rn0
        # tau can already hold at the initial guess (||r(z0)|| == 0); the reference
        # solver would then run on and terminate as "no strict decrease possible",
        # which would be recorded as censored.  Test it up front, exactly as
        # blat_common's lm_step_jit does with its reason 4.
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where((tau > 0) & (rn0 <= tol), jnp.int32(2),
                                          jnp.int32(0)))
        # z, r, J, rn, lam, attempts, accepted, rejected, n_res, n_jac, reason
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), jnp.int32(1), init_reason)

        def cond(s):
            return (s[10] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, rej, n_r, n_J, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            rn_new = jnp.where(finite, rn_fn(z_new), jnp.inf)
            accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new), lambda: (r, J))
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            rej = rej + (~accept).astype(jnp.int32)
            n_J = n_J + accept.astype(jnp.int32)
            n_r = n_r + finite.astype(jnp.int32) + accept.astype(jnp.int32)
            # TAU is tested FIRST: it is the stopping rule this study reports.
            reason = jnp.where(
                accept & (tau > 0) & (rn <= tol), jnp.int32(2),
                jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), jnp.int32(1),
                          jnp.where((~accept) & (lam >= 1e12),
                                    jnp.where(finite, jnp.int32(3), jnp.int32(4)),
                                    jnp.int32(0))))
            return (z, r2, J2, rn, lam, att + 1, acc, rej, n_r, n_J, reason)

        z, r, J, rn, lam, att, acc, rej, n_r, n_J, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, rn0, n_J, n_r, acc, rej, att, reason

    return jax.jit(lm)


# --------------------------------------------------------------------------
# Poisson: weak-form residual r(z) = Wl * (PhiT @ (wq * u(z, pts))) - f_m
# --------------------------------------------------------------------------
def lm_tau_poisson(dec, K, pts, wq, PhiT, Wl, budget):
    """`fu_eq.make_lm_jit` with the absolute stop replaced by tau*||r(z0)||.
    Returns lm(z0, f_m, tau) -> (z, val, val0, n_jac, accepted, attempts,
    reason)."""
    pts = jnp.asarray(pts); wq = jnp.asarray(wq)
    PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)

    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m

    rJ = lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m))
    rn_fn = lambda z, f_m: jnp.linalg.norm(r_of(z, f_m))

    def lm(z0, f_m, tau):
        r0, J0 = rJ(z0, f_m)
        v0 = jnp.linalg.norm(r0)
        tol = tau * v0
        init_reason = jnp.where(~jnp.isfinite(v0), jnp.int32(5),
                                jnp.where((tau > 0) & (v0 <= tol), jnp.int32(2),
                                          jnp.int32(0)))
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(1), init_reason)

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, acc, nJ, _ = s
            H = J.T @ J; g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            v_new = jnp.where(finite, rn_fn(z_new, f_m), jnp.inf)
            accept = finite & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, f_m), lambda: (r, J))
            z = jnp.where(accept, z_new, z); val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32); nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (tau > 0) & (val <= tol), jnp.int32(2),
                               jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)),
                                         jnp.int32(1),
                                         jnp.where((~accept) & (lam >= 1e12), jnp.int32(3),
                                                   jnp.int32(0))))
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason)

        z, J, r, val, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, val, v0, nJ, acc, att, reason

    return jax.jit(lm), jax.jit(rn_fn)


# --------------------------------------------------------------------------
# Burgers: tau-stopped latent rollout on the REFERENCE step kernel
# --------------------------------------------------------------------------
def rollout_tau_burgers(ops, num_steps, budget):
    """Fully on-device latent rollout (lax.scan) whose per-step absolute
    tolerance is tau * ||r_n(z_n)||, i.e. the relative reduction of THAT step's
    own objective from its warm start.

    The step kernel is `ops["step_jit"]` -- blat_common's `lm_step_jit`,
    unmodified.  The only addition is one residual-norm evaluation per step to
    form the step's own reference value (blat_common's own `rollout()` already
    evaluates the residual at the warm start through `rJ_lspg`; here it is
    evaluated once more so the tolerance can be formed before the kernel is
    entered, and that extra evaluation IS inside the timed region -- it is a
    real cost of the stopping rule, not an accounting artefact).

    Returns rollout(z0, nu, tau) -> (Z (T,K), rn (T,), rn0 (T,), n_jac (T,),
    attempts (T,), reason (T,))."""
    step_jit = ops["step_jit"]
    prev_of = ops["prev_of"]
    rn_fn = ops["rn"]

    def rollout(z0, nu, tau):
        def body(carry, _):
            z, prev_c = carry
            rn0 = rn_fn(z, prev_c, nu)
            z2, rn, nJ, acc, reason, att = step_jit(z, prev_c, nu, tau * rn0, budget)
            return (z2, prev_of(z2)), (z2, rn, rn0, nJ, att, reason)

        (zT, _), out = jax.lax.scan(body, (z0, prev_of(z0)), None, length=num_steps)
        return out

    return jax.jit(rollout)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def sha256_of(paths):
    """sha256 of each existing path, keyed by basename.  The executed bundle is
    assembled from several source worktrees, so the git commit of THIS tree does
    not identify it; hashing every staged module and checkpoint into the result
    JSON does."""
    import hashlib
    out = {}
    for p in paths:
        if p and os.path.isfile(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[os.path.basename(p)] = h.hexdigest()[:16]
    return out


def module_files(modules):
    return [getattr(m, "__file__", None) for m in modules]


# --------------------------------------------------------------------------
# agreement check against the reference solvers
# --------------------------------------------------------------------------
def check_tau_agreement(tau_lm, ref_lm, args_tau, args_ref, label, tol=1e-12):
    """Assert the tau solver at tau=0 reproduces the reference solver's latent.
    Returns the relative latent difference."""
    z_a = np.asarray(tau_lm(*args_tau)[0])
    z_b = np.asarray(ref_lm(*args_ref)[0])
    d = float(np.linalg.norm(z_a - z_b) / (1.0 + np.linalg.norm(z_b)))
    if not np.isfinite(d) or d > tol:
        raise SystemExit(f"tau-solver disagrees with the reference solver at tau=0 "
                         f"({label}): rel |dz| = {d:.3e} > {tol:.0e}")
    return d
