"""A field-metric ridge on the correction block, and nothing else.

The reduced state, the directions, the weak objective, the initializer policy, the
stopping rule and the output contract are the `b-ladder-top` budget-600 contract
verbatim:

    u(z, y) = G ( h_theta(z) + C_q y ),      w = (z, y) in R^{K+q}.

R1 replaces the per-step objective

    min_w || r_w(z, y) ||^2          by          min_w || r_w(z, y) ||^2 + lambda ||y||_W^2

with W the FIELD metric. Because the ladder's own directions are built as
C_q = R_G^{-1} Ctilde[:, :q] with G = Q_G R_G the thin QR of the bank and Ctilde
orthonormal, G C_q has orthonormal columns and

    || G C_q y ||_2 == || y ||_2 ,

so the correction coordinates are already field-whitened (the same R_G whitening the
prior-dial cell used) and the penalty is literally lambda ||y||^2. See DESIGN.md §3.1.

lambda is carried as a DIMENSIONLESS lambda_rel through

    lambda = lambda_rel * sigma_q^2 ,    sigma_q = || Phi^T G C_q ||_2 ,

the largest singular value of the exact, state-independent linear part of dr_w/dy: the
1/(1 + dt nu Lambda) row scaling of the weak residual cancels the diffusion term exactly,
leaving A C_q = Phi^T G C_q. This is prior-dial's scaling transported from the full bank
to the q-dimensional correction block.

lambda enters as an augmented residual, so no line of the solver is rewritten:

    F(z, y) = [ r_w(z, y) ; sqrt(lambda) y ] in R^{M+q},

and `varpro.make_block_lm` applied to F forms exactly the ridged inner normal equations

    ( J^T J + lambda P_y + diag(mu d . m_z + eps d . m_y) ) dw = -( J^T r + lambda [0; y] ).

Acceptance, the residual exit and the normalized-gradient exit are all taken on (F, J_F):
the same stopping rule, applied to the objective actually being minimised.

lambda == 0 takes a PYTHON branch to the retained path itself — `topfix.make_query` with
the `base` arm, which is `varpro.make_block_lm`, and at q = 0 `varpro.make_query` — so the
reproduction gate against b-ladder-top is exact rather than floating-point approximate.

The initial-condition fit is NOT ridged (DESIGN.md §3.4): it is a field fit in the
cold-start Gauss space, not a fit to the M weak test equations, and freezing it makes the
t = 0 output field bitwise invariant in lambda, which is an in-job gate.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import varpro as VP
import topfix as TF

corrected_head = VP.corrected_head
blocks = VP.blocks
joint_gradient = VP.joint_gradient
build_operators = TF.build_operators
enriched_codes = TF.enriched_codes


def sigma_q(data, C):
    """|| Phi^T G C_q ||_2 — the exact linear response of r_w to y.

    Formed from the SAME `A = Phi^T G` the residual uses, so it is the arm's own operator
    and not a re-derivation of it. At q = 0 there is no y block and sigma is reported as
    None.
    """
    if C.shape[1] == 0:
        return None
    return float(jnp.linalg.norm(jnp.asarray(data['A']) @ jnp.asarray(C), 2))


def make_query(params, C, K, q, L, dt, trust, quadrature, lam_abs, ic_budget=400,
               step_budget=600, gtol=1e-6, ic_gtol=None, linear='gj', inner_damping=1e-10):
    """The complete matched query: supplied dense field on GPU -> six dense fields.

    Returns, per invocation, exactly the `topfix.make_query` / `varpro.make_query` tuple:
      0 fields (6, L+1, L+1)   1 iterations/step        2 residual norms/step (of F)
      3 exit reasons/step      4 output latents         5 ic iterations
      6 ic reason              7 internal latents (steps+1, K+q)
      8 z-block gradient/step  9 ic outer gradient     10 ic residual   11 ||u_gauss||
     12 joint normalized gradient/step                 13 ic joint gradient
     14 y-block gradient/step

    Nothing is added to the timed contract: the per-step bank coefficients R3 needs are
    formed OUTSIDE this function from output 7, by `coefficients`, and are never timed.
    """
    ic_gtol = gtol if ic_gtol is None else ic_gtol
    lam = float(lam_abs)
    assert lam >= 0.
    if lam == 0.:
        # The retained path itself, not a re-implementation that happens to agree.
        return TF.make_query(params, C, K, q, L, dt, trust, quadrature, 'base',
                             ic_budget=ic_budget, step_budget=step_budget, gtol=gtol,
                             ic_gtol=ic_gtol, linear=linear, inner_damping=inner_damping)
    assert q > 0, 'lambda is vacuous at q = 0: there is no correction block to penalise'

    head = corrected_head(params, C, K)
    hz = VP.head_only(params)
    wk = A.weak_fn(quadrature)
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    root = jnp.sqrt(jnp.asarray(lam))

    def res_aug(z, y, prev, nu, data):
        return jnp.concatenate((wk(jnp.concatenate((z, y)), prev, nu, data, head, L, dt),
                                root * y))

    outer = VP.make_block_lm(res_aug, K, q, step_budget, trust, gtol, inner_damping)
    ic_lm = A.make_stationary_lm(lambda z, tgt, Rm, P: P(Rm @ hz(z) - tgt),
                                 ic_budget, gtol=ic_gtol, linear='gj')

    def body(u0, nu, data, cold):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        yv = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ yv)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))

        # --- the retained, UNRIDGED initial fit: y eliminated by orthogonal projection.
        RC = R @ C
        Qr, Rr = jnp.linalg.qr(RC, mode='reduced')
        proj = lambda v: v - Qr @ (Qr.T @ v)
        z, icrn, icit, icreason, icgn = ic_lm(Zcand[idx][:K], (yv, R, proj), 0.)
        ycor = jax.scipy.linalg.solve_triangular(Rr, Qr.T @ (yv - R @ hz(z)), lower=False)
        w0 = jnp.concatenate((z, ycor))
        icres = R @ head(w0) - yv
        icJz = jax.jacfwd(lambda zz: R @ head(jnp.concatenate((zz, ycor))) - yv)(z)
        icgj = jnp.linalg.norm(
            jnp.concatenate((icJz.T @ icres, RC.T @ icres))) / (
            jnp.sqrt(jnp.linalg.norm(icJz) ** 2 + jnp.linalg.norm(RC) ** 2)
            * jnp.linalg.norm(icres) + 1e-300)

        def step(carry, _):
            # The retained two-way initializer guard, verbatim, on the UNRIDGED residual:
            # the guard compares two candidate starting iterates, and the penalty term is
            # identical for the extrapolated and the held iterate only when y differs, so
            # the guard is taken on the ridged objective for consistency.
            wv, wprev = carry
            p = data['A'] @ head(wv)
            we = wv + (wv - wprev)
            r0 = jnp.linalg.norm(res_aug(wv[:K], wv[K:], p, nu, data))
            re = jnp.linalg.norm(res_aug(we[:K], we[K:], p, nu, data))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            z2, y2, rn, it, reason, _ = outer(wi[:K], wi[K:], (p, nu, data), 1e-9 * scale)
            gzn, nz, gyn, ny, rnorm = blocks(res_aug, z2, y2, (p, nu, data))
            gj = joint_gradient(gzn, nz, gyn, ny, rnorm)
            gz = gzn / (nz * rnorm + 1e-300)
            gy = gyn / (ny * rnorm + 1e-300)
            return (jnp.concatenate((z2, y2)), wv), (
                jnp.concatenate((z2, y2)), rn, it, reason, gz, gj, gy)

        _, out = jax.lax.scan(step, (w0, w0), None, length=steps)
        ws, rn, it, reason, gn, gj, gy = out
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = jax.vmap(lambda v: e.output_field(data['G'] @ head(v), L, L))(W)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn,
                icrn, jnp.linalg.norm(ui), gj, icgj, gy)

    return jax.jit(body)


def coefficients(params, C, K):
    """w -> h_theta(z) + C y, the bank coefficient vector, for a whole trajectory.

    Untimed and outside the query: R3 reconstructs every step's grid field as G c_n in
    NumPy from these, so the timed contract is not touched.
    """
    head = corrected_head(params, C, K)
    return jax.jit(jax.vmap(head))
