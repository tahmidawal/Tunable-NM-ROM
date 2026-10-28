"""Bounded latent transfers and common-fine-history coarse physics candidate."""
import jax
import jax.numpy as jnp
import p1core as T2
import hcore as H

INFO = ('accepted', 'down_it', 'down_reason', 'down_J', 'down_res',
        'coarse_it', 'coarse_reason', 'coarse_rejections',
        'up_it', 'up_reason', 'up_J', 'up_res', 'fine_f0', 'fine_f1',
        'down_initial_error', 'down_final_error')


def make_fit(head):
    """Retained arms.make_stationary_lm arithmetic, with logical evaluation counts.

    Count explicit residual calls separately from AD Jacobians. One accepted
    iteration reevaluates the residual and AD Jacobian, just as the original.
    """
    def fun(z, target, R):
        return R @ head(z) - target

    def fit(z0, target, R):
        def evaluate(z):
            r = fun(z, target, R)
            J = jax.jacfwd(fun)(z, target, R)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        rn0 = rn
        reason = jnp.where(jnp.isfinite(rn),
            jnp.where(grad(r, J) <= 1e-6, 4, jnp.where(rn <= 0., 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason, accepted = s
            B, g = J.T @ J, J.T @ r
            dz = H.Q.e.gj_solve(B + lam * jnp.diag(jnp.diag(B) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz))
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, target, R))
            take = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(take, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= 1e-6, 4, jnp.where(rn2 <= 0., 1,
                jnp.where(tiny, 2, jnp.where((~take) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(take, zn, z), r2, J2, rn2,
                    jnp.where(take, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason, accepted + take.astype(jnp.int32))

        z, r, J, rn, lam, it, reason, accepted = jax.lax.while_loop(
            lambda s: (s[5] < 4) & (s[6] == 0), body,
            (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason, jnp.int32(0)))
        z = jnp.where(rn < rn0, z, z0)
        return z, rn, it, reason, grad(r, J), 1 + accepted, 1 + it + accepted, rn0
    return fit


def make_candidate(hc, hf, nl, nlJ, coarse_trust, mode, force_failure=False):
    assert mode in ('physics', 'transfer')
    down, up = make_fit(hc), make_fit(hf)
    # Build the actual retained latent residual/Jacobian with the small head.
    _, coarse_res, coarse_J = H.make_head_evolve(hc, nl, nlJ, 16, coarse_trust)
    lm = T2.make_lm(coarse_J, 2, coarse_trust, 1e-10, solver='lu', clip=True)

    def seed(warm, args, coarse_tol):
        data = args[2]
        R, Hrot, Hnorm, Zcand = (data[k] for k in ('transfer_R', 'coarse_Hrot', 'coarse_Hnorm', 'coarse_Zcand'))
        target = R @ hf(warm)
        idx = jnp.argmin(Hnorm - 2 * Hrot @ target)
        z0 = Zcand[idx]
        zfit, drn, dit, dreason, dgn, dJ, dres, drn0 = down(z0, target, R)
        if mode == 'physics':
            zc, crn, cit, creason, cgn, crej, clam = lm(zfit, args, coarse_tol, 1e-3, jnp.asarray(1e-6))
        else:
            zc, cit, creason, cgn, crej = zfit, jnp.int32(0), jnp.int32(-1), jnp.asarray(0.), jnp.int32(0)
        zu, urn, uit, ureason, ugn, uJ, ures, urn0 = up(warm, R @ hc(zc), R)
        if force_failure:
            dreason = jnp.int32(3)
        # Evaluate precisely the same coefficient residual on both fine states.
        hist, nu, d, co, dt = args
        rc = T2.lspg_residual(nl, co, dt, nu, d, hist)
        r0, r1 = rc(hf(warm)), rc(hf(zu))
        f0, f1 = jnp.vdot(r0, r0), jnp.vdot(r1, r1)
        finite = (jnp.all(jnp.isfinite(zfit)) & jnp.all(jnp.isfinite(zc)) & jnp.all(jnp.isfinite(zu)) &
                  jnp.isfinite(drn) & jnp.isfinite(urn) & jnp.isfinite(dgn) & jnp.isfinite(ugn) & jnp.isfinite(cgn))
        take = (finite & (dreason != 3) & (creason != 3) & (ureason != 3) &
                jnp.isfinite(f0) & jnp.isfinite(f1) & (f1 < f0) & (f1 <= (1 - 1e-8) * f0))
        info = jnp.array([take, dit, dreason, dJ, dres, cit, creason, crej,
                          uit, ureason, uJ, ures, f0, f1, drn0, drn])
        codes = jnp.concatenate((z0, zfit, zc, zu))
        return jnp.where(take, zu, warm), info, codes
    return seed
