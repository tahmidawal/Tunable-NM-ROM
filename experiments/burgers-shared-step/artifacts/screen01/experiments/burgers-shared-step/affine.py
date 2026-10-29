"""One affine16-coordinate initialization of the unchanged fine64 LM.

The last 48 coordinates of the ORIGINAL fine predictor remain fixed throughout
the auxiliary solve. This is not the zero-tail D16 decoder and is not multigrid.
"""
import jax
import jax.numpy as jnp
import p1core as T2

INFO = ('accepted', 'aux_it', 'aux_reason', 'aux_gn', 'aux_rejections',
        'aux_rn', 'aux_final_damping', 'fine_f0', 'fine_f1', 'aux_J',
        'guard_residuals', 'active_displacement', 'tail_displacement',
        'forced_failure', 'finite', 'reserved')
DECISION_COLUMNS = (0, 1, 2, 4, 9, 10, 13, 14)


def make_active_residual(hf, nl, nlJ):
    """Use the same residual rows, quadrature, weights and fine history.

    AD differentiates only 16 arguments, then the existing analytic advection
    tangent multiplies that R512-by-16 matrix. No full64 Jacobian is formed.
    """
    def head(a, tail):
        return hf(jnp.concatenate((a, tail)))
    jh = jax.jacfwd(head, argnums=0)

    def res(a, packed):
        args, tail = packed
        hist, nu, data, co, dt = args
        rc = T2.lspg_residual(nl, co, dt, nu, data, hist)
        return rc(head(a, tail))

    def evalJ(a, packed):
        args, tail = packed
        hist, nu, data, co, dt = args
        c, Jh = head(a, tail), jh(a, tail)
        Ac = data['A'] @ c
        val, dNJ = nlJ(c, data, Jh)
        Acn, Acm, Fn = hist
        R = co[0]*Ac + co[1]*Acn + co[2]*Acm + dt*(co[3]*(val+nu*data['lam']*Ac)+co[4]*Fn)
        S = 1. / (co[0] + dt*co[3]*nu*data['lam'])
        return R*S, data['A'] @ Jh + (dt*co[3]*S)[:, None]*dNJ

    return res, evalJ


def make_candidate(hf, nl, nlJ, prefix_trust, force_failure=False):
    _, ev = make_active_residual(hf, nl, nlJ)
    lm = T2.make_lm(ev, 600, prefix_trust, 1e-10, solver='lu', clip=True)

    def seed(warm, args, coarse_tol):
        active, rn, it, reason, gn, rej, damping = lm(
            warm[:16], (args, warm[16:]), coarse_tol, 1e-3, jnp.asarray(1e-6))
        proposed = jnp.concatenate((active, warm[16:]))
        finite = jnp.all(jnp.isfinite(proposed)) & jnp.isfinite(rn) & jnp.isfinite(gn)
        reason = jnp.where(finite, reason, jnp.int32(3))
        if force_failure:
            reason = jnp.int32(3)
        hist, nu, data, co, dt = args
        rc = T2.lspg_residual(nl, co, dt, nu, data, hist)
        r0, r1 = rc(hf(warm)), rc(hf(proposed))
        f0, f1 = jnp.vdot(r0, r0), jnp.vdot(r1, r1)
        solved = ((reason == 1) | (reason == 4)) & ((gn <= 1e-3) | (rn <= coarse_tol))
        take = (finite & solved & jnp.isfinite(f0) & jnp.isfinite(f1) &
                (f1 < f0) & (f1 <= (1 - 1e-8)*f0))
        info = jnp.array([take, it, reason, gn, rej, rn, damping, f0, f1,
                          1+it, 2, jnp.linalg.norm(active-warm[:16]),
                          jnp.linalg.norm(proposed[16:]-warm[16:]), force_failure, finite, 0.])
        return jnp.where(take, proposed, warm), info, jnp.concatenate((warm, proposed))
    return seed
