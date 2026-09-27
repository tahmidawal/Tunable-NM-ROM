"""Exact extracted current block-LM primitive; source pinned in VENDOR.json."""
import numpy as np
import jax
import jax.numpy as jnp

def make_block_lm(res, K, q, budget, trust=np.inf, gtol=1e-6, ridge=1e-10):
    """One Jacobian per iteration, but the damping and the trust radius apply to z only.

    This is the audited joint LM with the augmented normal equations split into blocks:

        ( J^T J + lam D_z + eps D_y ) [dz; dy] = - J^T r,

    where D_z zeroes the y diagonal and D_y is a fixed tiny ridge. The y block is
    therefore an exact Gauss-Newton step on the current linearisation and is never
    restricted by the trust radius, which is the audited ladder's binding constraint at
    large q; the z block keeps the retained Levenberg damping and the q = 0 trust
    radius. Cost per iteration is the joint solver's, because the two Jacobian blocks
    come from one forward-mode pass. At q = 0 it IS the joint solver.
    """
    def lm(z0, y0, args, tol):
        def ev(w):
            r = res(w[:K], w[K:], *args)
            J = jax.jacfwd(lambda ww: res(ww[:K], ww[K:], *args))(w)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        mask_z = jnp.concatenate((jnp.ones(K), jnp.zeros(q)))
        mask_y = 1. - mask_z
        w0 = jnp.concatenate((z0, y0))
        r, J, rn = ev(w0)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, rn, lam, it, reason = s
            H = J.T @ J
            d = jnp.diag(H) + 1e-30
            step = jnp.linalg.solve(H + jnp.diag(lam * d * mask_z + ridge * d * mask_y), -(J.T @ r))
            ok = jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step[:K]) <= trust)
            wn = w + jnp.where(ok, step, 0.)
            rn2 = jnp.linalg.norm(res(wn[:K], wn[K:], *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: ev(wn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, wn, w), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        w, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (w0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return w[:K], w[K:], rn, it, reason, grad(r, J)
    return lm

