"""Backward-Euler LSPG time stepping on a trial manifold, Gauss-Newton as published.

x_n = argmin_z || r(D(z); D(z_{n-1})) ||_2^2, started from z_{n-1} (Kim et al. 2022, Eq. 3.8;
Lee & Carlberg 2020). `resfun(z, zprev, args)` returns the (possibly hyper-reduced, already
left-multiplied by (Z^T Phi_r)^+) residual vector. Plain Gauss-Newton, no line search; the
Jacobian is formed by forward mode over the latent directions and the step by thin QR.
"""
import jax
import jax.numpy as jnp


def make_rollout(resfun, nsteps, max_it=20, tol=1e-8, save_every=1, keep_residuals=False, jac_batch=None):
    def gn_step(z, zprev, args):
        r, lin = jax.linearize(lambda q: resfun(q, zprev, args), z)
        eye = jnp.eye(z.size, dtype=z.dtype)
        # jac_batch bounds the memory of the decoder tangent gather (n*P per latent direction)
        J = jax.vmap(lin, out_axes=1)(eye) if jac_batch is None else jax.lax.map(lin, eye, batch_size=jac_batch).T
        Q, R = jnp.linalg.qr(J)
        dz = -jax.scipy.linalg.solve_triangular(R, Q.T @ r)
        return dz, r

    def solve(zprev, args):
        def cond(s):
            z, it, dn, _ = s
            return (it < max_it) & (dn > tol * (1. + jnp.linalg.norm(z))) & jnp.isfinite(dn)
        def body(s):
            z, it, _, _ = s
            dz, r = gn_step(z, zprev, args)
            return z + dz, it + 1, jnp.linalg.norm(dz), jnp.linalg.norm(r)
        z, it, dn, rn = jax.lax.while_loop(cond, body, (zprev, jnp.int32(0), jnp.asarray(1e30, zprev.dtype),
                                                       jnp.asarray(0., zprev.dtype)))
        return z, it, dn, rn

    def rollout(z0, args):
        def block(z, _):
            def one(z, _):
                z2, it, dn, rn = solve(z, args)
                out = (it, dn, rn) + ((resfun(z2, z, args),) if keep_residuals else ())
                return z2, out
            z, out = jax.lax.scan(one, z, None, length=save_every)
            return z, (z, out)
        _, (Z, out) = jax.lax.scan(block, z0, None, length=nsteps // save_every)
        return jnp.concatenate((z0[None], Z)), out
    return rollout
