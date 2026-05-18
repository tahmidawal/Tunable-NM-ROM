"""NM-ROM solver specialized for AffineZDecoder.

   u(x; z) = Phi(x) @ A(z) + b(x)

Per solve:
  - Precompute Phi_st = vmap(phi)(stencil_coords)     # (n_eq*sw, h)
                bias_st = vmap(bias)(stencil_coords)  # (n_eq*sw,)
    These are constant across GN iterations -> compute ONCE.

Per GN iteration:
  - Compute A(z) ∈ R^h   (tiny MLP; ~1 µs)
  - Compute dA/dz ∈ R^{h x latent}   (jacrev on tiny MLP; ~10 µs)
  - u_st = (Phi_st @ A(z)) + bias_st     -> matmul + add
  - J_pt = Phi_st @ dA/dz                 -> matmul giving (n_eq*sw, latent)
  - Reshape, apply stencil, get R and J in residual-space

This is the CP-decoder asymptote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .nm_rom import INRNMROMSolver


@dataclass
class AffineNMROMSolver(INRNMROMSolver):
    """Same interface as INRNMROMSolver but exploits affine-in-z structure."""

    def __post_init__(self):
        super().__post_init__()
        # Precompute Phi(x_st) and bias(x_st) at every stencil point ONCE.
        # The autoencoder's decoder exposes phi() and bias() as methods.
        ae = self.autoencoder
        params = self.params

        def phi_one(x):
            return ae.apply(
                {"params": params}, x,
                method=lambda mod, xx: mod.decoder.phi(xx),
            )

        def bias_one(x):
            return ae.apply(
                {"params": params}, x,
                method=lambda mod, xx: mod.decoder.bias(xx),
            )

        # vmap precompute once; cast to float64 for solver consistency.
        Phi_st = jax.vmap(phi_one)(self.stencil_coords).astype(jnp.float64)
        bias_st = jax.vmap(bias_one)(self.stencil_coords).astype(jnp.float64)
        self._Phi_st = Phi_st                      # (n_eq*sw, h)
        self._bias_st = bias_st                    # (n_eq*sw,)

    def _coef_and_jac(self, z):
        """Return (A(z), dA/dz) using jacrev on the small coef MLP."""
        ae = self.autoencoder
        params = self.params

        def coef(zz):
            return ae.apply(
                {"params": params}, zz,
                method=lambda mod, zzz: mod.decoder.coef(zzz),
            )

        A_z = coef(z)
        dA_dz = jax.jacrev(coef)(z)               # (h, latent)
        return A_z.astype(jnp.float64), dA_dz.astype(jnp.float64)

    def _affine_residual_jac(self, z, F_eq):
        A_z, dA_dz = self._coef_and_jac(z)
        u_pt = self._Phi_st @ A_z + self._bias_st            # (n_eq*sw,)
        J_pt = self._Phi_st @ dA_dz                           # (n_eq*sw, latent)
        u_st = u_pt.reshape(self.n_eq, self.stencil_w)
        J_st = J_pt.reshape(self.n_eq, self.stencil_w, self.latent_dim)
        lap_neg = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        R = lap_neg - F_eq
        J = (self.coeff * J_st[:, 0, :]
             - jnp.sum(J_st[:, 1:, :], axis=1)) / self.dx**2
        return R, J

    def _solve_impl_affine(self, F_eq, z_init=None):
        z0 = (jnp.zeros((self.latent_dim,), dtype=jnp.float64)
              if z_init is None
              else jnp.asarray(z_init, dtype=jnp.float64))

        def body(carry):
            z, gnorm0, gnorm, itr = carry
            R, J = self._affine_residual_jac(z, F_eq)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / self.latent_dim, 1e-8)
            dz = jnp.linalg.solve(H + damp * jnp.eye(self.latent_dim), -g)
            steps = jnp.asarray([1.0, 0.5, 0.25, 0.125])

            def try_step(a):
                Rc, _ = self._affine_residual_jac(z + a * dz, F_eq)
                return 0.5 * jnp.sum(self.w_eq * Rc**2)

            losses = jax.vmap(try_step)(steps)
            best = jnp.argmin(losses)
            z_new = z + steps[best] * dz
            return (z_new, gnorm0, jnp.linalg.norm(g), itr + 1)

        def cond(carry):
            _, gnorm0, gnorm, itr = carry
            return jnp.logical_and(gnorm > self.gn_rel_tol * gnorm0, itr < self.gn_max_iters)

        R0, J0 = self._affine_residual_jac(z0, F_eq)
        g0 = (J0.T * self.w_eq[None, :]) @ R0
        gnorm0 = jnp.maximum(jnp.linalg.norm(g0), 1e-30)
        z_f, _, gnorm_f, iters = jax.lax.while_loop(
            cond, body, (z0, gnorm0, gnorm0, 0),
        )
        return z_f, gnorm_f, iters

    def make_solve(self):
        return jax.jit(self._solve_impl_affine)

    def make_solve_warm(self):
        def _solve(F_eq, z_init):
            return self._solve_impl_affine(F_eq, z_init=z_init)
        return jax.jit(_solve)
