"""Faster INRNMROMSolver variant that replaces jax.jacfwd on the
n_eq-vector residual with vmap(jacrev) at the per-point level.

Math is unchanged. The only difference is the Jacobian computation:

  baseline:  J = jacfwd(residual, argnums=z)
             -> latent_dim forward passes through all n_eq points each iter

  fast:      J_point[i] = jacrev(decoder, argnums=z)(z, x_i)
             -> 1 forward + 1 backward per point; vmapped across n_eq

For a scalar-output decoder, reverse-mode is cost ~2x the forward, so
J should be ~2x residual cost vs the current ~latent_dim x residual.

This file ONLY changes how J is assembled. Everything else is shared
with the base solver, including the LM body, line search, while_loop
control, and warm-start path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .nm_rom import INRNMROMSolver


@dataclass
class FastINRNMROMSolver(INRNMROMSolver):
    """Same interface as INRNMROMSolver. Jacobian assembly uses vmap+jacrev."""

    def _siren_residual_jac(self, z, F_eq):
        """Return (R, J) for SIREN decoders, with J via vmap(jacrev).

        For 5-point stencil 2D Poisson:
          u_st[i, j] = decoder(z, stencil_coords[i, j])    j = 0..2d
          R[i] = (coeff * u_st[i,0] - sum_{j>=1} u_st[i,j]) / dx^2 - F_eq[i]
          dR[i]/dz = (coeff * du_st[i,0]/dz - sum_{j>=1} du_st[i,j]/dz) / dx^2

        We get du_pt/dz via vmap(jacrev(decoder, argnums=0)) over the (n_eq*sw, d)
        flattened stencil coordinates, then reshape and contract.
        """
        ae = self.autoencoder
        params = self.params
        tokens = self._tokens_ref_j

        def u_at(x_single, z_in):
            # x_single: (coord_dim,), z_in: (latent,) -> scalar
            return ae.apply(
                {"params": params}, z_in, tokens, x_single[None, :],
                method=ae.decode_points,
            )[0]

        # vmap over the flat stencil coords; jacrev on z (scalar output, latent input)
        per_pt_J = jax.vmap(jax.jacrev(u_at, argnums=1), in_axes=(0, None))(
            self.stencil_coords, z,
        )  # shape (n_eq*sw, latent)
        per_pt_u = jax.vmap(u_at, in_axes=(0, None))(self.stencil_coords, z)
        u_st = per_pt_u.reshape(self.n_eq, self.stencil_w)
        J_st = per_pt_J.reshape(self.n_eq, self.stencil_w, self.latent_dim)
        lap_neg = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        R = lap_neg - F_eq
        J = (self.coeff * J_st[:, 0, :]
             - jnp.sum(J_st[:, 1:, :], axis=1)) / self.dx**2
        return R, J

    def _solve_impl_fast(self, F_eq, z_init=None):
        z0 = (jnp.zeros((self.latent_dim,), dtype=jnp.float64)
              if z_init is None
              else jnp.asarray(z_init, dtype=jnp.float64))

        def body(carry):
            z, gnorm0, gnorm, itr = carry
            R, J = self._siren_residual_jac(z, F_eq)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / self.latent_dim, 1e-8)
            dz = jnp.linalg.solve(H + damp * jnp.eye(self.latent_dim), -g)
            steps = jnp.asarray([1.0, 0.5, 0.25, 0.125])

            def try_step(a):
                Rc = self.residual(z + a * dz, F_eq)
                return 0.5 * jnp.sum(self.w_eq * Rc**2)

            losses = jax.vmap(try_step)(steps)
            best = jnp.argmin(losses)
            z_new = z + steps[best] * dz
            return (z_new, gnorm0, jnp.linalg.norm(g), itr + 1)

        def cond(carry):
            _, gnorm0, gnorm, itr = carry
            return jnp.logical_and(gnorm > self.gn_rel_tol * gnorm0, itr < self.gn_max_iters)

        R0, J0 = self._siren_residual_jac(z0, F_eq)
        g0 = (J0.T * self.w_eq[None, :]) @ R0
        gnorm0 = jnp.maximum(jnp.linalg.norm(g0), 1e-30)
        z_f, _, gnorm_f, iters = jax.lax.while_loop(
            cond, body, (z0, gnorm0, gnorm0, 0),
        )
        return z_f, gnorm_f, iters

    def make_solve(self):
        return jax.jit(self._solve_impl_fast)

    def make_solve_warm(self):
        def _solve(F_eq, z_init):
            return self._solve_impl_fast(F_eq, z_init=z_init)
        return jax.jit(_solve)
