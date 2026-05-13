"""Tunable NM-ROM solver for the Poisson equation.

A single Levenberg-Marquardt Gauss-Newton solve in latent space. The
residual is the discrete Poisson residual `R = K u - F` evaluated only
at EQ stencil nodes. The decoder Jacobian is materialised through
precomputed CP factors at the stencil nodes (V_eq), so cost is
N-independent.

Unlike the Heat solver, Poisson:
  * has no time loop (one elliptic solve per parameter),
  * starts each solve from z = 0 (cold start — relies on
    LinearCPDecoder for Jacobian regularity),
  * does interior-only residual evaluation (stencil neighbours stay
    inside the grid by construction).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class NMROMSolver:
    autoencoder: object
    params: dict
    N: int
    spatial_dim: int
    dx: float
    eq_flat_indices: np.ndarray
    eq_weights: np.ndarray
    v_eq_stencil: np.ndarray
    stencil_indices: np.ndarray
    gn_max_iters: int = 12
    gn_rel_tol: float = 1e-3
    lm_damping: float = 1e-4

    def __post_init__(self):
        d = self.spatial_dim
        self.coeff = 2.0 * d
        self.n_eq = self.eq_flat_indices.shape[0]
        self.stencil_w = 2 * d + 1
        dp = self.params["decoder"]
        self.b_scalar = float(dp["bias"])
        self.v_eq_st = jnp.asarray(self.v_eq_stencil)
        self.w_eq = jnp.asarray(self.eq_weights)

    def _mlp_apply(self, z):
        dp = self.params["decoder"]
        h_nl = jax.nn.swish(z @ dp["W1"]["kernel"] + dp["W1"]["bias"])
        h_nl = jax.nn.swish(h_nl @ dp["W2"]["kernel"] + dp["W2"]["bias"])
        h_nl = h_nl @ dp["W_rank"]["kernel"] + dp["W_rank"]["bias"]
        # Linear skip.
        h_lin = z @ dp["W_direct"]["kernel"]
        return h_lin + h_nl

    def u_at_stencil(self, z):
        """Decoded values at EQ stencil nodes (no boundary mask — interior-only EQ)."""
        h = self._mlp_apply(z)
        u_st = (h @ self.v_eq_st + self.b_scalar).reshape(self.n_eq, self.stencil_w)
        return u_st

    def residual(self, z, F_eq):
        """R_eq[i] = (coeff * u_c - sum_neighbours) / dx^2 - F_eq[i], at EQ centres."""
        u_st = self.u_at_stencil(z)
        lap_neg = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        return lap_neg - F_eq

    def solve(self, F_eq):
        """Cold-start GN-LM solve at z = 0. Returns (z_final, residual_norm, iters)."""
        latent_dim = self.params["decoder"]["W_direct"]["kernel"].shape[0]
        z0 = jnp.zeros((latent_dim,))

        def body(carry):
            z, gnorm0, gnorm, itr = carry
            R = self.residual(z, F_eq)
            J = jax.jacfwd(lambda zz: self.residual(zz, F_eq))(z)   # (n_eq, k)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / latent_dim, 1e-8)
            dz = jnp.linalg.solve(H + damp * jnp.eye(latent_dim), -g)
            # Backtracking line search.
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

        R0 = self.residual(z0, F_eq)
        J0 = jax.jacfwd(lambda zz: self.residual(zz, F_eq))(z0)
        g0 = (J0.T * self.w_eq[None, :]) @ R0
        gnorm0 = jnp.maximum(jnp.linalg.norm(g0), 1e-30)
        z_f, _, gnorm_f, iters = jax.lax.while_loop(cond, body, (z0, gnorm0, gnorm0, 0))
        return z_f, gnorm_f, iters

    def decode(self, z):
        return self.autoencoder.apply(
            {"params": self.params}, z, method=self.autoencoder.decode
        )
