"""Tunable NM-ROM solver for the Heat equation.

Latent-space Levenberg-Marquardt Gauss-Newton step with empirical-
quadrature (EQ) sparse residual evaluation. The residual is computed
only at EQ stencil nodes; the decoder Jacobian is materialised through
the V_eq precomputation, never via jax.jacfwd on the full grid.

The 50-step rollout is wrapped in jax.lax.fori_loop and the GN inner
iteration in jax.lax.while_loop, so the entire ROM solve compiles to a
single XLA program. Kappa is a traced runtime argument, not a closure,
to avoid recompilation per trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..fom.heat import DT


@dataclass
class NMROMSolver:
    autoencoder: object         # ViTCPAutoencoder
    params: dict                # trained AE params
    N: int
    spatial_dim: int
    dx: float
    eq_flat_indices: np.ndarray
    eq_weights: np.ndarray
    v_eq_stencil: np.ndarray    # (rank, n_eq * (2d+1))
    stencil_indices: np.ndarray  # (n_eq, 2d+1)
    gn_max_iters: int = 8
    gn_rel_tol: float = 1e-3
    lm_damping: float = 1e-3

    def __post_init__(self):
        d = self.spatial_dim
        self.coeff = 2.0 * d  # 4 in 2D, 6 in 3D
        self.n_eq = self.eq_flat_indices.shape[0]
        self.stencil_w = 2 * d + 1
        dec_params = self.params["decoder"]
        self.b_scalar = float(dec_params["bias"])
        self.W_x = jnp.asarray(dec_params["W_x"])
        self.W_y = jnp.asarray(dec_params["W_y"])
        self.W_z = jnp.asarray(dec_params.get("W_z")) if d == 3 else None
        # No boundary mask: EQ centres are restricted to [2, N-3]^d
        # (see eq.nnls._eq_candidate_indices, margin=2), so the full
        # (2d+1)-point stencil is strictly interior by construction.
        # Masking stencil values here corrupts the residual relative to
        # the FOM operator (the original defect; see DIAGNOSIS.md).
        self.v_eq_st = jnp.asarray(self.v_eq_stencil)  # (rank, n_eq * stencil_w)
        self.w_eq = jnp.asarray(self.eq_weights)
        self._mlp_apply = jax.jit(self._mlp_apply_impl)

    def _mlp_apply_impl(self, z):
        """Map latent z -> rank-channel weights via the trained MLP."""
        dp = self.params["decoder"]
        h = jax.nn.swish(z @ dp["W1"]["kernel"] + dp["W1"]["bias"])
        h = jax.nn.swish(h @ dp["W2"]["kernel"] + dp["W2"]["bias"])
        h = h @ dp["W_rank"]["kernel"] + dp["W_rank"]["bias"]
        return h

    def _u_stencil(self, z):
        """Unit-scale decoded field at EQ stencil nodes, shape (n_eq, 2d+1).

        No boundary mask: EQ centres are strictly interior (margin>=2),
        so every stencil neighbour is a genuine interior node and the
        decoded value flows in exactly as the FOM expects.
        """
        h = self._mlp_apply(z)                               # (rank,)
        u_st = (h @ self.v_eq_st + self.b_scalar)            # (n_eq * stencil_w,)
        return u_st.reshape(self.n_eq, self.stencil_w)

    def f_norm_eq(self, z, kappa):
        """A(u_hat) at EQ centres, for the unit-scale decoded field u_hat.

        Backward-Euler operator, term-for-term with fom.heat.implicit_op:
            Lap(u)_c = (sum_neighbours - 2d*u_c) / dx^2     (positive Lap)
            A(u)_c   = u_c - dt*kappa*Lap(u)_c
        """
        u_st = self._u_stencil(z)
        lap = (jnp.sum(u_st[:, 1:], axis=1) - self.coeff * u_st[:, 0]) / self.dx**2
        return u_st[:, 0] - DT * kappa * lap

    def u_center_eq(self, z):
        """Unit-scale decoded FIELD value at EQ centres (not the operator).

        This is the quantity that plays the role of u_{n-1} in the next
        step's residual R = scale*A(u_hat) - u_prev. It must be the field
        itself, not A(field) -- mixing the two breaks the backward-Euler
        recurrence (see DIAGNOSIS.md, u_prev representation seam).
        """
        return self._u_stencil(z)[:, 0]

    def residual(self, z, scale, u_prev_eq, kappa):
        """R = scale * A(u_hat(z)) - u_prev_eq, the backward-Euler step
        residual at EQ centres. u_prev_eq is the previous-step field."""
        return scale * self.f_norm_eq(z, kappa) - u_prev_eq

    def _step(self, z, scale, u_prev_eq, kappa):
        """One implicit-Euler ROM step via LM Gauss-Newton in (z, scale)."""
        def loss_only(z_s):
            z_local, s_local = z_s[:-1], z_s[-1]
            R = self.residual(z_local, s_local, u_prev_eq, kappa)
            return 0.5 * jnp.sum(self.w_eq * R**2)

        def step_body(carry):
            z_s, gnorm0, gnorm, itr = carry
            zc, sc = z_s[:-1], z_s[-1]
            R = self.residual(zc, sc, u_prev_eq, kappa)
            # Jacobian wrt (z, scale).
            J_z = jax.jacfwd(lambda zz: self.residual(zz, sc, u_prev_eq, kappa))(zc)  # (n_eq, k)
            f_norm_vec = self.f_norm_eq(zc, kappa)                                     # (n_eq,)
            J = jnp.concatenate([J_z, f_norm_vec[:, None]], axis=1)                    # (n_eq, k+1)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / (z_s.size), 1e-8)
            dz_s = jnp.linalg.solve(H + damp * jnp.eye(z_s.size), -g)
            # Backtracking line search on weighted residual norm.
            steps = jnp.asarray([1.0, 0.5, 0.25, 0.125])
            def try_step(a):
                cand = z_s + a * dz_s
                Rc = self.residual(cand[:-1], cand[-1], u_prev_eq, kappa)
                return 0.5 * jnp.sum(self.w_eq * Rc**2)
            losses = jax.vmap(try_step)(steps)
            best = jnp.argmin(losses)
            z_s_new = z_s + steps[best] * dz_s
            gnew = jnp.linalg.norm(g)
            return (z_s_new, gnorm0, gnew, itr + 1)

        def cond(carry):
            z_s, gnorm0, gnorm, itr = carry
            return jnp.logical_and(gnorm > self.gn_rel_tol * gnorm0, itr < self.gn_max_iters)

        zR0 = self.residual(z, scale, u_prev_eq, kappa)
        zJ_z0 = jax.jacfwd(lambda zz: self.residual(zz, scale, u_prev_eq, kappa))(z)
        f0 = self.f_norm_eq(z, kappa)
        J0 = jnp.concatenate([zJ_z0, f0[:, None]], axis=1)
        g0 = (J0.T * self.w_eq[None, :]) @ zR0
        gnorm0 = jnp.maximum(jnp.linalg.norm(g0), 1e-30)
        z_s0 = jnp.concatenate([z, jnp.array([scale])])
        z_s_f, _, _, iters = jax.lax.while_loop(cond, step_body, (z_s0, gnorm0, gnorm0, 0))
        return z_s_f[:-1], z_s_f[-1], iters

    def rollout(self, u0_flat, kappa, num_steps: int):
        """Run num_steps ROM steps starting from u0_flat. Returns (u_final, iters_buf)."""
        # Initial latent + scale from the AE.
        z0, scale0 = self.autoencoder.apply(
            {"params": self.params}, u0_flat, method=self.autoencoder.encode
        )
        # Initial u_prev at EQ centres (decoded full field projected).
        u0_eq = u0_flat[self.eq_flat_indices]

        def body(i, carry):
            z, s, u_prev_eq, iters_buf = carry
            z_new, s_new, it = self._step(z, s, u_prev_eq, kappa)
            # u_prev for the next step is the decoded FIELD at EQ centres
            # (u_n), not A(u_n) -- consistent with u0_eq above and with the
            # backward-Euler recurrence A(u_{n+1}) = u_n.
            u_new_eq = s_new * self.u_center_eq(z_new)
            iters_buf = iters_buf.at[i].set(it)
            return (z_new, s_new, u_new_eq, iters_buf)

        iters_buf0 = jnp.zeros((num_steps,), dtype=jnp.int32)
        z_f, s_f, _, iters_buf = jax.lax.fori_loop(
            0, num_steps, body, (z0, scale0, u0_eq, iters_buf0)
        )
        u_final = self.autoencoder.apply(
            {"params": self.params}, z_f, s_f, method=self.autoencoder.decode
        )
        return u_final, iters_buf
