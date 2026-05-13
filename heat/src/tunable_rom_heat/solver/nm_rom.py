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
        # Decoder + boundary mask split for fast stencil evaluation.
        dec_params = self.params["decoder"]
        self.b_scalar = float(dec_params["bias"])
        self.W_x = jnp.asarray(dec_params["W_x"])
        self.W_y = jnp.asarray(dec_params["W_y"])
        self.W_z = jnp.asarray(dec_params.get("W_z")) if d == 3 else None
        # Boundary mask at stencil nodes (1 = interior, 0 = boundary).
        flat = self.stencil_indices.reshape(-1)
        mask = np.ones(self.N**d, dtype=np.float32)
        # Simple boundary detection via index decomposition.
        if d == 2:
            ix = flat // self.N
            iy = flat % self.N
            self.mask_st = jnp.asarray(
                ((ix > 0) & (ix < self.N - 1) & (iy > 0) & (iy < self.N - 1)).astype(np.float32)
            ).reshape(self.n_eq, self.stencil_w)
        else:
            ix = flat // (self.N * self.N)
            iy = (flat // self.N) % self.N
            iz = flat % self.N
            inside = (
                (ix > 0) & (ix < self.N - 1)
                & (iy > 0) & (iy < self.N - 1)
                & (iz > 0) & (iz < self.N - 1)
            )
            self.mask_st = jnp.asarray(inside.astype(np.float32)).reshape(self.n_eq, self.stencil_w)
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

    def f_norm_eq(self, z, kappa):
        """Normalised decoded value at stencil nodes."""
        h = self._mlp_apply(z)                               # (rank,)
        u_st = (h @ self.v_eq_st + self.b_scalar)            # (n_eq * stencil_w,)
        u_st = u_st.reshape(self.n_eq, self.stencil_w) * self.mask_st
        lap = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        # Implicit-Euler update at center: u_c + dt*kappa*lap
        return u_st[:, 0] + DT * kappa * lap

    def residual(self, z, scale, u_prev_eq, kappa):
        """R = scale * f_norm(z, kappa) - u_prev_eq, at EQ centres."""
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
            # New u_prev at EQ centres for the next step.
            u_new_eq = s_new * self.f_norm_eq(z_new, kappa)
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
