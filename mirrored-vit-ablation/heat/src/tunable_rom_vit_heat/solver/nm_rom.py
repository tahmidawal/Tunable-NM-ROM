"""NM-ROM solver for the Heat equation with symmetric ViT decoder.

Identical projection scheme and EQ residual evaluation as the baseline,
but the decoder is evaluated on the FULL grid each step, then indexed
at stencil nodes. The CP-factor precomputation `v_eq_stencil` is no
longer used (set to None; kept in the signature so the eq module and
training code stay unchanged).

This is the load-bearing change for the ablation: self-attention
couples all output tokens, so there is no precomputable per-node
decoder — the decoder cost stays O(N^d * vit_cost) per GN iteration.

API matches `tunable_rom_heat.solver.NMROMSolver`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..fom.heat import DT


@dataclass
class NMROMSolver:
    autoencoder: object         # ViTViTAutoencoder
    params: dict                # trained AE params
    N: int
    spatial_dim: int
    dx: float
    eq_flat_indices: np.ndarray
    eq_weights: np.ndarray
    v_eq_stencil: Optional[np.ndarray] = None    # unused; kept for API parity
    stencil_indices: np.ndarray = None           # (n_eq, 2d+1)
    gn_max_iters: int = 8
    gn_rel_tol: float = 1e-3
    lm_damping: float = 1e-3

    def __post_init__(self):
        d = self.spatial_dim
        self.coeff = 2.0 * d
        self.n_eq = self.eq_flat_indices.shape[0]
        self.stencil_w = 2 * d + 1
        flat = self.stencil_indices.reshape(-1)
        if d == 2:
            ix = flat // self.N
            iy = flat % self.N
            inside = (ix > 0) & (ix < self.N - 1) & (iy > 0) & (iy < self.N - 1)
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
        self.w_eq = jnp.asarray(self.eq_weights)
        self.stencil_flat = jnp.asarray(self.stencil_indices.reshape(-1))

    def _decode_unit(self, z):
        """Evaluate the ViT decoder on the FULL grid, unit-scale, flat."""
        return self.autoencoder.apply(
            {"params": self.params}, z, method=self.autoencoder.decode_unit
        )

    def f_norm_eq(self, z, kappa):
        """Unit-scale Heat update at EQ centres: u_c + dt*kappa*lap_neg(u)."""
        u_full = self._decode_unit(z)
        u_st_flat = u_full[self.stencil_flat]
        u_st = u_st_flat.reshape(self.n_eq, self.stencil_w) * self.mask_st
        lap = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        return u_st[:, 0] + DT * kappa * lap

    def residual(self, z, scale, u_prev_eq, kappa):
        return scale * self.f_norm_eq(z, kappa) - u_prev_eq

    def _step(self, z, scale, u_prev_eq, kappa):
        def step_body(carry):
            z_s, gnorm0, gnorm, itr = carry
            zc, sc = z_s[:-1], z_s[-1]
            R = self.residual(zc, sc, u_prev_eq, kappa)
            J_z = jax.jacfwd(lambda zz: self.residual(zz, sc, u_prev_eq, kappa))(zc)
            f_norm_vec = self.f_norm_eq(zc, kappa)
            J = jnp.concatenate([J_z, f_norm_vec[:, None]], axis=1)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / (z_s.size), 1e-8)
            dz_s = jnp.linalg.solve(H + damp * jnp.eye(z_s.size), -g)
            steps = jnp.asarray([1.0, 0.5, 0.25, 0.125])
            def try_step(a):
                cand = z_s + a * dz_s
                Rc = self.residual(cand[:-1], cand[-1], u_prev_eq, kappa)
                return 0.5 * jnp.sum(self.w_eq * Rc**2)
            losses = jax.vmap(try_step)(steps)
            best = jnp.argmin(losses)
            z_s_new = z_s + steps[best] * dz_s
            return (z_s_new, gnorm0, jnp.linalg.norm(g), itr + 1)

        def cond(carry):
            _, gnorm0, gnorm, itr = carry
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
        z0, scale0 = self.autoencoder.apply(
            {"params": self.params}, u0_flat, method=self.autoencoder.encode
        )
        u0_eq = u0_flat[self.eq_flat_indices]

        def body(i, carry):
            z, s, u_prev_eq, iters_buf = carry
            z_new, s_new, it = self._step(z, s, u_prev_eq, kappa)
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
