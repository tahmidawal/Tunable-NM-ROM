"""NM-ROM solver for the Poisson equation with symmetric ViT decoder + skip.

Identical projection scheme as the baseline LinearCP solver, but the
decoder is evaluated on the FULL grid each GN iteration and then
indexed at the EQ stencil nodes. The CP-factor precomputation
`v_eq_stencil` is no longer used (kept in the signature for API
parity).

The linear skip W_direct (z -> full grid) is the load-bearing piece
for cold-start GN regularity. dU/dz |_{z=0} is dominated by the skip
because at z = 0 the ViT branch's positional-embedding + transformer
contribution feeds back through near-zero Dense weights into a
near-identity Jacobian — analogous to the LinearCP cold-start argument
in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    v_eq_stencil: Optional[np.ndarray] = None    # unused; kept for API parity
    stencil_indices: np.ndarray = None
    gn_max_iters: int = 12
    gn_rel_tol: float = 1e-3
    lm_damping: float = 1e-4

    def __post_init__(self):
        d = self.spatial_dim
        self.coeff = 2.0 * d
        self.n_eq = self.eq_flat_indices.shape[0]
        self.stencil_w = 2 * d + 1
        self.w_eq = jnp.asarray(self.eq_weights)
        self.stencil_flat = jnp.asarray(self.stencil_indices.reshape(-1))
        # Boundary mask on the FULL grid: 1 on interior, 0 on Dirichlet boundary.
        # The ViT decoder has no structural constraint that boundary values are
        # zero. Without masking, GN can find a "low-residual" z whose decoded
        # field has nonzero boundary, drifting away from the physical solution.
        # Apply the same FOM mask the CP decoder gets for free via its
        # axis-factor structure (factors learned ~0 on boundary indices).
        N, d = self.N, self.spatial_dim
        if d == 2:
            m = np.ones((N, N), dtype=np.float32)
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
        else:
            m = np.ones((N, N, N), dtype=np.float32)
            m[0, :, :] = m[-1, :, :] = 0.0
            m[:, 0, :] = m[:, -1, :] = 0.0
            m[:, :, 0] = m[:, :, -1] = 0.0
        self.bmask = jnp.asarray(m.reshape(-1))

    def _decode_full(self, z):
        """Evaluate the ViT decoder on the FULL grid, flat, masked to zero on boundary."""
        u = self.autoencoder.apply(
            {"params": self.params}, z, method=self.autoencoder.decode
        )
        return u * self.bmask

    def u_at_stencil(self, z):
        u_full = self._decode_full(z)
        return u_full[self.stencil_flat].reshape(self.n_eq, self.stencil_w)

    def residual(self, z, F_eq):
        u_st = self.u_at_stencil(z)
        lap_neg = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        return lap_neg - F_eq

    def _latent_dim(self):
        # Pull from the decoder's linear-skip kernel: W_direct: (k, N**d).
        return self.params["decoder"]["W_direct"]["kernel"].shape[0]

    def solve(self, F_eq):
        latent_dim = self._latent_dim()
        z0 = jnp.zeros((latent_dim,))

        def body(carry):
            z, gnorm0, gnorm, itr = carry
            R = self.residual(z, F_eq)
            J = jax.jacfwd(lambda zz: self.residual(zz, F_eq))(z)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / latent_dim, 1e-8)
            dz = jnp.linalg.solve(H + damp * jnp.eye(latent_dim), -g)
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
        u = self.autoencoder.apply(
            {"params": self.params}, z, method=self.autoencoder.decode
        )
        return u * self.bmask
