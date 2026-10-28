"""CP-tensor decoder for the Heat NM-ROM.

Plain CP factorization: a small MLP maps the latent code to rank-R
channel weights, which are contracted with CP factor matrices to
reconstruct the field on the grid.

For 2D:  u[i,j]   = sum_r  h[r] * W_x[r,i] * W_y[r,j]   + bias
For 3D:  u[i,j,k] = sum_r  h[r] * W_x[r,i] * W_y[r,j] * W_z[r,k]  + bias

Heat's NM-ROM warm-starts each timestep from the previous step's latent
code, so cold-start GN regularity at z=0 is not required. A plain CP
decoder (MLP + contraction) is sufficient. Contrast with the Poisson
repo, which adds a linear skip for cold-start convergence.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class CPDecoder(nn.Module):
    """Plain CP decoder: MLP -> rank-R weights -> CP tensor contraction."""

    N: int
    spatial_dim: int  # 2 or 3
    latent_dim: int
    rank: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z):
        # MLP to rank-R channel weights.
        h = nn.swish(nn.Dense(self.hidden_dim, name="W1")(z))
        h = nn.swish(nn.Dense(self.hidden_dim, name="W2")(h))
        h = nn.Dense(self.rank, name="W_rank")(h)

        # CP factor matrices, one per spatial axis.
        factor_init = nn.initializers.normal(stddev=0.01)
        if self.spatial_dim == 2:
            W_x = self.param("W_x", factor_init, (self.rank, self.N))
            W_y = self.param("W_y", factor_init, (self.rank, self.N))
            u = jnp.einsum("r,ri,rj->ij", h, W_x, W_y)
        elif self.spatial_dim == 3:
            W_x = self.param("W_x", factor_init, (self.rank, self.N))
            W_y = self.param("W_y", factor_init, (self.rank, self.N))
            W_z = self.param("W_z", factor_init, (self.rank, self.N))
            u = jnp.einsum("r,ri,rj,rk->ijk", h, W_x, W_y, W_z)
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {self.spatial_dim}")

        bias = self.param("bias", nn.initializers.zeros, ())
        return u + bias
