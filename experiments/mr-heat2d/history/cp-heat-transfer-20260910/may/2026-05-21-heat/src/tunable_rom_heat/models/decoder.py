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
        #
        # The contraction is done explicitly, one axis at a time, rather
        # than as a single `einsum("r,ri,rj,rk->ijk", ...)`. XLA's default
        # plan for the 4-operand einsum can materialize a (rank, N, N, N)
        # intermediate -- 2048 * 64^3 * 4 B = 2.1 TB at N=64 -- which
        # crashed Step-2 training with CUDA_ERROR_ILLEGAL_ADDRESS. The
        # staged form below never holds anything larger than (rank, N, N)
        # (~33 MB), so it is memory-safe at every resolution.
        factor_init = nn.initializers.normal(stddev=0.01)
        if self.spatial_dim == 2:
            W_x = self.param("W_x", factor_init, (self.rank, self.N))
            W_y = self.param("W_y", factor_init, (self.rank, self.N))
            hx = W_x * h[:, None]                       # (rank, N)
            u = jnp.einsum("ri,rj->ij", hx, W_y)        # contract rank
        elif self.spatial_dim == 3:
            W_x = self.param("W_x", factor_init, (self.rank, self.N))
            W_y = self.param("W_y", factor_init, (self.rank, self.N))
            W_z = self.param("W_z", factor_init, (self.rank, self.N))
            hx = W_x * h[:, None]                       # (rank, N)
            hxy = jnp.einsum("ri,rj->rij", hx, W_y)     # (rank, N, N)
            u = jnp.einsum("rij,rk->ijk", hxy, W_z)     # contract rank
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {self.spatial_dim}")

        bias = self.param("bias", nn.initializers.zeros, ())
        return u + bias
