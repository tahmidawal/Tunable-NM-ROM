"""LinearCPDecoder: linear skip + shallow MLP + CP-tensor contraction.

Three-branch decoder used for the parametric Poisson NM-ROM:

  h_lin = W_direct @ z                              -- linear skip
  h_nl  = W_rank @ swish(W2 @ swish(W1 @ z))        -- shallow MLP
  h     = h_lin + h_nl                              -- rank-R channels
  u     = einsum('r,ri,rj[,rk]->ij[k]', h, W_x, W_y[, W_z]) + bias

The linear skip is load-bearing for Gauss-Newton convergence from a
cold start (z = 0). With a plain MLP-only decoder, the Jacobian
dU/dz |_{z=0} is essentially zero in a wide neighbourhood of the
origin and GN's first step has nowhere to descend to. The linear skip
guarantees dU/dz |_{z=0} = W_direct @ (W_x [tensor] W_y [tensor] W_z),
a non-degenerate map of rank min(k, R), which restores GN convergence.

(Heat's NM-ROM warm-starts each step from the previous step's latent
code and so does not need the linear skip.)
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class LinearCPDecoder(nn.Module):
    N: int
    spatial_dim: int
    latent_dim: int
    rank: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z):
        # Nonlinear branch: small swish MLP -> rank head.
        h_nl = nn.swish(nn.Dense(self.hidden_dim, name="W1")(z))
        h_nl = nn.swish(nn.Dense(self.hidden_dim, name="W2")(h_nl))
        h_nl = nn.Dense(self.rank, name="W_rank")(h_nl)

        # Linear skip: latent -> rank.
        h_lin = nn.Dense(self.rank, name="W_direct", use_bias=False)(z)

        h = h_lin + h_nl

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
