"""Symmetric ViT decoder with linear skip for the Poisson NM-ROM ablation.

  u(z) = unpatchify( W_dir @ z reshaped ) + ViTDecoder(z) + bias

The ViT branch is the mirror image of the encoder: latent -> learned
query tokens -> L transformer blocks -> per-token linear head ->
un-patchify -> (N,)*d field.

The linear-skip branch maps z directly to the full grid through a
single Dense layer reshaped to (N,)*d. This is load-bearing for
cold-start Gauss-Newton: each Poisson solve starts from z = 0, so
dU/dz |_{z=0} has to be a well-conditioned map for the first step to
descend. Without it, the ViT branch's Jacobian is essentially zero
near z = 0 (every transformer block multiplies the propagated
quantity), GN's gradient direction is undefined, and the latent solve
diverges on a fraction of test cases.

The skip mirrors the role of W_direct in LinearCPDecoder, just routed
through an un-patchify instead of a CP contraction.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.embed_dim
        )(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(int(self.embed_dim * self.mlp_ratio))(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        return x + h


class LinearSkipViTDecoder(nn.Module):
    """Symmetric ViT decoder with a linear skip onto the full grid."""

    N: int
    spatial_dim: int  # 2 or 3
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    latent_dim: int

    def setup(self):
        assert self.N % self.patch_size == 0, "patch_size must divide N"
        self.n_per_side = self.N // self.patch_size
        self.num_patches = self.n_per_side ** self.spatial_dim
        self.patch_features = self.patch_size ** self.spatial_dim
        self.num_nodes = self.N ** self.spatial_dim

    @nn.compact
    def __call__(self, z):
        # Linear skip: latent -> full grid, single Dense.
        u_lin = nn.Dense(self.num_nodes, name="W_direct", use_bias=False)(z).reshape(
            (self.N,) * self.spatial_dim
        )

        # ViT branch.
        tokens = nn.Dense(
            self.num_patches * self.embed_dim, name="latent_to_tokens"
        )(z).reshape(self.num_patches, self.embed_dim)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.num_patches, self.embed_dim),
        )
        tokens = tokens + pos
        for i in range(self.num_layers):
            tokens = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                name=f"block_{i}",
            )(tokens)
        tokens = nn.LayerNorm()(tokens)
        patches = nn.Dense(self.patch_features, name="patch_head")(tokens)

        # Un-patchify.
        d = self.spatial_dim
        n = self.n_per_side
        p = self.patch_size
        if d == 2:
            u_nl = patches.reshape(n, n, p, p).transpose(0, 2, 1, 3).reshape(self.N, self.N)
        elif d == 3:
            u_nl = (
                patches.reshape(n, n, n, p, p, p)
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(self.N, self.N, self.N)
            )
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {d}")

        bias = self.param("bias", nn.initializers.zeros, ())
        return u_lin + u_nl + bias
