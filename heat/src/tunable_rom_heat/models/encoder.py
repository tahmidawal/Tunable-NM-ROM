"""ViT encoder shared across 2D and 3D Heat NM-ROM.

Patchify a (N,)*d field into (n_patches, patch_size^d) tokens, embed,
apply transformer blocks, mean-pool, project to latent_dim.
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


class ViTEncoder(nn.Module):
    """ViT encoder for d=2 or d=3 fields.

    Input: u with shape (N,)*d or (N**d,) flattened.
    Output: latent code of shape (latent_dim,).
    """

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

    @nn.compact
    def __call__(self, u):
        u = jnp.reshape(u, (self.N,) * self.spatial_dim)
        # Patchify: split each axis into n_per_side blocks of patch_size.
        if self.spatial_dim == 2:
            patches = u.reshape(
                self.n_per_side, self.patch_size,
                self.n_per_side, self.patch_size,
            ).transpose(0, 2, 1, 3).reshape(self.num_patches, self.patch_features)
        elif self.spatial_dim == 3:
            patches = u.reshape(
                self.n_per_side, self.patch_size,
                self.n_per_side, self.patch_size,
                self.n_per_side, self.patch_size,
            ).transpose(0, 2, 4, 1, 3, 5).reshape(self.num_patches, self.patch_features)
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {self.spatial_dim}")

        tokens = nn.Dense(self.embed_dim, name="patch_embed")(patches)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.num_patches, self.embed_dim),
        )
        tokens = tokens + pos
        for i in range(self.num_layers):
            tokens = TransformerBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, name=f"block_{i}"
            )(tokens)
        tokens = nn.LayerNorm()(tokens)
        pooled = tokens.mean(axis=0)
        z = nn.Dense(self.latent_dim, name="head")(pooled)
        return z
