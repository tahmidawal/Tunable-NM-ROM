"""Zero-anchored symmetric ViT decoder for the Heat NM-ROM ablation.

  u(z) = unpatchify( ViT(z) - ViT(0) )

Subtracting ViT(0) ensures decoder(0) = 0 by construction. Without it,
the pos_embed plus transformer biases give a rich non-trivial output
at z=0 that the network exploits as a "mean-field" cheat, collapsing
the encoder to near-zero latents.

No linear skip (Heat warm-starts each step from the previous latent,
so cold-start Jacobian regularity is not required).
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


class ViTBranch(nn.Module):
    N: int
    spatial_dim: int
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int

    def setup(self):
        assert self.N % self.patch_size == 0
        self.n_per_side = self.N // self.patch_size
        self.num_patches = self.n_per_side ** self.spatial_dim
        self.patch_features = self.patch_size ** self.spatial_dim

    @nn.compact
    def __call__(self, z):
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

        d = self.spatial_dim
        n = self.n_per_side
        p = self.patch_size
        if d == 2:
            return patches.reshape(n, n, p, p).transpose(0, 2, 1, 3).reshape(self.N, self.N)
        elif d == 3:
            return (
                patches.reshape(n, n, n, p, p, p)
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(self.N, self.N, self.N)
            )
        raise ValueError(f"spatial_dim must be 2 or 3, got {d}")


class ViTDecoder(nn.Module):
    """Zero-anchored symmetric ViT decoder for Heat."""

    N: int
    spatial_dim: int  # 2 or 3
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    latent_dim: int

    def setup(self):
        self.vit = ViTBranch(
            N=self.N,
            spatial_dim=self.spatial_dim,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            name="vit",
        )

    def __call__(self, z):
        return self.vit(z) - self.vit(jnp.zeros_like(z))
