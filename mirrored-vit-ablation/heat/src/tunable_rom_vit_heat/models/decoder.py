"""Symmetric ViT decoder for the Heat NM-ROM ablation.

Mirror image of the ViT encoder:

  z (k,)
   -> Dense(num_patches * embed_dim) -> reshape to (num_patches, embed_dim)
   -> + decoder positional embedding
   -> L transformer blocks (same depth / heads / mlp_ratio as encoder)
   -> LayerNorm
   -> Dense(patch_size**d) per token
   -> un-patchify -> (N,)*d field

This is the "MAE-style" decoder. Self-attention couples all output
tokens, so the field at a single mesh node depends on the entire latent
code through every transformer block. This breaks the per-node
locality that EQ hyper-reduction exploits with the CP decoder, and is
the central tradeoff this ablation is meant to expose.

Heat does NOT need a linear skip: each ROM time step warm-starts from
the previous step's latent, so the decoder Jacobian only has to be
well-conditioned around the trained latent codes, not at z = 0.
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


class ViTDecoder(nn.Module):
    """Symmetric ViT decoder: latent -> tokens -> transformer -> un-patchify."""

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

        # Un-patchify: inverse of the encoder's patchify.
        d = self.spatial_dim
        n = self.n_per_side
        p = self.patch_size
        if d == 2:
            u = patches.reshape(n, n, p, p).transpose(0, 2, 1, 3).reshape(self.N, self.N)
        elif d == 3:
            u = (
                patches.reshape(n, n, n, p, p, p)
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(self.N, self.N, self.N)
            )
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {d}")

        bias = self.param("bias", nn.initializers.zeros, ())
        return u + bias
