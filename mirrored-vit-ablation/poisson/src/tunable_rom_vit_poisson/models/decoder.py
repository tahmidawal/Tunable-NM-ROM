"""Symmetric ViT decoder with linear skip for the Poisson NM-ROM ablation.

Zero-anchored mirror: u(z) = unpatchify(W_dir @ z) + (ViT(z) - ViT(0))

The CP decoder gets `decoder(0) ≈ 0` for free from its multiplicative
tensor-factor structure: all output values are sums of products of
trained factors, and with k=0 all those products vanish. The ViT
decoder has no such structural guarantee; the pos_embed plus
transformer biases give a non-trivial output at z=0, and the network
quickly learns to put the "mean field" in this z-independent component.
Result: the encoder produces near-zero latents and the decoder
encodes everything in its biases — Gauss-Newton has no useful
direction to descend in.

Subtracting `ViT(0)` from `ViT(z)` enforces `decoder(0) = 0` by
construction, forcing the encoder to use the latent space.
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
    """The ViT branch only: latent -> tokens -> transformer -> un-patchify field."""
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


class LinearSkipViTDecoder(nn.Module):
    """Zero-anchored symmetric ViT decoder with linear skip."""

    N: int
    spatial_dim: int  # 2 or 3
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    latent_dim: int

    def setup(self):
        self.num_nodes = self.N ** self.spatial_dim
        self.W_direct = nn.Dense(self.num_nodes, name="W_direct", use_bias=False)
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
        u_lin = self.W_direct(z).reshape((self.N,) * self.spatial_dim)
        u_vit_z = self.vit(z)
        u_vit_0 = self.vit(jnp.zeros_like(z))
        u_nl = u_vit_z - u_vit_0
        return u_lin + u_nl
