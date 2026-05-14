"""Scale-aware autoencoder: ViT encoder + symmetric ViT decoder (Heat ablation).

Same scale-amplitude wrapping as the baseline ViT+CP autoencoder: the
encoder sees u/s, the decoder reconstructs the unit-scale field, and
the final output is multiplied by s. Only the decoder architecture
changes (CPDecoder -> ViTDecoder).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .encoder import ViTEncoder
from .decoder import ViTDecoder


class ViTViTAutoencoder(nn.Module):
    N: int
    spatial_dim: int
    patch_size: int
    embed_dim: int
    num_heads: int
    num_enc_layers: int
    latent_dim: int
    # Decoder may use independent shape; None -> mirror encoder.
    dec_patch_size: int | None = None
    dec_embed_dim: int | None = None
    dec_num_heads: int | None = None
    dec_num_layers: int | None = None

    def setup(self):
        self.encoder = ViTEncoder(
            N=self.N,
            spatial_dim=self.spatial_dim,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_enc_layers,
            latent_dim=self.latent_dim,
        )
        self.decoder = ViTDecoder(
            N=self.N,
            spatial_dim=self.spatial_dim,
            patch_size=self.dec_patch_size if self.dec_patch_size is not None else self.patch_size,
            embed_dim=self.dec_embed_dim if self.dec_embed_dim is not None else self.embed_dim,
            num_heads=self.dec_num_heads if self.dec_num_heads is not None else self.num_heads,
            num_layers=self.dec_num_layers if self.dec_num_layers is not None else self.num_enc_layers,
            latent_dim=self.latent_dim,
        )

    def encode(self, u_flat):
        """Returns (z, scale) for a flat input field."""
        scale = jnp.maximum(jnp.max(jnp.abs(u_flat)), 1e-6)
        z = self.encoder(u_flat / scale)
        return z, scale

    def decode(self, z, scale):
        u = self.decoder(z) * scale
        return u.reshape(-1)

    def decode_unit(self, z):
        """Unit-scale decoder output (no amplitude scale), flat. Used by the solver."""
        return self.decoder(z).reshape(-1)

    def __call__(self, u_flat):
        z, scale = self.encode(u_flat)
        return self.decode(z, scale)
