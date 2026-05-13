"""Scale-aware autoencoder combining ViT encoder and CP decoder.

The encoder operates on a unit-scale field u/s where s = max|u|. The
decoder reconstructs the unit-scale field, and the final output is
multiplied by s. This makes training robust across the wide amplitude
range of Heat trajectories (factor of ~10 from t=0 to t=T·0.25).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .encoder import ViTEncoder
from .decoder import CPDecoder


class ViTCPAutoencoder(nn.Module):
    N: int
    spatial_dim: int
    patch_size: int
    embed_dim: int
    num_heads: int
    num_enc_layers: int
    latent_dim: int
    rank: int
    hidden_dim: int = 256

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
        self.decoder = CPDecoder(
            N=self.N,
            spatial_dim=self.spatial_dim,
            latent_dim=self.latent_dim,
            rank=self.rank,
            hidden_dim=self.hidden_dim,
        )

    def encode(self, u_flat):
        """Returns (z, scale) for a flat input field."""
        scale = jnp.maximum(jnp.max(jnp.abs(u_flat)), 1e-6)
        z = self.encoder(u_flat / scale)
        return z, scale

    def decode(self, z, scale):
        u = self.decoder(z) * scale
        return u.reshape(-1)

    def __call__(self, u_flat):
        z, scale = self.encode(u_flat)
        return self.decode(z, scale)
