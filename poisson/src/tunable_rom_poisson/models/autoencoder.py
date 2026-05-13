"""ViT + LinearCPDecoder autoencoder for the Poisson NM-ROM.

Unlike the Heat autoencoder, Poisson reconstructs a single elliptic
field per parameter — no time integration, no amplitude scaling. The
encoder and decoder operate directly on the field.
"""
from __future__ import annotations

import flax.linen as nn

from .encoder import ViTEncoder
from .decoder import LinearCPDecoder


class ViTLinearCPAutoencoder(nn.Module):
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
        self.decoder = LinearCPDecoder(
            N=self.N,
            spatial_dim=self.spatial_dim,
            latent_dim=self.latent_dim,
            rank=self.rank,
            hidden_dim=self.hidden_dim,
        )

    def encode(self, u_flat):
        return self.encoder(u_flat)

    def decode(self, z):
        return self.decoder(z).reshape(-1)

    def __call__(self, u_flat):
        return self.decode(self.encode(u_flat))
