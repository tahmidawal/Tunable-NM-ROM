"""ViT encoder + symmetric ViT decoder (+ linear skip) for the Poisson ablation.

Same I/O as the baseline ViT+LinearCP autoencoder: encoder and decoder
operate directly on the raw field (no amplitude scaling, no time
integration). Only the decoder architecture changes
(LinearCPDecoder -> LinearSkipViTDecoder).
"""
from __future__ import annotations

import flax.linen as nn

from .encoder import ViTEncoder
from .decoder import LinearSkipViTDecoder


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
        self.decoder = LinearSkipViTDecoder(
            N=self.N,
            spatial_dim=self.spatial_dim,
            patch_size=self.dec_patch_size if self.dec_patch_size is not None else self.patch_size,
            embed_dim=self.dec_embed_dim if self.dec_embed_dim is not None else self.embed_dim,
            num_heads=self.dec_num_heads if self.dec_num_heads is not None else self.num_heads,
            num_layers=self.dec_num_layers if self.dec_num_layers is not None else self.num_enc_layers,
            latent_dim=self.latent_dim,
        )

    def encode(self, u_flat):
        return self.encoder(u_flat)

    def decode(self, z):
        return self.decoder(z).reshape(-1)

    def __call__(self, u_flat):
        return self.decode(self.encode(u_flat))
