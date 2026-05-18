"""Decoder-agnostic INR autoencoder.

Shared ViT encoder + one of {ModulatedSIREN, CrossAttnINR} decoder.
The encoder is identical across both ablation arms so all the FLOPs
difference lives in the decoder.

Decoder calling convention:
  - SIREN:    u(x) = decoder(z, x)             — does NOT consume tokens
  - XATTN:    state = decoder.prepare(z, T);   u(x) = decoder.query(state, x)

The AE exposes a `decode_points(params, u_input, x_query)` helper that
encodes once and decodes at M query points, doing the per-snapshot
work only once (this matters for the cross-attention decoder).
"""
from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from .encoder import ViTEncoder
from .decoders.modulated_siren import ModulatedSIREN
from .decoders.cross_attn_inr import CrossAttnINR


DecoderKind = Literal["siren", "xattn"]


class INRAutoencoder(nn.Module):
    decoder_kind: DecoderKind
    # Encoder.
    N: int
    spatial_dim: int = 2
    patch_size: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 16
    # Decoder common.
    coord_dim: int = 2
    hidden_dim: int = 256
    # SIREN-specific.
    siren_num_layers: int = 5
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 128
    # XATTN-specific.
    d_attn: int = 64
    num_fourier: int = 16
    xattn_num_layers: int = 3
    fourier_scale: float = 4.0

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
        if self.decoder_kind == "siren":
            self.decoder = ModulatedSIREN(
                coord_dim=self.coord_dim,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.siren_num_layers,
                omega_0=self.omega_0,
                omega=self.omega,
                modulator_hidden=self.modulator_hidden,
            )
        elif self.decoder_kind == "xattn":
            self.decoder = CrossAttnINR(
                coord_dim=self.coord_dim,
                latent_dim=self.latent_dim,
                embed_dim=self.embed_dim,
                d_attn=self.d_attn,
                num_fourier=self.num_fourier,
                hidden_dim=self.hidden_dim,
                num_layers=self.xattn_num_layers,
                fourier_scale=self.fourier_scale,
            )
        else:
            raise ValueError(f"unknown decoder_kind: {self.decoder_kind!r}")

    # ---- encoder convenience ----
    def encode(self, u_flat):
        """Returns (z, tokens). Tokens used only by xattn decoder."""
        return self.encoder(u_flat)

    # ---- decoder convenience ----
    def decode_one(self, z, tokens, x):
        """Single-coord decode. Used by tests and vmap callsites."""
        if self.decoder_kind == "siren":
            return self.decoder(z, x)
        else:
            state = self.decoder.prepare(z, tokens)
            return self.decoder.query(state, x)

    def decode_points(self, z, tokens, x_query):
        """Decode at M coords. Vectorised over coords.

        For xattn, prepare(z, tokens) is called ONCE and reused for all M
        queries (this is the cost-amortization story).

        x_query: (M, coord_dim)  ->  u: (M,)
        """
        if self.decoder_kind == "siren":
            # SIREN has no per-snapshot precompute beyond the modulator MLP,
            # which is consumed inside the decoder's __call__ — vmap handles it.
            return jax.vmap(lambda x: self.decoder(z, x))(x_query)
        else:
            state = self.decoder.prepare(z, tokens)
            return jax.vmap(lambda x: self.decoder.query(state, x))(x_query)

    def __call__(self, u_flat, x_query):
        """End-to-end: encode u, decode at M points."""
        z, tokens = self.encoder(u_flat)
        return self.decode_points(z, tokens, x_query)
