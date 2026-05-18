"""Variational + default-field INR autoencoder.

Encoder: same ViT trunk as `ViTEncoder`, but with two heads producing
`mu` and `logvar`. During training, `z = mu + sigma * eps` with
`sigma = exp(0.5 * logvar)` and `eps ~ N(0, I)`. A KL(N(mu, sigma^2) ||
N(0, I)) penalty is applied so that training latents concentrate near 0.

Decoder: `DefaultFieldSIREN`. Decode(z=0, x) ≈ u_default(x), the learned
mean training field — this is precisely the cold-start prior for NM-ROM.

Calling conventions match `INRAutoencoder`'s `decode_points(z, tokens, x)`
so the existing ROM solver works unchanged. For variational, `tokens` is
always None (we still keep the slot in the signature for compatibility).

`encode(u_flat)` returns `(z, tokens, mu, logvar)`. For ROM-time warm-
start, use `z = mu` (set via the convenience method `encode_mu`).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .encoder import TransformerBlock
from .decoders.default_field_siren import DefaultFieldSIREN


class VariationalViTEncoder(nn.Module):
    N: int
    spatial_dim: int = 2
    patch_size: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    latent_dim: int = 16

    def setup(self):
        assert self.N % self.patch_size == 0, "patch_size must divide N"
        self.n_per_side = self.N // self.patch_size
        self.num_patches = self.n_per_side ** self.spatial_dim
        self.patch_features = self.patch_size ** self.spatial_dim

    def _trunk(self, u_flat):
        u = jnp.reshape(u_flat, (self.N,) * self.spatial_dim)
        if self.spatial_dim == 2:
            patches = u.reshape(
                self.n_per_side, self.patch_size,
                self.n_per_side, self.patch_size,
            ).transpose(0, 2, 1, 3).reshape(self.num_patches, self.patch_features)
        else:
            raise ValueError(f"spatial_dim must be 2, got {self.spatial_dim}")
        tokens = nn.Dense(self.embed_dim, name="patch_embed")(patches)
        pos = self.param(
            "pos_embed", nn.initializers.normal(stddev=0.02),
            (self.num_patches, self.embed_dim),
        )
        tokens = tokens + pos
        for i in range(self.num_layers):
            tokens = TransformerBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, name=f"block_{i}",
            )(tokens)
        tokens = nn.LayerNorm(name="trunk_norm")(tokens)
        return tokens

    @nn.compact
    def __call__(self, u_flat):
        tokens = self._trunk(u_flat)
        pooled = tokens.mean(axis=0)
        mu = nn.Dense(self.latent_dim, name="head_mu")(pooled)
        # Initialise logvar near a slightly-negative value (small variance)
        # so the very first epochs aren't dominated by noise. bias_init=0,
        # kernel_init small.
        logvar = nn.Dense(
            self.latent_dim,
            name="head_logvar",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-2.0),  # sigma ≈ 0.37 at init
        )(pooled)
        return mu, logvar, tokens


class INRAutoencoderVDF(nn.Module):
    """Variational ViT encoder + DefaultFieldSIREN decoder."""

    N: int
    spatial_dim: int = 2
    patch_size: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 16
    coord_dim: int = 2

    # SIREN body.
    hidden_dim: int = 384
    siren_num_layers: int = 5
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 192
    # Default-field trunk.
    default_hidden_dim: int = 128
    default_num_layers: int = 4
    default_omega_0: float = 15.0

    # Kept only so the AE has the same fields as INRAutoencoder for cfg parsing.
    decoder_kind: str = "siren_vdf"

    def setup(self):
        self.encoder = VariationalViTEncoder(
            N=self.N, spatial_dim=self.spatial_dim,
            patch_size=self.patch_size, embed_dim=self.embed_dim,
            num_heads=self.num_heads, num_layers=self.num_enc_layers,
            latent_dim=self.latent_dim,
        )
        self.decoder = DefaultFieldSIREN(
            coord_dim=self.coord_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.siren_num_layers,
            omega_0=self.omega_0,
            omega=self.omega,
            modulator_hidden=self.modulator_hidden,
            default_hidden_dim=self.default_hidden_dim,
            default_num_layers=self.default_num_layers,
            default_omega_0=self.default_omega_0,
        )

    # ---- encoder accessors ----
    def encode(self, u_flat):
        """Variational encode. Returns (mu, logvar, tokens)."""
        mu, logvar, tokens = self.encoder(u_flat)
        return mu, logvar, tokens

    def encode_mu(self, u_flat):
        """ROM-time warm-start: deterministic latent (= mu)."""
        mu, _logvar, _tokens = self.encoder(u_flat)
        return mu

    # ---- decoder accessors ----
    def decode_points(self, z, tokens, x_query):
        """Decode at M coords. SIREN ignores `tokens`."""
        return jax.vmap(lambda x: self.decoder(z, x))(x_query)

    def decode_one(self, z, tokens, x):
        return self.decoder(z, x)

    # ---- end-to-end (training path; uses provided rng for reparam noise) ----
    def __call__(self, u_flat, x_query, rng=None, training=False):
        mu, logvar, tokens = self.encoder(u_flat)
        if training and rng is not None:
            eps = jax.random.normal(rng, mu.shape)
            sigma = jnp.exp(0.5 * logvar)
            z = mu + sigma * eps
        else:
            z = mu
        u = jax.vmap(lambda x: self.decoder(z, x))(x_query)
        return u, mu, logvar


def kl_divergence(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ) summed across latent dims.

    KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar ).
    """
    return 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar)
