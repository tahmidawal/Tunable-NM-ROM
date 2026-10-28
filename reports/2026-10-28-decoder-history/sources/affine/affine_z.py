"""Affine-in-z decoder.

   u(x; z) = Phi(x) @ A(z) + b(x)

   - Phi(x) in R^h is a fixed nonlinear feature trunk (SIREN-MLP from x only)
   - A(z)   in R^h is a 2-layer MLP from latent to feature-coefficient vector
   - b(x)   in R^h is a learnable bias field (single Dense from x)

The Jacobian wrt z factors cleanly:
   du/dz = Phi(x) @ dA/dz      shape (latent,)

For an NM-ROM solve, Phi(x_eq) can be precomputed ONCE per solve over the
EQ stencil, so per-iter cost = dA/dz (tiny) + matmul (Phi_eq @ dA/dz).
This is the CP-decoder-style asymptote with INR-style expressivity in
the encoder.

This module mirrors ModulatedSIREN's API: __call__(z, x) -> scalar.
Vmap is handled by decode_points in INRAutoencoder.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _siren_init(omega: float):
    def init(key, shape, dtype=jnp.float32):
        in_dim = shape[-2]
        bound = (6.0 / in_dim) ** 0.5 / omega
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def _first_layer_init(in_dim: int):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -1.0 / in_dim, +1.0 / in_dim)
    return init


class AffineZDecoder(nn.Module):
    """u(x; z) = Phi(x) @ A(z) + b(x).

    Args mirror ModulatedSIREN where possible so we can share configs.
    Different roles:
      - hidden_dim          : feature dim of Phi(x) (= rank of basis)
      - num_layers          : depth of Phi SIREN-MLP
      - omega_0/omega       : SIREN init schedule for Phi
      - modulator_hidden    : hidden width of A(z) MLP
      - latent_dim          : input dim of A(z)
    """

    coord_dim: int = 2
    latent_dim: int = 16
    hidden_dim: int = 256        # feature dim h
    num_layers: int = 4          # Phi SIREN depth (total layers incl. first sine)
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 128
    bias_hidden: int = 128       # b(x) SIREN width

    def setup(self):
        # Phi(x) SIREN trunk parameters.
        self._phi_first_W = self.param(
            "phi_W_in", _first_layer_init(self.coord_dim),
            (self.coord_dim, self.hidden_dim),
        )
        self._phi_first_b = self.param(
            "phi_b_in", nn.initializers.zeros, (self.hidden_dim,),
        )
        self._phi_W = [
            self.param(f"phi_W_{l}", _siren_init(self.omega),
                       (self.hidden_dim, self.hidden_dim))
            for l in range(1, self.num_layers)
        ]
        self._phi_b = [
            self.param(f"phi_b_{l}", nn.initializers.zeros, (self.hidden_dim,))
            for l in range(1, self.num_layers)
        ]
        # b(x) trunk: small SIREN MLP from x to scalar.
        # Shape: (coord_dim, bias_hidden) -> (bias_hidden, bias_hidden) -> (bias_hidden, 1)
        self._b_W_in = self.param(
            "bias_W_in", _first_layer_init(self.coord_dim),
            (self.coord_dim, self.bias_hidden),
        )
        self._b_b_in = self.param(
            "bias_b_in", nn.initializers.zeros, (self.bias_hidden,),
        )
        self._b_W_mid = self.param(
            "bias_W_mid", _siren_init(self.omega),
            (self.bias_hidden, self.bias_hidden),
        )
        self._b_b_mid = self.param(
            "bias_b_mid", nn.initializers.zeros, (self.bias_hidden,),
        )
        self._b_W_out = self.param(
            "bias_W_out", _siren_init(self.omega),
            (self.bias_hidden, 1),
        )
        self._b_b_out = self.param(
            "bias_b_out", nn.initializers.zeros, (1,),
        )
        # A(z) coefficient MLP: latent -> modulator_hidden -> hidden_dim.
        self._A_W0 = self.param(
            "A_W0", nn.initializers.lecun_normal(),
            (self.latent_dim, self.modulator_hidden),
        )
        self._A_b0 = self.param(
            "A_b0", nn.initializers.zeros, (self.modulator_hidden,),
        )
        self._A_W1 = self.param(
            "A_W1", nn.initializers.lecun_normal(),
            (self.modulator_hidden, self.hidden_dim),
        )
        self._A_b1 = self.param(
            "A_b1", nn.initializers.zeros, (self.hidden_dim,),
        )

    def phi(self, x):
        """Return Phi(x) in R^hidden_dim. Independent of z."""
        a = self.omega_0 * (x @ self._phi_first_W + self._phi_first_b)
        h = jnp.sin(a)
        for W, b in zip(self._phi_W, self._phi_b):
            h = jnp.sin(self.omega * (h @ W + b))
        return h

    def bias(self, x):
        """Learnable scalar bias field b(x): a 2-hidden-layer SIREN."""
        a = self.omega_0 * (x @ self._b_W_in + self._b_b_in)
        h = jnp.sin(a)
        h = jnp.sin(self.omega * (h @ self._b_W_mid + self._b_b_mid))
        return (h @ self._b_W_out + self._b_b_out).squeeze(-1)

    def coef(self, z):
        """A(z) in R^hidden_dim."""
        m = jnp.dot(z, self._A_W0) + self._A_b0
        m = nn.relu(m)
        m = jnp.dot(m, self._A_W1) + self._A_b1
        return m

    def __call__(self, z, x):
        """u(x; z) scalar."""
        return jnp.dot(self.phi(x), self.coef(z)) + self.bias(x)
