"""Modulated SIREN decoder.

Per-layer FiLM-style bias modulation of a sine-activation MLP. The
modulator network maps the latent `z` to a per-layer bias offset for
each hidden layer:

    beta = M(z)           # shape (L, h)
    h_0  = sin( omega_0 * ( W_in @ x + b_in + beta[0] ) )
    h_l  = sin( omega   * ( W_l  @ h_{l-1} + b_l + beta[l] ) )    l = 1..L-1
    u    = W_out @ h_{L-1} + b_out

We use bias-only modulation (Mehta et al. 2021 "Modulated PE") rather
than weight modulation because it's an order of magnitude cheaper and
empirically as good for parametric-PDE fits.

SIREN initialisation (Sitzmann et al. 2020):
    W_in:   U(-1/in_dim,         +1/in_dim)
    W_l:    U(-sqrt(6/h)/omega,  +sqrt(6/h)/omega)
    First layer scaled by omega_0 = 30; hidden by omega = 1 (i.e. no
    rescale beyond the SIREN-recommended sqrt(6/h)).

The decoder is vmapped over query points externally — `__call__`
returns the scalar u(x) for a single coord.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _siren_init(omega: float):
    """Sitzmann et al. SIREN uniform init scaled by sqrt(6/in)/omega."""
    def init(key, shape, dtype=jnp.float32):
        in_dim = shape[-2]
        bound = (6.0 / in_dim) ** 0.5 / omega
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def _first_layer_init(in_dim: int):
    """First-layer init: U(-1/in_dim, +1/in_dim)."""
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -1.0 / in_dim, +1.0 / in_dim)
    return init


class ModulatedSIREN(nn.Module):
    """Modulated SIREN INR. Apply on a single coord x; vmap externally for batches."""

    coord_dim: int = 2          # spatial dim
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 5         # total layers incl. first sine + L-2 hidden + 1 output
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 128

    @nn.compact
    def __call__(self, z, x):
        """z: (k,)   x: (coord_dim,)   -> u: scalar"""
        L = self.num_layers
        h = self.hidden_dim

        # Modulator network: 2-layer ReLU MLP, output (L-1, h) bias offsets
        # for the (L-1) sine layers (first + L-2 hidden); the output layer has
        # no nonlinearity and is not modulated.
        m = nn.Dense(self.modulator_hidden, name="mod_in")(z)
        m = nn.relu(m)
        m = nn.Dense((L - 1) * h, name="mod_out")(m)
        beta = m.reshape((L - 1, h))

        # First (input) sine layer.
        W_in = self.param(
            "W_in",
            _first_layer_init(self.coord_dim),
            (self.coord_dim, h),
        )
        b_in = self.param("b_in", nn.initializers.zeros, (h,))
        a = self.omega_0 * (x @ W_in + b_in + beta[0])
        out = jnp.sin(a)

        # Hidden sine layers (L-2 of them).
        for l in range(1, L - 1):
            W_l = self.param(
                f"W_{l}",
                _siren_init(self.omega),
                (h, h),
            )
            b_l = self.param(f"b_{l}", nn.initializers.zeros, (h,))
            a = self.omega * (out @ W_l + b_l + beta[l])
            out = jnp.sin(a)

        # Output layer: linear -> scalar.
        W_out = self.param(
            "W_out",
            _siren_init(self.omega),
            (h, 1),
        )
        b_out = self.param("b_out", nn.initializers.zeros, (1,))
        u = (out @ W_out + b_out).squeeze(-1)
        return u
