"""Default-field Modulated SIREN.

D(z, x) = u_default(x) + ModulatedSIREN(z, x)

`u_default(x)` is a small unmodulated SIREN trunk with no `z` input. Its
sole job is to fit the *mean* training solution field, so that when
z = 0 the decoder produces a sensible field (the mean, not garbage).
ModulatedSIREN then provides the parameter-dependent residual.

This pairs naturally with a variational encoder whose latent prior is
N(0, I): training pushes ‖z_train‖ → 1, and at ROM time cold-start z = 0
is a typical latent, decoding to the mean field.

The ModulatedSIREN body is bit-identical to decoders.modulated_siren.
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


class DefaultFieldSIREN(nn.Module):
    """ModulatedSIREN + a learned default-field trunk."""

    coord_dim: int = 2
    latent_dim: int = 16
    hidden_dim: int = 384
    num_layers: int = 5
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 192

    # Default-field network u_default(x): independent of z.
    default_hidden_dim: int = 128
    default_num_layers: int = 4
    default_omega_0: float = 15.0

    @nn.compact
    def __call__(self, z, x):
        L = self.num_layers
        h = self.hidden_dim

        # ---- (a) ModulatedSIREN body (bit-identical to modulated_siren) ----
        m = nn.Dense(self.modulator_hidden, name="mod_in")(z)
        m = nn.relu(m)
        m = nn.Dense((L - 1) * h, name="mod_out")(m)
        beta = m.reshape((L - 1, h))

        W_in = self.param(
            "W_in", _first_layer_init(self.coord_dim), (self.coord_dim, h),
        )
        b_in = self.param("b_in", nn.initializers.zeros, (h,))
        a = self.omega_0 * (x @ W_in + b_in + beta[0])
        out = jnp.sin(a)
        for l in range(1, L - 1):
            W_l = self.param(f"W_{l}", _siren_init(self.omega), (h, h))
            b_l = self.param(f"b_{l}", nn.initializers.zeros, (h,))
            a = self.omega * (out @ W_l + b_l + beta[l])
            out = jnp.sin(a)
        W_out = self.param("W_out", _siren_init(self.omega), (h, 1))
        b_out = self.param("b_out", nn.initializers.zeros, (1,))
        u_mod = (out @ W_out + b_out).squeeze(-1)

        # ---- (b) Default-field SIREN trunk u_default(x), no z input ----
        Ld = self.default_num_layers
        hd = self.default_hidden_dim
        Wd0 = self.param(
            "udef_W0",
            _first_layer_init(self.coord_dim),
            (self.coord_dim, hd),
        )
        bd0 = self.param("udef_b0", nn.initializers.zeros, (hd,))
        a_d = self.default_omega_0 * (x @ Wd0 + bd0)
        out_d = jnp.sin(a_d)
        for l in range(1, Ld - 1):
            Wdl = self.param(f"udef_W{l}", _siren_init(self.omega), (hd, hd))
            bdl = self.param(f"udef_b{l}", nn.initializers.zeros, (hd,))
            a_d = self.omega * (out_d @ Wdl + bdl)
            out_d = jnp.sin(a_d)
        Wd_out = self.param("udef_Wout", _siren_init(self.omega), (hd, 1))
        bd_out = self.param("udef_bout", nn.initializers.zeros, (1,))
        u_def = (out_d @ Wd_out + bd_out).squeeze(-1)

        return u_def + u_mod
