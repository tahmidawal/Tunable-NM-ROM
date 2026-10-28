"""Linear-skip Modulated SIREN.

Architecture (single coord x, latent z):

    u(z, x) = u_default(x) + sum_j z_j * g_j(x) + ModulatedSIREN(z, x)

where:
  - `u_default(x)` is a tiny SIREN (no z input) — the learned mean field.
    Decode(0, x) = u_default(x) + g(x) @ 0 + ModulatedSIREN(0, x)
                 ≈ u_default(x) + (modulator bias path).
  - `g(x): R^d -> R^k` is a small SIREN that outputs `k = latent_dim`
    spatial basis functions. The bilinear term z @ g(x) gives the decoder
    a **linear backbone** in z, so dD/dz at z=0 is at least rank k along
    those directions. This is the structural anchor that makes GN have a
    descent direction from z=0.
  - The ModulatedSIREN body is identical to decoders.modulated_siren.

The "default field" plus "linear backbone" jointly eliminate the two
failure modes diagnosed for cold-start NM-ROM:
  (i)  decode(0) is junk (large residual from a meaningless field),
  (ii) dD/dz at z=0 is rank-deficient (GN has no descent direction).
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


def _siren_stack(x, hidden_dim, num_layers, omega_0, omega, out_dim, name_prefix):
    """A plain (unmodulated) SIREN trunk for u_default(x) and g(x).

    Returns shape (out_dim,) for x: (coord_dim,).
    """
    L = num_layers
    h = hidden_dim
    coord_dim = x.shape[-1]
    a = x
    # First sine layer
    W_in = nn.Dense(
        h, use_bias=True, name=f"{name_prefix}_W0",
        kernel_init=_first_layer_init(coord_dim),
        bias_init=nn.initializers.zeros,
    )(a)
    out = jnp.sin(omega_0 * W_in)
    # Hidden sine layers
    for l in range(1, L - 1):
        a = nn.Dense(
            h, use_bias=True, name=f"{name_prefix}_W{l}",
            kernel_init=_siren_init(omega),
            bias_init=nn.initializers.zeros,
        )(out)
        out = jnp.sin(omega * a)
    # Linear output
    out = nn.Dense(
        out_dim, use_bias=True, name=f"{name_prefix}_Wout",
        kernel_init=_siren_init(omega),
        bias_init=nn.initializers.zeros,
    )(out)
    return out  # (out_dim,)


class LinearSkipSIREN(nn.Module):
    """Linear-skip modulated SIREN. Apply on a single coord x; vmap externally."""

    coord_dim: int = 2
    latent_dim: int = 16
    hidden_dim: int = 384
    num_layers: int = 5
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 192

    # Default-field network (no z input).
    default_hidden_dim: int = 128
    default_num_layers: int = 4
    default_omega_0: float = 15.0

    # Linear-skip basis network g(x): R^d -> R^latent_dim.
    basis_hidden_dim: int = 128
    basis_num_layers: int = 4
    basis_omega_0: float = 15.0

    @nn.compact
    def __call__(self, z, x):
        L = self.num_layers
        h = self.hidden_dim

        # ---- (a) Modulated SIREN body (same as ModulatedSIREN) ----
        m = nn.Dense(self.modulator_hidden, name="mod_in")(z)
        m = nn.relu(m)
        m = nn.Dense((L - 1) * h, name="mod_out")(m)
        beta = m.reshape((L - 1, h))

        W_in = self.param(
            "W_in",
            _first_layer_init(self.coord_dim),
            (self.coord_dim, h),
        )
        b_in = self.param("b_in", nn.initializers.zeros, (h,))
        a = self.omega_0 * (x @ W_in + b_in + beta[0])
        out = jnp.sin(a)
        for l in range(1, L - 1):
            W_l = self.param(
                f"W_{l}", _siren_init(self.omega), (h, h),
            )
            b_l = self.param(f"b_{l}", nn.initializers.zeros, (h,))
            a = self.omega * (out @ W_l + b_l + beta[l])
            out = jnp.sin(a)
        W_out = self.param("W_out", _siren_init(self.omega), (h, 1))
        b_out = self.param("b_out", nn.initializers.zeros, (1,))
        u_mod = (out @ W_out + b_out).squeeze(-1)

        # ---- (b) Default-field network u_default(x), no z dependence ----
        u_def = _siren_stack(
            x,
            hidden_dim=self.default_hidden_dim,
            num_layers=self.default_num_layers,
            omega_0=self.default_omega_0,
            omega=self.omega,
            out_dim=1,
            name_prefix="udef",
        ).squeeze(-1)

        # ---- (c) Linear-skip basis g(x): R^d -> R^latent_dim ----
        g = _siren_stack(
            x,
            hidden_dim=self.basis_hidden_dim,
            num_layers=self.basis_num_layers,
            omega_0=self.basis_omega_0,
            omega=self.omega,
            out_dim=self.latent_dim,
            name_prefix="gbasis",
        )  # (latent_dim,)

        # Linear skip term: scalar = z @ g(x). Cheap, dD/dz includes g(x).
        # We also give it a learnable global scale so it can decay if the
        # network finds it un-useful.
        skip_scale = self.param(
            "skip_scale", nn.initializers.constant(1.0), (),
        )
        u_skip = skip_scale * jnp.dot(z, g)

        return u_def + u_skip + u_mod
