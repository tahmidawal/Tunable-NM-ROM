"""Cross-attention INR decoder.

The query coord x cross-attends into the ViT encoder's per-token features T.
Then a small post-attention MLP combines the attended context, a Fourier
positional encoding of x, and the global latent z to produce u(x).

    q       = W_q @ gamma(x)
    K_tok   = W_k @ T                  # once per snapshot
    V_tok   = W_v @ T                  # once per snapshot
    attn    = softmax( q . K_tok^T / sqrt(d_q) )
    ctx     = attn . V_tok
    h_0     = [gamma(x); ctx; z]
    h_l     = gelu( W_l @ h_{l-1} + b_l ),   l = 1..L-1
    u       = w_out @ h_{L-1}

Fourier features gamma(x) = [sin(2 pi B x), cos(2 pi B x)] with B
random-normal frozen — standard practice. Single-head attention for
the first pass to keep the cost-comparison clean.

K_tok and V_tok depend only on the snapshot's tokens, NOT on x — so
they should be computed ONCE per snapshot and re-used for all query
points. The training loop achieves this via `prepare(z, tokens)` +
`query(state, x)`. We use setup() with named submodules (Flax forbids
multiple @nn.compact methods on the same class).
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class _SnapshotState(NamedTuple):
    K: jnp.ndarray   # (n_tok, d_attn)
    V: jnp.ndarray   # (n_tok, d_attn)
    z: jnp.ndarray   # (k,)


class CrossAttnINR(nn.Module):
    coord_dim: int = 2
    latent_dim: int = 16
    embed_dim: int = 64           # ViT token embed dim
    d_attn: int = 64              # K/V projected dim
    num_fourier: int = 16
    hidden_dim: int = 256
    num_layers: int = 3           # post-attention MLP depth
    fourier_scale: float = 4.0    # std of B

    def setup(self):
        self.W_k = nn.Dense(self.d_attn, name="W_k", use_bias=False)
        self.W_v = nn.Dense(self.d_attn, name="W_v", use_bias=False)
        self.W_q = nn.Dense(self.d_attn, name="W_q", use_bias=False)
        self.mlp_hidden = [
            nn.Dense(self.hidden_dim, name=f"mlp_{l}")
            for l in range(self.num_layers - 1)
        ]
        self.mlp_out = nn.Dense(1, name="mlp_out")
        # Fourier-feature matrix (learnable). Registered in setup() so it
        # can be referenced inside query() without violating Flax's
        # "params only in setup or @compact" rule.
        self.B_fourier = self.param(
            "B_fourier",
            nn.initializers.normal(stddev=self.fourier_scale),
            (self.num_fourier, self.coord_dim),
        )

    def prepare(self, z, tokens):
        """Compute snapshot-level state (K, V, z) once per snapshot."""
        K = self.W_k(tokens)
        V = self.W_v(tokens)
        return _SnapshotState(K=K, V=V, z=z)

    def query(self, state: _SnapshotState, x):
        """x: (coord_dim,)   -> u: scalar"""
        proj = 2.0 * jnp.pi * (self.B_fourier @ x)             # (F,)
        gamma = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)  # (2F,)

        q = self.W_q(gamma)                                    # (d_attn,)

        scores = state.K @ q / jnp.sqrt(self.d_attn)           # (n_tok,)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = attn @ state.V                                   # (d_attn,)

        h = jnp.concatenate([gamma, ctx, state.z], axis=-1)
        for l, dense in enumerate(self.mlp_hidden):
            h = dense(h)
            h = nn.gelu(h)
        u = self.mlp_out(h).squeeze(-1)
        return u

    def __call__(self, z, tokens, x):
        state = self.prepare(z, tokens)
        return self.query(state, x)
