"""Per-architecture forward-FLOPs counter (decoder only).

Counts a matmul (m,k)*(k,n) as 2*m*k*n FLOPs. Pointwise non-linearities
counted as 1 FLOP / element. Softmax counted as 3*N. Used to anchor the
fixed-cost comparison plot.

We split "snapshot-amortized" cost (paid once per encoded snapshot,
divided across M query points) from "per-query" cost. Total cost per
query at batch size M is

    total_per_query(M) = amortized / M + per_query
"""
from __future__ import annotations

from dataclasses import dataclass


def _matmul(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


@dataclass
class FlopsBreakdown:
    amortized: int    # per snapshot
    per_query: int    # per query point

    def total_per_query(self, M: int) -> float:
        return self.amortized / M + self.per_query


def siren_flops(
    coord_dim: int = 2,
    latent_dim: int = 16,
    hidden_dim: int = 256,
    num_layers: int = 5,
    modulator_hidden: int = 128,
) -> FlopsBreakdown:
    h = hidden_dim
    L = num_layers
    # Modulator MLP: latent_dim -> modulator_hidden -> (L-1)*h
    mod_amort = (
        _matmul(1, latent_dim, modulator_hidden)
        + modulator_hidden                                  # relu
        + _matmul(1, modulator_hidden, (L - 1) * h)
    )
    # Per query: first layer (coord_dim -> h), then (L-2) hidden (h -> h),
    # then output (h -> 1), plus sin per hidden activation.
    per_q = (
        _matmul(1, coord_dim, h) + 2 * h                    # +beta, sin
        + (L - 2) * (_matmul(1, h, h) + 2 * h)             # +beta, sin
        + _matmul(1, h, 1) + 1                             # output bias
    )
    return FlopsBreakdown(amortized=mod_amort, per_query=per_q)


def xattn_flops(
    coord_dim: int = 2,
    latent_dim: int = 16,
    embed_dim: int = 64,
    d_attn: int = 64,
    num_fourier: int = 16,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_tokens: int = 64,  # n_per_side**d. With N=128, patch=16 -> 8*8=64 in 2D.
) -> FlopsBreakdown:
    h = hidden_dim
    F = num_fourier
    # K, V projections of all tokens (per snapshot).
    amort = (
        _matmul(num_tokens, embed_dim, d_attn)         # W_k @ T
        + _matmul(num_tokens, embed_dim, d_attn)       # W_v @ T
    )
    # Per query: gamma(x) (2F sin + 2F cos), W_q @ gamma, q.K, softmax, attn.V, MLP.
    gamma_dim = 2 * F
    in_dim = gamma_dim + d_attn + latent_dim
    per_q = (
        _matmul(1, coord_dim, F)                         # B @ x
        + 2 * F                                            # sin + cos
        + _matmul(1, gamma_dim, d_attn)                  # W_q
        + _matmul(1, d_attn, num_tokens)                 # q . K^T  (1xD * Dx N_tok)
        + 3 * num_tokens                                  # softmax
        + _matmul(1, num_tokens, d_attn)                 # attn . V
        + _matmul(1, in_dim, h) + h                       # mlp_0 + gelu
        + (num_layers - 2) * (_matmul(1, h, h) + h)      # mid layers
        + _matmul(1, h, 1)                                # mlp_out
    )
    return FlopsBreakdown(amortized=amort, per_query=per_q)
