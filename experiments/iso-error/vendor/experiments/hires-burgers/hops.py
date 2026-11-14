"""hires-burgers: mesh-lean operators for the frozen Burgers bank at 2048^2 and 4096^2.

Nothing here ever materialises the (n, M) test matrix Phi. The weak test functions are the
M lowest discrete sine modes of `engines.modes`,

    Phi[(i, j), m] = (2 / L) sin(pi x_i kx_m) sin(pi y_j ky_m),

so a projection Phi^T v of a grid field v is the SEPARABLE product

    (Phi^T v)_m = (2 / L) sum_i sx[i, m] (V sy)[i, m],      V = v.reshape(L-1, L-1),

one (L-1, L-1) x (L-1, M) matmul. At 4096^2 and M = 1088 the dense Phi would be 146 GB; the
two sine tables are 36 MB each. `modes_lean` reproduces `engines.modes`'s selection (same
eigenvalue formula, same stable argsort) without the Phi array, and `phi_rows` evaluates the
rows of Phi at arbitrary nodes with the same floating-point expression. Parity against
`engines.modes` and `Phi.T @ v` is gated in the smoke and in every job at a small mesh.

Every large array is a jit ARGUMENT, never a closure.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e


# ------------------------------------------------------------------ modes ----

def modes_lean(L, M):
    """(kx, ky, lam) of `engines.modes(L, M)` without building Phi."""
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    return kx.ravel()[ind], ky.ravel()[ind], lam.ravel()[ind]


def sine_tables(L, kx, ky):
    """sx[i, m] = sin(pi x_i kx_m), sy[j, m] = sin(pi y_j ky_m), the expression of `engines.modes`."""
    x = np.arange(1, L) / L
    return np.sin(np.pi * x[:, None] * kx), np.sin(np.pi * x[:, None] * ky)


def phi_rows(L, kx, ky, ij):
    """Rows of Phi at grid nodes ij (1-based interior indices), shape (len(ij), M)."""
    ij = np.asarray(ij)
    return (2. / L) * np.sin(np.pi * (ij[:, 0] / L)[:, None] * kx) * np.sin(np.pi * (ij[:, 1] / L)[:, None] * ky)


def sep_project(V, sx, sy, L):
    """Phi^T v for v = V.reshape(-1); V is (..., L-1, L-1); returns (..., M)."""
    return (2. / L) * jnp.sum(sx * (V @ sy), axis=-2)


def unravel_nodes(nodes, L):
    return np.stack(np.unravel_index(np.asarray(nodes), (L - 1, L - 1)), 1) + 1


def ravel_nodes(ij, L):
    ij = np.asarray(ij)
    return np.ravel_multi_index((ij[:, 0] - 1, ij[:, 1] - 1), (L - 1, L - 1))


# ------------------------------------------------------------------- bank ----

MAX_GEMM_ELEMENTS = 2.146e9     # XLA's Triton gemm indexes with int32: hb4k02 (job 4059827) failed
                               # autotuning on f64[6, 16769025] x [512] = 8.6e9 elements


def build_bank(bank, L, nblocks=None):
    """G on the L-grid as a TUPLE of row blocks, each below the int32 gemm limit.

    hb4k01 died uploading one 64 GiB host array; hb4k02 built one 64 GiB device array and then
    every product with it failed Triton-gemm autotuning (more than 2^31 elements). Blocks are
    evaluated on the device by `bank.at` on the same points as `bank.on_grid`, never
    concatenated, never copied. `bank_apply` is the only way the driver multiplies by G."""
    n = (L - 1) ** 2
    nblocks = int(np.ceil(n * bank.dim / MAX_GEMM_ELEMENTS)) if nblocks is None else int(nblocks)
    if nblocks == 1:
        return (bank.on_grid(L),)
    x = np.arange(1, L) / L
    edges = np.linspace(0, L - 1, nblocks + 1).astype(int)
    out = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        xy = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
        out.append(jax.block_until_ready(bank.at(xy, chunk=8192)))
    return tuple(out)


def bank_apply(Gb, X):
    """G @ X for X of shape (R,) or (R, k), G a tuple of row blocks."""
    return jnp.concatenate([g @ X for g in Gb], axis=0)


def bank_rows(Gb):
    return int(sum(g.shape[0] for g in Gb))


def project_bank(G, sx, sy, L, cols=32):
    """A = Phi^T G by the separable projection, `cols` bank columns at a time."""
    f = jax.jit(lambda gs, a, b: sep_project(jnp.moveaxis(jnp.concatenate(gs, 0).reshape(L - 1, L - 1, -1), -1, 0), a, b, L))
    out = [f(tuple(g[:, s:s + cols] for g in G), sx, sy) for s in range(0, G[0].shape[1], cols)]
    return jnp.concatenate(out, axis=0).T                         # (M, R)


def advect_points(us, L):
    """The upwind advection of `engines.spatial` from five-point stencil values (.., 5)."""
    c, xp, xm, yp, ym = [us[..., i] for i in range(5)]
    return c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))


# ------------------------------------------------------------------ rules ----

def lattice_rule(L, s):
    """Uniform interior sub-lattice with s intervals per axis and equal weights (L/s)^2.

    It is the fine-grid sum restricted to every (L/s)-th line, i.e. the same composite rule on a
    coarser lattice: exact only on the discrete sine-product orthogonality class of that lattice,
    and the upwind branch switch is not band-limited at all, so it is certified empirically by
    held-out rho like any other rule. It needs no fit and no draw, and maps to any mesh with
    L % s == 0 at the same physical points.
    """
    assert L % s == 0
    k = np.arange(1, s) * (L // s)
    ii, jj = np.meshgrid(k, k, indexing='ij')
    ij = np.stack((ii.ravel(), jj.ravel()), 1)
    return ij, np.full(len(ij), float((L // s) ** 2))


def transfer_nodes(nodes_src, L_src, L):
    """The same physical points on a finer grid."""
    assert L % L_src == 0
    return unravel_nodes(nodes_src, L_src) * (L // L_src)


def rule_ops(bank, L, kx, ky, ij, w):
    keep = np.asarray(w) > 0
    ij, w = np.asarray(ij)[keep], np.asarray(w)[keep]
    return dict(G5=bank.stencil(ij, L), Pq=jnp.asarray(phi_rows(L, kx, ky, ij) * w[:, None])), ij, w


def dense_targets(G, coefficients, sx, sy, L, chunk=8):
    """Phi^T a(G c) for every coefficient row: the exact side of rho, (S, M)."""
    def one(c, g, a, b):
        return sep_project(e.spatial(bank_apply(g, c), L)[0].reshape(L - 1, L - 1), a, b, L)
    f = jax.jit(jax.vmap(one, in_axes=(0, None, None, None)))
    Cs = jnp.asarray(np.asarray(coefficients))
    return np.concatenate([np.asarray(f(Cs[s:s + chunk], G, sx, sy)) for s in range(0, len(Cs), chunk)])


def sampled_advection(G5, coefficients, L):
    """a(u)(x_j) at the rule's nodes for every state, (S, m), formed as `arms.weak_eq` forms it."""
    f = jax.jit(jax.vmap(lambda c, g5: advect_points(jnp.einsum('msr,r->ms', g5, c), L), in_axes=(0, None)))
    return np.asarray(f(jnp.asarray(np.asarray(coefficients)), G5))


def rho(Pq, adv_nodes, targets):
    """q-diag's rho per state, NumPy: || Pq^T a - Phi^T a || / || Phi^T a ||."""
    samp = np.asarray(adv_nodes) @ np.asarray(Pq)
    return np.linalg.norm(samp - targets, axis=1) / (np.linalg.norm(targets, axis=1) + 1e-300)


def rho_summary(v, bar, tight):
    v = np.asarray(v)
    return dict(states=int(len(v)), rho_max=float(v.max()), rho_p95=float(np.quantile(v, .95)),
                rho_median=float(np.median(v)), rho_mean=float(v.mean()), argmax=int(v.argmax()),
                certified_primary=bool(v.max() <= bar), certified_secondary=bool(np.quantile(v, .95) <= bar),
                certified_tight=bool(v.max() <= tight))


def refit_weights(Prow, adv_nodes, targets, scaling='state', max_seconds=None):
    """Nonnegative least squares for the weights on a FIXED support.

    Design rows (state s, mode m): Prow[j, m] a_s(x_j); targets Phi_m^T a_s. `scaling='state'`
    divides every row of state s by ||Phi^T a_s||, so the objective is sum_s rho_s^2 (b-eqtop's
    `rhow` arm); 'row' is the incumbent per-row unit scaling. The (S M, m) design is compressed
    exactly to (m, m) by a QR of [D | b] before the Lawson-Hanson solve, as b-eqtop does.
    The NNLS fit residual is recorded and is NEVER a certificate.
    """
    t0 = time.perf_counter()
    S, m = adv_nodes.shape
    M = Prow.shape[1]
    D = (adv_nodes[:, None, :] * Prow.T[None, :, :]).reshape(S * M, m)
    b = np.asarray(targets).reshape(S * M).copy()
    if scaling == 'state':
        sc = np.repeat(np.linalg.norm(targets, axis=1) + 1e-300, M)
    else:
        sc = np.linalg.norm(D, axis=1) + 1e-300
    D /= sc[:, None]
    b /= sc
    if D.shape[0] > D.shape[1] + 1:
        Rfull = np.asarray(jnp.linalg.qr(jnp.asarray(np.concatenate((D, b[:, None]), 1)), mode='r'))
        assert np.isfinite(Rfull).all(), 'QR compression returned non-finite values'
        Dc, bc = Rfull[:m, :m], Rfull[:m, m]
        tail = float(abs(Rfull[m, m])) if Rfull.shape[0] > m else 0.
    else:
        Dc, bc, tail = D, b, 0.
    w, _ = scipy.optimize.nnls(Dc, bc, maxiter=20 * m)
    fit = float(np.sqrt(np.linalg.norm(Dc @ w - bc) ** 2 + tail ** 2) / max(np.linalg.norm(b), 1e-300))
    return w, dict(relative_fit=fit, design_rows=int(S * M), support=int(m), nonzero_weights=int(np.sum(w > 0)),
                   fit_states=int(S), M=int(M), scaling=scaling, seconds=time.perf_counter() - t0,
                   note='NNLS fit residual; not a certificate')
