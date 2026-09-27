"""burgers-heldout: build a rank-R' mesh-free bank inside the span of a multi-block learned bank.

A block (bank-floor `bf_core.bank_block`) is dict(B, g, out_scale); its features are
`sep_common.features` = out_scale * bc(x) * mlp_g([sin 2pi xB, cos 2pi xB]). A multi-block bank's
features are the concatenation over blocks. `merge_blocks(blocks, V)` returns ONE separable-decoder
parameter set whose features equal concat_b(features_b) @ V exactly (block-diagonal hidden layers,
per-block out_scale and V folded into the last linear layer, merged out_scale = 1), so every existing
module that assumes one block (`sep_common`, `arms.CoordBank`, `hops`, `hfast`) runs unchanged.

`pod_directions` is the field-metric POD inside the span: with Gram = G^T G = L L^T, the whitened
coefficients a = L^T c satisfy ||G c|| = ||a||, Q = G L^{-T} is orthonormal, and the rank-r compressed
bank is G' = G V with V = s L^{-T} W, W the top-r right singular vectors of the weighted whitened rows.
"""
from __future__ import annotations

import hashlib

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import sep_common as sc

F64 = jnp.float64


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def block_features(blocks, xy):
    xy = jnp.asarray(xy, F64)
    return jnp.concatenate([sc.features(b, xy) for b in blocks], axis=1)


def merge_blocks(blocks, V=None):
    """One separable parameter set with features == concat_b(features_b) @ V (V=None: identity)."""
    nl = len(blocks[0]['g'])
    assert all(len(b['g']) == nl for b in blocks) and nl >= 2
    nff = [int(np.shape(b['B'])[1]) for b in blocks]
    NF = sum(nff)
    B = np.concatenate([np.asarray(b['B'], float) for b in blocks], axis=1)
    g = []
    # first layer: merged input is [sin(all NF), cos(all NF)]; block b reads its own sin and cos rows
    hs = [int(np.shape(b['g'][0][0])[1]) for b in blocks]
    W0 = np.zeros((2 * NF, sum(hs)))
    fo, ho = 0, 0
    for b, f_, h_ in zip(blocks, nff, hs):
        W = np.asarray(b['g'][0][0], float)
        assert W.shape == (2 * f_, h_)
        W0[fo:fo + f_, ho:ho + h_] = W[:f_]
        W0[NF + fo:NF + fo + f_, ho:ho + h_] = W[f_:]
        fo += f_
        ho += h_
    g.append((W0, np.concatenate([np.asarray(b['g'][0][1], float) for b in blocks])))
    for layer in range(1, nl - 1):            # hidden layers: block diagonal
        ins = [int(np.shape(b['g'][layer][0])[0]) for b in blocks]
        outs = [int(np.shape(b['g'][layer][0])[1]) for b in blocks]
        W = np.zeros((sum(ins), sum(outs)))
        io, oo = 0, 0
        for b, i_, o_ in zip(blocks, ins, outs):
            W[io:io + i_, oo:oo + o_] = np.asarray(b['g'][layer][0], float)
            io += i_
            oo += o_
        g.append((W, np.concatenate([np.asarray(b['g'][layer][1], float) for b in blocks])))
    ins = [int(np.shape(b['g'][-1][0])[0]) for b in blocks]
    outs = [int(np.shape(b['g'][-1][0])[1]) for b in blocks]
    Wl = np.zeros((sum(ins), sum(outs)))
    bl = np.zeros(sum(outs))
    io, oo = 0, 0
    for b, i_, o_ in zip(blocks, ins, outs):       # per-block out_scale folded into the last layer
        s = float(np.asarray(b['out_scale']))
        Wl[io:io + i_, oo:oo + o_] = s * np.asarray(b['g'][-1][0], float)
        bl[oo:oo + o_] = s * np.asarray(b['g'][-1][1], float)
        io += i_
        oo += o_
    if V is not None:
        V = np.asarray(V, float)
        assert V.shape[0] == sum(outs)
        Wl, bl = Wl @ V, bl @ V
    g.append((Wl, bl))
    return dict(B=B, g=g, out_scale=np.asarray(1.0))


def to_device(p):
    return dict(B=jnp.asarray(p['B'], F64), g=[(jnp.asarray(w, F64), jnp.asarray(b, F64)) for w, b in p['g']],
                out_scale=jnp.asarray(p['out_scale'], F64))


def merge_parity(blocks, merged, V=None, n=4096, seed=20260921):
    """max relative deviation (Frobenius, and worst column) of merged features vs concat @ V."""
    xy = np.random.default_rng(seed).uniform(0., 1., (n, 2))
    g = np.linspace(0., 1., 65)                 # grid nodes of several meshes, boundary included
    xy = np.concatenate([xy, np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2),
                         np.stack(np.meshgrid(np.arange(1, 256) / 256, [0., 1.], indexing='ij'), -1).reshape(-1, 2)])
    n = len(xy)
    ref = block_features([to_device(b) for b in blocks], xy)
    if V is not None:
        ref = ref @ jnp.asarray(V, F64)
    got = sc.features(to_device(merged), jnp.asarray(xy, F64))
    rel = float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref))
    col = float(jnp.max(jnp.linalg.norm(got - ref, axis=0) / jnp.maximum(jnp.linalg.norm(ref, axis=0), 1e-300)))
    return dict(points=n, seed=seed, relative_frobenius=rel, worst_column_relative=col,
                passed=bool(rel <= 1e-12 and col <= 1e-10))


def chol_whitener(Gram):
    """The jittered Cholesky the extraction and the head fit use (1e-12 trace / R)."""
    Gram = jnp.asarray(Gram, F64)
    R = Gram.shape[0]
    eps = 1e-12 * jnp.trace(Gram) / R
    return jnp.linalg.cholesky(Gram + eps * jnp.eye(R, dtype=F64))


def pod_directions(C, Lc, weights, rmax):
    """Top-rmax right singular vectors of diag(w) C L (the weighted whitened coefficient rows).

    QR first (the (S, R) matrix is tall), then an SVD of the triangular factor: no Gram squaring."""
    A = (jnp.asarray(C, F64) @ Lc) * jnp.asarray(weights, F64)[:, None]
    Rf = jnp.linalg.qr(A, mode='r')
    assert bool(jnp.all(jnp.isfinite(Rf))), 'non-finite QR factor'
    _, sv, Vt = jnp.linalg.svd(Rf, full_matrices=False)
    return np.asarray(Vt[:rmax].T), np.asarray(sv)


def compressed_V(Lc, W, s):
    """V = s L^{-T} W: G V = s (G L^{-T}) W has orthogonal columns of norm s in the fit metric."""
    return s * np.asarray(jax.scipy.linalg.solve_triangular(Lc.T, jnp.asarray(W, F64), lower=False))


def span_floor(G, U):
    """||u - P_G u|| per row of U, P_G the orthogonal projector onto span(G), by a guarded thin QR."""
    Q, _ = jnp.linalg.qr(jnp.asarray(G, F64), mode='reduced')
    Q, _ = jnp.linalg.qr(Q, mode='reduced')          # twice is enough
    U = jnp.asarray(U, F64)
    P = (U @ Q) @ Q.T
    return np.asarray(jnp.linalg.norm(U - P, axis=1))
