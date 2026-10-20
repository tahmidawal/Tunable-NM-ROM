"""jcp-speed3d streaming ROM tables at mesh n (DESIGN A1.9 as corrected by A2.4).

A_{m,i} = sum_{a,b,c=1}^{n-2} phi_{kx}(a) phi_{ky}(b) phi_{kz}(c) G_i((a,b,c)/(n-1)),  phi_k(j) = sqrt(2/(n-1)) sin(pi k j/(n-1)),
modes (one-based) and eigenvalues from common.modes(n, complete_M(n, 4R')). Per x-slab a: G at the slab's (n-2)^2 nodes
(chunked by rows), contracted over z then y with the sine tables restricted to the needed k, accumulated into
T (Kx, Ky, Kz, R') with weight phi_kx(a); finally A_{m,:} = T[kx-1, ky-1, kz-1, :]. Nothing of size N x R' is stored.
Gate G5 (`check65`): at n = 65 equal to tables.build_tables (kxyz, M, lam identical; A to 1e-11 relative).
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from c3common import C, TB, OM


def sine_table(n, K):
    a = np.arange(1, n - 1)[:, None]
    k = np.arange(1, K + 1)[None, :]
    return np.sqrt(2.0 / (n - 1)) * np.sin(np.pi * a * k / (n - 1))          # (n-2, K): column j-1 = mode j


def modes(n, Rp):
    order, lam_all = C.mode_order(n)
    M = int(C.complete_M(n, 4 * Rp, order, lam_all))
    kxyz, lam = C.modes(n, M)
    return M, np.asarray(kxyz), np.asarray(lam)


def build_A(n, bank, T, Rp, log=print, row_chunk=1 << 15):
    """(A (M, R'), lam (M,), kxyz (M, 3), M, seconds) at mesh n, streaming over x-slabs."""
    t0 = time.perf_counter()
    M, kxyz, lam = modes(n, Rp)
    K = kxyz.max(0)
    Px, Py, Pz = (jnp.asarray(sine_table(n, int(k))) for k in K)
    ni = n - 2
    ax = np.arange(1, n - 1) / (n - 1)
    Y, Z = np.meshgrid(ax, ax, indexing='ij')
    yz = np.stack([Y.ravel(), Z.ravel()], 1)                                  # (ni^2, 2), z fastest
    Tj = jnp.asarray(np.asarray(T)[:, :Rp])
    feat = jax.jit(lambda p, x, T_: C.features(p, x) @ T_)

    @jax.jit
    def slab(G, Py_, Pz_):
        Gs = G.reshape(ni, ni, -1)                                             # (y, z, R')
        Zc = jnp.einsum('yzi,zc->yci', Gs, Pz_)                                # (ni, Kz, R')
        return jnp.einsum('yci,yb->bci', Zc, Py_)                              # (Ky, Kz, R')

    acc = jnp.zeros((int(K[0]), int(K[1]), int(K[2]), Rp))
    upd = jax.jit(lambda acc_, Y_, w: acc_ + w[:, None, None, None] * Y_[None])
    for a in range(ni):
        pts = np.concatenate([np.full((len(yz), 1), ax[a]), yz], 1)
        G = jnp.concatenate([feat(bank, jnp.asarray(pts[s:s + row_chunk]), Tj) for s in range(0, len(pts), row_chunk)], 0)
        acc = upd(acc, slab(G, Py, Pz), Px[a])
        if a % 64 == 0:
            log(f'[bigtables n={n} R={Rp}] slab {a}/{ni} {time.perf_counter() - t0:.1f}s')
    k0 = kxyz - 1
    A = acc[k0[:, 0], k0[:, 1], k0[:, 2], :]
    A = jax.block_until_ready(A)
    return A, jnp.asarray(lam), kxyz, M, time.perf_counter() - t0


def rom_data(n, bank, T, Rp, X, w, base65, log=print):
    """ROM data at mesh n for the mesh-free online path: A_n, lam_n, P_n (tests of mesh n at the rule points), B, D
    (mesh-independent), QR of A_n, the quadrature projection (Bw, Lq) and the lattice rows of n = 65 (GTb) for decode."""
    A, lam, kxyz, M, secs = build_A(n, bank, T, Rp, log)
    B, D = OM.point_blocks(bank, np.asarray(T)[:, :Rp], X)
    P = OM.test_block(n, X, w, kxyz)
    QA, RA = jnp.linalg.qr(A, mode='reduced')
    Bw = B * jnp.asarray(w)[:, None]
    Lq = jnp.linalg.cholesky(B.T @ Bw)
    d = dict(A=A, lam=lam, B=B, D=D, P=P, QA=QA, RA=RA, Bw=Bw, Lq=Lq, GTb=base65['GTb'], L=base65['L'],
             hmask=(lam > jnp.median(lam)).astype(jnp.float64))
    return jax.tree_util.tree_map(jax.block_until_ready, d), dict(M=M, seconds_A=secs, kxyz_sha=C3sha(kxyz))


def C3sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def check65(bank, T, Rp, tb, log=print):
    """G5: streaming tables at n = 65 vs tables.build_tables (tb = c3common.Tables(65, ...))."""
    A, lam, kxyz, M, _ = build_A(65, bank, T, Rp, log)
    ref = tb.lean(Rp)
    kref = np.asarray(tb.tb['kxyz'])[:M]
    out = dict(M=M, M_ref=tb.M_of[Rp], kxyz_equal=bool(np.array_equal(kxyz, kref)),
               lam_equal=bool(np.array_equal(np.asarray(lam), np.asarray(ref['lam']))),
               A_rel=float(jnp.linalg.norm(A - ref['A']) / jnp.linalg.norm(ref['A'])))
    out['pass'] = bool(out['M'] == out['M_ref'] and out['kxyz_equal'] and out['lam_equal'] and out['A_rel'] <= 1e-11)
    # K7: mesh 66 restricted to the same mode triples must differ
    A66, lam66, k66, M66, _ = build_A(66, bank, T, Rp, log)
    idx = {tuple(k): i for i, k in enumerate(k66)}
    common = [(i, idx[tuple(k)]) for i, k in enumerate(kxyz) if tuple(k) in idx]
    ia, ib = np.array([c[0] for c in common]), np.array([c[1] for c in common])
    out['K7_rel'] = float(jnp.linalg.norm(A[ia] - A66[ib]) / jnp.linalg.norm(A[ia]))
    out['K7_fires'] = bool(out['K7_rel'] > 1e-4)
    return out
