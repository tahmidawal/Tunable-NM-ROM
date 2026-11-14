"""linear-query-speed: offline compression of the mesh-evaluated bank and the fused online query.

Nothing is trained and the reduced model is unchanged.  The rotated bank of the paper, G'[(x, y), r]
(n-1 interior nodes per side, R features), is a dense (n-1)^2 x R array whose product with the reduced
coefficients is the whole cost of a query on fine meshes.  Viewed as an array (x, y, r) it has low rank
in the two mesh directions, so offline

    G'[(x, y), r]  ~  sum_ab U[x, a] V[y, b] C[a, b, r]          (U, V orthonormal columns)

and online  u = U (sum_r a_r C[:, :, r]) V^T :  one small matrix-vector product and two matrix products.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ------------------------------------------------------------- offline --------

def compress_bank(rot, n, probes=224, seed=20261114):
    """Randomised range finder in y, then in x on the y-compressed array.  `rot` is the paper's rotated
    bank: a tuple of row chunks, each a tuple of column blocks of shape (rows * (n-1), width)."""
    t0 = time.perf_counter()
    m = n - 1
    R = int(sum(b.shape[1] for b in rot[0]))
    k = int(min(probes, m))
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def sketch_y(blocks, om):
        # sum_x T[x] (m x R) @ om[x] (R x k), block by block: no concatenated or transposed copy
        y, c2, off = 0.0, [], 0
        for b in blocks:
            w = b.shape[1]
            y = y + jnp.sum(jnp.matmul(b.reshape(-1, m, w), om[:, off:off + w, :]), axis=0)
            c2.append(jnp.sum(jnp.square(b), axis=0))
            off += w
        return y, jnp.concatenate(c2)

    @jax.jit
    def project_y(blocks, V):
        return jnp.concatenate([jnp.einsum('xyw,yb->xbw', b.reshape(-1, m, b.shape[1]), V) for b in blocks],
                               axis=2)                             # (rows, k, R)

    Y = jnp.zeros((m, k))
    col2 = jnp.zeros((R,))
    for i, blocks in enumerate(rot):
        rows = blocks[0].shape[0] // m
        om = jax.random.normal(jax.random.fold_in(key, i), (rows, R, k), dtype=jnp.float64)
        y, c2 = sketch_y(blocks, om)
        Y, col2 = Y + y, col2 + c2
        del om
    V0, _ = jnp.linalg.qr(Y, mode='reduced')
    Z = jnp.concatenate([project_y(blocks, V0) for blocks in rot], axis=0)   # (m, k, R)
    om = jax.random.normal(jax.random.fold_in(key, 10_000), (k * R, k), dtype=jnp.float64)
    U0, _ = jnp.linalg.qr(Z.reshape(m, k * R) @ om, mode='reduced')
    del om
    C0 = (U0.T @ Z.reshape(m, k * R)).reshape(k, k, R)
    jax.block_until_ready(C0)
    del Z
    col2 = np.asarray(col2)
    C0 = np.asarray(C0)
    kept2 = np.sum(C0 * C0, axis=(0, 1))
    info = dict(probes=k, seed=int(seed), features=R, intervals=int(n),
                bank_frobenius=float(np.sqrt(col2.sum())),
                range_finder_relative_residual={
                    str(Rp): float(np.sqrt(max(col2[:Rp].sum() - kept2[:Rp].sum(), 0.0) / col2[:Rp].sum()))
                    for Rp in (128, 512) if Rp <= R},
                note='residual from a difference of squared norms; resolution about 1e-8',
                seconds=time.perf_counter() - t0)
    return np.asarray(U0), np.asarray(V0), C0, col2, info


def truncate(U0, V0, C0, col2, Rp, tau):
    """Truncated higher-order SVD of the first Rp feature slices of the core to relative Frobenius tau."""
    C = C0[:, :, :Rp]
    k = C.shape[0]
    tot2 = float(col2[:Rp].sum())

    def mode(Mat):
        w, Q = np.linalg.eigh(Mat @ Mat.T)
        w, Q = w[::-1], Q[:, ::-1]
        tail = np.concatenate((np.cumsum(np.clip(w, 0, None)[::-1])[::-1], [0.0]))   # tail[p] = sum_{i>=p}
        p = int(np.argmax(tail <= 0.5 * tau ** 2 * tot2))
        return max(p, 1), Q

    p, W1 = mode(C.reshape(k, -1))
    q, W2 = mode(np.transpose(C, (1, 0, 2)).reshape(k, -1))
    W1, W2 = W1[:, :p], W2[:, :q]
    Cc = np.einsum('abr,ap,bq->pqr', C, W1, W2, optimize=True)
    resid = float(np.sqrt(max(tot2 - float(np.sum(Cc * Cc)), 0.0) / tot2))
    return dict(U=U0 @ W1, V=V0 @ W2, C=Cc, p=p, q=q, Rp=int(Rp), tau=float(tau),
                estimated_relative_residual=resid,
                stored_bytes_f64=int(8 * (U0.shape[0] * p + V0.shape[0] * q + Cc.size)))


def reduced_operator(Qt, Rr, W):
    """P = R^-1 Q^T diag(W): the paper's thin-QR least-squares solve applied once, offline."""
    import scipy.linalg as sl
    return sl.solve_triangular(np.asarray(Rr), np.asarray(Qt), lower=False) * np.asarray(W)[None, :]


# ------------------------------------------------------------- online ---------

def sine_rows(S, n, m):
    """Sine table rows for the nested sub-lattice with m intervals (m divides n), with the quadrature
    weight of the coarser lattice folded in.  m == n returns S."""
    assert n % m == 0, (n, m)
    st = n // m
    return (S if st == 1 else st * S[st - 1::st]), st


def device_arrays(tk, S, I, J, Pm, n, m, dtype):
    Sx, st = sine_rows(np.asarray(S), n, m)
    U = np.pad(tk['U'], ((1, 1), (0, 0)))
    Vt = np.pad(tk['V'], ((1, 1), (0, 0))).T
    C2 = tk['C'].reshape(tk['p'] * tk['q'], tk['Rp'])
    args = (jnp.asarray(Sx), jnp.asarray(I), jnp.asarray(J), jnp.asarray(Pm),
            jnp.asarray(C2, dtype=dtype), jnp.asarray(U, dtype=dtype), jnp.asarray(np.ascontiguousarray(Vt), dtype=dtype))
    return args, st


def make_kernel(p, q, st, dtype):
    """device f64 nodal source -> device f64 nodal field (n+1, n+1) and the reduced coefficients."""
    @jax.jit
    def kernel(source, Sx, I, J, Pm, C2, U, Vt):
        f = source[1:-1, 1:-1] if st == 1 else source[st:-1:st, st:-1:st]
        a = Pm @ (Sx.T @ f @ Sx)[I, J]
        core = (C2 @ a.astype(dtype)).reshape(p, q)
        return ((U @ core) @ Vt).astype(jnp.float64), a
    return kernel


def make_solve_only(st):
    @jax.jit
    def kernel(source, Sx, I, J, Pm):
        f = source[1:-1, 1:-1] if st == 1 else source[st:-1:st, st:-1:st]
        return Pm @ (Sx.T @ f @ Sx)[I, J]
    return kernel


def make_stages(p, q, st, dtype):
    @jax.jit
    def s_solve(source, Sx, I, J, Pm):
        f = source[1:-1, 1:-1] if st == 1 else source[st:-1:st, st:-1:st]
        return Pm @ (Sx.T @ f @ Sx)[I, J]

    @jax.jit
    def s_core(a, C2):
        return (C2 @ a.astype(dtype)).reshape(p, q)

    @jax.jit
    def s_field(core, U, Vt):
        return ((U @ core) @ Vt).astype(jnp.float64)

    def timed(fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        return out, time.perf_counter() - t

    def run(source, Sx, I, J, Pm, C2, U, Vt):
        a, t1 = timed(s_solve, source, Sx, I, J, Pm)
        core, t2 = timed(s_core, a, C2)
        field, t3 = timed(s_field, core, U, Vt)
        return field, dict(project_and_solve=t1, core_contraction=t2, field_products=t3)
    return run


def query(host_source, kernel, args, has_field=True):
    """The paper's timing scope (`pbk_core.trunc_linear_query`): host source in, host output out;
    GPU-query = fused_device_seconds (device source -> device result, synced)."""
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    out = kernel(source, *args)
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    out = jax.device_get(out)
    end = time.perf_counter()
    row = dict(total_seconds=end - start, input_seconds=input_end - start,
               fused_device_seconds=device_end - input_end, output_seconds=end - device_end)
    if has_field:
        return np.asarray(out[0], dtype=np.float64), np.asarray(out[1]), row
    return None, np.asarray(out), row
