"""hires-poisson: chunked bank, weak assembly and lean query kernels for large meshes.

The checkpoint, head, correction directions, LM kernel, test-mode rule and CG are the
unchanged parent code (`core`, `speed_core`, `kernel_solver`, `correction_core`,
`iterative_core`, `plin_core`).  What is new here is only HOW the mesh-sized bank is held
and applied: as a list of row chunks (never one (n-1)^2 x R array), so that 2048^2 and
4096^2 fit on one card.  Every lean path is parity-gated in the job against the retained
single-bank `correction_core` kernel wherever that kernel fits in memory.

Staged flat: every module sits beside this file on the cluster.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import sep_common as sc
from kernel_solver import make_lm_kernel


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


# ------------------------------------------------------------------ the bank --

def sine_table(n, maxmode):
    p = np.arange(1, n)
    return np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(p, np.arange(1, maxmode + 1)) / n)


def mode_set(n, modes):
    """Retained sine tests, exactly `core.assemble`'s rule (ties at the cut are kept)."""
    lam = C.eigenvalues(n)
    I, J = np.nonzero(lam <= np.sort(lam.ravel())[modes - 1])
    return I, J, lam[I, J] ** -1


def qr_r(G, retries=2):
    """Thin R factor with the GB10 first-call NaN guard of `pbh_core.bank_r`."""
    for attempt in range(retries + 1):
        R = jnp.linalg.qr(G, mode='r')
        if bool(jnp.isfinite(R).all()):
            return R
        print(f'WARNING: non-finite QR factor, attempt {attempt + 1}', flush=True)
    raise FloatingPointError('QR factor non-finite')


def build_bank(params, n, rows_per_chunk, eval_rows, maxmode, truth=None, keep64=True,
               keep32=False):
    """Evaluate the frozen feature network on the n-interval mesh, chunk by chunk.

    One pass produces everything that needs the f64 bank, so the f64 chunks need not be
    kept: the sine-sine contraction T[a,b,r] (from which every weak operator B is a
    gather), the stacked chunk R factors (TSQR -> exact bank rank and metric), and
    G^T u for the supplied truth fields (-> bank projection floor).
    """
    t0 = time.perf_counter()
    p = np.arange(1, n) / n
    S = jnp.asarray(sine_table(n, maxmode))
    feat = jax.jit(sc.features)

    @jax.jit
    def contract(G, Srows, S):
        cube = G.reshape((Srows.shape[0], n - 1, G.shape[-1]))
        tmp = jnp.einsum('xyr,yb->xbr', cube, S)
        return jnp.einsum('xa,xbr->abr', Srows, tmp)

    chunks64, chunks32, rfac = [], [], []
    T = None
    gtu = None
    Uj = None if truth is None else jnp.asarray(truth)          # (cases, n-1, n-1)
    for s in range(0, n - 1, rows_per_chunk):
        e = min(s + rows_per_chunk, n - 1)
        parts = []
        for a in range(s, e, eval_rows):
            b = min(a + eval_rows, e)
            xx, yy = np.meshgrid(p[a:b], p, indexing='ij')
            parts.append(feat(params, jnp.asarray(np.column_stack((xx.ravel(), yy.ravel())))))
        G = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        del parts
        c = contract(G, S[s:e], S)
        T = c if T is None else T + c
        rfac.append(qr_r(G))
        if Uj is not None:
            g = Uj[:, s:e].reshape(Uj.shape[0], -1) @ G
            gtu = g if gtu is None else gtu + g
        if keep32:
            chunks32.append(G.astype(jnp.float32))
        if keep64:
            chunks64.append(G)
        # bound the in-flight work: every consumer of this chunk finishes before the next
        jax.block_until_ready((T, rfac[-1], gtu, G))
        del G
    Rg = qr_r(jnp.concatenate(rfac, axis=0))
    values = np.asarray(jnp.linalg.svd(Rg, compute_uv=False))
    threshold = float(values[0] * (n - 1) ** 2 * np.finfo(float).eps)
    rank = int((values > threshold).sum())
    info = dict(intervals=n, rows_per_chunk=rows_per_chunk, chunks=len(rfac),
                stored_features=int(Rg.shape[1]), retained_bank_rank=rank,
                rank_threshold=threshold, condition_number=float(values[0] / values[-1]),
                bank_bytes_f64=int((n - 1) ** 2 * Rg.shape[1] * 8), maxmode=int(maxmode),
                kept_f64=bool(keep64), kept_f32=bool(keep32),
                contraction_sha256=sha_array(T), build_seconds=time.perf_counter() - t0)
    floor = None
    if Uj is not None:
        Tt = jax.scipy.linalg.solve_triangular(Rg.T, gtu.T, lower=True).T
        nu2 = jnp.sum(Uj * Uj, axis=(1, 2))
        perp2 = jnp.clip(nu2 - jnp.sum(Tt * Tt, axis=1), 0., None)
        floor = np.asarray(jnp.sqrt(perp2 / nu2))
    return dict(chunks64=tuple(chunks64), chunks32=tuple(chunks32), T=T, S=S, Rg=Rg,
                info=info, floor=floor)


def make_ops(bankd, params, codes, n, modes):
    """An `ops` dict with the fields the parent solver code reads (no bank inside)."""
    I, J, W = mode_set(n, modes)
    maxmode = int(max(I.max(), J.max())) + 1
    assert maxmode <= bankd['S'].shape[1], (maxmode, bankd['S'].shape)
    S = bankd['S'][:, :maxmode]
    B = bankd['T'][jnp.asarray(I), jnp.asarray(J)]

    @jax.jit
    def project(F, S, ii, jj, weight):
        c = S.T @ F[1:-1, 1:-1] @ S
        return c[ii, jj] * weight

    codes = np.asarray(codes)
    trust = float(np.linalg.norm(codes - codes.mean(0), axis=1).max())
    info = dict(intervals=n, requested_modes=int(modes), retained_modes=int(len(I)),
                trust_delta=trust, operator_sha256=C.sha(np.asarray(B)))
    return dict(B=B, S=S, I=jnp.asarray(I), J=jnp.asarray(J), W=jnp.asarray(W),
                z0=jnp.asarray(codes.mean(0)), project=project, params=params, info=info,
                intervals=n)


# ------------------------------------------------------------- lean kernels ---

def decode_chunks(chunks, coefficients, n):
    c = coefficients.astype(chunks[0].dtype)
    rows = [(G @ c).reshape(-1, n - 1) for G in chunks]
    return jnp.pad(jnp.concatenate(rows, axis=0), 1).astype(jnp.float64)


def make_lean(ops, engine, cfg, count):
    """The `correction_core` query without the in-kernel diagnostics, chunked decode.

    Identical numerics up to the decode: same projection, same projected nearest-code
    start, same LM kernel (built by the same factory from the same reduced operator),
    same triangular recovery of the eliminated coefficients.
    """
    n = ops['info']['intervals']
    reduced = {**ops, 'B': engine['Bp']}
    solve = make_lm_kernel(reduced, cfg['online_preset']['budget'], True, False,
                           cfg['linear_backward_error_limit'],
                           cfg['online_preset']['stationarity_stop'])
    tau = cfg['online_preset']['tau']

    @jax.jit
    def kernel(source, chunks, S, I, J, W, params, predictions, cached_codes, Q, R, Cq, B):
        f = ops['project'](source, S, I, J, W)
        fp = f - Q @ (Q.T @ f) if count else f
        squared = jnp.sum((predictions - fp[None, :]) ** 2, axis=1)
        _, indices = jax.lax.top_k(-squared, 1)
        index = indices[0]
        z0 = cached_codes[index]
        answer, stats, _ = solve(z0, fp, jnp.asarray(tau))
        h = sc.head(params, answer[0])
        if count:
            y = jax.scipy.linalg.solve_triangular(R, Q.T @ (f - B @ h), lower=False)
            coefficients = h + Cq @ y
        else:
            y = jnp.zeros((0,))
            coefficients = h
        return decode_chunks(chunks, coefficients, n), answer, stats, index, y
    return kernel


def make_diagnose(ops, engine, count):
    """Untimed post-query validation: every quantity `correction_core` computes in-kernel."""
    @jax.jit
    def diagnose(source, z, y, S, I, J, W, params, Q, R, Cq, B, Bp):
        f = ops['project'](source, S, I, J, W)
        fp = f - Q @ (Q.T @ f) if count else f
        h = sc.head(params, z)
        coefficients = h + Cq @ y if count else h
        residual = B @ coefficients - f
        D = jax.jacfwd(lambda zz: sc.head(params, zz))(z)
        Jfull = B @ D
        cols = B @ Cq
        grad = jnp.concatenate((Jfull.T @ residual, cols.T @ residual))
        full = jnp.linalg.norm(grad) / (jnp.sqrt(jnp.sum(Jfull * Jfull) + jnp.sum(cols * cols))
                                        * jnp.linalg.norm(residual) + 1e-300)
        Jr = Bp @ D
        rr = Bp @ h - fp
        red = jnp.linalg.norm(Jr.T @ rr) / (jnp.linalg.norm(Jr) * jnp.linalg.norm(rr) + 1e-300)
        if count:
            rhs = Q.T @ (f - B @ h)
            recovery = jnp.linalg.norm(R @ y - rhs) / (jnp.linalg.norm(R) * jnp.linalg.norm(y)
                                                     + jnp.linalg.norm(rhs) + 1e-300)
        else:
            recovery = jnp.asarray(0.)
        reconstruct = jnp.linalg.norm(residual - rr) / (jnp.linalg.norm(f)
                                                        + jnp.linalg.norm(B @ h) + 1e-300)
        singular = jnp.linalg.svd(Jr, compute_uv=False)
        rank = jnp.sum(singular > singular[0] * max(Jr.shape) * jnp.finfo(jnp.float64).eps)
        return full, red, jnp.linalg.norm(residual), jnp.linalg.norm(f), recovery, reconstruct, rank
    return diagnose


def lean_query(host_source, ops, engine, kernel, chunks, pre=None, post=None):
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    cache = engine['cache']
    if pre is not None:
        source = pre(source)
    out = kernel(source, chunks, ops['S'], ops['I'], ops['J'], ops['W'], ops['params'],
                 cache['predictions'], cache['codes'], engine['Q'], engine['R'], engine['C'],
                 ops['B'])
    if post is not None:
        out = (post(out[0]),) + tuple(out[1:])
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, answer, stats, index, y = jax.device_get(out)
    end = time.perf_counter()
    z, res, initial, njac, accepted, attempts, reason = answer
    return np.asarray(field, dtype=np.float64), dict(
        total_seconds=end - start, input_seconds=input_end - start,
        fused_device_seconds=device_end - input_end, output_seconds=end - device_end,
        latent=np.asarray(z).tolist(), correction_coefficients=np.asarray(y).tolist(),
        selected_training_code_index=int(index), residual=float(res),
        initial_residual=float(initial), reason=int(reason), attempts=int(attempts),
        accepted=int(accepted), jacobians=int(njac),
        max_linear_backward_error=float(stats[0]), fallback_count=int(stats[1]))


def make_linear(B, n):
    """q = R: the rank-R linear reduced model solved directly (`plin_core.make_linear_query`),
    chunked decode."""
    Qm, Rr = jnp.linalg.qr(jnp.asarray(B), mode='reduced')
    Qt = Qm.T

    @jax.jit
    def kernel(source, chunks, S, I, J, W, Qt, Rr):
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        y = jax.scipy.linalg.solve_triangular(Rr, Qt @ fm, lower=False)
        return decode_chunks(chunks, y, n), y
    values = np.asarray(jnp.linalg.svd(Rr, compute_uv=False))
    return kernel, Qt, Rr, dict(qr_condition_number=float(values[0] / values[-1]),
                                M=int(B.shape[0]))


def linear_query(host_source, ops, kernel, Qt, Rr, chunks, pre=None, post=None):
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    if pre is not None:
        source = pre(source)
    out = kernel(source, chunks, ops['S'], ops['I'], ops['J'], ops['W'], Qt, Rr)
    if post is not None:
        out = (post(out[0]),) + tuple(out[1:])
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, y = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field, dtype=np.float64), dict(total_seconds=end - start, input_seconds=input_end - start,
                                   fused_device_seconds=device_end - input_end,
                                   output_seconds=end - device_end, reason=4, attempts=0,
                                   accepted=0, jacobians=0,
                                   linear_coefficients=np.asarray(y).tolist())


# ------------------------------------------------------- coarse-grid controls --

def bilinear_up(u, requested):
    coarse = u.shape[0] - 1
    old = jnp.linspace(0., 1., coarse + 1)
    new = jnp.linspace(0., 1., requested + 1)
    along_x = jax.vmap(lambda col: jnp.interp(new, old, col), in_axes=1, out_axes=1)(u)
    return jax.vmap(lambda row: jnp.interp(new, old, row))(along_x)


def make_coarse(n, nc, cg_kernel=None):
    """Solve on an nc-interval grid from the point-sampled supplied source, then bilinear
    interpolation to the requested n-interval nodes (charged). DST if `cg_kernel` is None."""
    assert n % nc == 0
    f = n // nc
    lam = jnp.asarray(C.eigenvalues(nc))

    @jax.jit
    def dst_kernel(source, lam):
        return bilinear_up(C.dst_solve(source[::f, ::f], lam), n)

    @jax.jit
    def cg_wrap(source, tolerance):
        u, info = cg_kernel(source[::f, ::f], tolerance)
        return bilinear_up(u, n), info
    return (dst_kernel, lam) if cg_kernel is None else (cg_wrap, None)


def generic_query(host_source, fn, pre=None, post=None):
    """host source in -> host f64 nodal field out, same scope as every other subject."""
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    if pre is not None:
        source = pre(source)
    out = fn(source)
    if post is not None:
        out = (post(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else post(out)
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    out = jax.device_get(out)
    end = time.perf_counter()
    field, extra = (out if isinstance(out, tuple) else (out, None))
    return np.asarray(field, dtype=np.float64), dict(total_seconds=end - start, input_seconds=input_end - start,
                                   fused_device_seconds=device_end - input_end,
                                   output_seconds=end - device_end), extra
