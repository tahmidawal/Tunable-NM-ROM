"""poisson-bank-knob: deployment-time nested truncation of the frozen R=512 Poisson 2D bank.

Nothing is trained. The frozen bank G (n^2 x R) is rotated once, offline, so that its columns
are ordered by importance for the TRAINING coefficient distribution:

    samples a_i = R_G c_i / ||u_i||   (field coordinates in the exact QR metric G = Q_G R_G at
                                       the training mesh; c_i = bank projection of training
                                       field u_i, and c_i = h(z_i) at the stored training code)
    A = [a_i] = U S V_s^T  ->  G' = G T,  T = R_G^{-1} V_s,  a = L c,  L = V_s^T R_G = T^{-1}

This is the SVD of G Sigma^{1/2} (Sigma = empirical second moment of the training coefficient
vectors), i.e. POD of the training decoded fields inside span(G). A truncation R' keeps the
first R' rotated columns: c' = L[:R'] c, field = G'[:, :R'] c', and the weak operator of the
truncated model is B P_{R'} with P_{R'} = T[:, :R'] L[:R'] (an oblique projector on coefficient
space). The head, the correction directions and the LM kernel are the unchanged parent code; the
residual is simply evaluated through P_{R'}, so the y elimination, nearest-code start and LM are
the parent's with B -> B P_{R'}. At R' = R, P = I to round-off (parity gate).

The rotated bank is held as row chunks x nested COLUMN BLOCKS (block edges = the R' ladder), so
the R' decode reads exactly the first R' columns with no slicing copy.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import correction_core as CC
import pbh_core as K_
import sep_common as sc
import hp_core as H
from kernel_solver import make_lm_kernel


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


# ------------------------------------------------------------- rotation --------

def make_rotation(params, codes, train_draws, intervals):
    """Offline, training data only. Returns T (R x R), L = T^{-1} and diagnostics."""
    t0 = time.perf_counter()
    G = K_.bank_of(params, intervals)
    Rg, rank = K_.bank_r(G)
    assert rank['rank_valid'], rank
    U = K_.fields(np.asarray(train_draws), intervals)
    Tq, perp2, nu2 = K_.project_targets(G, Rg, U)            # Q_G^T u
    del G, U
    heads = jax.jit(jax.vmap(lambda z: sc.head(params, z)))(jnp.asarray(codes))
    Rg = np.asarray(Rg)
    scale = np.sqrt(np.asarray(nu2))[:, None]
    A_proj = np.asarray(Tq) / scale                          # projection coefficients (q = R limit)
    A_head = (np.asarray(heads) @ Rg.T) / scale              # head coefficients (q = 0)
    A = np.concatenate((A_proj, A_head), axis=0)
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    Vs = Vt.T
    T = np.linalg.solve(Rg, Vs)                              # R_G^{-1} V_s
    L = Vs.T @ Rg
    ident = float(np.linalg.norm(L @ T - np.eye(len(s))))
    sv_rg = np.linalg.svd(Rg, compute_uv=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    # energy captured per sample family at each prefix
    ep = np.cumsum(np.sum((A_proj @ Vs) ** 2, axis=0)) / np.sum(A_proj ** 2)
    eh = np.cumsum(np.sum((A_head @ Vs) ** 2, axis=0)) / np.sum(A_head ** 2)
    info = dict(training_intervals=int(intervals), training_sources=int(len(train_draws)),
                sample_rule='rows = [Q_G^T u_i ; R_G h(z_i)] / ||u_i||, training fit split only',
                R_G_condition_number=float(sv_rg[0] / sv_rg[-1]),
                L_times_T_identity_deviation=ident,
                singular_values=s.tolist(),
                cumulative_energy={str(r): float(energy[r - 1]) for r in (32, 64, 128, 256, 384, 512) if r <= len(s)},
                cumulative_energy_projection_samples={str(r): float(ep[r - 1]) for r in (32, 64, 128, 256, 384, 512) if r <= len(s)},
                cumulative_energy_head_samples={str(r): float(eh[r - 1]) for r in (32, 64, 128, 256, 384, 512) if r <= len(s)},
                T_sha256=sha_array(T), L_sha256=sha_array(L),
                seconds=time.perf_counter() - t0)
    return T, L, info


# ------------------------------------------------------------- the banks -------

def build_banks(params, n, rows_per_chunk, eval_rows, maxmode, Tm, edges, truth,
                keep_orig=True, keep_rot32=False):
    """One pass over the frozen feature network on the n-interval mesh.

    Produces: the sine-sine contraction of the ORIGINAL bank (every weak operator is a gather
    of it), R_G at this mesh (TSQR), G^T u for the truth fields (-> floors at every R'), the
    original row chunks (optional; the parity baseline) and the rotated row chunks split into
    nested column blocks at `edges`.
    """
    t0 = time.perf_counter()
    p = np.arange(1, n) / n
    S = jnp.asarray(H.sine_table(n, maxmode))
    feat = jax.jit(sc.features)
    Tj = jnp.asarray(Tm)

    @jax.jit
    def contract(G, Srows, S):
        cube = G.reshape((Srows.shape[0], n - 1, G.shape[-1]))
        tmp = jnp.einsum('xyr,yb->xbr', cube, S)
        return jnp.einsum('xa,xbr->abr', Srows, tmp)

    @jax.jit
    def rotate(G, Tj):
        Gr = G @ Tj
        return tuple(Gr[:, a:b] for a, b in zip(edges[:-1], edges[1:]))

    orig, rot, rot32, rfac = [], [], [], []
    Tc = None
    gtu = None
    Uj = jnp.asarray(truth)
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
        Tc = c if Tc is None else Tc + c
        rfac.append(H.qr_r(G))
        g = Uj[:, s:e].reshape(Uj.shape[0], -1) @ G
        gtu = g if gtu is None else gtu + g
        blocks = rotate(G, Tj)
        rot.append(blocks)
        if keep_rot32:
            rot32.append(tuple(x.astype(jnp.float32) for x in blocks))
        if keep_orig:
            orig.append(G)
        jax.block_until_ready((Tc, rfac[-1], gtu, blocks))
        del G
    Rg = H.qr_r(jnp.concatenate(rfac, axis=0))
    values = np.asarray(jnp.linalg.svd(Rg, compute_uv=False))
    threshold = float(values[0] * (n - 1) ** 2 * np.finfo(float).eps)
    rank = int((values > threshold).sum())
    # floors: t = Q_G^T u ; floor(R')^2 = 1 - ||proj of t on range(R_G T[:, :R'])||^2 / ||u||^2
    t = np.asarray(jax.scipy.linalg.solve_triangular(Rg.T, gtu.T, lower=True).T)
    nu2 = np.asarray(jnp.sum(Uj * Uj, axis=(1, 2)))
    RgT = np.asarray(Rg) @ Tm
    floors = {}
    for Rp in edges[1:]:
        Qx, _ = np.linalg.qr(RgT[:, :Rp])
        perp2 = np.clip(nu2 - np.sum((t @ Qx) ** 2, axis=1), 0., None)
        floors[int(Rp)] = np.sqrt(perp2 / nu2)
    info = dict(intervals=n, rows_per_chunk=rows_per_chunk, chunks=len(rfac),
                stored_features=int(Rg.shape[1]), retained_bank_rank=rank,
                condition_number=float(values[0] / values[-1]), column_block_edges=list(map(int, edges)),
                bank_bytes_f64=int((n - 1) ** 2 * Rg.shape[1] * 8), maxmode=int(maxmode),
                kept_original=bool(keep_orig), kept_rotated_f32=bool(keep_rot32),
                contraction_sha256=sha_array(Tc), build_seconds=time.perf_counter() - t0)
    return dict(orig=tuple(orig), rot=tuple(rot), rot32=tuple(rot32), T=Tc, S=S, Rg=Rg,
                info=info, floors=floors)


def nblocks(edges, Rp):
    k = list(edges).index(Rp)
    assert k > 0, (edges, Rp)
    return k


def decode_blocks(rot, a, n, nb):
    """field = G'[:, :R'] a, reading only the first nb column blocks of every row chunk."""
    rows = []
    off = 0
    for blocks in rot:
        acc = None
        col = 0
        for b in range(nb):
            w = blocks[b].shape[1]
            part = blocks[b] @ a[col:col + w].astype(blocks[b].dtype)
            acc = part if acc is None else acc + part
            col += w
        rows.append(acc.reshape(-1, n - 1))
    return jnp.pad(jnp.concatenate(rows, axis=0), 1).astype(jnp.float64)


# ------------------------------------------------------------- arms ------------

def make_trunc_lean(ops, engine, cfg, count, nb, n):
    """The parent lean query (hp_core.make_lean) with the residual through P_{R'} (already in
    ops['B'] / engine) and the decode from the rotated nested blocks: a = L[:R'] c."""
    reduced = {**ops, 'B': engine['Bp']}
    solve = make_lm_kernel(reduced, cfg['online_preset']['budget'], True, False,
                           cfg['linear_backward_error_limit'],
                           cfg['online_preset']['stationarity_stop'])
    tau = cfg['online_preset']['tau']

    @jax.jit
    def kernel(source, rot, Lr, S, I, J, W, params, predictions, cached_codes, Q, R, Cq, B):
        f = ops['project'](source, S, I, J, W)
        fp = f - Q @ (Q.T @ f) if count else f
        squared = jnp.sum((predictions - fp[None, :]) ** 2, axis=1)
        _, indices = jax.lax.top_k(-squared, 1)
        z0 = cached_codes[indices[0]]
        answer, stats, _ = solve(z0, fp, jnp.asarray(tau))
        h = sc.head(params, answer[0])
        if count:
            y = jax.scipy.linalg.solve_triangular(R, Q.T @ (f - B @ h), lower=False)
            coefficients = h + Cq @ y
        else:
            y = jnp.zeros((0,))
            coefficients = h
        return decode_blocks(rot, Lr @ coefficients, n, nb), answer, stats, indices[0], y
    return kernel


def trunc_query(host_source, ops, engine, kernel, rot, Lr):
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    cache = engine['cache']
    out = kernel(source, rot, Lr, ops['S'], ops['I'], ops['J'], ops['W'], ops['params'],
                 cache['predictions'], cache['codes'], engine['Q'], engine['R'], engine['C'],
                 ops['B'])
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
        reason=int(reason), attempts=int(attempts), accepted=int(accepted), jacobians=int(njac),
        max_linear_backward_error=float(stats[0]), fallback_count=int(stats[1]))


def make_trunc_linear(Bprime, n, nb):
    """q = R' linear model in the rotated truncated bank: thin QR solve, nested decode."""
    Qm, Rr = jnp.linalg.qr(jnp.asarray(Bprime), mode='reduced')
    Qt = Qm.T

    @jax.jit
    def kernel(source, rot, S, I, J, W, Qt, Rr):
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        a = jax.scipy.linalg.solve_triangular(Rr, Qt @ fm, lower=False)
        return decode_blocks(rot, a, n, nb), a
    values = np.asarray(jnp.linalg.svd(Rr, compute_uv=False))
    return kernel, Qt, Rr, dict(qr_condition_number=float(values[0] / values[-1]),
                                M=int(Bprime.shape[0]), R_prime=int(Bprime.shape[1]))


def trunc_linear_query(host_source, ops, kernel, Qt, Rr, rot):
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    out = kernel(source, rot, ops['S'], ops['I'], ops['J'], ops['W'], Qt, Rr)
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, a = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field, dtype=np.float64), dict(
        total_seconds=end - start, input_seconds=input_end - start,
        fused_device_seconds=device_end - input_end, output_seconds=end - device_end,
        linear_coefficients=np.asarray(a).tolist())


def arm_q_max(Rp, K, cap=256):
    return max(0, min(cap, Rp - K))
