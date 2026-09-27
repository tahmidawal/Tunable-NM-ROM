"""poisson-bank-knob-3d: shared pieces for the nested bank truncation R' of the frozen L-shaped
Poisson (K=16, R=512) and three-dimensional Poisson (K=16, R=128) models.

Nothing is trained. The frozen bank G (n x R) is rotated once, offline, from TRAINING data only
(the sibling lane's construction, `experiments/poisson-bank-knob/pbk_core.make_rotation`):

    rows  a_i = [ Q_G^T u_i ; R_G h(z_i) ] / ||u_i||     (G = Q_G R_G at the training mesh)
    A = U S V_s^T,   T = R_G^{-1} V_s,   L = V_s^T R_G = T^{-1}

so G T has columns ordered by training energy (POD of the training decoded fields inside
span(G)). A truncation R' keeps the first R' rotated columns: decode u = (G T)[:, :R'] (L[:R'] c),
and the residual uses B P_{R'} with P_{R'} = T[:, :R'] L[:R'] (P = I exactly at R' = R).
The rotated bank is held as nested COLUMN BLOCKS (edges = the R' ladder), so the R' decode reads
exactly R' columns.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def dump(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.floating):
            x = float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def gpu_uuid():
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    raw = (ctypes.c_ubyte * 16)()
    fn = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
    assert fn(ctypes.byref(raw), dev) == 0
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


_BURN = {}


def burn(seconds):
    """GPU burn-in before every timed invocation (the parents' `burn`)."""
    if 'fn' not in _BURN:
        _BURN['a'] = jnp.ones((384, 384), dtype=jnp.float64) * 0.001
        _BURN['fn'] = jax.jit(lambda x: x @ x + 0.0001)
    a, fn = _BURN['a'], _BURN['fn']
    fn(a).block_until_ready()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn(a).block_until_ready()


def qr_r(G):
    R = jnp.linalg.qr(jnp.asarray(G), mode='r')
    if not bool(jnp.isfinite(R).all()):          # GB10 first-QR NaN landmine (parent amendment 6)
        R = jnp.linalg.qr(jnp.asarray(G), mode='r')
    return R


def make_rotation(G, U, H):
    """Offline, TRAINING data only.

    G: (n, R) frozen bank at the training mesh; U: (S, n) training fields on that mesh;
    H: (S, R) head coefficients h(z_i) at the stored training codes, row-aligned with U.
    Returns T (R x R), L = T^{-1} and diagnostics."""
    t0 = time.perf_counter()
    Gj = jnp.asarray(G)
    Rg = qr_r(Gj)
    sv_rg = np.asarray(jnp.linalg.svd(Rg, compute_uv=False))
    Uj = jnp.asarray(U)
    GtU = np.asarray(Uj @ Gj)                                   # (S, R) = (G^T u_i)^T
    Rg = np.asarray(Rg)
    Tq = np.linalg.solve(Rg.T, GtU.T).T                         # Q_G^T u_i
    nu2 = np.asarray(jnp.sum(Uj * Uj, axis=1))
    scale = np.sqrt(nu2)[:, None]
    A_proj = Tq / scale
    A_head = (np.asarray(H) @ Rg.T) / scale
    A = np.concatenate((A_proj, A_head), axis=0)
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    Vs = Vt.T
    T = np.linalg.solve(Rg, Vs)
    L = Vs.T @ Rg
    R = Rg.shape[0]
    ident = float(np.linalg.norm(L @ T - np.eye(R)))
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    ep = np.cumsum(np.sum((A_proj @ Vs) ** 2, axis=0)) / np.sum(A_proj ** 2)
    eh = np.cumsum(np.sum((A_head @ Vs) ** 2, axis=0)) / np.sum(A_head ** 2)
    # training-projection floor at each prefix (in the exact QR metric): 1 - ||P_prefix t||^2/||u||^2
    perp_full = np.clip(1.0 - np.sum(A_proj ** 2, axis=1), 0, None)
    marks = sorted({r for r in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if r <= R} | {R})
    floor_train = {}
    for r in marks:
        inside = np.sum((A_proj @ Vs[:, :r]) ** 2, axis=1)
        floor_train[str(r)] = float(np.sqrt(np.max(np.clip(1.0 - inside, 0, None))))
    info = dict(training_states=int(len(U)),
                sample_rule='rows = [Q_G^T u_i ; R_G h(z_i)] / ||u_i||, training states only',
                R_G_condition_number=float(sv_rg[0] / sv_rg[-1]),
                L_times_T_identity_deviation=ident,
                singular_values=s.tolist(),
                cumulative_energy={str(r): float(energy[r - 1]) for r in marks},
                cumulative_energy_projection_samples={str(r): float(ep[r - 1]) for r in marks},
                cumulative_energy_head_samples={str(r): float(eh[r - 1]) for r in marks},
                training_projection_floor_worst_by_prefix=floor_train,
                training_projection_floor_worst_full=float(np.sqrt(perp_full.max())),
                T_sha256=sha_array(T), L_sha256=sha_array(L),
                seconds=time.perf_counter() - t0)
    return T, L, info


def split_blocks(Gr, edges):
    """(n, R) rotated bank -> tuple of column blocks at the ladder edges (each its own buffer)."""
    return tuple(jnp.array(Gr[:, a:b]) for a, b in zip(edges[:-1], edges[1:]))


def nblocks(edges, Rp):
    k = list(edges).index(Rp)
    assert k > 0, (edges, Rp)
    return k


def decode(blocks, a):
    """u = [B_1 ... B_k] a, reading only the blocks passed."""
    acc = None
    col = 0
    for b in blocks:
        w = b.shape[1]
        part = b @ a[col:col + w]
        acc = part if acc is None else acc + part
        col += w
    return acc


def floors_by_prefix(Rg_mesh, T, GtU_mesh, nu2, edges):
    """Same-mesh projection floor of each truth field onto span((G T)[:, :R']) for every edge.

    Rg_mesh: R factor of G at this mesh; GtU_mesh: (S, R) = U G; nu2: ||u||^2."""
    t = np.linalg.solve(np.asarray(Rg_mesh).T, np.asarray(GtU_mesh).T).T      # Q_G^T u
    RgT = np.asarray(Rg_mesh) @ np.asarray(T)
    out = {}
    for Rp in edges[1:]:
        Qx, _ = np.linalg.qr(RgT[:, :Rp])
        perp2 = np.clip(nu2 - np.sum((t @ Qx) ** 2, axis=1), 0., None)
        out[int(Rp)] = np.sqrt(perp2 / nu2)
    return out


def summarise(v):
    v = np.asarray(v, dtype=float)
    return dict(worst=float(v.max()), median=float(np.median(v)), mean=float(v.mean()), count=int(len(v)))


def q_set(ladder, Rp, K):
    """Correction ranks tested at truncation R': the parent's ladder values with q <= R' - K."""
    return sorted({q for q in ladder if q <= Rp - K})


def neighbour_phase(subjects, long_sub, invoke, record, main_rows, cfg, uuid0, order):
    """Order-effect gate. Each variant re-times every ROM arm on cases 0..neighbour_cases-1:
    [long CG solve if v['cg']] -> burn-in -> [device-guard call if v['guard']] -> the arm.
    The main phase runs burn-in -> device-guard call -> subject, so the variant with cg=True, guard=True
    differs from it ONLY by the long predecessor. Ratio = variant median / main-phase median on the
    same cases; the gate is the configured variant in the configured (Table-1) scope."""
    ncase = range(cfg['neighbour_cases'])
    variants = cfg.get('neighbour_variants', [dict(name='after_cg', cg=True, guard=True)])
    rows = []
    for v in variants:
        for case in ncase:
            for i in order.permutation(len(subjects)):
                sub = subjects[int(i)]
                if v['cg']:
                    invoke(long_sub, case)
                if v.get('burn', True):
                    burn(cfg['burn_seconds'])
                if v['guard']:
                    assert gpu_uuid() == uuid0
                if v.get('sleep'):
                    time.sleep(v['sleep'])
                for _ in range(v.get('repeat', 1) - 1):      # back-to-back calls; the last one is recorded
                    invoke(sub, case)
                field, row = invoke(sub, case)
                rec = record(sub, case, 0, 'neighbour', field, row, long_sub['name'] if v['cg'] else None)
                rec['variant'] = v['name']
                rows.append(rec)
    gate_rows = []
    for v in variants:
        for sub in subjects:
            for scope in ('total_seconds', 'fused_device_seconds'):
                base = np.median([x[scope] for x in main_rows if x['name'] == sub['name'] and x['case'] in ncase])
                after = np.median([x[scope] for x in rows if x['name'] == sub['name'] and x['variant'] == v['name']])
                gate_rows.append(dict(variant=v['name'], name=sub['name'], scope=scope, main_median=float(base),
                                      after_long_median=float(after), ratio=float(after / base)))
    gv, gs = cfg.get('neighbour_gate_variant', 'after_cg'), cfg['gate_scope']
    gate = dict(rows=gate_rows, limit=cfg['neighbour_limit'], neighbour=long_sub['name'], cases=len(ncase),
                gate_scope=gs, gate_variant=gv, variants=variants,
                passed=bool(all(r['ratio'] <= cfg['neighbour_limit'] for r in gate_rows
                                if r['scope'] == gs and r['variant'] == gv)))
    return rows, gate
