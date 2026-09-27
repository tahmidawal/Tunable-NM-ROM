"""PDE-agnostic numerics for the bank-floor lane.

A bank is a list of BLOCKS; each block is a `sep_common` parameter dict of which
only (B, g, out_scale) are used, and the bank is the column concatenation of the
blocks' `sep_common.features`. A one-block bank built from an incumbent
checkpoint is that checkpoint's bank exactly.

Floors are always measured in a twice-orthonormalised basis with an explicit
residual; large arrays are always explicit jit arguments.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import sep_common as sc

F64 = jnp.float64
BANK_KEYS = ('B', 'g', 'out_scale')


# --------------------------------------------------------------- utilities ---

def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def weights_sha(tree):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


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
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def summarise(errors):
    e = np.asarray(errors, dtype=float).ravel()
    return dict(worst=float(e.max()), median=float(np.median(e)), mean=float(e.mean()),
                rms=float(np.sqrt(np.mean(e * e))), count=int(e.size))


# ------------------------------------------------------------------- banks ---

def bank_block(params):
    """The bank-only part of a checkpoint's parameters, as f64 device arrays."""
    return dict(B=jnp.asarray(params['B'], F64),
                g=[(jnp.asarray(w, F64), jnp.asarray(b, F64)) for w, b in params['g']],
                out_scale=jnp.asarray(params['out_scale'], F64))


def fresh_block(key, r_feat, n_ff, ff_scales, g_hidden, g_layers, out_scale):
    p = sc.init_separable(key, 2, r_feat, n_ff=n_ff, g_hidden=g_hidden, g_layers=g_layers,
                          h_hidden=4, h_layers=1, out_scale=float(out_scale),
                          ff_scales=list(ff_scales))
    return bank_block(p)


def features(blocks, xy):
    return jnp.concatenate([sc.features(b, xy) for b in blocks], axis=1)


_features_jit = jax.jit(features)


def bank_of(blocks, xy, chunk=16384):
    xy = np.asarray(xy)
    parts = []
    for s in range(0, len(xy), chunk):
        parts.append(_features_jit(blocks, jnp.asarray(xy[s:s + chunk])))
    G = jnp.concatenate(parts, axis=0)
    G.block_until_ready()
    return G


def grid(intervals):
    p = np.arange(1, intervals) / intervals
    xx, yy = np.meshgrid(p, p, indexing='ij')
    return np.column_stack((xx.ravel(), yy.ravel()))


# ------------------------------------------------ orthonormal basis, guarded --

def _qr_r(G, retries=2):
    for attempt in range(retries + 1):
        Rf = jnp.linalg.qr(G, mode='r')
        if bool(jnp.isfinite(Rf).all()):
            return Rf, attempt
        print(f'WARNING: non-finite QR factor, attempt {attempt + 1}', flush=True)
    raise AssertionError('QR factor non-finite after retries')


def orth_basis(G):
    """Twice-orthonormalised basis of span(G) with a numerical-rank assert.

    Returns (Q (n, rank), info). Never returns a non-finite basis."""
    G = jnp.asarray(G, F64)
    assert bool(jnp.isfinite(G).all()), 'bank has non-finite entries'
    n, R = G.shape
    Rf, retried = _qr_r(G)
    _, S, Vt = jnp.linalg.svd(Rf, full_matrices=False)
    S = np.asarray(S)
    assert np.isfinite(S).all() and S[0] > 0, 'singular values non-finite'
    thr = float(S[0] * max(n, R) * np.finfo(float).eps)
    rank = int((S > thr).sum())
    assert rank >= 1
    Q1 = G @ (Vt[:rank].T / jnp.asarray(S[:rank])[None, :])
    Q, _ = jnp.linalg.qr(Q1)
    if not bool(jnp.isfinite(Q).all()):
        Q, _ = jnp.linalg.qr(Q1)
    assert bool(jnp.isfinite(Q).all()), 'orthonormal basis non-finite'
    dev = float(jnp.max(jnp.abs(Q.T @ Q - jnp.eye(rank, dtype=F64))))
    assert dev < 1e-10, f'orthonormality deviation {dev:.2e}'
    info = dict(columns=int(R), rank=rank, rank_valid=bool(rank == R), rank_threshold=thr,
                condition_number=float(S[0] / S[-1]), sigma_max=float(S[0]),
                sigma_min=float(S[-1]), qr_retries=int(retried), orthonormality_deviation=dev)
    return Q, info


@jax.jit
def _floor_block(Q, U):
    E = U - (U @ Q) @ Q.T
    return jnp.sqrt(jnp.sum(E * E, axis=1) / jnp.sum(U * U, axis=1))


def floors(Q, U, chunk=1024):
    """Per-snapshot relative projection floor, explicit residual."""
    out = []
    for s in range(0, U.shape[0], chunk):
        out.append(np.asarray(_floor_block(Q, jnp.asarray(U[s:s + chunk]))))
    e = np.concatenate(out)
    assert np.isfinite(e).all(), 'non-finite floor'
    return e


# ----------------------------------------------------------- POD from Gram ---

def gram(U, block=2048):
    """S x S Gram of device-resident snapshots, blockwise, f64."""
    S = U.shape[0]
    Gm = np.zeros((S, S))
    mm = jax.jit(lambda a, b: a @ b.T)
    for i in range(0, S, block):
        Ui = U[i:i + block]
        for j in range(i, S, block):
            g = np.asarray(mm(Ui, U[j:j + block]))
            Gm[i:i + g.shape[0], j:j + g.shape[1]] = g
            if j > i:
                Gm[j:j + g.shape[1], i:i + g.shape[0]] = g.T
    return Gm


def pod_deflated(U, scale, maxrank, idx=None, ranks=(), stage_ratio=1e-5, block=2048):
    """POD of {scale_i u_i : i in idx} by the snapshot Gram WITH DEFLATION.

    A single Gram eigen-decomposition squares the conditioning and is unreliable once
    sigma_r / sigma_1 < ~1e-7 (job 4052480 died on exactly that assert: Poisson 2D has
    sigma_2048 / sigma_1 = 1.1e-8). So each stage accepts only the modes with
    sigma_k / sigma_stage_1 > stage_ratio, removes them from the snapshots explicitly, and
    the next stage works on the well-scaled residual. Columns stay in descending order, so
    the first r columns span the leading rank-r POD subspace (nested prefixes).

    `scale` = 1/||u_i|| gives the optimum of the mean RELATIVE squared error; 1 = classical POD.
    Known-answer data: `tail_mean_sq[r]` = (residual energy after r modes) / snapshot count,
    from stage traces minus accepted eigenvalues (never from sums of tiny eigenvalues)."""
    idx = np.arange(U.shape[0]) if idx is None else np.asarray(idx)
    t0 = time.perf_counter()
    # built on the HOST and uploaded once: only one S x n array is ever device-resident here
    Ur = jnp.asarray(np.asarray(U)[idx] * np.asarray(scale)[idx][:, None])
    S, n = Ur.shape
    r_goal = int(min(maxrank, S))
    Q = jnp.zeros((n, 0), F64)
    tails, stages = {}, []
    defl = jax.jit(lambda ur, q: ur - (ur @ q) @ q.T, donate_argnums=0)
    acc = jax.jit(lambda m, u, wt: m + u.T @ wt)
    total = None
    while Q.shape[1] < r_goal:
        Gs = gram(Ur, block)
        tr = float(np.trace(Gs))
        total = tr if total is None else total
        w, V = jnp.linalg.eigh(jnp.asarray(Gs))
        del Gs
        w = np.asarray(w)[::-1]
        V = np.asarray(V[:, ::-1])
        assert np.isfinite(w).all() and w[0] > 0, 'Gram eigenvalues non-finite'
        good = int((np.sqrt(np.clip(w, 0, None) / w[0]) > stage_ratio).sum())
        take = int(min(good, r_goal - Q.shape[1]))
        assert take >= 1
        off = int(Q.shape[1])
        csum = np.concatenate(([0.], np.cumsum(w[:take])))
        for k in ranks:
            if off <= k <= off + take:
                tails[str(k)] = dict(mean_sq=float(max(tr - csum[k - off], 0.) / S),
                                     stage=len(stages), resolvable=bool((tr - csum[k - off]) / tr > 1e-8))
        Wt = V[:, :take] / np.sqrt(w[:take])[None, :]
        modes = jnp.zeros((n, take), F64)
        for a_ in range(0, S, block):
            modes = acc(modes, Ur[a_:a_ + block], jnp.asarray(Wt[a_:a_ + block]))
        for _ in range(2):                                        # twice is enough
            modes = modes - Q @ (Q.T @ modes)
            modes, _r = jnp.linalg.qr(modes)                      # order-preserving
        Q = jnp.concatenate([Q, modes], axis=1)
        Ur = defl(Ur, modes)
        stages.append(dict(offset=off, accepted=take, trace=tr, sigma_first=float(np.sqrt(w[0])),
                           sigma_last_accepted=float(np.sqrt(w[take - 1]))))
        print(f'   pod stage {len(stages)}: +{take} modes (total {Q.shape[1]}), '
              f'sigma {np.sqrt(w[0]):.3e} -> {np.sqrt(w[take - 1]):.3e} [{time.perf_counter() - t0:.0f}s]',
              flush=True)
    del Ur
    r = int(Q.shape[1])
    assert bool(jnp.isfinite(Q).all()), 'POD basis non-finite'
    dev = float(jnp.max(jnp.abs(Q.T @ Q - jnp.eye(r, dtype=F64))))
    assert dev < 1e-10, f'POD orthonormality deviation {dev:.2e}'
    info = dict(columns=r, rank=r, rank_valid=True, orthonormality_deviation=dev, snapshots=int(S),
                stages=stages, tail_mean_sq=tails, energy=total,
                sigma_ratio_r_over_1=float(stages[-1]['sigma_last_accepted'] / stages[0]['sigma_first']),
                seconds=time.perf_counter() - t0)
    return Q, info


# ------------------------------------------------ variable-projection training

def varpro_loss(blocks, xy, Ub, inv_un2, ridge):
    G = features(blocks, xy)
    s = jnp.sqrt(jnp.sum(G * G, axis=0)) + 1e-300      # differentiated: with a ridge it is not a pure gauge
    Gn = G / s[None, :]
    Gm = Gn.T @ Gn + ridge * jnp.eye(Gn.shape[1], dtype=F64)
    Y = Gn.T @ Ub.T                                              # (R, B)
    Lc = jnp.linalg.cholesky(Gm)
    Cc = jax.scipy.linalg.cho_solve((Lc, True), Y)               # (R, B)
    E = Ub - (Gn @ Cc).T
    return jnp.mean(inv_un2 * jnp.sum(E * E, axis=1))


def train_varpro(blocks, lrs, xy, U, steps, batch, seed, ridge=1e-9, log_every=500, tag='',
                 probe=None):
    """Adam on the bank blocks only. `lrs[i]` is block i's peak learning rate
    (0 freezes it). Fixed budget, final iterate, no selection."""
    S = U.shape[0]
    xy = jnp.asarray(xy)
    inv_un2 = 1.0 / jnp.sum(U * U, axis=1)
    nb = int(min(batch, S))

    def sched(lr):
        return optax.warmup_cosine_decay_schedule(0., lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    labels = [jax.tree_util.tree_map(lambda _: f'b{i}', b) for i, b in enumerate(blocks)]
    tx = {f'b{i}': (optax.adam(sched(lr)) if lr > 0 else optax.set_to_zero())
          for i, lr in enumerate(lrs)}
    opt = optax.multi_transform(tx, labels)
    state = opt.init(blocks)

    @jax.jit
    def step(bl, st, kk, U_, inv_):
        idx = jax.random.choice(kk, S, shape=(nb,), replace=False)
        val, gr = jax.value_and_grad(varpro_loss)(bl, xy, U_[idx], inv_[idx], ridge)
        for g in gr:
            g['out_scale'] = jnp.zeros_like(g['out_scale'])
        upd, st = opt.update(gr, st, bl)
        return optax.apply_updates(bl, upd), st, val

    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    curve = []
    ema = None
    for i in range(steps):
        key, kk = jax.random.split(key)
        blocks, state, val = step(blocks, state, kk, U, inv_un2)
        if (i + 1) % log_every == 0 or i == 0:
            v = float(val)
            assert np.isfinite(v), f'[{tag}] non-finite loss at step {i + 1}'
            ema = v if ema is None else 0.5 * ema + 0.5 * v
            row = dict(step=i + 1, batch_loss=v, batch_rms_floor=float(np.sqrt(v)),
                       seconds=time.time() - t0)
            if probe is not None and ((i + 1) % (log_every * 10) == 0 or i == 0):
                row['probe'] = probe(blocks)
            curve.append(row)
            print(f'   vp[{tag}] {i + 1:6d}/{steps} rms-floor {np.sqrt(v):.4e} '
                  f'{row.get("probe", "")} [{time.time() - t0:.0f}s]', flush=True)
    jax.block_until_ready(blocks)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(blocks)), \
        f'[{tag}] non-finite parameters after training'
    return blocks, dict(steps=int(steps), batch=nb, lrs=[float(x) for x in lrs], ridge=ridge,
                        seed=int(seed), seconds=time.time() - t0, curve=curve)


# --------------------------------------------------------------- cost model --

def cost_model(n, R, K, n_ref=None):
    """Analytic online-cost proxies for a bank of rank R on n points."""
    M_full = 4 * (K + R)
    return dict(R=int(R), decode_flops_per_field=int(n) * int(R),
                bank_bytes=8 * int(n) * int(R),
                full_bank_tests_M=int(M_full), full_bank_operator_entries=int(M_full) * int(R),
                q0_operator_entries=4 * K * int(R))
