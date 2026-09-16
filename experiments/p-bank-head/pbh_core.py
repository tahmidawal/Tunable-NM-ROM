"""Shared numerics for the p-bank-head cell: data, banks, floors, oracles, POD.

Everything here is mesh-explicit and memory-bounded. The two drivers
(`pbh_train.py`, `pbh_solve.py`) import this module; the evaluation path itself
is the unchanged head-ablation machinery in `arms.py` / `poisson_ablation.py`.

Staged flat: every module sits beside this file on the cluster.
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

import core as C
import sep_common as sc
import arms as A


# --------------------------------------------------------------- utilities ---

def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def weights_sha(params):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


def dump(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def descriptor(draws):
    """The family's own normalised parameter descriptor (ms_parametric.sample_params).

    OFFLINE ONLY. It regularises the training codes; no online query ever sees it.
    """
    d = np.asarray(draws, dtype=float)
    cx, cy, w, a = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    return np.column_stack(((cx - 0.5) / 0.35, (cy - 0.5) / 0.35,
                            (np.log(w) - np.log(0.045)) / 0.8, (a - 1.25) / 0.75))


def fit_validation_split(count, seed, validation_fraction):
    order = np.random.default_rng(seed).permutation(count)
    nval = max(1, int(round(validation_fraction * count)))
    val = np.sort(order[:nval])
    fit = np.sort(order[nval:])
    return fit, val


# -------------------------------------------------------------------- data ---

def fields(draws, intervals, chunk=64):
    """Same-mesh FD-DST interior solutions, exactly `pabl01`'s snapshot route."""
    lam = jnp.asarray(C.eigenvalues(intervals))
    out = []
    for s in range(0, len(draws), chunk):
        block = jnp.asarray(np.stack([C.full_source(intervals, q) for q in draws[s:s + chunk]]))
        sol = jax.vmap(lambda f: C.dst_solve(f, lam))(block)
        out.append(sol[:, 1:-1, 1:-1].reshape(len(block), -1))
        del block, sol
    U = jnp.concatenate(out)
    U.block_until_ready()
    return U


def coords_of(intervals):
    p = np.arange(1, intervals) / intervals
    xx, yy = np.meshgrid(p, p, indexing='ij')
    return np.column_stack((xx.ravel(), yy.ravel()))


def bank_of(params, intervals, chunk=16384):
    """The cached bank G, built exactly as `core.assemble` builds it."""
    xy = coords_of(intervals)
    feat = jax.jit(sc.features)
    parts = []
    for s in range(0, len(xy), chunk):
        block = feat(params, jnp.asarray(xy[s:s + chunk]))
        block.block_until_ready()
        parts.append(block)
    G = jnp.concatenate(parts, axis=0)
    G.block_until_ready()
    return G


def bank_r(G):
    """Thin R factor without ever materialising Q (4.3 GB at R=512, n=1024)."""
    R = jnp.linalg.qr(jnp.asarray(G), mode='r')
    values = np.asarray(jnp.linalg.svd(R, compute_uv=False))
    threshold = float(values[0] * max(np.asarray(G).shape) * np.finfo(float).eps)
    rank = int((values > threshold).sum())
    info = dict(rank=rank, rank_valid=rank == int(R.shape[1]), rank_threshold=threshold,
                condition_number=float(values[0] / values[-1]),
                singular_values=values.tolist(), R_sha256=sha_array(R))
    return R, info


def project_targets(G, R, U):
    """T = Q^T u and the perpendicular energy, from R alone:  R^T T = G^T u."""
    y = jnp.asarray(U) @ jnp.asarray(G)                       # (S, R)
    T = jax.scipy.linalg.solve_triangular(jnp.asarray(R).T, y.T, lower=True).T
    nu2 = jnp.sum(jnp.asarray(U) ** 2, axis=1)
    perp2 = jnp.clip(nu2 - jnp.sum(T * T, axis=1), 0., None)
    return T, perp2, nu2


def bank_floor(G, R, draws, intervals, chunk=64):
    """Worst/median/per-source relative bank projection floor, streamed by source."""
    lam = jnp.asarray(C.eigenvalues(intervals))
    errs = []
    for s in range(0, len(draws), chunk):
        block = jnp.asarray(np.stack([C.full_source(intervals, q) for q in draws[s:s + chunk]]))
        sol = jax.vmap(lambda f: C.dst_solve(f, lam))(block)
        U = sol[:, 1:-1, 1:-1].reshape(len(block), -1)
        _, perp2, nu2 = project_targets(G, R, U)
        errs.append(np.asarray(jnp.sqrt(perp2 / nu2)))
        del block, sol, U
    e = np.concatenate(errs)
    return e


def summarise(errors):
    e = np.asarray(errors, dtype=float)
    return dict(worst=float(e.max()), median=float(np.median(e)),
                mean=float(e.mean()), count=int(e.size), per_case=e.tolist())


# ------------------------------------------------------------- weak operators -

def sine_ops(intervals, modes):
    """The retained sine test family, exactly `core.assemble`'s construction."""
    p = np.arange(1, intervals)
    lam = C.eigenvalues(intervals)
    I, J = np.nonzero(lam <= np.sort(lam.ravel())[modes - 1])
    maxmode = int(max(I.max(), J.max())) + 1
    S = np.sqrt(2.0 / intervals) * np.sin(np.pi * np.outer(p, np.arange(1, maxmode + 1)) / intervals)
    return dict(S=jnp.asarray(S), I=jnp.asarray(I), J=jnp.asarray(J),
                W=jnp.asarray(lam[I, J] ** -1), retained_modes=int(len(I)), intervals=intervals)


def reduce_bank(V, ops, n):
    """Scaled sine coefficients of every bank column (`poisson_ablation.reduce_bank`)."""
    cubes = jnp.asarray(V).reshape((n - 1, n - 1, np.asarray(V).shape[-1]))
    c = jnp.einsum('xa,xyr,yb->abr', ops['S'], cubes, ops['S'])
    return c[ops['I'], ops['J']]


def weak_sources(draws, ops, n, chunk=64):
    """f_m = Lambda^{-1} Phi^T f for every source, the exact online projection."""
    S, I, J, W = ops['S'], ops['I'], ops['J'], ops['W']

    @jax.jit
    def one(src):
        c = S.T @ src[1:-1, 1:-1] @ S
        return c[I, J] * W
    out = []
    for s in range(0, len(draws), chunk):
        block = jnp.asarray(np.stack([C.full_source(n, q) for q in draws[s:s + chunk]]))
        out.append(jax.vmap(one)(block))
        del block
    F = jnp.concatenate(out)
    F.block_until_ready()
    return F


# ------------------------------------------------------------- head oracles ---

def oracle_errors(head, Rg, Zcand, T, perp2, nu2, budget, starts=8, gtol=1e-6,
                  linear='gj', block=64):
    """Worst/median best-found error over a cohort, multistart from candidate codes.

    Sources and starts are vmapped together, so one dispatch solves
    block x starts independent LM problems of dimension K with an R-dimensional
    residual. Returns per-source (error, iterations, exit reason)."""
    K = int(np.asarray(Zcand).shape[1])
    lm = A.make_stationary_lm(lambda z, t, R: R @ head(z) - t, budget, gtol=gtol, linear=linear)

    @jax.jit
    def fit(starts_b, T_b, R):
        out = jax.vmap(lambda zs, t: jax.vmap(lambda z0: lm(z0, (t, R), 0.))(zs))(starts_b, T_b)
        best = jnp.argmin(out[1], axis=1)
        idx = jnp.arange(out[1].shape[0])
        return out[1][idx, best], out[2][idx, best], out[3][idx, best]

    H = jax.jit(jax.vmap(head))(jnp.asarray(Zcand))
    Hr = H @ jnp.asarray(Rg).T
    Hn = jnp.sum(Hr * Hr, axis=1)
    Zc = jnp.asarray(Zcand)
    Rgj = jnp.asarray(Rg)
    total = int(T.shape[0])
    block = min(block, total)
    errs, iters, reasons = [], [], []
    for s in range(0, total, block):
        # Every block is padded to the SAME width by repeating its last row, so
        # `fit` is compiled exactly once however long the cohort is. Compilation
        # of this nested-vmap LM costs minutes at R=512, and the padded rows
        # cannot change the real ones: a batched while_loop freezes each element
        # once its own predicate is false.
        take = np.arange(s, min(s + block, total))
        keep = len(take)
        if keep < block:
            take = np.concatenate((take, np.full(block - keep, take[-1])))
        Tb = jnp.asarray(np.asarray(T)[take])
        score = Hn[None, :] - 2. * (Tb @ Hr.T)
        pick = jnp.argsort(score, axis=1)[:, :starts]
        rn, it, reason = jax.device_get(fit(Zc[pick], Tb, Rgj))
        p2 = np.asarray(perp2)[take[:keep]]
        n2 = np.asarray(nu2)[take[:keep]]
        errs.append(np.sqrt(np.asarray(rn)[:keep] ** 2 + p2) / np.sqrt(n2))
        iters.append(np.asarray(it)[:keep])
        reasons.append(np.asarray(reason)[:keep])
    return (np.concatenate(errs), np.concatenate(iters).astype(int),
            np.concatenate(reasons).astype(int))


def stored_code_errors(head, Rg, Z, T, perp2, nu2):
    """Head error AT the stored codes: the training fit the optimizer produced."""
    H = jax.jit(jax.vmap(head))(jnp.asarray(Z)) @ jnp.asarray(Rg).T
    res2 = jnp.sum((H - T) ** 2, axis=1)
    return np.asarray(jnp.sqrt((res2 + perp2) / nu2))


# ----------------------------------------------------------- streaming POD ----

def streaming_pod(draws, intervals, maxrank, block=512):
    """Exact POD of the same-mesh FD snapshots without holding the S x n matrix.

    Gram is assembled blockwise with regeneration (a 1024-interval DST solve is
    sub-millisecond, so regeneration is cheaper than the 25 GB residency the
    direct route would need at S = 3072).
    """
    t0 = time.perf_counter()
    S = len(draws)
    bounds = [(i, min(i + block, S)) for i in range(0, S, block)]
    gram = np.zeros((S, S))
    for a, (i0, i1) in enumerate(bounds):
        Ua = fields(draws[i0:i1], intervals)
        gram[i0:i1, i0:i1] = np.asarray(Ua @ Ua.T)
        for b, (j0, j1) in enumerate(bounds):
            if b <= a:
                continue
            Ub = fields(draws[j0:j1], intervals)
            g = np.asarray(Ua @ Ub.T)
            gram[i0:i1, j0:j1] = g
            gram[j0:j1, i0:i1] = g.T
            del Ub
        del Ua
    w, V = jnp.linalg.eigh(jnp.asarray(gram))
    w, V = w[::-1], V[:, ::-1]
    energy = float(jnp.sum(jnp.clip(w, 0., None)))
    Wt = V[:, :maxrank] / jnp.sqrt(jnp.clip(w[:maxrank], 1e-300, None))[None, :]
    nint = (intervals - 1) ** 2
    modes = jnp.zeros((nint, maxrank))
    for (i0, i1) in bounds:
        Ua = fields(draws[i0:i1], intervals)
        modes = modes + Ua.T @ Wt[i0:i1]
        del Ua
    coords = []
    for (i0, i1) in bounds:
        Ua = fields(draws[i0:i1], intervals)
        coords.append(np.asarray(Ua @ modes))
        del Ua
    modes.block_until_ready()
    info = dict(sources=S, blocks=len(bounds), intervals=intervals, maxrank=int(maxrank),
                total_energy=energy, eigenvalues=np.asarray(w[:maxrank]).tolist(),
                gram_sha256=sha_array(gram), modes_sha256=sha_array(modes),
                seconds=time.perf_counter() - t0)
    return modes, np.concatenate(coords), info
