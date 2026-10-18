"""Greedy Lawson-Hanson NNLS on the normal system, with tolerance checkpoints (DESIGN.md 3.5, A1.8).

    min_{xi >= 0} || G xi - b ||,   G (m, N) on the device, b (m,)

The support grows by the largest positive entries of the negative gradient w = G^T (b - G xi), in batches of
max(1, ceil(frac * |S|)); after each batch the support problem is refitted by the Lawson-Hanson inner loop on the
normal equations A_SS = G_S^T G_S (Cholesky on the host). G^T G_S is maintained incrementally, so the cost of one path
is about m * N * |S_final| flops on the device. Every first crossing of a tolerance tau (||G xi - b|| <= tau ||b||) is
recorded with its support and weights. Without tolerances the loop runs to the NNLS optimum (no positive gradient).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

_GtG = jax.jit(lambda G, cols: (G[:, cols].T @ G).T)      # no transpose of the full G (smk1: 27 GB copy, A15.4)
_bG = jax.jit(lambda b, G: b @ G)


def _lh_refit(A, g, x0, act_tol=0.0, max_inner=None, ls_cols=None):
    """Lawson-Hanson inner loop on the normal system restricted to the current support.
    A (k, k) = G_S^T G_S, g (k,) = G_S^T b, x0 (k,) >= 0 feasible start. Returns x (k,) >= 0 and a keep mask.
    ls_cols(idx) -> least-squares solution on the ORIGINAL columns G_S[:, idx] (A4.7), used when the passive-set
    Cholesky fails or is inaccurate."""
    k = len(g)
    x = x0.copy()
    P = np.ones(k, bool)
    max_inner = max_inner or 3 * k + 10
    for _ in range(max_inner):
        idx = np.nonzero(P)[0]
        if len(idx) == 0:
            break
        As = A[np.ix_(idx, idx)]
        z = None
        try:
            cf = sla.cho_factor(As, lower=True, check_finite=False)
            z = sla.cho_solve(cf, g[idx], check_finite=False)
            if not np.all(np.isfinite(z)) or np.linalg.norm(As @ z - g[idx]) > 1e-8 * (np.linalg.norm(g[idx]) + 1e-300):
                z = None
        except np.linalg.LinAlgError:
            z = None
        if z is None:          # (near-)dependent support columns: least squares on the original columns (A4.7)
            z = ls_cols(idx) if ls_cols is not None else sla.lstsq(As, g[idx], cond=1e-13, check_finite=False)[0]
        if np.all(z > act_tol):
            x[:] = 0.
            x[idx] = z
            return x, P
        neg = z <= act_tol
        xi = x[idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            al = np.where(neg, xi / (xi - z), np.inf)
        alpha = float(np.min(al))
        alpha = max(0.0, min(1.0, alpha))
        x[idx] = xi + alpha * (z - xi)
        drop = idx[(x[idx] <= 1e-300) | (neg & (al <= alpha + 1e-15))]
        x[drop] = 0.
        P[drop] = False
    return x, P


def kkt_check(G, bj, S, x, nb):
    """KKT diagnostics (A5.2), w = G^T (b - G xi): xi >= 0 and finite; |w_i| <= tol on the support; w_i <= tol off it
    (banned indices included). tol = 1e-9 ||G||_F ||b||."""
    G = jnp.asarray(G)
    S = np.asarray(S, np.int64)
    x = np.asarray(x, np.float64)
    res_ = bj - (G[:, jnp.asarray(S)] @ jnp.asarray(x) if len(S) else 0.)
    w = np.array(_bG(res_, G), dtype=np.float64, copy=True)        # writable copy; no transpose of G
    tol = 1e-9 * float(jnp.linalg.norm(G)) * nb
    on = np.zeros(len(w), bool)
    on[S] = True
    sup = float(np.max(np.abs(w[on]))) if on.any() else 0.
    off = float(np.max(w[~on])) if (~on).any() else -np.inf
    ok = bool(np.all(np.isfinite(w)) and np.all(np.isfinite(x)) and np.all(x >= 0) and sup <= tol and off <= tol)
    return dict(ok=ok, support_max_abs_gradient=sup, off_support_max_gradient=off, tol=tol)


def greedy_nnls(G, b, taus=(), cap=8192, frac=0.02, log=None, max_iter=100000, supports=(), max_drops=3):
    """-> dict(records={tau: dict(support, weights, relres) or None}, support, weights, relres, path, reason).
    reason: 'taus' (all tolerances reached), 'kkt' (no admissible positive gradient and the KKT check passed),
    'incomplete (KKT violated)' (no admissible gradient but KKT fails, e.g. only banned indices remain), 'cap' (support
    cap), 'max_iter'. `kkt` holds the final KKT diagnostics in every case; `banned` counts banned indices.
    supports: support-size checkpoints (A15.1): the first state with >= s positive weights is recorded under ('s', s).
    An index is banned from re-entry only after it has been dropped `max_drops` times (A15.3; LH drops are normal)."""
    G = jnp.asarray(G)
    b = np.asarray(b, np.float64)
    m, N = G.shape
    Gb = np.asarray(_bG(jnp.asarray(b), G))
    bb = float(b @ b)
    nb = math.sqrt(bb)
    taus = sorted(taus, reverse=True)
    rec = {t: None for t in taus}
    supports = sorted(supports)
    rec.update({('s', k): None for k in supports})
    S = np.zeros(0, np.int64)            # support (node indices)
    slots = np.zeros(0, np.int64)        # their columns in the preallocated G^T G_S buffer
    width = 2 * cap + 64
    buf = np.zeros((N, width))
    used = 0
    x = np.zeros(0)
    path = []
    reason = 'max_iter'
    bj = jnp.asarray(b)
    banned = np.zeros(N, bool)          # indices dropped max_drops times: never re-added (cycle guard)
    drops = np.zeros(N, np.int32)

    def relres():
        if len(S) == 0:
            return 1.0
        A = buf[S][:, slots]
        r2 = bb - 2 * x @ Gb[S] + x @ A @ x
        return math.sqrt(max(r2, 0.0)) / max(nb, 1e-300)

    def exact_relres():
        """||G_S x - b|| / ||b|| evaluated directly (the normal-equation estimate loses accuracy below ~1e-8)."""
        if len(S) == 0:
            return 1.0
        return float(jnp.linalg.norm(G[:, jnp.asarray(S)] @ jnp.asarray(x) - bj)) / max(nb, 1e-300)

    def record(rr):
        npos = int(np.sum(x > 0)) if len(S) else 0
        for k in supports:
            if rec[('s', k)] is None and npos >= k:
                rec[('s', k)] = dict(support=S[x > 0].copy(), weights=x[x > 0].copy(), relres=exact_relres(), size=npos)
        for t in taus:
            if rec[t] is None and rr <= t:
                ex = exact_relres()
                if ex <= t:
                    rec[t] = dict(support=S[x > 0].copy(), weights=x[x > 0].copy(), relres=ex, size=int(np.sum(x > 0)))

    for it in range(max_iter):
        rr = relres()
        record(rr)
        if (taus or supports) and all(v is not None for v in rec.values()):
            reason = 'taus'
            break
        w = Gb - (buf[:, slots] @ x if len(S) else 0.)
        w = np.asarray(w, np.float64)
        if len(S):
            w[S] = -np.inf
        w[banned] = -np.inf
        cand = np.nonzero(w > 1e-14 * max(np.max(np.abs(Gb)), 1e-300))[0]
        if len(cand) == 0:
            reason = 'kkt'
            break
        if len(S) >= cap:
            reason = 'cap'
            break
        B = max(1, int(math.ceil(frac * len(S))))
        B = min(B, len(cand), cap - len(S))
        new = cand[np.argsort(-w[cand], kind='stable')[:B]]
        if used + B > width:                       # compact the buffer (columns of dropped nodes are freed)
            buf[:, :len(S)] = buf[:, slots]
            slots = np.arange(len(S))
            used = len(S)
        Bp = 1 << (B - 1).bit_length()             # pad to a power of two: few compiled shapes
        cols = np.asarray(_GtG(G, jnp.asarray(np.pad(new, (0, Bp - B), mode='edge'))))[:, :B]
        buf[:, used:used + B] = cols
        slots = np.concatenate([slots, np.arange(used, used + B)])
        used += B
        S = np.concatenate([S, new])
        x = np.concatenate([x, np.zeros(B)])
        A = buf[S][:, slots]
        Scur = S.copy()
        ls = lambda idx, Sc=Scur: np.asarray(jnp.linalg.lstsq(G[:, jnp.asarray(Sc[idx])], bj, rcond=1e-13)[0])
        xs, keep = _lh_refit(A, Gb[S], x, ls_cols=ls)
        dropped = S[~keep]
        drops[dropped] += 1
        banned[dropped[drops[dropped] >= max_drops]] = True
        S, x, slots = S[keep], xs[keep], slots[keep]
        path.append((len(S), relres()))
        if log and it % 50 == 0:
            log(f'   nnls it {it}: support {len(S)} relres {path[-1][1]:.3e}')
    rr = relres()
    record(rr)
    rr = exact_relres()
    kkt = kkt_check(G, bj, S, x, nb)
    if reason == 'kkt' and not kkt['ok']:      # A4.7 / A5.2: a claimed optimum must satisfy KKT over ALL candidates
        reason = 'incomplete (KKT violated)'
    return dict(records=rec, support=S, weights=x, relres=rr, path=path, reason=reason, banned=int(banned.sum()), kkt=kkt)
