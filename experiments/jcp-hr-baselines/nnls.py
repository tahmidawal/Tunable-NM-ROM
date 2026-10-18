"""Greedy Lawson-Hanson NNLS with exact passive-set least squares, support and tolerance checkpoints (DESIGN.md 3.5,
A1.8, A15.1, A16, A17).

    min_{xi >= 0} || G xi - b ||,   G (m, N) on the device, b (m,)

Each outer iteration computes the exact negative gradient w = (b - G_S xi)^T G on the device, adds the largest
positive entry (classic single-index Lawson-Hanson; frac > 0 adds batches of ceil(frac |S|), which made the path cycle on
the dry-run fixture and is not used), and refits by the Lawson-Hanson inner loop in which every
passive-set least-squares problem is solved on the ORIGINAL columns G_S (device least squares; A17.1). The earlier
normal-equation refit (Cholesky of G_S^T G_S on the host) lost the positivity of a newly entered index to round-off and
made the path cycle (A17, smk1 and the CPU dry run), which is why it was replaced. An index that the refit drops
`max_drops` times is banned from re-entry (cycle guard). Every first crossing of a tolerance tau
(||G xi - b|| <= tau ||b||) and of a support target s (>= s positive weights) is recorded.
"""
from __future__ import annotations

import math

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

_bG = jax.jit(lambda b, G: b @ G)                         # no transpose of the full G (A15.4)
_res = jax.jit(lambda G, S, x, b: b - G[:, S] @ x)


@jax.jit
def _lsq(G, S, k, b):
    """Least squares on the original columns G_S by Householder QR (A17.1). S is padded to a power of two; the padded
    columns (positions >= k) are zeroed, so they do not affect the first k unknowns (upper-triangular back substitution);
    their zero diagonal entries are replaced by 1 to keep R invertible."""
    A = G[:, S]
    keep = (jnp.arange(S.shape[0]) < k).astype(A.dtype)
    A = A * keep[None, :]
    Q, R = jnp.linalg.qr(A, mode='reduced')
    R = R + jnp.diag(1. - keep)
    return jax.scipy.linalg.solve_triangular(R, Q.T @ b, lower=False)


def _pad(idx):
    """Pad an index list to a power of two (few compiled shapes)."""
    k = len(idx)
    kp = 1 << max(0, (k - 1).bit_length())
    return np.pad(np.asarray(idx, np.int64), (0, kp - k), mode='edge'), k


def _ls(G, S, b):
    Sp, k = _pad(S)
    z = np.array(_lsq(G, jnp.asarray(Sp), k, b), dtype=np.float64, copy=True)
    return z[:k]


def _lh_refit(G, b, S, x0, max_inner=None):
    """Lawson-Hanson inner loop on the support S (x0 >= 0 feasible). Returns (x >= 0, keep mask)."""
    k = len(S)
    x = x0.copy()
    P = np.ones(k, bool)
    max_inner = max_inner or 3 * k + 10
    for _ in range(max_inner):
        idx = np.nonzero(P)[0]
        if len(idx) == 0:
            break
        z = _ls(G, S[idx], b)
        if not np.all(np.isfinite(z)):
            z = np.where(np.isfinite(z), z, -1.)
        if np.all(z > 0):
            x[:] = 0.
            x[idx] = z
            return x, P
        neg = z <= 0
        xi = x[idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            al = np.where(neg, xi / (xi - z), np.inf)
        alpha = max(0.0, min(1.0, float(np.min(al))))
        x[idx] = xi + alpha * (z - xi)
        drop = idx[(x[idx] <= 1e-300) | (neg & (al <= alpha + 1e-15))]
        x[drop] = 0.
        P[drop] = False
    keep = P & (x > 0)
    return x, keep


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


def greedy_nnls(G, b, taus=(), cap=8192, frac=0.0, log=None, max_iter=100000, supports=(), max_drops=3):
    """-> dict(records={tau: ..., ('s', s): ...}, support, weights, relres, path, reason, banned, kkt).
    reason: 'taus' (every requested checkpoint reached), 'kkt' (no admissible positive gradient and the KKT check
    passed), 'incomplete (KKT violated)' (no admissible gradient but KKT fails, e.g. only banned indices remain), 'cap'
    (support cap), 'max_iter'."""
    G = jnp.asarray(G)
    b = np.asarray(b, np.float64)
    bj = jnp.asarray(b)
    m, N = G.shape
    nb = math.sqrt(float(b @ b))
    taus = sorted(taus, reverse=True)
    supports = sorted(supports)
    rec = {t: None for t in taus}
    rec.update({('s', k): None for k in supports})
    S = np.zeros(0, np.int64)
    x = np.zeros(0)
    path = []
    reason = 'max_iter'
    banned = np.zeros(N, bool)
    drops = np.zeros(N, np.int32)
    wmax0 = float(np.max(np.abs(np.asarray(_bG(bj, G)))))

    def residual():
        if len(S) == 0:
            return bj
        Sp, k = _pad(S)
        return _res(G, jnp.asarray(Sp), jnp.asarray(np.pad(x, (0, len(Sp) - k))), bj)

    def record(rr):
        npos = int(np.sum(x > 0)) if len(S) else 0
        for k in supports:
            if rec[('s', k)] is None and npos >= k:
                rec[('s', k)] = dict(support=S[x > 0].copy(), weights=x[x > 0].copy(), relres=rr, size=npos)
        for t in taus:
            if rec[t] is None and rr <= t:
                rec[t] = dict(support=S[x > 0].copy(), weights=x[x > 0].copy(), relres=rr, size=npos)

    rr = 1.0
    for it in range(max_iter):
        r = residual()
        rr = float(jnp.linalg.norm(r)) / max(nb, 1e-300)
        record(rr)
        if (taus or supports) and all(v is not None for v in rec.values()):
            reason = 'taus'
            break
        w = np.array(_bG(r, G), dtype=np.float64, copy=True)
        if len(S):
            w[S] = -np.inf
        w[banned] = -np.inf
        cand = np.nonzero(w > 1e-14 * max(wmax0, 1e-300))[0]
        if len(cand) == 0:
            reason = 'kkt'
            break
        if len(S) >= cap:
            reason = 'cap'
            break
        B = max(1, int(math.ceil(frac * len(S))))
        B = min(B, len(cand), cap - len(S))
        new = cand[np.argsort(-w[cand], kind='stable')[:B]]
        S = np.concatenate([S, new])
        x = np.concatenate([x, np.zeros(B)])
        xs, keep = _lh_refit(G, bj, S, x)
        dropped = S[~keep]
        drops[dropped] += 1
        banned[dropped[drops[dropped] >= max_drops]] = True
        S, x = S[keep], xs[keep]
        path.append((len(S), rr))
        if log and it % 50 == 0:
            log(f'   nnls it {it}: support {len(S)} relres {rr:.3e}')
    r = residual()
    rr = float(jnp.linalg.norm(r)) / max(nb, 1e-300)
    record(rr)
    kkt = kkt_check(G, bj, S, x, nb)
    if reason == 'kkt' and not kkt['ok']:
        reason = 'incomplete (KKT violated)'
    return dict(records=rec, support=S, weights=x, relres=rr, path=path, reason=reason, banned=int(banned.sum()), kkt=kkt)
