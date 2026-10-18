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
import time

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
    d = jnp.abs(jnp.diag(R))
    R = R + jnp.diag(1. - keep)
    return jax.scipy.linalg.solve_triangular(R, Q.T @ b, lower=False), d


_lsq_svd = jax.jit(lambda G, S, b: jnp.linalg.lstsq(G[:, S], b, rcond=1e-13)[0])


def _pad(idx, m=None):
    """Pad an index list to a power of two (few compiled shapes), never beyond the row count m."""
    k = len(idx)
    kp = 1 << max(0, (k - 1).bit_length())
    if m is not None:
        kp = max(k, min(kp, m))
    return np.pad(np.asarray(idx, np.int64), (0, kp - k), mode='edge'), k


def _ls(G, S, b):
    """Passive-set least squares (A17.1, A18.2): Householder QR on the original columns; if the support has more columns
    than rows, or QR shows numerical rank deficiency (min |R_ii| <= 1e-13 max |R_ii| over the first k), the minimum-norm
    SVD least-squares solution (A7.2) is used instead."""
    m = G.shape[0]
    k = len(S)
    if k > m:
        return np.array(_lsq_svd(G, jnp.asarray(S), b), dtype=np.float64, copy=True)
    Sp, k = _pad(S, m)
    zj, dj = _lsq(G, jnp.asarray(Sp), k, b)
    d = np.asarray(dj)[:k]
    if d.size and float(np.min(d)) <= 1e-13 * float(np.max(d)):
        return np.array(_lsq_svd(G, jnp.asarray(S), b), dtype=np.float64, copy=True)
    z = np.array(zj, dtype=np.float64, copy=True)
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


# ------------------------------------------------------------------------------------- updating QR (A18.5, A20) ---
# Thin QR of G_S held in fixed-shape device buffers: Q (m, K), R (K, K), qtb (K,), with k active columns. Unused columns
# of Q are zero; the unused block of R is the identity, so triangular solves on the full K x K system leave the first k
# unknowns exact (back substitution) and set the rest to 0 (qtb is zero there).

def _qr_append_impl(Q, R, qtb, k, g, b):
    """Append column g (classical Gram-Schmidt with one reorthogonalisation, CGS2). Returns (Q, R, qtb, rho, gnorm)."""
    c1 = Q.T @ g
    v = g - Q @ c1
    c2 = Q.T @ v
    v = v - Q @ c2
    c = c1 + c2
    rho = jnp.linalg.norm(v)
    q = v / jnp.where(rho > 0, rho, 1.)
    Q = Q.at[:, k].set(q)
    col = jnp.where(jnp.arange(R.shape[0]) < k, c, 0.).at[k].set(rho)
    R = R.at[:, k].set(col)
    qtb = qtb.at[k].set(q @ b)
    return Q, R, qtb, rho, jnp.linalg.norm(g)


_qr_append = jax.jit(_qr_append_impl, donate_argnums=(0, 1, 2))     # buffers updated in place (A22.2)


def _qr_reset_impl(Q, R, qtb, k):
    """Undo a rejected append: restore slot k to the unused state (zero Q column, identity R column, zero qtb)."""
    K = R.shape[0]
    return Q.at[:, k].set(0.), R.at[:, k].set(jnp.zeros(K).at[k].set(1.)), qtb.at[k].set(0.)


_qr_reset = jax.jit(_qr_reset_impl, donate_argnums=(0, 1, 2))


def _qr_delete_givens_impl(Q, R, qtb, k, j):
    """Reference (A20 original): delete active column j by sequential Givens rotations. Kept for the regression
    fixture only: its loop-carried column updates are not done in place by XLA at production widths (A38.1)."""
    K = R.shape[0]
    ar = jnp.arange(K)
    src = jnp.where((ar >= j) & (ar < k - 1), ar + 1, ar)
    R = R[:, src]

    def rot(i, c):
        Q, R, qtb = c
        a, bb = R[i, i], R[i + 1, i]
        h = jnp.hypot(a, bb)
        cs = jnp.where(h > 0, a / jnp.where(h > 0, h, 1.), 1.)
        sn = jnp.where(h > 0, bb / jnp.where(h > 0, h, 1.), 0.)
        ri, ri1 = R[i], R[i + 1]
        R = R.at[i].set(cs * ri + sn * ri1).at[i + 1].set(-sn * ri + cs * ri1)
        ti, ti1 = qtb[i], qtb[i + 1]
        qtb = qtb.at[i].set(cs * ti + sn * ti1).at[i + 1].set(-sn * ti + cs * ti1)
        qi, qi1 = Q[:, i], Q[:, i + 1]
        Q = Q.at[:, i].set(cs * qi + sn * qi1).at[:, i + 1].set(-sn * qi + cs * qi1)
        return Q, R, qtb
    Q, R, qtb = jax.lax.fori_loop(j, k - 1, rot, (Q, R, qtb))
    last = k - 1
    Q = Q.at[:, last].set(0.)
    R = R.at[last, :].set(0.).at[:, last].set(0.).at[last, last].set(1.)
    qtb = qtb.at[last].set(0.)
    return Q, R, qtb


_qr_delete_givens = jax.jit(_qr_delete_givens_impl)


def _qr_delete_window_impl(Q, R, qtb, k, j, w):
    """A38.1: delete active column j (0 <= j < k) by one blocked re-triangularisation. After shifting R's columns
    j+1..k-1 left, rows j..k-1 form an upper-Hessenberg block; a static window of width w (a power of two >= k - j)
    starting at j0 = min(j, K - w) contains it. Householder QR of the w x w window (sign-normalised to a non-negative
    diagonal, so the already-triangular part and the inactive identity part are left unchanged) gives Qs; then
    R_window <- Rs, qtb_window <- Qs^T qtb_window, Q[:, window] <- Q[:, window] Qs. Columns right of the window are
    inactive (>= k) with zero entries in these rows; columns left of it are zero in these rows. Finally slot k-1 is
    cleared. Cost O(m w^2 + w^3), no sequential loop."""
    K = R.shape[0]
    ar = jnp.arange(K)
    src = jnp.where((ar >= j) & (ar < k - 1), ar + 1, ar)
    R = R[:, src]
    j0 = jnp.minimum(j, K - w)
    B = jax.lax.dynamic_slice(R, (j0, j0), (w, w))
    Qs, Rs = jnp.linalg.qr(B)
    dg = jnp.diag(Rs)
    sg = jnp.where(dg < 0, -1., 1.)
    Qs = Qs * sg[None, :]
    Rs = Rs * sg[:, None]
    R = jax.lax.dynamic_update_slice(R, Rs, (j0, j0))
    qw = jax.lax.dynamic_slice(qtb, (j0,), (w,))
    qtb = jax.lax.dynamic_update_slice(qtb, Qs.T @ qw, (j0,))
    Qw = jax.lax.dynamic_slice(Q, (0, j0), (Q.shape[0], w))
    Q = jax.lax.dynamic_update_slice(Q, Qw @ Qs, (0, j0))
    last = k - 1
    Q = Q.at[:, last].set(0.)
    R = R.at[last, :].set(0.).at[:, last].set(0.).at[last, last].set(1.)
    qtb = qtb.at[last].set(0.)
    return Q, R, qtb


_qr_delete_window = jax.jit(_qr_delete_window_impl, static_argnums=(5,), donate_argnums=(0, 1, 2))


def _qr_delete(Q, R, qtb, k, j):
    """Delete active column j of the updating factorisation (A38.1: blocked window re-triangularisation)."""
    K = R.shape[0]
    span = max(int(k) - int(j), 2)
    w = min(K, 1 << (span - 1).bit_length())
    w = max(w, min(K, 64))
    return _qr_delete_window(Q, R, qtb, k, j, w)





@jax.jit
def _qr_solve(R, qtb):
    return jax.scipy.linalg.solve_triangular(R, qtb, lower=False)


@jax.jit
def _drift(Qb, k):
    """max |Q_k^T Q_k - I| over the active columns of a bucket slice Qb (columns >= k are zero and masked out)."""
    keep = (jnp.arange(Qb.shape[1]) < k).astype(Qb.dtype)
    M = Qb.T @ Qb - jnp.diag(keep)
    return jnp.max(jnp.abs(M))


@jax.jit
def _qr_active(G, Sb, kb_mask, b):
    """Householder QR of the active columns gathered at a bucket width kb (A22.1); padded columns are zero."""
    A = G[:, Sb] * kb_mask[None, :]
    Qa, Ra = jnp.linalg.qr(A, mode='reduced')
    sg = jnp.where(jnp.diag(Ra) < 0, -1., 1.)                  # A39.2: non-negative diagonal invariant
    Qa, Ra = Qa * sg[None, :], Ra * sg[:, None]
    Qa = Qa * kb_mask[None, :]
    Ra = Ra * kb_mask[:, None] * kb_mask[None, :] + jnp.diag(1. - kb_mask)
    return Qa, Ra, Qa.T @ b


DRIFT_TOL = 1e-8                       # A22.3: accepted orthogonality drift between refactorisations


class UpdQR:
    """Updating thin QR of G_S (A20). Columns are appended at the end and deleted anywhere; a full refactorisation runs
    every `refactor_every` appends, and the orthogonality drift max|Q^T Q - I| is checked at each refactorisation
    (recorded)."""

    def __init__(self, G, b, K, refactor_every=1024):
        m = G.shape[0]
        K = min(K, m)                               # A22.4: capacity never exceeds the row count
        self.G, self.b, self.K = G, b, K
        self.Q = jnp.zeros((m, K))
        self.R = jnp.eye(K)
        self.qtb = jnp.zeros(K)
        self.k = 0
        self.appends = 0
        self.refactor_every = refactor_every
        self.drift = []
        self.rmax = 0.

    def append(self, col, rank_tol=1e-12):
        """Append column `col` of G. Returns False (factorisation unchanged) if the column is numerically dependent:
        its orthogonal remainder rho <= rank_tol * ||g|| (A20.3)."""
        g = self.G[:, int(col)]
        self.Q, self.R, self.qtb, rho, gn = _qr_append(self.Q, self.R, self.qtb, self.k, g, self.b)
        rho, gn = float(rho), float(gn)
        if not (rho > rank_tol * gn):
            self.Q, self.R, self.qtb = _qr_reset(self.Q, self.R, self.qtb, self.k)
            return False
        self.k += 1
        self.appends += 1
        return True

    def delete(self, pos):
        self.Q, self.R, self.qtb = _qr_delete(self.Q, self.R, self.qtb, self.k, int(pos))
        self.k -= 1

    def refactor(self, S):
        """Drift check, then Householder refactorisation of the ACTIVE columns only (bucket width = next multiple of
        512, A22.1), written back into the fixed buffers. A drift above DRIFT_TOL is recorded as a stability event."""
        k = len(S)
        kb = min(self.K, max(512, -(-k // 512) * 512)) if k else 0
        if k:
            self.drift.append(float(_drift(self.Q[:, :kb], k)))
        if kb:
            Sb = np.pad(np.asarray(S, np.int64), (0, kb - k), mode='edge')
            mask = (np.arange(kb) < k).astype(np.float64)
            Qa, Ra, qa = _qr_active(self.G, jnp.asarray(Sb), jnp.asarray(mask), self.b)
            self.Q = self.Q.at[:, :kb].set(Qa).at[:, kb:].set(0.)
            self.R = jnp.eye(self.K).at[:kb, :kb].set(Ra)
            self.qtb = jnp.zeros(self.K).at[:kb].set(qa)
            del Qa, Ra, qa
        self.k = k

    def solve(self):
        return np.array(_qr_solve(self.R, self.qtb), dtype=np.float64, copy=True)[:self.k]


def _lh_refit_upd(F, S, x0, max_inner=None, deadline=None):
    """Lawson-Hanson inner loop using the updating factorisation F of G_S (columns in the order of S). Returns
    (x, S) after removing the dropped columns from both."""
    S = list(S)
    x = x0.copy()
    max_inner = max_inner or 3 * len(S) + 10
    dropped = []
    for _ in range(max_inner):
        if not S:
            break
        if deadline is not None and time.perf_counter() > deadline:
            break                               # A39.1: x stays feasible (>= 0) and consistent with F and S
        z = F.solve()
        if not np.all(np.isfinite(z)):
            z = np.where(np.isfinite(z), z, -1.)
        if np.all(z > 0):
            return z, S, dropped
        neg = z <= 0
        with np.errstate(divide='ignore', invalid='ignore'):
            al = np.where(neg, x / (x - z), np.inf)
        alpha = max(0.0, min(1.0, float(np.min(al))))
        x = x + alpha * (z - x)
        dmask = (x <= 1e-300) | (neg & (al <= alpha + 1e-15))
        for pos in sorted(np.nonzero(dmask)[0].tolist(), reverse=True):
            # a started deletion batch is completed (it keeps S, x and F consistent); overshoot <= one batch
            F.delete(pos)
            dropped.append(S.pop(pos))
            x = np.delete(x, pos)
    return np.maximum(x, 0.), S, dropped


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


def greedy_nnls(G, b, taus=(), cap=8192, frac=0.0, log=None, max_iter=100000, supports=(), max_drops=3,
                method='update', refactor_every=1024, max_seconds=None):
    """-> dict(records={tau: ..., ('s', s): ...}, support, weights, relres, path, reason, banned, kkt).
    reason: 'time' (the per-path wall budget max_seconds was exceeded, A38.2/A39.1: checked before every outer iteration,
    inside the Lawson-Hanson loop and before refactorisation; overshoot <= one append / solve / deletion batch),
    'taus' (every requested checkpoint reached), 'kkt' (no admissible positive gradient and the KKT check
    passed), 'incomplete (KKT violated)' (no admissible gradient but KKT fails, e.g. only banned indices remain), 'cap'
    (support cap), 'max_iter'. method 'update' (A20, default: updating QR, single-index entry) or 'full' (A17: full QR
    of the passive set at every refit; the reference the updating path is checked against)."""
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
    F = None
    dep_skip = np.zeros(N, bool)
    n_dep = 0
    n_drift_events = 0

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
    t_start = time.perf_counter()
    deadline = (t_start + max_seconds) if max_seconds is not None else None
    for it in range(max_iter):
        if deadline is not None and time.perf_counter() > deadline:
            reason = 'time'                 # A38.2: per-path wall budget; unreached targets = construction failures
            break
        r = residual()
        rr = float(jnp.linalg.norm(r)) / max(nb, 1e-300)
        if path:
            path[-1] = (path[-1][0], rr)      # (support, residual) of the same iterate
        record(rr)
        if (taus or supports) and all(v is not None for v in rec.values()):
            reason = 'taus'
            break
        w = np.array(_bG(r, G), dtype=np.float64, copy=True)
        if len(S):
            w[S] = -np.inf
        w[banned] = -np.inf
        w[dep_skip] = -np.inf
        cand = np.nonzero(w > 1e-14 * max(wmax0, 1e-300))[0]
        if len(cand) == 0:
            reason = 'kkt'
            break
        if len(S) >= cap:
            reason = 'cap'
            break
        if method == 'update':
            if F is None:
                F = UpdQR(G, bj, K=min(m, int(1 << max(0, (min(cap, m) - 1).bit_length()))),
                          refactor_every=refactor_every)
            new = int(cand[np.argmax(w[cand])])
            if not F.append(new):                   # numerically dependent on the support (A20.3): skipped until the
                dep_skip[new] = True                # support loses a column (it cannot enter before), never banned
                n_dep += 1
                continue
            S = np.concatenate([S, [new]]).astype(np.int64)
            x = np.concatenate([x, [0.]])
            xs, Sl, dropped = _lh_refit_upd(F, S, x, deadline=deadline)
            S, x = np.asarray(Sl, np.int64), xs
            for dd in dropped:
                drops[dd] += 1
                if drops[dd] >= max_drops:
                    banned[dd] = True
            if dropped:
                dep_skip[:] = False
            if F.appends and F.appends % F.refactor_every == 0 and not (deadline and time.perf_counter() > deadline):
                F.refactor(S)
                if F.drift and not (F.drift[-1] <= DRIFT_TOL):   # A22.3: non-finite or excessive drift = event
                    n_drift_events += 1
        else:
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
        path.append((len(S), None))          # residual of this iterate is filled in at the next iteration (A18.4)
        if log and it % 50 == 0:
            log(f'   nnls it {it}: support {len(S)} relres {rr:.3e}')
    r = residual()
    rr = float(jnp.linalg.norm(r)) / max(nb, 1e-300)
    if path and path[-1][1] is None:
        path[-1] = (path[-1][0], rr)
    record(rr)
    # raw evidence for offline checks (A18.1): checkpoint residual vectors, |b|, drop counts
    for v in rec.values():
        if v is not None and 'resvec' not in v:
            Sv, xv = v['support'], v['weights']
            v['resvec'] = np.asarray(bj - G[:, jnp.asarray(Sv)] @ jnp.asarray(xv)) if len(Sv) else np.asarray(b)
    kkt = kkt_check(G, bj, S, x, nb)
    if reason == 'kkt' and not kkt['ok']:
        reason = 'incomplete (KKT violated)'
    return dict(records=rec, support=S, weights=x, relres=rr, path=path, reason=reason, banned=int(banned.sum()), kkt=kkt,
                drops=drops, bnorm=nb, method=method, qr_drift=(F.drift if F is not None else []), dependent_skips=n_dep,
                refactor_every=refactor_every, capacity=(F.K if F is not None else None), drift_events=n_drift_events,
                drift_ok=bool(F is None or not F.drift or all(v <= DRIFT_TOL for v in F.drift)),
                terminal=dict(support=np.asarray(S), weights=np.asarray(x), resvec=np.asarray(r)),
                budget_seconds=max_seconds, elapsed_seconds=time.perf_counter() - t_start)
