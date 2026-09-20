"""Shallow masked autoencoder of Kim, Choi, Widemann, Zohdi (JCP 451:110841, 2022) in JAX.

Encoder  h(x) = We2 swish(We1 x + be1)            (dense, hidden width M1)
Decoder  g(z) = (S * W2) swish(W1 z + b1)         (hidden width M2=(n-1)*db+b, no output bias)

S is the 2D mask of their Section 3.2: a 1D building matrix whose row i has ones in the
b columns starting at i*db; the 2D row is the union of the building rows of the 5-point
stencil neighbours. Row i therefore has (at most) three contiguous blocks: the y-neighbour
union [i-1, i+1] of width b+2db and the two x-neighbour blocks of width b. The masked
product is stored as per-row value/index tables (n, P), P = 3b+2db, so the layer costs
n*P multiply-adds and a sub-network for a handful of outputs is a plain row gather.

All large arrays are function arguments; nothing here closes a jit over data.
"""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp


def swish(x):
    return x * jax.nn.sigmoid(x)


ACT = {'swish': swish, 'sigmoid': jax.nn.sigmoid}


def mask_tables(nx, ny, b, db):
    """Index/validity tables of the 2D mask for an nx*ny interior grid, C order (i=ix*ny+iy)."""
    assert 0 < db <= b, 'blocks of neighbouring rows must overlap or touch (audit finding 6)'
    n = nx * ny
    M2 = (n - 1) * db + b
    P = 3 * b + 2 * db
    ix, iy = np.divmod(np.arange(n), ny)
    idx = np.zeros((n, P), np.int32)
    valid = np.zeros((n, P), bool)
    # middle block: union of rows i-1, i, i+1 when those are grid (y) neighbours
    lo = np.where(iy > 0, np.arange(n) - 1, np.arange(n)) * db
    hi = np.where(iy < ny - 1, np.arange(n) + 1, np.arange(n)) * db + b
    w = b + 2 * db
    # the stored window always starts at (i-1)*db so that it is a strided window of the hidden vector
    cols = ((np.arange(n) - 1) * db)[:, None] + np.arange(w)[None]
    idx[:, :w] = cols
    valid[:, :w] = (cols >= lo[:, None]) & (cols < hi[:, None])
    # x-neighbour blocks (i-ny, i+ny); skip the part already covered by the middle block
    for k, (has, start) in enumerate(((ix > 0, (np.arange(n) - ny) * db),
                                      (ix < nx - 1, (np.arange(n) + ny) * db))):
        cols = start[:, None] + np.arange(b)[None]
        ok = has[:, None] & ((cols < lo[:, None]) | (cols >= hi[:, None])) & (cols >= 0)
        if k == 1:   # tiny grids only: do not list a column already in the i-ny block
            first = (np.arange(n) - ny) * db
            ok &= ~((ix > 0)[:, None] & (cols >= first[:, None]) & (cols < first[:, None] + b))
        idx[:, w + k * b:w + (k + 1) * b] = np.where(ok, cols, 0)
        valid[:, w + k * b:w + (k + 1) * b] = ok
    idx = np.where(valid, idx, 0)
    assert idx.max() < M2 and idx.min() >= 0
    return idx, valid, M2


def dense_mask(nx, ny, b, db):
    """Reference construction exactly as the paper words it; test use only."""
    n = nx * ny
    M2 = (n - 1) * db + b
    B = np.zeros((n, M2), bool)
    for i in range(n):
        B[i, i * db:i * db + b] = True
    S = B.copy()
    for i in range(n):
        ix, iy = divmod(i, ny)
        if iy > 0: S[i] |= B[i - 1]
        if iy < ny - 1: S[i] |= B[i + 1]
        if ix > 0: S[i] |= B[i - ny]
        if ix < nx - 1: S[i] |= B[i + ny]
    return S


def init(key, n, K, M1, M2, idx, valid, dtype):
    """PyTorch nn.Linear default (Kaiming-uniform a=sqrt5): U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    k = jax.random.split(key, 6)
    u = lambda kk, shape, fan: jax.random.uniform(kk, shape, dtype, -1., 1.) / float(np.sqrt(fan))
    return dict(We1=u(k[0], (M1, n), n), be1=u(k[1], (M1,), n), We2=u(k[2], (K, M1), M1),
                W1=u(k[3], (M2, K), K), b1=u(k[4], (M2,), K),
                W2=u(k[5], idx.shape, M2) * jnp.asarray(valid, dtype))


def encode(p, xn, act):
    return p['We2'] @ act(p['We1'] @ xn + p['be1'])


def hidden(p, z, act):
    return act(p['W1'] @ z + p['b1'])


def decode(p, z, idx, act):
    """Normalised decoder output for one latent vector; W2 is already zero where invalid.
    Gather form: used for sub-networks and online queries (small or forward-mode only)."""
    return jnp.einsum('np,np->n', p['W2'], hidden(p, z, act)[idx])


def masked_out(W2, h, ny, b, db):
    """(S*W2) h without gather/scatter, for training: every stored window is a run of b/db (+2)
    consecutive length-db blocks of h, so it is a sum of sliced products. XLA's scatter-add (the
    gather's transpose) took > 20 min per epoch on an A100 at n=3364; slices transpose to pads."""
    n = W2.shape[0]; c = b // db
    assert c * db == b and W2.shape[1] == 3 * b + 2 * db
    Hb = h.reshape(n - 1 + c, db)
    Hm = jnp.pad(Hb, ((1, 1), (0, 0)))
    Hq = jnp.pad(Hb, ((ny, ny), (0, 0)))
    Wm = W2[:, :b + 2 * db].reshape(n, c + 2, db)
    Wu = W2[:, b + 2 * db:2 * b + 2 * db].reshape(n, c, db)
    Wd = W2[:, 2 * b + 2 * db:].reshape(n, c, db)
    out = sum(jnp.sum(Wm[:, k] * Hm[k:k + n], 1) for k in range(c + 2))
    out = out + sum(jnp.sum(Wu[:, k] * Hq[k:k + n], 1) + jnp.sum(Wd[:, k] * Hq[2 * ny + k:2 * ny + k + n], 1) for k in range(c))
    return out


def n_params(p, valid):
    dense = sum(int(np.prod(p[k].shape)) for k in ('We1', 'be1', 'We2', 'W1', 'b1'))
    return dict(encoder=int(np.prod(p['We1'].shape) + p['be1'].size + np.prod(p['We2'].shape)),
                decoder=int(np.prod(p['W1'].shape) + p['b1'].size + valid.sum()),
                total=int(dense + valid.sum()))


# ---------------------------------------------------------------- training

def make_train(act, micro, ny, b, db):
    """Adam epoch over shuffled batches with micro-batch gradient accumulation.

    Returns epoch(p, opt, X, perm, lr, step0, idx, W2valid) and evalloss(p, X, idx); the index
    table is an argument, never a captured constant (334 MB at 512 intervals).
    X is the normalised snapshot matrix (N, n); perm is (nbatch, batch) int32.
    """
    def loss_rows(p, xb, idx):
        rec = jax.vmap(lambda x: masked_out(p['W2'], hidden(p, encode(p, x, act), act), ny, b, db))(xb)
        return jnp.sum((rec - xb) ** 2)

    def batch_grad(p, xb, idx):
        nb = xb.shape[0]
        m = max(d for d in range(1, min(micro, nb) + 1) if nb % d == 0)
        chunks = xb.reshape(nb // m, m, -1)
        def body(acc, c):
            l, g = jax.value_and_grad(loss_rows)(p, c, idx)
            return (acc[0] + l, jax.tree_util.tree_map(jnp.add, acc[1], g)), None
        zero = (jnp.zeros((), xb.dtype), jax.tree_util.tree_map(jnp.zeros_like, p))
        (l, g), _ = jax.lax.scan(body, zero, chunks)
        scale = 1. / (nb * xb.shape[1])          # MSE over elements, as torch MSELoss
        return l * scale, jax.tree_util.tree_map(lambda a: a * scale, g)

    def adam(p, opt, g, lr, t, b1=.9, b2=.999, eps=1e-8):
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt[0], g)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt[1], g)
        tf = t.astype(lr.dtype)
        c1, c2 = 1 - b1 ** tf, 1 - b2 ** tf
        p = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), p, m, v)
        return p, (m, v)

    @jax.jit
    def epoch(p, opt, X, perm, lr, step0, idx, W2valid):
        def body(carry, rows):
            p, opt, t = carry
            l, g = batch_grad(p, X[rows], idx)
            g['W2'] = g['W2'] * W2valid            # pruned weights stay exactly zero
            p, opt = adam(p, opt, g, lr, t)
            return (p, opt, t + 1), l
        (p, opt, _), losses = jax.lax.scan(body, (p, opt, step0), perm)
        return p, opt, jnp.mean(losses)

    @jax.jit
    def tail(p, opt, xb, lr, t, idx, W2valid):
        l, g = batch_grad(p, xb, idx)
        g['W2'] = g['W2'] * W2valid
        p, opt = adam(p, opt, g, lr, t)
        return p, opt, l

    @jax.jit
    def evalloss(p, X, idx):
        m = min(micro, X.shape[0])
        nfull = (X.shape[0] // m) * m
        tot = jax.lax.map(lambda c: loss_rows(p, c, idx), X[:nfull].reshape(-1, m, X.shape[1])).sum()
        if nfull < X.shape[0]:
            tot = tot + loss_rows(p, X[nfull:], idx)
        return tot / (X.shape[0] * X.shape[1])

    return epoch, tail, evalloss


def train(p, Xtr, Xva, idx, valid, act, *, ny, b, db, batch, micro, max_epochs, wall_seconds, seed,
          lr0=1e-3, lr_factor=.1, lr_patience=10, stop_patience=200, min_lr=1e-8, log=print, tag=''):
    """The paper's protocol: Adam, lr 1e-3, x0.1 when the TRAINING loss stagnates for 10 epochs
    (torch ReduceLROnPlateau defaults: relative threshold 1e-4), early stop when the VALIDATION
    loss has not improved for 200 epochs, at most max_epochs. The best-validation weights are kept.
    A wall budget is an addition of this lane; when it binds the run is reported as truncated.
    """
    dtype = Xtr.dtype
    idxj, validj = jnp.asarray(idx), jnp.asarray(valid, dtype)
    opt = (jax.tree_util.tree_map(jnp.zeros_like, p), jax.tree_util.tree_map(jnp.zeros_like, p))
    rng = np.random.default_rng(seed)
    N = Xtr.shape[0]
    batch = min(batch, N); micro = min(micro, batch)
    assert batch % micro == 0
    epoch, tail, evalloss = make_train(act, micro, ny, b, db)
    nb = N // batch                                  # full batches; the ragged tail is a separate step (torch drop_last=False)
    lr, best_tr, bad_tr, best_va, bad_va, best_p = lr0, np.inf, 0, np.inf, 0, p
    hist, t0, step, reason = [], time.monotonic(), 1, 'max_epochs'
    for ep in range(max_epochs):
        order = rng.permutation(N).astype(np.int32)
        p, opt, ltr = epoch(p, opt, Xtr, jnp.asarray(order[:nb * batch].reshape(nb, batch)), jnp.asarray(lr, dtype),
                            jnp.asarray(step, jnp.int32), idxj, validj)
        step += nb
        ltr = float(ltr) * nb * batch
        if N > nb * batch:
            p, opt, lt = tail(p, opt, Xtr[jnp.asarray(order[nb * batch:])], jnp.asarray(lr, dtype), jnp.asarray(step, jnp.int32), idxj, validj)
            step += 1
            ltr += float(lt) * (N - nb * batch)
        ltr, lva = ltr / N, float(evalloss(p, Xva, idxj))
        if ep == 0:
            t0 = time.monotonic()                     # the wall budget excludes compilation (first epoch)
        hist.append((ep, lr, ltr, lva, time.monotonic() - t0))
        if not (np.isfinite(ltr) and np.isfinite(lva)):
            reason = 'nonfinite'; break
        if ltr < best_tr * (1 - 1e-4): best_tr, bad_tr = ltr, 0
        else: bad_tr += 1
        if bad_tr > lr_patience:
            lr, bad_tr = max(lr * lr_factor, min_lr), 0
        if lva < best_va: best_va, bad_va, best_p = lva, 0, p
        else: bad_va += 1
        if ep % 25 == 0 or bad_va == 0 and ep % 5 == 0:
            log(f'{tag} ep={ep} lr={lr:.1e} train={ltr:.4e} val={lva:.4e} best={best_va:.4e} t={hist[-1][4]:.0f}s')
        if bad_va >= stop_patience:
            reason = 'early_stop'; break
        if time.monotonic() - t0 > wall_seconds:
            reason = 'wall_budget'; break
    return best_p, dict(stop_reason=reason, truncated=bool(reason == 'wall_budget'), epochs=len(hist), best_val=best_va, final_lr=lr,
                        seconds=time.monotonic() - t0, history=np.asarray(hist))


# ---------------------------------------------------------------- hyper-reduction helpers

def pod_basis(R, nr):
    """Left singular vectors of a (n, N) snapshot matrix through the N*N Gram in f64."""
    R = np.asarray(R, np.float64)
    if R.shape[1] <= R.shape[0]:
        w, V = np.linalg.eigh(R.T @ R)
        order = np.argsort(w)[::-1][:nr]
        U = R @ (V[:, order] / np.sqrt(np.maximum(w[order], 1e-300)))
    else:
        w, V = np.linalg.eigh(R @ R.T)
        U = V[:, np.argsort(w)[::-1][:nr]]
    U, _ = np.linalg.qr(U)
    return U


def greedy_samples(Phi, nz):
    """Oversampled greedy selection minimising the gappy reconstruction error of the basis
    columns (Carlberg et al. GNAT Algorithm 3 / Choi et al. SNS Algorithm 5 style)."""
    n, nr = Phi.shape
    assert nr <= nz <= n
    per = [nz // nr + (1 if j < nz % nr else 0) for j in range(nr)]
    chosen = []
    for j in range(nr):
        for _ in range(per[j]):
            if j == 0 or not chosen:
                r = np.abs(Phi[:, 0]).copy()
            else:
                Z = Phi[chosen]
                c = np.linalg.lstsq(Z[:, :j], Z[:, j], rcond=None)[0]
                r = np.abs(Phi[:, j] - Phi[:, :j] @ c)
            r[chosen] = -1.
            chosen.append(int(np.argmax(r)))
    assert len(set(chosen)) == nz
    return np.asarray(sorted(chosen))


def subnet(p, idx, valid, rows):
    """Active-path sub-network computing decoder outputs `rows` only (their Section 4.4.1)."""
    rows = np.asarray(rows)
    sub_idx, sub_valid = np.asarray(idx)[rows], np.asarray(valid)[rows]
    active = np.unique(sub_idx[sub_valid])
    remap = np.searchsorted(active, np.where(sub_valid, sub_idx, active[0]))
    return dict(W1=p['W1'][active], b1=p['b1'][active], W2=p['W2'][rows]), jnp.asarray(remap.astype(np.int32)), int(active.size)
