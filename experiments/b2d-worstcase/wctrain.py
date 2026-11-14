"""b2d-worstcase: fine-tune the coordinate bank for the LINEAR RUNG with a worst-case-aware span objective.

The linear rung only uses span{g_1..g_R} (the head is dropped), so the bank is trained directly on what the rung
needs: the relative projection error of every training snapshot onto the span,

    e_i(theta)^2 = min_c || u_i - G_theta c ||^2 / || u_{0,i} ||^2          (grid 2-norm, the paper's metric),

aggregated over a batch by a power mean  ( mean_i e_i^(2p) )^(1/p)  (p = 1: mean relative squared error; p > 1
leans on the worst snapshots), with hard-example sampling (part of every batch is drawn with probability
proportional to (current error)^alpha). The inner least squares is solved exactly per step (Cholesky of the Gram);
by the envelope theorem its solution carries no gradient. A scale-free Gram penalty keeps the bank conditioned.

Data: training trajectories P[lo:hi] of the baseline lanes' pool (26 time levels each, float32, read-only), held on
the GPU; validation P[4352:4608] is only monitored (projection error at the six output times), never trained on.

Start: the deployed checkpoint's bank, last layer folded with the first R columns of the deployed rotation (exact:
the fine-tune starts from the deployed width-R span), or random init (--scratch).

Writes <out>/bank.pkl (params B, g, out_scale; cfg), <out>/rotation.npz (T: orthonormal-on-grid columns ordered by
training energy of the relative projections; codes: rotated coefficients of a training subsample, for the trust
radius) and <out>/history.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import engines as e
import sep_common as sc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', required=True)
    p.add_argument('--pool-meta', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rotation', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--R', type=int, default=384)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-end', type=float, default=0.02)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--batch', type=int, default=384)
    p.add_argument('--points', type=int, default=0, help='0 = full interior grid every step')
    p.add_argument('--power', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=2.0, help='hard sampling exponent')
    p.add_argument('--hard-frac', type=float, default=0.5)
    p.add_argument('--lam-orth', type=float, default=1e-3)
    p.add_argument('--train-lo', type=int, default=0)
    p.add_argument('--train-hi', type=int, default=4352)
    p.add_argument('--val-lo', type=int, default=4352)
    p.add_argument('--val-hi', type=int, default=4608)
    p.add_argument('--levels', type=int, default=1, help='keep every k-th of the 26 levels')
    p.add_argument('--train-B', type=int, default=0)
    p.add_argument('--scratch', type=int, default=0)
    p.add_argument('--g-hidden', type=int, default=1024)
    p.add_argument('--g-layers', type=int, default=2)
    p.add_argument('--n-ff', type=int, default=128)
    p.add_argument('--ff-scale', type=float, default=4.0)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--time-cap', type=float, default=0., help='seconds; 0 = none')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow-cpu', type=int, default=0)
    p.add_argument('--smoke-traj', type=int, default=0, help='CPU smoke only: pool holds this many trajectories')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert a.allow_cpu or jax.default_backend() == 'gpu'
    print(f'jax_backend={jax.default_backend()} args={vars(a)}', flush=True)
    t_begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - t_begin, 1)
    L = 256
    n = (L - 1) ** 2
    rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------------ data ----
    meta = np.load(a.pool_meta)
    P, n0_all = np.asarray(meta['params']), np.asarray(meta['n0'])
    Pref = np.concatenate((e.params_draw(0, 576), e.params_draw(1000, 4032)))
    npool = a.smoke_traj or 4608
    assert np.array_equal(P, Pref[:npool]), 'pool parameter vectors differ from P'
    F = np.load(a.pool, mmap_mode='r')
    assert F.shape == (npool, 26, (L + 1) ** 2), F.shape
    lv = np.arange(0, 26, a.levels)
    inter = (np.arange(1, L)[:, None] * (L + 1) + np.arange(1, L)[None, :]).reshape(-1)

    def load(lo, hi, levels):
        parts = []
        for s in range(lo, hi, 64):
            blk = np.asarray(F[s:min(s + 64, hi)])[:, levels][:, :, inter]
            parts.append(jnp.asarray(blk.reshape(-1, n)))
        return jnp.concatenate(parts)

    Utr = load(a.train_lo, a.train_hi, lv)                                   # (Ntr * len(lv), n) float32 on device
    ntr = a.train_hi - a.train_lo
    n0tr = jnp.asarray(np.repeat(n0_all[a.train_lo:a.train_hi], len(lv)))
    vl = np.array([0, 5, 10, 15, 20, 25])
    Uva = load(a.val_lo, a.val_hi, vl)
    nva = a.val_hi - a.val_lo
    n0va = jnp.asarray(np.repeat(n0_all[a.val_lo:a.val_hi], len(vl)))
    nsnap = int(Utr.shape[0])
    print(f'DATA train {ntr} traj x {len(lv)} levels = {nsnap} snapshots ({Utr.nbytes/2**30:.1f} GiB), '
          f'val {nva} traj', el(), flush=True)
    xy = jnp.asarray(e.coords(L))

    # ----------------------------------------------------------------- model ----
    ck = pickle.load(open(a.checkpoint, 'rb'))
    if a.scratch:
        full = sc.init_separable(jax.random.PRNGKey(a.seed), 2, a.R, n_ff=a.n_ff, ff_scale=a.ff_scale,
                                 g_hidden=a.g_hidden, g_layers=a.g_layers,
                                 out_scale=float(np.asarray(ck['params']['out_scale'])))
        params = dict(B=full['B'], g=full['g'], out_scale=full['out_scale'])
    else:
        T0 = jnp.asarray(np.load(a.rotation)['T'][:, :a.R])
        g = [(jnp.asarray(w), jnp.asarray(b)) for w, b in ck['params']['g']]
        w, b = g[-1]
        g[-1] = (w @ T0, b @ T0)
        # rotated columns are orthonormal on the grid (norm 1); rescale to unit mean square so Adam sees O(1) outputs
        g[-1] = (g[-1][0] * np.sqrt(n), g[-1][1] * np.sqrt(n))
        params = dict(B=jnp.asarray(ck['params']['B']), g=g, out_scale=jnp.asarray(1.0))
    R = a.R
    train_keys = ['g'] + (['B'] if a.train_B else [])

    def feats(tp, fixed, pts):
        return sc.features(dict(fixed, **tp), pts)

    def proj_err2(G, U, n0, scale):
        """Relative squared projection errors of the rows of U (b, m) onto span G (m, R); no gradient through c."""
        gram = G.T @ G
        tr = jnp.trace(gram) / R
        cf = jax.scipy.linalg.cho_factor(gram + 1e-12 * tr * jnp.eye(R))
        C = jax.lax.stop_gradient(jax.scipy.linalg.cho_solve(cf, G.T @ U.T))             # (R, b)
        res = U - (G @ C).T
        return scale * jnp.sum(res * res, axis=1) / n0 ** 2, gram / tr

    def loss_fn(tp, fixed, Ub, n0b, pidx):
        pts = xy[pidx]
        G = feats(tp, fixed, pts)
        e2, gn = proj_err2(G, Ub[:, pidx].astype(jnp.float64), n0b, n / pidx.shape[0])
        agg = jnp.mean(e2 ** a.power) ** (1. / a.power)
        orth = jnp.mean((gn - jnp.eye(R)) ** 2)
        return agg + a.lam_orth * orth, (e2, agg, orth)

    sched = optax.warmup_cosine_decay_schedule(0., a.lr, a.warmup, a.steps, a.lr * a.lr_end)
    opt = optax.adam(sched)

    @jax.jit
    def step(tp, ema, st, fixed, U, n0, idx, pidx):
        Ub = U[idx]
        (val, (e2, agg, orth)), gr = jax.value_and_grad(loss_fn, has_aux=True)(tp, fixed, Ub, n0[idx], pidx)
        upd, st = opt.update(gr, st)
        tp = optax.apply_updates(tp, upd)
        ema = jax.tree_util.tree_map(lambda m, q: a.ema * m + (1 - a.ema) * q, ema, tp)
        return tp, ema, st, e2, agg, orth

    @jax.jit
    def errs_chunk(Q, U, n0, idx):
        Ub = U[idx].astype(jnp.float64)
        r = Ub - (Ub @ Q) @ Q.T
        return jnp.sqrt(jnp.sum(r * r, axis=1)) / n0[idx]

    @jax.jit
    def qr_grid(tp, fixed):
        return jnp.linalg.qr(feats(tp, fixed, xy), mode='reduced')

    def all_errs(tp, fixed, U, n0, chunk=1024):
        Q, _ = qr_grid(tp, fixed)
        N = int(U.shape[0])
        outs = [np.asarray(errs_chunk(Q, U, n0, jnp.arange(s, min(s + chunk, N)))) for s in range(0, N - N % chunk, chunk)]
        if N % chunk:
            outs.append(np.asarray(errs_chunk(Q, U, n0, jnp.arange(N - N % chunk, N))))
        return np.concatenate(outs)

    def val_stats(tp, fixed):
        ev = all_errs(tp, fixed, Uva, n0va, chunk=len(vl) * 64).reshape(nva, len(vl))
        m = ev[:, 1:].max(1)
        return dict(worst=float(m.max()), median=float(np.median(m)), p95=float(np.quantile(m, .95)),
                    mean=float(m.mean()), t0_worst=float(ev[:, 0].max()))

    def train_stats(err):
        m = err.reshape(ntr, len(lv)).max(1)
        return dict(worst=float(m.max()), median=float(np.median(m)), p95=float(np.quantile(m, .95)),
                    p99=float(np.quantile(m, .99)), mean=float(m.mean()))

    tp = {k: params[k] for k in train_keys}
    fixed = {k: v for k, v in params.items() if k not in train_keys}
    ema = tp
    st = opt.init(tp)
    hist = dict(args=vars(a), job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
                host=os.uname().nodename, snapshots=nsnap, evals=[])
    err_tab = all_errs(tp, fixed, Utr, n0tr)
    ev0 = dict(step=0, seconds=el(), train=train_stats(err_tab), val=val_stats(tp, fixed), which='init')
    hist['evals'].append(ev0)
    print('EVAL', json.dumps(ev0), flush=True)
    nh = int(round(a.batch * a.hard_frac))
    full_grid = jnp.arange(n)
    t_train = time.perf_counter()
    done = 0
    for i in range(a.steps):
        idx_u = rng.integers(0, nsnap, a.batch - nh)
        if nh:
            pr = err_tab ** a.alpha
            idx_h = rng.choice(nsnap, nh, replace=False, p=pr / pr.sum())
            idx = np.concatenate((idx_u, idx_h))
        else:
            idx = idx_u
        pidx = jnp.asarray(np.sort(rng.choice(n, a.points, replace=False))) if a.points else full_grid
        tp, ema, st, e2, agg, orth = step(tp, ema, st, fixed, Utr, n0tr, jnp.asarray(idx), pidx)
        if nh or (i + 1) % 100 == 0:
            eb = np.sqrt(np.asarray(e2))
            assert np.isfinite(eb).all(), f'non-finite loss at step {i}'
            err_tab[idx] = eb
        done = i + 1
        if (i + 1) % 100 == 0:
            print(f'step {i+1} agg {float(agg):.3e} sqrt {np.sqrt(float(agg)):.3e} batch max {eb.max():.3e} '
                  f'orth {float(orth):.2e} [{time.perf_counter()-t_train:.0f}s]', flush=True)
        capped = a.time_cap and (time.perf_counter() - t_begin) > a.time_cap
        if (i + 1) % a.eval_every == 0 or i + 1 == a.steps or capped:
            err_e = all_errs(ema, fixed, Utr, n0tr)
            err_tab = all_errs(tp, fixed, Utr, n0tr)
            ev = dict(step=i + 1, seconds=el(), train=train_stats(err_e), val=val_stats(ema, fixed), which='ema',
                      val_raw=val_stats(tp, fixed))
            hist['evals'].append(ev)
            print('EVAL', json.dumps(ev), flush=True)
            (out / 'history.json').write_text(json.dumps(hist))
        if capped:
            print('TIME CAP reached at step', i + 1, flush=True)
            break
    hist['steps_done'] = done
    hist['train_seconds'] = time.perf_counter() - t_train

    # ------------------------------------------------ final rotation, codes ----
    final = dict(fixed, **ema)
    G = sc.features(final, xy)
    Q, Rg = jnp.linalg.qr(G, mode='reduced')
    acc = jnp.zeros((R, R))
    cov = jax.jit(lambda Q_, U, n0, idx: (lambda c: c.T @ c)((U[idx].astype(jnp.float64) @ Q_) / n0[idx][:, None]))
    for s in range(0, nsnap, 1024):
        acc = acc + cov(Q, Utr, n0tr, jnp.arange(s, min(s + 1024, nsnap)))
    evals, V = np.linalg.eigh(np.asarray(acc))
    V = V[:, ::-1]
    T = np.linalg.solve(np.asarray(Rg), V)                     # G T = Q V: orthonormal on the grid, energy-ordered
    sub = np.sort(rng.choice(nsnap, min(16384, nsnap), replace=False))
    codes = np.concatenate([np.asarray(Utr[jnp.asarray(sub[s:s + 1024])].astype(jnp.float64) @ Q) @ V
                            for s in range(0, len(sub), 1024)])
    GT = np.asarray(G) @ T
    hist['rotation'] = dict(orthonormality_deviation=float(np.linalg.norm(GT.T @ GT - np.eye(R))),
                            energy_spectrum_head=evals[::-1][:8].tolist(), energy_spectrum_tail=evals[:8].tolist(),
                            Rg_condition=float(np.linalg.cond(np.asarray(Rg))))
    np.savez(out / 'rotation.npz', T=T, codes=codes)
    host = jax.tree_util.tree_map(np.asarray, final)
    with open(out / 'bank.pkl', 'wb') as f:
        pickle.dump(dict(params=host, cfg=dict(vars(a), kind='b2d-worstcase span fine-tune', steps_done=done)), f)
    hist['bank_sha256'] = hashlib.sha256((out / 'bank.pkl').read_bytes()).hexdigest()
    hist['rotation_sha256'] = hashlib.sha256((out / 'rotation.npz').read_bytes()).hexdigest()
    hist['seconds'] = el()
    (out / 'history.json').write_text(json.dumps(hist))
    print('TRAIN-DONE', json.dumps(hist['rotation']), el(), flush=True)


if __name__ == '__main__':
    main()
