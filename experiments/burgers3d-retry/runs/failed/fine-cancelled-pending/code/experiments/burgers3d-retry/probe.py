"""burgers3d-retry probes (diagnostic only; no validation or held-out cohort is touched).

Seeds used here are PROBE seeds, disjoint from every cohort of either lane:
    probe training 923601 (prefix-stable rows), probe validation 923651.

mode 'podscale' (one mesh): snapshots of the probe training trajectories at the training steps and of the probe
    validation trajectories at the six output times, on the mesh's point group (every interior node up to 250 047
    points, else the fixed random subset of train.py, seed 923002). POD of the normalised training snapshots for
    each nested training-set size; worst / rms validation projection floor by rank. Also the frozen R = 512 bank's
    floor on the same validation fields and points. Answers: does more training data lower the floor, and where
    is the linear n-width floor at this mesh?

mode 'fine' (meshes list): full-grid reference fields of a few probe validation cases at each mesh, the frozen bank's
    full-grid floor by R' (chunked Gram, then an exact residual pass), and Newton-BiCGStab GPU times of a few FOM
    settings (compiled once, median of reps). Answers: how expensive is the FOM at 257 nodes, and how accurate is
    the frozen bank there?
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'burgers3d-span'))
import common as C                 # noqa: E402
import train as TR                 # noqa: E402

T0 = time.perf_counter()


def log(s):
    print(f'[{time.perf_counter() - T0:.1f}s] {s}', flush=True)


def load_bank(model_dir):
    b = pickle.loads((Path(model_dir) / 'bank.pkl').read_bytes())
    return jax.tree_util.tree_map(jnp.asarray, b['params']), np.asarray(b['rotation'])


def floors_points(bank, T, x, V, ladder):
    """Projection floor of rows of V (normalised) onto G_hat[:, :R'] at points x (P small enough for one QR)."""
    G = C.bank_at(bank, T, x)
    Vn = jnp.asarray(V / np.linalg.norm(V, axis=1, keepdims=True))
    X = jax.jit(lambda G, Vn: Vn @ jnp.linalg.qr(G, mode='reduced')[0])(G, Vn)   # jitted (eager NaN on GB10)
    assert bool(jnp.all(jnp.isfinite(X)))
    out = {}
    for r in ladder:
        e = jnp.sqrt(jnp.maximum(1 - jnp.sum(X[:, :r] ** 2, axis=1), 0.))
        out[str(r)] = dict(worst=float(jnp.max(e)), rms=float(jnp.sqrt(jnp.mean(e ** 2))))
    return out


def chunked_gram(U, W=None, chunk=16384):
    """U (S, P) host -> U U^T (and U W^T) on the GPU, chunked over points."""
    S = U.shape[0]
    Gm = jnp.zeros((S, S))
    X = None if W is None else jnp.zeros((S, W.shape[0]))
    f = jax.jit(lambda a: a @ a.T)
    g = jax.jit(lambda a, b: a @ b.T)
    for s in range(0, U.shape[1], chunk):
        a = jnp.asarray(U[:, s:s + chunk])
        Gm = Gm + f(a)
        if W is not None:
            X = X + g(a, jnp.asarray(W[:, s:s + chunk]))
    return Gm, X


def podscale(cfg, out):
    n = cfg['mesh']
    idx, x = TR.group_points(n, dict(group_points_cap=cfg['points_cap'], subset_seed=cfg['subset_seed']))
    ttr = C.table(cfg['train_seed'], max(cfg['train_sizes']))
    tva = C.table(cfg['val_seed'], cfg['val_count'])
    gcfg = dict(data_ntol=1e-8, data_ltol=1e-9)
    U, _, wtr = TR.generate(gcfg, ttr, range(max(cfg['train_sizes'])), n, cfg['train_steps'], idx, log)
    V, _, wva = TR.generate(gcfg, tva, range(cfg['val_count']), n, [0, 10, 20, 30, 40, 50], idx, log)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    Gm, X = chunked_gram(U, Vn)
    log(f'gram {Gm.shape} cross {X.shape}')
    per = len(cfg['train_steps'])
    res = dict(config=cfg, points=int(len(idx)), worst_train_residual=wtr, worst_val_residual=wva, pod={})
    for s in cfg['train_sizes']:
        S = s * per
        lam, w = jnp.linalg.eigh(Gm[:S, :S])
        lam, w = lam[::-1], w[:, ::-1]
        proj = (w.T @ X[:S]) ** 2 / jnp.maximum(lam, 1e-300)[:, None]      # (S, V) energy per mode
        cum = jnp.cumsum(proj, axis=0)
        row = {}
        for r in cfg['ranks']:
            if r > S:
                continue
            e = jnp.sqrt(jnp.maximum(1 - cum[r - 1], 0.))
            row[str(r)] = dict(worst=float(jnp.max(e)), rms=float(jnp.sqrt(jnp.mean(e ** 2))))
        res['pod'][str(s)] = row
        log(f'train size {s}: ' + ' '.join(f"R{r}={v['worst']:.4f}/{v['rms']:.4f}" for r, v in row.items()))
    bank, T = load_bank(cfg['model'])
    res['bank_floor'] = floors_points(bank, T, x, V, cfg['ladder'])
    log('frozen bank floor ' + ' '.join(f"R{r}={v['worst']:.4f}/{v['rms']:.4f}" for r, v in res['bank_floor'].items()))
    res['seconds'] = time.perf_counter() - T0
    C.dump(out / f'podscale_n{n}.json', C.clean(res))


def bank_floor_full(bank, T, n, Y, ladder, chunk=1 << 16):
    """Full-grid floor of fields Y (F, N) on G_hat[:, :R'] at mesh n: Gram pass, prefix Cholesky, residual pass."""
    x = C.interior_coords(n)
    f = jax.jit(lambda p, xx, T: C.features(p, xx) @ T)
    Tj = jnp.asarray(T)
    Yj = jnp.asarray(Y)
    R = T.shape[1]
    acc = jax.jit(lambda Gm, B, g, y: (Gm + g.T @ g, B + g.T @ y.T))     # jitted: eager f64 GEMM gave NaN on GB10
    Gm = jnp.zeros((R, R))
    B = jnp.zeros((R, Y.shape[0]))
    for s in range(0, len(x), chunk):
        Gm, B = acc(Gm, B, f(bank, jnp.asarray(x[s:s + chunk]), Tj), Yj[:, s:s + chunk])
    assert bool(jnp.all(jnp.isfinite(Gm))) and bool(jnp.all(jnp.isfinite(B)))
    ev = jnp.linalg.eigvalsh(Gm)
    coefs = {}
    for r in ladder:
        L = jnp.linalg.cholesky(Gm[:r, :r])
        coefs[r] = jax.scipy.linalg.cho_solve((L, True), B[:r])            # (r, F)
    res2 = {r: jnp.zeros(Y.shape[0]) for r in ladder}
    rpass = jax.jit(lambda g, yc, c: jnp.sum((yc - (g[:, :c.shape[0]] @ c).T) ** 2, axis=1))
    for s in range(0, len(x), chunk):
        g = f(bank, jnp.asarray(x[s:s + chunk]), Tj)
        yc = Yj[:, s:s + chunk]
        for r in ladder:
            res2[r] = res2[r] + rpass(g, yc, coefs[r])
    nrm = jnp.linalg.norm(Yj, axis=1)
    out = {}
    for r in ladder:
        e = jnp.sqrt(res2[r]) / nrm
        out[str(r)] = dict(worst=float(jnp.max(e)), rms=float(jnp.sqrt(jnp.mean(e ** 2))))
    return out, float(jnp.sqrt(ev[-1] / ev[0]))


def fine(cfg, out):
    bank, T = load_bank(cfg['model'])
    tab = C.table(cfg['val_seed'], cfg['cases'])
    res = dict(config=cfg, meshes={})
    for n in cfg['meshes']:
        ref = C.make_fom(n, C.DT, 1e-10, 1e-11)
        U0 = [jnp.asarray(C.initial_interior(n, tab, j)) for j in range(cfg['cases'])]
        t = time.perf_counter()
        refs = []
        for j in range(cfg['cases']):
            f, it, rn = ref(U0[j], float(tab['nu'][j]))
            refs.append(np.asarray(f))
        log(f'[n={n}] references {time.perf_counter() - t:.1f}s (incl. compile) max newton {int(jnp.max(it))}')
        Y = np.concatenate(refs)                                               # (6 cases, N)
        fl, cond = bank_floor_full(bank, T, n, Y, cfg['ladder'])
        log(f'[n={n}] bank floor cond {cond:.3e} ' + ' '.join(f"R{r}={v['worst']:.4f}/{v['rms']:.4f}" for r, v in fl.items()))
        fl_ev, _ = bank_floor_full(bank, T, n, np.concatenate([r[1:] for r in refs]), cfg['ladder'])
        foms = {}
        for dt, nt, lt in cfg['fom_grid']:
            q = C.make_fom(n, dt, nt, lt)
            for j in range(cfg['cases']):
                C.block(q(U0[j], float(tab['nu'][j])))                         # compile + warm
            ts, errs = [], []
            for rep in range(cfg['reps']):
                for j in range(cfg['cases']):
                    t = time.perf_counter()
                    f, it, rn = q(U0[j], float(tab['nu'][j]))
                    C.block(f)
                    ts.append(time.perf_counter() - t)
                    if rep == 0:
                        errs.append(float(np.max(C.rel_errors(f, refs[j])[1:])))
            name = f'fom_dt{dt:g}_nt{nt:g}_lt{lt:g}'
            foms[name] = dict(ms=1e3 * float(np.median(ts)), worst=max(errs))
            log(f'[n={n}] {name}: {foms[name]["ms"]:.2f} ms worst {100 * foms[name]["worst"]:.3f}%')
        res['meshes'][str(n)] = dict(bank_floor_all_times=fl, bank_floor_evolved=fl_ev, cond=cond, fom=foms)
        del refs, Y, U0
        C.dump(out / 'fine.json', C.clean(res))
    res['seconds'] = time.perf_counter() - T0
    C.dump(out / 'fine.json', C.clean(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    assert jax.default_backend() == 'gpu' or cfg.get('local_smoke')
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f'jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64} mode={cfg["mode"]}')
    {'podscale': podscale, 'fine': fine}[cfg['mode']](cfg, out)
    log('PROBE COMPLETE')


if __name__ == '__main__':
    main()
