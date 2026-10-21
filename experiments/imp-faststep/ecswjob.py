"""imp-faststep `ecsw` job (DESIGN 3.5, A1.8, A2.8, A3.5): Galerkin-ECSW weights for the POD-Galerkin ETDRK4 arm.

Training states: hr-baselines `build.ecsw_states` (24 seeded trajectories, both 2D options, central FOM at the mesh).
For rank r (prefix of the hr-baselines POD basis) and state s with projection c_s = Phi_r^T u_s, u_p = Phi_r c_s:
    node contribution  g_i(s) = Phi_r[i, :]^T N_{h,i}(u_p)        (central advection, hrcore.adv_central)
    target             b(s)   = Phi_r^T N_h(u_p) = sum_i g_i(s)
Block scaling 1 / max(||b(s)||, floor) with build.ecsw_contrib's positive floor and dropped-block handling; candidates
32,768 (half uniform, half proportional to sum_s ||g_i(s)||, seed 20261018). Weights: build.ecsw_weights unchanged
(kappa path 2, 4, 8, 16), per-attempt wall budgets A2.8. Also control C11: the exact stacked block-scaled aggregate
relres of every checkpoint and the uniform-weight witness.

    python ecswjob.py --config configs/ecsw.json --out <dir>
Outputs: GECSW_r<r>.npz (k<kappa>_nodes / _weights / _relres / _size / _optimum), ecsw.json, COMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HRB = HERE.parent / 'jcp-hr-baselines'
sys.path.insert(0, str(HRB))
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import hrcore as H  # noqa: E402
import build as BD  # noqa: E402

BUDGET = {128: 240., 256: 360., 384: 600.}


def contributions(states, Phi_r, L, seed=BD.ECSW_SEED_CAND, n_cand=BD.CAND):
    """Stacked, block-scaled Galerkin contribution matrix G (rows = retained states x r, columns = candidates), target b,
    candidates, and the per-state data needed to rebuild columns for any node set (C11 witness)."""
    n = Phi_r.shape[0]
    r_ = Phi_r.shape[1]
    sh = (L - 1, L - 1)

    @jax.jit
    def one(u, P):
        c = P.T @ u
        N = H.adv_central((P @ c).reshape(sh), L).reshape(-1)
        return N, P.T @ N
    rowsn = jnp.linalg.norm(Phi_r, axis=1)
    imp = np.zeros(n)
    Ns, bs = [], []
    for s in states:
        N, b = one(jnp.asarray(s['u']), Phi_r)
        imp += np.asarray(jnp.abs(N) * rowsn)
        Ns.append(N)
        bs.append(np.asarray(b))
    nb = np.array([np.linalg.norm(b) for b in bs])
    if not (np.isfinite(nb).all() and np.isfinite(imp).all()):
        raise RuntimeError('ECSW construction failure: non-finite training contributions')
    floor = max(1e-8 * float(np.median(nb)), 1e-14 * float(np.max(nb)), 1e-300)
    keep = nb >= floor
    if not np.any(keep):
        raise RuntimeError('ECSW construction failure: every training target is below the floor')
    if n <= n_cand:                                    # build.ecsw_contrib's all-node branch
        cand = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        half = n_cand // 2
        uni = rng.choice(n, half, replace=False)
        rest = np.setdiff1d(np.arange(n), uni)
        mass = float(imp[rest].sum())
        if not (np.isfinite(mass) and mass > 0):
            raise RuntimeError('ECSW construction failure: zero importance mass outside the uniform half')
        pr = imp[rest] / mass
        k_imp = min(n_cand - half, int(np.sum(pr > 0)))
        cand = np.sort(np.concatenate([uni, rng.choice(rest, k_imp, replace=False, p=pr)]))
    sc = 1. / np.maximum(nb, floor)
    blocks = [(N, b, sc[i]) for i, (N, b) in enumerate(zip(Ns, bs)) if keep[i]]

    def columns(nodes):
        nj = jnp.asarray(nodes)
        Pc = Phi_r[nj]
        return jnp.concatenate([(Pc * N[nj][:, None]).T * s_ for N, _, s_ in blocks], 0)
    G = columns(cand)
    b = jnp.asarray(np.concatenate([b_ * s_ for _, b_, s_ in blocks]))
    info = dict(dropped_blocks=int(np.sum(~keep)), n_cand=int(len(cand)), floor=floor, block_norms=nb.tolist(),
                keep=keep.tolist(), all_nodes=bool(n <= n_cand))
    return G, b, cand, info, columns


def jclean(x):
    if isinstance(x, dict):
        return {str(k): jclean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jclean(v) for v in x]
    if isinstance(x, np.ndarray):
        return jclean(x.tolist()) if x.size <= 64 else dict(shape=list(x.shape), sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if hasattr(x, 'shape'):
        return jclean(np.asarray(x))
    return x


def mem():
    import resource
    try:
        s = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001
        s = {}
    return dict(device_peak=int(s.get('peak_bytes_in_use', -1)), device_in_use=int(s.get('bytes_in_use', -1)),
                host_maxrss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu'
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    L = int(cfg['mesh'])
    tdir = Path(cfg['tune_dir'])
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               gpu=jax.devices()[0].device_kind, ranks={}, timers={})
    t0 = time.perf_counter()
    h = hashlib.sha256()
    with open(tdir / 'artifacts' / 'Phi.npz', 'rb') as f_:
        for blk in iter(lambda: f_.read(1 << 24), b''):
            h.update(blk)
    rep['phi_sha256'] = h.hexdigest()
    assert rep['phi_sha256'] == cfg['phi_sha256'], 'POD basis differs from the pinned digest'
    zf = np.load(tdir / 'artifacts' / 'Phi.npz')
    rmax = max(cfg['ranks'])
    Phi = jnp.asarray(np.asarray(zf['Phi'])[:, :rmax])
    del zf
    rep['timers']['load'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fam = H.Family(2, 'train')
    states, sinfo = BD.ecsw_states(2, L, fam, None)
    rep['states'] = sinfo
    rep['timers']['states'] = time.perf_counter() - t0
    c11_ok = True
    for r in cfg['ranks']:
        t1 = time.perf_counter()
        G, b, cand, cinfo, columns = contributions(states, Phi[:, :r], L)
        jax.block_until_ready(G)
        tc = time.perf_counter() - t1
        t2 = time.perf_counter()
        rec, summ = BD.ecsw_weights(G, b, cand, r, BD.ECSW_KAPPA_PATH, max_seconds=BUDGET.get(r, 120.))
        tn = time.perf_counter() - t2
        bn = float(jnp.linalg.norm(b))
        arrays, cps = {}, {}
        for k, v in rec.items():
            if v is None:
                cps[f'k{k}'] = None
                continue
            Sv = jnp.asarray(np.searchsorted(cand, v['nodes']))
            assert np.array_equal(np.asarray(cand)[np.asarray(Sv)], np.asarray(v['nodes']))
            agg = float(jnp.linalg.norm(G[:, Sv] @ jnp.asarray(v['weights']) - b) / bn)
            rr_ = float(v['relres'])
            agree = bool(abs(agg - rr_) <= 1e-8 * rr_) if rr_ > 0 else bool(agg <= 1e-300)
            arrays.update({f'k{k}_nodes': np.asarray(v['nodes']), f'k{k}_weights': np.asarray(v['weights']),
                           f'k{k}_relres': np.float64(rr_), f'k{k}_size': np.int64(v['size']),
                           f'k{k}_optimum': np.bool_(v.get('optimum', False))})
            cps[f'k{k}'] = dict(size=int(v['size']), relres=rr_, aggregate_recomputed=agg, agree=agree,
                                optimum=bool(v.get('optimum', False)))
            c11_ok = c11_ok and agree
        # C11 witness (A2.7): uniform weights n/|E| on |E| (then |E|/4) nodes drawn uniformly from ALL n nodes
        avail = [k for k in (8, 16, 4, 2) if cps.get(f'k{k}')]
        wit = dict(fires=False)
        if avail:
            k0 = 8 if cps.get('k8') else max(k for k in (2, 4, 8, 16) if cps.get(f'k{k}'))
            E = cps[f'k{k0}']['size']
            rng = np.random.default_rng(20261021)
            n = Phi.shape[0]
            for frac in (1, 4):
                sz = max(1, E // frac)
                pick = np.sort(rng.choice(n, sz, replace=False))
                agg = float(jnp.linalg.norm(columns(pick) @ jnp.full(sz, n / sz) - b) / bn)
                ratio = agg / cps[f'k{k0}']['relres'] if cps[f'k{k0}']['relres'] > 0 else float('inf')
                wit[f'E/{frac}'] = dict(size=sz, relres=agg, ratio=ratio, nodes_sha256=hashlib.sha256(pick.tobytes()).hexdigest())
                if ratio >= 10:
                    wit['fires'] = True
                    break
            wit['kappa_ref'] = k0
        c11_ok = c11_ok and bool(wit['fires'])
        np.savez_compressed(out / f'GECSW_r{r}.npz', cand=cand, keep=np.asarray(cinfo['keep']), **arrays)
        rep['ranks'][str(r)] = dict(contrib=cinfo, contrib_seconds=tc, nnls_seconds=tn, checkpoints=cps, witness=wit,
                                    budget_per_attempt=BUDGET.get(r, 120.), selected=summ['selected'],
                                    attempts=jclean(summ['attempts']), memory=mem())
        json.dump(jclean(rep), open(out / 'ecsw.json', 'w'), indent=1)
        print('RANK', r, 'contrib', round(tc, 1), 'nnls', round(tn, 1), json.dumps(cps), 'witness', json.dumps(wit), flush=True)
        del G, b
    rep['c11_ok'] = c11_ok
    rep['memory'] = mem()
    rep['complete'] = True
    json.dump(jclean(rep), open(out / 'ecsw.json', 'w'), indent=1)
    if not c11_ok:
        (out / 'C11_FAILED').write_text('C11 aggregate or witness failed\n')
        print('ECSW C11 FAILED', flush=True)
        sys.exit(3)
    (out / 'COMPLETE').write_text('complete\n')
    print('ECSW COMPLETE', flush=True)


if __name__ == '__main__':
    main()
