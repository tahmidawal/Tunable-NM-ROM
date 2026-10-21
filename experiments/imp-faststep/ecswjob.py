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
    rng = np.random.default_rng(seed)
    half = n_cand // 2
    uni = rng.choice(n, half, replace=False)
    rest = np.setdiff1d(np.arange(n), uni)
    pr = imp[rest] / imp[rest].sum()
    k_imp = min(n_cand - half, int(np.sum(pr > 0)))
    cand = np.sort(np.concatenate([uni, rng.choice(rest, k_imp, replace=False, p=pr)]))
    cj = jnp.asarray(cand)
    nb = np.array([np.linalg.norm(b) for b in bs])
    floor = max(1e-8 * float(np.median(nb)), 1e-14 * float(np.max(nb)), 1e-300)
    keep = nb >= floor
    if not np.any(keep):
        raise RuntimeError('ECSW construction failure: every training target is below the floor')
    sc = 1. / np.maximum(nb, floor)
    Gh = np.empty((int(keep.sum()) * r_, len(cand)))
    bh = np.empty(int(keep.sum()) * r_)
    Pc = Phi_r[cj]
    q = 0
    for i, (N, b) in enumerate(zip(Ns, bs)):
        if not keep[i]:
            continue
        Gh[q * r_:(q + 1) * r_] = np.asarray((Pc * N[cj][:, None]).T) * sc[i]
        bh[q * r_:(q + 1) * r_] = b * sc[i]
        q += 1
    return jnp.asarray(Gh), jnp.asarray(bh), cand, dict(dropped_blocks=int(np.sum(~keep)), n_cand=int(len(cand)),
                                                        block_norm_min=float(nb.min()), block_norm_max=float(nb.max()))


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
    for r in cfg['ranks']:
        t1 = time.perf_counter()
        G, b, cand, cinfo = contributions(states, Phi[:, :r], L)
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
            agg = float(jnp.linalg.norm(G[:, Sv] @ jnp.asarray(v['weights']) - b) / bn)
            arrays.update({f'k{k}_nodes': np.asarray(v['nodes']), f'k{k}_weights': np.asarray(v['weights']),
                           f'k{k}_relres': np.float64(v['relres']), f'k{k}_size': np.int64(v['size']),
                           f'k{k}_optimum': np.bool_(v.get('optimum', False))})
            cps[f'k{k}'] = dict(size=int(v['size']), relres=float(v['relres']), aggregate_recomputed=agg,
                                agree=bool(abs(agg - v['relres']) <= 1e-8 * max(v['relres'], 1e-300) + 1e-14),
                                optimum=bool(v.get('optimum', False)))
        # C11 witness: uniform weights on |E| (then |E|/4) uniformly random nodes, |E| = kappa 8 support (or largest)
        avail = [k for k in (8, 16, 4, 2) if cps.get(f'k{k}')]
        wit = {}
        if avail:
            k0 = avail[0]
            E = cps[f'k{k0}']['size']
            rng = np.random.default_rng(20261021)
            n = Phi.shape[0]
            for frac in (1, 4):
                sz = max(1, E // frac)
                pick = np.sort(rng.choice(len(cand), sz, replace=False))
                xi = jnp.full(sz, n / sz)
                agg = float(jnp.linalg.norm(G[:, jnp.asarray(pick)] @ xi - b) / bn)
                wit[f'E/{frac}'] = dict(size=sz, relres=agg, ratio=agg / max(cps[f'k{k0}']['relres'], 1e-300))
            wit['fires'] = any(v['ratio'] >= 10 for v in wit.values() if isinstance(v, dict))
            wit['kappa_ref'] = k0
        np.savez_compressed(out / f'GECSW_r{r}.npz', cand=cand, **arrays)
        sel = summ['attempts'][summ['selected']]
        rep['ranks'][str(r)] = dict(contrib=cinfo, contrib_seconds=tc, nnls_seconds=tn, checkpoints=cps, witness=wit,
                                    nnls=dict(selected=summ['selected'], reason=sel['reason'], final_relres=sel['final_relres'],
                                              final_support=sel['final_support'], banned=sel['banned'],
                                              elapsed=sel.get('elapsed_seconds'), budget=BUDGET.get(r, 120.),
                                              attempts=list(summ['attempts'])))
        json.dump(rep, open(out / 'ecsw.json', 'w'), indent=1, default=float)
        print('RANK', r, 'contrib', round(tc, 1), 'nnls', round(tn, 1), json.dumps(cps), flush=True)
        del G, b
    rep['complete'] = True
    json.dump(rep, open(out / 'ecsw.json', 'w'), indent=1, default=float)
    (out / 'COMPLETE').write_text('complete\n')
    print('ECSW COMPLETE', flush=True)


if __name__ == '__main__':
    main()
