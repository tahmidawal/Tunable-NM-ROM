"""Local cost probe: which solver variant is worth putting on the cluster?

Runs the real mesh (256 intervals) and the real operator sizes with PROXY correction
directions — the field-metric POD of the training-snapshot bank coefficients about
their mean, from four trajectories — so the shapes, the quadrature and the solver cost
are realistic while the directions are not the audited ones. Accuracy here means
nothing; iteration counts, stopping status and per-query GPU time are what it is for.

This exists because the first smoke showed plain variable projection converging in ~4x
the outer iterations of the joint solver, which would cancel the per-iteration saving.
The decision it feeds is recorded in DESIGN.md.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/cheap-corrections'))
sys.path.insert(0, str(ROOT / 'experiments/head-ablation'))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import varpro as VP            # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'


def main():
    assert jax.default_backend() == 'gpu'
    print('jax_backend=gpu', flush=True)
    L, dt = 256, 0.005
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = Zold.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)

    train = e.params_draw(0, 4)
    fom, _ = e.make_fom(L, dt, None, .25, dt)
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    del fom
    jax.clear_caches()
    coef = np.asarray(jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T)
    tilde = jnp.asarray(coef - coef.mean(0)) @ Rb.T
    _, _, Vt = jnp.linalg.svd(tilde, full_matrices=False)
    # fewer snapshots than R, so complete the leading POD directions to a full
    # field-orthonormal basis; cost does not depend on which directions these are
    rng = np.random.default_rng(0)
    full = np.concatenate((np.asarray(Vt.T), rng.standard_normal((R, R))), axis=1)
    Cproxy = jnp.linalg.solve(Rb, jnp.asarray(np.linalg.qr(full)[0][:, :R]))
    Zsub = np.asarray(Zold[::max(1, len(Zold) // 2048)])
    phys = e.params_draw(7090702, 1)[0]
    u0 = jnp.asarray(e.initial(L, phys))
    nu = jnp.asarray(phys[4])

    strict = dict(ic_budget=400, step_budget=180, gtol=1e-6)
    rows = []
    plan = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        dict(q=64, M=256, quad='eq', m=1024), dict(q=64, M=320, quad='dense'),
        dict(q=256, M=1088, quad='dense'), dict(q=512, M=1056, quad='eq', m=2048)]
    for spec in plan:
        q, M, quad = spec['q'], spec['M'], spec['quad']
        C = Cproxy[:, :q]
        head = VP.corrected_head(params, C, K)
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        cold, _ = A.build_cold(bank, head, Zaug, 48)
        W = VP.enriched_codes(Zsub, np.zeros((len(Zsub), R)), Rb, jnp.eye(R), q) if quad == 'eq' else None
        t0 = time.perf_counter()
        data, info = VP.build_operators(bank, L, M, quad, head=head, Wcodes=W, m=spec.get('m'),
                                        fitter='bounded', max_fit_rows=8192, eq_seconds=120.)
        setup = time.perf_counter() - t0
        for variant in ('joint', 'block', 'varpro', 'alt'):
            lin = 'gj' if (K + q) <= 64 else 'lu'
            fn = VP.make_query(params, C, K, q, L, dt, trust, quad, variant, linear=lin,
                               inner_iters=2, alt_rounds=3, **strict)
            t0 = time.perf_counter()
            v = fn(u0, nu, data, cold)
            jax.block_until_ready(v)
            compile_s = time.perf_counter() - t0
            e.burn(.25)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(u0, nu, data, cold))
                ts.append(time.perf_counter() - t0)
            h = jax.device_get(v)
            rows.append(dict(q=q, M=M, quadrature=quad, m=info.get('m'), variant=variant,
                             setup_seconds=setup, compile_seconds=compile_s,
                             median_gpu_ms=float(np.median(ts) * 1e3),
                             median_iterations=float(np.median(h[1])),
                             total_iterations=int(np.sum(h[1])),
                             budget_exits=int(np.sum(np.asarray(h[3]) == 0)),
                             max_joint_stationarity=float(np.max(h[12])),
                             ic_joint_stationarity=float(h[13]),
                             max_inner_stationarity=float(np.max(h[14]))))
            print(json.dumps(rows[-1]), flush=True)
            del fn
            jax.clear_caches()
        del data, cold
        jax.clear_caches()
    out = dict(mesh=L, dt=dt, trust=trust, directions='proxy field-metric POD of snapshot '
               'coefficients; NOT the audited residual directions', rows=rows)
    Path('experiments/cheap-corrections/checks/probe-cost.json').write_text(
        json.dumps(out, indent=2) + '\n')
    print('PROBE OK', flush=True)


if __name__ == '__main__':
    main()
