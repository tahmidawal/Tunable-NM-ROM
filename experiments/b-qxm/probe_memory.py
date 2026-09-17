"""Device-memory probe at the production mesh (L = 256), no science.

Compiles a handful of reduced queries with RANDOM (orthonormalised) direction matrices at
the largest (q, M) cells each job will hold and records device bytes in use after each
compile, so the per-job subject count is sized from a measurement rather than from the
qtd01 anecdote (14 fit a 40 GB A100, 34 do not). Values are irrelevant; shapes are not.

    python probe_memory.py <out.json>
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
for sub in ('experiments/b-qxm', 'experiments/q-ridge', 'experiments/b-ladder-top',
            'experiments/cheap-corrections', 'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))
import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import topfix as TF            # noqa: E402
import ridge as RG             # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
CELLS = [(0, 64), (0, 1088), (32, 1088), (64, 1280), (256, 2176), (64, 4096)]


def stats():
    s = jax.devices()[0].memory_stats() or {}
    return dict(bytes_in_use=s.get('bytes_in_use'), peak=s.get('peak_bytes_in_use'))


def main():
    assert jax.default_backend() == 'gpu'
    L, dt, K = 256, .005, 16
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    rng = np.random.default_rng(0)
    Ct = np.linalg.qr(rng.standard_normal((R, 256)))[0]
    Cfull = jnp.linalg.solve(Rb, jnp.asarray(Ct))
    Zsub = np.asarray(Zold[::max(1, len(Zold) // 8192)])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    phys = e.params_draw(7090702, 1)[0]
    u0 = jnp.asarray(e.initial(L, phys))
    nu = float(phys[4])
    rows = [dict(stage='after_bank', **stats())]
    ops, colds, keep = {}, {}, []
    for q, M in CELLS:
        t = time.perf_counter()
        C = Cfull[:, :q]
        head = LD.corrected_head(params, C, K)
        if q not in colds:
            colds[q] = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1), 48)[0]
        if M not in ops:
            d, _ = TF.build_operators(bank, L, M, 'dense')
            d['G'] = G
            ops[M] = d
        after_ops = stats()
        linear = 'gj' if K + q <= 64 else 'lu'
        fn = RG.make_query(params, C, K, q, L, dt, trust, 'dense', 0., ic_budget=400,
                           step_budget=600, gtol=1e-6, ic_gtol=1e-6, linear=linear,
                           inner_damping=1e-10)
        v = fn(u0, nu, ops[M], colds[q])
        jax.block_until_ready(v)
        keep.append((fn, v))
        rows.append(dict(stage=f'q{q}_M{M}', after_operators=after_ops, after_compile=stats(),
                         seconds=time.perf_counter() - t, query_finite=bool(np.isfinite(np.asarray(v[0])).all())))
        print(rows[-1], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(dict(cells=CELLS, rows=rows,
                                                 device=jax.devices()[0].device_kind), indent=2) + '\n')
    print('PROBE DONE')


if __name__ == '__main__':
    main()
