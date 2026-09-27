"""Equivalence + speed check of make_query_fsc (cached predictor) and its float32-table variant against the lane-1
make_query_fs path, on probe seed 923651 rows 0-3 (no cohort touched). Diagnostic only."""
import os, sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'burgers3d-span'))
import numpy as np, jax, jax.numpy as jnp
import common as C, common2 as C2
assert jax.default_backend() == 'gpu' and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
b = pickle.loads(open('experiments/burgers3d-span/inputs/model_R512/bank.pkl', 'rb').read())
bank = jax.tree_util.tree_map(jnp.asarray, b['params']); T = np.asarray(b['rotation'])
for n in (33, 65, 129):
    order, lam = C.mode_order(n); M_of = lambda r: C.complete_M(n, 4 * r, order, lam)
    mesh = C.build_mesh(n, bank, T, None, None, [128, 192, 256, 512], M_of, build_eq=False)
    assert all(bool(jnp.all(jnp.isfinite(mesh[k]))) for k in ('G', 'Rq', 'A', 'Tsym'))
    tab = C.table(923651, 4)
    for Rp, dt in [(128, 0.01), (192, 0.01), (256, 0.01), (512, 0.01), (512, 0.005)]:
        M = M_of(Rp); d = C.arm_data(mesh, 'span', 'tensor', Rp, M); d['Ts32'] = d['Ts'].astype(jnp.float32)
        tr = 0.05 * b['coefficient_rms_spread'][str(Rp)]
        qs = dict(fs=C.make_query_fs(n, Rp, M, dt=dt, trust=tr, adaptive_first=3)[0],
                  fsc=C2.make_query_fsc(n, Rp, M, dt=dt, trust=tr, adaptive_first=3)[0],
                  fsc32=C2.make_query_fsc(n, Rp, M, dt=dt, trust=tr, adaptive_first=3, t32=True)[0])
        for j in range(4):
            u0 = jnp.asarray(C.initial_interior(n, tab, j)); nu = float(tab['nu'][j])
            o = {k: q(u0, nu, d, {}) for k, q in qs.items()}
            f0 = o['fs'][0]
            print(n, Rp, dt, j, 'rel diff fsc %.2e fsc32 %.2e' % tuple(float(jnp.linalg.norm(o[k][0] - f0) / jnp.linalg.norm(f0)) for k in ('fsc', 'fsc32')),
                  'nonstationary', {k: int((np.asarray(v[3]) != 4).sum()) for k, v in o.items()}, flush=True)
        for k, q in qs.items():
            ts = []
            for r in range(7):
                t = time.perf_counter(); jax.block_until_ready(q(u0, nu, d, {})); ts.append(time.perf_counter() - t)
            print(n, Rp, dt, k, 'ms %.2f' % (1e3 * np.median(ts)), flush=True)
    del mesh
print('CHECK COMPLETE')
