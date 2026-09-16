"""Bench the GPU fitter against scipy's NNLS on REAL quadrature designs.

Attempt `qrg301` failed because the inner solve dropped every negative weight without a KKT
re-entry test: the outer greedy re-selected the columns NNLS had just zeroed, cycled, and
stopped on its pass cap with 97 of 256 requested points. Attempt `qrg302` failed for the
opposite reason: replacing the greedy selection with a projected-gradient ranking removed the
cycling but chose a support that could not fit the design (relative fit 0.36 where the
incumbent reaches 5e-3).

Both would have been caught here, because this bench builds the **actual** design the cell
fits -- the decoder bank, the sine test modes and the five-point advection of real decoder
states -- rather than a synthetic one, and requires:

  * the achieved support to reach the target, or the fitter to stop on a genuine KKT
    `gradient` exit (no remaining candidate improves the fit);
  * every weight nonnegative;
  * the relative fit to be within `TOL` of scipy's block-greedy Lawson-Hanson at the same
    target, on the same design.

It runs at 64 intervals so it is minutes, not hours, on the shared box.

    python checks/fitter_bench.py [out.json]
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

ROOT = Path(__file__).resolve().parents[3]
for sub in ('experiments/q-ridge', 'experiments/b-ladder-top', 'experiments/cheap-corrections',
            'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e          # noqa: E402
import arms as A             # noqa: E402
import eqcert as EC          # noqa: E402
import varpro as VP          # noqa: E402

FITTER = sys.argv[2] if len(sys.argv) > 2 else 'bounded'

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
# (M, m, fit states) triples spanning the shapes the job fits.
CASES = [(64, 256, 16), (128, 512, 16), (256, 1024, 16), (320, 1280, 16)]
TOL = 2.0


def build_design(bank, G, Phi, L, codes, cand):
    """The cell's own design: rows are (state, test mode), columns are candidate points."""
    P = jnp.asarray(Phi)
    Pc = np.asarray(Phi)[cand]
    adv = jax.jit(lambda g, c: e.spatial(g @ c, L)[0])
    rows, targets = [], []
    for c in codes:
        n = adv(G, jnp.asarray(c))
        targets.append(np.asarray(P.T @ n))
        rows.append(Pc.T * np.asarray(n)[cand])
    D = np.concatenate(rows)
    b = np.concatenate(targets)
    scale = np.linalg.norm(D, axis=1) + 1e-300
    return D / scale[:, None], b / scale


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = Zold.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    L = 64
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    head = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))
    rng = np.random.default_rng(20260916)
    cand = np.sort(rng.choice((L - 1) ** 2, min(2048, (L - 1) ** 2), replace=False))
    rows = []
    for M, m, states in CASES:
        Phi, _, _ = e.modes(L, M)
        sel = np.sort(rng.choice(len(Zold), states, replace=False))
        codes = np.asarray(head(jnp.asarray(Zold[sel])))
        D, b = build_design(bank, G, Phi, L, codes, cand)
        s1, w1, i1 = EC.fit_via(D, b, m, fitter=FITTER, blocks=16, seconds=900.)
        t0 = time.perf_counter()
        s2, w2, i2 = VP.bounded_nnls(D, b, m, 900., block=max(1, m // 16))
        r2 = float(np.linalg.norm(D[:, s2] @ w2 - b) / np.linalg.norm(b))
        row = dict(M=M, m_target=m, fit_states=states, design_rows=int(D.shape[0]),
                   candidates=int(D.shape[1]), gpu_support=int(len(s1)),
                   gpu_relative=i1['relative_fit'],
                   gpu_stop=i1.get('stop_reason'), fitter=FITTER,
                   gpu_seconds=i1['seconds'], scipy_support=int(len(s2)),
                   scipy_relative=r2, scipy_seconds=time.perf_counter() - t0,
                   ratio_to_scipy=i1['relative_fit'] / max(r2, 1e-300),
                   nonnegative=bool((w1 >= 0).all()),
                   reached_target_or_kkt=bool(len(s1) >= m
                                              or i1.get('stop_reason') == 'gradient'))
        rows.append(row)
        print(row, flush=True)
        assert row['nonnegative'], row
        assert row['reached_target_or_kkt'], row
        assert row['ratio_to_scipy'] <= TOL, row
    out = dict(intervals=L, tolerance=TOL, fitter=FITTER, cases=rows,
               note=('real quadrature designs; qrg301 would fail reached_target and qrg302 '
                     'would fail ratio_to_scipy'))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('FITTER BENCH OK', flush=True)


if __name__ == '__main__':
    main()
