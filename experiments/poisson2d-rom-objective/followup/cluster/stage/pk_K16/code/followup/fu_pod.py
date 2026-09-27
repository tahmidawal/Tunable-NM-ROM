"""POD (linear-subspace) ROM ladder for the Poisson-2D testbed -- the linear
control for the k-ladder.  Basis V_k = top-k left singular vectors of the
TRAIN interior fields (f64, host SVD).  For each k, on the same 16 held-out
test sources as the coord ROM:

  proj      : projection floor ||V V^T u* - u*|| / ||u*||           (oracle)
  galerkin  : (V^T A V) c = V^T f     (= energy-norm / Ritz Galerkin ROM)
  fd        : min_c ||A V c - f||     (LSPG on the FD residual)
  weak_a1_M : min_c ||Phi_M^T V c - Lambda_M^{-1} Phi_M^T f||   (the coord ROM's
              objective, FULL GRID; requires M' >= k)
Every objective is quadratic in the POD coefficients, so the exact minimiser is
reported (an unregularised least-squares / k x k solve).  The damped LM used for
the nonlinear coordinate manifold converges to this same point whenever the
system has full column rank, which is recorded per k (`rank`, `cond`,
`square_system`) -- a square or rank-deficient Petrov-Galerkin system (M' <= k)
is labelled and must not be read as a fair comparison.
NOTE ON COMPARABILITY: these are FULL-GRID objectives.  The coordinate rows they
should be compared against are the coordinate ROM's own full-grid rows; the
NNLS-EQ coordinate rows use a different (hyper-reduced) quadrature.
Errors are mean rel-L2 over the full grid vs the FD solution -- the same metric
and the same 16 held-out sources as pro_colloc.py.

Usage: [N=64] [KS=2,4,6,8,12,16,24,32,48,64] [MS=64,256] [N_TEST=16] python fu_pod.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pro_common as pc  # noqa: E402
from pro_common import mp  # noqa: E402

OUT = sys.argv[1]
KS = [int(k) for k in os.environ.get("KS", "2,4,6,8,12,16,24,32,48,64").split(",")]
MS = [int(m) for m in os.environ.get("MS", "64,256").split(",")]
N_TEST = int(os.environ.get("N_TEST", "16"))


def main():
    N, N_TRAIN = mp.N, mp.N_TRAIN
    print(f"jax_backend={jax.default_backend()} N={N} KS={KS}", flush=True)
    grid = pc.Grid(N)
    U, z_true, coords, fom_res = mp.build_snapshots(N)
    U = np.asarray(U)
    n_i = grid.n_i
    ix, iy = grid.ix_full, grid.iy_full
    int_idx = ix * N + iy
    X = U[:N_TRAIN][:, int_idx]                              # (n_tr, n_i^2)
    t0 = time.time()
    # host f64 SVD of X^T (n_i^2 x n_tr): V = left singular vectors
    Vfull, sv, _ = np.linalg.svd(X.T, full_matrices=False)
    print(f"  POD: {X.shape[0]} snapshots, sv[0]={sv[0]:.3e} sv[63]={sv[min(63, len(sv)-1)]:.3e} [{time.time()-t0:.0f}s]", flush=True)
    U_test = U[N_TRAIN:N_TRAIN + N_TEST]
    tn = np.linalg.norm(U_test, axis=1)
    cx, cy, w, a, _ = mp.sample_params()
    F = np.stack([mp.source_interior(N, cx[N_TRAIN + i], cy[N_TRAIN + i], w[N_TRAIN + i], a[N_TRAIN + i])
                  for i in range(N_TEST)])                     # (n_test, n_i, n_i)
    op = jax.jit(lambda v2d: grid.op(v2d))
    spec = jax.jit(lambda v2d: grid.spec(v2d))
    lam = np.asarray(grid.lam)
    masks = {M: np.asarray(grid.mode_mask(M)).astype(bool) for M in MS}
    def full_field(c_int):
        u = np.zeros(N * N); u[int_idx] = c_int; return u
    def err(u_int_pred, i):
        return float(np.linalg.norm(full_field(u_int_pred) - U_test[i]) / tn[i])
    rows = []
    for k in KS:
        V = Vfull[:, :k]                                       # (n_i^2, k)
        AV = np.stack([np.asarray(op(jnp.asarray(V[:, j].reshape(n_i, n_i)))).reshape(-1) for j in range(k)], 1)
        SV = np.stack([np.asarray(spec(jnp.asarray(V[:, j].reshape(n_i, n_i)))) for j in range(k)], -1)  # (n_i,n_i,k)
        Ag = V.T @ AV                                          # k x k (SPD)
        e = {"proj": [], "galerkin": [], "fd": []} | {f"weak_a1_M{M}": [] for M in MS}
        cond_rank = {}
        for i in range(N_TEST):
            u_int = U_test[i][int_idx]
            f = F[i].reshape(-1)
            e["proj"].append(err(V @ (V.T @ u_int), i))
            c = np.linalg.solve(Ag, V.T @ f); e["galerkin"].append(err(V @ c, i))
            c, *_ = np.linalg.lstsq(AV, f, rcond=None); e["fd"].append(err(V @ c, i))
            Cf = np.asarray(spec(jnp.asarray(F[i])))
            for M in MS:
                mk = masks[M]
                A_ = SV[mk]                                    # (M', k)  Phi_M^T V
                b_ = Cf[mk] / lam[mk]                          # Lambda^{-1} Phi_M^T f
                c, *_ = np.linalg.lstsq(A_, b_, rcond=None); e[f"weak_a1_M{M}"].append(err(V @ c, i))
                cond_rank[M] = (float(np.linalg.cond(A_)), int(np.linalg.matrix_rank(A_)),
                                bool(A_.shape[0] <= k))
        row = dict(k=k, n_modes={M: int(masks[M].sum()) for M in MS},
                   cond_A_galerkin=float(np.linalg.cond(Ag)),
                   weak_cond={M: cond_rank[M][0] for M in MS},
                   weak_rank={M: cond_rank[M][1] for M in MS},
                   weak_square_or_underdetermined={M: cond_rank[M][2] for M in MS},
                   **{kk: dict(mean=float(np.mean(v)), median=float(np.median(v)), max=float(np.max(v)))
                      for kk, v in e.items()})
        rows.append(row)
        print(f"RESULT k={k:3d} proj {row['proj']['mean']:.3e} galerkin {row['galerkin']['mean']:.3e} "
              f"fd {row['fd']['mean']:.3e} " + " ".join(f"{kk} {row[kk]['mean']:.3e}" for kk in e if kk.startswith('weak')),
              flush=True)
    json.dump(dict(config=dict(N=N, n_train=N_TRAIN, n_test=N_TEST, ks=KS, ms=MS, seed=mp.SEED,
                               backend=jax.default_backend()),
                   fom_max_rel_residual=fom_res, singular_values=[float(s) for s in sv[:64]], rows=rows,
                   complete=True), open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
