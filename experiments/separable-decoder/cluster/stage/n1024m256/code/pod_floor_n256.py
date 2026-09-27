"""POD/SVD projection-floor diagnostic for the N=256 push (user directive
2026-08-23): the separable decoder's oracle is hard-lower-bounded by the
rank-R POD floor of the training family (ALL x-dependence lives in span{g},
dim R), so the rank-R floors tell us which R makes a ~1e-3 solve error
possible at all, per PDE, BEFORE any training job is sunk into it.

Everything is f64 (GRAM64-style Gram in f64 per project rules; an f32 Gram
once faked a 2e-4 floor).  This is a DIAGNOSTIC: SVD/POD is used to bound the
architecture, it is never used in any decoder, solve path, or reported ROM.

Outputs, per PDE:
  * singular-value spectrum of the training snapshot matrix (as trained on:
    Poisson = the 512 seed-0 training fields; Burgers = the same 8192-state
    subsample the auto-decoder trains on, identical rng stream),
  * family (Frobenius) rank-R floors,
  * per-test-field rank-R projection floors for the SAME held-out and
    fresh-seed cohorts the solve jobs use (mean/max over fields),
  * Poisson only: floors of the fresh cohort onto a DENSER family basis
    (2048 sources, seed 4242) -- measures how much of the fresh-cohort floor
    is sampling density rather than family rank.

Usage:  PDE=poisson|burgers N=256 RS=64,128,256,512[,1024] python pod_floor_n256.py
(one PDE per process: the poisson and burgers dependency stacks both import a
module named ms_parametric with different env-frozen architectures and cannot
coexist in one interpreter -- same reason sep_poisson/sep_burgers are separate
scripts).  Writes pod_floor_N<N>_<pde>.json in cwd and prints the tables.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc

import jax
import jax.numpy as jnp

F64 = jnp.float64

PDE = os.environ.get("PDE", "poisson")
if PDE == "poisson":
    import pro_common as pc                  # noqa: E402  (path set by sc)
    from pro_common import mp                # noqa: E402
else:
    import blat_common as bc                 # noqa: E402

N = int(os.environ.get("N", "256"))
RS = [int(v) for v in os.environ.get("RS", "64,128,256,512").split(",")]
N_TEST = int(os.environ.get("N_TEST", "16"))
N_TEST_B = int(os.environ.get("N_TEST_B", "8"))
FRESH_SEED = int(os.environ.get("FRESH_SEED", "777"))
DENSE_SEED = int(os.environ.get("DENSE_SEED", "4242"))
DENSE_M = int(os.environ.get("DENSE_M", "2048"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
SEED0 = int(os.environ.get("SEED0", "0"))
OUT = os.environ.get("OUT", f"pod_floor_N{N}_{PDE}.json")
CHUNK = int(os.environ.get("CHUNK", "1024"))


def gram_spectrum(X):
    """f64 Gram G = X X^T (chunked matmul on device), eigh, descending
    singular values + Gram eigenvectors.  X: (S, n) np.float64."""
    S = X.shape[0]
    Xd = jnp.asarray(X, dtype=F64)
    G = np.zeros((S, S), dtype=np.float64)
    for i in range(0, S, CHUNK):
        Gi = np.asarray(Xd[i:i + CHUNK] @ Xd.T)
        G[i:i + CHUNK] = Gi
    evals, evecs = np.linalg.eigh(G)         # ascending
    evals = evals[::-1].copy()
    evecs = evecs[:, ::-1].copy()
    evals = np.maximum(evals, 0.0)
    svals = np.sqrt(evals)
    return svals, evecs, Xd


def family_floors(svals, Rs):
    tot = float(np.sum(svals ** 2))
    out = {}
    for R in Rs:
        R_eff = min(R, svals.size)
        tail = max(0.0, 1.0 - float(np.sum(svals[:R_eff] ** 2)) / tot)
        out[R] = float(np.sqrt(tail))
    return out


def test_floors(svals, evecs, Xd, Y, Rs, eps=1e-13):
    """Rank-R projection floors of test fields Y (T, n) onto the training POD
    basis: coeff_i = v_i^T (X y) / s_i;  err^2 = ||y||^2 - sum coeff^2."""
    Yd = jnp.asarray(Y, dtype=F64)
    B = np.asarray(Xd @ Yd.T)                        # (S, T)
    ynorm2 = np.asarray(jnp.sum(Yd * Yd, axis=1))    # (T,)
    keep = svals > eps * svals[0]
    C = (evecs[:, keep].T @ B) / svals[keep][:, None]  # (rank, T)
    out = {}
    for R in Rs:
        R_eff = min(R, int(keep.sum()))
        proj2 = np.sum(C[:R_eff] ** 2, axis=0)
        rel = np.sqrt(np.maximum(ynorm2 - proj2, 0.0) / ynorm2)
        out[R] = dict(mean=float(np.mean(rel)), max=float(np.max(rel)),
                      per_field=[float(v) for v in rel])
    return out


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"POD-floor diagnostic PDE={PDE} N={N} Rs={RS}")
    t_all = time.time()
    report = dict(config=dict(pde=PDE, N=N, Rs=RS, n_test=N_TEST, n_test_b=N_TEST_B,
                              fresh_seed=FRESH_SEED, dense_seed=DENSE_SEED,
                              dense_m=DENSE_M, max_snaps=MAX_SNAPS, seed=SEED0,
                              f64=True, backend=dev.platform,
                              note="DIAGNOSTIC ONLY: POD bounds the separable "
                                   "architecture; no POD enters any model"),
                  complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    if PDE == "poisson":
        run_poisson(report, save)
    else:
        run_burgers(report, save)
    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE pod-floor {PDE} [{time.time()-t_all:.0f}s] -> {OUT}")


def run_poisson(report, save):
    grid = pc.Grid(N)
    int_idx = np.asarray(grid.ix_full * N + grid.iy_full)
    U_all = np.asarray(mp.build_snapshots(N)[0])
    U_tr = U_all[:mp.N_TRAIN][:, int_idx]
    cx, cy, w, a, _ = mp.sample_params()
    Fs_held = np.stack([mp.source_interior(N, cx[mp.N_TRAIN + i],
                                           cy[mp.N_TRAIN + i],
                                           w[mp.N_TRAIN + i],
                                           a[mp.N_TRAIN + i])
                        for i in range(N_TEST)])
    cxf, cyf, wf, af, _ = mp.sample_params(seed=FRESH_SEED, m=N_TEST)
    Fs_fresh = np.stack([mp.source_interior(N, cxf[i], cyf[i], wf[i], af[i])
                         for i in range(N_TEST)])
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])
    U_held = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs_held))
                        ).reshape(N_TEST, -1)
    U_fresh = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs_fresh))
                         ).reshape(N_TEST, -1)
    sc.log(f"  poisson: training X {U_tr.shape}, test 2x{N_TEST}")
    sv, ev, Xd = gram_spectrum(U_tr)
    pois = dict(n_snapshots=int(U_tr.shape[0]),
                svals_head=[float(v) for v in sv[:32]],
                svals=[float(v) for v in sv],
                family_floor=family_floors(sv, RS),
                held=test_floors(sv, ev, Xd, U_held, RS),
                fresh=test_floors(sv, ev, Xd, U_fresh, RS))
    for R in RS:
        sc.log(f"  poisson R={R:4d}: family {pois['family_floor'][R]:.3e}  "
               f"held {pois['held'][R]['mean']:.3e}/{pois['held'][R]['max']:.3e}  "
               f"fresh {pois['fresh'][R]['mean']:.3e}/{pois['fresh'][R]['max']:.3e}")
    del Xd
    # dense-family basis (sampling-density control for the fresh floor)
    cxd, cyd, wd, ad, _ = mp.sample_params(seed=DENSE_SEED, m=DENSE_M)
    U_dense = []
    for i0 in range(0, DENSE_M, 256):
        Fd = np.stack([mp.source_interior(N, cxd[i], cyd[i], wd[i], ad[i])
                       for i in range(i0, min(i0 + 256, DENSE_M))])
        U_dense.append(np.asarray(jax.lax.map(solve_one, jnp.asarray(Fd))
                                  ).reshape(len(Fd), -1))
    U_dense = np.concatenate(U_dense)
    sc.log(f"  poisson dense family: {U_dense.shape}")
    svd_, evd, Xdd = gram_spectrum(U_dense)
    pois["dense"] = dict(n_snapshots=int(U_dense.shape[0]),
                         family_floor=family_floors(svd_, RS),
                         fresh=test_floors(svd_, evd, Xdd, U_fresh, RS))
    for R in RS:
        sc.log(f"  poisson R={R:4d} DENSE({DENSE_M}): family "
               f"{pois['dense']['family_floor'][R]:.3e}  fresh "
               f"{pois['dense']['fresh'][R]['mean']:.3e}/"
               f"{pois['dense']['fresh'][R]['max']:.3e}")
    del Xdd
    report["poisson"] = pois
    save()


def run_burgers(report, save):
    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    rng = np.random.default_rng(SEED0)               # same stream as training
    n_states = n_traj * T
    if n_states > MAX_SNAPS:
        pick = np.sort(rng.choice(n_states, MAX_SNAPS, replace=False))
    else:
        pick = np.arange(n_states)
    S_tr = U.reshape(n_states, n2)[pick][:, interior]
    U_test = np.asarray(d["U_test"], dtype=np.float64)[:N_TEST_B]
    Y_test = U_test.reshape(-1, n2)[:, interior]     # all states of all trajs
    sc.log(f"  burgers: training X {S_tr.shape}, test states {Y_test.shape}")
    sv_b, ev_b, Xb = gram_spectrum(S_tr)
    burg = dict(n_snapshots=int(S_tr.shape[0]),
                svals_head=[float(v) for v in sv_b[:32]],
                svals=[float(v) for v in sv_b[:4096]],
                family_floor=family_floors(sv_b, RS),
                test_states=test_floors(sv_b, ev_b, Xb, Y_test, RS))
    # per-trajectory mean floor (what a rollout error is bounded by)
    for R in RS:
        per = np.asarray(burg["test_states"][R]["per_field"]).reshape(
            N_TEST_B, T)
        burg["test_states"][R]["per_traj_mean"] = [float(v)
                                                   for v in per.mean(axis=1)]
        sc.log(f"  burgers R={R:4d}: family {burg['family_floor'][R]:.3e}  "
               f"test-state {burg['test_states'][R]['mean']:.3e}/"
               f"{burg['test_states'][R]['max']:.3e}  traj-mean "
               f"{np.mean(per.mean(axis=1)):.3e}")
    report["burgers"] = burg
    save()


if __name__ == "__main__":
    main()
