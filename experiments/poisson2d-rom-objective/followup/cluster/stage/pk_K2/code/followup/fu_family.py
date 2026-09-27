"""Complexity ladder for the Poisson-2D testbed: does the coordinate ROM's
saturation latent dimension track the family's INTRINSIC dimension while POD's
grows with the sharpness of the fields?

The family is NB independent Gaussian bump sources,
    -lap u = sum_{b<NB} a_b exp(-|x - c_b|^2 / 2 w_b^2),   u = 0 on the walls,
so the intrinsic dimension is exactly 4*NB (4, 8, 12 for NB = 1, 2, 3).  NB=1 is
`ms_parametric.sample_params` verbatim, so that column is the same family as the
main study.  Poisson is linear, so the truth is one FD/CG solve on the summed
source at the testbed's own tolerance (residual asserted).

For each K in KS this trains a hard-BC FiLM auto-decoder with the SAME recipe and
the SAME budget as `fu_train.py`/`pro_train.py`, then measures, on N_TEST held-out
sources:
  train recon, ORACLE finite-budget inferred latent (LM on the data misfit to the
  held-out field -- not available to the ROM), coordinate ROM `weak_a1_M{M}` on the
  full grid and with meshfree NNLS-EQ at m points, and the FD-LSPG control.
The POD ladder (projection floor / Galerkin / the same weak objective, full grid,
exact minimiser) is computed once per family over the same k values.

Usage: NB=2 [KS=2,4,6,8,12,16,24,32] [M=64] [MQ=256] [N_TEST=16] [GN_ITERS=60]
       python followup/fu_family.py <outdir>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import pro_common as pc  # noqa: E402
from pro_common import mp  # noqa: E402
from fu_eq import eq_fit  # noqa: E402

NB = int(os.environ.get("NB", "1"))
KS = [int(k) for k in os.environ.get("KS", "2,4,6,8,12,16,24,32").split(",") if k]
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
POD_KS = [int(k) for k in os.environ.get("POD_KS", "2,4,6,8,12,16,24,32,48,64").split(",") if k]
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
N, N_TRAIN, N_VAL, STEPS = mp.N, mp.N_TRAIN, mp.N_VAL, mp.STEPS


def sample_family(nb, m):
    """Parameters of nb bumps.  Bump 0 IS ms_parametric.sample_params (so NB=1 is
    the main study's family); bumps 1.. come from their own fixed streams with the
    same ranges."""
    cx, cy, w, a, z = mp.sample_params(mp.SEED, m)
    P = [np.stack([cx, cy, w, a], 1)]
    Z = [z]
    for b in range(1, nb):
        r = np.random.default_rng(mp.SEED + 977 * b)
        cxb = r.uniform(0.15, 0.85, m); cyb = r.uniform(0.15, 0.85, m)
        wb = np.exp(r.uniform(np.log(0.02), np.log(0.1), m)); ab = r.uniform(0.5, 2.0, m)
        P.append(np.stack([cxb, cyb, wb, ab], 1))
        Z.append(np.stack([(cxb - 0.5) / 0.35, (cyb - 0.5) / 0.35,
                           (np.log(wb) - np.log(0.045)) / 0.8, (ab - 1.25) / 0.75], 1))
    return np.stack(P, 1), np.concatenate(Z, 1)          # (m, nb, 4), (m, 4*nb)


def source_interior(n, p):
    """p: (nb, 4) -> interior source field (n-2, n-2)."""
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    return sum(pb[3] * np.exp(-((Xi - pb[0]) ** 2 + (Yi - pb[1]) ** 2) / (2 * pb[2] ** 2))
               for pb in p)


def build(n, P, chunk=128):
    op = lambda v: mp.neg_lap_interior(v, n)
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL,
                                                             maxiter=mp.CG_MAXITER)[0])
    resid = jax.jit(jax.vmap(lambda u, F: jnp.linalg.norm(op(u) - F) / jnp.linalg.norm(F)))
    m = P.shape[0]
    U = np.zeros((m, n, n)); res_max = 0.0
    t0 = time.time()
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        F = jnp.asarray(np.stack([source_interior(n, P[i]) for i in range(s, e)]))
        Ui = jax.lax.map(solve_one, F)
        U[s:e, 1:-1, 1:-1] = np.asarray(Ui)
        res_max = max(res_max, float(np.max(np.asarray(resid(Ui, F)))))
    print(f"  FOM: {m} CG solves in {time.time()-t0:.0f}s, max rel residual {res_max:.2e}", flush=True)
    if not np.isfinite(res_max) or res_max > 1e-10:
        raise SystemExit(f"FOM not converged ({res_max:.2e})")
    return jnp.asarray(U.reshape(m, n * n))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"jax_backend={jax.default_backend()} NB={NB} intrinsic_dim={4*NB} KS={KS} "
          f"M={M_MODES} m={MQ}", flush=True)
    grid = pc.Grid(N)
    coords = grid.coords
    P, ztrue = sample_family(NB, N_TRAIN + N_VAL)
    U = build(N, P)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    U_test = np.asarray(U_va[:N_TEST]); tn = np.linalg.norm(U_test, axis=1)
    int_idx = grid.ix_full * N + grid.iy_full
    Fs = np.stack([source_interior(N, P[N_TRAIN + i]) for i in range(N_TEST)])
    nn_idx = np.argmin(((ztrue[N_TRAIN:, None, :] - ztrue[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
    spec = dict(kind="weak", alpha=1.0, M=M_MODES)
    report = dict(config=dict(NB=NB, intrinsic_dim=4 * NB, ks=KS, M=M_MODES, m=MQ, N=N,
                              n_train=N_TRAIN, n_test=N_TEST, gn_iters=GN_ITERS, steps=STEPS,
                              seed=mp.SEED, backend=jax.default_backend(),
                              device=str(jax.devices()[0])),
                  rows=[], pod=[])

    def save():
        json.dump(report, open(os.path.join(OUTDIR, f"family_NB{NB}.json"), "w"), indent=1,
                  default=float)

    # ---------------- POD ladder (linear control, exact minimiser) ----------------
    X = np.asarray(U_tr)[:, int_idx]
    Vfull, sv, _ = np.linalg.svd(X.T, full_matrices=False)
    op_j = jax.jit(lambda v2d: grid.op(v2d))
    spec_j = jax.jit(lambda v2d: grid.spec(v2d))
    lam = np.asarray(grid.lam)
    mask = np.asarray(grid.mode_mask(M_MODES)).astype(bool)

    def full_field(c_int):
        u = np.zeros(N * N); u[int_idx] = c_int; return u

    for k in POD_KS:
        V = Vfull[:, :k]
        AV = np.stack([np.asarray(op_j(jnp.asarray(V[:, j].reshape(grid.n_i, grid.n_i)))).reshape(-1)
                       for j in range(k)], 1)
        SV = np.stack([np.asarray(spec_j(jnp.asarray(V[:, j].reshape(grid.n_i, grid.n_i))))
                       for j in range(k)], -1)
        Ag = V.T @ AV
        e = {"proj": [], "galerkin": [], "fd": [], "weak": []}
        for i in range(N_TEST):
            u_int = U_test[i][int_idx]; f = Fs[i].reshape(-1)
            err = lambda c: float(np.linalg.norm(full_field(c) - U_test[i]) / tn[i])
            e["proj"].append(err(V @ (V.T @ u_int)))
            e["galerkin"].append(err(V @ np.linalg.solve(Ag, V.T @ f)))
            e["fd"].append(err(V @ np.linalg.lstsq(AV, f, rcond=None)[0]))
            Cf = np.asarray(spec_j(jnp.asarray(Fs[i])))
            A_ = SV[mask]; b_ = Cf[mask] / lam[mask]
            e["weak"].append(err(V @ np.linalg.lstsq(A_, b_, rcond=None)[0]))
        report["pod"].append(dict(k=k, n_modes=int(mask.sum()), square_or_underdetermined=bool(mask.sum() <= k),
                                  **{kk: float(np.mean(v)) for kk, v in e.items()}))
        print(f"RESULT pod NB={NB} k={k:3d} proj {np.mean(e['proj']):.3e} galerkin "
              f"{np.mean(e['galerkin']):.3e} weak {np.mean(e['weak']):.3e} fd {np.mean(e['fd']):.3e}",
              flush=True)
        save()

    # ---------------- coordinate ROM ladder ----------------
    for K in KS:
        t0 = time.time()
        np_rng = np.random.default_rng(mp.SEED)
        key = jax.random.PRNGKey(mp.SEED + 100 + K)
        stages, Z_tr, eps0, n_freq, adam_loss = pc.train_autodecoder_stage0(
            key, np_rng, coords, U_tr, K, True, STEPS, mp.BATCH, mp.P_SUB)
        dec = pc.make_decoder(stages, hard_bc=True)
        pred = jnp.concatenate([jax.vmap(lambda z: dec(z, coords))(jnp.asarray(Z_tr[s:s + 64]))
                                for s in range(0, N_TRAIN, 64)])
        g_tr, m_tr, _ = mp.rel_metrics(pred, U_tr)
        rJ = jax.jit(lambda z, u: (dec(z, coords) - u,
                                   jax.jacfwd(lambda zz: dec(zz, coords) - u)(z)))
        rn = jax.jit(lambda z, u: jnp.linalg.norm(dec(z, coords) - u))
        inits = {"mean": np.tile(Z_tr.mean(0), (N_TEST, 1)), "nearest": Z_tr[nn_idx][:N_TEST]}
        orc = {}
        for name, Z0 in inits.items():
            rels = []
            for i in range(N_TEST):
                u = jnp.asarray(U_test[i])
                _, r, _ = pc.lm_solve(lambda zz: rJ(zz, u), lambda zz: rn(zz, u),
                                      jnp.asarray(Z0[i]), GN_ITERS)
                rels.append(r / tn[i])
            orc[name] = float(np.mean(rels))
        row = dict(K=K, train_global_rel=g_tr, train_mean_rel_l2=m_tr, oracle=orc,
                   n_freq=int(n_freq), eps0=float(eps0), rom={})
        # ROM arms: weak full grid, weak + meshfree EQ (m=MQ), and the FD-LSPG control
        pts_eq, wq_eq, eq_info = eq_fit(dec, grid, Z_tr, K, M_MODES, MQ, "offgrid")
        row["eq_info"] = eq_info
        arms = [("weak_full", "grid", np.asarray(grid.coords_int), np.ones(grid.n_i ** 2),
                 dict(kind="weak", alpha=1.0, M=M_MODES)),
                ("weak_eq_meshfree", "offgrid", pts_eq, wq_eq, dict(kind="weak", alpha=1.0, M=M_MODES)),
                ("fd_full", "grid", np.asarray(grid.coords_int), np.ones(grid.n_i ** 2),
                 dict(kind="fd"))]
        for name, pk, pts, wq, sp in arms:
            HgV, Vv = pc.make_colloc_objective(dec, grid, sp, pk, bc_beta=0.0)
            if sp["kind"] == "weak":
                PhiT, Wl = pc.colloc_mode_table(grid, sp, pk, pts)
                pts_arg = jnp.asarray(pts)
            else:
                PhiT, Wl = jnp.zeros((1, 1)), jnp.zeros((1,))
                ix, iy = grid.ix_full, grid.iy_full
                pts_arg, keep_arg = grid.stencil(ix, iy)
            errs, its, reas = [], [], []
            for i in range(N_TEST):
                if sp["kind"] == "weak":
                    f_m = jnp.asarray(pc.weak_source_term(grid, sp, pk, Fs[i]))
                    args = (pts_arg, jnp.zeros((1, 2)), jnp.asarray(wq), PhiT, Wl, f_m)
                else:
                    f_m = jnp.asarray(Fs[i].reshape(-1))
                    args = (pts_arg, keep_arg, jnp.asarray(wq), PhiT, Wl, f_m)
                z, val, info = pc.lm_generic(lambda zz: HgV(zz, *args), lambda zz: Vv(zz, *args),
                                             jnp.asarray(inits["nearest"][i]), GN_ITERS)
                errs.append(float(np.linalg.norm(np.asarray(dec(z, coords)) - U_test[i]) / tn[i]))
                its.append(info["n_jac_evals"]); reas.append(info["reason"])
            row["rom"][name] = dict(rel_l2_mean=float(np.mean(errs)), rel_l2_med=float(np.median(errs)),
                                    rel_l2_max=float(np.max(errs)), iters_mean=float(np.mean(its)),
                                    reasons={r: reas.count(r) for r in set(reas)},
                                    m=int(len(wq)))
        row["secs"] = time.time() - t0
        report["rows"].append(row); save()
        with open(os.path.join(OUTDIR, f"family_NB{NB}_K{K}_stages.pkl"), "wb") as f:
            pickle.dump({"config": dict(mp.CONFIG, K_LAT=K, n_stages=1, hard_bc=1, NB=NB),
                         "stages": pc.stages_to_np(stages), "z_tr": Z_tr}, f)
        print(f"RESULT coord NB={NB} K={K:3d} train {m_tr:.3e} oracle(nearest) {orc['nearest']:.3e} "
              f"ROM full {row['rom']['weak_full']['rel_l2_mean']:.3e} EQ m={MQ} "
              f"{row['rom']['weak_eq_meshfree']['rel_l2_mean']:.3e} FD "
              f"{row['rom']['fd_full']['rel_l2_mean']:.3e} [{row['secs']:.0f}s]", flush=True)
    report["complete"] = True; save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
