"""Stage 2: latent-stepping ROM evaluation on HELD-OUT Heat-2D trajectories.

Inputs : hlat_ad_N{N}_K{K}.pkl (auto-decoder + POD basis) from hlat_train_ad.py;
         data regenerated from seed (fingerprint asserted equal).
The ROM knows: the initial condition u0, the diffusivity kappa, the PDE.  It never
sees the held-out trajectory (oracle floors are computed separately and
labelled).

Per variant "solver:colloc:objective" (env VARIANTS, comma-separated):
  solver    lspg | galerkin
  colloc    full | rand<m> | biased<m> | offgrid<m> (fd) | eq<m> | eqoff<m> (weak)
  objective fd | weak<M> | weakc<M> | weakall
POD control: POD_KS (default 6,8,16,32,64) x POD_VARIANTS.

Reports per variant: trajectory rel-L2 (mean over the 51 slices) mean/median/
max over N_TEST trajectories, per-time mean curve, iterations (cold step 0 vs
warm), termination reasons, blow-ups, per-step wall time; ROM-vs-FOM timing
(warm-up 2, median of 7, block_until_ready, same device, FOM = the same
jitted implicit solver at batch 1).

Usage: N=64 K_LAT=8 [N_TEST=16] [VARIANTS=...] python hlat_rom.py <ad.pkl> <outdir>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import hlat_common as bc
from hlat_common import F64, log, lm_solve

AD_PKL = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
N = bc.N
T1 = bc.NUM_STEPS + 1
# solver:colloc:objective.  objective weak<M> = weak-form Galerkin against M discrete
# sine modes with the DISCRETE eigenvalues (exact for the FOM); weakc<M> = the same
# with the continuum eigenvalues.  Both need only decoder OUTPUT at the quadrature
# nodes: colloc full | eq<m> (NNLS-EQ grid nodes) | eqoff<m> (NNLS-EQ meshfree pool).
# fd = strong FD residual control (full | rand<m> | biased<m> | offgrid<m>).
# weakall = all (N-2)^2 modes on the full grid: must equal lspg:full:fd (cross-check).
# Random / importance / off-grid STRONG-form collocation is deliberately NOT in the
# default list: it lost on both Poisson and Burgers and costs cluster time.
DEFAULT_VARIANTS = ("lspg:full:fd,galerkin:full:fd,lspg:full:weakall,galerkin:full:weakall,"
                    "lspg:full:weak16,lspg:full:weak32,lspg:full:weak64,"
                    "lspg:full:weak128,lspg:full:weak256,lspg:full:weak64a0,"
                    "galerkin:full:weak64,"
                    "lspg:eq256:weak64,lspg:eq512:weak64,"
                    "lspg:eqoff256:weak64,lspg:eqoff512:weak64,"
                    "lspg:eq512:weak256,lspg:eq1024:weak256,"
                    "lspg:full:weakc64,lspg:eqoff512:weakc64")
VARIANTS = [v for v in os.environ.get("VARIANTS", DEFAULT_VARIANTS).split(",") if v]
POD_KS = [int(k) for k in os.environ.get("POD_KS", "6,8,16,32,64").split(",") if k]
POD_VARIANTS = [v for v in os.environ.get(
    "POD_VARIANTS", "lspg:full:fd,galerkin:full:fd,lspg:full:weak64,lspg:eq512:weak64").split(",") if v]
EQ_RNG_SEED = 4321
FLOOR_BUDGET = int(os.environ.get("FLOOR_BUDGET", "60"))
DO_TIMING = int(os.environ.get("DO_TIMING", "1"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))


def summarize(runs):
    """Accuracy statistics are over COMPLETED rollouts only (n_completed of
    n_total); blow-ups are counted separately and NOT averaged in."""
    ok = [r for r in runs if r["n_done"] == bc.NUM_STEPS]
    tr = np.array([r["traj_rel"] for r in ok]) if ok else np.array([np.nan])
    it = np.array([r["iters"] for r in ok], dtype=float)
    at = np.array([r["attempts"] for r in ok], dtype=float)
    per = np.array([r["per_time"] for r in runs])
    reasons = {}
    for r in runs:
        for s_ in r["reasons"]:
            reasons[s_] = reasons.get(s_, 0) + 1
    out = dict(n_total=len(runs), n_completed=len(ok),
               n_blowup=int(sum(r["n_done"] < bc.NUM_STEPS for r in runs)),
               traj_rel_mean=float(np.mean(tr)), traj_rel_median=float(np.median(tr)),
               traj_rel_max=float(np.max(tr)),
               per_time_mean=np.nanmean(per, axis=0).tolist(),
               per_time_survivors=np.sum(np.isfinite(per), axis=0).tolist(),
               iters_cold_step0=float(it[:, 0].mean()) if it.size else float("nan"),
               iters_warm_mean=float(it[:, 1:].mean()) if it.size else float("nan"),
               iters_warm_max=float(it[:, 1:].max()) if it.size else float("nan"),
               attempts_cold_step0=float(at[:, 0].mean()) if at.size else float("nan"),
               attempts_warm_mean=float(at[:, 1:].mean()) if at.size else float("nan"),
               res_step_mean=float(np.mean([np.mean(r["res"]) for r in runs])),
               res_final_mean=float(np.mean([r["res"][-1] for r in runs])),
               res_over_res_init_mean=float(np.mean([
                   np.mean(np.asarray(r["res"]) / np.maximum(np.asarray(r["res_init"]), 1e-300))
                   for r in runs])),
               reasons=reasons,
               step_time_ms_median=float(1e3 * np.median(np.concatenate(
                   [r["step_time"] for r in runs]))),
               ic_misfit_mean=float(np.mean([r["ic_rel"] for r in runs])),
               ic_init_used={n: sum(r["ic_init"] == n for r in runs)
                             for n in set(r["ic_init"] for r in runs)})
    return out


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  variants={VARIANTS}")
    with open(AD_PKL, "rb") as f:
        ck = pickle.load(f)
    K = ck["k_lat"]
    for key_, val_ in (("bc_mode", bc.BC_MODE), ("N", N), ("ad_hidden", bc.AD_HIDDEN),
                       ("ad_layers", bc.AD_LAYERS), ("n_train", bc.N_TRAIN), ("seed", bc.SEED),
                       ("k_lat", bc.K_LAT)):
        if ck["config"][key_] != val_:
            raise SystemExit(f"checkpoint/config mismatch on {key_}: {ck['config'][key_]} vs {val_}")
    if K != bc.K_LAT or ck["Z_train"].shape[-1] != K or ck["V"].shape[0] != N * N:
        raise SystemExit(f"checkpoint shape mismatch: k_lat {K}/{bc.K_LAT}, "
                         f"Z {ck['Z_train'].shape}, V {ck['V'].shape}, N={N}")
    if not all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(ck["params"])):
        raise SystemExit("checkpoint contains non-finite parameters")
    dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], K)
    Ztr = ck["Z_train"]                                   # (n_tr, T1, K)
    V = ck["V"]
    if max(POD_KS) > V.shape[1]:
        raise SystemExit(f"POD_KS {POD_KS} exceeds stored basis rank {V.shape[1]}")
    d = bc.build_data(N)
    fp = bc.data_fingerprint(d["U"])
    fp_rel = max(abs(fp["sum"] - ck["data_fingerprint"]["sum"]) / abs(ck["data_fingerprint"]["sum"]),
                 abs(fp["sumsq"] - ck["data_fingerprint"]["sumsq"]) / ck["data_fingerprint"]["sumsq"])
    if fp_rel > 1e-6:
        raise SystemExit(f"data fingerprint mismatch ({fp_rel:.2e}): {fp} vs {ck['data_fingerprint']}")
    if fp_rel > 1e-12:
        log(f"  WARNING: data fingerprint differs by {fp_rel:.2e} (cross-machine rounding?)")
    U = d["U"]
    U_te = d["U_test"]                                     # fresh TEST_SEED trajectories
    kap_te = d["kappa_test"]
    U_tr = U[:bc.N_TRAIN]
    interior_np = bc.interior_indices(N)
    u0_rms = np.sqrt(np.mean(U_te[:, 0][:, interior_np] ** 2, axis=1))   # tolerance scales
    n2 = N * N
    coords = jnp.asarray(bc.grid_coords(N))
    interior = bc.interior_indices(N)
    report = dict(config=dict(bc.CONFIG, variants=VARIANTS, pod_ks=POD_KS,
                              pod_variants=POD_VARIANTS, floor_budget=FLOOR_BUDGET,
                              ad_pkl=os.path.basename(AD_PKL), ad_config=ck["config"]),
                  backend=jax.default_backend(), data_fingerprint=fp,
                  test_seed=bc.TEST_SEED, max_fom_rel_residual=d["max_fom_rel_residual"],
                  train_rel_mean=ck["train_rel_mean"],
                  oracle_pod_projection_floor_val=ck["pod_floors"])

    # ---------------- checks: point-local residual == FOM residual ----------------
    _, fom_res = bc.make_fom(N)
    st = jnp.asarray(bc.stencil_indices(interior, N))
    u1, u0 = jnp.asarray(U_te[0, 1]), jnp.asarray(U_te[0, 0])
    r_local = bc.be_residual_from_stencil(u1[st], u0[jnp.asarray(interior)],
                                          float(kap_te[0]), N)
    r_fom = fom_res(u1, u0, float(kap_te[0]))[jnp.asarray(interior)]
    chk = dict(local_vs_fom_maxabs=float(jnp.max(jnp.abs(r_local - r_fom))),
               fom_traj_step1_rel_res=float(jnp.linalg.norm(r_fom) / jnp.linalg.norm(u0)))
    # exactness of the weak form: with ALL (N-2)^2 sine modes and alpha=0 the test
    # matrix Phi is square-orthogonal, so r_weak = Phi^T r_fd -- the two residual
    # vectors must have identical norms and the LM/Galerkin normal equations are the
    # same.  Checked at 3 random latents against the FULL-grid strong-form operator.
    ops_fd_chk = bc.make_step_ops(dec, N, dict(kind="grid", idx=interior), "fd", "lspg")
    ops_wa_chk = bc.make_weak_ops(dec, N, dict(kind="grid", idx=interior, w=None),
                                  kind="weak", M=(N - 2) ** 2, alpha=0.0, solver="lspg")
    rngc = np.random.default_rng(99)
    devs, devs_g, devs_l, devs_h = [], [], [], []
    for _ in range(3):
        zc = jnp.asarray(rngc.normal(size=K) * float(np.std(Ztr)), dtype=F64)
        prev = jnp.asarray(rngc.normal(size=(N - 2) ** 2) * 1e-2)
        kc = float(kap_te[0])
        r_fd_ = ops_fd_chk["r_w"](zc, prev, kc)
        r_wa_ = ops_wa_chk["r_w"](zc, prev, kc)
        devs.append(float(abs(jnp.linalg.norm(r_fd_) - jnp.linalg.norm(r_wa_))
                          / jnp.linalg.norm(r_fd_)))
        _, J_fd_, JD_fd_ = ops_fd_chk["rJ"](zc, prev, kc)
        _, J_wa_, JD_wa_ = ops_wa_chk["rJ"](zc, prev, kc)
        g_fd = JD_fd_.T @ r_fd_
        g_wa = JD_wa_.T @ r_wa_
        devs_g.append(float(jnp.linalg.norm(g_fd - g_wa) / jnp.linalg.norm(g_fd)))
        # LSPG normal equations: J^T r and J^T J must match too
        jr_fd, jr_wa = J_fd_.T @ r_fd_, J_wa_.T @ r_wa_
        H_fd, H_wa = J_fd_.T @ J_fd_, J_wa_.T @ J_wa_
        devs_l.append(float(jnp.linalg.norm(jr_fd - jr_wa) / jnp.linalg.norm(jr_fd)))
        devs_h.append(float(jnp.linalg.norm(H_fd - H_wa) / jnp.linalg.norm(H_fd)))
    chk["weakall_vs_fd_resnorm_reldev"] = float(max(devs))
    chk["weakall_vs_fd_galerkin_grad_reldev"] = float(max(devs_g))
    chk["weakall_vs_fd_lspg_grad_reldev"] = float(max(devs_l))
    chk["weakall_vs_fd_lspg_hess_reldev"] = float(max(devs_h))
    del ops_fd_chk, ops_wa_chk
    # the hard-BC factor must be EXACTLY zero on the wall nodes (the ghost-zero
    # assumption behind -L phi = lam phi)
    wall = np.setdiff1d(np.arange(n2), interior)
    zc = jnp.asarray(np.random.default_rng(5).normal(size=K), dtype=F64)
    chk["decoder_wall_maxabs"] = float(jnp.max(jnp.abs(dec(zc, coords[jnp.asarray(wall)]))))
    log(f"  checks: {chk}")
    report["checks"] = chk
    bad = [k_ for k_, tolk in (("local_vs_fom_maxabs", 1e-12),
                               ("weakall_vs_fd_resnorm_reldev", 1e-10),
                               ("weakall_vs_fd_galerkin_grad_reldev", 1e-8),
                               ("weakall_vs_fd_lspg_grad_reldev", 1e-8),
                               ("weakall_vs_fd_lspg_hess_reldev", 1e-8),
                               ("decoder_wall_maxabs", 1e-300))
           if not np.isfinite(chk[k_]) or chk[k_] > tolk]
    if bad:
        raise SystemExit(f"identity checks FAILED: {[(k_, chk[k_]) for k_ in bad]}")

    # ---------------- floors on the test trajectories ----------------
    # (a) POD projection floors (linear ceiling of the control)
    Ute = U_te.reshape(-1, n2)
    pf = {}
    for k in POD_KS:
        rec = (Ute @ V[:, :k]) @ V[:, :k].T
        pf[k] = float(np.mean(np.linalg.norm(rec - Ute, axis=1) / np.linalg.norm(Ute, axis=1)))
    report["oracle_pod_projection_floor_test"] = pf
    # (b) ORACLE per-snapshot inferred latents (LM on the data misfit to the
    #     held-out field; NOT available to the ROM) -- floor 2 of the Poisson study
    zmean_t = Ztr.mean(axis=0)                            # (T1, K)
    f_mis = lambda z, u: dec(z, coords) - u
    rJ_mis = jax.jit(lambda z, u: (f_mis(z, u), jax.jacfwd(f_mis)(z, u)))
    rn_mis = jax.jit(lambda z, u: jnp.linalg.norm(f_mis(z, u)))
    Utr_flat = U_tr.reshape(-1, n2)
    Utr_sq = np.sum(Utr_flat ** 2, axis=1)
    t0 = time.time()
    orc = np.zeros((bc.N_TEST, T1)); orc_Z = np.zeros((bc.N_TEST, T1, K))
    for i in range(bc.N_TEST):
        for n in range(T1):
            u = jnp.asarray(U_te[i, n])
            # nearest TRAIN snapshot by field distance (oracle: uses held-out field)
            j = int(np.argmin(Utr_sq - 2.0 * (Utr_flat @ U_te[i, n])))   # argmin ||u_j - u||
            best = None
            for name, z0 in (("mean", zmean_t[n]), ("nearest", Ztr.reshape(-1, K)[j])):
                z, r, info = lm_solve(lambda zz: rJ_mis(zz, u), lambda zz: rn_mis(zz, u),
                                      jnp.asarray(z0), FLOOR_BUDGET)
                rel = r / float(jnp.linalg.norm(u))
                if best is None or rel < best[0]:
                    best = (rel, np.asarray(z))
            orc[i, n], orc_Z[i, n] = best
    report["oracle_inferred_latent_test"] = dict(
        traj_rel_mean=float(orc.mean()), per_time_mean=orc.mean(axis=0).tolist(),
        secs=time.time() - t0)
    log(f"  ORACLE inferred-latent floor (held-out, budget {FLOOR_BUDGET}): "
        f"{orc.mean():.3e} [{time.time()-t0:.0f}s]")

    # ---------------- cold starts from the KNOWN u0 ----------------
    # Two starts: the mean training latent at t=0, and the latent of the training
    # trajectory whose IC is nearest to u0 (legitimate: u0 is known online).  The
    # SAME algorithm is used for the accuracy run and for the timed pipeline: the
    # jitted residual/Jacobian closures are built ONCE here (bc.fit_ic rebuilds them
    # per call, which would charge the Python-LM timing with a recompile each time).
    Utr0 = U_tr[:, 0]
    Utr0_j = jnp.asarray(Utr0)
    Ztr_t0 = jnp.asarray(Ztr[:, 0], dtype=F64)
    zmean0 = jnp.asarray(zmean_t[0], dtype=F64)
    nearest_jit = jax.jit(lambda u0: jnp.argmin(jnp.sum((Utr0_j - u0) ** 2, axis=1)))
    f_ic = lambda z, u: dec(z, coords) - u
    rJ_ic = jax.jit(lambda z, u: (f_ic(z, u), jax.jacfwd(f_ic)(z, u)))
    rn_ic = jax.jit(lambda z, u: jnp.linalg.norm(f_ic(z, u)))

    def fit_ic_py(u0j, inits):
        """Python-loop LM on the u0 misfit, best of the given starts (prebuilt jits)."""
        best = None
        for name, z0 in inits:
            z, r, info = lm_solve(lambda zz: rJ_ic(zz, u0j), lambda zz: rn_ic(zz, u0j),
                                  jnp.asarray(z0, dtype=F64), bc.IC_BUDGET)
            if best is None or r < best[1]:
                best = (z, r, name)
        return best

    ic_jit = bc.make_ic_solver_jit(dec, N, coords=coords)

    def ic_pipeline_jit(u0j):
        """The whole cold start on device: nearest-training-IC search + two jitted LM
        solves + best-of on the misfit.  This is the realizable online path."""
        j = nearest_jit(u0j)
        zA, rA, _, _, _, _ = ic_jit(zmean0, u0j, bc.IC_BUDGET)
        zB, rB, _, _, _, _ = ic_jit(Ztr_t0[j], u0j, bc.IC_BUDGET)
        return jnp.where(rA <= rB, zA, zB), jnp.minimum(rA, rB)

    ic_pipeline_jit = jax.jit(ic_pipeline_jit)

    ics, ics_jit = [], []
    for i in range(bc.N_TEST):
        u0j = jnp.asarray(U_te[i, 0])
        nu0 = float(jnp.linalg.norm(u0j))
        j = int(nearest_jit(u0j))                               # legit: u0 known
        z0, r, name = fit_ic_py(u0j, (("mean_t0", zmean_t[0]), ("nearest_ic", Ztr[j, 0])))
        ics.append((np.asarray(z0), r / nu0, name))
        zj, rj = ic_pipeline_jit(u0j)
        ics_jit.append((np.asarray(zj), float(rj) / nu0))
    report["ic_fit"] = dict(rel_mean=float(np.mean([r for _, r, _ in ics])),
                            rel_max=float(np.max([r for _, r, _ in ics])))
    report["ic_fit_jit"] = dict(
        rel_mean=float(np.mean([r for _, r in ics_jit])),
        rel_max=float(np.max([r for _, r in ics_jit])),
        mean_abs_diff_vs_python=float(np.mean([abs(a[1] - b[1])
                                               for a, b in zip(ics, ics_jit)])))
    log(f"  IC fit (u0 misfit): python {report['ic_fit']['rel_mean']:.3e}  "
        f"jit {report['ic_fit_jit']['rel_mean']:.3e} "
        f"(|diff| {report['ic_fit_jit']['mean_abs_diff_vs_python']:.1e})")

    # ---------------- timing helpers ----------------
    # Every reported ROM time is a REALIZABLE online pipeline measured end to end on the
    # same device in the same process: cold start (nearest-training-IC search + LM on the
    # u0 misfit) -> latent rollout -> decode all 51 slices -> block.  The FOM baseline is
    # the same jitted implicit CG solver at batch 1, which also produces 51 slices.
    dec_all_coord = jax.jit(lambda ZZ: jax.vmap(lambda zz: dec(zz, coords))(ZZ))

    def time_variant(var, ops, solver, dec_all, ic_pipe_jit, ic_py_fn, z0_fixed, us0,
                     kap, u0j):
        tf = jnp.full((bc.NUM_STEPS,),
                      bc.GN_TOL * us0 * ops.get("tol_scale", np.sqrt(ops["m"])))
        out = {}
        if solver == "lspg":
            def roll_once():
                o = ops["rollout_jit"](z0_fixed, kap, tf, bc.GN_BUDGET)
                o[0].block_until_ready()
            med, ts = bc.time_fn(roll_once, reps=TIME_REPS, warm=2)
            o = ops["rollout_jit"](z0_fixed, kap, tf, bc.GN_BUDGET)
            out["iters_total"] = int(jnp.sum(o[3]))
            out["impl"] = "device_scan"

            def e2e_jit():
                z0_, _ = ic_pipe_jit(u0j)
                C = ops["rollout_jit"](z0_, kap, tf, bc.GN_BUDGET)[0]
                dec_all(jnp.concatenate([z0_[None], C], axis=0)).block_until_ready()
            e2e_j, _ = bc.time_fn(e2e_jit, reps=TIME_REPS, warm=2)

            def e2e_py():
                z0_ = jnp.asarray(ic_py_fn(u0j)[0], dtype=F64)
                C = ops["rollout_jit"](z0_, kap, tf, bc.GN_BUDGET)[0]
                dec_all(jnp.concatenate([z0_[None], C], axis=0)).block_until_ready()
            e2e_p, _ = bc.time_fn(e2e_py, reps=3, warm=1)
            out.update(rollout_s_median=med, all=ts, end_to_end_jit_ic_s=e2e_j,
                       end_to_end_py_ic_s=e2e_p)
        else:
            # the Galerkin root solver is a Python loop and bc.rollout ALREADY decodes
            # every slice, so its rollout time is not comparable to the device-scan LSPG
            # number and must NOT be charged for decoding a second time.
            def roll_once():
                bc.rollout(dec, N, ops, z0_fixed, kap, us0)
            med, ts = bc.time_fn(roll_once, reps=3, warm=1)

            def e2e_py():
                z0_ = jnp.asarray(ic_py_fn(u0j)[0], dtype=F64)
                bc.rollout(dec, N, ops, z0_, kap, us0)
            e2e_p, _ = bc.time_fn(e2e_py, reps=3, warm=1)
            out.update(rollout_s_median=med, all=ts, impl="python_loop_incl_decode",
                       end_to_end_py_ic_s=e2e_p, iters_total=None)
        return out

    # ---------------- ROM variants (coord decoder) ----------------
    eq_cache = {}
    _perm = np.random.default_rng(EQ_RNG_SEED).permutation(Ztr.shape[0] * T1)
    fit_rows, val_rows = _perm[:bc.EQ_SNAPS], _perm[bc.EQ_SNAPS:2 * bc.EQ_SNAPS]
    Z_snap = Ztr.reshape(-1, K)[fit_rows]
    Z_snap_val = Ztr.reshape(-1, K)[val_rows]

    def build_ops(decoder, var, i, cache):
        solver, colloc_name, objective = var.split(":")
        if objective.startswith("weak"):
            kind, M, alpha = bc.parse_objective(objective, N)
            if colloc_name == "full":
                col = dict(kind="grid", idx=interior, w=None)
            else:
                pool = "off" if colloc_name.startswith("eqoff") else "grid"
                m = int(colloc_name[5:] if pool == "off" else colloc_name[2:])
                if pool == "off" and decoder.kind != "coord":
                    return None                       # meshfree pool undefined for POD rows
                # the heat EQ fit reproduces phi_i^T u only, so it depends on
                # (decoder, M, m, pool) -- NOT on alpha nor on which eigenvalues
                # ('weak' vs 'weakc') the residual then multiplies them by.
                key = (decoder.kind, decoder.k, M, m, pool)
                if key not in cache:
                    zs = Z_snap if decoder.kind == "coord" else (
                        (U_tr.reshape(-1, n2) @ np.asarray(decoder.V))[fit_rows])
                    col_ = bc.fit_eq_weights(decoder, N, M, m, zs, kind=kind, pool=pool,
                                             rng=np.random.default_rng(EQ_RNG_SEED))
                    # out-of-fit check: quadratured mode projections vs exact grid sums
                    # at EQ_SNAPS training latents NOT used in the NNLS fit
                    zv = (Z_snap_val if decoder.kind == "coord" else
                          (U_tr.reshape(-1, n2) @ np.asarray(decoder.V))[val_rows])
                    col_["info"].update(bc.eq_validate(decoder, N, M, col_, zv))
                    log(f"    EQ out-of-fit proj err {col_['info']['val_rel_proj_err']:.2e} "
                        f"(max {col_['info']['val_rel_proj_err_max']:.2e}), "
                        f"w in [{col_['info']['w_min']:.2e}, {col_['info']['w_max']:.2e}]")
                    cache[key] = col_
                col = cache[key]
            return bc.make_weak_ops(decoder, N, col, kind=kind, M=M, alpha=alpha,
                                    solver=solver)
        col = bc.make_collocation(colloc_name, N, np.random.default_rng(1234 + i),
                                  data_row=dict(u0=U_te[i, 0]))
        return bc.make_step_ops(decoder, N, col, objective, solver)

    results = {}
    timing = {}
    for var in VARIANTS:
        solver, colloc_name, objective = var.split(":")
        runs = []
        t0 = time.time()
        ops = None
        for i in range(bc.N_TEST):
            if colloc_name.startswith("biased") or ops is None:
                ops = build_ops(dec, var, i, eq_cache)
            z0, ic_rel, ic_init = ics[i]
            r = bc.rollout(dec, N, ops, z0, float(kap_te[i]), float(u0_rms[i]), U_true=U_te[i])
            r["ic_rel"], r["ic_init"] = ic_rel, ic_init
            del r["fields"]
            runs.append(r)
        s = summarize(runs)
        s["m"] = int(ops["m"]); s["secs"] = time.time() - t0
        s["M"] = ops.get("M"); s["alpha"] = ops.get("alpha")
        s["eq_info"] = ops.get("colloc_info")
        results[var] = s
        log(f"  {var:24s} m={ops['m']:5d}  traj rel mean {s['traj_rel_mean']:.3e} "
            f"(med {s['traj_rel_median']:.3e}, max {s['traj_rel_max']:.3e}) "
            f"blowups {s['n_blowup']}  iters cold {s['iters_cold_step0']:.1f} "
            f"warm {s['iters_warm_mean']:.2f}  step {s['step_time_ms_median']:.1f} ms  "
            f"[{s['secs']:.0f}s]")
        if DO_TIMING:
            timing[var] = time_variant(var, ops, solver, dec_all_coord, ic_pipeline_jit,
                                       lambda u0j: fit_ic_py(
                                           u0j, (("mean_t0", zmean_t[0]),
                                                 ("nearest_ic", Ztr[int(nearest_jit(u0j)), 0]))),
                                       jnp.asarray(ics[0][0]), float(u0_rms[0]),
                                       float(kap_te[0]), jnp.asarray(U_te[0, 0]))
    report["rom"] = results

    # ---------------- POD control (same solver) ----------------
    podres = {}
    for k in POD_KS:
        pdec = bc.PODDecoder(V[:, :k])
        for var in POD_VARIANTS:
            solver, colloc_name, objective = var.split(":")
            ops = build_ops(pdec, var, 0, eq_cache)
            if ops is None:
                continue
            runs = []
            t0 = time.time()
            for i in range(bc.N_TEST):
                z0, ic_rel, _ = bc.fit_ic(pdec, N, U_te[i, 0], {})
                r = bc.rollout(pdec, N, ops, z0, float(kap_te[i]), float(u0_rms[i]), U_true=U_te[i])
                r["ic_rel"], r["ic_init"] = ic_rel, "projection"
                del r["fields"]
                runs.append(r)
            s = summarize(runs); s["m"] = int(ops["m"]); s["secs"] = time.time() - t0
            s["M"] = ops.get("M"); s["alpha"] = ops.get("alpha")
            s["eq_info"] = ops.get("colloc_info")
            podres[f"k{k}:{var}"] = s
            log(f"  POD k={k:3d} {var:20s} traj rel mean {s['traj_rel_mean']:.3e} "
                f"(med {s['traj_rel_median']:.3e}) blowups {s['n_blowup']} "
                f"iters warm {s['iters_warm_mean']:.2f} step {s['step_time_ms_median']:.1f} ms")
            if DO_TIMING and solver == "lspg":
                Vk = jnp.asarray(V[:, :k], dtype=F64)
                tf = jnp.full((bc.NUM_STEPS,),
                              bc.GN_TOL * float(u0_rms[0])
                              * ops.get("tol_scale", np.sqrt(ops["m"])))
                kap0 = float(kap_te[0]); u0j0 = jnp.asarray(U_te[0, 0])
                z0 = Vk.T @ u0j0
                def pod_roll():
                    out = ops["rollout_jit"](z0, kap0, tf, bc.GN_BUDGET)
                    out[0].block_until_ready()
                med, ts = bc.time_fn(pod_roll, reps=TIME_REPS, warm=2)
                def pod_e2e():                       # project u0 -> step -> reconstruct
                    c0 = Vk.T @ u0j0
                    C = ops["rollout_jit"](c0, kap0, tf, bc.GN_BUDGET)[0]
                    F_ = jnp.concatenate([c0[None], C], axis=0) @ Vk.T
                    F_.block_until_ready()
                e2e, _ = bc.time_fn(pod_e2e, reps=TIME_REPS, warm=2)
                timing[f"pod_k{k}:{var}"] = dict(rollout_s_median=med, all=ts,
                                                 impl="device_scan", end_to_end_s=e2e)
    report["pod_rom"] = podres

    # ---------------- direct reduced POD-Galerkin (production linear ROM) -------------
    # The same-solver POD arm above isolates the REPRESENTATION (it runs the nonlinear
    # LM/Newton machinery on a linear subspace).  For a linear PDE a real POD ROM would
    # never do that: V^T A_kappa V is a k x k matrix, so the whole rollout is 50 k x k
    # solves.  This arm is that ROM -- the honest linear competitor on speed.
    _, impl_op = bc.hf.make_rollout(N)          # (rollout, implicit_op) from the FOM itself
    poddirect = {}
    for k in POD_KS:
        Vk = jnp.asarray(V[:, :k], dtype=F64)
        AV1 = jax.vmap(lambda v: impl_op(v, 1.0), in_axes=1, out_axes=1)(Vk)
        Lr = Vk.T @ ((Vk - AV1) / bc.DT)                 # V^T lap V (ghost-zero walls)

        def make_run(Vk=Vk, Lr=Lr, k=k):
            Ik = jnp.eye(k, dtype=F64)
            @jax.jit
            def run(u0, kap):
                Ar = Ik - bc.DT * kap * Lr
                c0 = Vk.T @ u0
                def body(c, _):
                    c2 = jnp.linalg.solve(Ar, c)
                    return c2, c2
                _, C = jax.lax.scan(body, c0, None, length=bc.NUM_STEPS)
                return jnp.concatenate([c0[None], C], axis=0) @ Vk.T
            return run
        run = make_run()
        errs = []
        for i in range(bc.N_TEST):
            F_ = np.asarray(run(jnp.asarray(U_te[i, 0]), float(kap_te[i])))
            errs.append(float(np.mean(np.linalg.norm(F_ - U_te[i], axis=1)
                                      / np.linalg.norm(U_te[i], axis=1))))
        errs = np.array(errs)
        poddirect[f"k{k}"] = dict(traj_rel_mean=float(errs.mean()),
                                  traj_rel_median=float(np.median(errs)),
                                  traj_rel_max=float(errs.max()))
        log(f"  POD-direct (reduced BE Galerkin) k={k:3d}: traj rel mean {errs.mean():.3e} "
            f"(med {np.median(errs):.3e})")
        if DO_TIMING:
            u0j0 = jnp.asarray(U_te[0, 0]); kap0 = float(kap_te[0])
            def pd_once():
                run(u0j0, kap0).block_until_ready()
            med_, ts_ = bc.time_fn(pd_once, reps=TIME_REPS, warm=2)
            timing[f"pod_direct_k{k}"] = dict(rollout_s_median=med_, all=ts_,
                                              end_to_end_s=med_, impl="device_scan_kxk")
    report["pod_direct"] = poddirect

    # ---------------- FOM timing (same jitted solver, batch 1) ----------------
    if DO_TIMING:
        # the cold-start solvers alone (variant-independent), for the cost breakdown
        u0j0 = jnp.asarray(U_te[0, 0])
        def ic_jit_once():
            ic_pipeline_jit(u0j0)[0].block_until_ready()
        timing["ic_solve_jit"] = dict(rollout_s_median=bc.time_fn(
            ic_jit_once, reps=TIME_REPS, warm=2)[0], impl="device_while_loop")
        def ic_py_once():
            fit_ic_py(u0j0, (("mean_t0", zmean_t[0]),
                             ("nearest_ic", Ztr[int(nearest_jit(u0j0)), 0])))
        timing["ic_solve_py"] = dict(rollout_s_median=bc.time_fn(
            ic_py_once, reps=3, warm=1)[0], impl="python_loop")
        Zt = jnp.asarray(np.stack([ics[0][0]] * T1))
        def dec_once():
            dec_all_coord(Zt).block_until_ready()
        timing["decode_all_slices"] = dict(rollout_s_median=bc.time_fn(
            dec_once, reps=TIME_REPS, warm=2)[0], impl="device_vmap")

        roll, _ = bc.make_fom(N)
        U0 = jnp.asarray(U_te[0, 0])[None]
        kap1 = jnp.asarray([kap_te[0]])
        def fom_once():
            roll(U0, kap1).block_until_ready()
        med, ts = bc.time_fn(fom_once, reps=TIME_REPS, warm=2)
        timing["fom_rollout"] = dict(rollout_s_median=med, all=ts)
        for kk, v in timing.items():
            if kk in ("fom_rollout", "ic_solve_jit", "ic_solve_py", "decode_all_slices"):
                continue
            v["speedup_vs_fom_rollout_only"] = med / v["rollout_s_median"]
            for lbl in ("end_to_end_jit_ic_s", "end_to_end_py_ic_s", "end_to_end_s"):
                if lbl in v:
                    v["speedup_" + lbl[:-2]] = med / v[lbl]
        log("  timing: FOM %.0f ms; " % (med * 1e3) + "  ".join(
            f"{k}={v['rollout_s_median']*1e3:.0f}ms" for k, v in timing.items()
            if k != "fom_rollout"))
    report["timing"] = timing

    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"N{N}_K{K}"
    with open(os.path.join(OUTDIR, f"hlat_rom_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"wrote hlat_rom_{tag}.json")


if __name__ == "__main__":
    main()
