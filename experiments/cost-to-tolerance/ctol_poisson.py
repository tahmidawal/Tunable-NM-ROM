"""POISSON-2D cost-to-tolerance SURFACE: the (k, N) grid, one GPU, one process.

Every cell of the whole grid is measured SEQUENTIALLY IN ONE JOB ON ONE GPU.
Cross-N and cross-k timings measured on different GPUs are not comparable and
have burned this project before.

WHAT ONE CELL IS
----------------
(method, N, k, M, m, tau) with method in {coord, pod}.  For that cell:

  * the NNLS-EQ quadrature is REFIT (it depends on the decoder -- hence on the
    method and on k -- and on the mesh and on M);
  * a tau-stopped LM (`ctol_tol.lm_tau_poisson`) is built ONCE and reused for
    all three tau (tau is a runtime argument, so the three tolerances share one
    compilation and one kernel);
  * for EVERY one of the N_TEST held-out sources: TIME_WARM warm-ups then
    TIME_REPS timed, `block_until_ready`-synchronised repetitions of THAT
    solve; the latent used for the ERROR is the one returned by the LAST TIMED
    REPETITION.  Cost and accuracy therefore come from the same run -- same
    init, same sources, same solver invocation.  (The solve is deterministic,
    so every repetition returns the same latent.)

TOLERANCE.  The solver stops on the relative reduction of the objective it is
actually minimising, ||r(z)|| <= tau * ||r(z_0)||, measured from the run's own
initial guess.  No oracle is involved, so the rule is deployable.  The achieved
discrete residual ||A u - f|| / ||f|| is reported for reference only -- at the
weak-form solution it is ~2e-1 while the field error is ~8e-3, which is exactly
why the weak form exists, so it cannot serve as a stopping test.

CENSORING.  A cell is censored when the solver stopped for any reason other
than reaching tau, for at least one source.  Censored cells are reported with
`censored=true` and `censored_frac`, together with the error they did reach.
They are never dropped.

COMPARABILITY OF THE POD ARM.  POD uses the SAME weak objective, the SAME test
modes, the SAME NNLS-EQ hyper-reduction (fitted on POD-output snapshots) and
the SAME LM solver.  Because a POD decoder is only defined at grid nodes, the
EQ candidate pool is the interior grid for BOTH arms; the headline Poisson
recipe used a meshfree pool for the coordinate decoder, so a `pool_control` arm
re-measures the coordinate cell at k=8 with that meshfree pool at every mesh.
The exact linear POD minimiser (one precomputed pseudo-inverse matvec,
`pod_direct`) is measured too and reported as supplementary -- it is the
strongest possible POD implementation and is deliberately not handicapped.

TIME ACCOUNTING.  `time_ms` in the shared Pareto schema is the END-TO-END
ONLINE cost: input preprocessing (Lambda^-1 Phi^T f, one matvec against a
per-mesh table built offline) + latent solve + decode of the interior field.
The FOM baseline (`fom_cg_s`) is the testbed's own jitted CG returning exactly
that interior field, so the two are like-for-like.  `time_ms_solve` isolates
the latent solve, which is the quantity the cost(k) question asks about.

Usage:
  KS=2,4,6,8,12,16,24,32 NS=32,64,128,256,512 TAUS=1e-1,1e-2,1e-3 \
  PKL_DIR=../ckpt M=64 MQ=256 M_BIG=256 K_BIG=32 python ctol_poisson.py <out.json>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(HERE, "deps", "poisson2d-rom-objective"),
           os.path.abspath(os.path.join(HERE, "..", "poisson2d-rom-objective"))):
    if os.path.isfile(os.path.join(_c, "pro_common.py")):
        sys.path.insert(0, _c)
        sys.path.insert(0, os.path.join(_c, "followup"))
        PRO_DIR = _c
        break
else:
    raise ImportError("pro_common.py not found (deps/ or sibling experiment dir)")
sys.path.insert(0, HERE)

import pro_common as pc                                    # noqa: E402
from pro_common import mp                                  # noqa: E402
from fu_eq import make_lm_jit, weak_source_projector       # noqa: E402
import ctol_eq                                             # noqa: E402
import ctol_tol                                            # noqa: E402

OUT = sys.argv[1]
KS = [int(v) for v in os.environ.get("KS", "2,4,6,8,12,16,24,32").split(",") if v]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256,512").split(",") if v]
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-1,1e-2,1e-3").split(",") if v]
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
M_BIG = int(os.environ.get("M_BIG", "256"))
K_BIG = int(os.environ.get("K_BIG", "32"))
MQ_SUPP = int(os.environ.get("MQ_SUPP", "1024"))       # supplementary m ~ 4M at k >= K_BIG
DO_SUPP = int(os.environ.get("DO_SUPP", "1"))
POOL_CONTROL = int(os.environ.get("POOL_CONTROL", "1"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
WARM = int(os.environ.get("TIME_WARM", "2"))
PKL_DIR = os.environ.get("PKL_DIR", "../ckpt")
FOM_RES_TOL = float(os.environ.get("FOM_RES_TOL", "1e-10"))
POD_KMAX = max(KS)
SEED = mp.SEED
TAU_OK = ctol_tol.POISSON_TAU_OK
# CONFIGS: a JSON list of {method, N, k, M, m, tau} cells to measure instead of the
# full grid.  Used by the single-GPU CONSOLIDATION job that re-times the per-(method,N)
# argmin configurations across ALL meshes in one process, which is the only timing
# source the cross-N scaling figure may use (the fanned-out per-(pde,N) panel jobs are
# same-architecture but not the same physical GPU).
CONFIGS = os.environ.get("CONFIGS", "")
ARM_TAG = os.environ.get("ARM_TAG", "consolidated")
DO_POD_DIRECT = int(os.environ.get("DO_POD_DIRECT", "1"))
NODE = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME", "local")


def build_plan():
    """plan[N][(k, M, m, arm_tag)][method] = [tau, ...]"""
    plan = {}
    if CONFIGS:
        for s_ in json.load(open(CONFIGS)):
            if s_.get("pde", "poisson2d") != "poisson2d":
                continue
            if s_["method"] not in ("coord", "pod"):
                continue
            key = (int(s_["k"]), int(s_["M"]), int(s_["m"]), s_.get("arm", ARM_TAG))
            d_ = plan.setdefault(int(s_["N"]), {}).setdefault(key, {})
            d_.setdefault(s_["method"], [])
            if float(s_["tau"]) not in d_[s_["method"]]:
                d_[s_["method"]].append(float(s_["tau"]))
        return plan
    for n_ in NS:
        arms = {}
        for k_ in KS:
            arms[(k_, M_BIG if k_ >= K_BIG else M_MODES, MQ, "primary")] = {
                "coord": list(TAUS), "pod": list(TAUS)}
        if DO_SUPP:
            for k_ in KS:
                if k_ >= K_BIG:
                    arms[(k_, M_BIG, MQ_SUPP, "supp_m4M")] = {
                        "coord": list(TAUS), "pod": list(TAUS)}
        plan[n_] = arms
    return plan


def log(*a):
    print(*a, flush=True)


def git_commit():
    if os.environ.get("CTOL_COMMIT"):
        return os.environ["CTOL_COMMIT"]
    for d in (HERE, PRO_DIR):
        try:
            return subprocess.check_output(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
    return "unknown"


def time_fn(fn, reps=None, warm=None):
    reps = TIME_REPS if reps is None else reps
    warm = WARM if warm is None else warm
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


def pod_basis_host(S, kmax):
    """Top-kmax POD basis of the snapshot ROWS S (n_s, n_i^2), f64 on the HOST
    via the SMALLER Gram (an all-slice device Gram OOMs an 80 GB A100 at
    N >= 128).  Same construction as `blat_common.pod_basis`."""
    S = np.asarray(S, dtype=np.float64)
    G = S @ S.T
    ev, EV = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1]
    ev, EV = ev[o], EV[:, o]
    sv = np.sqrt(np.maximum(ev[:kmax], 0.0))
    V = (S.T @ EV[:, :kmax]) / np.maximum(sv, 1e-300)
    dev = float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))
    return V, sv, dev


def fom_solve(n, Fs):
    """The testbed's own jitted CG at the testbed's tolerance -- the function
    that generated the truth.  Aborts if the converged residual is too large."""
    op = lambda v: mp.neg_lap_interior(v, n)
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL,
                                                             maxiter=mp.CG_MAXITER)[0])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
    res = float(np.max([np.linalg.norm(np.asarray(op(jnp.asarray(U_int[i]))) - Fs[i])
                        / np.linalg.norm(Fs[i]) for i in range(Fs.shape[0])]))
    if not np.isfinite(res) or res > FOM_RES_TOL:
        raise SystemExit(f"N={n}: FOM CG rel residual {res:.2e} > {FOM_RES_TOL:.0e} -- "
                         f"refusing to time an unconverged baseline")
    return solve_one, U_int, res


def timed_sweep(lm, z0, f_ms_j, tau, err_fn, fd_fn):
    """Per source: WARM warm-ups, TIME_REPS timed reps, error from the LAST
    TIMED repetition.  Returns the per-source lists."""
    t, err, jac, att, reason, fd, red = [], [], [], [], [], [], []
    for fmi in f_ms_j:
        for _ in range(WARM):
            lm(z0, fmi, tau)[0].block_until_ready()
        ts = []
        for _ in range(TIME_REPS):
            t0 = time.perf_counter()
            out = lm(z0, fmi, tau)
            out[0].block_until_ready()
            ts.append(time.perf_counter() - t0)
        z_i, val, v0, nJ, acc, n_att, rsn = out          # LAST TIMED REPETITION
        t.append(float(np.median(ts)))
        err.append(err_fn(z_i, len(err)))
        fd.append(fd_fn(z_i, len(fd)))
        jac.append(int(nJ)); att.append(int(n_att)); reason.append(int(rsn))
        red.append(float(val) / max(float(v0), 1e-300))   # achieved ||r||/||r(z0)||
    return t, err, jac, att, reason, fd, red


# --------------------------------------------------------------------------
def main():
    dev = jax.devices()[0]
    gpu_name = getattr(dev, "device_kind", str(dev))
    log(f"jax_backend={dev.platform} device={dev} gpu={gpu_name} "
        f"x64={jax.config.jax_enable_x64} KS={KS} NS={NS} TAUS={TAUS} "
        f"M={M_MODES} m={MQ} reps={TIME_REPS} warm={WARM}")
    commit = git_commit()
    plan = build_plan()
    ks_used = sorted({k_ for arms in plan.values() for (k_, _M, _m, _t) in arms})
    log(f"  plan: {len(plan)} mesh(es) {sorted(plan)}, k values {ks_used}, "
        f"{sum(len(v) for a_ in plan.values() for v in a_.values())} (arm, method) cells"
        + (f"  [CONFIGS={CONFIGS}]" if CONFIGS else ""))

    # ---------------- checkpoints (the k ladder, all trained at N=64) --------
    ck = {}
    for k in ks_used:
        p = os.path.join(PKL_DIR, f"autodec_K{k}_N64_hbc_stages.pkl")
        d, cfg, stages, Z_tr, hb = pc.load_pkl(p)
        if cfg["K_LAT"] != k:
            raise SystemExit(f"{p}: K_LAT {cfg['K_LAT']} != {k}")
        ck[k] = dict(cfg=cfg, dec=pc.make_decoder(stages[:1], hard_bc=bool(hb)),
                     Z_tr=np.asarray(Z_tr), hard_bc=hb, path=os.path.basename(p))
        log(f"  ckpt k={k:2d}: {os.path.basename(p)} hard_bc={hb} "
            f"train_seed={cfg.get('train_seed')} latent_rms="
            f"{float(np.sqrt(np.mean(np.asarray(Z_tr)**2))):.3f}")

    cx, cy, w, a, _z = mp.sample_params()
    N_TRAIN = mp.N_TRAIN
    report = dict(
        config=dict(pde="poisson2d", ks=KS, ns=NS, taus=TAUS, M=M_MODES, m=MQ,
                    M_big=M_BIG, k_big=K_BIG, m_supp=MQ_SUPP, do_supp=DO_SUPP,
                    n_test=N_TEST, gn_iters=GN_ITERS, time_reps=TIME_REPS,
                    time_warm=WARM, seed=SEED, cg_tol=mp.CG_TOL,
                    cand_cap=ctol_eq.CAND_CAP, eq_snaps=ctol_eq.EQ_SNAPS,
                    eq_perturb=ctol_eq.EQ_PERTURB, eq_rows=ctol_eq.EQ_ROWS,
                    eq_seed=ctol_eq.EQ_SEED, eq_pool="grid",
                    eq_perturb_scale="0.05 * rms(Z_snap) (relative, symmetric across arms)",
                    init="mean training latent / mean training POD coefficient",
                    objective="weak_a1_M{M}: || Lambda^-1 Phi_M^T (A u - f) ||, "
                              "quadratured on m NNLS-EQ nodes",
                    stopping="||r(z)|| <= tau * ||r(z0)||",
                    x64=True,
                    matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                    backend=dev.platform, device=str(dev), gpu=gpu_name, commit=commit,
                    slurm_job=os.environ.get("SLURM_JOB_ID"),
                    time_ms_definition="preprocess + latent solve + interior-field decode",
                    node=NODE, configs=CONFIGS or None, arm_tag=ARM_TAG,
                    ns_measured=sorted(plan), ks_measured=ks_used,
                    ckpts={k: ck[k]["path"] for k in ks_used}),
        rows=[], fom=[], supplementary=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- the mesh ladder ---------------------------------------
    for n in sorted(plan):
        t_mesh = time.time()
        grid = pc.Grid(n)
        n_i, n_i2 = grid.n_i, grid.n_i ** 2
        int_idx = grid.ix_full * n + grid.iy_full
        Fs = np.stack([mp.source_interior(n, cx[N_TRAIN + i], cy[N_TRAIN + i],
                                          w[N_TRAIN + i], a[N_TRAIN + i])
                       for i in range(N_TEST)])
        solve_one, U_int, fom_res = fom_solve(n, Fs)
        F0 = jnp.asarray(Fs[0])
        fom_med, fom_all = time_fn(lambda: solve_one(F0).block_until_ready())
        tn = np.array([np.linalg.norm(U_int[i]) for i in range(N_TEST)])
        fn_ = np.array([np.linalg.norm(Fs[i]) for i in range(N_TEST)])
        report["fom"].append(dict(pde="poisson2d", method="fom", N=n, n_dof=n_i2,
                                  fom_cg_s=fom_med, all=fom_all,
                                  fom_max_rel_residual=fom_res, cg_tol=mp.CG_TOL,
                                  n_sources=N_TEST, gpu=gpu_name, node=NODE,
                                  slurm_job=os.environ.get("SLURM_JOB_ID"),
                                  jax_backend=dev.platform, commit=commit))
        log(f"== N={n:4d}  FOM CG {fom_med*1e3:8.2f} ms  (residual {fom_res:.1e})")

        # POD basis at THIS mesh from the SAME 512 training sources
        U_all = np.asarray(mp.build_snapshots(n)[0])
        X_tr = U_all[:N_TRAIN][:, int_idx]
        del U_all
        Vfull, sv, orth = pod_basis_host(X_tr, POD_KMAX)
        c_mean_full = X_tr.mean(0) @ Vfull
        log(f"   POD basis: {X_tr.shape[0]} snapshots, orthonormality dev {orth:.2e}, "
            f"sv0 {sv[0]:.3e} sv[{POD_KMAX-1}] {sv[-1]:.3e}")

        pre_cache, phi_cache = {}, {}

        def preprocess(M):
            if M not in pre_cache:
                spec = dict(kind="weak", alpha=1.0, M=M)
                ap, build_s = weak_source_projector(grid, spec, "grid")
                chk = float(jnp.max(jnp.abs(ap(jnp.asarray(Fs[0]))
                                            - pc.weak_source_term(grid, spec, "grid", Fs[0]))))
                med, _ = time_fn(lambda: ap(jnp.asarray(Fs[0])).block_until_ready())
                pre_cache[M] = (ap, med, build_s, chk)
                log(f"   preprocess M={M}: {med*1e3:.3f} ms/query "
                    f"(offline table {build_s:.1f} s, vs reference maxabs {chk:.1e})")
            return pre_cache[M]

        def phi_full(M):
            if M not in phi_cache:
                mask = np.asarray(grid.mode_mask(M)).astype(bool)
                I, Jm = np.nonzero(mask)
                S = np.asarray(grid.S)
                phi_cache[M] = S[grid.ix_full - 1][:, I] * S[grid.iy_full - 1][:, Jm]
            return phi_cache[M]

        cand_pos = ctol_eq.candidate_pool(n_i2)
        cand_np = np.asarray(grid.coords_int)[cand_pos]
        cand_j = jnp.asarray(cand_np)
        log(f"   EQ candidate pool: {cand_pos.size} of {n_i2} interior nodes "
            f"(cap {ctol_eq.CAND_CAP})")

        for (k, M, m, arm_tag), methods in sorted(plan[n].items()):
            spec = dict(kind="weak", alpha=1.0, M=M)
            ap, pre_med, pre_build_s, pre_chk = preprocess(M)
            Phi_f = phi_full(M)
            f_ms_j = [jnp.asarray(np.asarray(ap(jnp.asarray(Fs[i])))) for i in range(N_TEST)]

            for method in ("coord", "pod"):
                if method not in methods:
                    continue
                t_cell = time.time()
                if method == "coord":
                    dec_k = ck[k]["dec"]
                    Z_snap = ck[k]["Z_tr"]
                    z0 = jnp.asarray(Z_snap.mean(0))
                    u_cand = jax.jit(lambda z, _d=dec_k, _c=cand_j: _d(z, _c))
                    u_full = jax.jit(lambda z, _d=dec_k: _d(z, grid.coords_int))
                    dec_pts_fn = dec_k
                else:
                    Vk = np.ascontiguousarray(Vfull[:, :k])
                    Vk_j = jnp.asarray(Vk)
                    Vc_j = jnp.asarray(Vk[cand_pos])
                    Z_snap = X_tr @ Vk
                    z0 = jnp.asarray(c_mean_full[:k])
                    u_cand = jax.jit(lambda c, _V=Vc_j: _V @ c)
                    u_full = jax.jit(lambda c, _V=Vk_j: _V @ c)
                    dec_pts_fn = None                     # bound after the node set is known

                keep, wq, eq_info = ctol_eq.eq_fit_poisson(
                    u_cand, u_full, Phi_f[cand_pos], Phi_f, Z_snap, k, m,
                    f"poisson {method} N={n} k={k} M={M} m={m}", pc.nnls_capped)
                node_pos = cand_pos[keep]
                pts_np = np.asarray(grid.coords_int)[node_pos]
                PhiT, Wl = pc.colloc_mode_table(grid, spec, "grid", pts_np)
                n_modes = int(PhiT.shape[0])
                if n_modes <= k:
                    log(f"   WARNING N={n} k={k} M={M}: {n_modes} retained modes <= k -- the "
                        f"weak system is square/underdetermined (M > k is an operating rule)")
                if method == "pod":
                    Vq_j = jnp.asarray(Vk[node_pos])
                    dec_pts_fn = lambda z, xy, _V=Vq_j: _V @ z

                lm, _rn = ctol_tol.lm_tau_poisson(dec_pts_fn, k, pts_np, wq, PhiT, Wl, GN_ITERS)
                d_agree = None
                if k == ks_used[0]:  # the tau solver must reproduce the reference at tau = 0
                    lm_ref = make_lm_jit(dec_pts_fn, k, pts_np, wq, PhiT, Wl, GN_ITERS, 0.0)
                    d_agree = ctol_tol.check_tau_agreement(
                        lm, lm_ref, (z0, f_ms_j[0], 0.0), (z0, f_ms_j[0]),
                        f"poisson {method} N={n} k={k}")
                    log(f"   tau-solver vs reference solver at tau=0: rel |dz| {d_agree:.2e}")

                z_probe = lm(z0, f_ms_j[0], methods[method][0])[0]
                dec_med, _ = time_fn(lambda: u_full(z_probe).block_until_ready())

                def err_fn(z_i, i, _u=u_full):
                    ui = np.asarray(_u(z_i)).reshape(n_i, n_i)
                    return float(np.linalg.norm(ui - U_int[i]) / tn[i])

                def fd_fn(z_i, i, _u=u_full):
                    ui = jnp.asarray(np.asarray(_u(z_i)).reshape(n_i, n_i))
                    return float(np.linalg.norm(np.asarray(grid.op(ui)) - Fs[i]) / fn_[i])

                for tau in methods[method]:
                    per_t, per_err, per_jac, per_att, per_reason, per_fd, per_red = timed_sweep(
                        lm, z0, f_ms_j, tau, err_fn, fd_fn)
                    solve_ms = float(np.median(per_t)) * 1e3
                    e2e_ms = solve_ms + pre_med * 1e3 + dec_med * 1e3
                    cens = [r not in TAU_OK for r in per_reason]
                    row = dict(pde="poisson2d", method=method, N=n, k=k, M=M, m=int(len(wq)),
                               tau=tau, time_ms=e2e_ms, err_rel_l2=float(np.mean(per_err)),
                               iters=float(np.mean(per_att)), jac_evals=float(np.mean(per_jac)),
                               censored=bool(any(cens)), n_sources=N_TEST, seed=SEED,
                               gpu=gpu_name, jax_backend=dev.platform, commit=commit,
                               node=NODE, slurm_job=os.environ.get("SLURM_JOB_ID"),
                               # ---- beyond the shared schema: diagnostics / provenance
                               arm=arm_tag, time_ms_solve=solve_ms,
                               time_ms_pre=pre_med * 1e3, time_ms_decode=dec_med * 1e3,
                               time_ms_solve_per_source=[float(v) * 1e3 for v in per_t],
                               err_rel_l2_median=float(np.median(per_err)),
                               err_rel_l2_max=float(np.max(per_err)),
                               err_rel_l2_per_source=[float(v) for v in per_err],
                               fd_residual_rel_mean=float(np.mean(per_fd)),
                               fd_residual_rel_max=float(np.max(per_fd)),
                               censored_frac=float(np.mean(cens)),
                               rel_reduction_mean=float(np.mean(per_red)),
                               rel_reduction_max=float(np.max(per_red)),
                               reasons={str(r): per_reason.count(r) for r in set(per_reason)},
                               n_modes=n_modes, n_modes_le_k=bool(n_modes <= k),
                               eq_rel_fit=eq_info["rel_fit"], eq_info=eq_info,
                               ms_per_jac=solve_ms / max(float(np.mean(per_jac)), 1.0),
                               fom_cg_ms=fom_med * 1e3, speedup_e2e=fom_med * 1e3 / e2e_ms,
                               lm_agreement_rel_dz=d_agree)
                    report["rows"].append(row)
                    log(f"   {method:5s} N={n:4d} k={k:2d} M={M:3d} m={row['m']:4d} "
                        f"tau={tau:.0e}  solve {solve_ms:7.2f} ms  e2e {e2e_ms:7.2f} ms  "
                        f"jac {row['jac_evals']:5.1f}  err {row['err_rel_l2']:.3e}  "
                        f"fd {row['fd_residual_rel_mean']:.2e}  "
                        f"cens {row['censored_frac']*100:3.0f}%")
                    save()

                # supplementary: the exact linear POD minimiser (one pinv matvec)
                if method == "pod" and DO_POD_DIRECT:
                    A_ = (np.asarray(Wl)[:, None]
                          * (np.asarray(PhiT) * np.asarray(wq)[None, :])) @ np.asarray(Vk[node_pos])
                    pinv = jnp.asarray(np.linalg.pinv(A_))
                    apply_ = jax.jit(lambda b, _P=pinv: _P @ b)
                    med, _ = time_fn(lambda: apply_(f_ms_j[0]).block_until_ready())
                    errs = []
                    for i in range(N_TEST):
                        c = apply_(f_ms_j[i])
                        ui = np.asarray(u_full(c)).reshape(n_i, n_i)
                        errs.append(float(np.linalg.norm(ui - U_int[i]) / tn[i]))
                    report["supplementary"].append(dict(
                        pde="poisson2d", method="pod_direct", N=n, k=k, M=M,
                        m=int(len(wq)), tau=None, arm=arm_tag,
                        time_ms=med * 1e3 + pre_med * 1e3 + dec_med * 1e3,
                        time_ms_solve=med * 1e3, time_ms_pre=pre_med * 1e3,
                        time_ms_decode=dec_med * 1e3, err_rel_l2=float(np.mean(errs)),
                        iters=1.0, jac_evals=1.0, censored=False, n_sources=N_TEST,
                        seed=SEED, gpu=gpu_name, jax_backend=dev.platform, commit=commit,
                        cond=float(np.linalg.cond(A_)), rank=int(np.linalg.matrix_rank(A_)),
                        square_or_underdetermined=bool(A_.shape[0] <= k)))
                    log(f"   pod_direct N={n:4d} k={k:2d} solve {med*1e3:.4f} ms "
                        f"err {np.mean(errs):.3e} cond {np.linalg.cond(A_):.1e}")
                    save()
                log(f"   [cell {method} N={n} k={k} M={M} m={m}: {time.time()-t_cell:.0f}s]")

        # ---- control: the headline MESHFREE EQ pool, coordinate arm at k=8 ----
        if POOL_CONTROL and 8 in ks_used:
            k, M, m = 8, M_MODES, MQ
            spec = dict(kind="weak", alpha=1.0, M=M)
            dec_k = ck[k]["dec"]
            cand_off = np.random.default_rng(SEED + 12345).uniform(
                0.0, 1.0, size=(ctol_eq.CAND_CAP, 2))
            spec0 = dict(kind="weak", alpha=0.0, M=M)
            PhiT_c, _ = pc.colloc_mode_table(grid, spec0, "offgrid", cand_off)
            PhiT_fo, _ = pc.colloc_mode_table(grid, spec0, "offgrid", np.asarray(grid.coords_int))
            keep, wq, eq_info = ctol_eq.eq_fit_poisson(
                jax.jit(lambda z, _d=dec_k, _c=jnp.asarray(cand_off): _d(z, _c)),
                jax.jit(lambda z, _d=dec_k: _d(z, grid.coords_int)),
                np.asarray(PhiT_c).T, np.asarray(PhiT_fo).T * grid.dx ** 2,
                ck[k]["Z_tr"], k, m, f"poisson coord MESHFREE-POOL N={n} k={k}", pc.nnls_capped)
            pts_np = cand_off[keep]
            PhiT, Wl = pc.colloc_mode_table(grid, spec, "offgrid", pts_np)
            ap_o, _ = weak_source_projector(grid, spec, "offgrid")
            fmo = [jnp.asarray(np.asarray(ap_o(jnp.asarray(Fs[i])))) for i in range(N_TEST)]
            lm, _rn = ctol_tol.lm_tau_poisson(dec_k, k, pts_np, wq, PhiT, Wl, GN_ITERS)
            z0 = jnp.asarray(ck[k]["Z_tr"].mean(0))
            u_full_c = jax.jit(lambda z, _d=dec_k: _d(z, grid.coords_int))

            def err_c(z_i, i):
                ui = np.asarray(u_full_c(z_i)).reshape(n_i, n_i)
                return float(np.linalg.norm(ui - U_int[i]) / tn[i])

            for tau in sorted({t_ for arms in plan.values() for v in arms.values()
                               for tt in v.values() for t_ in tt}, reverse=True):
                per_t, per_err, per_jac, per_att, per_reason, _fd, per_red = timed_sweep(
                    lm, z0, fmo, tau, err_c, lambda z, i: float("nan"))
                report["supplementary"].append(dict(
                    pde="poisson2d", method="coord_meshfree_pool", N=n, k=k, M=M,
                    m=int(len(wq)), tau=tau, arm="pool_control",
                    time_ms_solve=float(np.median(per_t)) * 1e3,
                    err_rel_l2=float(np.mean(per_err)), jac_evals=float(np.mean(per_jac)),
                    iters=float(np.mean(per_att)),
                    censored=bool(any(r not in TAU_OK for r in per_reason)),
                    rel_reduction_mean=float(np.mean(per_red)),
                    n_sources=N_TEST, seed=SEED, gpu=gpu_name, jax_backend=dev.platform,
                    commit=commit, eq_rel_fit=eq_info["rel_fit"]))
                log(f"   [pool control] meshfree N={n} k=8 tau={tau:.0e} "
                    f"solve {np.median(per_t)*1e3:.2f} ms err {np.mean(per_err):.3e}")
            save()
        log(f"== N={n} done [{time.time()-t_mesh:.0f}s]")

    report["complete"] = True
    save()
    log("DONE")


if __name__ == "__main__":
    main()
