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
# m at k >= K_BIG.  The brief said "M=256 whenever k >= 32" while holding m=256, which
# lands exactly on the m = M corner and violates this project's own operating rule that
# m ~ 4M is the knee (HANDOFF.md rule 4).  The m ~ 4M setting is therefore PRIMARY here
# and the m = MQ run is kept as a labelled artefact of the original spec.
MQ_4M = int(os.environ.get("MQ_4M", str(4 * int(os.environ.get("M_BIG", "256")))))
FOM_LADDER = sorted({float(v) for v in os.environ.get(
    "FOM_LADDER", "3e-1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-8,1e-10,1e-13").split(",") if v}
    | {mp.CG_TOL}, reverse=True)     # the archived tolerance is always a rung
FOM_ONLY = int(os.environ.get("FOM_ONLY", "0"))
DO_CEILING = int(os.environ.get("DO_CEILING", "1"))
BURN_IN_S = float(os.environ.get("BURN_IN_S", "1.5"))
DO_SUPP = int(os.environ.get("DO_SUPP", "1"))
POOL_CONTROL = int(os.environ.get("POOL_CONTROL", "1"))
CAP_CONTROL = int(os.environ.get("CAP_CONTROL", "1"))
# the uncapped-pool control is only affordable while the ECSW design matrix fits:
# rows x n_cand doubles are ~2 GB at 16k candidates and M=64
CAP_CONTROL_MAX = int(os.environ.get("CAP_CONTROL_MAX", "16384"))
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
            # PRIMARY: m ~ 4M whenever M is raised, per the project's own EQ operating rule
            m_ = MQ_4M if k_ >= K_BIG else MQ
            arms[(k_, M_BIG if k_ >= K_BIG else M_MODES, m_, "primary")] = {
                "coord": list(TAUS), "pod": list(TAUS)}
        if DO_SUPP:
            for k_ in KS:
                if k_ >= K_BIG:
                    # the ORIGINAL spec's m = M = 256 corner, kept as a labelled artefact
                    arms[(k_, M_BIG, MQ, "artefact_m_eq_M")] = {
                        "coord": list(TAUS), "pod": list(TAUS)}
            if 8 in KS:
                # ISOLATOR at FIXED k = 8 of the M jump the SPEC makes at k >= K_BIG
                # (M: 64 -> 256 with m held at MQ).  m is held at MQ deliberately:
                # that is exactly the change the primary grid makes, and the ECSW
                # refit cost grows as m^3 -- m=1024 at M=256 is ~60 min per fit,
                # more than ten times the whole primary grid, for a supplementary arm.
                arms[(8, M_BIG, MQ, "supp_M256")] = {
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


def weak_source_projector_sel(grid, spec):
    """The DEPLOYABLE input preprocessing for the weak form on grid modes.

    `fu_eq.weak_source_projector` splits the mode table into an offline and an
    online half, but for pts_kind='grid' its online half is the FULL dense sine
    transform `S^T f S` (2 N^3 flops) followed by a gather of the M' retained
    modes.  Only the M' retained rows are ever used, so the deployable operation
    is the selected-mode contraction, `Phi_sel (M' x n_i^2)` times the flattened
    source: O(M' N^2).  Charging the full transform to the online path
    overcharges the ROM at fine meshes (it is the whole cost at N=512).

    Returns (apply, build_secs); `apply` is numerically identical to
    `pro_common.weak_source_term(grid, spec, 'grid', f)` -- asserted by the
    caller."""
    t0 = time.time()
    alpha, M = spec["alpha"], spec["M"]
    mask = np.asarray(grid.mode_mask(M)).astype(bool)
    I, Jm = np.nonzero(mask)
    lam = np.asarray(grid.lam)[I, Jm]
    S_ = np.asarray(grid.S)
    # Phi_sel[q, p] = S[px_p, I_q] * S[py_p, Jm_q]  over the interior nodes p
    Phi_sel = jnp.asarray((S_[grid.ix_full - 1][:, I] * S_[grid.iy_full - 1][:, Jm]).T)
    wgt = jnp.asarray(lam ** (-alpha))

    def apply(f_int2d):
        return (Phi_sel @ jnp.asarray(f_int2d).reshape(-1)) * wgt
    return jax.jit(apply), time.time() - t0


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


def make_pipeline(ap, lm, u_full, z0):
    """The whole ONLINE path as ONE jitted, once-synchronised function:
    preprocess the query source -> tau-stopped latent solve -> decode the
    interior field.  Timing this (rather than summing three separately measured
    medians) is what makes `time_ms` and `err_rel_l2` literally the same run."""
    def pipe(F, tau):
        f_m = ap(F)
        z, val, v0, nJ, acc, att, rsn = lm(z0, f_m, tau)
        return u_full(z), z, val, v0, nJ, acc, att, rsn
    return jax.jit(pipe)


def fom_ladder(n, Fs, U_ref, tn, fn_, time_fn_):
    """ISO-ACCURACY FOM baselines: the testbed's own CG at a LADDER of tolerances.

    The archived baseline is `cg(..., tol=mp.CG_TOL)` with CG_TOL = 1e-13 -- the
    tolerance that MANUFACTURED the truth data, not one any deployment would ask
    for.  A "speedup vs the FOM" against that number compares a 1e-2-accurate ROM
    with a 1e-13-accurate solve, which is two different questions, and the
    over-convergence factor is itself mesh dependent.  Every rung below is the
    SAME jitted CG, graded against the tightest rung and timed with the ROM's own
    protocol (all sources, warm-ups, median), so an iso-error comparison is
    possible.  The tightest rung is kept as the labelled 'exact' reference."""
    op = lambda v: mp.neg_lap_interior(v, n)
    out = []
    ctol_tol.burn_in(BURN_IN_S)
    for tol in sorted(FOM_LADDER, reverse=True):
        solve = jax.jit(lambda F, _t=tol: jax.scipy.sparse.linalg.cg(
            op, F, tol=_t, maxiter=mp.CG_MAXITER)[0])
        U = np.asarray(jax.lax.map(solve, jnp.asarray(Fs)))
        errs = [float(np.linalg.norm(U[i] - U_ref[i]) / tn[i]) for i in range(len(Fs))]
        res = [float(np.linalg.norm(np.asarray(op(jnp.asarray(U[i]))) - Fs[i]) / fn_[i])
               for i in range(len(Fs))]
        ts = []
        for i in range(len(Fs)):
            Fi = jnp.asarray(Fs[i])
            med_i, _ = time_fn_(lambda _F=Fi, _s=solve: _s(_F).block_until_ready())
            ts.append(med_i)
        out.append(dict(fom_tol=tol, fom_cg_s=float(np.median(ts)),
                        fom_cg_s_mean=float(np.mean(ts)),
                        per_source_s=[float(v) for v in ts],
                        err_rel_l2=float(np.mean(errs)),
                        err_rel_l2_max=float(np.max(errs)),
                        achieved_rel_residual=float(np.max(res))))
        log(f"   FOM CG tol={tol:.0e}: {np.median(ts)*1e3:8.3f} ms  err {np.mean(errs):.3e}"
            f"  achieved residual {np.max(res):.2e}")
    return out


def oracle_ceiling(dec, grid, Z_tr, U_int, tn, n_i, budget):
    """The DECODER'S OWN ceiling at this k: LM on the data misfit to the held-out
    field (an oracle -- the ROM never sees it).  Reported next to the ROM error so
    that non-monotone accuracy in k is a MEASURED property of the separately
    trained checkpoints rather than a caveat about them."""
    coords_int = grid.coords_int
    f = lambda z, u: dec(z, coords_int) - u
    rJ = jax.jit(lambda z, u: (f(z, u), jax.jacfwd(f)(z, u)))
    rn = jax.jit(lambda z, u: jnp.linalg.norm(f(z, u)))
    z0 = jnp.asarray(Z_tr.mean(0))
    rels = []
    for i in range(U_int.shape[0]):
        u = jnp.asarray(U_int[i].reshape(-1))
        z, r, _ = pc.lm_solve(lambda zz: rJ(zz, u), lambda zz: rn(zz, u), z0, budget)
        rels.append(float(r) / tn[i])
    return float(np.mean(rels)), float(np.max(rels))


def timed_sweep(pipeline, lm, z0, Fs_j, f_ms_j, tau, ref_int, tn, Fs, fn_, grid, n_i):
    """Per source: WARM warm-ups, then TIME_REPS timed, block_until_ready-
    synchronised repetitions of the FULL PIPELINE; the field that is GRADED is
    the one returned by the LAST TIMED repetition.  ONE timed quantity per cell,
    so `time_ms` and `err_rel_l2` are literally the same invocation.  The
    preprocess and decode stages are measured separately (they are
    value-independent and k-independent) and subtracted to give the
    latent-solve component, which is reported as DERIVED."""
    t_pipe = []
    err, jac, att_l, reason, fd, red = [], [], [], [], [], []
    ctol_tol.burn_in(BURN_IN_S)      # the EQ fit just spent minutes on the HOST
    for i, Fi in enumerate(Fs_j):
        for _ in range(WARM):
            pipeline(Fi, tau)[0].block_until_ready()
        ts = []
        for _ in range(TIME_REPS):
            t0 = time.perf_counter()
            out = pipeline(Fi, tau)
            out[0].block_until_ready()
            ts.append(time.perf_counter() - t0)
        u, z_i, val, v0, nJ, acc, n_att, rsn = out      # LAST TIMED REPETITION
        t_pipe.append(float(np.median(ts)))
        ui = np.asarray(u).reshape(n_i, n_i)
        err.append(float(np.linalg.norm(ui - ref_int[i]) / tn[i]))
        fd.append(float(np.linalg.norm(np.asarray(grid.op(jnp.asarray(ui))) - Fs[i]) / fn_[i]))
        jac.append(int(nJ)); att_l.append(int(n_att)); reason.append(int(rsn))
        red.append(float(val) / max(float(v0), 1e-300))     # achieved ||r||/||r(z0)||
    return t_pipe, err, jac, att_l, reason, fd, red


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
                    M_big=M_BIG, k_big=K_BIG, m_4M=MQ_4M, do_supp=DO_SUPP,
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
                    fom_ladder=FOM_LADDER, fom_only=FOM_ONLY, mq_4m=MQ_4M,
                    fom_baseline_note="mp.CG_TOL=1e-13 is the tolerance that MANUFACTURED "
                                      "the truth data; report['fom'] carries the full "
                                      "iso-accuracy ladder and every ROM row carries "
                                      "fom_iso_accuracy_ms",
                    node=NODE, configs=CONFIGS or None, arm_tag=ARM_TAG,
                    source_sha256=ctol_tol.sha256_of(
                        ctol_tol.module_files([ctol_tol, ctol_eq, pc, mp])
                        + [os.path.join(HERE, "ctol_poisson.py"),
                           os.path.join(HERE, "ctol_tables.py")]
                        + [os.path.join(PRO_DIR, "followup", "fu_eq.py")]),
                    ckpt_sha256=ctol_tol.sha256_of(
                        [os.path.join(PKL_DIR, ck[k]["path"]) for k in ks_used]),
                    src_commits=os.environ.get("CTOL_SRC_COMMITS"),
                    manifest_sha256=ctol_tol.sha256_of(
                        [os.path.abspath(os.path.join(HERE, "..", "MANIFEST.sha256"))]),
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
        # the FOM is timed on EVERY test source with the same warm-up / median-of-
        # TIME_REPS protocol and summarised with the SAME across-source statistic
        # as the ROM (CG iteration counts are source dependent, so timing source 0
        # alone would compare different workloads)
        fom_per = []
        for i in range(N_TEST):
            Fi = jnp.asarray(Fs[i])
            med_i, _ = time_fn(lambda _F=Fi: solve_one(_F).block_until_ready())
            fom_per.append(med_i)
        fom_med, fom_all = float(np.median(fom_per)), fom_per
        tn = np.array([np.linalg.norm(U_int[i]) for i in range(N_TEST)])
        fn_ = np.array([np.linalg.norm(Fs[i]) for i in range(N_TEST)])
        # ISO-ACCURACY FOM ladder.  The archived baseline (tol = CG_TOL = 1e-13) is the
        # truth-manufacturing tolerance, so it is kept only as the labelled `exact` rung.
        ladder = fom_ladder(n, Fs, U_int, tn, fn_, time_fn)
        for rung in ladder:
            report["fom"].append(dict(
                pde="poisson2d", method="fom", N=n, n_dof=n_i2,
                fom_rule=f"jax.scipy.sparse.linalg.cg(tol={rung['fom_tol']:.0e}, "
                         f"maxiter={mp.CG_MAXITER})",
                exact_reference=bool(rung["fom_tol"] == min(FOM_LADDER)),
                truth_manufacturing_tol=mp.CG_TOL,
                n_sources=N_TEST, gpu=gpu_name, node=NODE,
                slurm_job=os.environ.get("SLURM_JOB_ID"),
                jax_backend=dev.platform, commit=commit, **rung))
        fom_med = next(r["fom_cg_s"] for r in ladder if r["fom_tol"] == min(FOM_LADDER))
        # ---- shared vocabulary with the rom-warmstart-fom cell, so the two cells'
        # ---- JSONs compose and cannot publish two different Poisson denominators.
        _testbed = next(r for r in ladder if r["fom_tol"] == mp.CG_TOL)
        # MATCHED tolerance: chosen by what the archived baseline ACTUALLY ACHIEVED
        # (its true recomputed residual), not by its nominal tol.
        # the CHEAPEST rung that actually delivers at least the accuracy the archived
        # baseline delivered -- like-for-like on what was achieved, not on what was asked
        _match = min([r for r in ladder
                      if r["achieved_rel_residual"] <= _testbed["achieved_rel_residual"]],
                     key=lambda r: r["fom_cg_s"], default=_testbed)
        report["fom_baseline"] = report.get("fom_baseline", [])
        report["fom_baseline"].append(dict(
            N=n,
            t_fom_testbed_ms=_testbed["fom_cg_s"] * 1e3,
            fom_testbed_cg_tol=mp.CG_TOL,
            fom_testbed_true_rel_res=_testbed["achieved_rel_residual"],
            t_fom_baseline_native_ms=_match["fom_cg_s"] * 1e3,
            fom_baseline_tol=_match["fom_tol"],
            fom_baseline_true_rel_res=_match["achieved_rel_residual"],
            overconvergence_factor=_testbed["fom_cg_s"] / _match["fom_cg_s"],
            solver="jax.scipy.sparse.linalg.cg from x0=0, both rungs (like-for-like)",
            note="matched tolerance chosen by the ACHIEVED true residual of the "
                 "archived 1e-13 baseline, per the rom-warmstart-fom definition"))
        log(f"   overconvergence factor at N={n}: "
            f"{_testbed['fom_cg_s']/_match['fom_cg_s']:.2f}x "
            f"(testbed 1e-13 achieved {_testbed['achieved_rel_residual']:.1e}; "
            f"matched rung tol={_match['fom_tol']:.0e})")
        log(f"== N={n:4d}  FOM CG @1e-13 {fom_med*1e3:8.2f} ms  (residual {fom_res:.1e}) "
            f"-- the ladder above is the iso-accuracy baseline")
        if FOM_ONLY:
            save()
            continue

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
                ap, build_s = weak_source_projector_sel(grid, spec)
                ref = pc.weak_source_term(grid, spec, "grid", Fs[0])
                chk = float(jnp.max(jnp.abs(ap(jnp.asarray(Fs[0])) - ref))
                            / (float(jnp.max(jnp.abs(ref))) + 1e-300))
                if not np.isfinite(chk) or chk > 1e-10:
                    raise SystemExit(f"selected-mode preprocessor disagrees with "
                                     f"pro_common.weak_source_term by {chk:.2e}")
                med, _ = time_fn(lambda: ap(jnp.asarray(Fs[0])).block_until_ready())
                # the reference's full dense sine transform, for the record
                ap_full, _b = weak_source_projector(grid, spec, "grid")
                med_full, _ = time_fn(lambda: ap_full(jnp.asarray(Fs[0])).block_until_ready())
                pre_cache[M] = (ap, med, build_s, chk, med_full)
                log(f"   preprocess M={M}: selected-mode matvec {med*1e3:.3f} ms/query "
                    f"(full dense sine transform {med_full*1e3:.3f} ms; offline table "
                    f"{build_s:.1f} s; vs reference rel {chk:.1e})")
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
            ap, pre_med, pre_build_s, pre_chk, pre_med_full = preprocess(M)
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

                pipeline = make_pipeline(ap, lm, u_full, z0)
                z_probe = lm(z0, f_ms_j[0], methods[method][0])[0]
                dec_med, _ = time_fn(lambda: u_full(z_probe).block_until_ready())
                Fs_j = [jnp.asarray(Fs[i]) for i in range(N_TEST)]

                for tau in methods[method]:
                    (per_p, per_err, per_jac, per_att, per_reason, per_fd,
                     per_red) = timed_sweep(pipeline, lm, z0, Fs_j, f_ms_j, tau,
                                            U_int, tn, Fs, fn_, grid, n_i)
                    e2e_ms = float(np.median(per_p)) * 1e3          # the timed pipeline
                    # DERIVED: the pipeline minus the two value-independent,
                    # k-independent stages measured in isolation
                    solve_ms = e2e_ms - pre_med * 1e3 - dec_med * 1e3
                    cens = [r not in TAU_OK for r in per_reason]
                    row = dict(pde="poisson2d", method=method, N=n, k=k, M=M, m=int(len(wq)),
                               tau=tau, time_ms=e2e_ms, err_rel_l2=float(np.mean(per_err)),
                               iters=float(np.mean(per_att)), jac_evals=float(np.mean(per_jac)),
                               censored=bool(any(cens)), n_sources=N_TEST, seed=SEED,
                               gpu=gpu_name, jax_backend=dev.platform, commit=commit,
                               node=NODE, slurm_job=os.environ.get("SLURM_JOB_ID"),
                               # ---- beyond the shared schema: diagnostics / provenance
                               arm=arm_tag, time_ms_solve=solve_ms,
                               time_ms_solve_derivation="timed pipeline minus the "
                               "separately measured preprocess and decode medians",
                               time_ms_pre=pre_med * 1e3, time_ms_decode=dec_med * 1e3,
                               time_ms_pre_full_transform=pre_med_full * 1e3,
                               time_ms_e2e_per_source=[float(v) * 1e3 for v in per_p],
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
                               fom_rule_for_speedup=f"cg(tol={min(FOM_LADDER):.0e}) -- the "
                               f"TRUTH-MANUFACTURING tolerance; use the iso-accuracy ladder "
                               f"in report['fom'] for a like-for-like denominator",
                               fom_iso_accuracy_ms=next(
                                   (r["fom_cg_s"] * 1e3 for r in sorted(
                                       ladder, key=lambda r: r["fom_cg_s"])
                                    if r["err_rel_l2"] <= float(np.mean(per_err))), None),
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
                if DO_CEILING and method == "coord" and arm_tag == "primary":
                    cm, cx_ = oracle_ceiling(dec_k, grid, ck[k]["Z_tr"], U_int, tn,
                                             n_i, GN_ITERS)
                    report["supplementary"].append(dict(
                        pde="poisson2d", method="oracle_ceiling", N=n, k=k, M=M,
                        m=int(len(wq)), tau=None, arm="ceiling",
                        err_rel_l2=cm, err_rel_l2_max=cx_, n_sources=N_TEST,
                        seed=SEED, gpu=gpu_name, node=NODE,
                        jax_backend=dev.platform, commit=commit))
                    log(f"   ceiling  N={n:4d} k={k:2d}: oracle inferred-latent "
                        f"{cm:.3e} (max {cx_:.3e})")
                log(f"   [cell {method} N={n} k={k} M={M} m={m}: {time.time()-t_cell:.0f}s]")

        # ---- control arms -----------------------------------------------------
        # (a) POOL CONTROL: the headline recipe's MESHFREE EQ pool (the primary
        #     arms use the interior grid for both methods so POD, which is only
        #     defined at grid nodes, gets the identical hyper-reduction).
        # (b) CAP CONTROL: the SAME grid pool but UNCAPPED (every interior node as
        #     an ECSW candidate), so the default 4096-candidate cap is bounded by
        #     measurement rather than assumed harmless.  Only run where the design
        #     matrix fits (CAP_CONTROL_MAX candidates).
        all_taus = sorted({t_ for arms in plan.values() for v in arms.values()
                           for tt in v.values() for t_ in tt}, reverse=True)

        def run_control(label, arm, k, M, m, dec_pts_fn, pts_np, wq, PhiT, Wl,
                        z0, u_full_c, ap_c, eq_info):
            lm_c, _r = ctol_tol.lm_tau_poisson(dec_pts_fn, k, pts_np, wq, PhiT, Wl, GN_ITERS)
            pipe_c = make_pipeline(ap_c, lm_c, u_full_c, z0)
            fm_c = [jnp.asarray(np.asarray(ap_c(jnp.asarray(Fs[i])))) for i in range(N_TEST)]
            Fs_j = [jnp.asarray(Fs[i]) for i in range(N_TEST)]
            for tau in all_taus:
                (pp, pe, pj, pa, pr, pfd, prd) = timed_sweep(
                    pipe_c, lm_c, z0, Fs_j, fm_c, tau, U_int, tn, Fs, fn_, grid, n_i)
                report["supplementary"].append(dict(
                    pde="poisson2d", method=label, N=n, k=k, M=M, m=int(len(wq)),
                    tau=tau, arm=arm, time_ms=float(np.median(pp)) * 1e3,
                    err_rel_l2=float(np.mean(pe)), jac_evals=float(np.mean(pj)),
                    iters=float(np.mean(pa)),
                    censored=bool(any(r not in TAU_OK for r in pr)),
                    censored_frac=float(np.mean([r not in TAU_OK for r in pr])),
                    rel_reduction_mean=float(np.mean(prd)),
                    fd_residual_rel_mean=float(np.mean(pfd)),
                    n_sources=N_TEST, seed=SEED, gpu=gpu_name, node=NODE,
                    jax_backend=dev.platform, commit=commit,
                    eq_rel_fit=eq_info["rel_fit"], eq_n_cand=eq_info["n_cand"]))
                log(f"   [{arm}] {label} N={n} k={k} tau={tau:.0e} "
                    f"e2e {np.median(pp)*1e3:.2f} ms err {np.mean(pe):.3e} "
                    f"eqfit {eq_info['rel_fit']:.2e}")
            save()

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
            PhiT, Wl = pc.colloc_mode_table(grid, spec, "offgrid", cand_off[keep])
            ap_o, _ = weak_source_projector(grid, spec, "offgrid")
            run_control("coord_meshfree_pool", "pool_control", k, M, m, dec_k,
                        cand_off[keep], wq, PhiT, Wl,
                        jnp.asarray(ck[k]["Z_tr"].mean(0)),
                        jax.jit(lambda z, _d=dec_k: _d(z, grid.coords_int)), ap_o, eq_info)

        if CAP_CONTROL and 8 in ks_used and cand_pos.size < n_i2 and n_i2 <= CAP_CONTROL_MAX:
            k, M, m = 8, M_MODES, MQ
            spec = dict(kind="weak", alpha=1.0, M=M)
            ap_g, _pm, _pb, _pc2, _pf = preprocess(M)
            Phi_f = phi_full(M)
            full_pos = np.arange(n_i2)
            for method in ("coord", "pod"):
                if method == "coord":
                    dec_k = ck[k]["dec"]
                    Z_snap = ck[k]["Z_tr"]
                    z0c = jnp.asarray(Z_snap.mean(0))
                    u_c = jax.jit(lambda z, _d=dec_k: _d(z, grid.coords_int))
                    u_f = u_c
                else:
                    Vk = np.ascontiguousarray(Vfull[:, :k])
                    Vk_j = jnp.asarray(Vk)
                    Z_snap = X_tr @ Vk
                    z0c = jnp.asarray(c_mean_full[:k])
                    u_c = jax.jit(lambda c, _V=Vk_j: _V @ c)
                    u_f = u_c
                keep, wq, eq_info = ctol_eq.eq_fit_poisson(
                    u_c, u_f, Phi_f, Phi_f, Z_snap, k, m,
                    f"poisson {method} UNCAPPED-POOL N={n} k={k}", pc.nnls_capped)
                pts_np = np.asarray(grid.coords_int)[full_pos[keep]]
                PhiT, Wl = pc.colloc_mode_table(grid, spec, "grid", pts_np)
                if method == "coord":
                    dpf = dec_k
                else:
                    Vq_j = jnp.asarray(Vk[full_pos[keep]])
                    dpf = lambda z, xy, _V=Vq_j: _V @ z
                run_control(f"{method}_uncapped_pool", "cap_control", k, M, m, dpf,
                            pts_np, wq, PhiT, Wl, z0c, u_f, ap_g, eq_info)

        log(f"== N={n} done [{time.time()-t_mesh:.0f}s]")

    report["complete"] = True
    save()
    log("DONE")


if __name__ == "__main__":
    main()
