"""POISSON-2D arm of the ROM-WARM-STARTED-FOM cell.

Hand the coordinate ROM's decoded field to the FOM's conjugate-gradient solver as
its INITIAL GUESS and finish to a full-accuracy tolerance tau_FOM.  The answer is
then FOM-exact; the question is the total cost and how it scales with N.

  total  =  t_pre  +  t_rom  +  t_decode  +  t_fom(from the ROM guess)
  vs the honest baseline  t_fom_baseline  =  the SAME CG from a ZERO start.

Everything is reused from the reference Poisson harness:
  * decoder, grid, sine test modes, weak-form source term ...... pro_common
  * NNLS-EQ quadrature fit, jitted weak-form LM, source projector fu_eq
  * FD operator A = -Laplacian (ghost-zero Dirichlet) ........... ms_parametric
Only two things are NEW here, and both are deliberately shared by BOTH arms:
  1. `wsf_util.make_cg`  -- a counting CG.  The testbed's own
     `jax.scipy.sparse.linalg.cg` cannot report iterations, so it is kept as the
     CORRECTNESS REFERENCE (asserted below) rather than the timed baseline.  ONE
     compiled kernel serves both arms; only `x0` differs.
  2. `make_lm_obj_jit` -- fu_eq.make_lm_jit plus ONE extra stopping test:

        ROM TOLERANCE.  Stop at the first ACCEPTED LM iterate whose weak-form
        objective V(z) = || Wl * (PhiT @ (wq * dec(z, pts))) - f_m ||_2 satisfies

            V(z) <= rom_tau * V(z_0)

        i.e. a RELATIVE REDUCTION OF THE WEAK-FORM OBJECTIVE from the initial
        guess z_0.  rom_tau = 0 disables it, and the solver is then NUMERICALLY
        EQUIVALENT to fu_eq.make_lm_jit (the final latent is asserted equal to
        1e-12 relative below, and exact bitwise agreement is recorded when it
        holds).  A tolerance on ||Au-f||/||f|| is
        unreachable: at the weak-form solution that sits near 2e-1 while the
        FIELD error is ~8e-3.

Timing protocol (non-negotiable): every ladder point is measured SEQUENTIALLY IN
ONE PROCESS ON ONE GPU, warm-up TIME_WARM (2), median of TIME_REPS (7),
`block_until_ready`, f64, JAX_DEFAULT_MATMUL_PRECISION=highest.

Usage: PKL=<pkl> [NS=32,64,128,256,512] [ROM_TAUS=1e-1,1e-2,1e-3,0]
       [FOM_TAUS=1e-6,1e-8,1e-10] [M=64] [MQ=256] python wsf_poisson.py <out.json>
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
_EXPS = os.path.dirname(HERE)
for d in (os.path.join(HERE, "deps"),
          os.path.join(_EXPS, "poisson2d-rom-objective"),
          os.path.join(_EXPS, "poisson2d-rom-objective", "followup")):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import pro_common as pc                                        # noqa: E402
from pro_common import mp, F64                                 # noqa: E402
from fu_eq import eq_fit, make_lm_jit, weak_source_projector   # noqa: E402
import wsf_util as wu                                          # noqa: E402

OUT = sys.argv[1]
PKL = os.environ["PKL"]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256,512").split(",") if v]
ROM_TAUS = [float(v) for v in os.environ.get("ROM_TAUS", "1e-1,1e-2,1e-3,0").split(",")]
FOM_TAUS = [float(v) for v in os.environ.get("FOM_TAUS", "1e-6,1e-8,1e-10").split(",")]
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
N_TEST = int(os.environ.get("N_TEST", "16"))       # accuracy / iteration statistics
N_TIME = int(os.environ.get("N_TIME", "8"))        # test cases that are wall-clock timed
N_CHECK = int(os.environ.get("N_CHECK", "4"))      # cases used for the solver cross-checks
INIT = os.environ.get("INIT", "mean")
if INIT != "mean":
    raise SystemExit("INIT must be 'mean': the 'nearest' cold start needs a "
                     "PER-QUERY lookup that would also have to be timed and charged "
                     "to the hybrid total; it is not part of this benchmark.")
CG_MAXITER = int(os.environ.get("CG_MAXITER", "40000"))
REF_TAU = float(os.environ.get("REF_TAU", "1e-12"))   # tolerance of the reference solution
# The reference solution only has to be far more accurate than the tightest REPORTED
# tolerance so that error grading is meaningful.  Below ~1e-12 the achievable relative
# residual is limited by f64 rounding in the FD operator itself and grows with N
# (measured: 1.0e-13 at N=128, 5.7e-13 at N=256), so demanding REF_TAU exactly would
# abort on a floating-point floor rather than on a real defect.  The requirement is
# therefore "at least REF_MARGIN times tighter than min(FOM_TAUS)".
REF_MARGIN = float(os.environ.get("REF_MARGIN", "10"))
BURN_S = float(os.environ.get("BURN_S", "3"))   # GPU clock burn-in before timing
# NOTE for the README: because of that floor, `err_final` at the tightest tau is
# bounded below by the REFERENCE solution's own accuracy.  The actual correctness
# gate is reference-free -- the TRUE relative residual of the delivered iterate must
# be <= tau, and every row asserts that for both arms.
POOL = os.environ.get("POOL", "offgrid")
# "panel"        : one N per job, fanned out across GPUs.  Accuracy, iteration counts
#                  and the WITHIN-N timing breakdown are valid (one panel = one job =
#                  one GPU); its wall clock may NEVER be placed on a cross-N axis.
# "consolidated" : every N measured SEQUENTIALLY IN ONE JOB ON ONE GPU.  The ONLY
#                  timing source the cross-N figures and the crossover-N claim may use.
RUN_ROLE = os.environ.get("RUN_ROLE", "consolidated")

REASONS = {0: "budget", 1: "converged", 2: "abs_tol", 3: "lambda_max",
           5: "nan_at_init", 6: "rom_tau"}


# --------------------------------------------------------------------------
def make_lm_obj_jit(dec, K, pts, wq, PhiT, Wl, budget, obj_rel=0.0):
    """`fu_eq.make_lm_jit` (rel_tol=0) with ONE added stopping test: stop at the
    first ACCEPTED iterate with V(z) <= obj_rel * V(z_0).  obj_rel <= 0 disables
    it, and the solver is then identical to fu_eq.make_lm_jit (asserted in
    `check_lm_equivalence`).  Damping schedule, acceptance rule, relative-decrease
    and step-size stops, budget and iteration accounting are UNCHANGED.

    Reason codes: 0 budget, 1 converged (rel-dec/step), 3 lambda_max,
    5 nan_at_init, 6 rom_tau (the objective-reduction stop)."""
    pts = jnp.asarray(pts); wq = jnp.asarray(wq)
    PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)

    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m
    rJ = lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m))
    rn_fn = lambda z, f_m: jnp.linalg.norm(r_of(z, f_m))

    def lm(z0, f_m):
        r0, J0 = rJ(z0, f_m)
        v0 = jnp.linalg.norm(r0)
        obj_target = obj_rel * v0                     # 0 when the test is disabled
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(1), jnp.where(jnp.isfinite(v0), jnp.int32(0), jnp.int32(5)))

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, acc, nJ, _ = s
            H = J.T @ J; g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            v_new = jnp.where(finite, rn_fn(z_new, f_m), jnp.inf)
            accept = finite & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, f_m), lambda: (r, J))
            z = jnp.where(accept, z_new, z); val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32); nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (obj_rel > 0) & (val <= obj_target), jnp.int32(6),
                       jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), jnp.int32(1),
                        jnp.where((~accept) & (lam >= 1e12), jnp.int32(3), jnp.int32(0))))
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason)

        z, J, r, val, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, val, v0, nJ, acc, att, reason

    return jax.jit(lm)


def sources(n, cx, cy, w, a, n_train, n_test):
    return np.stack([mp.source_interior(n, cx[n_train + i], cy[n_train + i],
                                        w[n_train + i], a[n_train + i])
                     for i in range(n_test)])


def main():
    dev = jax.devices()[0]
    prov = wu.provenance(HERE)
    print(f"jax_backend={jax.default_backend()} device={dev} "
          f"commit={prov['commit_short']} NS={NS} rom_taus={ROM_TAUS} "
          f"fom_taus={FOM_TAUS} M={M_MODES} m={MQ} pool={POOL} "
          f"reps={TIME_REPS} warm={TIME_WARM}", flush=True)
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu -- refusing to produce timings")

    d, cfg, stages_all, Z_tr, HARD_BC = pc.load_pkl(PKL)
    K = cfg["K_LAT"]
    dec = pc.make_decoder(stages_all[:1], hard_bc=bool(HARD_BC))
    N_TRAIN = mp.N_TRAIN
    cx, cy, w, a, z_par = mp.sample_params()
    z_mean = jnp.asarray(Z_tr.mean(0))
    zt = np.asarray(z_par)
    nn_idx = np.argmin(((zt[N_TRAIN:, None, :] - zt[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
    z_init = z_mean if INIT == "mean" else jnp.asarray(Z_tr[nn_idx[0]])

    report = dict(
        config=dict(pde="poisson2d", pkl=os.path.basename(PKL), pkl_config=cfg,
                    hard_bc=HARD_BC, K=K, M=M_MODES, m=MQ, pool=POOL, ns=NS,
                    rom_taus=ROM_TAUS, fom_taus=FOM_TAUS, gn_iters=GN_ITERS,
                    n_test=N_TEST, n_time=N_TIME, init=INIT, seed=mp.SEED,
                    time_reps=TIME_REPS, time_warm=TIME_WARM, cg_maxiter=CG_MAXITER,
                    ref_tau=REF_TAU, run_role=RUN_ROLE,
                    rom_tau_definition="V(z) <= rom_tau * V(z0), V = weak-form "
                                       "objective ||Wl*(PhiT@(wq*dec(z,pts))) - f_m||_2, "
                                       "z0 = initial latent; rom_tau=0 -> reference LM stops"),
        provenance=prov, rows=[], checks=[])

    def save():
        # allow_nan=False: an invalid number must fail loudly here rather than be
        # serialised as a bare NaN and silently averaged into a headline later.
        json.dump(report, open(OUT, "w"), indent=1, default=float, allow_nan=False)

    for n in NS:
        t_n0 = time.time()
        grid = pc.Grid(n)
        n_i = grid.n_i
        op = lambda v: mp.neg_lap_interior(v, n)
        cg = wu.make_cg(op, maxiter=CG_MAXITER)
        Fs = sources(n, cx, cy, w, a, N_TRAIN, N_TEST)
        Fj = [jnp.asarray(Fs[i]) for i in range(N_TEST)]
        zero = jnp.zeros((n_i, n_i), F64)

        # ---- reference solutions (used ONLY to grade errors, never fed to a solve)
        U_ref, ref_iters, ref_res, ref_flag = [], [], [], []
        for i in range(N_TEST):
            x, k, rr, fl = cg(Fj[i], zero, REF_TAU)
            U_ref.append(np.asarray(x)); ref_iters.append(int(k))
            ref_res.append(float(jnp.linalg.norm(op(x) - Fj[i]) / jnp.linalg.norm(Fj[i])))
            ref_flag.append(int(fl))
        U_ref = np.stack(U_ref)
        ref_norm = np.linalg.norm(U_ref.reshape(N_TEST, -1), axis=1)
        # cross-check the counting CG against the testbed's own jax.scipy CG at
        # EVERY tolerance that will be reported and on several right-hand sides,
        # comparing TRUE residuals -- one RHS at one tolerance is not enough to
        # validate a replacement solver (Codex).
        native = {}
        subchecks = []
        ref_accept = min(FOM_TAUS) / REF_MARGIN
        for tt in sorted(set(FOM_TAUS + [REF_TAU])):
            # a REPORTED tolerance must be met exactly; the reference tolerance only has
            # to clear the f64 floor by REF_MARGIN (see the note at REF_TAU above)
            acc = tt if tt in FOM_TAUS else max(tt, ref_accept)
            for i in range(min(N_CHECK, N_TEST)):
                ref_fn, c = wu.cg_reference_check(op, Fj[i], tt, cg)
                c["case"] = i; c["accept_threshold"] = acc
                subchecks.append(c)
                if c["counting_cg_true_rel_res"] > acc:
                    raise SystemExit(f"N={n} tau={tt}: counting CG true res "
                                     f"{c['counting_cg_true_rel_res']:.2e} > {acc:.2e}")
                if tt in FOM_TAUS and c["counting_cg_flag"] != 0:
                    raise SystemExit(f"N={n} tau={tt}: counting CG flag "
                                     f"{c['counting_cg_flag']}")
                if c["rel_diff_vs_jax_scipy_cg"] > 1e-6:
                    raise SystemExit(f"N={n} tau={tt} case {i}: counting CG disagrees "
                                     f"with the testbed CG by "
                                     f"{c['rel_diff_vs_jax_scipy_cg']:.2e}")
            if tt in FOM_TAUS:
                native[tt] = ref_fn
        if np.max(ref_res) > ref_accept:
            raise SystemExit(f"N={n}: the REFERENCE solutions only reached a true relative "
                             f"residual of {np.max(ref_res):.2e}, which is not "
                             f"{REF_MARGIN:g}x tighter than the tightest reported tolerance "
                             f"{min(FOM_TAUS):.0e} -- error grading would not be meaningful")
        chk = dict(N=n, subchecks=subchecks, ref_accept_threshold=ref_accept,
                   ref_iters_max=int(np.max(ref_iters)),
                   ref_true_rel_res_max=float(np.max(ref_res)),
                   ref_flags=sorted(set(ref_flag)),
                   rel_diff_vs_jax_scipy_cg=float(max(c["rel_diff_vs_jax_scipy_cg"]
                                                      for c in subchecks)))
        report["checks"].append(chk)
        print(f"== N={n}: reference CG {np.max(ref_iters)} iters, true rel res "
              f"{np.max(ref_res):.2e}; counting-CG vs jax.scipy.cg max rel diff "
              f"{chk['rel_diff_vs_jax_scipy_cg']:.2e} over {len(subchecks)} checks",
              flush=True)

        # ---- ROM pieces at this N (EQ weights refit on THIS grid, as in fu_timing)
        spec = dict(kind="weak", alpha=1.0, M=M_MODES)
        if POOL == "full":
            pts = np.asarray(grid.coords_int); wq = np.ones(pts.shape[0]); eq_info = None
            kind = "grid"
        else:
            pts, wq, eq_info = eq_fit(dec, grid, Z_tr, K, M_MODES, MQ, POOL)
            kind = "grid" if POOL == "grid" else "offgrid"
        PhiT, Wl = pc.colloc_mode_table(grid, spec, kind, pts)
        pre_apply, pre_build_s = weak_source_projector(grid, spec, kind)
        f_ms = [pre_apply(Fj[i]) for i in range(N_TEST)]
        pre_chk = float(jnp.max(jnp.abs(f_ms[0]
                                        - pc.weak_source_term(grid, spec, kind, Fs[0]))))
        dec_int = jax.jit(lambda z: dec(z, grid.coords_int).reshape(n_i, n_i))

        # ---- the reference LM must be reproduced exactly when rom_tau = 0
        lm_ref = make_lm_jit(dec, K, pts, wq, PhiT, Wl, GN_ITERS, 0.0)
        lm0 = make_lm_obj_jit(dec, K, pts, wq, PhiT, Wl, GN_ITERS, 0.0)
        z_a = lm_ref(z_init, f_ms[0])[0]
        z_b = lm0(z_init, f_ms[0])[0]
        lm_equiv = float(jnp.max(jnp.abs(z_a - z_b)))
        lm_equiv_rel = lm_equiv / (1.0 + float(jnp.linalg.norm(z_a)))
        if lm_equiv_rel > 1e-12:
            raise SystemExit(f"N={n}: rom_tau=0 LM does not reproduce fu_eq.make_lm_jit "
                             f"(max|dz| = {lm_equiv:.3e})")
        report["checks"][-1].update(lm_rom_tau0_vs_reference_maxabs=lm_equiv,
                                    lm_rom_tau0_bitwise_identical=bool(lm_equiv == 0.0),
                                    preprocess_vs_reference_maxabs=pre_chk,
                                    eq_info=eq_info)

        # ---- per-N constant online costs
        pre_med, pre_all = wu.time_fn(lambda: pre_apply(Fj[0]).block_until_ready(),
                                      TIME_REPS, TIME_WARM)
        dec_med, dec_all = wu.time_fn(lambda: dec_int(z_init).block_until_ready(),
                                      TIME_REPS, TIME_WARM)
        dec_full = jax.jit(lambda z: dec(z, grid.coords))
        decfull_med, _ = wu.time_fn(lambda: dec_full(z_init).block_until_ready(),
                                    TIME_REPS, TIME_WARM)

        # ---- GPU BURN-IN: bring the device to a steady clock before ANY timing at
        # this mesh (see wsf_util.gpu_burn for the 17% bias this removes).
        burn_n = wu.gpu_burn(lambda: cg(Fj[0], zero, FOM_TAUS[0])[0].block_until_ready(),
                             BURN_S)
        print(f"   burn-in: {burn_n} CG solves in {BURN_S:.0f}s", flush=True)

        # ---- DIRECT-SOLVER baseline (the reviewers' "a direct solver beats you").
        # The FD system on a square with Dirichlet walls is diagonalised EXACTLY by
        # the same discrete sine basis the ROM uses as test modes, so u = S ((S^T f S)
        # / Lambda) S^T is the exact inverse of the SAME operator A -- not an
        # approximation, and O(n^{3/2}) here (two dense (n_i x n_i) products).
        direct = jax.jit(lambda F: grid.S @ ((grid.S.T @ F @ grid.S) / grid.lam) @ grid.S.T)
        d_err = [float(np.linalg.norm(np.asarray(direct(Fj[i])) - U_ref[i]) / ref_norm[i])
                 for i in range(N_TEST)]
        d_res = [float(jnp.linalg.norm(op(direct(Fj[i])) - Fj[i]) / jnp.linalg.norm(Fj[i]))
                 for i in range(N_TEST)]
        d_t = [wu.time_fn(lambda ii=i: direct(Fj[ii]).block_until_ready(),
                          TIME_REPS, TIME_WARM)[0] for i in range(N_TIME)]
        direct_ms = float(np.mean(d_t)) * 1e3
        report["checks"][-1].update(direct_solver_ms=direct_ms,
                                    direct_solver_rel_err=float(np.mean(d_err)),
                                    direct_solver_rel_residual=float(np.mean(d_res)))
        print(f"   DIRECT (sine-diagonalised exact inverse): {direct_ms:.2f} ms, "
              f"err {np.mean(d_err):.2e}, rel resid {np.mean(d_res):.2e}", flush=True)

        # ---- pure-FOM baseline: the SAME CG from a ZERO start
        base = {}
        for ft in FOM_TAUS:
            it, res, tms, err = [], [], [], []
            for i in range(N_TEST):
                x, k, rr, fl = cg(Fj[i], zero, ft)
                it.append(int(k)); res.append(float(jnp.linalg.norm(op(x) - Fj[i])
                                                    / jnp.linalg.norm(Fj[i])))
                err.append(float(np.linalg.norm(np.asarray(x) - U_ref[i]) / ref_norm[i]))
                if int(fl) != 0:
                    raise SystemExit(f"N={n} tau={ft}: baseline CG flag {int(fl)}")
                if i < N_TIME:
                    # UNPAIRED timing, kept only as a diagnostic.  The authoritative
                    # baseline time is measured BACK TO BACK with the warm arm inside
                    # the ROM ladder below, because a device-clock drift between two
                    # separated measurement blocks is larger than the effect under test.
                    m_, _a = wu.time_fn(
                        lambda ii=i: cg(Fj[ii], zero, ft)[0].block_until_ready(),
                        TIME_REPS, TIME_WARM)
                    tms.append(m_)
            base[ft] = dict(iters=it, true_rel_res=res, t_s=tms,
                            t_ms_unpaired=float(np.mean(tms)) * 1e3, err=err)
            print(f"   FOM baseline tau={ft:.0e}: {np.mean(it):.1f} iters, "
                  f"{np.mean(tms)*1e3:.2f} ms (unpaired diagnostic), "
                  f"err {np.mean(err):.2e}", flush=True)

        # ---- POST-HOC ORACLE DIAGNOSTIC (never timed, never on the hybrid path):
        # the plain-CG error curve from a zero start, graded against the reference
        # solution.  It answers "how many CG iterations is the ROM answer worth?"
        # WITHOUT ever stopping a solver on the reference (Codex leakage rule).
        curves = []
        for i in range(min(N_CHECK, N_TEST)):
            nmax = int(max(base[ft]["iters"][i] for ft in FOM_TAUS))
            curves.append(np.asarray(wu.cg_error_curve(op, Fj[i], jnp.asarray(U_ref[i]),
                                                       nmax)))

        # ---- ROM tolerance ladder
        for rt in ROM_TAUS:
            lm = make_lm_obj_jit(dec, K, pts, wq, PhiT, Wl, GN_ITERS, max(rt, 0.0))
            zs, vals, v0s, nJs, atts, rsn = [], [], [], [], [], []
            for i in range(N_TEST):
                z, val, v0, nJ, acc, att, r_ = lm(z_init, f_ms[i])
                zs.append(z); vals.append(float(val)); v0s.append(float(v0))
                nJs.append(int(nJ)); atts.append(int(att)); rsn.append(int(r_))
            rom_t = []
            for i in range(N_TIME):
                m_, _a = wu.time_fn(
                    lambda ii=i: lm(z_init, f_ms[ii])[0].block_until_ready(),
                    TIME_REPS, TIME_WARM)
                rom_t.append(m_)
            t_rom_ms = float(np.mean(rom_t)) * 1e3
            U_rom = [np.asarray(dec_int(zs[i])) for i in range(N_TEST)]
            err_rom = [float(np.linalg.norm(U_rom[i] - U_ref[i]) / ref_norm[i])
                       for i in range(N_TEST)]
            res_rom = [float(jnp.linalg.norm(op(jnp.asarray(U_rom[i])) - Fj[i])
                             / jnp.linalg.norm(Fj[i])) for i in range(N_TEST)]
            # A-norm error ratio: CG's convergence is governed by ||u - u*||_A, so
            # this is the quantity that decides how many iterations a warm start
            # can possibly save (a zero start has ratio 1 by definition).
            def a_norm(v):
                v = jnp.asarray(v)
                return float(jnp.sqrt(jnp.abs(jnp.sum(v * op(v)))))
            a_rom = [a_norm(U_rom[i] - U_ref[i]) / max(a_norm(U_ref[i]), 1e-300)
                     for i in range(N_TEST)]
            obj_red = [vals[i] / max(v0s[i], 1e-300) for i in range(N_TEST)]
            # ORACLE DIAGNOSTIC (never timed, never on the hybrid's path): how many
            # plain CG iterations from a zero start reach the ROM's own field accuracy?
            # This is the "what is the ROM's answer worth, in CG iterations" number.
            eq_it = []
            for i, curve in enumerate(curves):
                hit = np.nonzero(curve <= err_rom[i])[0]
                eq_it.append(int(hit[0]) if hit.size else -1)
            print(f"   ROM tau={rt:g}: {np.mean(nJs):.1f} LM iters, {t_rom_ms:.2f} ms, "
                  f"obj red {np.mean(obj_red):.2e}, field err {np.mean(err_rom):.3e}, "
                  f"rel resid {np.mean(res_rom):.2e}, A-norm err ratio "
                  f"{np.mean(a_rom):.2e}, worth {np.mean(eq_it):.1f} CG iters, reasons "
                  f"{ {REASONS[r]: rsn.count(r) for r in set(rsn)} }", flush=True)

            for ft in FOM_TAUS:
                it, res, tms, errf, flg = [], [], [], [], []
                btms, ntw, ntb = [], [], []
                for i in range(N_TEST):
                    x0 = jnp.asarray(U_rom[i])
                    x, k, rr, fl = cg(Fj[i], x0, ft)
                    it.append(int(k)); flg.append(int(fl))
                    res.append(float(jnp.linalg.norm(op(x) - Fj[i]) / jnp.linalg.norm(Fj[i])))
                    errf.append(float(np.linalg.norm(np.asarray(x) - U_ref[i]) / ref_norm[i]))
                    if i < N_TIME:
                        # PAIRED: warm and zero start measured back to back on the same
                        # case, so any device-clock drift hits both arms equally.
                        m_, _a = wu.time_fn(
                            lambda ii=i, xx=x0: cg(Fj[ii], xx, ft)[0].block_until_ready(),
                            TIME_REPS, TIME_WARM)
                        b_, _b = wu.time_fn(
                            lambda ii=i: cg(Fj[ii], zero, ft)[0].block_until_ready(),
                            TIME_REPS, TIME_WARM)
                        nw_, _c = wu.time_fn(
                            lambda ii=i, xx=x0: native[ft](Fj[ii], xx).block_until_ready(),
                            TIME_REPS, TIME_WARM)
                        nb_, _d = wu.time_fn(
                            lambda ii=i: native[ft](Fj[ii], zero).block_until_ready(),
                            TIME_REPS, TIME_WARM)
                        tms.append(m_); btms.append(b_); ntw.append(nw_); ntb.append(nb_)
                if max(flg) != 0:
                    raise SystemExit(f"N={n} rom_tau={rt} fom_tau={ft}: warm CG flag {max(flg)}")
                if max(res) > ft:
                    raise SystemExit(f"N={n} rom_tau={rt} fom_tau={ft}: warm CG true "
                                     f"residual {max(res):.2e} > {ft:.0e}")
                ntw = [wu.time_fn(lambda ii=i: native[ft](
                           Fj[ii], jnp.asarray(U_rom[ii])).block_until_ready(),
                       TIME_REPS, TIME_WARM)[0] for i in range(N_TIME)]
                t_fom_ms = float(np.mean(tms)) * 1e3
                t_total_ms = pre_med * 1e3 + t_rom_ms + dec_med * 1e3 + t_fom_ms
                t_base_ms = float(np.mean(btms)) * 1e3      # PAIRED baseline
                row = dict(
                    pde="poisson2d", N=n, n_dof=n_i ** 2, rom_tau=rt, fom_tau=ft,
                    t_rom_ms=t_rom_ms, t_pre_ms=pre_med * 1e3, t_decode_ms=dec_med * 1e3,
                    t_decode_full_grid_ms=decfull_med * 1e3,
                    t_fom_ms=t_fom_ms, t_total_ms=t_total_ms,
                    t_fom_baseline_ms=t_base_ms,
                    iters_from_rom=float(np.mean(it)),
                    iters_from_baseline=float(np.mean(base[ft]["iters"])),
                    # the wall clock is the mean over the FIRST N_TIME cases, so the
                    # iteration counts over exactly that subset are reported next to it
                    # (the headline iteration columns use all N_TEST cases)
                    iters_from_rom_timed=float(np.mean(it[:N_TIME])),
                    iters_from_baseline_timed=float(np.mean(base[ft]["iters"][:N_TIME])),
                    iters_from_rom_all=it, iters_from_baseline_all=base[ft]["iters"],
                    iter_saving_frac=1.0 - float(np.mean(it))
                                     / max(float(np.mean(base[ft]["iters"])), 1e-30),
                    err_rel_l2_rom=float(np.mean(err_rom)),
                    rom_rel_residual=float(np.mean(res_rom)),
                    rom_err_Anorm_ratio=float(np.mean(a_rom)),
                    cg_iters_equivalent_to_rom=(float(np.mean([v for v in eq_it if v >= 0]))
                                               if any(v >= 0 for v in eq_it) else None),
                    cg_iters_equivalent_not_reached=int(sum(v < 0 for v in eq_it)),
                    rom_obj_reduction=float(np.mean(obj_red)),
                    rom_lm_iters=float(np.mean(nJs)),
                    rom_lm_attempts=float(np.mean(atts)),
                    rom_lm_reasons={REASONS[r]: rsn.count(r) for r in set(rsn)},
                    err_final=float(np.mean(errf)),
                    err_final_baseline=float(np.mean(base[ft]["err"])),
                    final_rel_residual=float(np.max(res)),
                    reference_true_rel_residual=float(np.max(ref_res)),
                    final_rel_residual_baseline=float(np.max(base[ft]["true_rel_res"])),
                    speedup_vs_fom=t_base_ms / t_total_ms,
                    speedup_fom_stage_only=t_base_ms / t_fom_ms,
                    t_fom_direct_ms=direct_ms,
                    t_fom_native_ms=float(np.mean(ntw)) * 1e3,
                    t_fom_baseline_native_ms=float(np.mean(ntb)) * 1e3,
                    t_fom_baseline_unpaired_ms=base[ft]["t_ms_unpaired"],
                    speedup_vs_fom_native=(float(np.mean(ntb)) * 1e3) / (
                        pre_med * 1e3 + t_rom_ms + dec_med * 1e3
                        + float(np.mean(ntw)) * 1e3),
                    direct_rel_err=float(np.mean(d_err)),
                    speedup_vs_direct=direct_ms / t_total_ms,
                    n_test=N_TEST, n_time=N_TIME, seed=mp.SEED, run_role=RUN_ROLE,
                    gpu=prov["gpu"], gpu_kind=prov["gpu_kind"],
                    jax_backend=prov["jax_backend"], commit=prov["commit"],
                    slurm_job_id=prov["slurm_job_id"])
                report["rows"].append(row); save()
                print(f"     tau_FOM={ft:.0e}: CG {np.mean(it):.1f} vs "
                      f"{np.mean(base[ft]['iters']):.1f} iters  "
                      f"total {t_total_ms:.2f} ms vs FOM {t_base_ms:.2f} ms  "
                      f"({t_base_ms/t_total_ms:.2f}x)  err_final {np.mean(errf):.2e}",
                      flush=True)
        print(f"== N={n} done [{time.time()-t_n0:.0f}s]", flush=True)

    report["complete"] = True
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
