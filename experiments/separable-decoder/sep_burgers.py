"""Separable-decoder Burgers-2D cell(s): train (no POD), weak NM-ROM rollout.

N-scaling round (2026-08-23).  Differences from the N=64 first cell, each one
mandated by AUDIT-2026-08-23 / HANDOFF.md:
  * END-TO-END timed online path: IC latent fit (from the known u0) + latent
    rollout + FULL-GRID decode of all 51 states, with the (IC, rollout+decode)
    split reported separately -- the timed ROM returns exactly what the timed
    FOM returns (51 full fields);
  * the reported ROM error is computed from a timed invocation's own decoded
    fields; the incumbent untimed rollout is run as a cross-check and the max
    field deviation between the two is recorded;
  * raw timing repetitions retained; balanced alternating-order schedule
    across every ROM and FOM arm (sep_common.time_multi);
  * STRONG classical baseline in-job: a tolerance-terminated inexact-Newton
    rollout ladder (NEWTON_LADDER env, "ntol:lintol" pairs) alongside the
    truth-generating fixed-8-Newton rollout, which is labelled over-solved and
    never the headline comparator;
  * IC fit repair: inits = mean code + the IC_TOPK nearest decoded training
    t=0 states (nearest to the KNOWN u0; training data only), solved by the
    incumbent-identical jitted LM (ctol_tol.lm_tau_generic, tau=0);
  * per-step stop-reason distributions reported next to every error;
  * EQ fit through the mesh-safe capped candidate pool (ctol_eq.eq_fit_burgers,
    the cost-to-tolerance recipe) -- the full-grid pool builds a 17 GB design
    matrix at N=512; per-row EQ diagnostics recorded;
  * several (K, R, arch) cells per job from ONE in-job seed-regenerated data
    build (CELLS env); training states subsampled BEFORE the interior copy
    (the naive path holds two 61 GB copies at N=512), t=0 states always kept.

Two ROM arms through the SAME incumbent weak operators and LM stepper
(blat_common.make_weak_ops / _finish_ops / rollout_jit).  GATE 0 (asserted per
cell): weak residual/Jacobian of the two arms agree <= 1e-12 relative.  The
FOM-exact sign-upwind stencil is preserved exactly.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402  (path set by sc)
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
CELLS = os.environ.get("CELLS", "")           # "K:R[:NFF[:FFSCALE[:STEPS]]]" csv
K_DEF = int(os.environ.get("K", "16"))
R_DEF = int(os.environ.get("R", str(4 * K_DEF)))
NFF = int(os.environ.get("NFF", "64"))
FF_SCALE = float(os.environ.get("FF_SCALE", "4.0"))
G_HID = int(os.environ.get("G_HID", "128"))
H_HID = int(os.environ.get("H_HID", "128"))
G_LAY = int(os.environ.get("G_LAY", "2"))
H_LAY = int(os.environ.get("H_LAY", "2"))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "4"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
IC_TOPK = int(os.environ.get("IC_TOPK", "3"))
OUT_DIR = os.environ.get("OUT_DIR", ".")
TIME_REPS = int(os.environ.get("TIME_REPS", "5"))
TIME_WARM = int(os.environ.get("TIME_WARM", "1"))
NEWTON_LADDER = os.environ.get("NEWTON_LADDER", "1e-2:1e-3,1e-4:1e-5,1e-6:1e-7")
NEWTON_MAX = int(os.environ.get("NEWTON_MAX", "20"))
# cells (by index in CELLS) whose timing block includes the FULL newton-tol
# ladder; other cells time ROM arms + the truth-generator anchor only (the
# ladder is cell-independent and same-job, so once per job suffices)
BASELINE_CELLS = {int(v) for v in os.environ.get("BASELINE_CELLS", "0").split(",")}

STEP_REASONS = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}
IC_REASONS = {0: "budget", 1: "converged", 2: "tau", 3: "lambda_max",
              4: "lambda_max_nonfinite", 5: "nan_at_init"}


def parse_cells():
    if not CELLS:
        return [dict(K=K_DEF, R=R_DEF, nff=NFF, ffs=FF_SCALE, steps=STEPS)]
    out = []
    for tok in CELLS.split(","):
        p = tok.split(":")
        k = int(p[0])
        out.append(dict(K=k, R=int(p[1]) if len(p) > 1 and p[1] else 4 * k,
                        nff=int(p[2]) if len(p) > 2 and p[2] else NFF,
                        ffs=float(p[3]) if len(p) > 3 and p[3] else FF_SCALE,
                        steps=int(p[4]) if len(p) > 4 and p[4] else STEPS))
    return out


def reason_hist(codes, names):
    h = {}
    for c in codes:
        nm = names.get(int(c), str(int(c)))
        h[nm] = h.get(nm, 0) + 1
    return h


def make_newton_tol_rollout(n, ntol, lin_tol, residual, max_newton=NEWTON_MAX):
    """Tolerance-terminated inexact-Newton BE rollout on the SAME discrete
    residual as the truth generator (bf.make_rollout's `residual`), stopping
    each step's Newton loop at ||R|| <= ntol * ||u_prev|| with BiCGStab linear
    tolerance lin_tol.  This is the STRONG classical baseline arm; the fixed
    eight-Newton truth generator is over-solved by construction."""

    def step(u_prev, nu):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def cond(s):
            u, rn, it = s
            return (rn > ntol * u_scale) & (it < max_newton)

        def body(s):
            u, rn, it = s
            r = residual(u, u_prev, nu)
            Jv = lambda v: jax.jvp(lambda uu: residual(uu, u_prev, nu),
                                   (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=bc.bf.LIN_MAXITER)
            ok = jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            rn2 = jnp.linalg.norm(residual(u2, u_prev, nu))
            # keep the better iterate (a diverging inexact step must not
            # destroy the rollout; counted honestly either way)
            better = jnp.isfinite(rn2) & (rn2 < rn)
            return (jnp.where(better, u2, u), jnp.where(better, rn2, rn), it + 1)

        rn0 = jnp.linalg.norm(residual(u_prev, u_prev, nu))
        u, rn, it = jax.lax.while_loop(cond, body, (u_prev, rn0, 0))
        return u, rn / u_scale, it

    def roll(u0, nu):
        def body(u, _):
            u2, rr, it = step(u, nu)
            return u2, (u2, rr, it)
        _, (snaps, rrs, its) = jax.lax.scan(body, u0, None, length=bc.NUM_STEPS)
        return jnp.concatenate([u0[None], snaps], axis=0), rrs, its

    return jax.jit(roll)


def run_cell(cell, cell_idx, data, baselines, report_common):
    K, R = cell["K"], cell["R"]
    M_MODES = 4 * K
    MQ = 4 * M_MODES
    tag = f"K{K}_R{R}_nff{cell['nff']}_ffs{cell['ffs']:g}"
    OUT = os.path.join(OUT_DIR, f"sep_burgers_{tag}.json")
    CKPT = os.path.join(OUT_DIR, f"sep_burgers_N{N}_{tag}.pkl")
    t_cell = time.time()
    (S_tr, pick, t0_states_mask, U_test, nu_test, interior, coords, coords_int,
     n2, T, fingerprint, n_states_total) = data
    fom_roll, nt_pairs, nt_rollers, base_rows = baselines
    full_baseline = cell_idx in BASELINE_CELLS
    dev = jax.devices()[0]
    report = dict(config=dict(
        pde="burgers2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=cell["steps"],
        lr=LR, n_ff=cell["nff"], ff_scale=cell["ffs"], g_hidden=G_HID,
        h_hidden=H_HID, g_layers=G_LAY, h_layers=H_LAY,
        n_test=N_TEST, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
        num_steps=bc.NUM_STEPS, dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0,
        data_seed=bc.SEED, test_seed=bc.TEST_SEED, max_snaps=MAX_SNAPS,
        ic_topk=IC_TOPK, ic_budget=bc.IC_BUDGET,
        newton_ladder=NEWTON_LADDER, newton_max=NEWTON_MAX,
        time_reps=TIME_REPS, time_warm=TIME_WARM,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"incumbent weak upwind M={M_MODES}, NNLS-EQ m={MQ} "
                  f"(capped candidate pool, ctol_eq recipe)",
        solver="blat_common lm_step_jit / rollout_jit (incumbent), both arms",
        timing="END-TO-END: IC fit + rollout + full-grid decode of 51 states; "
               "split reported; balanced alternating order; raw reps retained; "
               "error from a timed invocation's decoded fields",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local"),
        **report_common), rows=[], complete=False)
    report["data"] = dict(fingerprint=fingerprint, n_states_total=n_states_total,
                          n_states_trained=int(S_tr.shape[0]),
                          t0_states_kept=int(t0_states_mask.sum()),
                          subsample="all t=0 states + random rest (seeded)")

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, S_tr, K, R,
        steps=cell["steps"], lr=LR, tag=f"burgers N={N} {tag}",
        n_ff=cell["nff"], ff_scale=cell["ffs"], g_hidden=G_HID,
        h_hidden=H_HID, g_layers=G_LAY, h_layers=H_LAY)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    # trust region exactly as the accepted recipe: factor x training radius
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)

    # ------------------ EQ (capped pool) + incumbent weak ops ----------------
    rng = np.random.default_rng(SEED0)
    kx, ky, Phi, lam, _lamc = bc.test_modes(N, M_MODES)
    xy_int = jnp.asarray(coords_int)
    u_full_int = jax.jit(lambda z: dec(z, xy_int))
    adv_full = jax.jit(lambda u: bc.upwind_adv_field(u, N))
    cand_pos = ctol_eq.candidate_pool(interior.size)
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    keep, wq_, eq_info = ctol_eq.eq_fit_burgers(
        u_full_int, adv_full, Phi, cand_pos, Z_tr[eq_pick], K, MQ,
        f"sep burgers N={N} k={K} M={M_MODES} m={MQ}", bc.nnls_capped)
    colloc = dict(kind="grid", idx=interior[cand_pos[keep]], w=wq_, info=eq_info)
    report["eq"] = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    ops_ref = bc.make_weak_ops(dec, N, colloc, kind="weak", M=M_MODES,
                               solver="lspg")

    # ------------------ cached fast ops (same formulas, banks cached) --------
    idx = np.asarray(colloc["idx"])
    m = idx.size
    w_q = jnp.asarray(colloc["w"], dtype=F64)
    pos = np.searchsorted(interior, idx)
    assert np.all(interior[pos] == idx)
    Phi_q = jnp.asarray(Phi[pos]) * w_q[:, None]                  # (m, M')
    lam_j = jnp.asarray(lam, dtype=F64)
    st = bc.stencil_indices(idx, N)                               # (m, 5)
    G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)
    G_all = dec.feat_at(coords)                                   # (n^2, r)
    h_fn = dec.head_fn()
    dx = 1.0 / (N - 1)

    def u_and_N_fast(z):
        us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
        c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
        return c, c * (ux + uy)

    def prev_of_fast(z):
        return jnp.einsum("mr,r->m", G_st[:, 0, :], h_fn(z))

    def r_w_fast(z, prev_c, nu):
        wt = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
        u, Nu = u_and_N_fast(z)
        pu = Phi_q.T @ u
        return wt * (Phi_q.T @ (u - prev_c) + bc.DT * ((Phi_q.T @ Nu)
                                                       + nu * lam_j * pu))

    def d_c_fast(z):
        return u_and_N_fast(z)[0]

    def rJ_fast(z, prev_c, nu):
        return (r_w_fast(z, prev_c, nu), jax.jacfwd(r_w_fast)(z, prev_c, nu),
                Phi_q.T @ jax.jacfwd(d_c_fast)(z))

    def full_fast(z):
        return G_all @ h_fn(z)

    ops_fast = bc._finish_ops(rJ_fast, r_w_fast, prev_of_fast, full_fast, m,
                              "lspg")
    ops_fast["M"] = M_MODES
    ops_fast["tol_scale"] = float(np.sqrt(interior.size))
    ops_fast["colloc_used"] = colloc

    # ------------------ GATE 0: identity of the two arms ---------------------
    g0 = []
    for _ in range(5):
        zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))]
                         + 0.05 * rng.standard_normal(K))
        zp = jnp.asarray(Z_tr[rng.integers(len(Z_tr))])
        prev_c = ops_ref["prev_of"](zp)
        nu = float(np.median(nu_test))
        ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
        rb, Jb, _ = ops_fast["rJ"](zt, prev_c, nu)
        pa = ops_ref["prev_of"](zt); pb = ops_fast["prev_of"](zt)
        g0.append(max(
            float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
            float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300)),
            float(jnp.max(jnp.abs(pa - pb)) / (jnp.max(jnp.abs(pa)) + 1e-300))))
    report["gate0_max_rel_dev"] = float(np.max(g0))
    sc.log(f"  GATE 0 (incumbent vs cached weak r/J identity): "
           f"max rel dev {np.max(g0):.2e}")
    assert np.max(g0) < 1e-12, "gate 0 failed"
    save()

    # ------------------ IC fit: jitted, nearest-t0 inits ---------------------
    # bank: DECODED training t=0 states (training data only; u0 is known data)
    t0_codes = jnp.asarray(Z_tr[t0_states_mask]) if t0_states_mask.any() \
        else jnp.asarray(Z_tr[:64])
    zbar = jnp.asarray(Z_tr.mean(0))
    coords_j = jnp.asarray(coords)
    interior_j = jnp.asarray(interior)

    def make_ic_fit(dec_full_one):
        """dec_full_one(z) -> (n^2,) full-grid decode for ONE latent.  The
        decoded-t0 bank (n_t0, n^2; ~1.2 GB at N=512) is an EXPLICIT argument
        -- never a jit-captured constant (CLAUDE.md landmine)."""
        def ic_fit(u0, bank, bank_sq):
            d2 = bank_sq - 2.0 * (bank @ u0)
            top = jax.lax.top_k(-d2, IC_TOPK)[1]
            inits = jnp.concatenate([zbar[None, :], t0_codes[top]], axis=0)
            lm = ctol_tol.lm_tau_generic(lambda z: dec_full_one(z) - u0, K,
                                         bc.IC_BUDGET)
            z_a, rn_a, rn0_a, nJ_a, nr_a, acc_a, rej_a, att_a, rsn_a = \
                jax.vmap(lambda z0: lm(z0, 0.0))(inits)
            best = jnp.argmin(jnp.where(jnp.isfinite(rn_a), rn_a, jnp.inf))
            return (z_a[best], rn_a[best] / jnp.linalg.norm(u0), best,
                    rn_a, rsn_a, nJ_a)
        return ic_fit

    dec_full_cached = lambda z: G_all @ h_fn(z)
    dec_full_mesh = lambda z: dec(z, coords_j)
    ic_fit_cached = jax.jit(make_ic_fit(dec_full_cached))
    ic_fit_mesh = jax.jit(make_ic_fit(dec_full_mesh))
    # one shared bank (decoded t0 fields; identical between arms to ~1e-15,
    # used ONLY for init selection)
    ic_bank = jax.lax.map(jax.jit(dec_full_cached), t0_codes)
    ic_bank_sq = jnp.sum(ic_bank * ic_bank, axis=1)
    ic_bank.block_until_ready()

    # ------------------ end-to-end jitted paths ------------------------------
    def make_paths(ops, ic_fit, dec_all):
        """dec_all(Z (T+1,K)) -> (T+1, n^2) full-grid fields."""
        def us_of(u0):
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            return jnp.full((bc.NUM_STEPS,),
                            bc.GN_TOL * u_scale * ops["tol_scale"], dtype=F64)

        def rodec(z0, nu, us):
            Z, rns, nJs, reasons = ops["rollout_jit"](z0, nu, us, bc.GN_BUDGET)
            fields = dec_all(jnp.concatenate([z0[None, :], Z], axis=0))
            return fields, Z, rns, nJs, reasons

        def e2e(u0, nu, bank, bank_sq):
            z0, ic_rel, ic_best, ic_rns, ic_rsn, ic_nJ = ic_fit(u0, bank,
                                                               bank_sq)
            fields, Z, rns, nJs, reasons = rodec(z0, nu, us_of(u0))
            return (fields, z0, ic_rel, ic_best, ic_rns, ic_rsn, ic_nJ,
                    rns, nJs, reasons)
        return jax.jit(e2e), jax.jit(rodec), jax.jit(us_of)

    dec_all_cached = lambda Z: (jax.vmap(h_fn)(Z)) @ G_all.T
    dec_all_mesh = lambda Z: jax.lax.map(lambda z: dec(z, coords_j), Z)
    e2e_cached, rodec_cached, us_of = make_paths(ops_fast, ic_fit_cached,
                                                 dec_all_cached)
    e2e_mesh, rodec_mesh, _ = make_paths(ops_ref, ic_fit_mesh, dec_all_mesh)

    # ------------------ per-trajectory: error (timed-path outputs) -----------
    n_test = min(N_TEST, U_test.shape[0])
    arms = dict(cached=(e2e_cached, rodec_cached, ic_fit_cached, ops_fast),
                meshfree=(e2e_mesh, rodec_mesh, ic_fit_mesh, ops_ref))
    arm_rows = {a: [] for a in arms}
    for i in range(n_test):
        u0 = jnp.asarray(U_test[i, 0])
        nu = float(nu_test[i])
        Ut = np.asarray(U_test[i])
        utn = np.linalg.norm(Ut, axis=1)
        for arm, (e2e, rodec_, ic_fit, ops) in arms.items():
            out = e2e(u0, nu, ic_bank, ic_bank_sq)
            (fields, z0, ic_rel, ic_best, ic_rns, ic_rsn, ic_nJ,
             rns, nJs, reasons) = [np.asarray(o) for o in out]
            per = np.linalg.norm(fields - Ut, axis=1) / utn
            # cross-check vs the incumbent untimed rollout from the same z0
            u_scale = float(np.sqrt(np.mean(np.asarray(u0)[interior] ** 2)))
            ro = bc.rollout(dec if arm == "meshfree" else dec, N, ops,
                            jnp.asarray(z0), nu, u_scale, U_true=U_test[i])
            dev_fields = float(np.max(np.abs(
                fields[:ro["fields"].shape[0]] - ro["fields"])))
            row = dict(traj=i, nu=nu,
                       ic_rel=float(ic_rel), ic_best_init=int(ic_best),
                       ic_rns=[float(v) for v in ic_rns],
                       ic_reasons=reason_hist(ic_rsn, IC_REASONS),
                       ic_jacs=[int(v) for v in ic_nJ],
                       traj_rel=float(np.mean(per)),
                       traj_rel_frob=float(np.linalg.norm(fields - Ut)
                                           / np.linalg.norm(Ut)),
                       per_time_max=float(np.max(per)),
                       jac_total=int(np.sum(nJs)),
                       stop_reasons=reason_hist(reasons, STEP_REASONS),
                       n_done_incumbent=int(ro["n_done"]),
                       incumbent_traj_rel=ro.get("traj_rel"),
                       timed_vs_incumbent_max_abs_dev=dev_fields)
            arm_rows[arm].append(row)
            sc.log(f"   {arm:8s} traj {i}: ic {float(ic_rel):.2e} "
                   f"(init {int(ic_best)})  err {row['traj_rel']:.3e}  "
                   f"jac {row['jac_total']}  reasons {row['stop_reasons']}  "
                   f"dev(incumbent) {dev_fields:.1e}")
    for arm in arms:
        rows = arm_rows[arm]
        errs = [r_["traj_rel"] for r_ in rows if np.isfinite(r_["traj_rel"])]
        report["rows"].append(dict(
            pde="burgers2d", method=f"sep_{arm}", N=N, k=K, r=R, M=M_MODES,
            m=int(m), err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            n_blowups=int(sum(r_["n_done_incumbent"] < bc.NUM_STEPS
                              for r_ in rows)),
            per_traj=rows, n_test=n_test))
    save()

    # ------------------ classical baselines (built once per job) -------------
    report["fom_newton_tol"] = base_rows
    report["fom_truth_note"] = ("fom_truth arm is the truth-generating fixed-8-"
                                "Newton rollout: OVER-SOLVED, never a headline "
                                "comparator; its error vs the test data is 0 "
                                "by construction")
    save()

    # ------------------ balanced timing block --------------------------------
    timing = dict(reps=TIME_REPS, warm=TIME_WARM,
                  schedule="sep_common.time_multi alternating forward/reverse",
                  full_baseline=full_baseline, per_traj={})
    for i in range(n_test):
        u0 = jnp.asarray(U_test[i, 0])
        U0b = jnp.asarray(U_test[i:i + 1, 0])
        nub = jnp.asarray(nu_test[i:i + 1])
        nu = float(nu_test[i])
        us = us_of(u0)
        z0_cached = e2e_cached(u0, nu, ic_bank, ic_bank_sq)[1]
        z0_mesh = e2e_mesh(u0, nu, ic_bank, ic_bank_sq)[1]
        z0_cached.block_until_ready(); z0_mesh.block_until_ready()
        ctol_tol.burn_in(1.0)
        thunks = {
            "rom_cached_e2e": lambda: e2e_cached(u0, nu, ic_bank, ic_bank_sq)[0]
                .block_until_ready(),
            "rom_cached_icfit": lambda: ic_fit_cached(u0, ic_bank, ic_bank_sq)[0]
                .block_until_ready(),
            "rom_cached_rollout_decode":
                lambda: rodec_cached(z0_cached, nu, us)[0].block_until_ready(),
            "rom_meshfree_e2e": lambda: e2e_mesh(u0, nu, ic_bank, ic_bank_sq)[0]
                .block_until_ready(),
            "fom_truth_newton8":
                lambda: fom_roll(U0b, nub)[0].block_until_ready(),
        }
        if full_baseline:
            for p in nt_pairs:
                nm = f"fom_newton_ntol{p[0]:g}_lin{p[1]:g}"
                thunks[nm] = (lambda _p=p: nt_rollers[_p](u0, nu)[0]
                              .block_until_ready())
        raw, order = sc.time_multi(thunks, reps=TIME_REPS, warm=TIME_WARM)
        timing["per_traj"][str(i)] = {nm: [t * 1e3 for t in ts]
                                      for nm, ts in raw.items()}
        if i == 0:
            timing["order_traj0"] = order
        sc.log("   timed traj %d: " % i + "  ".join(
            f"{nm}={np.median(ts)*1e3:.1f}ms"
            for nm, ts in raw.items()))
    meds = {}
    for nm in timing["per_traj"]["0"]:
        per_tr = [float(np.median(timing["per_traj"][str(i)][nm]))
                  for i in range(n_test)]
        meds[nm] = dict(median_ms=float(np.median(per_tr)),
                        per_traj_median_ms=per_tr)
    timing["summary"] = meds
    report["timing"] = timing
    # attach the end-to-end medians to the arm rows
    for row in report["rows"]:
        if row["method"] == "sep_cached":
            row["e2e_ms_median"] = meds["rom_cached_e2e"]["median_ms"]
            row["icfit_ms_median"] = meds["rom_cached_icfit"]["median_ms"]
            row["rollout_decode_ms_median"] = \
                meds["rom_cached_rollout_decode"]["median_ms"]
        if row["method"] == "sep_meshfree":
            row["e2e_ms_median"] = meds["rom_meshfree_e2e"]["median_ms"]
    report["complete"] = True
    report["cell_seconds"] = time.time() - t_cell
    save()
    sc.log(f"CELL DONE burgers {tag} [{time.time()-t_cell:.0f}s] -> {OUT}")


def main():
    dev = jax.devices()[0]
    cells = parse_cells()
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} cells={cells} seed={SEED0}")
    t_all = time.time()

    # ------------------ data (regenerated from seed, ONCE per job) -----------
    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)            # (n_traj, T, n^2)
    U_test = np.asarray(d["U_test"], dtype=np.float64)  # (N_TEST_all, T, n^2)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    coords_int = coords[interior]
    fingerprint = bc.data_fingerprint(U)
    max_fom_res = d.get("max_fom_rel_residual")

    # subsample training states BEFORE the interior copy; ALWAYS keep all t=0
    # states (they are the IC-fit init bank)
    n_states = n_traj * T
    rng = np.random.default_rng(SEED0)
    t0_idx = np.arange(n_traj) * T
    if n_states > MAX_SNAPS:
        rest = np.setdiff1d(np.arange(n_states), t0_idx)
        n_rand = max(MAX_SNAPS - t0_idx.size, 0)
        pick = np.sort(np.concatenate(
            [t0_idx, rng.choice(rest, n_rand, replace=False)]))
    else:
        pick = np.arange(n_states)
    rows_flat = U.reshape(n_states, n2)
    S_tr = np.empty((pick.size, interior.size), dtype=np.float64)
    for j, p in enumerate(pick):
        S_tr[j] = rows_flat[p][interior]
    t0_states_mask = (pick % T) == 0
    del rows_flat, U, d
    sc.log(f"  training states: {S_tr.shape[0]} of {n_states} "
           f"({int(t0_states_mask.sum())} t=0 states kept)")

    data = (S_tr, pick, t0_states_mask, U_test, nu_test, interior, coords,
            coords_int, n2, T, fingerprint, n_states)
    report_common = dict(max_fom_rel_residual=max_fom_res)

    # classical baselines: built + error-passed ONCE per job (cell-independent)
    n_test = min(N_TEST, U_test.shape[0])
    fom_roll, residual = bc.bf.make_rollout(N)
    nt_pairs = []
    for tok in NEWTON_LADDER.split(","):
        a, b = tok.split(":")
        nt_pairs.append((float(a), float(b)))
    nt_rollers = {p: make_newton_tol_rollout(N, p[0], p[1], residual)
                  for p in nt_pairs}
    base_rows = []
    for p in nt_pairs:
        rows = []
        for i in range(n_test):
            u0 = jnp.asarray(U_test[i, 0])
            snaps, rrs, its = nt_rollers[p](u0, float(nu_test[i]))
            F = np.asarray(snaps)
            Ut = np.asarray(U_test[i])
            per = np.linalg.norm(F - Ut, axis=1) / np.linalg.norm(Ut, axis=1)
            rows.append(dict(traj=i, traj_rel=float(np.mean(per)),
                             traj_rel_frob=float(np.linalg.norm(F - Ut)
                                                 / np.linalg.norm(Ut)),
                             newton_iters_total=int(np.sum(np.asarray(its))),
                             newton_iters_per_step_mean=float(
                                 np.mean(np.asarray(its))),
                             achieved_rel_res_max=float(
                                 np.max(np.asarray(rrs)))))
        base_rows.append(dict(method="fom_newton_tol", ntol=p[0], lin_tol=p[1],
                              err_traj_rel_mean=float(np.mean(
                                  [r_["traj_rel"] for r_ in rows])),
                              err_traj_rel_max=float(np.max(
                                  [r_["traj_rel"] for r_ in rows])),
                              per_traj=rows))
        sc.log(f"   FOM newton ntol={p[0]:.0e} lin={p[1]:.0e}: err "
               f"{base_rows[-1]['err_traj_rel_mean']:.3e}  iters/step "
               f"{np.mean([r_['newton_iters_per_step_mean'] for r_ in rows]):.1f}")
    baselines = (fom_roll, nt_pairs, nt_rollers, base_rows)

    for ci, cell in enumerate(cells):
        run_cell(cell, ci, data, baselines, report_common)
    sc.log(f"ALL CELLS DONE burgers [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
