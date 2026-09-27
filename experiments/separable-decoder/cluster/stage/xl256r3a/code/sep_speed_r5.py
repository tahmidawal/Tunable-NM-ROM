"""ROUND 5 -- PROFILE AND SPEED, on an ALREADY-TRAINED OR REFINED checkpoint.

`sep_speed_r4.py` with one capability added, so that a decoder produced by the
round-5 coefficient-space refinement can be measured end to end on exactly the
round-4 protocol.  A refined decoder's codes may correspond to a TRAJ_FIT
subset, or to a draw with appended trajectories, neither of which is
reconstructible from (max_snaps, t_early, n_traj).  So when the checkpoint
carries `cfg["hfit_pick"]`, this driver READS the state ids the codes belong to
instead of reconstructing them, and builds the trajectory parameters as the
canonical seed-0 draw plus the recorded appended draw.  With `hfit_pick`
absent it reproduces `sep_speed_r4.py` exactly.

--- original round-4 header follows ---

ROUND 4 -- PROFILE AND SPEED, on an ALREADY-TRAINED checkpoint.

Round 3 measured, at N=1024 (job 2835794, runs/push_r3d):

    end-to-end 73.4 ms = 26.2 ms encoder IC fit + ~45.8 ms rollout
                          + 1.4 ms full-grid decode
    rollout: 256 Jacobians over 50 steps (~5.1 LM iterations/step),
             399/400 steps stop on 'stalled', 0 on 'tol'
    against the standardised classical ladder the ROM is 0.86x (it LOSES),
    but batched over 8 queries it is 1.31x (it wins)

This driver answers "where does the time actually go" WITH MEASUREMENTS and
then tests speed variants.  It TRAINS NOTHING: it loads a committed .pkl and
varies only the solve path, so a full pass is minutes rather than hours.

What it produces:

 1. COST BREAKDOWN by ablation -- e2e / IC / rollout / decode, and inside one
    LM iteration: the h-track MLP, its jacfwd Jacobian, the G_st bank
    products, the weak projection, and the K x K normal-equation solve.  Each
    is timed with the same balanced harness as every other number in this
    project.
 2. ROOFLINE EVIDENCE, not inference: XLA `cost_analysis()` (FLOPs and bytes
    accessed) plus a fusion count for the compiled step and rollout, so
    "launch-bound vs bandwidth-bound vs compute-bound" is settled by the
    compiled artefact rather than by arithmetic on the back of an envelope.
 3. SPEED VARIANTS, each gated and each reported with its Jacobian count:
      ic_gram  -- the IC latent fit moved into the R-dimensional Gram space.
                  ||G h(z) - u0||^2 = h^T (G^T G) h - 2 h^T (G^T u0) + const,
                  so with Gram = L L^T precomputed OFFLINE and b = G^T u0
                  computed ONCE online (that cost is inside the timed region),
                  the LM residual is  L^T h(z) - L^{-1} b.  Its Gauss-Newton
                  system  J~^T J~ = (dh/dz)^T Gram (dh/dz)  and gradient
                  J~^T r~ = (dh/dz)^T (Gram h - b) are ALGEBRAICALLY IDENTICAL
                  to the full-grid ones, and acceptance (strict decrease) is
                  equivalent because the two objectives differ by a constant.
                  Only the *stopping* test sees a different ||r||, so this is
                  reported as a VARIANT with its latent deviation from the
                  incumbent full-grid fit recorded, never as a free lunch.
      stall    -- the LM stall threshold (relative decrease below which a step
                  gives up).  The incumbent value is 1e-12; since no step ever
                  reaches 'tol', that spends iterations for nothing.
      budget   -- a cap on LM iterations per step.
      extrap   -- warm-start extrapolation order for the latent.
 4. A BATCH SWEEP (1,2,4,...) for the best ROM arm and for the matched
    classical configuration, i.e. the amortization curve.
 5. The standardised classical (newton_tol, lin_tol) ladder in the SAME job on
    the SAME GPU, matched-accuracy selection, and a paired AB/BA head-to-head.

Unchanged from round 3: incumbent discretization, residual and Jacobian
definitions; gate 0 (<=1e-12) per EQ set; no test truth in any solve path;
end-to-end timing including the IC fit and the full-grid decode with all raw
repetitions retained.  Every variant is additionally gated: the configurable
LM step must reproduce the incumbent step EXACTLY at the incumbent settings.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc
import sep_solvers as ss

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402
from sep_burgers_r3 import make_tol_newton_pc, build_test_full   # noqa: E402

F64 = jnp.float64

CKPT = os.environ["CKPT"]
N = int(os.environ.get("N", "256"))
N_TEST = int(os.environ.get("N_TEST", "8"))
EQ_MS = [int(v) for v in os.environ.get("EQ_MS", "64").split(",")]
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "5"))
WARM = int(os.environ.get("WARM", "2"))
PAIR_REPS = int(os.environ.get("PAIR_REPS", "3"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
IC_ENC_BUDGET = int(os.environ.get("IC_ENC_BUDGET", "50"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
STALLS = [float(v) for v in os.environ.get(
    "STALLS", "1e-12,1e-6,1e-4,1e-3,1e-2").split(",")]
BUDGETS = [int(v) for v in os.environ.get("BUDGETS", "30,5,3,2,1").split(",")]
EXTRAPS = [float(v) for v in os.environ.get("EXTRAPS", "0,1.0").split(",")]
BATCHES = [int(v) for v in os.environ.get("BATCHES", "1,2,4,8").split(",")]
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4").split(",")]
LIN_FRACS = [float(v) for v in os.environ.get("LIN_FRACS", "0.05,0.5").split(",")]
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "0"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}


def cost_of(fn, *args, static=()):
    """FLOPs / bytes accessed / fusion count of the COMPILED artefact."""
    try:
        low = jax.jit(fn, static_argnums=static).lower(*args)
        comp = low.compile()
        ca = comp.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        txt = comp.as_text()
        return dict(flops=float(ca.get("flops", 0.0)),
                    bytes_accessed=float(ca.get("bytes accessed", 0.0)),
                    n_fusion=txt.count("fusion("),
                    n_hlo_lines=txt.count("\n"))
    except Exception as e:                      # never let profiling kill a run
        return dict(error=f"{type(e).__name__}: {e}")


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} ROUND4-SPEED N={N} "
           f"ckpt={os.path.basename(CKPT)}")
    t_all = time.time()
    params, Z_tr, cfg = sc.load_pkl(CKPT)
    K, R = int(cfg["k"]), int(cfg["r"])
    assert int(cfg["N"]) == N, f"ckpt N={cfg['N']} != N={N}"
    dec = sc.SeparableDecoder(params, K, R)
    h_fn = dec.head_fn()
    OUT = f"{OUT_PREFIX}sep_speed_r5_{os.environ.get('OUT_TAG', f'N{N}_K{K}_R{R}')}.json"

    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    n_i2 = interior.size
    T = bc.NUM_STEPS + 1
    feat_chunk = FEAT_CHUNK or (0 if N <= 512 else 131072)

    report = dict(config=dict(
        pde="burgers2d", round=5, kind="profile+speed", N=N, k=K, r=R,
        ckpt=os.path.basename(CKPT), ckpt_cfg=cfg, n_test=N_TEST, eq_Ms=EQ_MS,
        step_tol=STEP_TOL, extrap=EXTRAP, stalls=STALLS, budgets=BUDGETS,
        extraps=EXTRAPS, batches=BATCHES, newton_tols=NEWTON_TOLS,
        lin_fracs=LIN_FRACS, reps=REPS, warm=WARM, pair_reps=PAIR_REPS,
        gn_budget=bc.GN_BUDGET, ic_enc_budget=IC_ENC_BUDGET,
        num_steps=bc.NUM_STEPS, dt=bc.DT, seed=SEED0, test_seed=bc.TEST_SEED,
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        gates={}, profile={}, variants=[], rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    U_test, nu_test, worst_res_test = build_test_full(N, N_TEST, sc.log)
    report["data"] = dict(n_test=int(N_TEST),
                          max_fom_rel_residual_test=worst_res_test)
    save()

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)
    zbar = Z_tr.mean(0)

    G_all = dec.feat_at(coords, chunk=feat_chunk)
    interior_j = jnp.asarray(interior)
    G_int = G_all[interior_j]
    coords_j = jnp.asarray(coords)
    coords_int = coords[interior]

    # ---------------- EQ sets + cached ops + gate 0 -------------------------
    rng = np.random.default_rng(SEED0)
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    dx = 1.0 / (N - 1)
    cand_pos = ctol_eq.candidate_pool(n_i2, cap=EQ_CAND_CAP)
    bank_apply = jax.jit(lambda Gb, z: Gb @ h_fn(z))
    u_full_int = lambda z: bank_apply(G_int, z)
    _zb = jnp.asarray(Z_tr[eq_pick[0]])
    _a, _b = u_full_int(_zb), dec(_zb, jnp.asarray(coords_int))
    dv = float(jnp.max(jnp.abs(_a - _b)) / (jnp.max(jnp.abs(_b)) + 1e-300))
    report["gates"]["eq_bank_vs_meshfree"] = dv
    assert dv < 1e-12
    adv_full = jax.jit(lambda uf: bc.upwind_adv_field(uf, N))
    eq_ops = {}
    for Mi in EQ_MS:
        name = "ctrl" if Mi == EQ_MS[0] else f"M{Mi}"
        kx, ky, Phi, lam, _ = bc.test_modes(N, Mi)
        keep, wq_np, eq_info = ctol_eq.eq_fit_burgers(
            u_full_int, adv_full, np.asarray(Phi), cand_pos, Z_eq, K, 4 * Mi,
            f"sep speed r4 N={N} k={K} M={Mi} m={4*Mi}", bc.nnls_capped)
        cl = dict(kind="grid", idx=interior[cand_pos[keep]], w=wq_np,
                  info=eq_info)
        idx = np.asarray(cl["idx"])
        m = idx.size
        w_q = jnp.asarray(cl["w"], dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx)
        Phi_q = jnp.asarray(np.asarray(Phi)[pos]) * w_q[:, None]
        lam_j = jnp.asarray(lam, dtype=F64)
        st = bc.stencil_indices(idx, N)
        G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)
        del Phi

        def mk(G_st=G_st, Phi_q=Phi_q, lam_j=lam_j):
            def u_and_N_fast(z):
                us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
                c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2], us[:, 3],
                                     us[:, 4])
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
                return c, c * (ux + uy)

            def prev_of_fast(z):
                return jnp.einsum("mr,r->m", G_st[:, 0, :], h_fn(z))

            def r_w_fast(z, prev_c, nu):
                wt = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                u, Nu = u_and_N_fast(z)
                pu = Phi_q.T @ u
                return wt * (Phi_q.T @ (u - prev_c)
                             + bc.DT * ((Phi_q.T @ Nu) + nu * lam_j * pu))

            def d_c_fast(z):
                return u_and_N_fast(z)[0]

            def rJ_fast(z, prev_c, nu):
                return (r_w_fast(z, prev_c, nu),
                        jax.jacfwd(r_w_fast)(z, prev_c, nu),
                        Phi_q.T @ jax.jacfwd(d_c_fast)(z))

            def full_fast(z):
                return G_st[:, 0, :] @ h_fn(z)
            return (r_w_fast, rJ_fast, prev_of_fast, full_fast, u_and_N_fast,
                    d_c_fast)
        (r_w_f, rJ_f, prev_f, full_f, uN_f, dc_f) = mk()
        ops_fast = bc._finish_ops(rJ_f, r_w_f, prev_f, full_f, m, "lspg")
        ops_fast["M"] = Mi
        ops_fast["tol_scale"] = float(np.sqrt(n_i2))
        ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=Mi, solver="lspg")
        g0 = []
        grng = np.random.default_rng(SEED0 + 50)
        for _ in range(5):
            zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                             + 0.05 * grng.standard_normal(K))
            zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
            prev_c = ops_ref["prev_of"](zp)
            nu = float(np.median(nu_test))
            ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
            rb, Jb, _ = ops_fast["rJ"](zt, prev_c, nu)
            g0.append(max(
                float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
        gate0 = float(np.max(g0))
        sc.log(f"  GATE 0 [{name}]: {gate0:.2e}")
        assert gate0 < 1e-12
        info_rep = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
        info_rep["gate0"] = gate0
        report.setdefault("eq", {})[name] = info_rep
        eq_ops[name] = dict(ops_fast=ops_fast, ops_ref=ops_ref, idx=idx,
                            pos=pos, m=m, r_w=r_w_f, rJ=rJ_f, prev=prev_f,
                            uN=uN_f, dc=dc_f, G_st=G_st, Phi_q=Phi_q)
        save()
    del G_int
    e0 = eq_ops["ctrl"]
    ops0 = e0["ops_fast"]
    idx0_j = jnp.asarray(e0["idx"])
    G_q0 = dec.feat_at(coords[e0["idx"]])
    m0 = e0["m"]

    # ---------------- IC encoder, retrained on t=0 pairs --------------------
    # The checkpoint stores Z_tr but not the training fields.  The t=0 states
    # are ANALYTIC (`blob_ic`), and the state pick is reproducible from SEED0,
    # so the encoder is retrained on exactly the trajectories the decoder was
    # trained on, using training data only.  Its achieved IC error is reported
    # next to round 3's so the comparison is auditable.
    if cfg.get("hfit_pick") is not None:
        pick = np.asarray(cfg["hfit_pick"], dtype=np.int64)
        report["config"]["pick_source"] = "cfg[hfit_pick] (round-5 refined)"
    else:
        cfg_ms = int(cfg.get("max_snaps", 16384))
        cfg_te = int(cfg.get("t_early", 5))
        n_traj = int(cfg.get("n_traj") or (bc.bf.N_TRAIN + bc.bf.N_VAL))
        rng2 = np.random.default_rng(SEED0)
        tidx_of = np.arange(n_traj * T) % T
        early = np.nonzero(tidx_of <= cfg_te)[0]
        rest = np.nonzero(tidx_of > cfg_te)[0]
        if early.size >= cfg_ms:
            pick = np.sort(rng2.choice(early, cfg_ms, replace=False))
        else:
            extra = rng2.choice(rest, min(cfg_ms - early.size, rest.size),
                                replace=False)
            pick = np.sort(np.concatenate([early, extra]))
        report["config"]["pick_source"] = "reconstructed from cfg"
    assert pick.size == len(Z_tr), (pick.size, len(Z_tr))
    t0_rows = np.nonzero((pick % T) == 0)[0]
    cx, cy, w, a, nu_all, _ = bc.bf.sample_params(seed=bc.SEED)
    ex_seed = int(cfg.get("hfit_extra_seed", 0) or 0)
    ex_traj = int(cfg.get("hfit_extra_traj", 0) or 0)
    if ex_seed:
        exd = bc.bf.sample_params(seed=ex_seed, m=ex_traj)
        cx = np.concatenate([cx, exd[0]]); w = np.concatenate([w, exd[2]])
        cy = np.concatenate([cy, exd[1]]); a = np.concatenate([a, exd[3]])
        nu_all = np.concatenate([nu_all, exd[4]])
        sc.log(f"  trajectory parameters: canonical 576 + {ex_traj} appended "
               f"from seed {ex_seed} (recorded in the checkpoint)")
    U0_tr = np.stack([bc.bf.blob_ic(N, cx[i], cy[i], w[i], a[i])
                      for i in (pick[t0_rows] // T)])
    X_tr = U0_tr[:, np.asarray(e0["idx"])]
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr[t0_rows], steps=ENC_STEPS,
        tag=f"speed r4 u0->z0 N={N}")
    report["encoder"] = enc_info
    report["encoder"]["note"] = ("retrained on t=0 pairs only (analytic "
                                 "blob_ic + the checkpoint's own codes); "
                                 "training data only, never test data")
    cands_j = jnp.asarray(np.concatenate([Z_tr[t0_rows], zbar[None]], axis=0))
    del U0_tr, X_tr

    # ---------------- IC fits: incumbent full-grid vs Gram-space ------------
    def ic_enc_full(Gb, u0):
        z0 = enc_apply(enc_params, u0[idx0_j])

        def f(z):
            return Gb @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, rn, nJ

    Gram_all = G_all.T @ G_all
    eps_g = 1e-13 * jnp.trace(Gram_all) / R
    L_all = jnp.linalg.cholesky(Gram_all + eps_g * jnp.eye(R, dtype=F64))

    def ic_enc_gram(Gb, u0):
        z0 = enc_apply(enc_params, u0[idx0_j])
        b = Gb.T @ u0                                   # the ONE full pass
        y = jax.scipy.linalg.solve_triangular(L_all, b, lower=True)
        c2 = jnp.maximum(u0 @ u0 - y @ y, 0.0)          # orthogonal remainder

        def f(z):
            return L_all.T @ h_fn(z) - y
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, jnp.sqrt(rn * rn + c2), nJ

    ic_variants = dict(full=ic_enc_full, gram=ic_enc_gram)
    ic_jit = {k: jax.jit(v) for k, v in ic_variants.items()}
    zf, rf, _ = ic_jit["full"](G_all, jnp.asarray(U_test[0, 0], dtype=F64))
    zg, rg, _ = ic_jit["gram"](G_all, jnp.asarray(U_test[0, 0], dtype=F64))
    report["gates"]["ic_gram_vs_full_latent_dev"] = float(
        jnp.linalg.norm(zg - zf) / (1.0 + jnp.linalg.norm(zf)))
    report["gates"]["ic_gram_vs_full_resnorm_rel"] = float(
        abs(rg - rf) / (abs(rf) + 1e-300))
    sc.log(f"  IC gram vs full: |dz|/(1+|z|) "
           f"{report['gates']['ic_gram_vs_full_latent_dev']:.2e}, "
           f"||r|| rel diff {report['gates']['ic_gram_vs_full_resnorm_rel']:.2e}")

    # ---------------- variant LM step must equal the incumbent --------------
    step_ref = ops0["step_jit"]
    step_var = ss.make_step_lspg_var(e0["r_w"], K, stall_rel=1e-12)
    zt = jnp.asarray(Z_tr[3])
    pc = ops0["prev_of"](jnp.asarray(Z_tr[5]))
    nug = float(np.median(nu_test))
    tolg = STEP_TOL * float(np.sqrt(np.mean(U_test[0, 0][interior] ** 2))) \
        * ops0["tol_scale"]
    a_ = step_ref(zt, pc, nug, tolg, bc.GN_BUDGET)
    b_ = step_var(zt, pc, nug, tolg, bc.GN_BUDGET)
    sdev = float(jnp.max(jnp.abs(a_[0] - b_[0])))
    report["gates"]["step_variant_vs_incumbent_at_1e-12"] = sdev
    sc.log(f"  variant LM step vs incumbent at stall=1e-12: max |dz| {sdev:.2e}")
    assert sdev == 0.0, "configurable-stall step is not the incumbent step"
    save()

    # ---------------- PROFILE: ablation of the e2e path ---------------------
    delta0 = jnp.asarray(float(bc.TR_DELTA), dtype=F64)
    u0_j = jnp.asarray(U_test[0, 0], dtype=F64)
    nu0 = float(nu_test[0])
    z0_p, _, _ = ic_jit["full"](G_all, u0_j)
    prev_c0 = ops0["prev_of"](z0_p)
    us_full = jnp.full((bc.NUM_STEPS,), tolg, dtype=F64)
    dhdz = jax.jit(jax.jacfwd(h_fn))
    roll_ex = ss.make_rollout_v2("incumbent", ops=ops0,
                                 num_steps=bc.NUM_STEPS, extrap=EXTRAP)
    decode_j = jax.jit(lambda Gb, Zf: jax.vmap(h_fn)(Zf) @ Gb.T)
    Zf_p = jnp.tile(z0_p[None], (T, 1))

    def blk(f):
        return lambda: jax.block_until_ready(f())

    prof_subs = [
        ("ic_full", blk(lambda: ic_jit["full"](G_all, u0_j))),
        ("ic_gram", blk(lambda: ic_jit["gram"](G_all, u0_j))),
        ("rollout_extrap", blk(lambda: roll_ex(z0_p, nu0, us_full,
                                               bc.GN_BUDGET, delta0, delta0,
                                               delta0))),
        ("decode_full_grid_51", blk(lambda: decode_j(G_all, Zf_p))),
        ("h_track_value", blk(lambda: h_fn(z0_p))),
        ("h_track_jacobian", blk(lambda: dhdz(z0_p))),
        ("bank_value_Gst_h", blk(jax.jit(lambda: jnp.einsum(
            "msr,r->ms", e0["G_st"], h_fn(z0_p))))),
        ("bank_jacobian_Gst_dhdz", blk(jax.jit(lambda: jnp.einsum(
            "msr,rk->msk", e0["G_st"], dhdz(z0_p))))),
        ("weak_residual_r_w", blk(jax.jit(lambda: e0["r_w"](z0_p, prev_c0, nu0)))),
        ("weak_residual_and_jacobian", blk(jax.jit(
            lambda: e0["rJ"](z0_p, prev_c0, nu0)))),
        ("prev_of", blk(jax.jit(lambda: ops0["prev_of"](z0_p)))),
    ]
    for bgt in (1, 2, 4, 8):
        prof_subs.append((f"lm_step_budget{bgt}",
                          blk(lambda _b=bgt: step_ref(z0_p, prev_c0, nu0,
                                                      tolg, _b))))
    ctol_tol.burn_in(1.5)
    raw_p, _ = sc.balanced_time(prof_subs, reps=max(REPS, 7), warm=WARM)
    report["profile"]["ablation_ms"] = {
        k: dict(median_ms=float(np.median(v)) * 1e3,
                raw_s=[float(t) for t in v]) for k, v in raw_p.items()}
    for k, v in sorted(report["profile"]["ablation_ms"].items(),
                       key=lambda kv: -kv[1]["median_ms"]):
        sc.log(f"   PROFILE {k:32s} {v['median_ms']:9.4f} ms")

    report["profile"]["cost_analysis"] = dict(
        lm_step=cost_of(lambda z, p, nu, t: step_ref(z, p, nu, t,
                                                     bc.GN_BUDGET),
                        z0_p, prev_c0, nu0, tolg),
        weak_residual_and_jacobian=cost_of(e0["rJ"], z0_p, prev_c0, nu0),
        h_track_jacobian=cost_of(jax.jacfwd(h_fn), z0_p),
        decode_full_grid_51=cost_of(lambda Gb, Zf: jax.vmap(h_fn)(Zf) @ Gb.T,
                                    G_all, Zf_p),
        ic_full=cost_of(ic_variants["full"], G_all, u0_j),
        ic_gram=cost_of(ic_variants["gram"], G_all, u0_j))
    report["profile"]["shapes"] = dict(
        m_quadrature=int(m0), R=int(R), K=int(K), n_grid=int(N * N),
        G_st_bytes=int(e0["G_st"].size * 8), G_all_bytes=int(G_all.size * 8),
        Phi_q_shape=list(e0["Phi_q"].shape))
    save()

    # ---------------- SPEED VARIANTS ----------------------------------------
    def make_e2e(ic_kind, stall, budget, extrap):
        step_fn = (ops0["step_jit"] if stall == 1e-12
                   else ss.make_step_lspg_var(e0["r_w"], K, stall_rel=stall))
        ops_v = dict(ops0)
        ops_v["step_jit"] = step_fn
        rv = ss.make_rollout_v2("incumbent", ops=ops_v,
                                num_steps=bc.NUM_STEPS, extrap=extrap)
        icf = ic_variants[ic_kind]
        tol_scale = ops0["tol_scale"]

        def e2e(Gb, u0, nu):
            z0, ic_rn, ic_nJ = icf(Gb, u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((bc.NUM_STEPS,), STEP_TOL * u_scale * tol_scale,
                          dtype=F64)
            Z, rns, nJs, reasons, _ = rv(z0, nu, us, budget, delta0, delta0,
                                         delta0)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)
            return (decode_j(Gb, Zfull), z0, Z, rns, nJs, reasons, ic_rn,
                    ic_nJ)
        return jax.jit(e2e)

    specs = [("base", "full", 1e-12, bc.GN_BUDGET, EXTRAP)]
    specs += [(f"gram", "gram", 1e-12, bc.GN_BUDGET, EXTRAP)]
    specs += [(f"gram+stall{s:.0e}", "gram", s, bc.GN_BUDGET, EXTRAP)
              for s in STALLS if s != 1e-12]
    specs += [(f"gram+budget{b}", "gram", 1e-12, b, EXTRAP)
              for b in BUDGETS if b != bc.GN_BUDGET]
    specs += [(f"gram+extrap{e:g}", "gram", 1e-12, bc.GN_BUDGET, e)
              for e in EXTRAPS if e != EXTRAP]
    v_raw = {n_: make_e2e(i_, s_, b_, e_) for n_, i_, s_, b_, e_ in specs}
    v_arm = {n_: (lambda u0, nu, _f=f_: _f(G_all, u0, nu))
             for n_, f_ in v_raw.items()}
    report["config"]["variant_specs"] = [
        dict(name=n_, ic=i_, stall=s_, budget=b_, extrap=e_)
        for n_, i_, s_, b_, e_ in specs]

    fom_roll, _ = bc.bf.make_rollout(N)
    tol_newton = make_tol_newton_pc(N)
    base_cfgs = [(nt, max(nt * lf, 1e-12))
                 for nt in NEWTON_TOLS for lf in LIN_FRACS]

    n_test = min(N_TEST, U_test.shape[0])
    per_arm = {a: [] for a in v_arm}
    base_rows = {c: [] for c in base_cfgs}
    tg_rows = []
    ctol_tol.burn_in(1.5)
    for i in range(n_test):
        u0 = jnp.asarray(U_test[i, 0], dtype=F64)
        nu = float(nu_test[i])
        tnorm = np.linalg.norm(U_test[i], axis=1)
        subs = [(f"v|{a}", lambda _u=u0, _n=nu, _f=v_arm[a]:
                 (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u, _n)))
                for a in v_arm]
        subs += [(f"fom|nt{nt:.0e}|lt{lt:.0e}",
                  lambda _u=u0, _n=nu, _t=nt, _l=lt:
                  (lambda o: (o[0].block_until_ready(), o)[1])(
                      tol_newton(_u, _n, _t, _l)))
                 for (nt, lt) in base_cfgs]
        subs.append(("fom_newton8_truthgen",
                     lambda _u=jnp.asarray(U_test[i:i+1, 0]),
                            _n=jnp.asarray(nu_test[i:i+1]):
                     (lambda o: (o[0].block_until_ready(), o)[1])(
                         fom_roll(_u, _n))))
        raw, res = sc.balanced_time(subs, reps=REPS, warm=WARM)
        for a in v_arm:
            F, z0_t, Z_t, rns, nJs, reasons, ic_rn, ic_nJ = res[f"v|{a}"]
            Fh = np.asarray(F)
            pt = np.linalg.norm(Fh - U_test[i], axis=1) / tnorm
            rn_ = [int(v) for v in np.asarray(reasons)]
            per_arm[a].append(dict(
                traj=i, nu=nu, traj_rel=float(np.mean(pt)),
                per_time=[float(v) for v in pt],
                per_time_max=float(np.max(pt)),
                ic_rel=float(ic_rn) / float(np.linalg.norm(U_test[i, 0])),
                ic_jac=int(ic_nJ), jac_total=int(np.sum(np.asarray(nJs))),
                stop_reasons={REASON_NAMES[r_]: rn_.count(r_) for r_ in set(rn_)},
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                e2e_ms=float(np.median(raw[f"v|{a}"])) * 1e3,
                e2e_raw_s=[float(t) for t in raw[f"v|{a}"]]))
        for (nt, lt) in base_cfgs:
            key = f"fom|nt{nt:.0e}|lt{lt:.0e}"
            snaps, its, rels = res[key]
            pt = np.linalg.norm(np.asarray(snaps) - U_test[i], axis=1) / tnorm
            base_rows[(nt, lt)].append(dict(
                traj=i, traj_rel=float(np.mean(pt)),
                newton_iters_total=int(np.sum(np.asarray(its))),
                time_ms=float(np.median(raw[key])) * 1e3,
                time_raw_s=[float(t) for t in raw[key]]))
        snaps, resid = res["fom_newton8_truthgen"]
        pt = np.linalg.norm(np.asarray(snaps)[:, 0, :] - U_test[i], axis=1) / tnorm
        tg_rows.append(dict(traj=i, traj_rel=float(np.mean(pt)),
                            time_ms=float(np.median(raw["fom_newton8_truthgen"])) * 1e3))
        save()

    for a in v_arm:
        rows = per_arm[a]
        agg = {}
        for r_ in rows:
            for k_, v in r_["stop_reasons"].items():
                agg[k_] = agg.get(k_, 0) + v
        report["variants"].append(dict(
            name=a, err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
            ic_rel_mean=float(np.mean([r_["ic_rel"] for r_ in rows])),
            ic_jac_mean=float(np.mean([r_["ic_jac"] for r_ in rows])),
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            e2e_ms_median=float(np.median([r_["e2e_ms"] for r_ in rows])),
            stop_reasons=agg,
            n_blowups=int(sum(r_["n_finite_steps"] < bc.NUM_STEPS for r_ in rows)),
            per_traj=rows, n_test=n_test))
        v = report["variants"][-1]
        sc.log(f"   VARIANT {a:24s} {v['e2e_ms_median']:8.2f} ms  err "
               f"{v['err_traj_rel_mean']:.3e}  jac {v['jac_total_mean']:.0f}"
               f"  ic_jac {v['ic_jac_mean']:.1f}  {v['stop_reasons']}")
    for (nt, lt) in base_cfgs:
        rows = base_rows[(nt, lt)]
        report["rows"].append(dict(
            pde="burgers2d", method="fom_newton_tol_pc", N=N, newton_tol=nt,
            lin_tol=lt, preconditioner="exact Helmholtz (sine basis)",
            err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
            time_ms_median=float(np.median([r_["time_ms"] for r_ in rows])),
            newton_iters_mean=float(np.mean([r_["newton_iters_total"] for r_ in rows])),
            per_traj=rows, n_test=n_test))
    report["rows"].append(dict(
        pde="burgers2d", method="fom_newton8_truthgen", N=N, oversolved=True,
        err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in tg_rows])),
        time_ms_median=float(np.median([r_["time_ms"] for r_ in tg_rows])),
        per_traj=tg_rows))
    save()

    # ---------------- matched accuracy + paired -----------------------------
    # champion = the FASTEST variant that does not regress accuracy.  Picking
    # the fastest variant outright would let a budget-capped arm win by giving
    # up error, which the matched-accuracy comparison would then reward.
    base_err = [v for v in report["variants"]
                if v["name"] == "base"][0]["err_traj_rel_mean"]
    ok = [v for v in report["variants"]
          if v["err_traj_rel_mean"] <= 1.02 * base_err]
    best = min(ok, key=lambda v: v["e2e_ms_median"])
    report["config"]["champion_rule"] = (
        "fastest variant whose mean trajectory error is within 1.02x of the "
        f"base arm's ({base_err:.4e})")
    champ = best["name"]
    cerr = best["err_traj_rel_mean"]
    cands_m = [c for c in base_cfgs
               if float(np.mean([r_["traj_rel"] for r_ in base_rows[c]])) <= cerr]
    match = min(cands_m, key=lambda c: float(np.median(
        [r_["time_ms"] for r_ in base_rows[c]]))) if cands_m else None
    report["matched_accuracy"] = dict(
        rom_arm=champ, rom_err=cerr, rom_e2e_ms=best["e2e_ms_median"],
        rule="cheapest (newton_tol, lin_tol) at least as accurate as the "
             "fastest ROM variant",
        matched=None if match is None else dict(
            newton_tol=match[0], lin_tol=match[1],
            err=float(np.mean([r_["traj_rel"] for r_ in base_rows[match]])),
            ms=float(np.median([r_["time_ms"] for r_ in base_rows[match]]))))
    if match is not None:
        pairs = []
        for i in range(n_test):
            u0 = jnp.asarray(U_test[i, 0], dtype=F64)
            nu = float(nu_test[i])
            pairs.append(dict(traj=i, **sc.time_pair(
                lambda _u=u0, _n=nu: v_arm[champ](_u, _n)[0].block_until_ready(),
                lambda _u=u0, _n=nu, _t=match[0], _l=match[1]:
                tol_newton(_u, _n, _t, _l)[0].block_until_ready(),
                reps=PAIR_REPS, warm=WARM)))
        report["matched_accuracy"]["paired"] = dict(
            rom_ms=float(np.median([p["a_ms"] for p in pairs])),
            base_ms=float(np.median([p["b_ms"] for p in pairs])),
            per_traj=pairs)
        pa = report["matched_accuracy"]["paired"]
        sc.log(f"  MATCHED paired: ROM[{champ}] {pa['rom_ms']:.2f} ms vs "
               f"tol-Newton(nt={match[0]:.0e},lt={match[1]:.0e}) "
               f"{pa['base_ms']:.2f} ms -> {pa['base_ms']/pa['rom_ms']:.2f}x")
    save()

    # ---------------- batch sweep -------------------------------------------
    report["batch_sweep"] = []
    bmax = max(b for b in BATCHES if b <= n_test)
    for B in [b for b in BATCHES if b <= n_test]:
        u0b = jnp.asarray(U_test[:B, 0], dtype=F64)
        nub = jnp.asarray(nu_test[:B], dtype=F64)
        subs = []
        for a in (champ, "base"):
            if a not in v_raw:
                continue
            be = jax.jit(jax.vmap(v_raw[a], in_axes=(None, 0, 0)))
            subs.append((f"rom|{a}", lambda _b=be, _u=u0b, _n=nub:
                         (lambda o: (o[0].block_until_ready(), o)[1])(
                             _b(G_all, _u, _n))))
        if match is not None:
            br = jax.jit(jax.vmap(lambda u, n_, _t=match[0], _l=match[1]:
                                  tol_newton(u, n_, _t, _l)))
            subs.append((f"fom|nt{match[0]:.0e}|lt{match[1]:.0e}",
                         lambda _b=br, _u=u0b, _n=nub:
                         (lambda o: (o[0].block_until_ready(), o)[1])(
                             _b(_u, _n))))
        ctol_tol.burn_in(1.0)
        raw_b, res_b = sc.balanced_time(subs, reps=REPS, warm=WARM)
        for name in raw_b:
            ts = raw_b[name]
            Fb = np.asarray(res_b[name][0])
            errs = [float(np.mean(np.linalg.norm(Fb[j] - U_test[j], axis=1)
                                  / np.linalg.norm(U_test[j], axis=1)))
                    for j in range(B)]
            ent = dict(batch=B, subject=name,
                       total_ms_median=float(np.median(ts)) * 1e3,
                       amortized_ms=float(np.median(ts)) * 1e3 / B,
                       err_traj_rel_mean=float(np.mean(errs)),
                       raw_s=[float(t) for t in ts])
            report["batch_sweep"].append(ent)
            sc.log(f"   BATCH {B:3d} {name:34s} total "
                   f"{ent['total_ms_median']:8.2f} ms -> "
                   f"{ent['amortized_ms']:7.3f} ms/query  err "
                   f"{ent['err_traj_rel_mean']:.3e}")
        save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE speed r4 N={N} [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
