"""QUADRATURE-FREE exact weak residual for the separable Poisson-2D ROM.

Part B of the 2026-08-27 two-part study (exp/2026-08-27-b1d-poissonqf).
Design provenance: understand/2026-08-26-autonomous-run-handoff.md ("Poisson
goes fully quadrature-free") and understand/2026-08-26-ideation-handoff.md:
with the separable decoder u(x;z) = g(x)^T h(z), the weak Poisson residual is
values-only, so the ENTIRE residual collapses to a precomputed matrix:

    r(z) = Wl * (Phi^T_all (wq_all * G_all h(z))) - f_m
         = Wl * (B h(z)) - f_m,     B = Phi^T_all diag(wq_all) G_all  (M', R)

with G_all the feature bank on ALL interior nodes and wq_all == 1 (the discrete
sine basis is orthonormal in the node inner product).  B is algebraically
IDENTICAL to the full-grid reference -- no empirical-quadrature sample points,
no NNLS fit, no sampling error.  This cell measures that claim (gates) and its
online cost against the incumbent EQ path, over FROZEN decoders (NO training).

Three residual paths through the SAME incumbent trust-LM weak solver
(ctol_tol.lm_tau_poisson), same Wl/alpha/M/f_m conventions as sep_poisson.py:
  FULL : incumbent weak objective on ALL interior nodes, exact weights
         (wq_all=1), cached feature bank G_all (the incumbent 'cached' arm
         evaluated on the full grid; gate F certifies identity to the meshfree
         evaluation).  The reference.
  EQ   : incumbent NNLS-fitted m-node rule, fitted IN-JOB per the incumbent
         recipe (same candidate pool, same NNLS, m=EQ_MQ), cached bank at the
         EQ nodes (the incumbent's fast arm; gate E certifies identity to the
         meshfree evaluation).
  QF   : r(z) = Wl * (B h(z)) - f_m, B precomputed once.  NEW.

GATES (recorded in the JSON; the run aborts on failure):
  gate B : backend gpu (unless ALLOW_CPU=1), f64 on, matmul precision highest.
  gate S : f_m identical between paths -- one shared jitted projection,
           checked against the incumbent pc.weak_source_term to <= 1e-12.
  gate F/E : cached banks reproduce the meshfree weak residual/Jacobian
           to <= 1e-12 relative (the incumbent's gate 0, both point sets).
  gate Q : QF vs FULL residual rel err <= 1e-12 and gradient J^T r rel err
           <= 1e-10 at GATE_NZ seeded random states; at every captured solve
           solution the same residual bound holds and the gradient must pass
           rel <= 1e-10 OR cancellation-aware abs <= 1e-12 (near a solver
           stop g = J^T r loses leading digits to cancellation; the absolute
           form ||g_qf - g_full|| / (||J_full||_F ||r_full||) is the
           c1_abs normalisation of the EQ ladder).

MEASUREMENT RULES (inherited from sep_poisson.py / HANDOFF.md):
  * cost and error from the SAME timed invocation: the timed pipe projects the
    grid source to f_m, solves, decodes the full interior field; errors and
    counters are extracted from the captured outputs of a timed repetition.
  * ALL raw timing repetitions retained per (path, tau, source).
  * balanced order: every source's 3x|TAUS| subjects are swept together,
    forward/reversed on alternate repetitions (sc.balanced_time).
  * TWO cohorts: held-out same-seed sources (indices N_TRAIN.. of the seed-0
    draw) and a fresh-seed cohort (FRESH_SEED, default = the checkpoint's own
    fresh_seed) that no model selection ever saw.
  * one-time setup costs timed separately: EQ NNLS fit vs QF B-matrix build.

Env: CKPT (required), N (asserted == ckpt cfg), M / EQ_MQ / GN_ITERS /
TR_FACTOR / FRESH_SEED (defaults = the checkpoint's own training conventions),
SEED0=0, N_SRC=16 (per cohort), REPS=12, BURN=3 (>=3), TAUS="1e-3,1e-2",
FEAT_CHUNK (0 = auto), GATE_NZ=32, OUT_TAG, OUT_PREFIX, ALLOW_CPU=0.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

import numpy as np

import sep_common as sc

import jax
import jax.numpy as jnp

import pro_common as pc                      # noqa: E402  (path set by sc)
from pro_common import mp                    # noqa: E402
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402

F64 = jnp.float64
HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = os.environ["CKPT"]
N = int(os.environ.get("N", "64"))
N_SRC = int(os.environ.get("N_SRC", os.environ.get("N_TEST", "16")))
REPS = int(os.environ.get("REPS", "12"))
BURN = int(os.environ.get("BURN", "3"))
SEED0 = int(os.environ.get("SEED0", "0"))
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-3,1e-2").split(",")]
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "0"))
GATE_NZ = int(os.environ.get("GATE_NZ", "32"))
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
ALLOW_CPU = int(os.environ.get("ALLOW_CPU", "0"))

PATHS = ("full", "eq", "qf")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit():
    p = os.path.join(HERE, "GIT_COMMIT")
    if os.path.isfile(p):
        return open(p).read().strip()
    try:
        return subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    assert BURN >= 3, f"BURN={BURN} < 3 (warmup floor of the spec)"
    t_all = time.time()

    # ---------------- gate B: backend / precision ----------------------------
    dev = jax.devices()[0]
    mmp = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION")
    sc.log(f"jax_backend={dev.platform} device={dev} "
           f"x64={jax.config.jax_enable_x64} matmul_precision={mmp}")
    assert jax.config.jax_enable_x64, "gate B failed: f64 not active"
    assert mmp == "highest", "gate B failed: JAX_DEFAULT_MATMUL_PRECISION != highest"
    if not ALLOW_CPU:
        assert dev.platform == "gpu", "gate B failed: backend is not gpu"

    # ---------------- frozen checkpoint (NO training anywhere) ---------------
    ck_sha = sha256_of(CKPT)
    params, Z_tr, cfg = sc.load_pkl(CKPT)
    K, R = int(cfg["k"]), int(cfg["r"])
    assert int(cfg["N"]) == N, f"ckpt N={cfg['N']} != env N={N}"
    M_MODES = int(os.environ.get("M", str(cfg.get("M", 4 * K))))
    MQ = int(os.environ.get("EQ_MQ", str(cfg.get("m", 4 * M_MODES))))
    GN_ITERS = int(os.environ.get("GN_ITERS", str(cfg.get("gn_iters", 60))))
    TR_FACTOR = float(os.environ.get("TR_FACTOR", str(cfg.get("tr_factor", 1.0))))
    FRESH_SEED = int(os.environ.get("FRESH_SEED", str(cfg.get("fresh_seed", 777))))
    dec = sc.SeparableDecoder(params, K, R)
    h_fn = dec.head_fn()
    sc.log(f"ckpt={CKPT}\n  sha256={ck_sha}\n  N={N} K={K} R={R} M={M_MODES} "
           f"m={MQ} gn_iters={GN_ITERS} tr_factor={TR_FACTOR} "
           f"seed0={SEED0} fresh_seed={FRESH_SEED} n_src={N_SRC}/cohort "
           f"reps={REPS} burn={BURN} taus={TAUS}")

    tag = OUT_TAG or f"N{N}_K{K}_R{R}"
    OUT = f"{OUT_PREFIX}sep_poisson_qf_{tag}.json"
    if os.path.dirname(OUT):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)

    report = dict(config=dict(
        pde="poisson2d", kind="quadrature_free_exact_weak_residual",
        N=N, k=K, r=R, M=M_MODES, m=MQ, taus=TAUS, n_src_per_cohort=N_SRC,
        gn_iters=GN_ITERS, tr_factor=TR_FACTOR, seed=SEED0,
        fresh_seed=FRESH_SEED, data_seed=mp.SEED, n_train=mp.N_TRAIN,
        cg_tol=mp.CG_TOL, reps=REPS, burn=BURN, gate_nz=GATE_NZ,
        feat_chunk=FEAT_CHUNK,
        ckpt=CKPT, ckpt_sha256=ck_sha, ckpt_cfg=cfg,
        objective=f"weak alpha=1 M={M_MODES} (incumbent sep_poisson.py "
                  f"conventions: Wl=lam^(1-alpha), f_m=lam^-alpha Phi^T f)",
        solver="ctol_tol.lm_tau_poisson (incumbent trust-LM), all 3 paths",
        paths=dict(full="all interior nodes, exact weights (wq=1), cached "
                        "bank G_all (reference)",
                   eq=f"incumbent NNLS-EQ m={MQ} grid nodes, cached bank",
                   qf="r(z)=Wl*(B h(z))-f_m, B=PhiT_all @ (wq_all*G_all) "
                      "precomputed once; NO quadrature points"),
        timing="balanced forward/reversed sweeps per source (sc.balanced_time)"
               ", raw reps retained, error from the captured timed invocation "
               "(source->f_m projection inside the timed region)",
        x64=True, matmul_precision=mmp, backend=dev.platform,
        gpu=getattr(dev, "device_kind", str(dev)),
        jax_version=jax.__version__, git_commit=git_commit(),
        hostname=os.uname().nodename,
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local"),
        allow_cpu=bool(ALLOW_CPU)),
        gates=dict(), setup=dict(), rungs=dict(), rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- sources + truth (regenerated from seeds) ---------------
    grid = pc.Grid(N)
    n_i = grid.n_i
    n_i2 = n_i * n_i
    coords_int = np.asarray(grid.coords_int)
    cx, cy, w, a, _z = mp.sample_params()
    assert mp.N_TRAIN + N_SRC <= len(cx), "N_SRC exceeds the held-out draw"
    Fs_held = np.stack([mp.source_interior(N, cx[mp.N_TRAIN + i],
                                           cy[mp.N_TRAIN + i],
                                           w[mp.N_TRAIN + i], a[mp.N_TRAIN + i])
                        for i in range(N_SRC)])
    cxf, cyf, wf, af, _zf = mp.sample_params(seed=FRESH_SEED, m=N_SRC)
    Fs_fresh = np.stack([mp.source_interior(N, cxf[i], cyf[i], wf[i], af[i])
                         for i in range(N_SRC)])
    Fs = np.concatenate([Fs_held, Fs_fresh])            # (2*N_SRC, n_i, n_i)
    cohort_of = (["heldout_seed0"] * N_SRC + [f"fresh_seed{FRESH_SEED}"] * N_SRC)
    cohorts = ["heldout_seed0", f"fresh_seed{FRESH_SEED}"]
    n_src = Fs.shape[0]
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
    res = float(np.max([np.linalg.norm(np.asarray(
        mp.neg_lap_interior(jnp.asarray(U_int[i]), N)) - Fs[i])
        / np.linalg.norm(Fs[i]) for i in range(n_src)]))
    sc.log(f"  truth: {n_src} sources (2 cohorts), FOM CG rel residual {res:.2e}")
    assert res < 1e-10, "unconverged truth"
    report["config"]["truth_res"] = res
    U_int = U_int.reshape(n_src, -1)
    tn = np.array([np.linalg.norm(U_int[i]) for i in range(n_src)])

    # ---------------- weak-form tables (incumbent conventions) ---------------
    spec = dict(kind="weak", alpha=1.0, M=M_MODES)
    mask = np.asarray(grid.mode_mask(M_MODES)).astype(bool)
    I, Jm = np.nonzero(mask)
    S_ = np.asarray(grid.S)
    Phi_f = S_[grid.ix_full - 1][:, I] * S_[grid.iy_full - 1][:, Jm]  # (n_i2, M')
    n_modes = Phi_f.shape[1]
    report["config"]["n_modes_actual"] = int(n_modes)
    PhiT_all, Wl = pc.colloc_mode_table(grid, spec, "grid", coords_int)
    assert float(np.max(np.abs(np.asarray(PhiT_all) - Phi_f.T))) == 0.0, \
        "full-grid mode table != incumbent Phi_f"
    wq_all = jnp.ones((n_i2,), F64)      # discrete sine basis: exact weights = 1
    f_ms = [jnp.asarray(np.asarray(pc.weak_source_term(grid, spec, "grid", Fs[i])))
            for i in range(n_src)]

    # gate S: in-pipe source projection == incumbent pc.weak_source_term, and
    # ONE shared jitted projection feeds every path (bitwise-identical f_m).
    S_j = jnp.asarray(grid.S)
    I_j, Jm_j = jnp.asarray(I), jnp.asarray(Jm)
    Wsrc = jnp.asarray(np.asarray(grid.lam)[I, Jm] ** (-spec["alpha"]))

    def f_m_of(F2d):
        C = S_j.T @ F2d @ S_j
        return C[I_j, Jm_j] * Wsrc

    fm_dev = max(float(jnp.max(jnp.abs(f_m_of(jnp.asarray(Fs[i])) - f_ms[i]))
                       / (float(jnp.max(jnp.abs(f_ms[i]))) + 1e-300))
                 for i in (0, N_SRC, n_src - 1))
    fm_rep = float(jnp.max(jnp.abs(jax.jit(f_m_of)(jnp.asarray(Fs[0]))
                                   - jax.jit(f_m_of)(jnp.asarray(Fs[0])))))
    report["gates"]["S"] = dict(fm_max_rel_dev_vs_incumbent=fm_dev,
                                bitwise_repeat_dev=fm_rep,
                                shared_projection_across_paths=True)
    sc.log(f"  gate S: f_m vs pc.weak_source_term {fm_dev:.2e}; repeat {fm_rep:.1e}")
    assert fm_dev < 1e-12 and fm_rep == 0.0, "gate S failed"
    save()

    # ---------------- EQ path: incumbent NNLS fit (in-job) -------------------
    t0 = time.time()
    cand_pos = ctol_eq.candidate_pool(n_i2)
    cand_j = jnp.asarray(coords_int[cand_pos])
    u_cand = jax.jit(lambda z: dec(z, cand_j))
    u_full = jax.jit(lambda z: dec(z, jnp.asarray(coords_int)))
    keep, wq, eq_info = ctol_eq.eq_fit_poisson(
        u_cand, u_full, Phi_f[cand_pos], Phi_f, Z_tr, K, MQ,
        f"qf-cell poisson N={N} k={K} M={M_MODES} m={MQ}", pc.nnls_capped)
    eq_fit_secs = time.time() - t0
    report["eq"] = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    node_pos = cand_pos[keep]
    pts_np = coords_int[node_pos]
    t0 = time.time()
    PhiT_eq, Wl_eq = pc.colloc_mode_table(grid, spec, "grid", pts_np)
    G_q = dec.feat_at(pts_np)
    G_q.block_until_ready()
    eq_bank_secs = time.time() - t0
    assert np.array_equal(np.asarray(Wl_eq), np.asarray(Wl)), \
        "Wl differs between EQ and full tables"

    # ---------------- QF path: exact B matrix, built ONCE --------------------
    chunk = FEAT_CHUNK or (65536 if n_i2 > 262144 else 0)
    t0 = time.time()
    G_all = dec.feat_at(coords_int, chunk=chunk)
    G_all.block_until_ready()
    g_all_secs = time.time() - t0
    t0 = time.time()
    B_qf = jnp.asarray(PhiT_all) @ (wq_all[:, None] * G_all)      # (M', R)
    B_qf.block_until_ready()
    b_matmul_secs = time.time() - t0
    report["setup"] = dict(
        eq_nnls_fit_s=eq_fit_secs, eq_bank_and_table_s=eq_bank_secs,
        qf_g_all_bank_s=g_all_secs, qf_b_matmul_s=b_matmul_secs,
        note="G_all is also the shared full-field readout bank every path's "
             "timed pipe decodes with (the incumbent cached arm builds it "
             "too), so the marginal QF setup over the incumbent is only "
             "qf_b_matmul_s; the EQ path additionally pays eq_nnls_fit_s "
             "+ eq_bank_and_table_s.")
    sc.log(f"  setup: EQ NNLS fit {eq_fit_secs:.1f}s + bank {eq_bank_secs:.2f}s"
           f"  |  QF G_all {g_all_secs:.2f}s + B matmul {b_matmul_secs:.3f}s"
           f"  (B: {tuple(B_qf.shape)})")
    save()

    # ---------------- the three paths through the SAME solver ----------------
    dec_full = lambda z, xy: G_all @ h_fn(z)          # xy baked into the bank
    dec_eq = lambda z, xy: G_q @ h_fn(z)
    dec_qf = lambda z, xy: h_fn(z)                    # bank folded into B_qf
    path_ops = dict(
        full=(dec_full, coords_int, np.asarray(wq_all), np.asarray(PhiT_all)),
        eq=(dec_eq, pts_np, np.asarray(wq), np.asarray(PhiT_eq)),
        qf=(dec_qf, np.zeros((R, 2)), np.ones((R,)), np.asarray(B_qf)))

    def make_rJ(dfn, pts, wq_, PhiT_):
        pts_j, wq_j = jnp.asarray(pts), jnp.asarray(wq_)
        PhiT_j, Wl_j = jnp.asarray(PhiT_), jnp.asarray(Wl)

        def r_of(z, f_m):     # EXACTLY the residual inside lm_tau_poisson
            return Wl_j * (PhiT_j @ (wq_j * dfn(z, pts_j))) - f_m
        return (jax.jit(r_of),
                jax.jit(lambda z, f_m: (r_of(z, f_m),
                                        jax.jacfwd(r_of)(z, f_m))))

    rJ = {p: make_rJ(*path_ops[p]) for p in PATHS}
    _, rJ_mesh_full = make_rJ(dec, coords_int, wq_all, PhiT_all)
    _, rJ_mesh_eq = make_rJ(dec, pts_np, wq, PhiT_eq)

    # gates F/E: cached banks == meshfree decoder through the incumbent weak map
    rng = np.random.default_rng(SEED0)
    for gname, rj_cached, rj_mesh in (("F", rJ["full"][1], rJ_mesh_full),
                                      ("E", rJ["eq"][1], rJ_mesh_eq)):
        devs = []
        for _ in range(5):
            zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))]
                             + 0.05 * rng.standard_normal(K))
            ra, Ja = rj_cached(zt, f_ms[0])
            rb, Jb = rj_mesh(zt, f_ms[0])
            devs.append(max(
                float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(rb)) + 1e-300)),
                float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Jb)) + 1e-300))))
        report["gates"][gname] = dict(max_rel_dev=float(np.max(devs)))
        sc.log(f"  gate {gname} (cached vs meshfree weak r/J): "
               f"max rel dev {np.max(devs):.2e}")
        assert np.max(devs) < 1e-12, f"gate {gname} failed"

    # gate Q (random states): QF == FULL residual and gradient
    rngq = np.random.default_rng(SEED0 + 1)
    qr, qg = [], []
    for j in range(GATE_NZ):
        zt = jnp.asarray(Z_tr[rngq.integers(len(Z_tr))]
                         + 0.05 * rngq.standard_normal(K))
        fm = f_ms[j % n_src]
        rf, Jf = rJ["full"][1](zt, fm)
        rq, Jq = rJ["qf"][1](zt, fm)
        rf, Jf, rq, Jq = map(np.asarray, (rf, Jf, rq, Jq))
        qr.append(rel(rq, rf))
        qg.append(rel(Jq.T @ rq, Jf.T @ rf))
    report["gates"]["Q_random"] = dict(
        n=GATE_NZ, resid_rel_max=float(np.max(qr)), grad_rel_max=float(np.max(qg)))
    sc.log(f"  gate Q (random, n={GATE_NZ}): resid rel max {np.max(qr):.2e}, "
           f"grad rel max {np.max(qg):.2e}")
    assert np.max(qr) < 1e-12 and np.max(qg) < 1e-10, "gate Q (random) failed"
    save()

    # ---------------- fidelity rungs at z0 (b / c1, ladder conventions) ------
    z0 = jnp.asarray(Z_tr.mean(0))
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    trust = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(trust)

    def rungs_at(z, fm):
        rf, Jf = rJ["full"][1](z, fm)
        rf, Jf = np.asarray(rf), np.asarray(Jf)
        gf = Jf.T @ rf
        out = {}
        for p in ("eq", "qf"):
            rs, Js = rJ[p][1](z, fm)
            rs, Js = np.asarray(rs), np.asarray(Js)
            gs = Js.T @ rs
            out[p] = dict(
                b_resid=rel(rs, rf), c1_grad=rel(gs, gf),
                c1_cos=cosine(gs, gf),
                c1_abs=float(np.linalg.norm(gs - gf)
                             / (np.linalg.norm(Jf) * np.linalg.norm(rf) + 1e-300)))
        return out

    rungs_z0 = [dict(source=i, cohort=cohort_of[i], **rungs_at(z0, f_ms[i]))
                for i in range(n_src)]
    report["rungs"]["at_z0"] = rungs_z0
    for p in ("eq", "qf"):
        br = [r_[p]["b_resid"] for r_ in rungs_z0]
        c1 = [r_[p]["c1_grad"] for r_ in rungs_z0]
        sc.log(f"  rungs@z0 {p:4s}: b_resid mean {np.mean(br):.2e} "
               f"max {np.max(br):.2e}  c1_grad mean {np.mean(c1):.2e}")
    save()

    # ---------------- timed pipes: source -> f_m -> solve -> decode ----------
    u_full_fast = jax.jit(lambda z: G_all @ h_fn(z))   # shared readout, all paths
    pipes, compile_s = {}, {}
    for p in PATHS:
        dfn, pts, wq_, PhiT_ = path_ops[p]
        lm, _ = ctol_tol.lm_tau_poisson(dfn, K, pts, wq_, PhiT_, Wl,
                                        GN_ITERS, trust_delta=trust)

        def pipe_fn(F2d, tau, _lm=lm):
            out = _lm(z0, f_m_of(F2d), tau)
            return (u_full_fast(out[0]), out[0]) + out[1:]
        pipes[p] = jax.jit(pipe_fn)
        t0 = time.time()
        pipes[p](jnp.asarray(Fs[0]), TAUS[0])[0].block_until_ready()
        compile_s[p] = time.time() - t0
    report["setup"]["pipe_compile_s"] = compile_s
    sc.log(f"  pipe jit-compile: " +
           "  ".join(f"{p} {compile_s[p]:.1f}s" for p in PATHS))

    # ---------------- balanced timing over every source ----------------------
    acc = {}
    ctol_tol.burn_in(1.5)
    for i in range(n_src):
        Fi = jnp.asarray(Fs[i])
        subs = []
        for p in PATHS:
            for tau in TAUS:
                def fn(_p=pipes[p], _F=Fi, _tau=tau):
                    out = _p(_F, _tau)
                    out[0].block_until_ready()
                    return out
                subs.append((f"{p}|{tau:.0e}", fn))
        raw, results = sc.balanced_time(subs, reps=REPS, warm=BURN)
        for name in raw:
            acc.setdefault(name, []).append((raw[name], results[name]))
        if i == 0:
            sc.log(f"   timing block: {len(subs)} subjects x {REPS} reps "
                   f"(+{BURN} warm), balanced forward/reversed sweeps")

    # ---------------- rows (from the captured timed invocations) -------------
    # pipe output: (u, z, val, val0, n_jac, accepted, attempts, reason)
    for p in PATHS:
        for tau in TAUS:
            name = f"{p}|{tau:.0e}"
            for cname in cohorts:
                idxs = [i for i in range(n_src) if cohort_of[i] == cname]
                per = dict(t=[], err=[], jac=[], reason=[], val=[], val0=[],
                           gnorm=[], raw={})
                for i in idxs:
                    times, out = acc[name][i]
                    u = np.asarray(out[0]).reshape(-1)
                    z_sol = jnp.asarray(out[1])
                    r_s, J_s = rJ[p][1](z_sol, f_ms[i])
                    g_s = np.asarray(J_s).T @ np.asarray(r_s)
                    per["err"].append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
                    per["val"].append(float(out[2]))
                    per["val0"].append(float(out[3]))
                    per["jac"].append(int(out[4]))
                    per["reason"].append(int(out[7]))
                    per["gnorm"].append(float(np.linalg.norm(g_s)))
                    per["t"].append(float(np.median(times)))
                    per["raw"][str(i)] = [float(t) for t in times]
                cens = [r_ not in ctol_tol.POISSON_TAU_OK for r_ in per["reason"]]
                reasons_hist = {str(r_): per["reason"].count(r_)
                                for r_ in set(per["reason"])}
                row = dict(pde="poisson2d", method=p, N=N, k=K, r=R, M=M_MODES,
                           m=(int(len(wq)) if p == "eq"
                              else (n_i2 if p == "full" else 0)),
                           tau=tau, cohort=cname,
                           time_ms=float(np.median(per["t"])) * 1e3,
                           time_ms_all=[t * 1e3 for t in per["t"]],
                           time_raw_s=per["raw"],
                           err_rel_l2=float(np.mean(per["err"])),
                           err_rel_l2_max=float(np.max(per["err"])),
                           resid_final=float(np.mean(per["val"])),
                           resid_init=float(np.mean(per["val0"])),
                           grad_norm_final=float(np.mean(per["gnorm"])),
                           jac_evals=float(np.mean(per["jac"])),
                           censored_frac=float(np.mean(cens)),
                           stop_reasons=reasons_hist,
                           n_sources=len(idxs), trust_delta=float(trust))
                report["rows"].append(row)
                sc.log(f"   {p:4s} tau={tau:.0e} [{cname:14s}] "
                       f"proj+solve+decode {row['time_ms']:8.3f} ms  "
                       f"jac {row['jac_evals']:5.1f}  err {row['err_rel_l2']:.3e}  "
                       f"cens {row['censored_frac']*100:3.0f}%  "
                       f"reasons {reasons_hist}")
            save()

    # ---------------- gate Q at solve solutions + rungs at solutions ---------
    tau_min = min(TAUS)
    sol_r, sol_g_rel, sol_g_abs, sol_fail = [], [], [], 0
    for p in PATHS:
        for tau in TAUS:
            name = f"{p}|{tau:.0e}"
            for i in range(n_src):
                z_sol = jnp.asarray(acc[name][i][1][1])
                rf, Jf = rJ["full"][1](z_sol, f_ms[i])
                rq, Jq = rJ["qf"][1](z_sol, f_ms[i])
                rf, Jf, rq, Jq = map(np.asarray, (rf, Jf, rq, Jq))
                gf, gq = Jf.T @ rf, Jq.T @ rq
                rr = rel(rq, rf)
                gr = rel(gq, gf)
                ga = float(np.linalg.norm(gq - gf)
                           / (np.linalg.norm(Jf) * np.linalg.norm(rf) + 1e-300))
                sol_r.append(rr)
                sol_g_rel.append(gr)
                sol_g_abs.append(ga)
                if rr > 1e-12 or (gr > 1e-10 and ga > 1e-12):
                    sol_fail += 1
    report["gates"]["Q_solutions"] = dict(
        n=len(sol_r), resid_rel_max=float(np.max(sol_r)),
        grad_rel_max=float(np.max(sol_g_rel)),
        grad_abs_max=float(np.max(sol_g_abs)), n_fail=int(sol_fail),
        rule="resid<=1e-12 AND (grad_rel<=1e-10 OR grad_abs<=1e-12); grad_abs "
             "= ||g_qf-g_full||/(||J_full||_F ||r_full||), the ladder's c1_abs "
             "normalisation (g cancels near solver stops)")
    sc.log(f"  gate Q (solutions, n={len(sol_r)}): resid rel max "
           f"{np.max(sol_r):.2e}, grad rel max {np.max(sol_g_rel):.2e}, "
           f"grad abs max {np.max(sol_g_abs):.2e}, fails {sol_fail}")
    assert sol_fail == 0, "gate Q (solutions) failed"

    rungs_sol = []
    for i in range(n_src):
        z_sol = jnp.asarray(acc[f"full|{tau_min:.0e}"][i][1][1])
        rungs_sol.append(dict(source=i, cohort=cohort_of[i], tau=tau_min,
                              **rungs_at(z_sol, f_ms[i])))
    report["rungs"]["at_full_solution"] = rungs_sol
    for p in ("eq", "qf"):
        br = [r_[p]["b_resid"] for r_ in rungs_sol]
        c1 = [r_[p]["c1_grad"] for r_ in rungs_sol]
        cc = [r_[p]["c1_cos"] for r_ in rungs_sol]
        report["rungs"][f"summary_{p}"] = dict(
            b_resid_z0_mean=float(np.mean([r_[p]["b_resid"] for r_ in rungs_z0])),
            b_resid_sol_mean=float(np.mean(br)), b_resid_sol_max=float(np.max(br)),
            c1_grad_sol_mean=float(np.mean(c1)), c1_grad_sol_max=float(np.max(c1)),
            c1_cos_sol_min=float(np.min(cc)))
        sc.log(f"  rungs@sol {p:4s}: b_resid mean {np.mean(br):.2e} "
               f"max {np.max(br):.2e}  c1_grad mean {np.mean(c1):.2e}  "
               f"c1_cos min {np.min(cc):.6f}")

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE poisson-qf [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
