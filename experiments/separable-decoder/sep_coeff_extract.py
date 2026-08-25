"""ROUND 5 (burgers-accuracy campaign) -- COEFFICIENT EXTRACTION.

Purpose: make the h-fitting problem cheap enough to iterate on locally, WITHOUT
approximating it.  The observation this rests on is an exact algebraic identity,
not a modelling choice.

With the separable decoder u(.;z) = G h(z) restricted to a fixed point set
(G = features at those points, (n_pts, r)), write Gram = G^T G and, for any
target field u,

    c*(u) = argmin_c ||G c - u||   (the span least-squares coefficients),
    f(u)^2 = ||u||^2 - c*^T Gram c*   (the span floor residual, orthogonal),

and then FOR EVERY z

    ||G h(z) - u||^2 = (h(z) - c*)^T Gram (h(z) - c*) + f(u)^2.            (*)

The cross term vanishes because G c* is the orthogonal projection of u onto the
span.  (*) is exact in exact arithmetic and is asserted numerically below.

Consequences, and why this job exists:
  * reconstruction, the span LS floor, and the K-dim representation ORACLE of a
    FROZEN bank are all computable from (Gram, c*, ||u||) alone -- an object of
    size r^2 + S r instead of S n_pts.  At N=1024 that is 70 MB instead of
    137 GB.
  * so the entire "can h reach into its own span" question -- the rung round 3
    identified as binding, by a factor of ~32 -- can be attacked at zero
    cluster cost once this job has run once per (N, checkpoint).

This job therefore: regenerates the CANONICAL training data from the seed with
the incumbent generator and its <=1e-8 truth gate, regenerates the fresh-seed
test trajectories, and emits Gram plus the projections.  It changes no
discretization and trains nothing.  The bank comes from an ALREADY-TRAINED
checkpoint and is not modified.

Diagnostic status: c* uses truth (training snapshots, and test snapshots for
the test block).  The test block is a DIAGNOSTIC ONLY -- it is the same truth
the span-floor and oracle diagnostics already use in the r3/r4 drivers, and it
never enters any solve path or any training loss.  The training block is
training data and may be used for fitting.
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

F64 = jnp.float64

N = int(os.environ.get("N", "256"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "512"))
CKPT = os.environ["CKPT"]
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "16384"))
T_EARLY = int(os.environ.get("T_EARLY", "5"))
N_TRAJ = int(os.environ.get("N_TRAJ", "0"))
N_TEST = int(os.environ.get("N_TEST", "8"))
SEED0 = int(os.environ.get("SEED0", "0"))
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "64" if N <= 256 else "8"))
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "0" if N <= 512 else "131072"))
PROJ_CHUNK = int(os.environ.get("PROJ_CHUNK", "512"))
IDENT_ROWS = int(os.environ.get("IDENT_ROWS", "64"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
# LOOSE=1 is for SMOKE TESTS ONLY: it downgrades the "checkpoint codes must
# align with the reproduced state pick" assertion to a warning and replaces
# the codes with a recycled stand-in.  The Gram identity gate (*) holds for
# ANY z, so it still tests what it is there to test; the checkpoint
# reconstruction number becomes meaningless and is flagged as such.
LOOSE = int(os.environ.get("LOOSE", "0"))


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} "
           f"x64={jax.config.jax_enable_x64} EXTRACT N={N} K={K} R={R} "
           f"ckpt={CKPT}")
    t_all = time.time()
    TAG = f"N{N}_K{K}_R{R}"
    OUT = f"{OUT_PREFIX}sep_coeff_{TAG}.json"
    NPZ = f"{OUT_PREFIX}sep_coeff_{TAG}.npz"

    params, Z_ck, cfg_ck = sc.load_pkl(CKPT)
    dec = sc.SeparableDecoder(params, K, R)
    assert dec.r == R

    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    n_i2 = interior.size
    T = bc.NUM_STEPS + 1
    n_traj = N_TRAJ or (bc.bf.N_TRAIN + bc.bf.N_VAL)

    # ---- reproduce the r3/r4 state pick EXACTLY (same rng call sequence) ----
    rng = np.random.default_rng(SEED0)
    n_states = n_traj * T
    tidx_of = np.arange(n_states) % T
    early = np.nonzero(tidx_of <= T_EARLY)[0]
    rest = np.nonzero(tidx_of > T_EARLY)[0]
    if early.size >= MAX_SNAPS:
        pick = np.sort(rng.choice(early, MAX_SNAPS, replace=False))
    else:
        extra = rng.choice(rest, min(MAX_SNAPS - early.size, rest.size),
                           replace=False)
        pick = np.sort(np.concatenate([early, extra]))
    aligned = (pick.size == Z_ck.shape[0])
    if not aligned:
        msg = (f"pick {pick.size} != checkpoint codes {Z_ck.shape[0]}: the "
               "state pick does not match the checkpoint's training set")
        if not LOOSE:
            raise SystemExit(msg)
        sc.log(f"  LOOSE SMOKE MODE: {msg}")
        reps = int(np.ceil(pick.size / Z_ck.shape[0]))
        Z_ck = np.tile(Z_ck, (reps, 1))[:pick.size]
    else:
        sc.log(f"  pick reproduced: {pick.size} states, matches checkpoint "
               "codes")

    report = dict(config=dict(
        pde="burgers2d", job="coeff_extract", N=N, k=K, r=R, ckpt=CKPT,
        max_snaps=MAX_SNAPS, t_early=T_EARLY, n_traj=int(n_traj),
        n_test=N_TEST, seed=SEED0, data_seed=bc.SEED, test_seed=bc.TEST_SEED,
        num_steps=bc.NUM_STEPS, dt=bc.DT, x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local"),
        loose=bool(LOOSE), ckpt_cfg=cfg_ck), gates={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------------- the frozen bank ------------------------------
    coords_int = coords[interior]
    G_int = dec.feat_at(coords_int, chunk=FEAT_CHUNK)          # (n_i2, R)
    Gram_j = G_int.T @ G_int
    Gram = np.asarray(Gram_j, dtype=np.float64)
    ev = np.linalg.eigvalsh(Gram)[::-1]
    svr = np.sqrt(np.maximum(ev, 0.0)); svr = svr / max(svr[0], 1e-300)
    report["span"] = dict(
        note="DIAGNOSTIC ONLY -- no SVD/least-squares enters the model",
        gram_sv_ratio={str(i): float(svr[i])
                       for i in sorted({0, R // 4, R // 2, 3 * R // 4, R - 1})},
        numerical_rank_1e8=int(np.sum(svr > 1e-8)),
        cond_G=float(svr[0] / max(svr[R - 1], 1e-300)))
    sc.log(f"  bank rank(>1e-8) {report['span']['numerical_rank_1e8']}/{R}  "
           f"cond(G) {report['span']['cond_G']:.3e}")
    eps = 1e-12 * jnp.trace(Gram_j) / R
    Lc = jnp.linalg.cholesky(Gram_j + eps * jnp.eye(R, dtype=F64))
    report["span"]["chol_jitter_rel"] = 1e-12
    save()

    # NOTE: G_int is 4.3 GB at N=1024.  It is passed as an EXPLICIT jit
    # ARGUMENT, never captured -- a closed-over device array is lowered as an
    # HLO literal (+10 GB host RSS and ~16 s of compile per jit; CLAUDE.md).
    @jax.jit
    def _project(U, Gb, Lb, Gr):
        """U: (b, n_i2) -> c* (b, R), ||u||^2 (b,), floor^2 (b,).

        The normal equations are formed on Gram, whose condition number is the
        SQUARE of the bank's (cond(G) ~ 2.6e4 at N=256/R=512, so cond(Gram) ~
        7e8).  One step of iterative refinement, with the residual taken
        through G itself rather than through Gram, recovers most of what the
        squaring costs and is one extra pass over the bank."""
        B = Gb.T @ U.T                                      # (R, b)
        C = jax.scipy.linalg.cho_solve((Lb, True), B).T     # (b, R)
        Rres = B - Gb.T @ (Gb @ C.T)                        # (R, b)
        C = C + jax.scipy.linalg.cho_solve((Lb, True), Rres).T
        un2 = jnp.sum(U * U, axis=1)
        E = U - C @ Gb.T
        return C, un2, jnp.sum(E * E, axis=1)

    def project(U):
        return _project(U, G_int, Lc, Gram_j)

    # ------------------- training data: stream + project --------------------
    cx, cy, w, a, nu, _z = bc.bf.sample_params(seed=bc.SEED)
    assert n_traj <= len(cx)
    rollout, res_fn = bc.bf.make_rollout(N)
    chk = jax.jit(jax.vmap(lambda u1, u0, nu_:
                           jnp.linalg.norm(res_fn(u1, u0, nu_))
                           / jnp.linalg.norm(u0)))
    interior_j = jnp.asarray(interior)
    pick_set = {int(v): i for i, v in enumerate(pick)}
    C_tr = np.zeros((pick.size, R), dtype=np.float64)
    un2_tr = np.zeros(pick.size, dtype=np.float64)
    fl2_tr = np.zeros(pick.size, dtype=np.float64)
    # identity-check rows: keep the FULL interior field for a few picked states
    id_rows = np.sort(rng.choice(pick.size, min(IDENT_ROWS, pick.size),
                                 replace=False))
    id_set = {int(pick[i]): j for j, i in enumerate(id_rows)}
    U_id = np.zeros((len(id_rows), n_i2), dtype=np.float64)
    worst = 0.0
    fp_sum = fp_sumsq = 0.0
    t0 = time.time()
    for s in range(0, n_traj, GEN_CHUNK):
        e = min(s + GEN_CHUNK, n_traj)
        U0 = np.stack([bc.bf.blob_ic(N, cx[i], cy[i], w[i], a[i])
                       for i in range(s, e)])
        nu_j = jnp.asarray(nu[s:e])
        snaps, res = rollout(jnp.asarray(U0), nu_j)          # (T, b, n^2)
        cm = float(jnp.max(res))
        if not np.isfinite(cm) or cm > worst:
            worst = cm
        for kk in range(bc.NUM_STEPS):
            wr = float(jnp.max(chk(snaps[kk + 1], snaps[kk], nu_j)))
            if not np.isfinite(wr) or wr > worst:
                worst = wr
        fp_sum += float(jnp.sum(snaps))
        fp_sumsq += float(jnp.sum(snaps * snaps))
        rows, tl, bl = [], [], []
        for b in range(e - s):
            base = (s + b) * T
            for t in range(T):
                r_ = pick_set.get(base + t)
                if r_ is not None:
                    rows.append(r_); tl.append(t); bl.append(b)
        if rows:
            tl_j = jnp.asarray(np.asarray(tl)); bl_j = jnp.asarray(np.asarray(bl))
            rows = np.asarray(rows)
            for q in range(0, rows.size, PROJ_CHUNK):
                sl = slice(q, min(q + PROJ_CHUNK, rows.size))
                Uq = snaps[tl_j[sl], bl_j[sl]][:, interior_j]
                Cq, uq, fq = project(Uq)
                C_tr[rows[sl]] = np.asarray(Cq)
                un2_tr[rows[sl]] = np.asarray(uq)
                fl2_tr[rows[sl]] = np.asarray(fq)
                for jj, gid in enumerate(range(sl.start, sl.stop)):
                    idr = id_set.get(int(pick[rows[gid]]))
                    if idr is not None:
                        U_id[idr] = np.asarray(Uq[jj])
                del Uq, Cq
        del snaps
        sc.log(f"   gen+proj: trajectories {e}/{n_traj}  worst FOM rel "
               f"residual {worst:.2e}  [{time.time()-t0:.0f}s]")
    if not np.isfinite(worst) or worst > 1e-8:
        raise SystemExit(f"FOM residual {worst:.2e} > 1e-8: data not converged")
    report["data"] = dict(
        n_traj=int(n_traj), T=int(T), n_i2=int(n_i2),
        n_states_trained=int(pick.size),
        fingerprint=dict(sum=fp_sum, sumsq=fp_sumsq,
                         shape=[int(n_traj), int(T), int(N * N)]),
        max_fom_rel_residual=worst)
    save()

    # ---------------- IDENTITY GATE for the reformulation -------------------
    # Recompute the checkpoint's reconstruction two ways on the identity rows:
    # (a) full field ||G h(z) - u|| / ||u||, (b) the Gram-space form (*).
    Hid = sc.head(params, jnp.asarray(Z_ck[id_rows]))
    Uh = jnp.asarray(U_id)
    a_full = np.asarray(jnp.linalg.norm(G_int @ Hid.T - Uh.T, axis=0)
                        / jnp.linalg.norm(Uh, axis=1))
    d = Hid - jnp.asarray(C_tr[id_rows])
    a_gram = np.asarray(jnp.sqrt(jnp.maximum(
        jnp.sum(d * (d @ Gram_j), axis=1) + jnp.asarray(fl2_tr[id_rows]), 0.0))
        / jnp.sqrt(jnp.asarray(un2_tr[id_rows])))
    dev_rel = np.abs(a_full - a_gram) / np.maximum(a_full, 1e-300)
    ident = float(np.max(dev_rel))
    ident_mean = float(np.mean(dev_rel))
    report["gates"]["gram_identity_rel_dev_max"] = ident
    report["gates"]["gram_identity_rel_dev_mean"] = ident_mean
    report["gates"]["gram_identity_rows"] = int(len(id_rows))
    sc.log(f"  GRAM IDENTITY gate on {len(id_rows)} training states: max rel "
           f"deviation {ident:.3e} mean {ident_mean:.3e}  (full mean "
           f"{a_full.mean():.6e} vs gram mean {a_gram.mean():.6e})")
    # the gate is genuinely independent: a_full comes from the FULL FIELDS and h
    # only, a_gram from (c*, floor^2, Gram) only, so agreeing to ~1e-10 validates
    # the projection and the floor as well as the identity.
    # Bars, and why they are where they are.  The identity is exact in exact
    # arithmetic, so a FORMULATION error would show up at O(1).  What is left
    # is f64 round-off in c*, whose forward error is amplified by cond(Gram) =
    # cond(G)^2 ~ 7e8; in FIELD terms that is ~1e-12 of ||u||, i.e. eight
    # orders below the 1e-4 accuracy this campaign is chasing.  The bars below
    # are therefore set to catch a formulation error, not to police round-off,
    # and the achieved values are recorded in the report.
    report["gates"]["gram_identity_implied_field_rel"] = float(
        ident_mean * a_full.mean())
    assert ident_mean < 1e-6 and ident < 1e-4, \
        "Gram-space reformulation is not exact"
    # Direct span-floor cross-check on the same rows.  Since the projection now
    # returns floor^2 as ||u - G c*||^2 computed inside the jit, this mostly
    # re-checks the round-trip through the npz rather than the formula; the
    # INDEPENDENT check on the floor is the identity gate above, which compares
    # a field-space number against a (c*, floor, Gram) number.
    Cid = jnp.asarray(C_tr[id_rows])
    fl_direct = np.asarray(jnp.linalg.norm(G_int @ Cid.T - Uh.T, axis=0)
                           / jnp.linalg.norm(Uh, axis=1))
    fl_gram = np.sqrt(fl2_tr[id_rows] / np.maximum(un2_tr[id_rows], 1e-300))
    fdev = float(np.max(np.abs(fl_direct - fl_gram)
                        / np.maximum(fl_direct, 1e-300)))
    report["gates"]["span_floor_direct_vs_gram_rel_dev"] = fdev
    sc.log(f"  span-floor direct-vs-Gram deviation on {len(id_rows)} rows: "
           f"{fdev:.3e}  (direct mean {fl_direct.mean():.6e})")
    assert fdev < 1e-6, "span floor round-trip is inaccurate"
    # and the checkpoint's own recon over ALL picked states, in Gram space
    Hall = np.zeros((pick.size, R))
    for q in range(0, pick.size, 4096):
        sl = slice(q, min(q + 4096, pick.size))
        Hall[sl] = np.asarray(sc.head(params, jnp.asarray(Z_ck[sl])))
    dd = jnp.asarray(Hall - C_tr)
    rec = np.asarray(jnp.sqrt(jnp.maximum(
        jnp.sum(dd * (dd @ Gram_j), axis=1) + jnp.asarray(fl2_tr), 0.0))
        / jnp.sqrt(jnp.asarray(un2_tr)))
    report["ckpt_recon"] = dict(
        mean=float(rec.mean()), max=float(rec.max()), codes_aligned=aligned,
        note=("checkpoint h,Z re-evaluated via (*)" if aligned else
              "MEANINGLESS: LOOSE smoke mode, codes are a recycled stand-in"))
    fl_tr = np.sqrt(fl2_tr / np.maximum(un2_tr, 1e-300))
    report["train_span_floor"] = dict(mean=float(fl_tr.mean()),
                                      max=float(fl_tr.max()))
    sc.log(f"  checkpoint recon via (*): mean {rec.mean():.6e}  "
           f"(train span LS floor mean {fl_tr.mean():.3e})")
    del dd, Hall
    save()

    # ------------------------- fresh test states ----------------------------
    cxt, cyt, wt, at, nut, _ = bc.bf.sample_params(seed=bc.TEST_SEED, m=N_TEST)
    U0t = np.stack([bc.bf.blob_ic(N, cxt[i], cyt[i], wt[i], at[i])
                    for i in range(N_TEST)])
    snaps_t, res_t = rollout(jnp.asarray(U0t), jnp.asarray(nut))
    wt_res = float(jnp.max(res_t))
    for kk in range(bc.NUM_STEPS):
        wr = float(jnp.max(chk(snaps_t[kk + 1], snaps_t[kk],
                               jnp.asarray(nut))))
        wt_res = max(wt_res, wr)
    if not np.isfinite(wt_res) or wt_res > 1e-8:
        raise SystemExit(f"TEST FOM residual {wt_res:.2e} > 1e-8")
    C_te = np.zeros((N_TEST * T, R)); un2_te = np.zeros(N_TEST * T)
    fl2_te = np.zeros(N_TEST * T)
    for i in range(N_TEST):
        Ui = snaps_t[:, i][:, interior_j]                     # (T, n_i2)
        Ci, ui, fi = project(Ui)
        C_te[i * T:(i + 1) * T] = np.asarray(Ci)
        un2_te[i * T:(i + 1) * T] = np.asarray(ui)
        fl2_te[i * T:(i + 1) * T] = np.asarray(fi)
    fl_te = np.sqrt(fl2_te / np.maximum(un2_te, 1e-300))
    report["test_span_floor"] = dict(mean=float(fl_te.mean()),
                                     max=float(fl_te.max()),
                                     n_states=int(N_TEST * T))
    report["data"]["max_fom_rel_residual_test"] = wt_res
    sc.log(f"  fresh test span LS floor: mean {fl_te.mean():.3e} "
           f"max {fl_te.max():.3e}  (test FOM residual {wt_res:.2e})")
    del snaps_t

    # ------------------------------ emit ------------------------------------
    tid_tr = (pick // T).astype(np.int32)
    tim_tr = (pick % T).astype(np.int32)
    mu_tr = np.stack([cx, cy, w, a, nu], axis=1)[tid_tr]
    mu_te = np.repeat(np.stack([cxt, cyt, wt, at, nut], axis=1), T, axis=0)
    np.savez_compressed(
        NPZ, Gram=Gram, C_tr=C_tr, un2_tr=un2_tr, fl2_tr=fl2_tr,
        traj_tr=tid_tr, t_tr=tim_tr, mu_tr=mu_tr, pick=pick, Z_ck=Z_ck,
        C_te=C_te, un2_te=un2_te, fl2_te=fl2_te, mu_te=mu_te,
        traj_te=np.repeat(np.arange(N_TEST), T).astype(np.int32),
        t_te=np.tile(np.arange(T), N_TEST).astype(np.int32),
        U_id=U_id, id_rows=id_rows)
    report["npz"] = os.path.basename(NPZ)
    report["total_seconds"] = time.time() - t_all
    report["complete"] = True
    save()
    sc.log(f"EXTRACT done in {report['total_seconds']:.0f}s -> {NPZ}")


if __name__ == "__main__":
    main()
