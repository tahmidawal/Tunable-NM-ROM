"""Generalizable cascade + LM ROM on POISSON-2D (held-out sources).

Pipeline (see cn_common.py for the design):
  data   : mp.build_snapshots (f64 FD/CG, seed family); TRAIN = first N_TRAIN,
           HELD-OUT = last N_VAL (never used for any fit); ROM on the first
           N_TEST held-out sources.  Encoder input = source f on the fixed
           16x16 lattice (standardized with TRAIN stats).
  stage 0: E(f) -> z (K_LAT) + D0, joint.  TRUE_Z=1 replaces the encoder by
           the true 4 parameters (control arm; K_LAT forced to 4).
  gate   : probe (eff_rank / POD-r / whitened NN-corr) of the residual before
           each stage; GATE=1 stops stacking at the first flagged residual.
  stages : residual FiLM decoders on frozen z (K_EXTRA / LAT_SMOOTH levers).
  eval   : on HELD-OUT, per number of stages S:
             (i)  plug-in error at z = E(f)               ["encoded"; the ROM's
                                                             free generalizable init]
             (ii) finite-budget LM-inferred latent on the data misfit, init
                  E(f) (and mean) — labelled inferred, NOT a lower bound
             (iii) ROM: LM on the ghost-zero-Dirichlet FD residual, init E(f),
                  collocation full + m subset,
                  objectives: 'fd' (plain), 'hinv' (residual preconditioned by
                  the exact discrete inverse Laplacian via CG — for the LINEAR
                  Poisson operator this equals the field misfit to the FOM
                  solution, so it is an OBJECTIVE DIAGNOSTIC / upper bound at
                  FOM-solve cost, not a cheap ROM), 'hinvK' (K CG iterations
                  as an approximate preconditioner — cheap-ish).
  report : JSON with config manifest, per-stage rows, probe rows, ROM rows,
           `complete` flag; stages pkl (deployable: encoder + stages).

Usage: [K_LAT=8] [N=64] [N_TRAIN=512] [N_VAL=64] [N_STAGES=3] [STEPS=20000]
       [P_SUB=1024] [BATCH=32] [N_TEST=16] [GN_ITERS=60] [M_EQ=512]
       [K_EXTRA=0] [LAT_SMOOTH=0] [GATE=0] [TRUE_Z=0] [HINV_ITERS=20]
       python cn_poisson.py [outdir]
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

import cn_common as cn
from cn_common import mp, lm_solve

TRUE_Z = int(os.environ.get("TRUE_Z", "0"))
K_LAT = 4 if TRUE_Z else cn.K_LAT
N, N_TRAIN, N_VAL, STEPS = mp.N, mp.N_TRAIN, mp.N_VAL, mp.STEPS
N_STAGES = int(os.environ.get("N_STAGES", "3"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = cn.GN_ITERS
M_EQ = int(os.environ.get("M_EQ", "512"))
HINV_ITERS = int(os.environ.get("HINV_ITERS", "20"))
OBJECTIVES = os.environ.get("OBJECTIVES", "fd,hinv,hinvK").split(",")
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
F64 = jnp.float64
CONFIG = dict(mp.CONFIG, **cn.COMMON_CONFIG, pde="poisson2d", true_z=TRUE_Z,
              K_LAT=K_LAT, n_stages=N_STAGES, n_test=N_TEST, m_eq=M_EQ,
              hinv_iters=HINV_ITERS, objectives=OBJECTIVES)
TAG = f"{'truez' if TRUE_Z else 'enc'}_K{K_LAT}_X{cn.K_EXTRA}_S{cn.LAT_SMOOTH:g}"


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}  {TAG}",
          flush=True)
    print("CONFIG " + json.dumps(CONFIG), flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    U, z_true, coords, fom_res = mp.build_snapshots(N)
    cx, cy, w, a, _ = mp.sample_params()
    n_all = U.shape[0]
    U_tr, U_ho = U[:N_TRAIN], U[N_TRAIN:]
    zt = np.asarray(z_true)
    # encoder input: source on the lattice, standardized with TRAIN stats
    F_lat = cn.gaussian_on_lattice(cx, cy, w, a)
    mu, sd = cn.standardize_fit(F_lat[:N_TRAIN])
    inp = jnp.asarray((F_lat - mu) / sd)
    if TRUE_Z:
        inp = jnp.asarray(zt)                       # control: identity "encoder"
    w_tr = mp.sample_weights(U_tr)
    np_rng = np.random.default_rng(cn.SEED)
    coll_rng = np.random.default_rng(cn.SEED + 12345)
    key = jax.random.PRNGKey(cn.SEED + 7 + K_LAT)
    report = {"config": CONFIG, "fom_max_rel_residual": fom_res, "stages": [],
              "probes": [], "rom": [], "complete": False}

    def save(phase):
        with open(os.path.join(OUTDIR, f"cn_poisson_{TAG}_report.json"), "w") as f:
            json.dump(dict(report, phase=phase), f, indent=2)

    def dump_pkl(enc_params, stages):
        with open(os.path.join(OUTDIR, f"cn_poisson_{TAG}_stages.pkl"), "wb") as f:
            pickle.dump({"config": CONFIG, "encoder": None if TRUE_Z else
                         jax.tree_util.tree_map(np.asarray, enc_params),
                         "lattice_mu": mu, "lattice_sd": sd,
                         "stages": cn.stages_to_np(stages)}, f)

    # ---------------- stage 0: encoder + decoder ----------------
    eps0 = float(jnp.sqrt(jnp.mean(U_tr ** 2)))
    n_freq = mp.freq_schedule(mp.dominant_radial_freq(U_tr, N), 0, N)
    print(f"stage 0: eps={eps0:.3e} n_freq={n_freq}", flush=True)
    key, k0 = jax.random.split(key)
    row2inp = np.arange(N_TRAIN)
    if TRUE_Z:
        # identity encoder: fit the decoder only via mp.fit_stage on true z
        dec0, _, loss0, secs0 = mp.fit_stage(
            k0, np_rng, coords, U_tr / eps0, w_tr, n_freq, jnp.asarray(zt[:N_TRAIN]),
            k_lat=4, z_ff=0, learn_latents=False, steps=STEPS, tag="A")
        enc0 = None
        Z_all = jnp.asarray(zt)
    else:
        dec0, enc0, Z_tr, loss0, secs0 = cn.fit_stage0(
            k0, np_rng, coords, U_tr / eps0, w_tr, n_freq, inp[:N_TRAIN], row2inp,
            np.ones(N_TRAIN, bool), K_LAT, steps=STEPS, tag="A")
        Z_all = jax.vmap(lambda r: cn.enc_apply(enc0, inp[r]))(jnp.arange(n_all))
    stages = [{"params": dec0, "n_freq": n_freq, "eps": eps0, "k_extra": 0}]
    Z_tr, Z_ho = Z_all[:N_TRAIN], Z_all[N_TRAIN:]
    C_all = [None]

    def extras_for(rows):
        return [None if s.get("k_extra", 0) == 0 else
                jax.vmap(lambda r: cn.enc_apply(s["enc_extra"], inp[r]))(jnp.asarray(rows))
                for s in stages]

    def evaluate(S_stages_note):
        C_tr = extras_for(np.arange(N_TRAIN)); C_ho = extras_for(np.arange(N_TRAIN, n_all))
        pred_tr = cn.predict_rows(stages, Z_tr, C_tr, coords)
        pred_ho = cn.predict_rows(stages, Z_ho, C_ho, coords)
        g_tr, m_tr, _ = mp.rel_metrics(pred_tr, U_tr)
        g_ho, m_ho, per_ho = mp.rel_metrics(pred_ho, U_ho)
        return pred_tr, g_tr, m_tr, g_ho, m_ho, per_ho

    pred_tr, g_tr, m_tr, g_ho, m_ho, per_ho = evaluate(0)
    e_tr = U_tr - pred_tr
    pr = cn.probe(e_tr, np.asarray(Z_tr)); pr["after_stage"] = 0
    pr_f = cn.probe(U_tr, np.asarray(Z_tr)); pr_f["after_stage"] = -1
    report["probes"] += [pr_f, pr]
    row = {"stage": 0, "eps_in": eps0, "n_freq": n_freq, "adam_final_batch_loss": loss0,
           "secs": secs0, "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
           "heldout_encoded_global_rel": g_ho, "heldout_encoded_mean_rel_l2": m_ho,
           "heldout_encoded_by_amp_quartile": mp.quartile_errors(per_ho, U_ho),
           "latent_rms": float(jnp.sqrt(jnp.mean(Z_tr ** 2)))}
    report["stages"].append(row)
    print(f"  after stage 0: TRAIN {g_tr:.3e}/{m_tr:.3e}  HELD-OUT encoded {g_ho:.3e}/"
          f"{m_ho:.3e}  probe eff_rank {pr['eff_rank']:.1f} pod8 {pr['pod8_rel_err']:.2e} "
          f"nn5 {pr['nn5_corr']:.2f} stop? {pr['stop_suggested']}", flush=True)
    dump_pkl(enc0, stages); save("A")

    # ---------------- stages 1..S-1 (frozen z) ----------------
    stopped_at = None
    for k in range(1, N_STAGES):
        if cn.GATE and pr["stop_suggested"]:
            stopped_at = k; print(f"GATE: stopping before stage {k}", flush=True); break
        eps = float(jnp.sqrt(jnp.mean(e_tr ** 2)))
        f_d = mp.dominant_radial_freq(e_tr, N)
        n_freq = mp.freq_schedule(f_d, n_freq, N)
        print(f"stage {k}: eps={eps:.3e} f_d~{f_d:.1f} n_freq={n_freq}", flush=True)
        key, sub = jax.random.split(key)
        params, enc_x, loss, secs = cn.fit_residual_stage(
            sub, np_rng, coords, e_tr / eps, w_tr, n_freq, Z_tr, inp, row2inp, K_LAT,
            k_extra=cn.K_EXTRA, steps=STEPS, tag=f"B{k}")
        st = {"params": params, "n_freq": n_freq, "eps": eps, "k_extra": cn.K_EXTRA}
        if cn.K_EXTRA > 0:
            st["enc_extra"] = enc_x
        stages.append(st)
        pred_tr, g_tr, m_tr, g_ho, m_ho, per_ho = evaluate(k)
        e_tr = U_tr - pred_tr
        pr = cn.probe(e_tr, np.asarray(Z_tr)); pr["after_stage"] = k
        report["probes"].append(pr)
        row = {"stage": k, "eps_in": eps, "f_d_cyc": f_d, "n_freq": n_freq,
               "adam_final_batch_loss": loss, "secs": secs,
               "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
               "heldout_encoded_global_rel": g_ho, "heldout_encoded_mean_rel_l2": m_ho,
               "heldout_encoded_by_amp_quartile": mp.quartile_errors(per_ho, U_ho)}
        report["stages"].append(row)
        print(f"  after stage {k}: TRAIN {g_tr:.3e}/{m_tr:.3e}  HELD-OUT encoded "
              f"{g_ho:.3e}/{m_ho:.3e}  probe eff_rank {pr['eff_rank']:.1f} pod8 "
              f"{pr['pod8_rel_err']:.2e} nn5 {pr['nn5_corr']:.2f} stop? "
              f"{pr['stop_suggested']}", flush=True)
        dump_pkl(enc0, stages); save("B")
    report["gate_stopped_before_stage"] = stopped_at
    save("B-done")

    # ---------------- held-out: inferred latents + ROM ----------------
    U_test = np.asarray(U_ho[:N_TEST]); test_norms = np.linalg.norm(U_test, axis=1)
    Z_init = np.asarray(Z_ho[:N_TEST])
    z_mean = np.asarray(jnp.mean(Z_tr, axis=0))
    C_test = extras_for(np.arange(N_TRAIN, N_TRAIN + N_TEST))
    dx = 1.0 / (N - 1)
    ii, jj = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1), indexing="ij")
    ix_full, iy_full = ii.reshape(-1), jj.reshape(-1)
    n_int = len(ix_full)
    sub = coll_rng.choice(n_int, size=min(M_EQ, n_int), replace=False)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    bmask = np.zeros((N, N), bool); bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bpts = jnp.asarray(np.stack([X[bmask], Y[bmask]], axis=1))
    F_int = [jnp.asarray(mp.source_interior(N, cx[N_TRAIN + i], cy[N_TRAIN + i],
                                             w[N_TRAIN + i], a[N_TRAIN + i]))
             for i in range(N_TEST)]                       # (N-2, N-2)

    def stencil(ix, iy):
        offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        pts, keep = [], []
        for ox, oy in offs:
            jx, jy = ix + ox, iy + oy
            pts.append(np.stack([jx * dx, jy * dx], axis=1))
            keep.append(~((jx == 0) | (jx == N - 1) | (jy == 0) | (jy == N - 1)))
        return jnp.asarray(np.stack(pts)), jnp.asarray(np.stack(keep).astype(float))

    op = lambda v: mp.neg_lap_interior(v, N)              # (N-2,N-2) -> same

    def make_solver(n_stages, objective, ix, iy):
        st = stages[:n_stages]
        pts, keep = stencil(ix, iy)
        def dec_at(z, cs, xy):
            return cn.cascade_apply(st, z, cs[:n_stages], xy)
        if objective == "fd":
            def residual(z, cs, f_rows):
                u = dec_at(z, cs, pts.reshape(-1, 2)).reshape(5, -1) * keep
                lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (dx * dx)
                return -lap - f_rows
            rJ = jax.jit(lambda z, cs, f: (residual(z, cs, f), jax.jacfwd(residual)(z, cs, f)))
            rn = jax.jit(lambda z, cs, f: jnp.linalg.norm(residual(z, cs, f)))
        else:
            iters = None if objective == "hinv" else HINV_ITERS
            def full_res(z, cs, F):
                u_int = dec_at(z, cs, coords).reshape(N, N)[1:-1, 1:-1]
                return op(u_int) - F                       # (N-2,N-2)
            def precond(r):
                if iters is None:
                    return jax.scipy.sparse.linalg.cg(op, r, tol=1e-12, maxiter=20000)[0]
                return jax.scipy.sparse.linalg.cg(op, r, tol=0.0, maxiter=iters)[0]
            def wres(z, cs, F):
                return precond(full_res(z, cs, F)).reshape(-1)
            def rJ_fn(z, cs, F):
                r = full_res(z, cs, F)
                Jr = jax.jacfwd(full_res)(z, cs, F)          # (N-2,N-2,K)
                rw = precond(r).reshape(-1)
                Jw = jax.vmap(lambda col: precond(col).reshape(-1), in_axes=2, out_axes=1)(Jr)
                return rw, Jw
            rJ = jax.jit(rJ_fn)
            rn = jax.jit(lambda z, cs, F: jnp.linalg.norm(wres(z, cs, F)))
        bnorm = jax.jit(lambda z, cs: jnp.linalg.norm(dec_at(z, cs, bpts)))
        dec_full = jax.jit(lambda z, cs: dec_at(z, cs, coords))
        return rJ, rn, bnorm, dec_full

    def cs_of(i):
        return [None if C is None else C[i] for C in C_test]

    # (ii) finite-budget inferred latents (data misfit) per n_stages
    inferred = {}
    for S in range(1, len(stages) + 1):
        st = stages[:S]
        dec = lambda z, cs, u: cn.cascade_apply(st, z, cs[:S], coords) - u
        rJ = jax.jit(lambda z, cs, u: (dec(z, cs, u), jax.jacfwd(dec)(z, cs, u)))
        rn = jax.jit(lambda z, cs, u: jnp.linalg.norm(dec(z, cs, u)))
        res = {}
        for init_name, Z0 in (("encoded", Z_init), ("mean", np.tile(z_mean, (N_TEST, 1)))):
            rels, accs = [], []
            for i in range(N_TEST):
                u = jnp.asarray(U_test[i]); cs = cs_of(i)
                z, r, info = lm_solve(lambda zz: rJ(zz, cs, u), lambda zz: rn(zz, cs, u),
                                      jnp.asarray(Z0[i]), GN_ITERS)
                rels.append(r / test_norms[i]); accs.append(info["accepted"])
            res[init_name] = {"rel": np.asarray(rels), "acc_med": float(np.median(accs))}
        best = np.minimum(res["encoded"]["rel"], res["mean"]["rel"])
        inferred[S] = {"encoded": res["encoded"], "mean": res["mean"], "best": best}
        report.setdefault("inferred", []).append(
            {"n_stages": S, "init_encoded_mean_rel_l2": float(res["encoded"]["rel"].mean()),
             "init_mean_mean_rel_l2": float(res["mean"]["rel"].mean()),
             "best_of_starts_mean_rel_l2": float(best.mean()),
             "budget_attempts": GN_ITERS})
        print(f"INFERRED stages={S}: encoded-init {res['encoded']['rel'].mean():.3e} "
              f"mean-init {res['mean']['rel'].mean():.3e} best {best.mean():.3e}", flush=True)
    save("inferred")

    # (iii) ROM
    colls = {"full": (ix_full, iy_full), f"m{M_EQ}": (ix_full[sub], iy_full[sub])}
    for objective in OBJECTIVES:
        for cname, (ix, iy) in colls.items():
            if objective != "fd" and cname != "full":
                continue                                    # hinv needs the full grid
            for S in range(1, len(stages) + 1):
                rJ, rn, bnorm, dec_full = make_solver(S, objective, ix, iy)
                per = {k: [] for k in ("err", "err_enc", "err_inf", "r_lm", "r_enc", "f_norm",
                                       "b_lm", "acc", "rej", "reason", "lam", "z_nn")}
                t0 = time.time()
                for i in range(N_TEST):
                    cs = cs_of(i)
                    if objective == "fd":
                        f_rows = jnp.asarray(a[N_TRAIN + i] * np.exp(
                            -((ix * dx - cx[N_TRAIN + i]) ** 2 + (iy * dx - cy[N_TRAIN + i]) ** 2)
                            / (2 * w[N_TRAIN + i] ** 2)))
                    else:
                        f_rows = F_int[i]
                    z0 = jnp.asarray(Z_init[i])
                    z, r_lm, info = lm_solve(lambda zz: rJ(zz, cs, f_rows),
                                             lambda zz: rn(zz, cs, f_rows), z0, GN_ITERS)
                    pred = np.asarray(dec_full(z, cs))
                    pred0 = np.asarray(dec_full(z0, cs))
                    per["err"].append(np.linalg.norm(pred - U_test[i]) / test_norms[i])
                    per["err_enc"].append(np.linalg.norm(pred0 - U_test[i]) / test_norms[i])
                    per["err_inf"].append(float(inferred[S]["encoded"]["rel"][i]))
                    per["r_lm"].append(r_lm); per["r_enc"].append(float(rn(z0, cs, f_rows)))
                    per["f_norm"].append(float(jnp.linalg.norm(f_rows)))
                    per["b_lm"].append(float(bnorm(z, cs)))
                    per["acc"].append(info["accepted"]); per["rej"].append(info["rejected"])
                    per["reason"].append(info["reason"]); per["lam"].append(info["final_lambda"])
                    per["z_nn"].append(float(np.min(np.linalg.norm(
                        np.asarray(Z_tr) - np.asarray(z), axis=1))))
                e = np.asarray(per["err"])
                row = {"objective": objective, "colloc": cname, "n_stages": S, "init": "encoded",
                       "budget_attempts": GN_ITERS,
                       "rom_rel_l2_mean": float(e.mean()), "rom_rel_l2_med": float(np.median(e)),
                       "rom_rel_l2_max": float(e.max()),
                       "encoded_plugin_rel_l2_mean": float(np.mean(per["err_enc"])),
                       "inferred_latent_rel_l2_mean": float(np.mean(per["err_inf"])),
                       "resid_lm_med": float(np.median(per["r_lm"])),
                       "resid_encoded_med": float(np.median(per["r_enc"])),
                       "f_norm_med": float(np.median(per["f_norm"])),
                       "boundary_block_lm_med": float(np.median(per["b_lm"])),
                       "lm_accepted_med": float(np.median(per["acc"])),
                       "lm_rejected_med": float(np.median(per["rej"])),
                       "lm_final_lambda_med": float(np.median(per["lam"])),
                       "lm_reasons": {r: per["reason"].count(r) for r in set(per["reason"])},
                       "z_nn_dist_med": float(np.median(per["z_nn"])),
                       "per_sample_rom_rel_l2": [float(v) for v in e], "secs": time.time() - t0}
                report["rom"].append(row)
                print(f"RESULT {TAG} obj={objective:5s} colloc={cname:5s} stages={S}  "
                      f"ROM {row['rom_rel_l2_mean']:.3e} (med {row['rom_rel_l2_med']:.3e} "
                      f"max {row['rom_rel_l2_max']:.3e})  encoded-plugin "
                      f"{row['encoded_plugin_rel_l2_mean']:.3e}  inferred "
                      f"{row['inferred_latent_rel_l2_mean']:.3e}  ||r|| lm {row['resid_lm_med']:.2e} "
                      f"enc {row['resid_encoded_med']:.2e} f {row['f_norm_med']:.2e}  bnd "
                      f"{row['boundary_block_lm_med']:.1e}  acc/rej {row['lm_accepted_med']:.0f}/"
                      f"{row['lm_rejected_med']:.0f} {row['lm_reasons']}", flush=True)
                save("C")
    report["complete"] = True
    save("done")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
