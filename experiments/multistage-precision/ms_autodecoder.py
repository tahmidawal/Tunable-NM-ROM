"""Experiment (2): multi-stage-precision AUTO-DECODER + Levenberg-Marquardt
Gauss-Newton ROM on Poisson 2D.

Question: can an autoencoder-style decoder be trained multi-stage (Wang & Lai)
to high precision and then used to SOLVE the PDE as a ROM — how much decoder
precision survives the solve?  Staging needs a FIXED target, so the encoder is
never live: auto-decoder (DeepSDF-style learned per-snapshot latents).

  (A) Base FiLM decoder D0(x; z_i), latents z_i (dim K_LAT) learned jointly
      (lazy per-row Adam), f64, inverse-energy sample weights.  Reported:
      TRAIN fit at the learned latents (both metrics), and the held-out
      "finite-budget inferred-latent error": the SAME LM solver as the ROM,
      applied to the DATA-MISFIT residual dec(z)-u at all grid nodes,
      multi-start (mean train latent / nearest-in-source-parameter training
      latent), same attempt budget as the ROM arms.  Not a lower bound.
  (B) FREEZE the train latents; Wang-Lai stages over (x; z), frequency
      schedule in half-cycle units from the radial-mean spectrum probe
      (ms_parametric.freq_schedule).  Val error with (i) the stage-0 inferred
      latents held fixed and (ii) latents re-inferred through the full staged
      decoder (LM, same budget).
  (C) ROM on the held-out val sources: online the solver knows ONLY the source
      f and the decoder.  Residual = ghost-zero-Dirichlet 5-point FD operator
      applied to the decoded INTERIOR field minus f (boundary rows DROPPED;
      == the FOM operator, so a perfect decoder at some z zeros it exactly);
      the boundary block ||dec(z, boundary)|| is recorded separately.
      Collocation: full interior, and an m-node EQ-style random subset
      (stencil neighbours on the wall contribute 0).  LM with tolerance, NaN
      guard, budget accounting; inits: mean latent, nearest-source-parameter
      training latent, and a STAGED solve (stage-0 decoder for half the
      budget, then the full sum for the other half).  Per arm we record
      ||r(z_LM)||, ||r(z_oracle)||, ||f|| (restricted), field error at the
      oracle latent, latent norms and nearest-training-latent distances.

Provenance: report + pkl carry the config manifest; the phase-B report is
written before phase C starts; `complete` flag at the end; the held-out-derived
val latents live in a separate tainted file (never in the deployable pkl).

Usage: [K_LAT=4] [N=64] [N_TRAIN=512] [N_VAL=64] [N_STAGES=3] [STEPS=20000]
       [P_SUB=1024] [N_TEST=16] [GN_ITERS=60] [M_EQ=512] [Z_FF=0]
       python ms_autodecoder.py [outdir]
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
import optax

import ms_parametric as mp

K_LAT = int(os.environ.get("K_LAT", "4"))
N, N_TRAIN, N_VAL = mp.N, mp.N_TRAIN, mp.N_VAL
N_STAGES = int(os.environ.get("N_STAGES", "3"))
STEPS = mp.STEPS
LAT_LR = float(os.environ.get("LAT_LR", "5e-3"))
LAT_REG = float(os.environ.get("LAT_REG", "1e-4"))
ADAM_INFER_STEPS = int(os.environ.get("ADAM_INFER_STEPS", "1500"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
M_EQ = int(os.environ.get("M_EQ", "512"))
Z_FF = mp.Z_FF
SEED = mp.SEED
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))
F64 = jnp.float64
CONFIG = dict(mp.CONFIG, K_LAT=K_LAT, n_stages=N_STAGES, lat_lr=LAT_LR,
              lat_reg=LAT_REG, adam_infer_steps=ADAM_INFER_STEPS, n_test=N_TEST,
              gn_iters=GN_ITERS, m_eq=M_EQ)


# --------------------------- Levenberg-Marquardt ---------------------------

def lm_solve(rJ, rnorm, z0, budget, lam0=1e-6):
    """LM on min_z ||r(z)||^2.  budget = attempts (each = 1 residual eval;
    accepted attempts add 1 Jacobian eval).  Returns z, ||r||, accounting."""
    z = z0
    lam = lam0
    r, J = rJ(z)
    n_r, n_J = 1, 1
    rn = float(jnp.linalg.norm(r))
    acc = rej = 0
    reason = "budget"
    if not np.isfinite(rn):
        return z, rn, dict(accepted=0, rejected=0, n_resid_evals=1,
                           n_jac_evals=1, final_lambda=lam, reason="nan_at_init",
                           attempts=0)
    for attempt in range(1, budget + 1):
        H = J.T @ J
        g = J.T @ r
        D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(H.shape[0], dtype=F64)
        dz = jnp.linalg.solve(H + lam * D, -g)
        if not bool(jnp.all(jnp.isfinite(dz))):
            lam = min(lam * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "nan_step_lambda_max"; break
            continue
        z_new = z + dz
        rn_new = float(rnorm(z_new)); n_r += 1
        if np.isfinite(rn_new) and rn_new < rn:
            rel_dec = (rn - rn_new) / rn
            step = float(jnp.linalg.norm(dz)) / (1.0 + float(jnp.linalg.norm(z)))
            z, rn = z_new, rn_new
            r, J = rJ(z); n_J += 1; n_r += 1
            lam = max(lam / 3.0, 1e-12); acc += 1
            if rel_dec < 1e-12 or step < 1e-13:
                reason = "converged"; break
        else:
            lam = min(lam * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "lambda_max"; break
    return z, rn, dict(accepted=acc, rejected=rej, n_resid_evals=n_r,
                       n_jac_evals=n_J, final_lambda=float(lam), reason=reason,
                       attempts=attempt)


def make_data_misfit(stages, coords):
    dec = lambda z, u: mp.combined_apply(stages, z, coords) - u
    rJ = jax.jit(lambda z, u: (dec(z, u), jax.jacfwd(dec)(z, u)))
    rn = jax.jit(lambda z, u: jnp.linalg.norm(dec(z, u)))
    return rJ, rn


def infer_latents_lm(stages, coords, U_target, inits, budget):
    """Finite-budget latent inference by LM on the data misfit; inits: dict
    name -> (n, K) starting latents.  Returns per-start results + best-of."""
    rJ, rn = make_data_misfit(stages, coords)
    out = {}
    for name, Z0 in inits.items():
        Z, rels, accs = [], [], []
        for i in range(U_target.shape[0]):
            u = U_target[i]
            z, r, info = lm_solve(lambda zz: rJ(zz, u), lambda zz: rn(zz, u),
                                  Z0[i], budget)
            Z.append(np.asarray(z))
            rels.append(r / float(jnp.linalg.norm(u)))
            accs.append(info["accepted"])
        out[name] = {"Z": np.stack(Z), "rel": np.asarray(rels),
                     "acc_med": float(np.median(accs))}
    names = list(out)
    best_rel = np.min(np.stack([out[n]["rel"] for n in names]), axis=0)
    best_pick = np.argmin(np.stack([out[n]["rel"] for n in names]), axis=0)
    Zbest = np.stack([out[names[b]]["Z"][i] for i, b in enumerate(best_pick)])
    out["best"] = {"Z": Zbest, "rel": best_rel}
    return out


def infer_latents_adam(stages, coords, U_target, Z0, steps=ADAM_INFER_STEPS,
                       lr=2e-2):
    """Secondary: Adam on latents only, per-sample RELATIVE loss (normalized)."""
    msq = jnp.mean(U_target ** 2, axis=1)
    def total(Z):
        pred = jax.vmap(lambda z: mp.combined_apply(stages, z, coords))(Z)
        return jnp.sum(jnp.mean((pred - U_target) ** 2, axis=1) / msq)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=1e-4))
    st = opt.init(Z0)
    @jax.jit
    def step(Z, st):
        val, g = jax.value_and_grad(total)(Z)
        up, st = opt.update(g, st, Z)
        return optax.apply_updates(Z, up), st, val
    Z = Z0
    for _ in range(steps):
        Z, st, _ = step(Z, st)
    pred = jax.vmap(lambda z: mp.combined_apply(stages, z, coords))(Z)
    _, m_rel, per = mp.rel_metrics(pred, U_target)
    return np.asarray(Z), per


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}"
          f"  K_LAT={K_LAT}", flush=True)
    print("CONFIG " + json.dumps(CONFIG), flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"K{K_LAT}"
    U, z_true_all, coords, fom_res = mp.build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    zt_all = np.asarray(z_true_all)
    zt_tr, zt_va = zt_all[:N_TRAIN], zt_all[N_TRAIN:]
    w_tr = mp.sample_weights(U_tr)
    np_rng = np.random.default_rng(SEED)
    coll_rng = np.random.default_rng(SEED + 12345)     # collocation subsets only
    key = jax.random.PRNGKey(SEED + 100 + K_LAT)
    report = {"config": CONFIG, "fom_max_rel_residual": fom_res, "stages": [],
              "rom": [], "complete": False}
    # nearest training sample in SOURCE-parameter space (known to the ROM)
    nn_idx = np.argmin(((zt_va[:, None, :] - zt_tr[None, :, :]) ** 2).sum(-1), axis=1)

    def save(phase):
        with open(os.path.join(OUTDIR, f"ms_autodecoder_{tag}_report.json"), "w") as f:
            json.dump(dict(report, phase=phase), f, indent=2)

    # ---------------- (A) auto-decoder stage 0 ----------------
    key, k0, kz = jax.random.split(key, 3)
    Z_tr = 0.1 * jax.random.normal(kz, (N_TRAIN, K_LAT), dtype=F64)
    eps0 = float(jnp.sqrt(jnp.mean(U_tr ** 2)))
    n_freq = mp.freq_schedule(mp.dominant_radial_freq(U_tr, N), 0, N)
    print(f"stage 0 (auto-decoder, joint latents): eps={eps0:.3e} n_freq={n_freq}",
          flush=True)
    params0, Z_tr, adam_loss, secs = mp.fit_stage(
        k0, np_rng, coords, U_tr / eps0, w_tr, n_freq, Z_tr, k_lat=K_LAT,
        z_ff=Z_FF, learn_latents=True, lat_lr=LAT_LR, lat_reg=LAT_REG, tag="A")
    stages = [{"params": params0, "n_freq": n_freq, "eps": eps0, "z_ff": Z_FF}]
    z_mean = jnp.mean(Z_tr, axis=0)
    Z_tr_np = np.asarray(Z_tr)

    def eval_stage(k, e_tr_prev=None):
        pred_tr = mp.predict_all(stages, Z_tr, coords)
        e_tr = U_tr - pred_tr
        g_tr, m_tr, _ = mp.rel_metrics(pred_tr, U_tr)
        return e_tr, g_tr, m_tr

    e_tr, g_tr, m_tr = eval_stage(0)
    inits = {"mean": np.tile(np.asarray(z_mean), (N_VAL, 1)),
             "nearest": Z_tr_np[nn_idx]}
    inf0 = infer_latents_lm(stages, coords, U_va, inits, GN_ITERS)
    Zva0 = inf0["best"]["Z"]
    Za, per_adam = infer_latents_adam(stages, coords, U_va,
                                      jnp.asarray(inits["mean"]))
    row = {"stage": 0, "eps_in": eps0, "n_freq": n_freq,
           "adam_final_batch_loss": adam_loss, "secs": secs,
           "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
           "val_lm_inferred_mean_rel_l2": {n: float(inf0[n]["rel"].mean())
                                           for n in inf0},
           "val_lm_inferred_by_amp_quartile": mp.quartile_errors(
               inf0["best"]["rel"], U_va),
           "val_adam_inferred_mean_rel_l2": float(per_adam.mean()),
           "val_fixed_stage0_latents_mean_rel_l2": float(inf0["best"]["rel"].mean()),
           "latent_rms": float(jnp.sqrt(jnp.mean(Z_tr ** 2))),
           "latent_absmax": float(jnp.max(jnp.abs(Z_tr)))}
    report["stages"].append(row)
    print(f"  after stage 0: TRAIN global {g_tr:.3e} / mean-rel {m_tr:.3e}  "
          f"VAL LM-inferred best {row['val_lm_inferred_mean_rel_l2']['best']:.3e} "
          f"(mean-start {row['val_lm_inferred_mean_rel_l2']['mean']:.3e}, "
          f"nearest-start {row['val_lm_inferred_mean_rel_l2']['nearest']:.3e}); "
          f"Adam-inferred {per_adam.mean():.3e}; latent rms {row['latent_rms']:.2f}",
          flush=True)
    save("A")

    # ---------------- (B) frozen-latent staging ----------------
    for k in range(1, N_STAGES):
        eps = float(jnp.sqrt(jnp.mean(e_tr ** 2)))
        f_d = mp.dominant_radial_freq(e_tr, N)
        n_freq = mp.freq_schedule(f_d, n_freq, N)
        print(f"stage {k}: eps={eps:.3e}  f_d~{f_d:.1f} cyc/unit  n_freq={n_freq}",
              flush=True)
        key, sub = jax.random.split(key)
        params, _, adam_loss, secs = mp.fit_stage(
            sub, np_rng, coords, e_tr / eps, w_tr, n_freq, Z_tr, k_lat=K_LAT,
            z_ff=Z_FF, learn_latents=False, tag=f"B{k}")
        stages.append({"params": params, "n_freq": n_freq, "eps": eps, "z_ff": Z_FF})
        e_tr, g_tr, m_tr = eval_stage(k)
        pred_fix = mp.predict_all(stages, jnp.asarray(Zva0), coords)
        g_fix, m_fix, per_fix = mp.rel_metrics(pred_fix, U_va)
        infk = infer_latents_lm(stages, coords, U_va,
                                {"mean": inits["mean"], "nearest": inits["nearest"],
                                 "stage0lat": Zva0}, GN_ITERS)
        row = {"stage": k, "eps_in": eps, "f_d_cyc": f_d, "n_freq": n_freq,
               "adam_final_batch_loss": adam_loss, "secs": secs,
               "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
               "val_fixed_stage0_latents_global_rel": g_fix,
               "val_fixed_stage0_latents_mean_rel_l2": m_fix,
               "val_lm_inferred_mean_rel_l2": {n: float(infk[n]["rel"].mean())
                                               for n in infk},
               "val_lm_inferred_by_amp_quartile": mp.quartile_errors(
                   infk["best"]["rel"], U_va)}
        report["stages"].append(row)
        print(f"  after stage {k}: TRAIN global {g_tr:.3e} / mean-rel {m_tr:.3e}   "
              f"VAL fixed-lat {m_fix:.3e}   VAL LM re-inferred best "
              f"{row['val_lm_inferred_mean_rel_l2']['best']:.3e}", flush=True)
        with open(os.path.join(OUTDIR, f"ms_autodecoder_{tag}_stages.pkl"), "wb") as f:
            pickle.dump({"config": CONFIG, "stages": mp.stages_to_np(stages),
                         "z_tr": Z_tr_np}, f)
        save("B")
    with open(os.path.join(OUTDIR, f"ms_autodecoder_{tag}_stages.pkl"), "wb") as f:
        pickle.dump({"config": CONFIG, "stages": mp.stages_to_np(stages),
                     "z_tr": Z_tr_np}, f)
    with open(os.path.join(OUTDIR, f"ms_autodecoder_{tag}_val_latents_TAINTED.pkl"), "wb") as f:
        pickle.dump({"config": CONFIG, "z_va0_lm_best": Zva0,
                     "note": "derived from held-out fields; never use in ROM"}, f)
    save("B-done")

    # ---------------- (C) ROM on held-out sources ----------------
    cx, cy, w, a, _ = mp.sample_params()
    sl = slice(N_TRAIN, N_TRAIN + N_TEST)
    cx, cy, w, a = cx[sl], cy[sl], w[sl], a[sl]
    U_test = np.asarray(U_va[:N_TEST])
    test_norms = np.linalg.norm(U_test, axis=1)
    dx = 1.0 / (N - 1)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    bmask = np.zeros((N, N), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bpts = jnp.asarray(np.stack([X[bmask], Y[bmask]], axis=1))

    def stencil(ix, iy):
        """5 point sets (centre, +x, -x, +y, -y) and a 0/1 mask that zeroes
        neighbours lying ON the wall (ghost-zero Dirichlet)."""
        offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        pts, keep = [], []
        for ox, oy in offs:
            jx, jy = ix + ox, iy + oy
            pts.append(np.stack([jx * dx, jy * dx], axis=1))
            keep.append(~((jx == 0) | (jx == N - 1) | (jy == 0) | (jy == N - 1)))
        return jnp.asarray(np.stack(pts)), jnp.asarray(np.stack(keep).astype(float))

    ii, jj = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1), indexing="ij")
    ix_full, iy_full = ii.reshape(-1), jj.reshape(-1)
    sub = coll_rng.choice(len(ix_full), size=min(M_EQ, len(ix_full)), replace=False)
    colls = {"full": (ix_full, iy_full), f"m{M_EQ}": (ix_full[sub], iy_full[sub])}

    def make_solver(n_stages, pts, keep):
        st = stages[:n_stages]
        dec = lambda z, xy: mp.combined_apply(st, z, xy)

        def residual(z, f_vals):
            u = dec(z, pts.reshape(-1, 2)).reshape(5, -1) * keep
            lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (dx * dx)
            return -lap - f_vals

        rJ = jax.jit(lambda z, f: (residual(z, f), jax.jacfwd(residual)(z, f)))
        rn = jax.jit(lambda z, f: jnp.linalg.norm(residual(z, f)))
        bnorm = jax.jit(lambda z: jnp.linalg.norm(dec(z, bpts)))
        dec_full = jax.jit(lambda z: dec(z, coords))
        return rJ, rn, bnorm, dec_full

    # oracle (finite-budget inferred latents) on the test subset per n_stages,
    # same LM solver / same budget, multi-start
    oracle = {}
    for n_stages in range(1, len(stages) + 1):
        o = infer_latents_lm(stages[:n_stages], coords, U_va[:N_TEST],
                             {"mean": inits["mean"][:N_TEST],
                              "nearest": inits["nearest"][:N_TEST]}, GN_ITERS)
        oracle[n_stages] = o

    for cname, (ix, iy) in colls.items():
        pts, keep = stencil(ix, iy)
        F_rows = [jnp.asarray(a[i] * np.exp(-((ix * dx - cx[i]) ** 2
                                              + (iy * dx - cy[i]) ** 2)
                                            / (2 * w[i] ** 2)))
                  for i in range(N_TEST)]
        solvers = {ns: make_solver(ns, pts, keep) for ns in range(1, len(stages) + 1)}
        for n_stages in range(1, len(stages) + 1):
            rJ, rn, bnorm, dec_full = solvers[n_stages]
            for init_name in ("mean", "nearest", "staged"):
                if init_name == "staged" and n_stages == 1:
                    continue
                per = {"err": [], "err_oracle": [], "r_lm": [], "r_oracle": [],
                       "f_norm": [], "b_lm": [], "acc": [], "rej": [], "reason": [],
                       "lam": [], "z_norm": [], "z_nn_dist": [], "z_norm_oracle": []}
                t0 = time.time()
                for i in range(N_TEST):
                    f = F_rows[i]
                    if init_name == "mean":
                        z0 = z_mean; budget = GN_ITERS
                    elif init_name == "nearest":
                        z0 = jnp.asarray(Z_tr_np[nn_idx[i]]); budget = GN_ITERS
                    else:                    # staged: half budget on stage-0
                        rJ1, rn1, _, _ = solvers[1]
                        z0, _, _ = lm_solve(lambda zz: rJ1(zz, f), lambda zz: rn1(zz, f),
                                            jnp.asarray(Z_tr_np[nn_idx[i]]), GN_ITERS // 2)
                        budget = GN_ITERS - GN_ITERS // 2
                    z, r_lm, info = lm_solve(lambda zz: rJ(zz, f), lambda zz: rn(zz, f),
                                             z0, budget)
                    pred = np.asarray(dec_full(z))
                    # oracle latent for the matching start (equal budget arms)
                    oname = "nearest" if init_name != "mean" else "mean"
                    z_or = jnp.asarray(oracle[n_stages][oname]["Z"][i])
                    per["err"].append(np.linalg.norm(pred - U_test[i]) / test_norms[i])
                    per["err_oracle"].append(float(oracle[n_stages][oname]["rel"][i]))
                    per["r_lm"].append(r_lm)
                    per["r_oracle"].append(float(rn(z_or, f)))
                    per["f_norm"].append(float(jnp.linalg.norm(f)))
                    per["b_lm"].append(float(bnorm(z)))
                    per["acc"].append(info["accepted"]); per["rej"].append(info["rejected"])
                    per["reason"].append(info["reason"]); per["lam"].append(info["final_lambda"])
                    zn = np.asarray(z)
                    per["z_norm"].append(float(np.linalg.norm(zn)))
                    per["z_nn_dist"].append(float(np.min(np.linalg.norm(Z_tr_np - zn, axis=1))))
                    per["z_norm_oracle"].append(float(jnp.linalg.norm(z_or)))
                e = np.asarray(per["err"])
                row = {"colloc": cname, "n_stages": n_stages, "init": init_name,
                       "budget_attempts": GN_ITERS,
                       "rom_rel_l2_mean": float(e.mean()), "rom_rel_l2_med": float(np.median(e)),
                       "rom_rel_l2_max": float(e.max()),
                       "oracle_start_used": "nearest" if init_name != "mean" else "mean",
                       "oracle_rel_l2_mean": float(np.mean(per["err_oracle"])),
                       "oracle_best_of_starts_rel_l2_mean": float(oracle[n_stages]["best"]["rel"].mean()),
                       "resid_lm_med": float(np.median(per["r_lm"])),
                       "resid_oracle_med": float(np.median(per["r_oracle"])),
                       "f_norm_med": float(np.median(per["f_norm"])),
                       "boundary_block_lm_med": float(np.median(per["b_lm"])),
                       "lm_accepted_med": float(np.median(per["acc"])),
                       "lm_rejected_med": float(np.median(per["rej"])),
                       "lm_final_lambda_med": float(np.median(per["lam"])),
                       "lm_reasons": {r: per["reason"].count(r) for r in set(per["reason"])},
                       "z_norm_med": float(np.median(per["z_norm"])),
                       "z_nn_dist_med": float(np.median(per["z_nn_dist"])),
                       "z_norm_oracle_med": float(np.median(per["z_norm_oracle"])),
                       "per_sample_rom_rel_l2": [float(v) for v in e],
                       "secs": time.time() - t0}
                report["rom"].append(row)
                print(f"RESULT {tag} colloc={cname:5s} stages={n_stages} init={init_name:7s}  "
                      f"ROM {row['rom_rel_l2_mean']:.3e} (med {row['rom_rel_l2_med']:.3e}, "
                      f"max {row['rom_rel_l2_max']:.3e})  oracle({row['oracle_start_used']}) "
                      f"{row['oracle_rel_l2_mean']:.3e}  ||r||: lm {row['resid_lm_med']:.2e} "
                      f"oracle {row['resid_oracle_med']:.2e} f {row['f_norm_med']:.2e}  "
                      f"bnd {row['boundary_block_lm_med']:.1e}  acc/rej "
                      f"{row['lm_accepted_med']:.0f}/{row['lm_rejected_med']:.0f} "
                      f"{row['lm_reasons']}", flush=True)
                save("C")
    report["complete"] = True
    save("done")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
