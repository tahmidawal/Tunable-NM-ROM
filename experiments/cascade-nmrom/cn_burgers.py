"""Generalizable cascade + LATENT TIME-STEPPING LM ROM on BURGERS-2D (pilot).

Data: the burgers2d testbed FOM (implicit backward-Euler, guarded Newton /
BiCGStab, f64; family z=(cx,cy,w,a,log nu)) — snapshots U (n_traj, T+1, n^2)
with T=NUM_STEPS=50, dt=0.005.  TRAIN = first N_TRAIN trajectories, HELD-OUT =
last N_VAL (never fitted); ROM on the first N_TEST held-out trajectories.

Model: decoder D(x, y; z) — NO time input; time is carried by the latent
trajectory z_{i,n}.  Stage 0 is an AUTO-DECODER over all snapshots (rows =
(traj, time)) with:  z_{i,0} = E(u0_i on the 16x16 lattice)  [encoder, the
generalizable entry point], free latents z_{i,n>=1} (lazy per-row Adam) with a
temporal-smoothness penalty T_SMOOTH*||z_{i,n} - z_{i,n-1}||^2.  Then E and
all latents are FROZEN and residual stages are stacked (fixed targets), with
the probe gate as in Poisson.  The physics parameter nu is NOT a decoder
input: the ROM knows it and uses it in the residual.

ROM (held-out): z_0 = E(u0); for n = 0..T-1 solve with LM
  min_z || R_int( D(z) ; D(z_n), nu ) ||,   R = the FOM's backward-Euler
residual (interior rows; decoded fields have their wall nodes zeroed =
ghost-zero Dirichlet), warm start z = z_n.  Reported per number of stages S:
rollout rel-L2 vs the FOM trajectory (mean over held-out trajectories and
all T+1 slices, plus per-time), the encoded plug-in error at t=0, and the
finite-budget inferred-latent error per snapshot (data-misfit LM, init from
the previous solved latent) as the representation reference.

Scope note: this is the PILOT arm — configs are kept small (N=32, ~128
trajectories) so the whole thing fits one A100 job; the Poisson script is the
fully-instrumented experiment.

Usage: [K_LAT=8] [N=32] [N_TRAIN=128] [N_VAL=16] [N_STAGES=3] [STEPS=20000]
       [P_SUB=512] [BATCH=32] [N_TEST=8] [GN_ITERS=30] [ROM_STEPS=50]
       [K_EXTRA=0] [GATE=0] python cn_burgers.py [outdir]
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
import burgers2d_film as bf                      # FOM + family (f64)

K_LAT = cn.K_LAT
N = int(os.environ.get("N", "32"))
N_TRAIN = int(os.environ.get("N_TRAIN", "128"))
N_VAL = int(os.environ.get("N_VAL", "16"))
STEPS = mp.STEPS
N_STAGES = int(os.environ.get("N_STAGES", "3"))
N_TEST = int(os.environ.get("N_TEST", "8"))
GN_ITERS = cn.GN_ITERS
ROM_STEPS = min(int(os.environ.get("ROM_STEPS", str(bf.NUM_STEPS))), bf.NUM_STEPS)
T1 = bf.NUM_STEPS + 1
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
F64 = jnp.float64
CONFIG = dict(mp.CONFIG, **cn.COMMON_CONFIG, pde="burgers2d", N=N, n_train=N_TRAIN,
              n_val=N_VAL, K_LAT=K_LAT, n_stages=N_STAGES, n_test=N_TEST,
              rom_steps=ROM_STEPS, dt=bf.DT, num_steps=bf.NUM_STEPS,
              newton_iters=bf.NEWTON_ITERS)
TAG = f"enc_K{K_LAT}_X{cn.K_EXTRA}"


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}  {TAG}",
          flush=True)
    print("CONFIG " + json.dumps(CONFIG), flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    # --- data (bf reads N_TRAIN/N_VAL from env; assert consistency) ---
    assert bf.N_TRAIN == N_TRAIN and bf.N_VAL == N_VAL, "set N_TRAIN/N_VAL in env"
    U_all, z_true, cx, cy, w, a, nu = bf.build_trajectories(N)   # (m, T+1, n^2)
    m_all = U_all.shape[0]
    n_pts = N * N
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    coords = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1))
    bmask_np = np.asarray(bf.boundary_mask(N))
    bmask = jnp.asarray(bmask_np.reshape(-1))
    # rows = (traj, time)
    U_rows = jnp.asarray(U_all.reshape(m_all * T1, n_pts))
    tr_rows = np.arange(N_TRAIN * T1)
    ho_rows = np.arange(N_TRAIN * T1, m_all * T1)
    U_tr, U_ho = U_rows[tr_rows], U_rows[ho_rows]
    row2traj = np.repeat(np.arange(m_all), T1)
    row_t = np.tile(np.arange(T1), m_all)
    enc_mask_all = (row_t == 0)
    prev_row_all = np.where(row_t >= 1, np.arange(m_all * T1) - 1, -1)
    # encoder input: u0 on the lattice
    F_lat = cn.gaussian_on_lattice(cx, cy, w, a)
    mu, sd = cn.standardize_fit(F_lat[:N_TRAIN])
    inp = jnp.asarray((F_lat - mu) / sd)
    w_tr = mp.sample_weights(U_tr)
    np_rng = np.random.default_rng(cn.SEED)
    key = jax.random.PRNGKey(cn.SEED + 11 + K_LAT)
    report = {"config": CONFIG, "stages": [], "probes": [], "rom": [], "complete": False}

    def save(phase):
        with open(os.path.join(OUTDIR, f"cn_burgers_{TAG}_report.json"), "w") as f:
            json.dump(dict(report, phase=phase), f, indent=2)

    def dump_pkl(enc0, stages, Z_tr):
        with open(os.path.join(OUTDIR, f"cn_burgers_{TAG}_stages.pkl"), "wb") as f:
            pickle.dump({"config": CONFIG, "encoder": jax.tree_util.tree_map(np.asarray, enc0),
                         "lattice_mu": mu, "lattice_sd": sd,
                         "stages": cn.stages_to_np(stages),
                         "z_train_rows": np.asarray(Z_tr)}, f)

    # ---------------- stage 0 ----------------
    eps0 = float(jnp.sqrt(jnp.mean(U_tr ** 2)))
    n_freq = mp.freq_schedule(mp.dominant_radial_freq(U_tr[::7], N), 0, N)
    print(f"stage 0: rows={len(tr_rows)} eps={eps0:.3e} n_freq={n_freq}", flush=True)
    key, k0 = jax.random.split(key)
    dec0, enc0, Z_tr, loss0, secs0 = cn.fit_stage0(
        k0, np_rng, coords, U_tr / eps0, w_tr, n_freq, inp, row2traj[tr_rows],
        enc_mask_all[tr_rows], K_LAT, steps=STEPS, prev_row=prev_row_all[tr_rows], tag="A")
    stages = [{"params": dec0, "n_freq": n_freq, "eps": eps0, "k_extra": 0}]

    def extras_for(rows):
        return [None if s.get("k_extra", 0) == 0 else
                jax.vmap(lambda r: cn.enc_apply(s["enc_extra"], inp[row2traj[r]]))(jnp.asarray(rows))
                for s in stages]

    def eval_train():
        pred = cn.predict_rows(stages, Z_tr, extras_for(tr_rows), coords)
        g, m_, _ = mp.rel_metrics(pred, U_tr)
        return pred, g, m_

    pred_tr, g_tr, m_tr = eval_train()
    e_tr = U_tr - pred_tr
    # held-out t=0 plug-in (encoded) error — the only held-out number that
    # needs no solve; the rest of the trajectory needs the ROM
    Z_ho0 = jax.vmap(lambda r: cn.enc_apply(enc0, inp[r]))(jnp.arange(N_TRAIN, m_all))
    ho0_rows = ho_rows[row_t[ho_rows] == 0]
    pred_ho0 = cn.predict_rows(stages, Z_ho0, extras_for(ho0_rows), coords)
    g0, m0, _ = mp.rel_metrics(pred_ho0, U_rows[ho0_rows])
    pr = cn.probe(e_tr, np.asarray(Z_tr)); pr["after_stage"] = 0
    prf = cn.probe(U_tr, np.asarray(Z_tr)); prf["after_stage"] = -1
    report["probes"] += [prf, pr]
    report["stages"].append({"stage": 0, "eps_in": eps0, "n_freq": n_freq,
                             "adam_final_batch_loss": loss0, "secs": secs0,
                             "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
                             "heldout_t0_encoded_global_rel": g0,
                             "heldout_t0_encoded_mean_rel_l2": m0,
                             "latent_rms": float(jnp.sqrt(jnp.mean(Z_tr ** 2)))})
    print(f"  after stage 0: TRAIN {g_tr:.3e}/{m_tr:.3e}  HELD-OUT t0 encoded {g0:.3e}/{m0:.3e}"
          f"  probe eff_rank {pr['eff_rank']:.1f} pod8 {pr['pod8_rel_err']:.2e} nn5 "
          f"{pr['nn5_corr']:.2f} stop? {pr['stop_suggested']}", flush=True)
    dump_pkl(enc0, stages, Z_tr); save("A")

    # ---------------- stages 1..S-1 ----------------
    stopped_at = None
    for k in range(1, N_STAGES):
        if cn.GATE and pr["stop_suggested"]:
            stopped_at = k; print(f"GATE: stopping before stage {k}", flush=True); break
        eps = float(jnp.sqrt(jnp.mean(e_tr ** 2)))
        f_d = mp.dominant_radial_freq(e_tr[::7], N)
        n_freq = mp.freq_schedule(f_d, n_freq, N)
        print(f"stage {k}: eps={eps:.3e} f_d~{f_d:.1f} n_freq={n_freq}", flush=True)
        key, sub = jax.random.split(key)
        params, enc_x, loss, secs = cn.fit_residual_stage(
            sub, np_rng, coords, e_tr / eps, w_tr, n_freq, Z_tr, inp, row2traj[tr_rows],
            K_LAT, k_extra=cn.K_EXTRA, steps=STEPS, tag=f"B{k}")
        st = {"params": params, "n_freq": n_freq, "eps": eps, "k_extra": cn.K_EXTRA}
        if cn.K_EXTRA > 0:
            st["enc_extra"] = enc_x
        stages.append(st)
        pred_tr, g_tr, m_tr = eval_train()
        e_tr = U_tr - pred_tr
        pred_ho0 = cn.predict_rows(stages, Z_ho0, extras_for(ho0_rows), coords)
        g0, m0, _ = mp.rel_metrics(pred_ho0, U_rows[ho0_rows])
        pr = cn.probe(e_tr, np.asarray(Z_tr)); pr["after_stage"] = k
        report["probes"].append(pr)
        report["stages"].append({"stage": k, "eps_in": eps, "f_d_cyc": f_d, "n_freq": n_freq,
                                 "adam_final_batch_loss": loss, "secs": secs,
                                 "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
                                 "heldout_t0_encoded_global_rel": g0,
                                 "heldout_t0_encoded_mean_rel_l2": m0})
        print(f"  after stage {k}: TRAIN {g_tr:.3e}/{m_tr:.3e}  HELD-OUT t0 encoded "
              f"{g0:.3e}/{m0:.3e}  probe eff_rank {pr['eff_rank']:.1f} pod8 "
              f"{pr['pod8_rel_err']:.2e} nn5 {pr['nn5_corr']:.2f} stop? {pr['stop_suggested']}",
              flush=True)
        dump_pkl(enc0, stages, Z_tr); save("B")
    report["gate_stopped_before_stage"] = stopped_at
    save("B-done")

    # ---------------- ROM: latent time stepping on held-out ----------------
    _, fom_residual = bf.make_rollout(N)          # residual(u_flat, u_prev_flat, nu)
    int_mask = bmask                               # 1 interior / 0 wall
    test_traj = np.arange(N_TRAIN, N_TRAIN + N_TEST)
    Z0_test = np.asarray(Z_ho0[:N_TEST])

    for S in range(1, len(stages) + 1):
        st = stages[:S]
        def dec(z, cs):
            return cn.cascade_apply(st, z, cs[:S], coords) * bmask     # ghost-zero walls
        def res_step(z, cs, u_prev, nu_):
            return fom_residual(dec(z, cs), u_prev, nu_) * int_mask   # interior rows
        rJ = jax.jit(lambda z, cs, up, nu_: (res_step(z, cs, up, nu_),
                                             jax.jacfwd(res_step)(z, cs, up, nu_)))
        rn = jax.jit(lambda z, cs, up, nu_: jnp.linalg.norm(res_step(z, cs, up, nu_)))
        dec_j = jax.jit(dec)
        # data-misfit inference (representation reference), init = previous inferred latent
        dmis = lambda z, cs, u: cn.cascade_apply(st, z, cs[:S], coords) - u
        rJd = jax.jit(lambda z, cs, u: (dmis(z, cs, u), jax.jacfwd(dmis)(z, cs, u)))
        rnd = jax.jit(lambda z, cs, u: jnp.linalg.norm(dmis(z, cs, u)))
        per_time_err = np.zeros((N_TEST, ROM_STEPS + 1))
        per_time_inf = np.zeros((N_TEST, ROM_STEPS + 1))
        iters, reasons, resids = [], [], []
        t0 = time.time()
        for ti, tj in enumerate(test_traj):
            cs = extras_for(np.array([tj * T1]))
            cs = [None if c is None else c[0] for c in cs]
            nu_j = jnp.asarray(nu[tj])
            traj = U_all[tj]                                   # (T+1, n^2)
            nrm = np.linalg.norm(traj, axis=1)
            z = jnp.asarray(Z0_test[ti]); z_inf = z
            u_prev = dec_j(z, cs)
            per_time_err[ti, 0] = np.linalg.norm(np.asarray(u_prev) - traj[0]) / nrm[0]
            z_inf, r_i, _ = lm_solve(lambda zz: rJd(zz, cs, jnp.asarray(traj[0])),
                                     lambda zz: rnd(zz, cs, jnp.asarray(traj[0])), z_inf, GN_ITERS)
            per_time_inf[ti, 0] = r_i / nrm[0]
            for n in range(1, ROM_STEPS + 1):
                z, r_lm, info = lm_solve(lambda zz: rJ(zz, cs, u_prev, nu_j),
                                         lambda zz: rn(zz, cs, u_prev, nu_j), z, GN_ITERS)
                u_prev = dec_j(z, cs)
                per_time_err[ti, n] = np.linalg.norm(np.asarray(u_prev) - traj[n]) / nrm[n]
                iters.append(info["accepted"]); reasons.append(info["reason"]); resids.append(r_lm)
                z_inf, r_i, _ = lm_solve(lambda zz: rJd(zz, cs, jnp.asarray(traj[n])),
                                         lambda zz: rnd(zz, cs, jnp.asarray(traj[n])), z_inf,
                                         GN_ITERS)
                per_time_inf[ti, n] = r_i / nrm[n]
        row = {"n_stages": S, "init": "encoded_u0", "budget_attempts_per_step": GN_ITERS,
               "rom_steps": ROM_STEPS,
               "rom_rel_l2_mean_all_t": float(per_time_err.mean()),
               "rom_rel_l2_final_t": float(per_time_err[:, -1].mean()),
               "rom_rel_l2_per_time": [float(v) for v in per_time_err.mean(0)],
               "encoded_t0_rel_l2": float(per_time_err[:, 0].mean()),
               "inferred_latent_rel_l2_mean_all_t": float(per_time_inf.mean()),
               "inferred_latent_rel_l2_per_time": [float(v) for v in per_time_inf.mean(0)],
               "lm_accepted_med_per_step": float(np.median(iters)),
               "lm_reasons": {r: reasons.count(r) for r in set(reasons)},
               "resid_lm_med": float(np.median(resids)),
               "per_traj_rom_rel_l2_all_t": [float(v) for v in per_time_err.mean(1)],
               "secs": time.time() - t0}
        report["rom"].append(row)
        print(f"RESULT {TAG} stages={S}  ROM rollout {row['rom_rel_l2_mean_all_t']:.3e} "
              f"(final-t {row['rom_rel_l2_final_t']:.3e}, t0 encoded {row['encoded_t0_rel_l2']:.3e})"
              f"  inferred-latent {row['inferred_latent_rel_l2_mean_all_t']:.3e}  "
              f"acc/step {row['lm_accepted_med_per_step']:.0f} {row['lm_reasons']}", flush=True)
        save("C")
    report["complete"] = True
    save("done")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
