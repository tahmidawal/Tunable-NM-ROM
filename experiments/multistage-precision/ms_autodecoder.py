"""Experiment D: multi-stage-precision AUTO-DECODER + Gauss-Newton ROM (Poisson 2D).

The user's question: can an autoencoder-style decoder be trained multi-stage
(Wang & Lai) to high precision and then used to SOLVE the PDE as a ROM — and
how much of the decoder precision survives the solve?

Staging needs a FIXED target, so the encoder is never live during staging.
Auto-decoder formulation (DeepSDF-style latents), three phases:

  (A) Base: FiLM decoder D0(x; z_i) with LEARNED per-snapshot latents z_i
      (dim K_LAT, family has 4 true params), latents + weights jointly
      optimized (Adam, f64) on target U/eps0.  Manifold/representation floor
      = train fit error at the learned latents.  Val latents come from
      auto-decoder INFERENCE (Adam on z only against the val field).
  (B) FREEZE the train latents; Wang-Lai stages over (x; z): stage j fits
      r_j(x; z_i) = U_i - sum_{l<j} eps_l D_l(x; z_i), normalized by RMS,
      fresh FiLM net with Fourier bandwidth from the residual's dominant
      frequency.  Val error reported with (i) stage-0-inferred latents held
      fixed and (ii) latents re-inferred through the full staged decoder
      (the "oracle-latent" floor the ROM is measured against).
  (C) ROM on the held-out val sources: only the source f(x; cx,cy,w,a) and the
      decoder are known online.  min_z ||discrete FD residual(D(x; z))|| via
      Levenberg-Marquardt Gauss-Newton (adaptive damping, accept/reject) on
      the K_LAT-dim latent, full-interior collocation and an m-point EQ-style
      random subset.  Latent init: mean train latent (cold) and the latent of
      the nearest training sample in SOURCE-parameter space (the ROM knows f,
      hence its parameters).  Also a STAGED solve: stage-0 decoder first, then
      refine with the full sum.  All decoders/algebra f64.

Usage: [K_LAT=4] [N=64] [N_TRAIN=512] [N_VAL=64] [N_STAGES=3] [STEPS=25000]
       [N_TEST=16] python ms_autodecoder.py [outdir]
Reuses family / FOM / FiLM building blocks from ms_parametric.py.
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
N = mp.N
N_TRAIN, N_VAL = mp.N_TRAIN, mp.N_VAL
N_STAGES = int(os.environ.get("N_STAGES", "3"))
STEPS = int(os.environ.get("STEPS", "25000"))
BATCH = mp.BATCH
HIDDEN, N_LAYERS = mp.HIDDEN, mp.N_LAYERS
PEAK_LR = mp.PEAK_LR
P_SUB = mp.P_SUB
LAT_LR = float(os.environ.get("LAT_LR", "5e-3"))
LAT_REG = float(os.environ.get("LAT_REG", "1e-4"))
INFER_STEPS = int(os.environ.get("INFER_STEPS", "1500"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
M_EQ = int(os.environ.get("M_EQ", "512"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))
F64 = jnp.float64


# ------------------------- FiLM net with K_LAT latent input -------------------------

def init_film_net(key, n_freq):
    d_in = 2 * (2 * n_freq + 1)
    keys = jax.random.split(key, N_LAYERS + 4)
    trunk = [mp.init_dense(keys[0], d_in, HIDDEN)]
    for i in range(1, N_LAYERS):
        trunk.append(mp.init_dense(keys[i], HIDDEN, HIDDEN))
    out = mp.init_dense(keys[N_LAYERS], HIDDEN, 1)
    z_embed = mp.init_dense(keys[N_LAYERS + 1], K_LAT, 64)
    film = mp.init_dense(keys[N_LAYERS + 2], 64, N_LAYERS * 2 * HIDDEN)
    film["W"] = film["W"] * 0.01
    return {"trunk": trunk, "out": out, "z_embed": z_embed, "film": film}


film_apply = mp.film_apply          # generic in the latent dim
combined_apply = mp.combined_apply


def train_stage(key, np_rng, coords, target, n_freq, z_tr, learn_latents,
                tag):
    """Fit one FiLM stage to `target` (n_train, n^2), RMS ~ 1.
    learn_latents=True -> joint (weights, latents) Adam (phase A);
    False -> weights only, latents frozen (phase B)."""
    params = init_film_net(key, n_freq)
    sched = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, max(1, STEPS // 20), STEPS, end_value=1e-9)
    opt = optax.adamw(sched, weight_decay=1e-6)
    state = opt.init(params)
    lat_sched = optax.warmup_cosine_decay_schedule(
        0.0, LAT_LR, max(1, STEPS // 20), STEPS, end_value=1e-9)
    lat_opt = optax.adam(lat_sched)
    lat_state = lat_opt.init(z_tr)

    def loss_fn(ps, z_b, t_b, pidx):
        pred = jax.vmap(lambda zi: film_apply(ps, zi, coords[pidx], n_freq))(z_b)
        return jnp.mean((pred - t_b[:, pidx]) ** 2)

    @jax.jit
    def step_w(ps, st, z_b, t_b, pidx):
        val, g = jax.value_and_grad(loss_fn)(ps, z_b, t_b, pidx)
        up, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, up), st, val

    @jax.jit
    def step_wz(ps, st, z_all, lst, bi, t_b, pidx):
        z_b = z_all[bi]
        def lz(ps_, z_b_):
            return loss_fn(ps_, z_b_, t_b, pidx) + LAT_REG * jnp.mean(z_b_ ** 2)
        val, (gp, gz) = jax.value_and_grad(lz, argnums=(0, 1))(ps, z_b)
        up, st = opt.update(gp, st, ps)
        ps = optax.apply_updates(ps, up)
        gz_full = jnp.zeros_like(z_all).at[bi].set(gz)
        upz, lst = lat_opt.update(gz_full, lst, z_all)
        z_all = optax.apply_updates(z_all, upz)
        return ps, st, z_all, lst, val

    n_pts = coords.shape[0]
    t0 = time.time()
    for it in range(STEPS):
        bi = np_rng.choice(N_TRAIN, size=BATCH, replace=False)
        pidx = (jnp.arange(n_pts) if P_SUB <= 0 else
                jnp.asarray(np_rng.choice(n_pts, size=P_SUB, replace=False)))
        if learn_latents:
            params, state, z_tr, lat_state, val = step_wz(
                params, state, z_tr, lat_state, jnp.asarray(bi), target[bi], pidx)
        else:
            params, state, val = step_w(params, state, z_tr[bi], target[bi], pidx)
        if it % 5000 == 0:
            print(f"  [{tag}] step {it:6d}  loss {float(val):.3e}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"  [{tag}] trained {STEPS} steps in {time.time()-t0:.0f}s "
          f"(final batch loss {float(val):.3e})", flush=True)
    return params, z_tr


def infer_latents(stages, coords, U_target, z_init, steps=INFER_STEPS,
                  lr=2e-2):
    """Auto-decoder inference: Adam on latents only, decoder frozen, batched
    over samples.  Returns latents and per-sample rel-L2."""
    def one_loss(z, u):
        pred = combined_apply(stages, z, coords)
        return jnp.mean((pred - u) ** 2)
    total = lambda Z: jnp.sum(jax.vmap(one_loss)(Z, U_target))
    sched = optax.cosine_decay_schedule(lr, steps, alpha=1e-4)
    opt = optax.adam(sched)
    st = opt.init(z_init)

    @jax.jit
    def step(Z, st):
        val, g = jax.value_and_grad(total)(Z)
        up, st = opt.update(g, st, Z)
        return optax.apply_updates(Z, up), st, val

    Z = z_init
    for _ in range(steps):
        Z, st, _ = step(Z, st)
    pred = jax.vmap(lambda z: combined_apply(stages, z, coords))(Z)
    rel = jnp.linalg.norm(pred - U_target, axis=1) / jnp.linalg.norm(
        U_target, axis=1)
    return Z, np.asarray(rel)


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}"
          f"  K_LAT={K_LAT}", flush=True)
    nyq = (N - 1) // 2
    U, z_true_all, coords = mp.build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_true_va = np.asarray(z_true_all[N_TRAIN:])
    u_norm_tr = float(jnp.sqrt(jnp.mean(U_tr ** 2)))
    va_norms = np.asarray(jnp.linalg.norm(U_va, axis=1))
    np_rng = np.random.default_rng(SEED)
    key = jax.random.PRNGKey(SEED + 100 + K_LAT)
    report = {"K_LAT": K_LAT, "N": N, "n_train": N_TRAIN, "n_val": N_VAL,
              "steps": STEPS, "p_sub": P_SUB, "hidden": HIDDEN, "n_layers": N_LAYERS,
              "lat_lr": LAT_LR, "lat_reg": LAT_REG, "seed": SEED,
              "stages": [], "rom": []}

    def train_resid(stages, z_tr):
        preds = []
        for s in range(0, N_TRAIN, 64):
            e = min(s + 64, N_TRAIN)
            preds.append(jax.vmap(
                lambda zi: combined_apply(stages, zi, coords))(z_tr[s:e]))
        return U_tr - jnp.concatenate(preds, axis=0)

    # ---------------- (A) auto-decoder stage 0 ----------------
    key, k0, kz = jax.random.split(key, 3)
    z_tr = 0.1 * jax.random.normal(kz, (N_TRAIN, K_LAT), dtype=F64)
    eps0 = float(jnp.sqrt(jnp.mean(U_tr ** 2)))
    n_freq = 16
    print(f"stage 0 (auto-decoder, joint latents): eps={eps0:.3e} "
          f"n_freq={n_freq}", flush=True)
    params0, z_tr = train_stage(k0, np_rng, coords, U_tr / eps0, n_freq, z_tr,
                                True, "A")
    stages = [{"params": params0, "n_freq": n_freq, "eps": eps0}]
    e_tr = train_resid(stages, z_tr)
    fit_rel = float(jnp.sqrt(jnp.mean(e_tr ** 2))) / u_norm_tr
    z_mean = jnp.mean(z_tr, axis=0)
    # val latents by inference through stage 0
    Zva0, rel_va0 = infer_latents(stages, coords, U_va,
                                  jnp.tile(z_mean, (N_VAL, 1)))
    lat_stats = {"z_tr_rms": float(jnp.sqrt(jnp.mean(z_tr ** 2))),
                 "z_tr_absmax": float(jnp.max(jnp.abs(z_tr)))}
    print(f"  after stage 0: TRAIN fit rel {fit_rel:.3e} (manifold floor, "
          f"K={K_LAT})  VAL inferred-latent rel-L2 {rel_va0.mean():.3e}  "
          f"latent rms {lat_stats['z_tr_rms']:.2f}", flush=True)
    report["stages"].append({"stage": 0, "eps_in": eps0, "n_freq": n_freq,
                             "train_fit_rel_rms": fit_rel,
                             "val_rel_l2_stage0_latents": float(rel_va0.mean()),
                             "val_rel_l2_reinferred": float(rel_va0.mean()),
                             **lat_stats})

    # ---------------- (B) frozen-latent staging ----------------
    Zva_fixed = Zva0
    for k in range(1, N_STAGES):
        eps = float(jnp.sqrt(jnp.mean(e_tr ** 2)))
        f_d = mp.dominant_radial_freq(e_tr[:32], N)
        n_freq = int(min(max(np.ceil(1.5 * f_d) + 4, n_freq), nyq))
        print(f"stage {k}: eps={eps:.3e}  f_d~{f_d:.0f}  n_freq={n_freq}",
              flush=True)
        key, sub = jax.random.split(key)
        params, _ = train_stage(sub, np_rng, coords, e_tr / eps, n_freq, z_tr,
                                False, f"B{k}")
        stages.append({"params": params, "n_freq": n_freq, "eps": eps})
        e_tr = train_resid(stages, z_tr)
        fit_rel = float(jnp.sqrt(jnp.mean(e_tr ** 2))) / u_norm_tr
        # val (i): stage-0 latents held fixed
        pred = jax.vmap(lambda z: combined_apply(stages, z, coords))(Zva_fixed)
        rel_fixed = np.asarray(jnp.linalg.norm(pred - U_va, axis=1)) / va_norms
        # val (ii): re-infer latents through the full staged decoder
        Zva_re, rel_re = infer_latents(stages, coords, U_va, Zva_fixed)
        print(f"  after stage {k}: TRAIN fit rel {fit_rel:.3e}   VAL fixed-lat "
              f"{rel_fixed.mean():.3e}   VAL re-inferred {rel_re.mean():.3e}",
              flush=True)
        report["stages"].append({"stage": k, "eps_in": eps, "f_d": f_d,
                                 "n_freq": n_freq, "train_fit_rel_rms": fit_rel,
                                 "val_rel_l2_stage0_latents": float(rel_fixed.mean()),
                                 "val_rel_l2_reinferred": float(rel_re.mean())})

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"ms_autodecoder_K{K_LAT}_stages.pkl"), "wb") as f:
        pickle.dump({"stages": [{"params": jax.tree_util.tree_map(np.asarray, s["params"]),
                                 "n_freq": s["n_freq"], "eps": s["eps"]}
                                for s in stages],
                     "z_tr": np.asarray(z_tr), "z_va0": np.asarray(Zva0)}, f)

    # ---------------- (C) ROM: LM Gauss-Newton on held-out sources ----------------
    cx, cy, w, a, _ = mp.sample_params()
    sl = slice(N_TRAIN, N_TRAIN + N_TEST)
    cx, cy, w, a = cx[sl], cy[sl], w[sl], a[sl]
    cx_tr, cy_tr, w_tr, a_tr = (v[:N_TRAIN] for v in mp.sample_params()[:4])
    U_test = np.asarray(U_va[:N_TEST])
    test_norms = va_norms[:N_TEST]
    dx = 1.0 / (N - 1)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    def stencil_pts(ix, iy):
        px, py = ix * dx, iy * dx
        return np.stack([np.stack([px, py], axis=1),
                         np.stack([px + dx, py], axis=1),
                         np.stack([px - dx, py], axis=1),
                         np.stack([px, py + dx], axis=1),
                         np.stack([px, py - dx], axis=1)])
    ii, jj = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1), indexing="ij")
    ix_full, iy_full = ii.reshape(-1), jj.reshape(-1)
    sub = np_rng.choice(len(ix_full), size=min(M_EQ, len(ix_full)), replace=False)
    colls = {"full": (ix_full, iy_full), f"m{M_EQ}": (ix_full[sub], iy_full[sub])}
    bmask = np.zeros((N, N), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bpts = jnp.asarray(np.stack([X[bmask], Y[bmask]], axis=1))
    bw = 1.0 / dx ** 2

    # nearest-training-source init (source params are known to the ROM)
    ztrue_all = np.asarray(z_true_all)
    zt_tr, zt_te = ztrue_all[:N_TRAIN], ztrue_all[N_TRAIN:N_TRAIN + N_TEST]
    nn_idx = np.argmin(((zt_te[:, None, :] - zt_tr[None, :, :]) ** 2).sum(-1), axis=1)
    z_tr_np = np.asarray(z_tr)

    def make_solver(n_stages, pts, f_vals):
        st = stages[:n_stages]
        dec = lambda z, xy: combined_apply(st, z, xy)

        def residual(z):
            u = dec(z, pts.reshape(-1, 2)).reshape(5, -1)
            lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (dx * dx)
            return jnp.concatenate([-lap - f_vals, bw * dec(z, bpts)])

        @jax.jit
        def rJ(z):
            return residual(z), jax.jacfwd(residual)(z)

        @jax.jit
        def rnorm(z):
            return jnp.linalg.norm(residual(z))

        def solve(z0, iters):
            """Levenberg-Marquardt: adaptive lambda, accept/reject."""
            z = z0
            lam = 1e-6
            r, J = rJ(z)
            rn = float(jnp.linalg.norm(r))
            n_acc = 0
            for _ in range(iters):
                H = J.T @ J
                g = J.T @ r
                D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(H.shape[0], dtype=F64)
                dz = jnp.linalg.solve(H + lam * D, -g)
                z_new = z + dz
                rn_new = float(rnorm(z_new))
                if rn_new < rn:
                    z, rn = z_new, rn_new
                    r, J = rJ(z)
                    lam = max(lam / 3.0, 1e-12)
                    n_acc += 1
                else:
                    lam = min(lam * 10.0, 1e12)
                if lam >= 1e12:
                    break
            return z, rn, n_acc

        return solve, dec

    # oracle-latent floors on the test subset for each n_stages (re-inference)
    oracle = {}
    for n_stages in range(1, len(stages) + 1):
        _, rel = infer_latents(stages[:n_stages], coords, U_va[:N_TEST],
                               Zva0[:N_TEST])
        oracle[n_stages] = float(rel.mean())

    for cname, (ix, iy) in colls.items():
        pts = jnp.asarray(stencil_pts(ix, iy))
        for n_stages in range(1, len(stages) + 1):
            for init_name in ("mean", "nearest", "staged"):
                if init_name == "staged" and n_stages == 1:
                    continue
                errs, rns, accs = [], [], []
                t0 = time.time()
                for i in range(N_TEST):
                    f_vals = jnp.asarray(a[i] * np.exp(
                        -((ix * dx - cx[i]) ** 2 + (iy * dx - cy[i]) ** 2)
                        / (2 * w[i] ** 2)))
                    solve, dec = make_solver(n_stages, pts, f_vals)
                    if init_name == "mean":
                        z0 = z_mean
                    elif init_name == "nearest":
                        z0 = jnp.asarray(z_tr_np[nn_idx[i]])
                    else:                       # staged: stage-0 solve first
                        s1, _ = make_solver(1, pts, f_vals)
                        z0, _, _ = s1(jnp.asarray(z_tr_np[nn_idx[i]]), GN_ITERS)
                    z, rn, n_acc = solve(z0, GN_ITERS)
                    pred = np.asarray(dec(z, coords))
                    errs.append(np.linalg.norm(pred - U_test[i]) / test_norms[i])
                    rns.append(rn)
                    accs.append(n_acc)
                row = {"colloc": cname, "n_stages": n_stages, "init": init_name,
                       "rom_rel_l2_mean": float(np.mean(errs)),
                       "rom_rel_l2_med": float(np.median(errs)),
                       "rom_rel_l2_max": float(np.max(errs)),
                       "oracle_latent_rel_l2": oracle[n_stages],
                       "gn_resid_med": float(np.median(rns)),
                       "gn_accepted_med": float(np.median(accs)),
                       "secs": time.time() - t0}
                report["rom"].append(row)
                print(f"RESULT K={K_LAT} colloc={cname:5s} stages={n_stages} "
                      f"init={init_name:7s}  ROM rel-L2 mean {row['rom_rel_l2_mean']:.3e} "
                      f"(med {row['rom_rel_l2_med']:.3e}, max {row['rom_rel_l2_max']:.3e})  "
                      f"oracle-latent {oracle[n_stages]:.3e}  "
                      f"acc {row['gn_accepted_med']:.0f}/{GN_ITERS}", flush=True)
            with open(os.path.join(OUTDIR, f"ms_autodecoder_K{K_LAT}_report.json"), "w") as f:
                json.dump(report, f, indent=2)

    with open(os.path.join(OUTDIR, f"ms_autodecoder_K{K_LAT}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
