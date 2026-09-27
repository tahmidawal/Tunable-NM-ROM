"""Train the Burgers-2D FiLM AUTO-DECODER u(x; z) (one latent per (trajectory,
time) snapshot) + POD basis for the linear control.

  data    : bf.build_trajectories(N) from seed (TRAIN = first N_TRAIN traj)
  target  : U / eps, eps = global RMS of the training snapshots; per-row
            inverse-energy weights (mean 1) => relative-error training
  latents : Z (n_tr, T+1, K), init = top-K POD coefficients (per-dim
            standardized) -- data-driven, smooth in time, no true-z used
  net     : ms_parametric FiLM net (f64), n_freq Nyquist-capped (N-1)//2,
            hard-BC factor b(x,y) (blat_common.bc_factor)
  loss    : mean_w ||b*film(x;z) - u/eps||^2 over P_SUB points
            + LAT_REG mean z^2 + T_SMOOTH mean ||z_{i,n+1} - z_{i,n}||^2
            (batch = B/2 random (i,n) rows PLUS their (i,n+1) neighbours so the
            time-smoothness term is always defined); lazy per-row Adam on the
            latents as in ms_parametric.fit_stage (moments only touched for
            rows in the batch)

Usage: N=64 K_LAT=8 [AD_STEPS=40000] [AD_BATCH=128] [P_SUB=0] [T_SMOOTH=1e-2]
       [LAT_REG=1e-4] [LAT_LR=5e-3] python blat_train_ad.py <outdir>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import blat_common as bc
from blat_common import mp, F64, log

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
AD_STEPS = int(os.environ.get("AD_STEPS", "40000"))
AD_BATCH = int(os.environ.get("AD_BATCH", "128"))
P_SUB = int(os.environ.get("P_SUB", "0"))
T_SMOOTH = float(os.environ.get("T_SMOOTH", "1e-2"))
LAT_REG = float(os.environ.get("LAT_REG", "1e-4"))
LAT_LR = float(os.environ.get("LAT_LR", "5e-3"))
PEAK_LR = float(os.environ.get("PEAK_LR", "2e-3"))
POD_KMAX = int(os.environ.get("POD_KMAX", "64"))
# TRAINING seed.  It controls the FiLM network initialisation and the minibatch /
# collocation-point sampling ONLY -- the per-snapshot latents are initialised
# DETERMINISTICALLY from the top-K POD coefficients (see below), so the multi-seed
# sweep here is "net init + batch order", not "latent init".  Default = the data seed
# bc.SEED, which reproduces the frozen runs' weights and latents exactly.  The DATA
# draw, the train/val split and the TEST_SEED test set never see this variable.
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", str(bc.SEED)))
K = bc.K_LAT
N = bc.N
T1 = bc.NUM_STEPS + 1


def main():
    log(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}  "
        f"N={N} K={K} steps={AD_STEPS} batch={AD_BATCH} p_sub={P_SUB} "
        f"t_smooth={T_SMOOTH} bc={bc.BC_MODE} hidden={bc.AD_HIDDEN}x{bc.AD_LAYERS}")
    d = bc.build_data(N)
    U = d["U"]
    U_tr = U[:bc.N_TRAIN]                          # TEST split (TEST_SEED) untouched here
    fp = bc.data_fingerprint(U)
    log(f"  data fingerprint {fp}")
    n2 = N * N
    n_tr = U_tr.shape[0]
    S = U_tr.reshape(-1, n2)                       # rows (i, n)
    n_rows = S.shape[0]

    # ---------------- POD (control basis + latent init) ----------------
    t0 = time.time()
    V, sv, dev = bc.pod_basis(S, kmax=max(POD_KMAX, K))
    log(f"  POD: {n_rows} snapshots, top-{V.shape[1]} ortho dev {dev:.1e} "
        f"[{time.time()-t0:.0f}s]")
    U_va = U[bc.N_TRAIN:].reshape(-1, n2)
    floors = {}
    for r in (4, 6, 8, 16, 32, 64):
        if r <= V.shape[1]:
            rec = (U_va @ V[:, :r]) @ V[:, :r].T
            floors[r] = float(np.mean(np.linalg.norm(rec - U_va, axis=1)
                                      / np.linalg.norm(U_va, axis=1)))
    log("  POD val floors: " + "  ".join(f"r{r}={e:.3e}" for r, e in floors.items()))

    # ---------------- targets, weights, latent init ----------------
    eps = float(np.sqrt(np.mean(S ** 2)))
    target = jnp.asarray(S / eps)
    w = 1.0 / np.mean(S ** 2, axis=1)
    weights = jnp.asarray(w / w.mean())
    C = S @ V[:, :K]                                # (rows, K)
    C = C / np.maximum(C.std(axis=0, keepdims=True), 1e-300)
    Z = jnp.asarray(C)
    coords = jnp.asarray(bc.grid_coords(N))
    bfac = bc.bc_factor(coords)
    n_freq = (N - 1) // 2
    key = jax.random.PRNGKey(TRAIN_SEED)
    params = mp.init_film_net(key, n_freq, K, 0)
    n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    log(f"  film params {n_par}, n_freq {n_freq}, eps {eps:.4e}")

    opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, max(1, AD_STEPS // 20), AD_STEPS, end_value=1e-9),
        weight_decay=1e-6)
    state = opt.init(params)
    lat_sched = optax.warmup_cosine_decay_schedule(
        0.0, LAT_LR, max(1, AD_STEPS // 20), AD_STEPS, end_value=1e-9)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    Bh = AD_BATCH // 2

    def dec_pts(ps, z, pidx):
        return bfac[pidx] * mp.film_apply(ps, z, coords[pidx], n_freq)

    def loss_fn(ps, z_b, t_b, w_b, pidx):
        pred = jax.vmap(lambda zi: dec_pts(ps, zi, pidx))(z_b)
        se = jnp.mean((pred - t_b[:, pidx]) ** 2, axis=1)
        data = jnp.mean(w_b * se)
        smooth = jnp.mean(jnp.sum((z_b[:Bh] - z_b[Bh:]) ** 2, axis=1))
        return data + LAT_REG * jnp.mean(z_b ** 2) + T_SMOOTH * smooth, data

    @jax.jit
    def step(ps, st, Z, m, v, cnt, gstep, bi, t_b, w_b, pidx):
        z_b = Z[bi]
        (val, data), (gp, gz) = jax.value_and_grad(loss_fn, argnums=(0, 1),
                                                    has_aux=True)(ps, z_b, t_b, w_b, pidx)
        up, st = opt.update(gp, st, ps)
        ps = optax.apply_updates(ps, up)
        m_b = b1 * m[bi] + (1 - b1) * gz
        v_b = b2 * v[bi] + (1 - b2) * gz ** 2
        c_b = cnt[bi] + 1.0
        mhat = m_b / (1 - b1 ** c_b[:, None])
        vhat = v_b / (1 - b2 ** c_b[:, None])
        z_new = z_b - lat_sched(gstep) * mhat / (jnp.sqrt(vhat) + eps_a)
        return (ps, st, Z.at[bi].set(z_new), m.at[bi].set(m_b),
                v.at[bi].set(v_b), cnt.at[bi].set(c_b), val, data)

    m_ = jnp.zeros_like(Z); v_ = jnp.zeros_like(Z)
    cnt = jnp.zeros((n_rows,), dtype=F64)
    rng = np.random.default_rng(TRAIN_SEED + 7)
    all_pts = jnp.arange(n2)
    t0 = time.time()
    hist = []
    assert AD_BATCH % 2 == 0 and AD_BATCH >= 2 and Bh <= n_tr
    for it in range(AD_STEPS):
        ti = rng.choice(n_tr, Bh, replace=False)      # distinct trajectories -> unique rows
        tn = rng.integers(0, T1 - 1, Bh)
        rows = np.concatenate([ti * T1 + tn, ti * T1 + tn + 1])
        pidx = (all_pts if P_SUB <= 0 or P_SUB >= n2 else
                jnp.asarray(rng.choice(n2, size=P_SUB, replace=False)))
        params, state, Z, m_, v_, cnt, val, data = step(
            params, state, Z, m_, v_, cnt, it, jnp.asarray(rows), target[rows],
            weights[rows], pidx)
        if it % 2000 == 0 or it == AD_STEPS - 1:
            hist.append((it, float(val), float(data)))
            log(f"  step {it:6d}  loss {float(val):.3e}  data {float(data):.3e}  "
                f"[{time.time()-t0:.0f}s]")
    train_secs = time.time() - t0

    # ---------------- train reconstruction at learned latents ----------------
    dec = bc.CoordDecoder(params, n_freq, eps, K)
    pred_fn = jax.jit(lambda z: dec(z, coords))
    rels = []
    CH = 32                                         # 512 rows x 16384 pts OOM'd a 40 GB A100
    for r in range(0, n_rows, CH):
        P = jax.vmap(pred_fn)(Z[r:r + CH])
        rels.append(np.asarray(jnp.linalg.norm(P - jnp.asarray(S[r:r + CH]), axis=1)
                               / jnp.linalg.norm(jnp.asarray(S[r:r + CH]), axis=1)))
    rels = np.concatenate(rels)
    Zn = np.asarray(Z).reshape(n_tr, T1, K)
    dz = np.linalg.norm(np.diff(Zn, axis=1), axis=2)
    log(f"  TRAIN recon at learned latents: mean rel {rels.mean():.3e} "
        f"(median {np.median(rels):.3e}, max {rels.max():.3e}); latent RMS "
        f"{np.sqrt(np.mean(np.asarray(Z)**2)):.3f}, mean |dz| per step "
        f"{dz.mean():.3e} [{train_secs:.0f}s]")

    os.makedirs(OUTDIR, exist_ok=True)
    # non-default training seeds get their own file name so two multi-seed cells
    # can never overwrite or read each other's checkpoint after a pull
    tag = f"N{N}_K{K}" + (f"_S{TRAIN_SEED}" if TRAIN_SEED != bc.SEED else "")
    ck = dict(params=jax.tree_util.tree_map(np.asarray, params), n_freq=n_freq,
              eps=eps, k_lat=K, Z_train=Zn, V=V, sv=sv, pod_floors=floors,
              config=dict(bc.CONFIG, train_seed=TRAIN_SEED, ad_steps=AD_STEPS, ad_batch=AD_BATCH,
                          p_sub=P_SUB, t_smooth=T_SMOOTH, lat_reg=LAT_REG,
                          lat_lr=LAT_LR, peak_lr=PEAK_LR, n_params=n_par),
              data_fingerprint=fp, train_rel_mean=float(rels.mean()),
              train_rel_median=float(np.median(rels)), train_secs=train_secs,
              hist=hist, backend=jax.default_backend())
    with open(os.path.join(OUTDIR, f"blat_ad_{tag}.pkl"), "wb") as f:
        pickle.dump(ck, f)
    rep = {k: v for k, v in ck.items() if k not in ("params", "Z_train", "V")}
    rep["sv"] = [float(s) for s in sv[:32]]
    with open(os.path.join(OUTDIR, f"blat_ad_{tag}_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    log(f"wrote {tag}")


if __name__ == "__main__":
    main()
