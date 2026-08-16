"""Cells 4/5: (re)train a stage-0 FiLM AUTO-DECODER (same recipe as
ms_autodecoder phase A: joint lazy-Adam latents, inverse-energy weights,
warmup-cosine AdamW, f64) — optionally with a HARD-BC multiplier
b(x,y) = 16 x(1-x) y(1-y) on the output — and write a multistage-format pkl
consumable by pro_objective.py / pro_colloc.py (HARD_BC=1 must be passed to
those when the pkl was trained with it; the pkl config records it).

Reports the training fit and the held-out finite-budget inferred-latent error
(LM on the data misfit, mean/nearest inits, GN_ITERS budget) so the new
decoder's floors 1 and 2 are on record next to the ROM floor.

Usage: K_LAT=8 [HARD_BC=0|1] [N=64] [N_TRAIN=512] [N_VAL=64] [STEPS=20000]
       [BATCH=32] [P_SUB=1024] [GN_ITERS=60] python pro_train.py <outdir>
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

import pro_common as pc
from pro_common import mp

K_LAT = int(os.environ.get("K_LAT", "8"))
HARD_BC = int(os.environ.get("HARD_BC", "0"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
OUTDIR = sys.argv[1]
N, N_TRAIN, N_VAL, STEPS = mp.N, mp.N_TRAIN, mp.N_VAL, mp.STEPS


def main():
    print(f"jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64} K_LAT={K_LAT} HARD_BC={HARD_BC}",
          flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = dict(mp.CONFIG, K_LAT=K_LAT, n_stages=1, lat_lr=5e-3, lat_reg=1e-4, hard_bc=HARD_BC,
               gn_iters=GN_ITERS)
    print("CONFIG " + json.dumps(cfg), flush=True)
    U, z_true, coords, fom_res = mp.build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    zt = np.asarray(z_true)
    np_rng = np.random.default_rng(mp.SEED)
    key = jax.random.PRNGKey(mp.SEED + 100 + K_LAT)
    stages, Z_tr, eps0, n_freq, adam_loss = pc.train_autodecoder_stage0(
        key, np_rng, coords, U_tr, K_LAT, bool(HARD_BC), STEPS, mp.BATCH, mp.P_SUB)
    dec = pc.make_decoder(stages, hard_bc=bool(HARD_BC))
    pred_tr = jnp.concatenate([jax.vmap(lambda z: dec(z, coords))(jnp.asarray(Z_tr[s:s + 64]))
                               for s in range(0, N_TRAIN, 64)])
    g_tr, m_tr, _ = mp.rel_metrics(pred_tr, U_tr)
    # held-out inferred latents (finite budget), mean + nearest inits
    nn_idx = np.argmin(((zt[N_TRAIN:, None, :] - zt[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
    inits = {"mean": np.tile(Z_tr.mean(0), (N_VAL, 1)), "nearest": Z_tr[nn_idx]}
    rJ = jax.jit(lambda z, u: (dec(z, coords) - u, jax.jacfwd(lambda zz: dec(zz, coords) - u)(z)))
    rn = jax.jit(lambda z, u: jnp.linalg.norm(dec(z, coords) - u))
    inf = {}
    for name, Z0 in inits.items():
        rels = []
        for i in range(N_VAL):
            u = U_va[i]
            _, r, _ = pc.lm_solve(lambda zz: rJ(zz, u), lambda zz: rn(zz, u), jnp.asarray(Z0[i]), GN_ITERS)
            rels.append(r / float(jnp.linalg.norm(u)))
        inf[name] = np.asarray(rels)
    best = np.minimum(inf["mean"], inf["nearest"])
    bnd = float(np.median([float(jnp.linalg.norm(dec(jnp.asarray(z), pc.Grid(N).bpts))) for z in Z_tr[:32]]))
    report = dict(config=cfg, fom_max_rel_residual=fom_res, eps0=eps0, n_freq=n_freq,
                  adam_final_batch_loss=adam_loss, train_global_rel=g_tr, train_mean_rel_l2=m_tr,
                  val_lm_inferred_mean_rel_l2={k: float(v.mean()) for k, v in inf.items()} | {"best": float(best.mean())},
                  boundary_block_train_med=bnd, latent_rms=float(np.sqrt(np.mean(Z_tr ** 2))),
                  complete=True)
    tag = f"K{K_LAT}_N{N}" + ("_hbc" if HARD_BC else "")
    with open(os.path.join(OUTDIR, f"autodec_{tag}_stages.pkl"), "wb") as f:
        pickle.dump({"config": cfg, "stages": pc.stages_to_np(stages), "z_tr": Z_tr}, f)
    json.dump(report, open(os.path.join(OUTDIR, f"autodec_{tag}_report.json"), "w"), indent=1)
    print(f"RESULT {tag}: TRAIN {g_tr:.3e}/{m_tr:.3e}  VAL inferred mean {inf['mean'].mean():.3e} "
          f"nearest {inf['nearest'].mean():.3e} best {best.mean():.3e}  boundary-block(train) {bnd:.1e}",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
