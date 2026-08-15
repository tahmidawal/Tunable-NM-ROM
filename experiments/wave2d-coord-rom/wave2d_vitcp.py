"""Published CP-decoder arm for the wave-2D testbed (Task: ViT-CP port).

Trains the repo's REAL published CP decoders -- imported verbatim from the
clean packages (heat/src CPDecoder, poisson/src LinearCPDecoder, flax) --
as (z, t)-conditioned decoders on the same seed-0 wave data, so the testbed
comparison includes the literal published baseline next to the simplified
grid-tied control.

Faithful pieces and deviations: see burgers2d_vitcp.py in the sibling
worktree -- identical port (published rank=256, hidden=256, per-N optimizer
settings, AdamW warmup-cosine, published squared-relative-L2 full-field
loss with eps 1e-6; latent input = concat(z (5,), 2*tau-1), latent_dim=6;
both 'cp' and 'lincp' variants; STEPS=120000 parity budget).

Wave-specific note: the published per-sample relative loss up-weights
near-silent snapshots (wave snapshot norms pass near zero during energy
exchange -- the reason the film arm trains against a per-trajectory msq
normalizer instead). Kept as published: that behavior is part of what this
arm measures. Metrics reported in BOTH testbed conventions: traj-RMS
primary (diff norm / per-trajectory RMS norm) and per-snapshot secondary.

Usage:  N=64 [STEPS=120000] [SEED=0] python wave2d_vitcp.py <outdir>
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "heat", "src"))
sys.path.insert(0, os.path.join(REPO, "poisson", "src"))

import wave2d_film as wf
from tunable_rom_heat.models.decoder import CPDecoder
from tunable_rom_poisson.models.decoder import LinearCPDecoder

N = wf.N
STEPS = int(os.environ.get("STEPS", "120000"))
SEED = wf.SEED
RANK = int(os.environ.get("RANK", "256"))
HIDDEN = 256
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
os.makedirs(OUTDIR, exist_ok=True)

if N >= 256:
    BATCH, PEAK_LR, WD = 8, 1e-3, 2e-3
elif N >= 128:
    BATCH, PEAK_LR, WD = 16, 1e-3, 2e-3
else:
    BATCH, PEAK_LR, WD = 32, 2e-3, 5e-4
WARMUP_FRAC = 0.1

F32 = jnp.float32


def latent_inputs(z, k):
    tau = 2.0 * (k / wf.NUM_STEPS) - 1.0
    return np.concatenate([z, [tau]]).astype(np.float32)


def train_variant(name, module, U_tr32, Z6_tr, U_va, z_va, rms_va, np_rng):
    model = module(N=N, spatial_dim=2, latent_dim=6, rank=RANK,
                   hidden_dim=HIDDEN)
    params = model.init(jax.random.PRNGKey(SEED), jnp.zeros((6,), F32))["params"]
    n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))

    warmup = max(1, int(STEPS * WARMUP_FRAC))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-6)
    opt = optax.adamw(learning_rate=schedule, weight_decay=WD)
    opt_state = opt.init(params)

    def loss_fn(p, zb, ub):
        preds = jax.vmap(
            lambda zi: model.apply({"params": p}, zi).reshape(-1))(zb)
        num = jnp.sum((preds - ub) ** 2, axis=1)
        den = jnp.sum(ub ** 2, axis=1) + 1e-6      # published eps
        return jnp.mean(num / den)

    @jax.jit
    def step(p, s, zb, ub):
        loss, grads = jax.value_and_grad(loss_fn)(p, zb, ub)
        upd, s = opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, loss

    n_samp = U_tr32.shape[0]
    t0 = time.time()
    last = 0.0
    for it in range(STEPS):
        idx = np_rng.integers(0, n_samp, BATCH)
        params, opt_state, last = step(params, opt_state,
                                       Z6_tr[idx], U_tr32[idx])
        if it % 20000 == 0:
            print(f"  [{name}] step {it:7d}  loss {float(last):.3e}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"  [{name}] trained {STEPS} steps in {time.time()-t0:.0f}s "
          f"(final batch loss {float(last):.3e})", flush=True)

    apply_j = jax.jit(lambda p, zi: model.apply({"params": p}, zi).reshape(-1))
    rels_t, rels_s = [], []
    for i in range(wf.N_VAL):
        rt, rs = [], []
        for k in range(wf.NUM_STEPS + 1):
            z6 = jnp.asarray(latent_inputs(z_va[i], k))
            pred = np.asarray(apply_j(params, z6), dtype=np.float64)
            d = np.linalg.norm(pred - U_va[i, k])
            rt.append(d / rms_va[i])
            rs.append(d / max(np.linalg.norm(U_va[i, k]), 1e-300))
        rels_t.append(np.mean(rt))
        rels_s.append(np.mean(rs))
    err_t, err_s = float(np.mean(rels_t)), float(np.mean(rels_s))
    print(f"  [{name}] params={n_par}  traj-RMS = {err_t:.3e}  "
          f"snap = {err_s:.3e}", flush=True)
    return err_t, err_s, n_par, params


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    print(f"ViT-CP arm N={N} rank={RANK} steps={STEPS} batch={BATCH} "
          f"lr={PEAK_LR} wd={WD}", flush=True)

    U, z, cx, cy, w, a, c = wf.build_trajectories(N)
    U_tr, U_va = U[:wf.N_TRAIN], U[wf.N_TRAIN:]
    z_tr, z_va = z[:wf.N_TRAIN], z[wf.N_TRAIN:]
    rms_va, _ = wf.traj_norms(U_va)

    n_tr, n_t = U_tr.shape[:2]
    U_tr32 = jnp.asarray(U_tr.reshape(n_tr * n_t, -1), dtype=F32)
    Z6_tr = jnp.asarray(np.stack([
        latent_inputs(z_tr[i], k)
        for i in range(n_tr) for k in range(n_t)]))

    np_rng = np.random.default_rng(SEED + 7)
    results = {"N": N, "rank": RANK, "steps": STEPS, "batch": BATCH,
               "peak_lr": PEAK_LR, "weight_decay": WD, "seed": SEED}
    for name, module in [("cp", CPDecoder), ("lincp", LinearCPDecoder)]:
        err_t, err_s, n_par, params = train_variant(
            name, module, U_tr32, Z6_tr, U_va, z_va, rms_va, np_rng)
        results[name] = err_t
        results[f"{name}_snap"] = err_s
        results[f"{name}_params"] = n_par
        with open(os.path.join(
                OUTDIR, f"wave2d_vitcp_{name}_N{N}.pkl"), "wb") as f:
            pickle.dump(jax.device_get(params), f)

    with open(os.path.join(
            OUTDIR, f"wave2d_vitcp_results_N{N}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"RESULT N={N}  vitcp-cp={results['cp']:.3e}  "
          f"vitcp-lincp={results['lincp']:.3e}  (traj-RMS; snap: "
          f"cp={results['cp_snap']:.3e} lincp={results['lincp_snap']:.3e})",
          flush=True)


if __name__ == "__main__":
    main()
