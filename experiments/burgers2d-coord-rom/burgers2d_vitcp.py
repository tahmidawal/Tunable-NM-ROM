"""Published CP-decoder arm for the Burgers-2D testbed (Task: ViT-CP port).

Trains the repo's REAL published CP decoders -- imported verbatim from the
clean packages (heat/src CPDecoder and poisson/src LinearCPDecoder, flax) --
as (z, t)-conditioned decoders on the same seed-0 Burgers data, so the
testbed comparison includes the literal published baseline next to the
simplified grid-tied control.

Faithful pieces: decoder modules imported from the packages unmodified
(2-layer swish MLP hidden=256 -> rank-R channel weights -> CP contraction
over W_x, W_y factor matrices (normal 0.01 init) + scalar bias; LinearCP
adds the published linear skip); published rank=256 and per-N optimizer
settings (N<=64: batch 32, peak_lr 2e-3, wd 5e-4; N=128: 16/1e-3/2e-3;
N=256: 8/1e-3/2e-3), AdamW warmup-cosine (warmup_frac 0.1), published
squared-relative-L2 full-field loss (eps 1e-6).

Documented deviations, forced by the (z, t) -> field testbed protocol:
  1. The decoder's latent input is the TRUE normalized parameter vector
     concat(z (5,), 2*tau-1) (latent_dim=6), not a ViT-encoder latent
     (latent_dim=64): at eval the arms may not see the field, so the
     encoder is unused. This matches how the grid-tied/FiLM arms are
     conditioned and gives the published decoder its best case.
  2. Training budget STEPS=120000 for parity with the film arms
     (published 2D configs use 80k-150k).
  3. Both variants are trained: 'cp' (heat package, plain) and 'lincp'
     (poisson package, linear skip) -- the skip is the published cure for
     cold-start GN, irrelevant here but included for completeness.

Metrics identical to the film arms: mean over the 64 val trajectories and
all 51 slices of per-snapshot rel-L2, network f32, norms in f64.

Usage:  N=64 [STEPS=120000] [SEED=0] python burgers2d_vitcp.py <outdir>
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

import burgers2d_film as bf
from tunable_rom_heat.models.decoder import CPDecoder
from tunable_rom_poisson.models.decoder import LinearCPDecoder

N = bf.N
STEPS = int(os.environ.get("STEPS", "120000"))
SEED = bf.SEED
RANK = int(os.environ.get("RANK", "256"))
HIDDEN = 256
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
os.makedirs(OUTDIR, exist_ok=True)

# published per-N optimizer settings (2D configs; N<=64 uses the n64 values)
if N >= 256:
    BATCH, PEAK_LR, WD = 8, 1e-3, 2e-3
elif N >= 128:
    BATCH, PEAK_LR, WD = 16, 1e-3, 2e-3
else:
    BATCH, PEAK_LR, WD = 32, 2e-3, 5e-4
WARMUP_FRAC = 0.1

F32 = jnp.float32


def latent_inputs(z, k):
    """concat(normalized params, 2*tau-1) -> (6,) latent for the decoder."""
    tau = 2.0 * (k / bf.NUM_STEPS) - 1.0
    return np.concatenate([z, [tau]]).astype(np.float32)


def train_variant(name, module, U_tr32, Z6_tr, U_va, z_va, np_rng):
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

    # eval: identical conventions to the film arms
    apply_j = jax.jit(lambda p, zi: model.apply({"params": p}, zi).reshape(-1))
    rels = []
    for i in range(bf.N_VAL):
        rel_i = []
        for k in range(bf.NUM_STEPS + 1):
            z6 = jnp.asarray(latent_inputs(z_va[i], k))
            pred = np.asarray(apply_j(params, z6), dtype=np.float64)
            rel_i.append(np.linalg.norm(pred - U_va[i, k])
                         / max(np.linalg.norm(U_va[i, k]), 1e-300))
        rels.append(np.mean(rel_i))
    err = float(np.mean(rels))
    print(f"  [{name}] params={n_par}  val rel-L2 = {err:.3e}", flush=True)
    return err, n_par, params


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    print(f"ViT-CP arm N={N} rank={RANK} steps={STEPS} batch={BATCH} "
          f"lr={PEAK_LR} wd={WD}", flush=True)

    U, z, cx, cy, w, a, nu = bf.build_trajectories(N)
    U_tr, U_va = U[:bf.N_TRAIN], U[bf.N_TRAIN:]
    z_tr, z_va = z[:bf.N_TRAIN], z[bf.N_TRAIN:]

    # flatten (traj, time) -> samples, f32, with (z, tau) latents
    n_tr, n_t = U_tr.shape[:2]
    U_tr32 = jnp.asarray(U_tr.reshape(n_tr * n_t, -1), dtype=F32)
    Z6_tr = jnp.asarray(np.stack([
        latent_inputs(z_tr[i], k)
        for i in range(n_tr) for k in range(n_t)]))

    np_rng = np.random.default_rng(SEED + 7)
    results = {"N": N, "rank": RANK, "steps": STEPS, "batch": BATCH,
               "peak_lr": PEAK_LR, "weight_decay": WD, "seed": SEED}
    for name, module in [("cp", CPDecoder), ("lincp", LinearCPDecoder)]:
        err, n_par, params = train_variant(
            name, module, U_tr32, Z6_tr, U_va, z_va, np_rng)
        results[name] = err
        results[f"{name}_params"] = n_par
        with open(os.path.join(
                OUTDIR, f"burgers2d_vitcp_{name}_N{N}.pkl"), "wb") as f:
            pickle.dump(jax.device_get(params), f)

    with open(os.path.join(
            OUTDIR, f"burgers2d_vitcp_results_N{N}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"RESULT N={N}  vitcp-cp={results['cp']:.3e}  "
          f"vitcp-lincp={results['lincp']:.3e}", flush=True)


if __name__ == "__main__":
    main()
