"""1D Poisson decoder-architecture diagnostic.

Question: does a grid-tied learned basis (the 1D analog of the CP decoder)
degrade with resolution while a coordinate-network decoder does not?

Problem:  -u'' = A sin(k pi x),  u(0)=u(1)=0,  k ~ U[1,3], A = 10.
Exact Dirichlet solution:  u(x) = A [sin(k pi x) - x sin(k pi)] / (k pi)^2.
Ground truth is exact at every N, so resolution only affects the models.

Both decoders are conditioned on the true parameter z = k - 2 (no encoder),
isolating the decoder representation + trainability.

Arm A (grid-tied): MLP(z) -> h in R^R ; u = h @ W + b, W in R^{R x N} learned.
Arm B (coord-net): u_i = MLP(fourier(x_i), z), parameters independent of N.
Reference: rank-R POD projection error of val snapshots (best linear basis).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

A_AMP = 10.0
RANK = 24
N_TRAIN, N_VAL = 256, 64
STEPS = 15_000
BATCH = 32
PEAK_LR = 2e-3
WEIGHT_DECAY = 5e-4
WARMUP_FRAC = 0.05
SEED = 0
NS = [64, 256, 1024, 4096]


def exact_u(k: float, x: np.ndarray) -> np.ndarray:
    return A_AMP * (np.sin(k * np.pi * x) - x * np.sin(k * np.pi)) / (k * np.pi) ** 2


def make_data(N: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    ks = rng.uniform(1.0, 3.0, N_TRAIN + N_VAL)
    x = np.linspace(0.0, 1.0, N)
    U = np.stack([exact_u(k, x) for k in ks])  # float64
    z = (ks - 2.0).astype(np.float32)[:, None]
    return (U[:N_TRAIN], z[:N_TRAIN]), (U[N_TRAIN:], z[N_TRAIN:]), x


def pod_floor(U_train: np.ndarray, U_val: np.ndarray, rank: int) -> float:
    """Mean rel-L2 of projecting val snapshots onto top-`rank` right singular vecs."""
    _, _, Vt = np.linalg.svd(U_train, full_matrices=False)
    V = Vt[:rank].T  # (N, rank)
    proj = (U_val @ V) @ V.T
    rel = np.linalg.norm(U_val - proj, axis=1) / np.linalg.norm(U_val, axis=1)
    return float(rel.mean())


class GridTiedDecoder(nn.Module):
    N: int
    rank: int
    hidden: int = 64

    @nn.compact
    def __call__(self, z):
        h = nn.swish(nn.Dense(self.hidden)(z))
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.Dense(self.rank)(h)
        W = self.param("W", nn.initializers.normal(0.01), (self.rank, self.N))
        b = self.param("b", nn.initializers.zeros, ())
        return h @ W + b


class CoordDecoder(nn.Module):
    n_freq: int = 8
    hidden: int = 64

    @nn.compact
    def __call__(self, x, z):
        # x: (P,) coordinates, z: (dz,) latent/parameter
        j = jnp.arange(1, self.n_freq + 1, dtype=jnp.float32)
        ff = jnp.concatenate(
            [x[:, None], jnp.sin(jnp.pi * j * x[:, None]), jnp.cos(jnp.pi * j * x[:, None])],
            axis=1,
        )
        zz = jnp.broadcast_to(z, (x.shape[0], z.shape[-1]))
        h = jnp.concatenate([ff, zz], axis=1)
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.swish(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h)[:, 0]


def rel_l2_sq(pred, true):
    return jnp.sum((pred - true) ** 2) / (jnp.sum(true**2) + 1e-12)


def train_model(apply_fn, params, U_train, z_train, U_val, z_val, seed: int):
    """apply_fn(params, u_z) -> pred field, vmapped over the batch outside."""
    warmup = max(1, int(STEPS * WARMUP_FRAC))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-6
    )
    opt = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)
    opt_state = opt.init(params)

    def batched_loss(params, U_b, z_b):
        preds = jax.vmap(lambda z: apply_fn(params, z))(z_b)
        return jnp.mean(jax.vmap(rel_l2_sq)(preds, U_b))

    @jax.jit
    def step(params, opt_state, U_b, z_b):
        loss, grads = jax.value_and_grad(batched_loss)(params, U_b, z_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    eval_loss = jax.jit(batched_loss)

    np_rng = np.random.default_rng(seed)
    U_train32 = jnp.asarray(U_train, dtype=jnp.float32)
    z_train32 = jnp.asarray(z_train)
    U_val32 = jnp.asarray(U_val, dtype=jnp.float32)
    z_val32 = jnp.asarray(z_val)

    best_val = float("inf")
    best_params = params
    for it in range(STEPS):
        idx = np_rng.choice(N_TRAIN, size=BATCH, replace=False)
        params, opt_state, _ = step(params, opt_state, U_train32[idx], z_train32[idx])
        if it % 500 == 0 or it == STEPS - 1:
            v = float(eval_loss(params, U_val32, z_val32))
            if v < best_val:
                best_val = v
                best_params = params
    return best_params, best_val


def eval_rel_l2(apply_fn, params, U_val, z_val) -> float:
    preds = jax.vmap(lambda z: apply_fn(params, z))(jnp.asarray(z_val))
    preds = np.asarray(preds, dtype=np.float64)
    rel = np.linalg.norm(preds - U_val, axis=1) / np.linalg.norm(U_val, axis=1)
    return float(rel.mean())


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    results = {"meta": {"rank": RANK, "steps": STEPS, "batch": BATCH,
                        "n_train": N_TRAIN, "n_val": N_VAL, "seed": SEED,
                        "peak_lr": PEAK_LR, "weight_decay": WEIGHT_DECAY}}
    coord_ckpts = {}

    for N in NS:
        (U_tr, z_tr), (U_va, z_va), x = make_data(N)
        t0 = time.time()

        pod = pod_floor(U_tr, U_va, RANK)

        # Arm A: grid-tied
        rng = jax.random.PRNGKey(SEED)
        modelA = GridTiedDecoder(N=N, rank=RANK)
        pA = modelA.init(rng, jnp.zeros((1,), jnp.float32))["params"]
        applyA = lambda p, z: modelA.apply({"params": p}, z)
        pA, _ = train_model(applyA, pA, U_tr, z_tr, U_va, z_va, seed=SEED)
        errA = eval_rel_l2(applyA, pA, U_va, z_va)
        nparamsA = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(pA))

        # Arm B: coordinate net
        xj = jnp.asarray(x, dtype=jnp.float32)
        modelB = CoordDecoder()
        pB = modelB.init(rng, xj, jnp.zeros((1,), jnp.float32))["params"]
        applyB = lambda p, z, xg=xj: modelB.apply({"params": p}, xg, z)
        pB, _ = train_model(applyB, pB, U_tr, z_tr, U_va, z_va, seed=SEED)
        errB = eval_rel_l2(applyB, pB, U_va, z_va)
        nparamsB = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(pB))
        coord_ckpts[N] = pB

        dt = time.time() - t0
        results[N] = {"pod_rank%d" % RANK: pod, "grid_tied": errA,
                      "coord_net": errB, "params_grid": nparamsA,
                      "params_coord": nparamsB, "seconds": round(dt, 1)}
        print(f"N={N:5d}  POD-{RANK}={pod:.3e}  grid-tied={errA:.3e} "
              f"({nparamsA} params)  coord-net={errB:.3e} ({nparamsB} params)  "
              f"[{dt:.0f}s]", flush=True)

    # Mesh-transfer: coord net trained at the coarsest N, evaluated on the finest grid.
    N_src, N_dst = NS[0], NS[-1]
    (_, _), (U_va_dst, z_va_dst), x_dst = make_data(N_dst)
    modelB = CoordDecoder()
    xj_dst = jnp.asarray(x_dst, dtype=jnp.float32)
    applyB_dst = lambda p, z: modelB.apply({"params": p}, xj_dst, z)
    err_transfer = eval_rel_l2(applyB_dst, coord_ckpts[N_src], U_va_dst, z_va_dst)
    results["mesh_transfer"] = {"train_N": N_src, "eval_N": N_dst, "rel_l2": err_transfer}
    print(f"mesh-transfer: coord net trained N={N_src}, evaluated N={N_dst}: "
          f"rel_l2={err_transfer:.3e}", flush=True)

    out = sys.argv[1] if len(sys.argv) > 1 else "poisson1d_decoder_diag_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
