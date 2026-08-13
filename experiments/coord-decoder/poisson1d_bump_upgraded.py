"""Upgraded-budget head-to-head at N=1024 on the bump family.

Same data/protocol as poisson1d_bump_diag.py, but both arms get:
  steps 40k, hidden 128, weight_decay 1e-5; coord net additionally
  n_freq 32 and 4 hidden layers. Reference: POD-3 (= coord net's fair
  linear comparator, 3 reduced variables) and POD-24 (= grid-tied's).
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.linalg import solveh_banded
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

N = 1024
RANK = 24
N_TRAIN, N_VAL = 512, 128
STEPS = 40_000
BATCH = 32
PEAK_LR = 2e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05
SEED = 0


def fd_solve(F, n):
    dx = 1.0 / (n - 1)
    ab = np.zeros((2, n - 2))
    ab[0, 1:] = -1.0 / dx**2
    ab[1, :] = 2.0 / dx**2
    u = np.zeros(n)
    u[1:-1] = solveh_banded(ab, F[1:-1], lower=False)
    return u


def make_data(n, seed=SEED):
    rng = np.random.default_rng(seed)
    m = N_TRAIN + N_VAL
    c = rng.uniform(0.15, 0.85, m)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), m))
    a = rng.uniform(0.5, 2.0, m)
    x = np.linspace(0.0, 1.0, n)
    U = np.stack([fd_solve(a[i] * np.exp(-((x - c[i]) ** 2) / (2 * w[i] ** 2)), n)
                  for i in range(m)])
    z = np.stack([
        (c - 0.5) / 0.35,
        (np.log(w) - np.log(0.045)) / 0.8,
        (a - 1.25) / 0.75,
    ], axis=1).astype(np.float32)
    return (U[:N_TRAIN], z[:N_TRAIN]), (U[N_TRAIN:], z[N_TRAIN:]), x


class GridTiedDecoder(nn.Module):
    N: int
    rank: int
    hidden: int = 128

    @nn.compact
    def __call__(self, z):
        h = nn.swish(nn.Dense(self.hidden)(z))
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.Dense(self.rank)(h)
        W = self.param("W", nn.initializers.normal(0.01), (self.rank, self.N))
        b = self.param("b", nn.initializers.zeros, ())
        return h @ W + b


class CoordDecoder(nn.Module):
    n_freq: int = 32
    hidden: int = 128

    @nn.compact
    def __call__(self, x, z):
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
        h = nn.swish(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h)[:, 0]


def rel_l2_sq(pred, true):
    return jnp.sum((pred - true) ** 2) / (jnp.sum(true**2) + 1e-12)


def train_model(apply_fn, params, U_train, z_train, U_val, z_val, seed):
    warmup = max(1, int(STEPS * WARMUP_FRAC))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-7
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

    best_val, best_params = float("inf"), params
    for it in range(STEPS):
        idx = np_rng.choice(N_TRAIN, size=BATCH, replace=False)
        params, opt_state, _ = step(params, opt_state, U_train32[idx], z_train32[idx])
        if it % 500 == 0 or it == STEPS - 1:
            v = float(eval_loss(params, U_val32, z_val32))
            if v < best_val:
                best_val, best_params = v, params
    return best_params, best_val


def eval_rel_l2(apply_fn, params, U_val, z_val):
    preds = jax.vmap(lambda z: apply_fn(params, z))(jnp.asarray(z_val))
    preds = np.asarray(preds, dtype=np.float64)
    rel = np.linalg.norm(preds - U_val, axis=1) / np.linalg.norm(U_val, axis=1)
    return float(rel.mean())


def pod_err(U_train, U_val, rank):
    _, _, Vt = np.linalg.svd(U_train, full_matrices=False)
    V = Vt[:rank].T
    proj = (U_val @ V) @ V.T
    return float((np.linalg.norm(U_val - proj, axis=1) / np.linalg.norm(U_val, axis=1)).mean())


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    (U_tr, z_tr), (U_va, z_va), x = make_data(N)

    pod3 = pod_err(U_tr, U_va, 3)
    pod24 = pod_err(U_tr, U_va, RANK)
    print(f"POD-3={pod3:.3e}  POD-24={pod24:.3e}", flush=True)

    rng = jax.random.PRNGKey(SEED)

    t0 = time.time()
    modelA = GridTiedDecoder(N=N, rank=RANK)
    pA = modelA.init(rng, jnp.zeros((3,), jnp.float32))["params"]
    applyA = lambda p, z: modelA.apply({"params": p}, z)
    pA, _ = train_model(applyA, pA, U_tr, z_tr, U_va, z_va, seed=SEED)
    errA = eval_rel_l2(applyA, pA, U_va, z_va)
    print(f"grid-tied (rank 24, 40k steps): {errA:.3e}  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    xj = jnp.asarray(x, dtype=jnp.float32)
    modelB = CoordDecoder()
    pB = modelB.init(rng, xj, jnp.zeros((3,), jnp.float32))["params"]
    applyB = lambda p, z: modelB.apply({"params": p}, xj, z)
    pB, _ = train_model(applyB, pB, U_tr, z_tr, U_va, z_va, seed=SEED)
    errB = eval_rel_l2(applyB, pB, U_va, z_va)
    print(f"coord-net (3 latent, 40k steps): {errB:.3e}  [{time.time()-t0:.0f}s]", flush=True)

    with open("results_bump_upgraded.json", "w") as f:
        json.dump({"pod3": pod3, "pod24": pod24, "grid_tied": errA,
                   "coord_net": errB, "steps": STEPS}, f, indent=2)
    print("wrote results_bump_upgraded.json", flush=True)


if __name__ == "__main__":
    main()
