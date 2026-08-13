"""Does error IMPROVE as the training resolution increases?

Train on FD data at N in {16..512}; evaluate every model against a
near-continuum reference (FD at N_ref=8192) on the reference grid.

Curves:
  data-floor : rel-L2 of the N-grid FD solution itself (interpolated to
               the reference grid) vs the reference — the best any model
               trained on N-grid data could possibly do vs the continuum.
  coord-net  : trained on N-grid data, evaluated NATIVELY on the
               reference grid (mesh-free).
  grid-tied  : trained on N-grid data, predictions linearly interpolated
               to the reference grid (only option for a grid-tied model).

Expected if the coordinate decoder restores the POD property:
  its curve tracks the falling O(dx^2) data-floor until it hits the
  network fit floor, i.e. error DECREASES with resolution.
Bump family as before: F = a*exp(-(x-c)^2/(2w^2)), c~U[.15,.85],
w~logU[.02,.1], a~U[.5,2]; z = normalized (c, ln w, a).
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

N_REF = 8192
NS = [16, 32, 64, 128, 256, 512]
RANK = 24
N_TRAIN, N_VAL = 512, 128
STEPS = 25_000
BATCH = 32
PEAK_LR = 2e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05
SEED = 0
N_FREQ = 16  # Nyquist-safe for N >= 33; N=16 is knowingly aliased


def fd_solve(F, n):
    dx = 1.0 / (n - 1)
    ab = np.zeros((2, n - 2))
    ab[0, 1:] = -1.0 / dx**2
    ab[1, :] = 2.0 / dx**2
    u = np.zeros(n)
    u[1:-1] = solveh_banded(ab, F[1:-1], lower=False)
    return u


def sample_params(seed=SEED):
    rng = np.random.default_rng(seed)
    m = N_TRAIN + N_VAL
    c = rng.uniform(0.15, 0.85, m)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), m))
    a = rng.uniform(0.5, 2.0, m)
    z = np.stack([
        (c - 0.5) / 0.35,
        (np.log(w) - np.log(0.045)) / 0.8,
        (a - 1.25) / 0.75,
    ], axis=1).astype(np.float32)
    return c, w, a, z


def solutions_on_grid(c, w, a, n):
    x = np.linspace(0.0, 1.0, n)
    U = np.stack([fd_solve(a[i] * np.exp(-((x - c[i]) ** 2) / (2 * w[i] ** 2)), n)
                  for i in range(len(c))])
    return U, x


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
    n_freq: int = N_FREQ
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


def train_model(apply_fn, params, U_train, z_train, U_val, z_val, seed=SEED):
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
    return best_params


def mean_rel(preds, ref):
    return float((np.linalg.norm(preds - ref, axis=1) / np.linalg.norm(ref, axis=1)).mean())


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    c, w, a, z = sample_params()
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]

    print(f"building reference solutions at N_ref={N_REF}", flush=True)
    U_ref_val, x_ref = solutions_on_grid(c[N_TRAIN:], w[N_TRAIN:], a[N_TRAIN:], N_REF)
    x_ref32 = jnp.asarray(x_ref, dtype=jnp.float32)

    results = {"meta": {"n_ref": N_REF, "steps": STEPS, "rank": RANK, "n_freq": N_FREQ,
                        "n_train": N_TRAIN, "n_val": N_VAL, "seed": SEED}}
    for N in NS:
        t0 = time.time()
        U_all, x = solutions_on_grid(c, w, a, N)
        U_tr, U_va = U_all[:N_TRAIN], U_all[N_TRAIN:]

        # Data floor: N-grid FD truth interpolated up to the reference grid.
        interp_data = np.stack([np.interp(x_ref, x, U_va[i]) for i in range(N_VAL)])
        e_data = mean_rel(interp_data, U_ref_val)

        rng = jax.random.PRNGKey(SEED)

        # Coordinate net: train on N-grid, evaluate natively on the reference grid.
        xj = jnp.asarray(x, dtype=jnp.float32)
        modelB = CoordDecoder()
        pB = modelB.init(rng, xj, jnp.zeros((3,), jnp.float32))["params"]
        applyB_train = lambda p, zz: modelB.apply({"params": p}, xj, zz)
        pB = train_model(applyB_train, pB, U_tr, z_tr, U_va, z_va)
        preds_ref = jax.vmap(lambda zz: modelB.apply({"params": pB}, x_ref32, zz))(
            jnp.asarray(z_va))
        e_coord = mean_rel(np.asarray(preds_ref, dtype=np.float64), U_ref_val)

        # Grid-tied: train on N-grid, linearly interpolate predictions to reference.
        modelA = GridTiedDecoder(N=N, rank=RANK)
        pA = modelA.init(rng, jnp.zeros((3,), jnp.float32))["params"]
        applyA = lambda p, zz: modelA.apply({"params": p}, zz)
        pA = train_model(applyA, pA, U_tr, z_tr, U_va, z_va)
        predsA = np.asarray(jax.vmap(lambda zz: applyA(pA, zz))(jnp.asarray(z_va)),
                            dtype=np.float64)
        interpA = np.stack([np.interp(x_ref, x, predsA[i]) for i in range(N_VAL)])
        e_grid = mean_rel(interpA, U_ref_val)

        dt = time.time() - t0
        results[N] = {"data_floor": e_data, "coord_net": e_coord,
                      "grid_tied": e_grid, "seconds": round(dt, 1)}
        print(f"N={N:4d}  data-floor={e_data:.3e}  coord-net={e_coord:.3e}  "
              f"grid-tied={e_grid:.3e}  [{dt:.0f}s]", flush=True)

    with open("results_convergence.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("wrote results_convergence.json", flush=True)


if __name__ == "__main__":
    main()
