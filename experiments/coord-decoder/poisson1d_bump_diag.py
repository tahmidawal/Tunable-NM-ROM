"""1D Poisson decoder diagnostic, part 2: translated sharp features.

Problem:  -u'' = F(x; mu),  u(0)=u(1)=0
Source:   F = a * exp(-(x-c)^2 / (2 w^2)),  c ~ U[0.15,0.85],
          w ~ logU[0.02, 0.1],  a ~ U[0.5, 2].
Truth:    FD tridiagonal solve per N in float64 (data matches operator).

The solution is a smoothed 'tent' whose kink location moves with c —
a slowly-decaying linear-width family (1D analog of the repo's
translated Gaussian blobs). This is where a fixed linear basis must
struggle and a coordinate network (nonlinear in x) can win.

Same three arms as part 1: POD-R floor, grid-tied decoder, coord-net.
Conditioned on true params z = normalized (c, ln w, a); no encoder.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.linalg import solveh_banded
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

RANK = 24
N_TRAIN, N_VAL = 512, 128
STEPS = 15_000
BATCH = 32
PEAK_LR = 2e-3
WEIGHT_DECAY = 5e-4
WARMUP_FRAC = 0.05
SEED = 0
NS = [64, 256, 1024, 4096]


def fd_solve(F: np.ndarray, N: int) -> np.ndarray:
    """Solve -u'' = F with homogeneous Dirichlet BCs, 2nd-order FD, f64."""
    dx = 1.0 / (N - 1)
    M = N - 2
    ab = np.zeros((2, M))
    ab[0, 1:] = -1.0 / dx**2   # upper diagonal
    ab[1, :] = 2.0 / dx**2     # main diagonal
    u_int = solveh_banded(ab, F[1:-1], lower=False)
    u = np.zeros(N)
    u[1:-1] = u_int
    return u


def make_data(N: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL
    c = rng.uniform(0.15, 0.85, n)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), n))
    a = rng.uniform(0.5, 2.0, n)
    x = np.linspace(0.0, 1.0, N)
    U = np.empty((n, N))
    for i in range(n):
        F = a[i] * np.exp(-((x - c[i]) ** 2) / (2 * w[i] ** 2))
        U[i] = fd_solve(F, N)
    # normalize params to O(1) conditioning inputs
    z = np.stack([
        (c - 0.5) / 0.35,
        (np.log(w) - np.log(0.045)) / 0.8,
        (a - 1.25) / 0.75,
    ], axis=1).astype(np.float32)
    return (U[:N_TRAIN], z[:N_TRAIN]), (U[N_TRAIN:], z[N_TRAIN:]), x


def pod_floor(U_train, U_val, rank):
    _, _, Vt = np.linalg.svd(U_train, full_matrices=False)
    V = Vt[:rank].T
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
    n_freq: int = 16
    hidden: int = 64

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
        return nn.Dense(1)(h)[:, 0]


def rel_l2_sq(pred, true):
    return jnp.sum((pred - true) ** 2) / (jnp.sum(true**2) + 1e-12)


def train_model(apply_fn, params, U_train, z_train, U_val, z_val, seed):
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


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    results = {"meta": {"rank": RANK, "steps": STEPS, "batch": BATCH,
                        "n_train": N_TRAIN, "n_val": N_VAL, "seed": SEED,
                        "peak_lr": PEAK_LR, "weight_decay": WEIGHT_DECAY,
                        "family": "gaussian-bump source, moving-kink solutions"}}
    coord_ckpts = {}

    for N in NS:
        (U_tr, z_tr), (U_va, z_va), x = make_data(N)
        t0 = time.time()

        pod = pod_floor(U_tr, U_va, RANK)

        rng = jax.random.PRNGKey(SEED)
        modelA = GridTiedDecoder(N=N, rank=RANK)
        pA = modelA.init(rng, jnp.zeros((3,), jnp.float32))["params"]
        applyA = lambda p, z: modelA.apply({"params": p}, z)
        pA, _ = train_model(applyA, pA, U_tr, z_tr, U_va, z_va, seed=SEED)
        errA = eval_rel_l2(applyA, pA, U_va, z_va)

        xj = jnp.asarray(x, dtype=jnp.float32)
        modelB = CoordDecoder()
        pB = modelB.init(rng, xj, jnp.zeros((3,), jnp.float32))["params"]
        applyB = lambda p, z, xg=xj: modelB.apply({"params": p}, xg, z)
        pB, _ = train_model(applyB, pB, U_tr, z_tr, U_va, z_va, seed=SEED)
        errB = eval_rel_l2(applyB, pB, U_va, z_va)
        coord_ckpts[N] = pB

        dt = time.time() - t0
        results[N] = {"pod": pod, "grid_tied": errA, "coord_net": errB,
                      "seconds": round(dt, 1)}
        print(f"N={N:5d}  POD-{RANK}={pod:.3e}  grid-tied={errA:.3e}  "
              f"coord-net={errB:.3e}  [{dt:.0f}s]", flush=True)

    # Mesh transfer: train coarse, evaluate fine (truth = fine-grid FD solve).
    N_src, N_dst = NS[0], NS[-1]
    (_, _), (U_va_dst, z_va_dst), x_dst = make_data(N_dst)
    modelB = CoordDecoder()
    xj_dst = jnp.asarray(x_dst, dtype=jnp.float32)
    applyB_dst = lambda p, z: modelB.apply({"params": p}, xj_dst, z)
    err_transfer = eval_rel_l2(applyB_dst, coord_ckpts[N_src], U_va_dst, z_va_dst)
    results["mesh_transfer"] = {"train_N": N_src, "eval_N": N_dst, "rel_l2": err_transfer}
    print(f"mesh-transfer: coord net trained N={N_src}, evaluated N={N_dst}: "
          f"rel_l2={err_transfer:.3e}", flush=True)

    out = sys.argv[1] if len(sys.argv) > 1 else "poisson1d_bump_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
