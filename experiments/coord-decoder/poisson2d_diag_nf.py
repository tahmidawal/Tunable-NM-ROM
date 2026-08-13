"""2D Poisson decoder-architecture diagnostic (pure JAX + optax, no flax).

Problem:  -Lap(u) = a * exp(-((x-cx)^2+(y-cy)^2)/(2 w^2)),  u|_boundary = 0
on [0,1]^2.  cx,cy ~ U[0.15,0.85], w ~ logU[0.02,0.1], a ~ U[0.5,2].
Ground truth: 5-point FD + matrix-free CG in float64, regenerated from seed.

Arms (decoders conditioned on true normalized params z in R^4, no encoder):
  POD-r      : SVD floor for a rank sweep (yardsticks: r=4 equal-dim, r=24).
  grid-tied  : MLP(z)->h in R^24 ; u = h @ W + b, W in R^{24 x N^2} learned.
  coord-net  : u(x,y;z) = MLP([ff(x), ff(y), z]), params independent of N.

Both arms train on the same per-step random point subsets (P<=4096); final
eval is full-grid mean rel-L2 in float64 on the val set.

Usage:  N=64 [STEPS=20000] python poisson2d_diag_nf.py [outdir]
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

N = int(os.environ.get("N", "16"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
RANK = 24
N_TRAIN, N_VAL = 512, 128
STEPS = int(os.environ.get("STEPS", "20000"))
BATCH = 32
P_POINTS = int(os.environ.get("P_POINTS", "4096"))
PEAK_LR = 2e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05
SEED = int(os.environ.get("SEED", "0"))
N_FREQ = int(os.environ.get("N_FREQ", "16"))
HIDDEN = 128
EVAL_EVERY = 1000
POD_RANKS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
CG_TOL = 1e-11
CG_MAXITER = 20_000

F32 = jnp.float32


# ----------------------------- FOM (float64) -----------------------------

def neg_lap_interior(u_int, n):
    dx = 1.0 / (n - 1)
    u = jnp.pad(u_int, 1)
    lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
           - 4.0 * u[1:-1, 1:-1]) / dx**2
    return -lap


def fd_solve_batch(F_int_batch, n):
    op = lambda v: neg_lap_interior(v, n)

    def solve_one(F_int):
        u, _ = jax.scipy.sparse.linalg.cg(op, F_int, tol=CG_TOL, maxiter=CG_MAXITER)
        return u

    return jax.lax.map(jax.jit(solve_one), F_int_batch)


def sample_params(seed=SEED):
    rng = np.random.default_rng(seed)
    m = N_TRAIN + N_VAL
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), m))
    a = rng.uniform(0.5, 2.0, m)
    z = np.stack([
        (cx - 0.5) / 0.35,
        (cy - 0.5) / 0.35,
        (np.log(w) - np.log(0.045)) / 0.8,
        (a - 1.25) / 0.75,
    ], axis=1).astype(np.float32)
    return cx, cy, w, a, z


def build_snapshots(n):
    cx, cy, w, a, z = sample_params()
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    m = len(cx)
    F = np.empty((m, n - 2, n - 2))
    for i in range(m):
        F[i] = a[i] * np.exp(-((Xi - cx[i]) ** 2 + (Yi - cy[i]) ** 2) / (2 * w[i] ** 2))
    t0 = time.time()
    U_int = np.asarray(fd_solve_batch(jnp.asarray(F), n))
    res = []
    for i in range(0, m, max(1, m // 5)):
        r = np.asarray(neg_lap_interior(jnp.asarray(U_int[i]), n)) - F[i]
        res.append(np.linalg.norm(r) / np.linalg.norm(F[i]))
    print(f"  FOM: {m} CG solves in {time.time()-t0:.0f}s, "
          f"max rel residual {max(res):.2e}", flush=True)
    U = np.zeros((m, n, n))
    U[:, 1:-1, 1:-1] = U_int
    return U.reshape(m, n * n), z


# ----------------------------- models (pure jax) -----------------------------

def init_dense(key, d_in, d_out):
    kw, _ = jax.random.split(key)
    W = jax.random.normal(kw, (d_in, d_out), dtype=F32) * np.sqrt(1.0 / d_in)
    b = jnp.zeros((d_out,), dtype=F32)
    return {"W": W, "b": b}


def mlp_apply(layers, h):
    for i, lyr in enumerate(layers):
        h = h @ lyr["W"] + lyr["b"]
        if i < len(layers) - 1:
            h = jax.nn.swish(h)
    return h


def init_mlp(key, dims):
    keys = jax.random.split(key, len(dims) - 1)
    return [init_dense(k, dims[i], dims[i + 1]) for i, k in enumerate(keys)]


def init_grid_tied(key, n_nodes):
    k1, k2 = jax.random.split(key)
    return {
        "mlp": init_mlp(k1, [4, HIDDEN, HIDDEN, RANK]),
        "W": jax.random.normal(k2, (RANK, n_nodes), dtype=F32) * 0.01,
        "b": jnp.zeros((), dtype=F32),
    }


def grid_tied_apply(params, z, idx):
    h = mlp_apply(params["mlp"], z)
    return h @ params["W"][:, idx] + params["b"]


COORD_IN = 2 * (2 * N_FREQ + 1) + 4


def init_coord(key):
    return {"mlp": init_mlp(key, [COORD_IN, HIDDEN, HIDDEN, HIDDEN, HIDDEN, 1])}


def coord_features(xy, z):
    j = jnp.arange(1, N_FREQ + 1, dtype=F32)

    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)

    zz = jnp.broadcast_to(z, (xy.shape[0], z.shape[-1]))
    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1]), zz], axis=1)


def make_coord_apply(coords32):
    def apply(params, z, idx):
        feats = coord_features(coords32[idx], z)
        return mlp_apply(params["mlp"], feats)[:, 0]
    return apply


# ----------------------------- training -----------------------------

def rel_l2_sq(pred, true):
    return jnp.sum((pred - true) ** 2) / (jnp.sum(true**2) + 1e-12)


def train_model(apply_at_idx, params, U_tr32, z_tr32, U_va32, z_va32, n_nodes, tag):
    warmup = max(1, int(STEPS * WARMUP_FRAC))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-7)
    opt = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)
    opt_state = opt.init(params)

    def batched_loss(params, U_b, z_b, idx):
        preds = jax.vmap(lambda z: apply_at_idx(params, z, idx))(z_b)
        return jnp.mean(jax.vmap(rel_l2_sq)(preds, U_b[:, idx]))

    @jax.jit
    def step(params, opt_state, U_b, z_b, idx):
        loss, grads = jax.value_and_grad(batched_loss)(params, U_b, z_b, idx)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    all_idx = jnp.arange(n_nodes)

    @jax.jit
    def val_loss(params):
        def one(z_and_u):
            z, u = z_and_u
            pred = apply_at_idx(params, z, all_idx)
            return rel_l2_sq(pred, u)
        losses = jax.lax.map(one, (z_va32, U_va32))
        return jnp.mean(losses)

    np_rng = np.random.default_rng(SEED)
    P = min(P_POINTS, n_nodes)
    best_val, best_params = float("inf"), params
    t0 = time.time()
    for it in range(STEPS):
        b_idx = np_rng.choice(N_TRAIN, size=BATCH, replace=False)
        p_idx = jnp.asarray(np_rng.choice(n_nodes, size=P, replace=False))
        params, opt_state, _ = step(params, opt_state, U_tr32[b_idx], z_tr32[b_idx], p_idx)
        if it % EVAL_EVERY == 0 or it == STEPS - 1:
            v = float(val_loss(params))
            if v < best_val:
                best_val, best_params = v, params
    print(f"  {tag}: {STEPS} steps in {time.time()-t0:.0f}s "
          f"(best val loss {best_val:.3e})", flush=True)
    return best_params


def eval_full(apply_at_idx, params, U_val, z_val, n_nodes):
    all_idx = jnp.arange(n_nodes)
    preds = []
    for i in range(z_val.shape[0]):
        preds.append(np.asarray(apply_at_idx(params, jnp.asarray(z_val[i]), all_idx),
                                dtype=np.float64))
    preds = np.stack(preds)
    rel = np.linalg.norm(preds - U_val, axis=1) / np.linalg.norm(U_val, axis=1)
    return float(rel.mean())


def n_params(tree):
    return sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(tree))


def main():
    backend = jax.default_backend()
    print(f"jax_backend={backend}", flush=True)
    n_nodes = N * N
    print(f"N={N} ({n_nodes} nodes), steps={STEPS}", flush=True)

    U, z = build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]

    t0 = time.time()
    _, S, Vt = np.linalg.svd(U_tr, full_matrices=False)
    pods = {}
    for r in POD_RANKS:
        V = Vt[:r].T
        proj = (U_va @ V) @ V.T
        pods[r] = float((np.linalg.norm(U_va - proj, axis=1)
                         / np.linalg.norm(U_va, axis=1)).mean())
    print(f"  POD ({time.time()-t0:.0f}s): "
          + "  ".join(f"r{r}={pods[r]:.3e}" for r in [4, 8, 16, 24, 64]), flush=True)

    U_tr32 = jnp.asarray(U_tr, dtype=F32)
    U_va32 = jnp.asarray(U_va, dtype=F32)
    z_tr32 = jnp.asarray(z_tr, dtype=F32)
    z_va32 = jnp.asarray(z_va, dtype=F32)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    coords32 = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), dtype=F32)

    key = jax.random.PRNGKey(SEED)

    pA = init_grid_tied(key, n_nodes)
    pA = train_model(grid_tied_apply, pA, U_tr32, z_tr32, U_va32, z_va32,
                     n_nodes, "grid-tied")
    errA = eval_full(grid_tied_apply, pA, U_va, z_va, n_nodes)

    coord_apply = make_coord_apply(coords32)
    pB = init_coord(key)
    pB = train_model(coord_apply, pB, U_tr32, z_tr32, U_va32, z_va32,
                     n_nodes, "coord-net")
    errB = eval_full(coord_apply, pB, U_va, z_va, n_nodes)

    print(f"RESULT N={N}  POD-4={pods[4]:.3e}  POD-24={pods[24]:.3e}  "
          f"grid-tied={errA:.3e} ({n_params(pA)} params)  "
          f"coord-net={errB:.3e} ({n_params(pB)} params)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"results_2d_N{N}.json"), "w") as f:
        json.dump({"N": N, "backend": backend, "pod": pods,
                   "grid_tied": errA, "coord_net": errB,
                   "params_grid": n_params(pA), "params_coord": n_params(pB),
                   "steps": STEPS, "seed": SEED,
                   "singular_values": [float(s) for s in S[:80]]},
                  f, indent=2, default=float)
    with open(os.path.join(OUTDIR, f"coord_params_2d_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, pB), f)
    print("wrote results + coord checkpoint", flush=True)


if __name__ == "__main__":
    main()
