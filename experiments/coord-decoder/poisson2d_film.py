"""2D Poisson coord-net, upgraded: FiLM conditioning + more data + importance
sampling. Goal: push the fit floor down at each resolution so the
error-vs-resolution behavior is visible over a wider range.

Same bump family / FD-CG f64 ground truth as poisson2d_diag_nf.py, but:
  - n_train 2048, n_val 256 (4x data)
  - FiLM: z -> per-layer scale/shift of a 5x256 trunk (translation-friendly)
  - n_freq 32 Fourier features per axis
  - per-sample point sampling: half uniform, half concentrated near the
    source center (the loss signal for the sharp region)
  - 120k steps

Saves an in-resolution result json + checkpoint; cross-resolution reference
eval happens separately from the checkpoints.

Usage:  N=64 [STEPS=120000] python poisson2d_film.py [outdir]
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
N_TRAIN = int(os.environ.get("N_TRAIN", "2048"))
N_VAL = int(os.environ.get("N_VAL", "256"))
STEPS = int(os.environ.get("STEPS", "120000"))
BATCH = 32
P_POINTS = int(os.environ.get("P_POINTS", "8192"))
PEAK_LR = 2e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05
SEED = int(os.environ.get("SEED", "0"))
N_FREQ = int(os.environ.get("N_FREQ", "32"))
HIDDEN = int(os.environ.get("HIDDEN", "256"))
N_LAYERS = 5
EVAL_EVERY = 2000
CG_TOL = 1e-11
CG_MAXITER = 40_000

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


def sample_params(seed=SEED, m=None):
    rng = np.random.default_rng(seed)
    m = m or (N_TRAIN + N_VAL)
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


def build_snapshots(n, chunk=512):
    cx, cy, w, a, z = sample_params()
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    m = len(cx)
    t0 = time.time()
    U = np.zeros((m, n, n))
    res_max = 0.0
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        F = np.stack([a[i] * np.exp(-((Xi - cx[i]) ** 2 + (Yi - cy[i]) ** 2)
                                    / (2 * w[i] ** 2)) for i in range(s, e)])
        U_int = np.asarray(fd_solve_batch(jnp.asarray(F), n))
        U[s:e, 1:-1, 1:-1] = U_int
        r = np.asarray(neg_lap_interior(jnp.asarray(U_int[0]), n)) - F[0]
        res_max = max(res_max, np.linalg.norm(r) / np.linalg.norm(F[0]))
    print(f"  FOM: {m} CG solves in {time.time()-t0:.0f}s, "
          f"spot rel residual {res_max:.2e}", flush=True)
    return U.reshape(m, n * n), z, cx, cy, w


# ----------------------------- FiLM coord net -----------------------------

COORD_FEATS = 2 * (2 * N_FREQ + 1)


def init_dense(key, d_in, d_out):
    W = jax.random.normal(key, (d_in, d_out), dtype=F32) * np.sqrt(1.0 / d_in)
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F32)}


def init_film_net(key):
    keys = jax.random.split(key, N_LAYERS + 4)
    trunk = [init_dense(keys[0], COORD_FEATS, HIDDEN)]
    for i in range(1, N_LAYERS):
        trunk.append(init_dense(keys[i], HIDDEN, HIDDEN))
    out = init_dense(keys[N_LAYERS], HIDDEN, 1)
    z_embed = init_dense(keys[N_LAYERS + 1], 4, 64)
    film = init_dense(keys[N_LAYERS + 2], 64, N_LAYERS * 2 * HIDDEN)
    # film output starts near zero -> identity modulation at init
    film["W"] = film["W"] * 0.01
    return {"trunk": trunk, "out": out, "z_embed": z_embed, "film": film}


def coord_features(xy):
    j = jnp.arange(1, N_FREQ + 1, dtype=F32)

    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)

    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1])], axis=1)


def film_apply(params, z, xy):
    g = jax.nn.swish(z @ params["z_embed"]["W"] + params["z_embed"]["b"])
    film = (g @ params["film"]["W"] + params["film"]["b"]).reshape(
        N_LAYERS, 2, HIDDEN)
    h = coord_features(xy)
    for i, lyr in enumerate(params["trunk"]):
        h = h @ lyr["W"] + lyr["b"]
        h = h * (1.0 + film[i, 0]) + film[i, 1]
        h = jax.nn.swish(h)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]


# ----------------------------- training -----------------------------

def rel_l2_sq(pred, true):
    return jnp.sum((pred - true) ** 2) / (jnp.sum(true**2) + 1e-12)


def sample_points(np_rng, b_idx, cx, cy, w, n, P):
    """Half uniform grid indices, half near each sample's source center."""
    B = len(b_idx)
    Pu, Pb = P // 2, P - P // 2
    idx = np.empty((B, P), dtype=np.int64)
    idx[:, :Pu] = np_rng.integers(0, n * n, size=(B, Pu))
    for bi, i in enumerate(b_idx):
        px = cx[i] + np_rng.normal(0.0, 2.0 * w[i], Pb)
        py = cy[i] + np_rng.normal(0.0, 2.0 * w[i], Pb)
        ix = np.clip(np.round(px * (n - 1)), 0, n - 1).astype(np.int64)
        iy = np.clip(np.round(py * (n - 1)), 0, n - 1).astype(np.int64)
        idx[bi, Pu:] = ix * n + iy
    return idx


def main():
    backend = jax.default_backend()
    print(f"jax_backend={backend}", flush=True)
    n_nodes = N * N
    print(f"N={N} ({n_nodes} nodes), steps={STEPS}, n_train={N_TRAIN}, "
          f"hidden={HIDDEN}x{N_LAYERS}, n_freq={N_FREQ}", flush=True)

    U, z, cx, cy, w = build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]

    U_tr32 = jnp.asarray(U_tr, dtype=F32)
    U_va32 = jnp.asarray(U_va, dtype=F32)
    z_tr32 = jnp.asarray(z_tr, dtype=F32)
    z_va32 = jnp.asarray(z_va, dtype=F32)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    coords32 = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1),
                           dtype=F32)

    params = init_film_net(jax.random.PRNGKey(SEED))
    n_par = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
    print(f"  params: {n_par}", flush=True)

    warmup = max(1, int(STEPS * WARMUP_FRAC))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-7)
    opt = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)
    opt_state = opt.init(params)

    def batched_loss(params, U_b, z_b, idx):
        def one(z_i, u_i, idx_i):
            pred = film_apply(params, z_i, coords32[idx_i])
            return rel_l2_sq(pred, u_i[idx_i])
        return jnp.mean(jax.vmap(one)(z_b, U_b, idx))

    @jax.jit
    def step(params, opt_state, U_b, z_b, idx):
        loss, grads = jax.value_and_grad(batched_loss)(params, U_b, z_b, idx)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def val_loss(params):
        def one(zu):
            z_i, u_i = zu
            pred = film_apply(params, z_i, coords32)
            return rel_l2_sq(pred, u_i)
        return jnp.mean(jax.lax.map(one, (z_va32, U_va32)))

    np_rng = np.random.default_rng(SEED)
    P = min(P_POINTS, n_nodes)
    best_val, best_params = float("inf"), params
    t0 = time.time()
    for it in range(STEPS):
        b_idx = np_rng.choice(N_TRAIN, size=BATCH, replace=False)
        p_idx = jnp.asarray(sample_points(np_rng, b_idx, cx, cy, w, N, P))
        params, opt_state, _ = step(params, opt_state, U_tr32[b_idx],
                                    z_tr32[b_idx], p_idx)
        if it % EVAL_EVERY == 0 or it == STEPS - 1:
            v = float(val_loss(params))
            if v < best_val:
                best_val, best_params = v, params
            if it % (EVAL_EVERY * 10) == 0:
                print(f"  step {it:7d}  val {v:.3e}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"  trained {STEPS} steps in {time.time()-t0:.0f}s "
          f"(best val loss {best_val:.3e})", flush=True)

    # full-grid f64 eval
    preds = []
    for i in range(N_VAL):
        preds.append(np.asarray(film_apply(best_params, jnp.asarray(z_va[i]),
                                           coords32), dtype=np.float64))
    preds = np.stack(preds)
    rel = np.linalg.norm(preds - U_va, axis=1) / np.linalg.norm(U_va, axis=1)
    err = float(rel.mean())
    print(f"RESULT N={N}  film-coord={err:.3e} ({n_par} params)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"film_results_N{N}.json"), "w") as f:
        json.dump({"N": N, "backend": backend, "film_coord": err,
                   "params": n_par, "steps": STEPS, "seed": SEED,
                   "n_train": N_TRAIN, "hidden": HIDDEN, "n_freq": N_FREQ},
                  f, indent=2)
    with open(os.path.join(OUTDIR, f"film_params_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, best_params), f)
    print("wrote results + checkpoint", flush=True)


if __name__ == "__main__":
    main()
