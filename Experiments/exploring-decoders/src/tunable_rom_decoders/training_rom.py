"""ROM-faithful retraining for the 3 Pareto-frontier decoders.

Key design choices versus training.py (the analytical-data AE training):
  - Snapshots come from CG-discrete data (data-must-match-operator rule).
  - Sampling pattern from training.py is preserved: M sparse query coords
    per snapshot per step, mix of on-grid and off-mesh, regenerated each
    step. This was empirically essential — full-grid loss was found to
    not descend in 1000 epochs whereas the sparse-coord regime descended
    cleanly in the original AE training.
  - GT at query coords is bilinear interpolation of the CG snapshot
    (exact at grid nodes, smooth off-mesh). The ROM residual lives only
    at grid nodes, so this preserves operator-matching where it matters.
  - Loss is mean rel-L2 across the batch, matching training.py.
  - For xattn we also compute and stash mean training tokens so the ROM
    solver can use a frozen token reference at solve time.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .autoencoder import INRAutoencoder


def _rel_l2(u_pred, u_true):
    num = jnp.linalg.norm(u_pred - u_true)
    den = jnp.linalg.norm(u_true) + 1e-12
    return num / den


def _bilinear_sample_2d(u_grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear interp on a uniform N x N grid spanning [0, 1]^2.

    u_grid: (N, N), x: (M, 2) in [0, 1]. Returns (M,).
    """
    N = u_grid.shape[0]
    xs = np.clip(x[:, 0] * (N - 1), 0.0, N - 1)
    ys = np.clip(x[:, 1] * (N - 1), 0.0, N - 1)
    x0 = np.floor(xs).astype(np.int32); x1 = np.minimum(x0 + 1, N - 1)
    y0 = np.floor(ys).astype(np.int32); y1 = np.minimum(y0 + 1, N - 1)
    fx = xs - x0; fy = ys - y0
    w00 = (1 - fx) * (1 - fy)
    w01 = (1 - fx) * fy
    w10 = fx * (1 - fy)
    w11 = fx * fy
    v = w00 * u_grid[x0, y0] + w01 * u_grid[x0, y1] \
      + w10 * u_grid[x1, y0] + w11 * u_grid[x1, y1]
    return v.astype(np.float32)


def _make_coords_and_targets(
    U_batch: np.ndarray,            # (B, N**d)
    M: int,
    on_grid_frac: float,
    N: int,
    spatial_dim: int,
    rng: np.random.Generator,
):
    """For each of B snapshots, sample M coords (mix of on-grid + off-mesh)
    and bilinear-interpolate the CG snapshot at those coords."""
    assert spatial_dim == 2
    B = U_batch.shape[0]
    on_grid_M = int(round(on_grid_frac * M))
    off_mesh_M = M - on_grid_M

    x_all = np.empty((B, M, spatial_dim), dtype=np.float32)
    u_all = np.empty((B, M), dtype=np.float32)

    for i in range(B):
        if on_grid_M > 0:
            idx = rng.integers(0, N, size=(on_grid_M, spatial_dim))
            x_grid = (idx.astype(np.float32) / (N - 1))
        else:
            x_grid = np.empty((0, spatial_dim), dtype=np.float32)
        if off_mesh_M > 0:
            x_off = rng.uniform(0.0, 1.0, size=(off_mesh_M, spatial_dim)).astype(np.float32)
        else:
            x_off = np.empty((0, spatial_dim), dtype=np.float32)
        xs = np.concatenate([x_grid, x_off], axis=0)
        u_grid = U_batch[i].reshape(N, N)
        u_all[i] = _bilinear_sample_2d(u_grid, xs)
        x_all[i] = xs
    return x_all, u_all


def make_val_set(
    U_val: np.ndarray, M_val: int, N: int, spatial_dim: int, seed: int = 1234
):
    """Fixed validation random-coord point set + bilinear-interpolated GT."""
    rng = np.random.default_rng(seed)
    n_val = U_val.shape[0]
    x_val = rng.uniform(0.0, 1.0, size=(n_val, M_val, spatial_dim)).astype(np.float32)
    u_val = np.empty((n_val, M_val), dtype=np.float32)
    for i in range(n_val):
        u_val[i] = _bilinear_sample_2d(U_val[i].reshape(N, N), x_val[i])
    return x_val, u_val


def train_rom_ready(
    model: INRAutoencoder,
    U_train: np.ndarray,
    U_val: np.ndarray,
    *,
    N: int,
    spatial_dim: int,
    num_epochs: int,
    batch_size: int,
    points_per_sample_train: int,
    points_per_sample_val: int,
    on_grid_frac_train: float,
    peak_lr: float,
    weight_decay: float,
    warmup_steps: int,
    seed: int,
    log_every: int = 500,
):
    """Train the INRAutoencoder on CG-discrete data via sparse off-mesh queries
    with bilinear-interpolated GT."""
    n_train = U_train.shape[0]

    rng = np.random.default_rng(seed)
    x_val_np, u_val_np = make_val_set(U_val, points_per_sample_val, N, spatial_dim)
    U_val_j = jnp.asarray(U_val)
    x_val = jnp.asarray(x_val_np)
    u_val = jnp.asarray(u_val_np)

    init_u = jnp.asarray(U_train[0])
    init_x = jnp.asarray(x_val_np[0, :8])
    rng_jax = jax.random.PRNGKey(seed)
    params = model.init(rng_jax, init_u, init_x)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(num_epochs, warmup_steps + 1),
        end_value=1e-6,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    def _loss_fn(params, U_batch, x_batch, u_t_batch):
        def per_sample(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return _rel_l2(u_pred, u_t)
        return jnp.mean(jax.vmap(per_sample)(U_batch, x_batch, u_t_batch))

    @jax.jit
    def step(params, opt_state, U_batch, x_batch, u_t_batch):
        loss, grads = jax.value_and_grad(_loss_fn)(params, U_batch, x_batch, u_t_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_val(params):
        def per_sample(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return _rel_l2(u_pred, u_t)
        return jnp.mean(jax.vmap(per_sample)(U_val_j, x_val, u_val))

    history = []
    best_val = float("inf")
    best_params = params

    print(
        f"[train_rom] decoder={model.decoder_kind} N={N} latent={model.latent_dim} "
        f"epochs={num_epochs} batch={batch_size} M_train={points_per_sample_train} "
        f"M_val={points_per_sample_val} on_grid_frac={on_grid_frac_train}"
    )
    print(f"[train_rom] n_train={n_train} n_val={U_val.shape[0]}")
    t0 = time.time()
    initial_val = float(eval_val(params))
    print(f"[train_rom] initial val rel-L2 = {initial_val:.4e}")
    if np.isfinite(initial_val):
        best_val = initial_val

    for epoch in range(num_epochs):
        idx = rng.choice(n_train, size=batch_size, replace=False)
        U_batch = jnp.asarray(U_train[idx])
        x_np, u_np = _make_coords_and_targets(
            U_train[idx], points_per_sample_train, on_grid_frac_train, N, spatial_dim, rng,
        )
        params, opt_state, loss = step(
            params, opt_state, U_batch, jnp.asarray(x_np), jnp.asarray(u_np)
        )
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val = float(eval_val(params))
            elapsed = time.time() - t0
            history.append((epoch, float(loss), val, elapsed))
            print(
                f"[train_rom] epoch={epoch:>7d}  loss={float(loss):.4e}  "
                f"val_relL2={val:.4e}  best={best_val:.4e}  t={elapsed:6.1f}s",
                flush=True,
            )
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = params

    return best_params, {
        "best_val_rel_l2": best_val,
        "history": history,
        "decoder_kind": model.decoder_kind,
    }


def compute_tokens_ref(model: INRAutoencoder, params, U_train: np.ndarray) -> np.ndarray:
    """Mean of encoder.tokens(u) over the training set, used as ROM-time T_ref."""
    U_j = jnp.asarray(U_train)
    @jax.jit
    def tokens_of(u):
        _z, T = model.apply({"params": params}, u, method=model.encode)
        return T
    Ts = jax.vmap(tokens_of)(U_j)
    return np.asarray(jnp.mean(Ts, axis=0))


def save_rom_checkpoint(path, params, config: dict, meta: dict, tokens_ref=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"params": params, "config": config, "meta": meta, "tokens_ref": tokens_ref},
            f,
        )
