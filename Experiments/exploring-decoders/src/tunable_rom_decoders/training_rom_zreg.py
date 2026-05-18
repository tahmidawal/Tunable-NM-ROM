"""Baseline ModulatedSIREN training with an L2 penalty on the encoder
output `z`. Pushes the trained latent distribution toward 0.

Loss = mean rel-L2(u_pred, u_t) + lambda_z * mean ||z||^2.

`lambda_z` is annealed in linearly from 0 over `lambda_z_warmup_steps`.

Everything else (sparse coord sampling, bilinear GT, optimizer) matches
training_rom.train_rom_ready exactly.
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
from .training_rom import _bilinear_sample_2d, make_val_set


def _rel_l2(u_pred, u_true):
    num = jnp.linalg.norm(u_pred - u_true)
    den = jnp.linalg.norm(u_true) + 1e-12
    return num / den


def _make_coords_and_targets(U_batch, M, on_grid_frac, N, spatial_dim, rng):
    assert spatial_dim == 2
    B = U_batch.shape[0]
    on_grid_M = int(round(on_grid_frac * M))
    off_mesh_M = M - on_grid_M
    x_all = np.empty((B, M, spatial_dim), dtype=np.float32)
    u_all = np.empty((B, M), dtype=np.float32)
    for i in range(B):
        if on_grid_M > 0:
            idx = rng.integers(0, N, size=(on_grid_M, spatial_dim))
            x_grid = idx.astype(np.float32) / (N - 1)
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


def train_zreg(
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
    lambda_z: float,
    lambda_z_warmup_steps: int,
    seed: int,
    log_every: int = 500,
):
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

    def _loss_fn(params, U_batch, x_batch, u_t_batch, lam):
        def per_sample(u_in, x_q, u_t):
            z, tokens = model.apply({"params": params}, u_in, method=model.encode)
            u_pred = model.apply(
                {"params": params}, z, tokens, x_q,
                method=model.decode_points,
            )
            rec = _rel_l2(u_pred, u_t)
            znorm2 = jnp.sum(z * z)
            return rec, znorm2
        recs, znorm2s = jax.vmap(per_sample)(U_batch, x_batch, u_t_batch)
        rec_mean = jnp.mean(recs)
        zn_mean = jnp.mean(znorm2s)
        return rec_mean + lam * zn_mean, (rec_mean, zn_mean)

    @jax.jit
    def step(params, opt_state, U_batch, x_batch, u_t_batch, lam):
        (loss, (rec, zn)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params, U_batch, x_batch, u_t_batch, lam,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, rec, zn

    @jax.jit
    def eval_val(params):
        def per_sample(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return _rel_l2(u_pred, u_t)
        return jnp.mean(jax.vmap(per_sample)(U_val_j, x_val, u_val))

    @jax.jit
    def mean_znorm(params, U_batch):
        def per(u_in):
            z, _ = model.apply({"params": params}, u_in, method=model.encode)
            return jnp.sqrt(jnp.sum(z * z))
        return jnp.mean(jax.vmap(per)(U_batch))

    history = []
    best_val = float("inf")
    best_params = params

    print(
        f"[train_zreg] decoder={model.decoder_kind} N={N} latent={model.latent_dim} "
        f"epochs={num_epochs} batch={batch_size} M_train={points_per_sample_train} "
        f"lambda_z={lambda_z} lambda_z_warmup={lambda_z_warmup_steps}"
    )
    t0 = time.time()
    initial_val = float(eval_val(params))
    initial_zn = float(mean_znorm(params, jnp.asarray(U_train[:64])))
    print(f"[train_zreg] initial val={initial_val:.4e}  mean ||z||={initial_zn:.3e}")
    if np.isfinite(initial_val):
        best_val = initial_val

    for epoch in range(num_epochs):
        idx = rng.choice(n_train, size=batch_size, replace=False)
        U_batch = jnp.asarray(U_train[idx])
        x_np, u_np = _make_coords_and_targets(
            U_train[idx], points_per_sample_train, on_grid_frac_train, N, spatial_dim, rng,
        )
        lam = float(lambda_z * min(1.0, epoch / max(lambda_z_warmup_steps, 1)))
        params, opt_state, loss, rec, zn = step(
            params, opt_state, U_batch, jnp.asarray(x_np), jnp.asarray(u_np),
            jnp.asarray(lam),
        )
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val = float(eval_val(params))
            zbar = float(mean_znorm(params, jnp.asarray(U_train[:64])))
            elapsed = time.time() - t0
            history.append((epoch, float(loss), float(rec), float(zn), val, zbar, lam, elapsed))
            print(
                f"[train_zreg] ep={epoch:>7d}  loss={float(loss):.4e}  "
                f"rec={float(rec):.4e}  ||z||²_mean={float(zn):.3e}  "
                f"lam={lam:.3e}  val={val:.4e}  best={best_val:.4e}  "
                f"||z||_mean={zbar:.3e}  t={elapsed:6.1f}s",
                flush=True,
            )
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = params

    return best_params, {
        "best_val_rel_l2": best_val,
        "history": history,
        "decoder_kind": model.decoder_kind,
        "lambda_z": lambda_z,
        "lambda_z_warmup_steps": lambda_z_warmup_steps,
    }


def save_zreg_checkpoint(path, params, config: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"params": params, "config": config, "meta": meta, "tokens_ref": None},
            f,
        )
