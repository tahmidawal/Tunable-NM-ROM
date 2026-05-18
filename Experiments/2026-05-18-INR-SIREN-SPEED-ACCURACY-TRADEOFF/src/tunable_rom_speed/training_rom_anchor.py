"""Baseline ModulatedSIREN training with TWO additions:

  (1) Strong L2 penalty λ_z · mean ‖z‖²  on the encoder's output.
  (2) Anchor-at-zero loss λ_a · rel_l2(decode(z=0, x), u_mean(x)):
      forces decode(0, ·) to recover the *training mean* field. This is
      the cold-start prior for NM-ROM.

The mean training field u_mean is computed once over U_train (on the
grid). Per-batch, the same sparse query coords used for the recon term
are used to bilinear-interp u_mean as the anchor target.

Everything else (sparse coord sampling, bilinear GT, optimizer) is
identical to training_rom.train_rom_ready.
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


def _bilinear_batch(u_mean_grid, x_batch):
    """Bilinear-interp `u_mean_grid` (N x N) at coordinates `x_batch` (B, M, 2).

    Returns (B, M). Implemented in numpy outside JIT for simplicity (batches
    are small; cost is negligible relative to GPU training step).
    """
    B, M, _ = x_batch.shape
    out = np.empty((B, M), dtype=np.float32)
    for i in range(B):
        out[i] = _bilinear_sample_2d(u_mean_grid, x_batch[i])
    return out


def train_anchor(
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
    lambda_anchor: float,
    lambda_anchor_warmup_steps: int,
    seed: int,
    log_every: int = 500,
):
    n_train = U_train.shape[0]
    rng = np.random.default_rng(seed)

    # Mean training field — the anchor target at z=0.
    u_mean_grid = U_train.mean(axis=0).reshape(N, N).astype(np.float32)
    print(f"[anchor] u_mean: shape={u_mean_grid.shape}  "
          f"min={u_mean_grid.min():.3e}  max={u_mean_grid.max():.3e}  "
          f"norm={np.linalg.norm(u_mean_grid):.3e}")

    x_val_np, u_val_np = make_val_set(U_val, points_per_sample_val, N, spatial_dim)
    U_val_j = jnp.asarray(U_val)
    x_val = jnp.asarray(x_val_np)
    u_val = jnp.asarray(u_val_np)

    init_u = jnp.asarray(U_train[0])
    init_x = jnp.asarray(x_val_np[0, :8])
    rng_jax = jax.random.PRNGKey(seed)
    params = model.init(rng_jax, init_u, init_x)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(num_epochs, warmup_steps + 1),
        end_value=1e-6,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    latent_dim = model.latent_dim
    z_zero = jnp.zeros((latent_dim,), dtype=jnp.float32)

    def _loss_fn(params, U_batch, x_batch, u_t_batch, u_mean_batch, lam_z, lam_a):
        # Recon term (with encoder).
        def per_sample_rec(u_in, x_q, u_t):
            z, tokens = model.apply({"params": params}, u_in, method=model.encode)
            u_pred = model.apply(
                {"params": params}, z, tokens, x_q,
                method=model.decode_points,
            )
            return _rel_l2(u_pred, u_t), jnp.sum(z * z)
        recs, znorm2s = jax.vmap(per_sample_rec)(U_batch, x_batch, u_t_batch)
        rec_mean = jnp.mean(recs)
        zn_mean = jnp.mean(znorm2s)

        # Anchor term: decode at z=0 with the SAME coord batch used for recon.
        def per_sample_anchor(x_q, u_m):
            u_pred0 = model.apply(
                {"params": params}, z_zero, None, x_q,
                method=model.decode_points,
            )
            return _rel_l2(u_pred0, u_m)
        anchors = jax.vmap(per_sample_anchor)(x_batch, u_mean_batch)
        anchor_mean = jnp.mean(anchors)

        loss = rec_mean + lam_z * zn_mean + lam_a * anchor_mean
        return loss, (rec_mean, zn_mean, anchor_mean)

    @jax.jit
    def step(params, opt_state, U_batch, x_batch, u_t_batch, u_mean_batch, lam_z, lam_a):
        (loss, (rec, zn, an)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params, U_batch, x_batch, u_t_batch, u_mean_batch, lam_z, lam_a,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, rec, zn, an

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
        f"[train_anchor] decoder={model.decoder_kind} N={N} latent={latent_dim} "
        f"epochs={num_epochs} batch={batch_size} M_train={points_per_sample_train} "
        f"λ_z={lambda_z} λ_z_warmup={lambda_z_warmup_steps} "
        f"λ_a={lambda_anchor} λ_a_warmup={lambda_anchor_warmup_steps}"
    )
    t0 = time.time()
    initial_val = float(eval_val(params))
    print(f"[train_anchor] initial val rel-L2 = {initial_val:.4e}")
    if np.isfinite(initial_val):
        best_val = initial_val

    for epoch in range(num_epochs):
        idx = rng.choice(n_train, size=batch_size, replace=False)
        U_batch_np = U_train[idx]
        x_np, u_np = _make_coords_and_targets(
            U_batch_np, points_per_sample_train, on_grid_frac_train, N, spatial_dim, rng,
        )
        u_mean_np = _bilinear_batch(u_mean_grid, x_np)
        lam_z_t = float(lambda_z * min(1.0, epoch / max(lambda_z_warmup_steps, 1)))
        lam_a_t = float(lambda_anchor * min(1.0, epoch / max(lambda_anchor_warmup_steps, 1)))
        params, opt_state, loss, rec, zn, an = step(
            params, opt_state, jnp.asarray(U_batch_np),
            jnp.asarray(x_np), jnp.asarray(u_np), jnp.asarray(u_mean_np),
            jnp.asarray(lam_z_t), jnp.asarray(lam_a_t),
        )
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val = float(eval_val(params))
            zbar = float(mean_znorm(params, jnp.asarray(U_train[:64])))
            elapsed = time.time() - t0
            history.append((
                epoch, float(loss), float(rec), float(zn), float(an),
                val, zbar, lam_z_t, lam_a_t, elapsed,
            ))
            print(
                f"[anchor] ep={epoch:>7d}  loss={float(loss):.4e}  "
                f"rec={float(rec):.4e}  anchor={float(an):.4e}  "
                f"λz={lam_z_t:.2e} λa={lam_a_t:.2e}  "
                f"val={val:.4e}  best={best_val:.4e}  "
                f"||z||={zbar:.3e}  t={elapsed:6.1f}s",
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
        "lambda_anchor": lambda_anchor,
        "lambda_anchor_warmup_steps": lambda_anchor_warmup_steps,
    }


def save_anchor_checkpoint(path, params, config: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"params": params, "config": config, "meta": meta, "tokens_ref": None},
            f,
        )
