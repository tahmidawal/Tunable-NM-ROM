"""Training loop for the INR autoencoder.

Each step samples:
  - a batch of B parameter snapshots (indexes into U_train),
  - for each snapshot, M_train query points: a mix of on-grid coords and
    off-mesh uniform coords. Ground truth at these coords comes from the
    analytical Poisson formula evaluated at the same x.

Loss: mean rel-L2 across the batch, computed per-sample as
    rel_L2(u_pred, u_true) = ||u_pred - u_true||_2 / ||u_true||_2

Eval (every `log_every` steps): mean rel-L2 on the validation set, computed
on a fixed off-mesh grid of `points_per_sample_val` points per snapshot.
The eval point set is regenerated once at the start and held fixed across
the run for direct comparability.
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
from .fom.poisson_analytical import PoissonAnalytical


def _rel_l2(u_pred, u_true):
    num = jnp.linalg.norm(u_pred - u_true)
    den = jnp.linalg.norm(u_true) + 1e-12
    return num / den


def _make_coords_and_targets(
    freqs_batch: np.ndarray,        # (B, d)
    M: int,
    on_grid_frac: float,
    N: int,
    spatial_dim: int,
    rng: np.random.Generator,
    fom: PoissonAnalytical,
):
    """For each of B snapshots, sample M coords (mix of on-grid + off-mesh)
    and evaluate the analytical solution at those coords."""
    B = freqs_batch.shape[0]
    on_grid_M = int(round(on_grid_frac * M))
    off_mesh_M = M - on_grid_M

    x_all = np.empty((B, M, spatial_dim), dtype=np.float32)
    u_all = np.empty((B, M), dtype=np.float32)

    for i in range(B):
        freqs = freqs_batch[i]
        if on_grid_M > 0:
            idx = rng.integers(0, N, size=(on_grid_M, spatial_dim))
            x_grid = (idx.astype(np.float32) / (N - 1)) * fom.L
        else:
            x_grid = np.empty((0, spatial_dim), dtype=np.float32)
        if off_mesh_M > 0:
            x_off = rng.uniform(0.0, fom.L, size=(off_mesh_M, spatial_dim)).astype(np.float32)
        else:
            x_off = np.empty((0, spatial_dim), dtype=np.float32)
        xs = np.concatenate([x_grid, x_off], axis=0)
        x_all[i] = xs
        u_all[i] = fom.u_at_points(freqs, xs)
    return x_all, u_all


def make_val_set(
    freqs_val: np.ndarray,
    M_val: int,
    N: int,
    spatial_dim: int,
    fom: PoissonAnalytical,
    seed: int = 1234,
):
    """Fixed validation off-mesh point set + ground truth.

    Returns (x_val, u_val) with shapes
        x_val: (n_val, M_val, d)
        u_val: (n_val, M_val)
    All M_val points per snapshot are random-uniform off-mesh (no on-grid
    portion) so the eval metric reflects the partial-decoding capability.
    """
    rng = np.random.default_rng(seed)
    n_val = freqs_val.shape[0]
    x_val = rng.uniform(0.0, fom.L, size=(n_val, M_val, spatial_dim)).astype(np.float32)
    u_val = np.empty((n_val, M_val), dtype=np.float32)
    for i in range(n_val):
        u_val[i] = fom.u_at_points(freqs_val[i], x_val[i])
    return x_val, u_val


def train_inr_autoencoder(
    model: INRAutoencoder,
    U_train: np.ndarray,          # (n_train, N**d) — on-grid snapshots for the encoder
    freqs_train: np.ndarray,      # (n_train, d)    — for analytical GT at query coords
    U_val: np.ndarray,            # (n_val, N**d)
    freqs_val: np.ndarray,        # (n_val, d)
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
    fom = PoissonAnalytical(N=N, spatial_dim=spatial_dim)

    rng = np.random.default_rng(seed)
    n_train = U_train.shape[0]

    # Build val coords once and keep them fixed.
    x_val_np, u_val_np = make_val_set(
        freqs_val, points_per_sample_val, N, spatial_dim, fom
    )
    U_val_jnp = jnp.asarray(U_val)
    x_val = jnp.asarray(x_val_np)
    u_val = jnp.asarray(u_val_np)

    # Init params.
    init_u = jnp.asarray(U_train[0])
    init_x = jnp.asarray(x_val_np[0, :8])     # any small coord batch
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

    # Loss is vmapped over batch.
    def _loss_fn(params, U_batch, x_batch, u_target_batch):
        # x_batch: (B, M, d), u_target_batch: (B, M)
        def per_sample(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return _rel_l2(u_pred, u_t)
        return jnp.mean(jax.vmap(per_sample)(U_batch, x_batch, u_target_batch))

    @jax.jit
    def step(params, opt_state, U_batch, x_batch, u_target_batch):
        loss, grads = jax.value_and_grad(_loss_fn)(params, U_batch, x_batch, u_target_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_val(params):
        def per_sample(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return _rel_l2(u_pred, u_t)
        return jnp.mean(jax.vmap(per_sample)(U_val_jnp, x_val, u_val))

    history = []
    best_val = float("inf")
    best_params = params

    print(
        f"[train] decoder={model.decoder_kind} N={N} latent={model.latent_dim} "
        f"epochs={num_epochs} batch={batch_size} M_train={points_per_sample_train} "
        f"M_val={points_per_sample_val} on_grid_frac={on_grid_frac_train}"
    )
    print(
        f"[train] n_train={n_train} n_val={U_val.shape[0]}"
    )
    t0 = time.time()
    initial_val = float(eval_val(params))
    print(f"[train] initial val rel-L2 = {initial_val:.4e}")
    if np.isfinite(initial_val):
        best_val = initial_val

    for epoch in range(num_epochs):
        idx = rng.choice(n_train, size=batch_size, replace=False)
        U_batch = jnp.asarray(U_train[idx])
        x_np, u_np = _make_coords_and_targets(
            freqs_train[idx],
            points_per_sample_train,
            on_grid_frac_train,
            N,
            spatial_dim,
            rng,
            fom,
        )
        x_batch = jnp.asarray(x_np)
        u_target_batch = jnp.asarray(u_np)
        params, opt_state, loss = step(params, opt_state, U_batch, x_batch, u_target_batch)

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val = float(eval_val(params))
            elapsed = time.time() - t0
            history.append((epoch, float(loss), val, elapsed))
            print(
                f"[train] epoch={epoch:>7d}  loss={float(loss):.4e}  "
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


def save_checkpoint(path, params, config: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "config": config, "meta": meta}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)
