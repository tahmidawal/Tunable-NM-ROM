"""Anchor recipe + ROM-aware Laplacian loss.

Loss = rec + λ_z · ‖z‖² + λ_a · anchor + λ_L · laplacian_residual,
where laplacian_residual is

   mean over batch / mean over M_lap interior nodes of
   ((coeff * u_c - sum_neighbours) / dx² - F[interior_idx])²

evaluated by decoding the autoencoder at the 5-point stencil of M_lap
random interior nodes per snapshot. The Forcing F is taken from data
(boundary-masked).

This directly trains the decoder to produce ROM-compatible fields, which
should both (a) lower the achievable cold-start ROM rel-L² floor (since
the network learns to satisfy the discrete operator) and (b) sharpen
the local minimum around the true z so GN finds it from z=0.
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
    B, M, _ = x_batch.shape
    out = np.empty((B, M), dtype=np.float32)
    for i in range(B):
        out[i] = _bilinear_sample_2d(u_mean_grid, x_batch[i])
    return out


def _sample_lap_stencil(N: int, B: int, M_lap: int, rng):
    """Sample M_lap interior nodes per snapshot. For each, return:
      stencil_coords: (B, M_lap, 5, 2) normalised to [0,1]
      flat_center: (B, M_lap) flat index of the centre node
    """
    # interior grid is i in [1, N-2], j in [1, N-2]
    ii = rng.integers(1, N - 1, size=(B, M_lap))
    jj = rng.integers(1, N - 1, size=(B, M_lap))
    cx = ii.astype(np.float32) / (N - 1)
    cy = jj.astype(np.float32) / (N - 1)
    sx = jnp = None  # not used
    # 5-point stencil: centre, N, S, E, W
    coords = np.zeros((B, M_lap, 5, 2), dtype=np.float32)
    coords[:, :, 0, 0] = cx;             coords[:, :, 0, 1] = cy
    coords[:, :, 1, 0] = (ii + 1) / (N - 1);  coords[:, :, 1, 1] = cy
    coords[:, :, 2, 0] = (ii - 1) / (N - 1);  coords[:, :, 2, 1] = cy
    coords[:, :, 3, 0] = cx;                   coords[:, :, 3, 1] = (jj + 1) / (N - 1)
    coords[:, :, 4, 0] = cx;                   coords[:, :, 4, 1] = (jj - 1) / (N - 1)
    flat = (ii * N + jj).astype(np.int32)
    return coords, flat


def train_lap(
    model: INRAutoencoder,
    U_train: np.ndarray,
    U_val: np.ndarray,
    F_train: np.ndarray,
    *,
    N: int,
    spatial_dim: int,
    num_epochs: int,
    batch_size: int,
    points_per_sample_train: int,
    points_per_sample_val: int,
    on_grid_frac_train: float,
    M_lap: int,
    peak_lr: float,
    weight_decay: float,
    warmup_steps: int,
    lambda_z: float,
    lambda_z_warmup_steps: int,
    lambda_anchor: float,
    lambda_anchor_warmup_steps: int,
    lambda_lap: float,
    lambda_lap_warmup_steps: int,
    seed: int,
    log_every: int = 500,
):
    n_train = U_train.shape[0]
    rng = np.random.default_rng(seed)

    u_mean_grid = U_train.mean(axis=0).reshape(N, N).astype(np.float32)
    dx = 1.0 / (N - 1)
    coeff = 2.0 * spatial_dim
    print(f"[lap] u_mean range [{u_mean_grid.min():.3e}, {u_mean_grid.max():.3e}]  "
          f"dx={dx:.3e}  coeff={coeff}  M_lap={M_lap}")

    x_val_np, u_val_np = make_val_set(U_val, points_per_sample_val, N, spatial_dim)
    U_val_j = jnp.asarray(U_val); x_val = jnp.asarray(x_val_np); u_val = jnp.asarray(u_val_np)

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

    def _loss_fn(params, U_b, x_b, u_t_b, u_m_b, lap_coords_b, F_lap_b,
                 lam_z, lam_a, lam_L):
        def per_sample(u_in, x_q, u_t, u_m, lap_c, F_lap):
            # Recon
            z, tokens = model.apply({"params": params}, u_in, method=model.encode)
            u_pred = model.apply(
                {"params": params}, z, tokens, x_q,
                method=model.decode_points,
            )
            rec = _rel_l2(u_pred, u_t)
            znorm2 = jnp.sum(z * z)
            # Anchor at z=0
            u_pred0 = model.apply(
                {"params": params}, z_zero, None, x_q,
                method=model.decode_points,
            )
            anchor = _rel_l2(u_pred0, u_m)
            # Laplacian residual at M_lap interior nodes
            # lap_c shape (M_lap, 5, 2) → flatten to (M_lap*5, 2) for decode_points
            lap_flat = lap_c.reshape(-1, 2)
            u_lap = model.apply(
                {"params": params}, z, tokens, lap_flat,
                method=model.decode_points,
            ).reshape(-1, 5)  # (M_lap, 5)
            R_lap = (coeff * u_lap[:, 0] - jnp.sum(u_lap[:, 1:], axis=1)) / (dx ** 2) - F_lap
            # Use relative residual norm so the scale matches recon.
            lap_resid = jnp.linalg.norm(R_lap) / (jnp.linalg.norm(F_lap) + 1e-12)
            return rec, znorm2, anchor, lap_resid
        recs, znorm2s, anchors, lapres = jax.vmap(per_sample)(
            U_b, x_b, u_t_b, u_m_b, lap_coords_b, F_lap_b,
        )
        rec_m = jnp.mean(recs)
        zn_m = jnp.mean(znorm2s)
        an_m = jnp.mean(anchors)
        lap_m = jnp.mean(lapres)
        loss = rec_m + lam_z * zn_m + lam_a * an_m + lam_L * lap_m
        return loss, (rec_m, zn_m, an_m, lap_m)

    @jax.jit
    def step(params, opt_state, U_b, x_b, u_t_b, u_m_b, lap_c_b, F_lap_b,
             lam_z, lam_a, lam_L):
        (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params, U_b, x_b, u_t_b, u_m_b, lap_c_b, F_lap_b, lam_z, lam_a, lam_L,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, *aux

    @jax.jit
    def eval_val(params):
        def per(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return _rel_l2(u_pred, u_t)
        return jnp.mean(jax.vmap(per)(U_val_j, x_val, u_val))

    @jax.jit
    def mean_znorm(params, U_b):
        def per(u_in):
            z, _ = model.apply({"params": params}, u_in, method=model.encode)
            return jnp.sqrt(jnp.sum(z * z))
        return jnp.mean(jax.vmap(per)(U_b))

    history = []
    best_val = float("inf")
    best_params = params

    print(
        f"[train_lap] decoder={model.decoder_kind} N={N} latent={latent_dim} "
        f"epochs={num_epochs} batch={batch_size} M_train={points_per_sample_train} "
        f"M_lap={M_lap} λ_z={lambda_z} λ_a={lambda_anchor} λ_L={lambda_lap}"
    )
    t0 = time.time()
    initial_val = float(eval_val(params))
    print(f"[train_lap] initial val rel-L2 = {initial_val:.4e}")
    if np.isfinite(initial_val):
        best_val = initial_val

    for epoch in range(num_epochs):
        idx = rng.choice(n_train, size=batch_size, replace=False)
        U_b_np = U_train[idx]
        F_b_np = F_train[idx]  # already boundary-masked at save time
        x_np, u_np = _make_coords_and_targets(
            U_b_np, points_per_sample_train, on_grid_frac_train, N, spatial_dim, rng,
        )
        u_m_np = _bilinear_batch(u_mean_grid, x_np)
        lap_c_np, lap_flat_np = _sample_lap_stencil(N, batch_size, M_lap, rng)
        # Gather F values at the centres of each sample.
        F_lap_np = np.take_along_axis(F_b_np, lap_flat_np, axis=1)
        lam_z_t = float(lambda_z * min(1.0, epoch / max(lambda_z_warmup_steps, 1)))
        lam_a_t = float(lambda_anchor * min(1.0, epoch / max(lambda_anchor_warmup_steps, 1)))
        lam_L_t = float(lambda_lap * min(1.0, epoch / max(lambda_lap_warmup_steps, 1)))
        params, opt_state, loss, rec, zn, an, lap = step(
            params, opt_state,
            jnp.asarray(U_b_np), jnp.asarray(x_np),
            jnp.asarray(u_np), jnp.asarray(u_m_np),
            jnp.asarray(lap_c_np), jnp.asarray(F_lap_np),
            jnp.asarray(lam_z_t), jnp.asarray(lam_a_t), jnp.asarray(lam_L_t),
        )
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val = float(eval_val(params))
            zbar = float(mean_znorm(params, jnp.asarray(U_train[:64])))
            elapsed = time.time() - t0
            history.append((epoch, float(loss), float(rec), float(zn), float(an), float(lap),
                            val, zbar, lam_z_t, lam_a_t, lam_L_t, elapsed))
            print(
                f"[lap] ep={epoch:>7d}  loss={float(loss):.4e}  "
                f"rec={float(rec):.4e}  anch={float(an):.4e}  "
                f"lap={float(lap):.4e}  λL={lam_L_t:.2e}  "
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
        "lambda_z": lambda_z, "lambda_z_warmup_steps": lambda_z_warmup_steps,
        "lambda_anchor": lambda_anchor, "lambda_anchor_warmup_steps": lambda_anchor_warmup_steps,
        "lambda_lap": lambda_lap, "lambda_lap_warmup_steps": lambda_lap_warmup_steps,
        "M_lap": M_lap,
    }


def save_lap_checkpoint(path, params, config: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"params": params, "config": config, "meta": meta, "tokens_ref": None},
            f,
        )
