"""Autoencoder training loop and checkpoint I/O for Poisson."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..models.autoencoder import ViTViTAutoencoder
from ..fom.poisson import PoissonFOM


def _rel_l2_sq(u_pred, u_true):
    return jnp.sum((u_pred - u_true) ** 2) / (jnp.sum(u_true**2) + 1e-6)


def _batched_loss(params, model, batch, *, lap_weight=0.0, K_op=None):
    """Field rel-L2² loss + optional Laplacian-aware term.

    The Laplacian term `||K(u_pred) - K(u_true)||² / ||K(u_true)||²` penalizes
    high-frequency error in the decoder output that the plain field loss
    ignores. Without it, the un-patchify decoder produces fields with
    patch-boundary discontinuities whose Laplacian is huge; the NM-ROM
    Poisson residual `K u - F` becomes uninformative because almost any
    decoder output has a wildly varying Laplacian.
    """
    preds = jax.vmap(lambda u: model.apply({"params": params}, u))(batch)
    field_loss = jnp.mean(jax.vmap(_rel_l2_sq)(preds, batch))
    if lap_weight > 0.0 and K_op is not None:
        Kp = jax.vmap(K_op)(preds)
        Kt = jax.vmap(K_op)(batch)
        lap_loss = jnp.mean(jax.vmap(_rel_l2_sq)(Kp, Kt))
        return field_loss + lap_weight * lap_loss
    return field_loss


def train_autoencoder(
    model: ViTViTAutoencoder,
    U_train: np.ndarray,
    U_val: np.ndarray,
    *,
    num_epochs: int = 100_000,
    batch_size: int = 32,
    peak_lr: float = 1e-3,
    weight_decay: float = 5e-4,
    warmup_frac: float = 0.05,
    seed: int = 42,
    log_every: int | None = None,
    lap_weight: float = 0.0,
    N: int = 0,
    spatial_dim: int = 0,
) -> Tuple[dict, dict]:
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.asarray(U_train[0]))["params"]

    # Optional Laplacian-aware loss: needs the FOM K operator.
    K_op = None
    if lap_weight > 0.0:
        assert N > 0 and spatial_dim > 0, "lap_weight>0 requires N and spatial_dim"
        fom = PoissonFOM(N=N, spatial_dim=spatial_dim)
        K_op = fom.K_op

    warmup_steps = max(1, int(num_epochs * warmup_frac))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=num_epochs - warmup_steps,
        end_value=1e-6,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        loss_fn = lambda p: _batched_loss(p, model, batch, lap_weight=lap_weight, K_op=K_op)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_loss_chunk(params, val_chunk):
        # Eval the field loss only (Laplacian loss is just for shaping training).
        return _batched_loss(params, model, val_chunk)

    val_eval_chunk = max(1, batch_size)

    def eval_loss(params, val_array):
        # Chunked eval to keep VRAM bounded for the symmetric ViT decoder.
        nv = val_array.shape[0]
        total = 0.0
        for start in range(0, nv, val_eval_chunk):
            end = min(start + val_eval_chunk, nv)
            chunk = jnp.asarray(val_array[start:end])
            total += float(eval_loss_chunk(params, chunk)) * (end - start)
        return total / nv

    np_rng = np.random.default_rng(seed)
    M = U_train.shape[0]
    log_every = log_every or max(500, num_epochs // 100)
    best_val = float("inf")
    best_params = params
    history = []

    for epoch in range(num_epochs):
        idx = np_rng.choice(M, size=batch_size, replace=False)
        batch = jnp.asarray(U_train[idx])
        params, opt_state, loss = step(params, opt_state, batch)
        if epoch % log_every == 0:
            v = eval_loss(params, U_val)
            history.append((epoch, float(loss), v))
            if v < best_val:
                best_val = v
                best_params = params

    return best_params, {"best_val": best_val, "history": history}


def save_checkpoint(path, params, config: dict, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "config": config, "meta": meta or {}}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)
