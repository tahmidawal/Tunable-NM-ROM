"""Autoencoder training loop and checkpoint I/O."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..models.autoencoder import ViTCPAutoencoder


def _rel_l2_sq(u_pred, u_true):
    num = jnp.sum((u_pred - u_true) ** 2)
    den = jnp.sum(u_true**2) + 1e-6
    return num / den


def _batched_loss(params, model, batch):
    """Mean squared-relative-L2 over a batch of flat fields."""
    preds = jax.vmap(lambda u: model.apply({"params": params}, u))(batch)
    losses = jax.vmap(_rel_l2_sq)(preds, batch)
    return jnp.mean(losses)


def train_autoencoder(
    model: ViTCPAutoencoder,
    snapshots_train: np.ndarray,  # (M, N**d) float32
    snapshots_val: np.ndarray,
    *,
    num_epochs: int = 80_000,
    batch_size: int = 32,
    peak_lr: float = 2e-3,
    weight_decay: float = 5e-4,
    warmup_frac: float = 0.1,
    seed: int = 0,
    log_every: int | None = None,
) -> Tuple[dict, dict]:
    """Train the autoencoder, return (best_params, train_meta)."""
    rng = jax.random.PRNGKey(seed)
    init_u = jnp.asarray(snapshots_train[0])
    params = model.init(rng, init_u)["params"]

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
        loss, grads = jax.value_and_grad(_batched_loss)(params, model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_loss(params, val):
        return _batched_loss(params, model, val)

    np_rng = np.random.default_rng(seed)
    M = snapshots_train.shape[0]
    log_every = log_every or max(100, num_epochs // 100)
    best_val = float("inf")
    best_params = params
    history = []

    for epoch in range(num_epochs):
        idx = np_rng.choice(M, size=batch_size, replace=False)
        batch = jnp.asarray(snapshots_train[idx])
        params, opt_state, loss = step(params, opt_state, batch)
        if epoch % log_every == 0:
            v = float(eval_loss(params, jnp.asarray(snapshots_val)))
            history.append((epoch, float(loss), v))
            if v < best_val:
                best_val = v
                best_params = params

    train_meta = {"best_val": best_val, "history": history}
    return best_params, train_meta


def save_checkpoint(path: str | Path, params: dict, config: dict, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "config": config, "meta": meta or {}}, f)


def load_checkpoint(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
