"""Autoencoder training loop and checkpoint I/O for Poisson."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..models.autoencoder import ViTLinearCPAutoencoder


def _rel_l2_sq(u_pred, u_true):
    return jnp.sum((u_pred - u_true) ** 2) / (jnp.sum(u_true**2) + 1e-6)


def _batched_loss(params, model, batch):
    preds = jax.vmap(lambda u: model.apply({"params": params}, u))(batch)
    return jnp.mean(jax.vmap(_rel_l2_sq)(preds, batch))


def train_autoencoder(
    model: ViTLinearCPAutoencoder,
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
) -> Tuple[dict, dict]:
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.asarray(U_train[0]))["params"]

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
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_loss(params, val):
        return _batched_loss(params, model, val)

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
            v = float(eval_loss(params, jnp.asarray(U_val)))
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
