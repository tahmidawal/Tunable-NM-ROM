"""Fixed-bank head/code refinement with exact compressed field objectives."""
import time

import heat_core as hc
import jax
import jax.numpy as jnp
import numpy as np
import optax


def split_head(params):
    return {key: params[key] for key in ("h", "h_lin")}


def full_params(base, head):
    return dict(base, **head)


def compressed_loss(pz, triangular, target, norm2, perpendicular2, indices):
    head, codes = pz
    predicted = hc.sc.head(head, codes[indices])@triangular.T
    return jnp.mean((jnp.sum((predicted-target[indices])**2, axis=1)+perpendicular2[indices])/norm2[indices])


def compression(fields, projection):
    target = fields@projection
    norm2 = jnp.sum(fields**2, axis=1)
    perpendicular2 = jnp.maximum(norm2-jnp.sum(target**2, axis=1), 0)
    return target, norm2, perpendicular2


def train(base, codes0, triangular, target, norm2, perpendicular2, seed, settings):
    updates, batch = settings["updates"], settings["batch_size"]
    head = split_head(base)
    pz = (head, codes0)
    schedule = optax.warmup_cosine_decay_schedule(0, settings["learning_rate"], min(200, updates//4),
                                                updates, settings["learning_rate"]*.01)
    opt = optax.adam(schedule)
    state = opt.init(pz)
    @jax.jit
    def step(pz, state, key, triangular, target, norm2, perpendicular2):
        indices = jax.random.randint(key, (batch,), 0, target.shape[0])
        value, gradient = jax.value_and_grad(compressed_loss)(pz, triangular, target, norm2, perpendicular2, indices)
        update, state = opt.update(gradient, state)
        return optax.apply_updates(pz, update), state, value
    key = jax.random.PRNGKey(seed)
    history = []
    start = time.perf_counter()
    for i in range(updates):
        key, batch_key = jax.random.split(key)
        pz, state, value = step(pz, state, batch_key, triangular, target, norm2, perpendicular2)
        if i == 0 or (i+1) % 1000 == 0:
            record = dict(update=i+1, loss=float(value), seconds=time.perf_counter()-start)
            history.append(record)
            print("refine", seed, target.shape[0], record, flush=True)
    jax.block_until_ready(pz)
    elapsed = time.perf_counter()-start
    all_loss = float(compressed_loss(pz, triangular, target, norm2, perpendicular2, jnp.arange(len(target))))
    return full_params(base, pz[0]), pz[1], dict(seconds=elapsed, final_full_relative_mse=all_loss,
                                               history=history, seed=seed, snapshots=len(target),
                                               sampled_snapshots=updates*batch,
                                               trained_head_parameters=sum(x.size for x in jax.tree.leaves(pz[0])),
                                               code_scalars=pz[1].size)


def fit_fields(params, codes, bank, projection, triangular, fields, settings):
    """Evaluation only: extra starts never seed autonomous rollout or training."""
    fields = jnp.asarray(fields)
    target = fields@projection
    library = hc.sc.head(params, codes)@triangular.T
    nearest = jnp.argsort(jnp.sum((target[:, None]-library[None])**2, axis=2), axis=1)[:, :settings["fit_starts"]-1]
    starts = jnp.concatenate((codes[nearest], jnp.broadcast_to(jnp.mean(codes, axis=0), (len(fields), 1, codes.shape[1]))), axis=1)
    solve = hc.make_lm(hc.sc.head, settings["fit_budget"], settings["fit_gradient_tolerance"])
    @jax.jit
    def all_fits(params, triangular, target, starts):
        def one(t, zs):
            return jax.vmap(lambda z: solve(params, triangular, t, z))(zs)
        return jax.lax.map(lambda inputs: one(*inputs), (target, starts))
    zs, info = all_fits(params, triangular, target, starts)
    best = jnp.argmin(info[:, :, 3], axis=1)
    selected = zs[jnp.arange(len(fields)), best]
    reconstructed = hc.sc.head(params, selected)@bank.T
    return reconstructed, selected, info, best, nearest
