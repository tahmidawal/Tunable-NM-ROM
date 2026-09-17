"""Training phases for the lshape cell: the parent's `pbh_fit.py` with the boundary factor
injected through a `features(params, xy)` callable and the head objective reduced to plain
reconstruction (DESIGN.md sections 4 and 5).

Spatial phases (joint / bank): Adam on the selected parameter subtrees plus the
per-snapshot latent, minibatched over sources and nodes, relative field MSE with a
feature-Gram orthonormality term. The bank phase optimises FREE per-snapshot coefficients.

Head phase: with the bank frozen the reconstruction loss separates exactly through the thin
QR, so the head is fitted full batch in R dimensions.

Large arrays are explicit jit ARGUMENTS, never closed over.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import sep_common as sc
import lsh_core as K_


PHASE_KEYS = {'joint': ('B', 'g', 'h', 'h_lin'), 'bank': ('B', 'g')}


def _schedule(lr, steps, cfg):
    return optax.warmup_cosine_decay_schedule(
        0., lr, max(1, int(cfg['warmup_fraction'] * steps)), steps,
        lr * cfg['final_learning_rate_fraction'])


def train_spatial_phase(features, params, latent, U, coords, cfg, phase, steps, lr, seed, tag,
                        wall_cap):
    assert phase in PHASE_KEYS
    start = time.perf_counter()
    keys = PHASE_KEYS[phase]
    train = {k: params[k] for k in keys}
    frozen = {k: params[k] for k in params if k not in keys}
    U = jnp.asarray(U)
    coords = jnp.asarray(coords)
    denom = jnp.sum(U * U, axis=1)
    npts = int(U.shape[1])
    opt = optax.adam(_schedule(lr, steps, cfg))
    v = (train, jnp.asarray(latent))
    state = opt.init(v)
    ow = float(cfg['orthogonality_weight'])
    sb, pb = int(cfg['source_batch']), int(cfg['point_batch'])

    def loss(v, U, coords, denom, si, pi):
        w, lat = v
        p = {**frozen, **w}
        G = features(p, coords[pi])
        H = lat[si] if phase == 'bank' else sc.head(p, lat[si])
        err = H @ G.T - U[si[:, None], pi[None, :]]
        rec = jnp.mean(jnp.mean(err * err, axis=1) / (denom[si] / npts))
        gram = G.T @ G / (G.shape[0] * p['out_scale'] ** 2)
        orth = jnp.mean((gram - jnp.eye(gram.shape[0], dtype=jnp.float64)) ** 2)
        return rec + ow * orth, (rec, orth)

    @jax.jit
    def step(v, state, key, U, coords, denom):
        key, ks, kp = jax.random.split(key, 3)
        si = jax.random.randint(ks, (sb,), 0, U.shape[0])
        pi = jax.random.randint(kp, (pb,), 0, U.shape[1])
        (value, parts), grad = jax.value_and_grad(loss, has_aux=True)(v, U, coords, denom, si, pi)
        update, state = opt.update(grad, state, v)
        return optax.apply_updates(v, update), state, key, value, parts

    key = jax.random.PRNGKey(seed)
    compile_start = time.perf_counter()
    step = step.lower(v, state, key, U, coords, denom).compile()
    compile_seconds = time.perf_counter() - compile_start
    records, done, finite, capped = [], 0, True, False
    loop_start = time.perf_counter()
    block = int(cfg['timing_block_updates'])
    while done < steps:
        count = min(block, steps - done)
        for _ in range(count):
            v, state, key, value, parts = step(v, state, key, U, coords, denom)
        jax.block_until_ready((v, value, parts))
        done += count
        scalar = float(value)
        finite = bool(np.isfinite(scalar))
        if done == block or done % int(cfg['log_every_updates']) == 0 or done == steps or not finite:
            records.append(dict(updates=done, loss=scalar, reconstruction=float(parts[0]),
                                orth=float(parts[1]), seconds=time.perf_counter() - loop_start))
            print('train', tag, records[-1], flush=True)
        if not finite:
            break
        if time.perf_counter() - loop_start > wall_cap:
            capped = done < steps
            break
    trained, lat = v
    final = {**frozen, **trained}
    jax.block_until_ready(final)
    info = dict(tag=tag, phase=phase, updates=done, requested=steps, learning_rate=lr,
                seed=seed, finite=finite, wall_capped=capped, source_batch=sb, point_batch=pb,
                source_exposures=done * sb, sources=int(U.shape[0]), points=npts,
                compile_seconds=compile_seconds, optimizer_seconds=time.perf_counter() - loop_start,
                total_seconds=time.perf_counter() - start, progress=records,
                final_weights_sha256=K_.weights_sha(final), final_latent_sha256=K_.sha_array(lat))
    return final, np.asarray(lat), info


def train_head_phase(bank_params, K, Rtot, Rg, T, perp2, nu2, cfg, steps, lr, seed, tag, wall_cap,
                     init_head=True, head_params=None, latent=None):
    """Frozen-bank head fit, plain reconstruction objective (DESIGN.md section 5)."""
    start = time.perf_counter()
    frozen = {k: bank_params[k] for k in bank_params if k not in ('h', 'h_lin')}
    if init_head:
        fresh = sc.init_separable(jax.random.PRNGKey(seed + 1), K, Rtot, **cfg['arch'])
        train = dict(h=fresh['h'], h_lin=fresh['h_lin'])
        Z = 0.1 * jax.random.normal(jax.random.PRNGKey(seed + 2), (int(T.shape[0]), K), dtype=jnp.float64)
    else:
        train = dict(h=head_params['h'], h_lin=head_params['h_lin'])
        Z = jnp.asarray(latent)
    opt = optax.adam(_schedule(lr, steps, cfg))
    v = (train, Z)
    state = opt.init(v)

    def loss(v, Rg, T, perp2, nu2):
        w, Z = v
        p = {**frozen, **w}
        H = sc.head(p, Z)
        res = H @ Rg.T - T
        return jnp.mean((jnp.sum(res * res, axis=1) + perp2) / nu2)

    @jax.jit
    def step(v, state, Rg, T, perp2, nu2):
        value, grad = jax.value_and_grad(loss)(v, Rg, T, perp2, nu2)
        update, state = opt.update(grad, state, v)
        return optax.apply_updates(v, update), state, value

    args = (jnp.asarray(Rg), jnp.asarray(T), jnp.asarray(perp2), jnp.asarray(nu2))
    compile_start = time.perf_counter()
    step = step.lower(v, state, *args).compile()
    compile_seconds = time.perf_counter() - compile_start
    records, done, finite, capped = [], 0, True, False
    loop_start = time.perf_counter()
    block = int(cfg['timing_block_updates'])
    while done < steps:
        count = min(block, steps - done)
        for _ in range(count):
            v, state, value = step(v, state, *args)
        jax.block_until_ready((v, value))
        done += count
        scalar = float(value)
        finite = bool(np.isfinite(scalar))
        if done == block or done % int(cfg['log_every_updates']) == 0 or done == steps or not finite:
            records.append(dict(updates=done, loss=scalar, seconds=time.perf_counter() - loop_start))
            print('head', tag, records[-1], flush=True)
        if not finite:
            break
        if time.perf_counter() - loop_start > wall_cap:
            capped = done < steps
            break
    trained, Z = v
    final = {**frozen, **trained}
    jax.block_until_ready(final)
    info = dict(tag=tag, phase='head_frozen_bank', K=int(K), R=int(Rtot), objective='reconstruction',
                updates=done, requested=steps, learning_rate=lr, seed=seed, finite=finite,
                wall_capped=capped, full_batch=True, sources=int(T.shape[0]),
                compile_seconds=compile_seconds, optimizer_seconds=time.perf_counter() - loop_start,
                total_seconds=time.perf_counter() - start, progress=records,
                final_weights_sha256=K_.weights_sha(final), final_latent_sha256=K_.sha_array(Z))
    return final, np.asarray(Z), info
