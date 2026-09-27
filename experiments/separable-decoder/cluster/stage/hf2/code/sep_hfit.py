"""ROUND 5 (burgers-accuracy) -- h REFINEMENT IN COEFFICIENT SPACE.

Round 3/4 localised the entire remaining Burgers error in ONE rung: the map
h: R^K -> R^R reaches only ~1/32 of the span its own bank provides (oracle
8.24e-3 against a span least-squares floor of 2.60e-4 at N=1024; the ratio is
resolution-independent, 24.6-36.8x at N=256).  This module attacks that rung
directly, using the exact identity established in `sep_coeff_extract.py`:

    ||G h(z) - u||^2 = (h(z) - c*)^T Gram (h(z) - c*) + f(u)^2 .          (*)

With the bank G FROZEN, (*) says fitting h against full fields and fitting h
against the precomputed span coefficients c* IN THE GRAM METRIC are the SAME
optimisation problem -- not an approximation of it.  The second costs
O(S r^2) instead of O(S n_pts r) and needs a 70 MB file instead of the data,
so the whole h search runs locally in minutes per arm.

Two further consequences are used here:

  * WHITENING.  Gram = L L^T; put q(z) = L^T h(z) and a = L^T c*.  Then the
    objective is the plain Euclidean ||q(z) - a||^2, and the last linear layer
    of q converts back to h EXACTLY (h_row = q_row @ L^{-1}), so nothing about
    the model class changes -- it is a reparameterisation of the SAME decoder.
    This matters because the trained bank is badly conditioned (cond(G) ~ 2.6e4
    at N=256/R=512, so the field-space loss weights coefficient directions by
    up to ~7e8 in energy).  Every arm here optimises the whitened problem; the
    reported errors are the field-space ones, via (*).
  * EXACT DIAGNOSTICS.  recon, the span floor, and the K-dimensional
    representation ORACLE on fresh states are all computable from
    (Gram, c*, ||u||^2, f^2) alone, so the accuracy ladder above the rollout is
    reproduced locally, bit-comparably with the cluster drivers.

PURE NEURAL is preserved: no POD, no data SVD.  L is the Cholesky factor of the
LEARNED bank's own Gram matrix over the grid -- a function of the network and
the coordinates only, with no PDE solution data in it.

Truth usage: c* on TRAINING states is training data and is fitted.  c* on the
fresh TEST states is used only by the labelled oracle/span-floor DIAGNOSTICS,
exactly as the r3/r4 drivers already use test truth for those two rungs.  No
part of this file enters a solve path.
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

F64 = jnp.float64


def log(*a):
    print(*a, flush=True)


# ----------------------------- model pieces ---------------------------------
# Deliberately identical in FORM to sep_common.init_separable / head, so a
# fitted q converts into a drop-in `params["h"]` / `params["h_lin"]` for the
# incumbent decoder with no change to the decoder code.

def init_head(key, k_lat, r_feat, hidden=256, layers=2, lin_scale=0.3,
              h_ff=0, h_ff_scale=1.0):
    """Mirrors sep_common.init_separable's h-track exactly, including the
    default-off latent Fourier features (`h_ff`)."""
    sizes = [k_lat + 2 * int(h_ff)] + [hidden] * layers + [r_feat]
    ps = []
    for i in range(len(sizes) - 1):
        key, k1 = jax.random.split(key)
        ps.append((jax.random.normal(k1, (sizes[i], sizes[i + 1]), dtype=F64)
                   * jnp.sqrt(2.0 / sizes[i]),
                   jnp.zeros((sizes[i + 1],), dtype=F64)))
    key, k2 = jax.random.split(key)
    lin = jax.random.normal(k2, (k_lat, r_feat), dtype=F64) * lin_scale
    out = dict(h=ps, h_lin=lin)
    if h_ff:
        key, k3 = jax.random.split(key)
        out["hB"] = (jax.random.normal(k3, (k_lat, int(h_ff)), dtype=F64)
                     * float(h_ff_scale))
    return out


def head_apply(hp, z):
    x = z
    if "hB" in hp:
        ang = 2.0 * jnp.pi * (z @ hp["hB"])
        x = jnp.concatenate([z, jnp.sin(ang), jnp.cos(ang)], axis=-1)
    for w, b in hp["h"][:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = hp["h"][-1]
    return x @ w + b + z @ hp["h_lin"]


def to_q(hp, L):
    """h-parameterisation -> q = L^T h.  Exact, folds into the last layer."""
    out = {k: v for k, v in hp.items()}
    w, b = hp["h"][-1]
    out["h"] = hp["h"][:-1] + [(w @ L, b @ L)]
    out["h_lin"] = hp["h_lin"] @ L
    return out


def to_h(qp, Linv):
    """q-parameterisation -> h = L^{-T} q.  The inverse of `to_q`."""
    out = {k: v for k, v in qp.items()}
    w, b = qp["h"][-1]
    out["h"] = qp["h"][:-1] + [(w @ Linv, b @ Linv)]
    out["h_lin"] = qp["h_lin"] @ Linv
    return out


# ------------------------------- metrics ------------------------------------

def rel_from_q(Q, A, fl2, un2):
    """field-space relative L2 of every state, via (*).  Q,A: (S,r)."""
    d = Q - A
    return jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=1) + fl2, 0.0)
                    / jnp.maximum(un2, 1e-300))


def batched_rel(qp, Z, A, fl2, un2, chunk=4096):
    out = []
    for s in range(0, A.shape[0], chunk):
        e = min(s + chunk, A.shape[0])
        out.append(rel_from_q(head_apply(qp, Z[s:e]), A[s:e], fl2[s:e],
                              un2[s:e]))
    return jnp.concatenate(out)


# --------------------------- latent LM (oracle) -----------------------------

def make_lm(qp, iters=120):
    """Damped LM on ||q(z) - a|| for one state; vmapped by the caller.
    Truth-using DIAGNOSTIC (this is the representation oracle)."""
    def r_of(z, a):
        return head_apply(qp, z[None, :])[0] - a

    def rJ(z, a):
        return r_of(z, a), jax.jacfwd(r_of)(z, a)

    def solve(z0, a):
        k = z0.shape[0]
        eye = jnp.eye(k, dtype=F64)

        def body(st, _):
            z, lam, val = st
            r, J = rJ(z, a)
            H = J.T @ J
            g = J.T @ r
            dz = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H)) + 1e-30 * eye,
                                  -g)
            zn = z + dz
            vn = jnp.linalg.norm(r_of(zn, a))
            acc = jnp.isfinite(vn) & (vn < val)
            z = jnp.where(acc, zn, z)
            val = jnp.where(acc, vn, val)
            lam = jnp.where(acc, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            return (z, lam, val), None

        v0 = jnp.linalg.norm(r_of(z0, a))
        (z, _, val), _ = jax.lax.scan(body, (z0, 1e-6, v0), None, length=iters)
        return z, val

    return jax.jit(jax.vmap(solve))


def oracle(qp, A, fl2, un2, Z_init, iters=120, chunk=512):
    """min over z of the field-space relative L2, per state.  Z_init: (S,n0,k)
    -- n0 restarts per state, the best kept."""
    lm = make_lm(qp, iters)
    S, n0, k = Z_init.shape
    best = jnp.full((S,), jnp.inf)
    bestz = jnp.zeros((S, k), dtype=F64)
    for j in range(n0):
        vals, zs = [], []
        for s in range(0, S, chunk):
            e = min(s + chunk, S)
            z, v = lm(Z_init[s:e, j], A[s:e])
            zs.append(z)
            vals.append(v)
        v = jnp.concatenate(vals)
        z = jnp.concatenate(zs)
        take = v < best
        best = jnp.where(take, v, best)
        bestz = jnp.where(take[:, None], z, bestz)
    rel = jnp.sqrt(jnp.maximum(best ** 2 + fl2, 0.0)
                   / jnp.maximum(un2, 1e-300))
    return np.asarray(rel), np.asarray(bestz)


# ------------------------------- fitting ------------------------------------

def fit(key, A, un2, fl2, k_lat, r_feat, steps, lr, hidden, layers,
        batch=4096, wd=0.0, ema_decay=0.0, w_state=None, qp0=None, Z0=None,
        z_polish_every=0, z_polish_iters=40, log_every=5000, tag="",
        time_cap=0.0, norm="snap", h_ff=0, h_ff_scale=1.0):
    """Adam(W) on (q, Z) against the exact whitened objective.

    norm: 'snap' divides each state's squared error by its own ||u||^2 (so the
        minimised quantity is the mean per-snapshot relative MSE -- the square
        of the reported metric); 'global' divides by the mean ||u||^2 over all
        states, reproducing the loss the round-3 reference recipe actually
        used.  The round-3 snap_norm arm was WORSE than the global one, so
        both are kept as arms rather than assumed.
    w_state: optional per-state weight (S,), normalised to mean 1.
    qp0/Z0:  warm start (e.g. the checkpoint's own h, whitened).
    z_polish_every: run an exact code-only LM refinement of Z every so many
        steps (the codes are FREE VARIABLES; joint Adam has never been checked
        for their convergence -- lever 2 of the handoff)."""
    S = A.shape[0]
    key, kq, kz = jax.random.split(key, 3)
    qp = (qp0 if qp0 is not None else
          init_head(kq, k_lat, r_feat, hidden, layers, h_ff=h_ff,
                    h_ff_scale=h_ff_scale))
    Z = (jnp.asarray(Z0, dtype=F64) if Z0 is not None
         else 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64))
    w = (jnp.ones((S,), dtype=F64) if w_state is None
         else jnp.asarray(w_state, dtype=F64))
    w = w / jnp.mean(w)
    den = (jnp.maximum(un2, 1e-300) if norm == "snap"
           else jnp.full_like(un2, float(jnp.mean(un2))))
    inv = w / den                             # per-state loss weight

    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)

    def wd_mask(pz):
        q, z = pz
        m = {"h": [(True, False) for _ in q["h"]], "h_lin": False}
        if "hB" in q:
            m["hB"] = False            # fixed random latent frequencies
        return (m, False)

    opt = (optax.adamw(sched, weight_decay=wd, mask=wd_mask) if wd > 0
           else optax.adam(sched))
    state = opt.init((qp, Z))

    def loss_fn(pz, Ab, invb, idx):
        q, z = pz
        d = head_apply(q, z[idx]) - Ab
        return jnp.mean(invb * jnp.sum(d * d, axis=1))

    @jax.jit
    def step(pz, st, ema, kk, A_, inv_):
        idx = jax.random.choice(kk, S, shape=(min(batch, S),), replace=False)
        val, gr = jax.value_and_grad(loss_fn)(pz, A_[idx], inv_[idx], idx)
        upd, st = opt.update(gr, st, pz)
        pz = optax.apply_updates(pz, upd)
        ema = jax.tree_util.tree_map(
            lambda e, q_: ema_decay * e + (1.0 - ema_decay) * q_, ema, pz)
        return pz, st, ema, val

    pz = (qp, Z)
    ema = pz
    t0 = time.time()
    done = 0
    val = jnp.inf
    for i in range(steps):
        key, kk = jax.random.split(key)
        pz, state, ema, val = step(pz, state, ema, kk, A, inv)
        done = i + 1
        if done % log_every == 0 or i == 0:
            r = batched_rel(pz[0], pz[1], A, fl2, un2)
            log(f"   hfit[{tag}] {done:7d}/{steps} loss {float(val):.4e} "
                f"recon {float(jnp.mean(r)):.4e} [{time.time()-t0:.0f}s]")
        if z_polish_every and done % z_polish_every == 0 and done < steps:
            q_, z_ = pz
            lm = make_lm(q_, z_polish_iters)
            zs = []
            for s in range(0, S, 4096):
                e = min(s + 4096, S)
                zz, _ = lm(z_[s:e], A[s:e])
                zs.append(zz)
            pz = (q_, jnp.concatenate(zs))
            state = opt.init(pz)          # codes moved: Adam moments are stale
        if time_cap and (time.time() - t0) > time_cap:
            log(f"   hfit[{tag}] TIME CAP at step {done}")
            break
    if ema_decay > 0:
        r_raw = float(jnp.mean(batched_rel(pz[0], pz[1], A, fl2, un2)))
        r_ema = float(jnp.mean(batched_rel(ema[0], ema[1], A, fl2, un2)))
        if r_ema < r_raw:
            pz = ema
        used_ema = r_ema < r_raw
    else:
        used_ema = False
    return pz[0], pz[1], dict(steps=steps, steps_done=done, lr=lr,
                              hidden=hidden, layers=layers, batch=int(batch),
                              wd=wd, ema_decay=ema_decay, used_ema=used_ema,
                              norm=norm, z_polish_every=int(z_polish_every),
                              h_ff=int(h_ff), h_ff_scale=float(h_ff_scale),
                              seconds=time.time() - t0,
                              final_loss=float(val))
