"""imp-geometry R-BANK: Stage-A bank with a domain envelope, its ordered span and its rom.Space (DESIGN 5.2-5.3, A1).

Architecture and hyper-parameters = the dn256b Stage-A recipe (jcp-robust/train2d/stageA/smoothtrain.py RECIPE):
Fourier features (n_ff = 128, scale 4, fixed), g-MLP 2 x 1024 SiLU -> R = 512, head h(z) = MLP(z) (2 x 256) + z h_lin,
K = 16, loss = mean(err^2)/mean(U^2) + lam_orth * feature-Gram orthonormality, Adam (warm-up cosine), weight decay 1e-5
on g/h kernels, EMA 0.999, p_sub = 4096 random points per step, 300k steps, the last 10k steps full-batch.
Model code is vendor/sep_common.py (byte-identical, sha 1f1c2796); the only change is `features`: the box `bc_poly`
is replaced by the domain envelope geom.envelope_jnp(dom, .) and the Fourier map acts on X = x / 2 (unit-square
coordinates, as the incumbent). Deviations (stated in DESIGN A1): Stage A only; the full-batch phase uses a fixed
65,536-point subset when the sampling lattice has more points; ordering from the training codes in the L2(Omega) metric
of a frozen reference rule.
"""
from __future__ import annotations

import os
import sys
import time
import pickle
import hashlib

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))
import sep_common as sc          # noqa: E402  (vendored, byte-identical)
import geom as G                 # noqa: E402
import rom as RM                 # noqa: E402

F64 = jnp.float64
RECIPE = dict(K=16, R=512, steps=300000, lr=1e-3, p_sub=4096, wd=1e-5, ema_decay=0.999, full_last=10000,
              lam_orth=1e-4, arch=dict(n_ff=128, g_hidden=1024, h_hidden=256), full_cap=65536)


def features(dom, p, xy, box_only=False):
    """envelope(x) * g(FF(x / 2)) * out_scale: (n, R)."""
    X = xy / G.SIDE
    ang = 2.0 * jnp.pi * (X @ p['B'])
    ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return (p['out_scale'] * G.envelope_jnp(dom, xy, box_only))[..., None] * sc.apply_mlp(p['g'], ff)


def features_grad(dom, p, xy, box_only=False):
    """values, d/dx, d/dy of the features at xy (n, 2) by forward-mode AD."""
    f = lambda pt: features(dom, p, pt[None], box_only)[0]
    val = features(dom, p, xy, box_only)
    jx = jax.vmap(lambda pt: jax.jvp(f, (pt,), (jnp.array([1.0, 0.0]),))[1])(xy)
    jy = jax.vmap(lambda pt: jax.jvp(f, (pt,), (jnp.array([0.0, 1.0]),))[1])(xy)
    return val, jx, jy


def train(dom, coords, U, seed, steps=None, log_every=5000, log=print, time_marks=(), full_last=None):
    """Stage A on states U (S, n_pts) at coords (n_pts, 2). Returns params, codes Z, info."""
    rc = dict(RECIPE)
    steps = rc['steps'] if steps is None else steps
    if full_last is not None:
        rc['full_last'] = full_last
    coords = jnp.asarray(coords, F64)
    U = jnp.asarray(U, F64)
    S, n_pts = U.shape
    key = jax.random.PRNGKey(seed)
    key, kz, kp, kf = jax.random.split(key, 4)
    u_rms = float(jnp.sqrt(jnp.mean(U * U)))
    params = sc.init_separable(kp, rc['K'], rc['R'], out_scale=u_rms, **rc['arch'])
    Z = 0.1 * jax.random.normal(kz, (S, rc['K']), dtype=F64)
    u_ms = jnp.mean(U * U)
    if full_last is None and rc['full_last'] == 10000 and steps < 20000:                   # short (smoke) runs keep a sub-sampled phase
        rc['full_last'] = max(1, steps // 10)
    full_idx = jnp.arange(n_pts) if n_pts <= rc['full_cap'] else \
        jnp.sort(jax.random.choice(kf, n_pts, (rc['full_cap'],), replace=False))
    sched = optax.warmup_cosine_decay_schedule(0.0, rc['lr'], min(500, steps // 10 + 1), steps, rc['lr'] * 1e-2)

    def wd_mask(pz):
        p, z = pz
        mp = {k_: ([(True, False) for _ in v] if k_ in ('g', 'h') else jax.tree_util.tree_map(lambda _: False, v))
              for k_, v in p.items()}
        return (mp, jax.tree_util.tree_map(lambda _: False, z))
    opt = optax.adamw(sched, weight_decay=rc['wd'], mask=wd_mask)
    state = opt.init((params, Z))

    def loss_at(pz, U_, C_):
        p, z = pz
        Gf = features(dom, p, C_)
        H = sc.head(p, z)
        err = H @ Gf.T - U_
        rel = jnp.mean(err * err) / u_ms
        C = (Gf.T @ Gf) / (Gf.shape[0] * p['out_scale'] ** 2)
        orth = jnp.mean((C - jnp.eye(C.shape[0], dtype=F64)) ** 2)
        return rel + rc['lam_orth'] * orth, rel

    def _apply(pz, st, ema, U_, C_):
        (val, rel), grads = jax.value_and_grad(loss_at, has_aux=True)(pz, U_, C_)
        grads[0]['out_scale'] = jnp.zeros_like(grads[0]['out_scale'])
        upd, st = opt.update(grads, st, pz)
        pz = optax.apply_updates(pz, upd)
        ema = jax.tree_util.tree_map(lambda e, q: rc['ema_decay'] * e + (1.0 - rc['ema_decay']) * q, ema, pz)
        return pz, st, ema, rel

    @jax.jit
    def step_sub(pz, st, ema, k_, U_all, C_all):
        idx = jax.random.choice(k_, n_pts, shape=(rc['p_sub'],), replace=False)
        return _apply(pz, st, ema, U_all[:, idx], C_all[idx])

    @jax.jit
    def step_full(pz, st, ema, U_all, C_all, fidx):
        return _apply(pz, st, ema, U_all[:, fidx], C_all[fidx])

    pz = (params, Z)
    ema = pz
    t0 = time.time()
    hist = []
    marks_t = {}
    for i in range(steps):
        if i < steps - rc['full_last']:
            key, k_ = jax.random.split(key)
            pz, state, ema, rel = step_sub(pz, state, ema, k_, U, coords)
        else:
            pz, state, ema, rel = step_full(pz, state, ema, U, coords, full_idx)
        if (i + 1) in time_marks:                                         # measurement mode (A8): synchronised
            jax.block_until_ready(rel)
            marks_t[i + 1] = time.perf_counter()
        if (i + 1) % log_every == 0 or i == 0:
            rv = float(rel)
            hist.append(dict(step=i + 1, rel_mse=rv, seconds=time.time() - t0))
            log(f'  bank[{dom}] step {i + 1}/{steps} rel-MSE {rv:.3e} [{time.time() - t0:.0f}s]')
            if not np.isfinite(rv):
                raise SystemExit(f'non-finite loss at step {i + 1}')

    def recon(pz_):
        p, z = pz_
        Gf = features(dom, p, coords)
        H = sc.head(p, z)
        per = []
        for s in range(0, S, 512):
            Uh = H[s:s + 512] @ Gf.T
            per.append(jnp.linalg.norm(Uh - U[s:s + 512], axis=1) / jnp.linalg.norm(U[s:s + 512], axis=1))
        per = jnp.concatenate(per)
        return float(jnp.mean(per)), float(jnp.max(per))
    rm, rx = recon(pz)
    em, ex = recon(ema)
    use_ema = em < rm
    params, Z = ema if use_ema else pz
    info = dict(steps=steps, seconds=time.time() - t0, used_ema=bool(use_ema), recon_mean=min(em, rm),
                recon_max=ex if use_ema else rx, n_states=int(S), n_points=int(n_pts), history=hist, seed=int(seed),
                full_points=int(len(full_idx)), time_marks=marks_t)
    return jax.tree_util.tree_map(np.asarray, params), np.asarray(Z), info


def rotation(dom, params, Z, Xq, wq, rank_tol=1e-10):
    """Ordering (DESIGN A1): W^1/2 G = Q R_G on the frozen reference rule (Xq, wq); c_i = h(z_i); rows R_G c_i/||.||;
    SVD -> V_s; T = R_G^-1 V_s (columns ordered by training energy). Returns T (R, R_eff), singular values, cond(R_G)."""
    p = jax.tree_util.tree_map(jnp.asarray, params)
    Gq = np.asarray(features(dom, p, jnp.asarray(Xq)))
    Qm, Rg = np.linalg.qr(Gq * np.sqrt(wq)[:, None])
    d = np.abs(np.diag(Rg))
    condR = float(d.max() / d.min())      # ratio of |diag R_G| extremes (not a condition number)
    C = np.asarray(sc.head(p, jnp.asarray(Z)))                  # (S, R)
    A = C @ Rg.T
    nrm = np.linalg.norm(A, axis=1)
    keep = nrm > 1e-14 * nrm.max()
    A = A[keep] / nrm[keep, None]
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    R_eff = int(np.sum(sv > rank_tol * sv[0]))
    T = np.linalg.solve(Rg, Vt.T[:, :R_eff])
    return T, sv, condR, int((~keep).sum())


def bank_space(dom, params, T, k_q, q, lattice_pts=None, ref_rule=None, box_only=False):
    """rom.Space of the rotated bank on the composite q x q Gauss rule with k_q elements per macro edge."""
    p = jax.tree_util.tree_map(jnp.asarray, params)
    Xq, wq = G.composite_rule(dom, k_q, q)
    Tj = jnp.asarray(T)
    v, gx, gy = [], [], []
    for s in range(0, len(Xq), 8192):
        a, b_, c = features_grad(dom, p, jnp.asarray(Xq[s:s + 8192]), box_only)
        v.append(np.asarray(a @ Tj))
        gx.append(np.asarray(b_ @ Tj))
        gy.append(np.asarray(c @ Tj))
    P, Gx, Gy = np.concatenate(v), np.concatenate(gx), np.concatenate(gy)
    M = (P * wq[:, None]).T @ P
    K = (Gx * wq[:, None]).T @ Gx + (Gy * wq[:, None]).T @ Gy
    pts = G.lattice(dom)[0] if lattice_pts is None else lattice_pts
    Pl = np.asarray(features(dom, p, jnp.asarray(pts), box_only) @ Tj)
    return RM.Space(P, Gx + Gy, wq, M, K, P, wq, Pl, name=f'bank-{dom}-k{k_q}-q{q}',
                    meta=dict(dom=dom, ic_points=Xq, k_q=k_q, q=q, M_dev=float(np.linalg.norm(M - np.eye(len(M))))))


def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
