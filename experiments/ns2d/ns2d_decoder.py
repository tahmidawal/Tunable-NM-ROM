"""ns2d_decoder.py -- the PERIODIC separable decoder for the torus (DESIGN.md "Bank, head").

    omega(x; z) = G(x)^T h(z),   G: R^2 -> R^R  (integer-Fourier-feature MLP, periodic
                                                  by construction, NO boundary mask),
                                 h: R^K -> R^R  (sep_common.head: MLP + linear skip).

Differences from sep_common.features, stated so nothing is inherited silently:
  * the Fourier feature matrix B is INTEGER-valued (frequencies k in [-KFF, KFF]^2, drawn
    once from the seed), so every feature is exactly 1-periodic on the unit torus;
  * there is no bc(x) polynomial (the torus has no wall);
  * every bank column is made MEAN-ZERO ON THE GRID, because the vorticity has zero mean
    and the Poisson solve needs it; this is a constant per column per grid.

Training: the sep_common.train_autodecoder recipe (joint Adam over g, h, and one free code
per snapshot; relative MSE + feature-Gram orthonormality regulariser), with an optional
mini-batch over snapshot ROWS (the separable form makes a batch cost S_b x n).  The training
array is a jit ARGUMENT, never a closure constant.
"""
from __future__ import annotations

import time

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                              # noqa: E402
import optax                                                         # noqa: E402

import sep_common as sc                                              # noqa: E402

F64 = jnp.float64


def init_periodic(key, k_lat, r_feat, n_ff=64, kff=6, g_hidden=128, g_layers=2,
                  h_hidden=128, h_layers=2, out_scale=1.0):
    kb, kg, kh, kl = jax.random.split(key, 4)
    B = jax.random.randint(kb, (2, n_ff), -kff, kff + 1).astype(F64)   # integer frequencies
    g_mlp = sc.init_mlp(kg, [2 * n_ff] + [g_hidden] * g_layers + [r_feat])
    h_mlp = sc.init_mlp(kh, [k_lat] + [h_hidden] * h_layers + [r_feat])
    h_lin = jax.random.normal(kl, (k_lat, r_feat), dtype=F64) * 0.3
    return dict(B=B, g=g_mlp, h=h_mlp, h_lin=h_lin,
                out_scale=jnp.asarray(float(out_scale), dtype=F64))


def features_raw(params, xy):
    """g~(x) without mean removal: (n_pts, R)."""
    ang = 2.0 * jnp.pi * (xy @ params['B'])
    ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return params['out_scale'] * sc.apply_mlp(params['g'], ff)


def bank_on_grid(params, N, chunk=16384):
    """The mean-zero bank on the N x N grid, (N*N, R), row order i*N + j (x-major)."""
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    xy = jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1))
    if xy.shape[0] <= chunk:
        G = features_raw(params, xy)
    else:
        G = jnp.concatenate([features_raw(params, xy[s:s + chunk])
                             for s in range(0, xy.shape[0], chunk)], axis=0)
    return G - jnp.mean(G, axis=0, keepdims=True)


head = sc.head


def train_autodecoder(key, N, U, k_lat, r_feat, steps=30000, lr=1e-3, lam_orth=1e-4,
                      batch=0, log_every=1000, tag='', **arch):
    """U: (S, N*N) f64 snapshots on the N-grid.  Returns (params, Z, info)."""
    U = jnp.asarray(U, dtype=F64)
    S = U.shape[0]
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    xy = jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1))
    key, kz, kp = jax.random.split(key, 3)
    u_rms = float(jnp.sqrt(jnp.mean(U * U)))
    params = init_periodic(kp, k_lat, r_feat, out_scale=u_rms, **arch)
    Z = 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64)
    u_ms = float(jnp.mean(U * U))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    state = opt.init((params, Z))
    nb = int(batch) if batch and batch < S else S

    def loss_fn(pz, Ub, idx):
        p, z = pz
        G = features_raw(p, xy)
        G = G - jnp.mean(G, axis=0, keepdims=True)
        H = sc.head(p, z[idx])
        err = H @ G.T - Ub
        rel = jnp.mean(err * err) / u_ms
        C = (G.T @ G) / (G.shape[0] * p['out_scale'] ** 2)
        orth = jnp.mean((C - jnp.eye(C.shape[0], dtype=F64)) ** 2)
        return rel + lam_orth * orth, rel

    @jax.jit
    def step(pz, st, Ub, idx):
        (val, rel), grads = jax.value_and_grad(loss_fn, has_aux=True)(pz, Ub, idx)
        grads[0]['out_scale'] = jnp.zeros_like(grads[0]['out_scale'])
        upd, st = opt.update(grads, st)
        return optax.apply_updates(pz, upd), st, val, rel

    rng = np.random.default_rng(0)
    pz = (params, Z)
    t0 = time.time()
    rel = jnp.inf
    all_idx = jnp.arange(S)
    for i in range(steps):
        if nb < S:
            idx = jnp.asarray(np.sort(rng.choice(S, nb, replace=False)))
            pz, state, val, rel = step(pz, state, U[idx], idx)
        else:
            pz, state, val, rel = step(pz, state, U, all_idx)
        if (i + 1) % log_every == 0 or i == 0:
            print(f'   train[{tag}] step {i+1:6d}/{steps}  rel-MSE {float(rel):.3e}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
    params, Z = pz
    G = bank_on_grid(params, N)
    per = []
    for s in range(0, S, 1024):
        Uh = sc.head(params, Z[s:s + 1024]) @ G.T
        per.append(np.asarray(jnp.linalg.norm(Uh - U[s:s + 1024], axis=1)
                              / jnp.linalg.norm(U[s:s + 1024], axis=1)))
    per = np.concatenate(per)
    info = dict(final_rel_mse=float(rel), steps=steps, lr=lr, lam_orth=lam_orth, batch=nb,
                seconds=time.time() - t0, recon_rel_l2_mean=float(per.mean()),
                recon_rel_l2_median=float(np.median(per)), recon_rel_l2_max=float(per.max()),
                n_snapshots=int(S), n_points=int(N * N), arch=arch, u_rms=u_rms)
    print(f'   train[{tag}] done: recon rel-L2 mean {info["recon_rel_l2_mean"]:.3e} '
          f'max {info["recon_rel_l2_max"]:.3e}  [{info["seconds"]:.0f}s]', flush=True)
    return params, np.asarray(Z), info


def make_lm_fit(head_fn, budget=300, gtol=1e-6):
    """Field-metric latent fit  min_z ||Rb (h(z) - c)||  by damped LM (arms.make_stationary_lm
    semantics: reasons 0 budget, 1 tolerance, 2 tiny step, 3 rejected, 4 stationary)."""
    def fun(z, c, Rb):
        return Rb @ (head_fn(z) - c)

    def lm(z0, c, Rb, tol=0.0):
        def evaluate(z):
            r = fun(z, c, Rb)
            J = jax.jacfwd(fun)(z, c, Rb)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        reason = jnp.where(jnp.isfinite(rn), jnp.where(grad(r, J) <= gtol, 4,
                                                       jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason = s
            H = J.T @ J
            g = J.T @ r
            dz = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz))
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, c, Rb))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4, jnp.where(rn2 <= tol, 1, jnp.where(
                tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason
    return lm


def oracle_fit(head_fn, Rb, C, Zcand, n_starts=8, budget=300, gtol=1e-6, chunk=32):
    """Best-found latent fit for every row of C (S, R) in the whitened metric Rb.  Starts:
    the n_starts training codes whose head output is nearest in the field metric.  Returns
    (Z (S,K), residual norms (S,), iterations, reasons) -- all in the field metric."""
    lm = make_lm_fit(head_fn, budget, gtol)
    Hc = jax.jit(jax.vmap(head_fn))(jnp.asarray(Zcand))
    Hrot = Hc @ Rb.T
    Hn = jnp.sum(Hrot * Hrot, 1)

    @jax.jit
    def fit(c):
        score = Hn - 2 * (Hrot @ (Rb @ c))
        starts = jnp.asarray(Zcand)[jnp.argsort(score)[:n_starts]]
        outs = jax.vmap(lambda z0: lm(z0, c, Rb, 0.))(starts)
        best = jnp.argmin(outs[1])
        return outs[0][best], outs[1][best], outs[2][best], outs[3][best]

    zs, rns, its, rs = [], [], [], []
    C = jnp.asarray(C)
    for s in range(0, C.shape[0], chunk):
        v = jax.tree_util.tree_map(np.asarray, jax.vmap(fit)(C[s:s + chunk]))
        zs.append(v[0]); rns.append(v[1]); its.append(v[2]); rs.append(v[3])
    return (np.concatenate(zs), np.concatenate(rns), np.concatenate(its), np.concatenate(rs))
