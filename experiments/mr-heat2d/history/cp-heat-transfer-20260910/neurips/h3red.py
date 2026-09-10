"""Heat-3D reduced-Gram ROM.  Separate file so h3opt.py -- which the 3.8x
headline rests on -- is not restructured.

What this changes and what it does not
--------------------------------------
NOTHING about the objective.  The empirical-quadrature support and weights are
used exactly as released: same `eq_indices`, same `eq_weights`, same 7-point
gather, same weighted least-squares problem, same Levenberg-style damping, same
4-point backtracking line search, same relative-gradient stopping test.  This is
the same iteration, formed differently.

What changes is how H_aug and g_aug are assembled.  `_solve_step_rt` calls
`jax.jacfwd(_f_norm)` every Gauss-Newton iteration, which pushes k tangents
through `h @ V_eq` and materialises an (n_eq x k) Jacobian -- at N=128 that is
40 tangents over 7*640 = 4480 gathered entries, about 92 Mflop per iteration.

But `_f_norm` is AFFINE in the rank-512 feature vector h(z):

    f(z) = h(z) @ Q(t) + q(t),   t = DT*kappa
    Q(t) = Q0 + t*Q1,  q(t) = q0 + t*q1

with Q0 the centre column of `mask_sp (*) V_eq` and Q1 its 7-point Laplacian
combination -- both (rank, n_eq) and kappa-free.  So every weighted contraction
the solve needs is a polynomial in t whose coefficients are

    M_ij = Q_i W Q_j^T   (rank x rank),  a_ij = Q_i W q_j  (rank),
    s_ij = q_i W q_j     (scalar)

i.e. three rank x rank matrices and a handful of vectors, 6.3 MB total, built
once offline in about a millisecond (n_eq <= 1280).  Per iteration the cost
becomes Gw @ Jh at 512x512x40 ~ 10.5 Mflop, with no (n_eq x k) Jacobian and no
4480-wide gather: about 10x fewer flops for the identical normal equations.

Precision
---------
The reduced algebra runs in f64 (X64=1).  In 2D the f32 version of exactly this
reformulation put J^T J off by 7% and J^T r by 51% and flipped Gauss-Newton
counts on 4 of 12 configurations (test_red2d.py) -- the same lesson as GRAM64.
x64 is a global flag, so this file casts every float array in ctx back to f32
after `h3opt.build` and h3opt's dtype-less array constructors are pinned to f32;
the X64=0 control run is what demonstrates the frozen paths did not move.

The FOM is untouched: h3opt.make_fom, same operator, CG tol 1e-6, 1000-iter cap,
50 steps, jitted with kappa as a runtime argument.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import jax

X64 = os.environ.get('X64', '0') == '1'
if X64:
    jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

import h3opt as H

F32 = jnp.float32
FRED = jnp.float64 if X64 else jnp.float32


def pin_ctx_f32(ctx):
    """Undo any x64 promotion in the setup arrays, so the dense control and the
    FOM see exactly the dtypes they see when X64=0."""
    n = 0
    for key, val in list(ctx.items()):
        if isinstance(val, jnp.ndarray) and val.dtype == jnp.float64:
            ctx[key] = val.astype(F32)
            n += 1

    # Belt and braces: a single f64 leak anywhere upstream of `encode` moves the
    # whole solve -- including the frozen dense control -- to f64 without saying
    # so.  One did (jnp.linspace in the initial-condition builder).  Force f32 at
    # the two entry points rather than trusting that it was the only one.
    def _f32(x):
        return jax.tree_util.tree_map(
            lambda a: a.astype(F32) if getattr(a, 'dtype', None) == jnp.float64
            else a, x)

    _enc, _ic = ctx['encode'], ctx.get('make_gaussian_ic')
    ctx['encode'] = lambda u: _f32(_enc(_f32(u)))
    if _ic is not None:
        ctx['make_gaussian_ic'] = lambda *a, **kw: _f32(_ic(*a, **kw))
    print(f'[pin] cast {n} ctx arrays back to f32; encode/IC wrapped', flush=True)
    return ctx


def build_reduced3(ctx):
    """OFFLINE, once per cell.  Returns the kappa-free contractions."""
    cfg = ctx['cfg']
    n_eq, dx = ctx['n_eq'], ctx['dx']
    V_eq, b_sparse = ctx['V_eq'], ctx['b_sparse']
    mask_sp, ug_sp = ctx['mask_sp'], ctx['ug_sp']
    w = ctx['eq_w_jnp'].astype(FRED)
    t0 = time.perf_counter()

    # u_sp(z) = h @ P + c, reshaped to (n_eq, 7)
    P = (V_eq * mask_sp[None, :]).astype(FRED)
    c = (mask_sp * b_sparse + ug_sp).astype(FRED)
    rank = P.shape[0]
    P3 = P.reshape(rank, n_eq, 7)
    c3 = c.reshape(n_eq, 7)

    # f = u_sp[:,0] + DT*kappa*lap  ->  Q0 = centre, Q1 = Laplacian combination
    Q0 = P3[:, :, 0]
    Q1 = (6.0 * P3[:, :, 0] - P3[:, :, 1] - P3[:, :, 2] - P3[:, :, 3]
          - P3[:, :, 4] - P3[:, :, 5] - P3[:, :, 6]) / dx ** 2
    q0 = c3[:, 0]
    q1 = (6.0 * c3[:, 0] - c3[:, 1] - c3[:, 2] - c3[:, 3]
          - c3[:, 4] - c3[:, 5] - c3[:, 6]) / dx ** 2

    Q0w, Q1w = Q0 * w[None, :], Q1 * w[None, :]
    R = dict(
        M00=Q0w @ Q0.T, M01=Q0w @ Q1.T, M11=Q1w @ Q1.T,
        a00=Q0w @ q0, a01=Q0w @ q1, a10=Q1w @ q0, a11=Q1w @ q1,
        s00=float(q0 @ (w * q0)), s01=float(q0 @ (w * q1)),
        s11=float(q1 @ (w * q1)),
        Q0=Q0, Q1=Q1, q0=q0, q1=q1, w=w, rank=rank, n_eq=n_eq)
    for key in ('M00', 'M01', 'M11', 'a00', 'a01', 'a10', 'a11'):
        jax.block_until_ready(R[key])
    R['build_seconds'] = time.perf_counter() - t0
    R['offline_bytes'] = int(sum(R[key].size * R[key].dtype.itemsize
                                 for key in ('M00', 'M01', 'M11', 'a00', 'a01',
                                             'a10', 'a11')))
    print(f'[red3] rank={rank} n_eq={n_eq} offline build '
          f'{R["build_seconds"]:.3f}s  kappa-free {R["offline_bytes"]/1e6:.2f} MB'
          f'  dtype={Q0.dtype}', flush=True)
    return R


def make_rom_reduced(ctx, R, masked_iters=False, chol=True):
    """Same iteration as h3opt.make_rom_base, assembled from R."""
    cfg = ctx['cfg']
    k_dim = cfg['k_dim']
    IT = cfg['max_iters']
    REL_TOL = cfg['gn_tol']
    W1k, b1k, W2k, b2k, Wrk, brk, Wdk, bdk = ctx['mlp_w']
    EYE = jnp.eye(k_dim + 1, dtype=FRED)
    M00, M01, M11 = R['M00'], R['M01'], R['M11']
    a00, a01, a10, a11 = R['a00'], R['a01'], R['a10'], R['a11']
    s00, s01, s11 = R['s00'], R['s01'], R['s11']
    Q0, Q1, q0, q1, w = R['Q0'], R['Q1'], R['q0'], R['q1'], R['w']

    def _h(z):
        h_nl = jax.nn.swish(z @ W1k + b1k)
        h_nl = jax.nn.swish(h_nl @ W2k + b2k)
        h_nl = h_nl @ Wrk + brk
        return z @ Wdk + bdk + h_nl

    def _h_and_J(z):
        """h and dh/dz in closed form -- see h2opt.make_analytic_features."""
        x1 = z @ W1k + b1k
        g1 = jax.nn.sigmoid(x1)
        a1 = x1 * g1
        d1 = g1 * (1.0 + x1 * (1.0 - g1))
        x2 = a1 @ W2k + b2k
        g2 = jax.nn.sigmoid(x2)
        a2 = x2 * g2
        d2 = g2 * (1.0 + x2 * (1.0 - g2))
        h = z @ Wdk + bdk + a2 @ Wrk + brk
        Mj = d1[:, None] * W1k.T
        Mj = d2[:, None] * (W2k.T @ Mj)
        return h, Wdk.T + Wrk.T @ Mj

    def _solve_step(z_init, s_init, u_prev_eq, Gw, wq, gg, t):
        # projections of u_prev; n_eq <= 1280 so these are two small matvecs
        wu = w * u_prev_eq
        Wup = Q0 @ wu + t * (Q1 @ wu)
        gup = (q0 @ wu) + t * (q1 @ wu)
        upWup = u_prev_eq @ wu

        def _parts(z, s):
            h = _h(z).astype(FRED)
            Gwh = Gw @ h
            fnWfn = h @ Gwh + 2.0 * (h @ wq) + gg
            term = h @ Wup + gup
            return s * s * fnWfn - 2.0 * s * term + upWup    # R^T W R

        def _grad(z, s):
            h32, Jh32 = _h_and_J(z)
            h, Jh = h32.astype(FRED), Jh32.astype(FRED)
            Gwh = Gw @ h
            Qwfn = Gwh + wq                                  # Q W fn
            fnWfn = h @ Gwh + 2.0 * (h @ wq) + gg
            JtWJ = Jh.T @ (Gw @ Jh)
            JtWfn = Jh.T @ Qwfn
            JtWR = Jh.T @ (s * Qwfn - Wup)
            fnWR = s * fnWfn - (h @ Wup + gup)
            f0v = s * s * fnWfn - 2.0 * s * (h @ Wup + gup) + upWup
            return JtWJ, JtWfn, fnWfn, JtWR, fnWR, f0v

        # gnorm0, for the relative stopping test
        _, _, _, JtWR0, fnWR0, _ = _grad(z_init.astype(F32),
                                         s_init.astype(FRED))
        gz0 = s_init.astype(FRED) * JtWR0
        gnorm0 = jnp.maximum(jnp.sqrt(gz0 @ gz0 + fnWR0 ** 2), 1e-30)

        def _body(carry):
            z, s, _, itr = carry
            zf, sf = z.astype(F32), s.astype(FRED)
            JtWJ, JtWfn, fnWfn, JtWR, fnWR, f0v = _grad(zf, sf)
            H_aug = jnp.block([[sf ** 2 * JtWJ, (sf * JtWfn)[:, None]],
                               [(sf * JtWfn)[None, :], fnWfn[None, None]]])
            g_aug = jnp.append(sf * JtWR, fnWR)
            lam = jnp.maximum(1e-3 * jnp.trace(H_aug) / (k_dim + 1), 1e-8)
            _A = H_aug + lam * EYE
            if chol:
                _L = jnp.linalg.cholesky(_A)
                _y = jax.scipy.linalg.solve_triangular(_L, -g_aug, lower=True)
                delta = jax.scipy.linalg.solve_triangular(_L.T, _y, lower=False)
            else:
                delta = jnp.linalg.solve(_A, -g_aug)
            dz, ds = delta[:k_dim], delta[k_dim]

            def _f(a):
                return _parts((zf + a * dz.astype(F32)), sf + a * ds)

            f1, f2, f3, f4 = _f(1.), _f(.5), _f(.25), _f(.125)
            step = jnp.where(f1 < f0v, 1., jnp.where(
                f2 < f0v, .5, jnp.where(f3 < f0v, .25,
                                        jnp.where(f4 < f0v, .125, 0.))))
            gz = sf * JtWR
            gnorm = jnp.sqrt(gz @ gz + fnWR ** 2)
            # the iterate itself stays f32, as in the released solver; only the
            # normal equations are assembled in f64
            return (z + (step * dz).astype(F32),
                    jnp.maximum(s + (step * ds).astype(F32), H.S_MIN),
                    gnorm, itr + 1)

        def _cond(carry):
            _, _, gnorm, itr = carry
            return (gnorm > REL_TOL * gnorm0) & (itr < IT)

        _init = (z_init, s_init, jnp.array(jnp.inf, dtype=FRED),
                 jnp.array(0, jnp.int32))
        if masked_iters:
            def _masked(_i, cr):
                z, s, gn, itr = cr
                done = ~_cond(cr)
                zn, sn, gn2, it2 = _body(cr)
                return (jnp.where(done, z, zn), jnp.where(done, s, sn),
                        jnp.where(done, gn, gn2), jnp.where(done, itr, it2))
            z_f, s_f, gnorm_f, itr_f = jax.lax.fori_loop(0, IT, _masked, _init)
        else:
            z_f, s_f, gnorm_f, itr_f = jax.lax.while_loop(_cond, _body, _init)

        res_norm = jnp.sqrt(jnp.maximum(_parts(z_f.astype(F32),
                                               s_f.astype(FRED)), 0.0))
        h_f = _h(z_f.astype(F32)).astype(FRED)
        u_eq_new = s_f.astype(FRED) * (h_f @ Q0 + q0)
        return z_f, s_f, gnorm_f, itr_f, res_norm, u_eq_new

    @jax.jit
    def rollout(z0, s0, u_cur_eq0, kappa_f32):
        t = (H.DT * kappa_f32).astype(FRED)
        Gw = M00 + t * (M01 + M01.T) + t * t * M11
        wq = a00 + t * (a01 + a10) + t * t * a11
        gg = s00 + 2.0 * t * s01 + t * t * s11
        iters_arr = jnp.zeros(H.NUM_STEPS, dtype=jnp.int32)
        res_arr = jnp.zeros(H.NUM_STEPS, dtype=FRED)

        def _step(i, carry):
            z_, s_, u_eq_, iters_, res_ = carry
            z_n, s_n, _, n_it, rn, u_eq_n = _solve_step(z_, s_, u_eq_,
                                                        Gw, wq, gg, t)
            return (z_n, s_n, u_eq_n, iters_.at[i].set(n_it),
                    res_.at[i].set(rn))

        return jax.lax.fori_loop(
            0, H.NUM_STEPS, _step,
            (z0.astype(F32), s0.astype(F32), u_cur_eq0.astype(FRED),
             iters_arr, res_arr))

    return rollout


# ---------------------------------------------------------------------------
# Register the reduced variants into h3opt's own registry and let h3opt.main
# drive the measurement.  Duplicating that loop would risk measuring these
# variants under a different protocol than the 3.8x headline was measured under,
# which is the one thing that must not happen.
# ---------------------------------------------------------------------------
_GRAM_CACHE = {}


def _grams(ctx):
    """Built once per cell.  The Grams depend only on the EQ support/weights and
    the decoder, never on max_iters or gn_tol, so knob variants share them."""
    key = ctx['cfg']['ckpt'], int(ctx['n_eq'])
    if key not in _GRAM_CACHE:
        _GRAM_CACHE[key] = build_reduced3(ctx)
    return _GRAM_CACHE[key]


def _register():
    if not X64:
        return
    H.VARIANTS['red'] = lambda ctx: make_rom_reduced(
        ctx, _grams(ctx), masked_iters=False, chol=True)
    H.VARIANTS['red_m'] = lambda ctx: make_rom_reduced(
        ctx, _grams(ctx), masked_iters=True, chol=True)
    H.VARIANTS['red_lu'] = lambda ctx: make_rom_reduced(
        ctx, _grams(ctx), masked_iters=False, chol=False)

    _build = H.build

    def build_pinned(cell, eq_cache_dir):
        return pin_ctx_f32(_build(cell, eq_cache_dir))

    H.build = build_pinned


def main():
    if not X64:
        raise SystemExit('FATAL: h3red needs X64=1 (f32 reduced Grams put '
                         'J^T J off by 7% in the 2D test); use h3opt.py '
                         'directly for the X64=0 control')
    _register()
    H.main()


if __name__ == '__main__':
    main()
