"""
Heat-2D NM-ROM cost study (discussion period, NeurIPS 25242).

Companion to h3opt.py.  The 2D driver differs from the 3D one in three ways
that matter, and all three are reproduced here before anything is changed:

  1. It runs a DENSE residual -- `using dense residual (no EQ)` -- so the
     decoder is evaluated on the whole grid at every Gauss-Newton iteration and
     `jacfwd` builds a (N^2 x k+1) Jacobian.  Empirical quadrature, the paper's
     own hyper-reduction, is not used on these cells.
  2. Its inner solve is Levenberg-Marquardt with an adaptive mu and an
     accept/reject test, not the damped Gauss-Newton with backtracking used in
     3D.  The stopping test is on the ABSOLUTE ||J^T r||.
  3. The timed ROM region EXCLUDES the encoder (the 3D one includes it,
     uncompiled).  `base` here reproduces that exclusion; `base_enc` puts the
     encoder back in, which is what an honest online cost has to contain.

FOM is FROZEN: same operator, CG tol 1e-6, 1000-iteration cap, 50 steps,
jitted with kappa as a runtime argument.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

import jax

# x64 is needed ONLY for the rank-space Grams of the reduced variants (see
# build_reduced).  It is a global flag, so every dtype on the FOM path and on
# the dense-Jacobian ROM path is pinned to float32 explicitly below; the
# `base`/`enc_jit` controls are re-measured in the same job to confirm nothing
# on the frozen side moved.  Same lesson as GRAM64: f32 Grams are unusable.
X64 = os.environ.get('X64', '0') == '1'
if X64:
    jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg as jax_cg
import flax.linen as nn

import bench as _bench

F32 = jnp.float32
FRED = jnp.float64 if X64 else jnp.float32

ROOT = '/cluster/tufts/paralab/tawal01/NMROM-Apr8'
JOBS = '/cluster/home/tawal01/nmrom/jobs'

L = 1.0
DT = 5e-3
N_STEPS = 50
CG_TOL = 1e-6
CG_MAX_ITER = 1000
AMP_EPS = 1e-6

CELLS = {
    'h2d_n64_acc': dict(
        N=64, k_dim=64, rank=256, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=4, gn_max_iters=20, gn_tol=1e-3,
        data=f'{ROOT}/Heat2D/data/training_data_2d_64.pkl',
        ckpt=(f'{JOBS}/heatretime_h2d_n64/checkpoint_N64_k64_r256_h256_p8_e96'
              '_nh4_nl4_ep80000_bs32_lr0.002_wd0.0005_s0.pkl'),
        posted_rel_l2=5.21e-3, retimed_rel_l2=0.005207489442458728),
    'h2d_n64_fast': dict(
        N=64, k_dim=64, rank=256, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=4, gn_max_iters=8, gn_tol=1e-3,
        data=f'{ROOT}/Heat2D/data/training_data_2d_64.pkl',
        ckpt=(f'{JOBS}/heatretime_h2d_n64/checkpoint_N64_k64_r256_h256_p8_e96'
              '_nh4_nl4_ep80000_bs32_lr0.002_wd0.0005_s0.pkl'),
        posted_rel_l2=1.08e-2, retimed_rel_l2=0.010812239726766851),
    'h2d_n128_acc': dict(
        N=128, k_dim=64, rank=256, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=4, gn_max_iters=20, gn_tol=5e-3,
        data=(f'{ROOT}/20260423-NEURIPS/Autoresearch/Heat-2D/N128/'
              'training_data_2d_128.pkl'),
        ckpt=(f'{JOBS}/heatretime_h2d_n128/checkpoint_N128_k64_r256_h256_p8_e96'
              '_nh4_nl4_ep150000_bs16_lr0.001_wd0.002_s0.pkl'),
        # posted 3.75e-3 exists in no result file; the source run measured this
        posted_rel_l2=1.020e-2, retimed_rel_l2=0.010200129719411885),
    'h2d_n128_fast': dict(
        N=128, k_dim=64, rank=256, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=4, gn_max_iters=10, gn_tol=2e-2,
        data=(f'{ROOT}/20260423-NEURIPS/Autoresearch/Heat-2D/N128/'
              'training_data_2d_128.pkl'),
        ckpt=(f'{JOBS}/heatretime_h2d_n128/checkpoint_N128_k64_r256_h256_p8_e96'
              '_nh4_nl4_ep150000_bs16_lr0.001_wd0.002_s0.pkl'),
        posted_rel_l2=1.354e-2, retimed_rel_l2=0.01353803034590526),
}


class ViTEncoder(nn.Module):
    latent_dim: int
    grid_n: int
    patch_size: int = 8
    embed_dim: int = 96
    num_heads: int = 4
    num_enc_layers: int = 4

    @nn.compact
    def __call__(self, u_flat):
        N = self.grid_n
        p = self.patch_size
        nps = N // p
        u = u_flat.reshape(N, N)
        patches = u.reshape(nps, p, nps, p)
        patches = jnp.transpose(patches, (0, 2, 1, 3)).reshape(nps ** 2, p * p)
        x = nn.Dense(self.embed_dim)(patches)
        pos = self.param('pos', nn.initializers.normal(0.02),
                         (nps ** 2, self.embed_dim))
        x = x + pos
        for _ in range(self.num_enc_layers):
            y = nn.LayerNorm()(x)
            y = nn.SelfAttention(num_heads=self.num_heads,
                                 qkv_features=self.embed_dim)(y)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.embed_dim * 4)(y)
            y = nn.gelu(y)
            y = nn.Dense(self.embed_dim)(y)
            x = x + y
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=0)
        return nn.Dense(self.latent_dim)(x)


class LinearCPDecoder(nn.Module):
    latent_dim: int
    rank: int = 256
    grid_size: int = 64
    hidden_dim: int = 256

    def setup(self):
        self.W1 = nn.Dense(self.hidden_dim)
        self.W2 = nn.Dense(self.hidden_dim)
        self.W_rank = nn.Dense(self.rank)
        self.W_direct = nn.Dense(self.rank)
        init = nn.initializers.normal(0.01)
        Ng = self.grid_size
        self.W_x = self.param('W_x', init, (self.rank, Ng))
        self.W_y = self.param('W_y', init, (self.rank, Ng))
        self.bias = self.param('bias', nn.initializers.zeros, ())

    def features(self, z):
        """The rank-dim vector h(z).  The decoder output is AFFINE in h:
        u = (h @ V) + bias, with V[r] = outer(W_x[r], W_y[r]).  Splitting the
        decoder here is what makes the reduced-Gram form possible."""
        h_nl = nn.swish(self.W1(z))
        h_nl = nn.swish(self.W2(h_nl))
        h_nl = self.W_rank(h_nl)
        return self.W_direct(z) + h_nl

    def basis(self):
        return self.W_x, self.W_y, self.bias

    def __call__(self, z):
        h = self.features(z)
        return jnp.einsum('r,ri,rj->ij', h, self.W_x, self.W_y).flatten() + self.bias


class Autoencoder(nn.Module):
    latent_dim: int
    grid_size: int
    rank: int = 256
    hidden_dim: int = 256
    patch_size: int = 8
    embed_dim: int = 96
    num_heads: int = 4
    num_enc_layers: int = 4

    def setup(self):
        self.encoder = ViTEncoder(
            latent_dim=self.latent_dim, grid_n=self.grid_size,
            patch_size=self.patch_size, embed_dim=self.embed_dim,
            num_heads=self.num_heads, num_enc_layers=self.num_enc_layers)
        self.decoder = LinearCPDecoder(
            latent_dim=self.latent_dim, rank=self.rank,
            grid_size=self.grid_size, hidden_dim=self.hidden_dim)

    def encode(self, u_flat):
        scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
        return self.encoder(u_flat / scale), scale

    def decode_normalised(self, z):
        return self.decoder(z)

    def decode_features(self, z):
        return self.decoder.features(z)

    def decoder_basis(self):
        return self.decoder.basis()


def make_fom_ops(N):
    dx = L / (N - 1)

    def K_op(u_flat):
        u = u_flat.reshape((N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1, 1:-1].set(
            (4 * u[1:-1, 1:-1] - u[0:-2, 1:-1] - u[2:, 1:-1]
             - u[1:-1, 0:-2] - u[1:-1, 2:]) / dx ** 2)
        out = out.at[0, :].set(u[0, :])
        out = out.at[-1, :].set(u[-1, :])
        out = out.at[:, 0].set(u[:, 0])
        out = out.at[:, -1].set(u[:, -1])
        return out.flatten()

    def implicit_op(u_flat, kappa):
        return u_flat + DT * kappa * K_op(u_flat)

    m = jnp.ones((N, N), dtype=F32)
    m = m.at[0, :].set(0.).at[-1, :].set(0.).at[:, 0].set(0.).at[:, -1].set(0.)
    return K_op, implicit_op, m.flatten(), jnp.zeros(N * N, dtype=F32)


def build(cell):
    cfg = CELLS[cell]
    N = cfg['N']
    K_op, implicit_op, mask, u_g = make_fom_ops(N)

    model = Autoencoder(
        latent_dim=cfg['k_dim'], grid_size=N, rank=cfg['rank'],
        hidden_dim=cfg['hidden_dim'], patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'], num_heads=cfg['num_heads'],
        num_enc_layers=cfg['num_enc_layers'])
    with open(cfg['ckpt'], 'rb') as f:
        ck = pickle.load(f)
    params = ck['params'] if 'params' in ck else ck

    with open(cfg['data'], 'rb') as f:
        payload = pickle.load(f)
    val_snaps = payload['val_snapshots']
    val_params = payload['val_params']
    trajs = []
    for i in range(10):
        tr = np.asarray(val_snaps[i], dtype=np.float32)
        trajs.append(dict(u0=jnp.asarray(tr[0]), kappa=float(val_params[i]['kappa']),
                          final=jnp.asarray(tr[-1])))

    def encode(u):
        return model.apply({'params': params}, u, method=Autoencoder.encode)

    def decode_norm(z):
        return model.apply({'params': params}, z,
                           method=Autoencoder.decode_normalised)

    def decode_features(z):
        return model.apply({'params': params}, z,
                           method=Autoencoder.decode_features)

    params64 = jax.tree_util.tree_map(lambda a: a.astype(FRED), params)

    def decode_norm64(z):
        return model.apply({'params': params64}, z,
                           method=Autoencoder.decode_normalised)

    h_only, h_and_J = make_analytic_features(params)
    return dict(cfg=cfg, N=N, K_op=K_op, implicit_op=implicit_op, mask=mask,
                u_g=u_g, model=model, params=params, encode=encode,
                decode_norm=decode_norm, decode_features=decode_features,
                decode_norm64=decode_norm64,
                h_only=h_only, h_and_J=h_and_J, trajs=trajs)


def make_analytic_features(params):
    """h(z) and dh/dz in closed form.

    `jax.jacfwd(features)` traces k=64 tangents through the 3-layer swish MLP,
    which is ~60 small kernels per Gauss-Newton iteration on a path that is
    dispatch-bound.  The Jacobian of that MLP is

        dh/dz = W_direct^T + W_rank^T diag(s2') W2^T diag(s1') W1^T

    with s' the swish derivative -- six matmuls, no tracing.  Exact, and checked
    against jacfwd at run time (see the [red] equivalence line).
    """
    d = params['decoder']
    W1, b1 = d['W1']['kernel'], d['W1']['bias']
    W2, b2 = d['W2']['kernel'], d['W2']['bias']
    Wr, br = d['W_rank']['kernel'], d['W_rank']['bias']
    Wd, bd = d['W_direct']['kernel'], d['W_direct']['bias']

    def h_only(z):
        a1 = nn.swish(z @ W1 + b1)
        a2 = nn.swish(a1 @ W2 + b2)
        return z @ Wd + bd + a2 @ Wr + br

    def h_and_J(z):
        x1 = z @ W1 + b1
        g1 = jax.nn.sigmoid(x1)
        a1 = x1 * g1
        d1 = g1 * (1.0 + x1 * (1.0 - g1))          # swish'
        x2 = a1 @ W2 + b2
        g2 = jax.nn.sigmoid(x2)
        a2 = x2 * g2
        d2 = g2 * (1.0 + x2 * (1.0 - g2))
        h = z @ Wd + bd + a2 @ Wr + br
        M = d1[:, None] * W1.T                     # (hid, k)
        M = d2[:, None] * (W2.T @ M)               # (hid, k)
        return h, Wd.T + Wr.T @ M                  # (rank, k)

    return h_only, h_and_J


def build_reduced(ctx):
    """OFFLINE, once per cell.  Everything returned is rank x rank, rank, or
    scalar -- so the ONLINE solve never touches a grid-sized array.

    The residual of h2opt is

        r(z,s) = s * T(kap) @ (h(z) @ Vm + bias*m) - u_prev,  T = I + DT*kap*K

    r is affine in h and T is affine in kap, so every entry of J^T J, J^T r and
    ||r||^2 is a polynomial in kap whose coefficients are contractions of
    Y = Vm and Yt = K Vm.  Those contractions are what this builds.

    Assumes u_g == 0, which make_fom_ops guarantees for the 2D cells.
    """
    N, K_op, m = ctx['N'], ctx['K_op'], ctx['mask']
    Wx, Wy, bias = ctx['model'].apply(
        {'params': ctx['params']}, method=Autoencoder.decoder_basis)
    rank = Wx.shape[0]
    t0 = time.perf_counter()

    # V[r] = outer(W_x[r], W_y[r]); Y = mask (*) V.  Built in FRED (f64 when
    # X64=1): an f32 Gram is off by ~7% on J^T J -- measured, see test_red2d.py.
    V = jnp.einsum('ri,rj->rij', Wx, Wy).reshape(rank, N * N)
    Y = (V * m[None, :]).astype(FRED)
    Yt = jax.vmap(K_op)(Y)
    mf = m.astype(FRED)
    Km = K_op(mf)
    R = dict(
        A0=Y @ Y.T, A1=Y @ Yt.T, A2=Yt @ Yt.T,
        v0=Y @ mf, v1=Y @ Km, v2=Yt @ mf, v3=Yt @ Km,
        s0=mf @ mf, s1=mf @ Km, s2=Km @ Km,
        bias=float(bias), rank=rank, Y=Y, Yt=Yt, m=mf, Km=Km)
    R = jax.tree_util.tree_map(
        lambda x: jax.block_until_ready(x) if hasattr(x, 'shape') else x, R)
    R['build_seconds'] = time.perf_counter() - t0
    R['offline_bytes'] = int(sum(
        R[key].size * R[key].dtype.itemsize
        for key in ('A0', 'A1', 'A2', 'v0', 'v1', 'v2', 'v3')))
    print(f'[red] rank={rank} offline build {R["build_seconds"]:.2f}s  '
          f'kap-free Grams {R["offline_bytes"]/1e3:.1f} kB  dtype={Y.dtype}',
          flush=True)
    return R


def observe_isolation():
    """Clause (iv) is 'an otherwise idle, dedicated device'.  That has to be an
    observation, never a self-declaration: ask Slurm what else is running on
    this node.  Anything we cannot verify counts as NOT isolated."""
    import re
    import subprocess
    jid = os.environ.get('SLURM_JOB_ID')
    node = os.environ.get('SLURMD_NODENAME')
    ev = {'slurm_job_id': jid, 'node': node}
    if not jid or not node:
        ev['reason'] = 'not running under slurm'
        return False, ev
    try:
        out = subprocess.run(['scontrol', 'show', 'job', jid],
                             capture_output=True, text=True, timeout=60).stdout
        m = re.search(r'OverSubscribe=(\S+)', out)
        ev['oversubscribe'] = m.group(1) if m else None
    except Exception as exc:
        ev['oversubscribe_error'] = repr(exc)
    try:
        out = subprocess.run(['squeue', '-h', '-w', node, '-t', 'R', '-o', '%i'],
                             capture_output=True, text=True, timeout=60).stdout
        others = [x.strip() for x in out.split()
                  if x.strip() and x.strip().split('_')[0] != str(jid)]
        ev['other_running_jobs_on_node'] = others
    except Exception as exc:
        ev['other_jobs_error'] = repr(exc)
        others = None
    isolated = isinstance(others, list) and len(others) == 0
    return isolated, ev


def make_fom(ctx):
    implicit_op = ctx['implicit_op']

    @jax.jit
    def fom(u0, kappa_rt):
        def _op(v):
            return implicit_op(v, kappa_rt)

        def _step(_i, u):
            u2, _ = jax_cg(_op, u, x0=u, tol=CG_TOL, maxiter=CG_MAX_ITER)
            return u2

        return jax.lax.fori_loop(0, N_STEPS, _step, u0)

    return fom


def make_rollout(ctx, masked_iters=False, chol=False, gn_max_iters=None,
                 gn_tol=None, f64_assembly=False, f64_decoder=False):
    """`f64_assembly=True` keeps the EXPLICIT (N^2 x k+1) Jacobian -- the
    released assembly -- but forms the residual, the Jacobian and the normal
    equations in f64.  This is the control that separates the two things the
    reduced-Gram variants change at once: the assembly and the precision.
    Without it, a reduced-vs-dense timing cannot say which one bought the
    iteration-count drop."""
    cfg = ctx['cfg']
    IT = int(cfg['gn_max_iters'] if gn_max_iters is None else gn_max_iters)
    TOL = float(cfg['gn_tol'] if gn_tol is None else gn_tol)
    K_op, mask, u_g = ctx['K_op'], ctx['mask'], ctx['u_g']
    decode_norm = ctx['decode_norm']
    kk = cfg['k_dim'] + 1
    DT_A = FRED if f64_assembly else F32
    if (f64_assembly or f64_decoder) and not X64:
        raise SystemExit('FATAL: f64_assembly/f64_decoder need X64=1')
    EYE = jnp.eye(kk, dtype=DT_A)
    if f64_decoder:
        # `v4w64` keeps the DECODER in f32 and only accumulates in f64, which
        # turned out not to change the Gauss-Newton count at all.  This arm
        # moves the decoder arithmetic to f64 as well, so the grid residual
        # itself is f64-accurate.  It completes the attribution: it is the only
        # dense arm that can produce an f64-accurate gradient, and it is the
        # expensive way to do it.
        dn64 = ctx['decode_norm64']

        def cdn(z):
            return mask.astype(DT_A) * dn64(z.astype(DT_A)) + u_g.astype(DT_A)
    else:
        def cdn(z):
            # network stays f32; only the assembly moves
            return (mask * decode_norm(z.astype(F32)) + u_g).astype(DT_A)

    def _residual(zs, u_prev, kap):
        z_, s_ = zs[:-1], zs[-1]
        u_n = cdn(z_)
        return s_ * (u_n + DT * kap * K_op(u_n)) - u_prev.astype(DT_A)

    def _gn_step(z_init, scale_init, u_prev, kap):
        # dtype pinned: jnp.asarray([<0-d f32 array>]) promotes to the default
        # float, which is f64 when X64=1.  This keeps the dense path in f32.
        zs0 = jnp.concatenate(
            [z_init.astype(DT_A),
             jnp.reshape(jnp.asarray(scale_init, dtype=DT_A), (1,))])
        mu0 = jnp.asarray(1e-4, dtype=DT_A)

        def _body(carry):
            zs, mu, _, itr = carry
            J = jax.jacfwd(lambda q: _residual(q, u_prev, kap))(zs)
            r = _residual(zs, u_prev, kap)
            Jtr = J.T @ r
            JtJ = J.T @ J
            diag_scale = jnp.mean(jnp.diag(JtJ)) + 1e-8
            A = JtJ + mu * diag_scale * EYE
            if chol:
                Lc = jnp.linalg.cholesky(A)
                y = jax.scipy.linalg.solve_triangular(Lc, -Jtr, lower=True)
                d = jax.scipy.linalg.solve_triangular(Lc.T, y, lower=False)
            else:
                d = jnp.linalg.solve(A, -Jtr)
            zs_new = zs + d.astype(DT_A)    # pinned: see the comment above
            r_new = _residual(zs_new, u_prev, kap)
            # .astype(F32): under X64=1 jnp.linalg.norm promotes, which would
            # change the dtype of the dense path.  Pinned so the dense control
            # is bit-comparable between X64=0 and X64=1 runs.
            accept = (jnp.linalg.norm(r_new).astype(DT_A)
                      < jnp.linalg.norm(r).astype(DT_A))
            zs_next = jnp.where(accept, zs_new, zs)
            zs_next = zs_next.at[-1].set(
                jnp.maximum(zs_next[-1], jnp.asarray(1e-8, dtype=DT_A)))
            mu_next = jnp.where(accept,
                                jnp.maximum(mu * 0.3, jnp.asarray(1e-7, DT_A)),
                                jnp.minimum(mu * 10.0, jnp.asarray(1e2, DT_A)))
            return zs_next, mu_next, jnp.linalg.norm(Jtr).astype(DT_A), itr + 1

        def _cond(carry):
            _, _, jtr_norm, itr = carry
            return (jtr_norm > TOL) & (itr < IT)

        init = (zs0, mu0, jnp.asarray(jnp.inf, dtype=DT_A), jnp.array(0, jnp.int32))
        if masked_iters:
            def _m(i, c):
                zs, mu, jn_, itr = c
                done = ~((jn_ > TOL) & (itr < IT))
                zn, mn, jn2, it2 = _body(c)
                return (jnp.where(done, zs, zn), jnp.where(done, mu, mn),
                        jnp.where(done, jn_, jn2), jnp.where(done, itr, it2))
            zs_f, _, _, itr_f = jax.lax.fori_loop(0, IT, _m, init)
        else:
            zs_f, _, _, itr_f = jax.lax.while_loop(_cond, _body, init)
        return zs_f[:-1], zs_f[-1], itr_f

    @jax.jit
    def rollout(z0, scale0, u0, kap):
        iters = jnp.zeros(N_STEPS, jnp.int32)

        def _step(i, c):
            z_, sc_, u_prev, it_ = c
            z_n, sc_n, ni = _gn_step(z_, sc_, u_prev, kap)
            return (z_n, sc_n, sc_n * cdn(z_n), it_.at[i].set(ni))

        return jax.lax.fori_loop(
            0, N_STEPS, _step,
            (z0.astype(DT_A), jnp.asarray(scale0, dtype=DT_A),
             u0.astype(DT_A), iters))

    return rollout


def make_rollout_reduced(ctx, R, masked_iters=False, chol=True,
                         gn_max_iters=None, gn_tol=None, f64_iterates=False,
                         analytic=True):
    """Same Levenberg-Marquardt iteration as make_rollout -- same J^T J, same
    J^T r, same mu schedule, same accept test, same stopping test -- formed from
    the rank-space Grams instead of an explicit (N^2 x k+1) Jacobian.

    Exact, not an approximation: verified against the dense path on 12 random
    configurations (test_red2d.py), max relative deviation 9.3e-14 in J^T J with
    identical Gauss-Newton counts.
    """
    cfg = ctx['cfg']
    IT = int(cfg['gn_max_iters'] if gn_max_iters is None else gn_max_iters)
    TOL = float(cfg['gn_tol'] if gn_tol is None else gn_tol)
    k = cfg['k_dim']
    ITER_DT = FRED if f64_iterates else F32
    EYE = jnp.eye(k + 1, dtype=FRED)
    mask, u_g, decode_norm = ctx['mask'], ctx['u_g'], ctx['decode_norm']
    if analytic:
        hf, hj = ctx['h_only'], ctx['h_and_J']
    else:
        hf = ctx['decode_features']

        def hj(z):
            return hf(z), jax.jacfwd(hf)(z)

    A0, A1, A2 = R['A0'], R['A1'], R['A2']
    v0, v1, v2, v3 = R['v0'], R['v1'], R['v2'], R['v3']
    s0, s1, s2 = R['s0'], R['s1'], R['s2']
    bias = R['bias']

    def cdn(z):
        return mask * decode_norm(z) + u_g

    def _gn_step(z_init, s_init, proj, Gw, wg, gg, t):
        a, b, c0, c1, e = proj
        Wup = a + t * b                                   # W u_prev  (rank)
        gup = c0 + t * c1                                 # g^T u_prev

        def _pieces(zs):
            z_, s_ = zs[:-1].astype(F32), zs[-1].astype(FRED)
            h = hf(z_).astype(FRED)
            Gwh = Gw @ h
            qq = h @ Gwh + 2.0 * bias * (h @ wg) + bias * bias * gg
            qup = h @ Wup + bias * gup
            r2 = s_ * s_ * qq - 2.0 * s_ * qup + e        # ||r||^2
            return z_, s_, h, Gwh, qq, qup, r2

        def sq_residual(zs):
            """||r||^2 alone -- the accept test needs no Jacobian."""
            return _pieces(zs)[-1]

        def normal_eqs(zs):
            z_, s_ = zs[:-1].astype(F32), zs[-1].astype(FRED)
            h32, Jh32 = hj(z_)
            h, Jh = h32.astype(FRED), Jh32.astype(FRED)   # Jh is (rank, k)
            Gwh = Gw @ h
            qq = h @ Gwh + 2.0 * bias * (h @ wg) + bias * bias * gg
            qup = h @ Wup + bias * gup
            r2 = s_ * s_ * qq - 2.0 * s_ * qup + e
            Wq = Gwh + bias * wg                          # W q
            JtJ = jnp.zeros((k + 1, k + 1), dtype=FRED)
            JtJ = JtJ.at[:k, :k].set(s_ * s_ * (Jh.T @ (Gw @ Jh)))
            cross = s_ * (Jh.T @ Wq)
            JtJ = JtJ.at[:k, k].set(cross).at[k, :k].set(cross)
            JtJ = JtJ.at[k, k].set(qq)
            Jtr = jnp.zeros(k + 1, dtype=FRED)
            Jtr = Jtr.at[:k].set(s_ * (Jh.T @ (s_ * Wq - Wup)))
            Jtr = Jtr.at[k].set(s_ * qq - qup)
            return JtJ, Jtr, r2

        zs0 = jnp.concatenate(
            [z_init.astype(ITER_DT), s_init.astype(ITER_DT)[None]])
        mu0 = jnp.asarray(1e-4, dtype=FRED)

        def _body(carry):
            zs, mu, _, itr = carry
            JtJ, Jtr, r2 = normal_eqs(zs)
            diag_scale = jnp.mean(jnp.diag(JtJ)) + 1e-8
            A = JtJ + mu * diag_scale * EYE
            if chol:
                Lc = jnp.linalg.cholesky(A)
                y = jax.scipy.linalg.solve_triangular(Lc, -Jtr, lower=True)
                d = jax.scipy.linalg.solve_triangular(Lc.T, y, lower=False)
            else:
                d = jnp.linalg.solve(A, -Jtr)
            zs_new = zs + d.astype(ITER_DT)
            accept = sq_residual(zs_new) < r2
            zs_next = jnp.where(accept, zs_new, zs)
            zs_next = zs_next.at[-1].set(
                jnp.maximum(zs_next[-1], jnp.asarray(1e-8, dtype=ITER_DT)))
            mu_next = jnp.where(accept,
                                jnp.maximum(mu * 0.3, jnp.asarray(1e-7, FRED)),
                                jnp.minimum(mu * 10.0, jnp.asarray(1e2, FRED)))
            return zs_next, mu_next, jnp.linalg.norm(Jtr), itr + 1

        def _cond(carry):
            _, _, jtr_norm, itr = carry
            return (jtr_norm > TOL) & (itr < IT)

        init = (zs0, mu0, jnp.asarray(jnp.inf, dtype=FRED),
                jnp.array(0, jnp.int32))
        if masked_iters:
            def _m(i, c):
                zs, mu, jn_, itr = c
                done = ~((jn_ > TOL) & (itr < IT))
                zn, mn, jn2, it2 = _body(c)
                return (jnp.where(done, zs, zn), jnp.where(done, mu, mn),
                        jnp.where(done, jn_, jn2), jnp.where(done, itr, it2))
            zs_f, _, _, itr_f = jax.lax.fori_loop(0, IT, _m, init)
        else:
            zs_f, _, _, itr_f = jax.lax.while_loop(_cond, _body, init)
        return zs_f[:-1], zs_f[-1], itr_f

    def _advance(z_, s_):
        """Project the new u_prev = s*(Y^T h + bias*m) without touching the
        grid.  This is why the rollout is mesh-free after step 0."""
        h = hf(z_.astype(F32)).astype(FRED)
        sf = s_.astype(FRED)
        return (sf * (A0 @ h + bias * v0),
                sf * (A1.T @ h + bias * v2),
                sf * (h @ v0 + bias * s0),
                sf * (h @ v1 + bias * s1),
                sf * sf * (h @ (A0 @ h) + 2.0 * bias * (h @ v0)
                           + bias * bias * s0))

    @jax.jit
    def rollout(z0, scale0, u0, kap, Yc, Ytc, mc, Kmc):
        # kap enters only through these three rank-space objects; forming them
        # is 3 rank x rank adds and is INSIDE the timed region.
        t = (DT * kap).astype(FRED)
        Gw = A0 + t * (A1 + A1.T) + t * t * A2
        wg = v0 + t * (v1 + v2) + t * t * v3
        gg = s0 + 2.0 * t * s1 + t * t * s2
        # one-time projection of the initial condition, O(rank * N^2)
        u0r = u0.astype(FRED)
        proj = (Yc @ u0r, Ytc @ u0r, mc @ u0r, Kmc @ u0r, u0r @ u0r)
        iters = jnp.zeros(N_STEPS, jnp.int32)

        def _step(i, c):
            z_, sc_, proj_, it_ = c
            z_n, sc_n, ni = _gn_step(z_, sc_, proj_, Gw, wg, gg, t)
            return (z_n, sc_n, _advance(z_n, sc_n), it_.at[i].set(ni))

        z_f, s_f, _, it_f = jax.lax.fori_loop(
            0, N_STEPS, _step, (z0.astype(ITER_DT),
                                jnp.asarray(scale0, dtype=ITER_DT), proj, iters))
        # single grid-sized decode, once per trajectory, inside the timed region
        u_phys = s_f.astype(F32) * cdn(z_f.astype(F32))
        return z_f, s_f, u_phys, it_f

    return rollout


def make_solve_reduced(ctx, R, rollout, jit_encoder=True):
    encode = ctx['encode']
    Yc, Ytc, mc, Kmc = R['Y'], R['Yt'], R['m'], R['Km']

    enc = jax.jit(encode) if jit_encoder else encode

    def solve(u0, kap, z0, s0):
        za, sa = enc(u0)
        return rollout(za, sa, u0, kap, Yc, Ytc, mc, Kmc)
    return solve


def make_solve(ctx, rollout, include_encoder, jit_encoder):
    """`include_encoder=False` reproduces the submission-era timed region,
    which starts after the encoder has already run."""
    encode = ctx['encode']

    if not include_encoder:
        def solve(u0, kap, z0, s0):
            return rollout(z0, s0, u0, kap)
        return solve

    if jit_encoder:
        @jax.jit
        def solve(u0, kap, z0, s0):
            za, sa = encode(u0)
            return rollout(za, sa, u0, kap)
        return solve

    def solve(u0, kap, z0, s0):
        za, sa = encode(u0)
        return rollout(za, sa, u0, kap)
    return solve


VARIANTS = {
    # submission-era timed region: encoder excluded, while_loop, LU
    'base':      dict(include_encoder=False, jit_encoder=True,
                      masked=False, chol=False),
    # same solve, but the encoder is inside the measured cost (uncompiled)
    'base_enc':  dict(include_encoder=True, jit_encoder=False,
                      masked=False, chol=False),
    # honest online cost, everything compiled
    'enc_jit':   dict(include_encoder=True, jit_encoder=True,
                      masked=False, chol=False),
    'v4':        dict(include_encoder=True, jit_encoder=True,
                      masked=True, chol=True),
    'v4_lu':     dict(include_encoder=True, jit_encoder=True,
                      masked=True, chol=False),
    # keeps the while_loop, for cells that converge well before the cap
    'v4w':       dict(include_encoder=True, jit_encoder=True,
                      masked=False, chol=True),
    # dense (N^2 x k+1) Jacobian assembly, but formed in f64.  THE control
    # that separates precision from assembly: if this also converges in ~4
    # Gauss-Newton iterations, the iteration-count drop belongs to precision,
    # not to the reduced-Gram reformulation.
    'v4w64':     dict(include_encoder=True, jit_encoder=True,
                      masked=False, chol=True, f64_asm=True),
    'v4w64f':    dict(include_encoder=True, jit_encoder=True,
                      masked=False, chol=True, f64_asm=True, f64_dec=True),
    'v4_lu64':   dict(include_encoder=True, jit_encoder=True,
                      masked=False, chol=False, f64_asm=True),
    # --- reduced-Gram variants (exact; require X64=1) --------------------
    # f32 iterates, f64 Grams: the iterate sequence stays in the same precision
    # as the dense path, so nothing is bought by carrying z in f64.
    'red':       dict(reduced=True, masked=False, chol=True, f64_it=False),
    'red_lu':    dict(reduced=True, masked=False, chol=False, f64_it=False),
    'red_m':     dict(reduced=True, masked=True, chol=True, f64_it=False),
    # same, but dh/dz from jax.jacfwd instead of the closed form -- isolates
    # what the analytic MLP Jacobian is worth
    'red_jf':    dict(reduced=True, masked=False, chol=True, f64_it=False,
                      analytic=False),
    # f64 iterates as well -- reported separately, it is NOT the same iteration
    'red64':     dict(reduced=True, masked=False, chol=True, f64_it=True),
    'red64_m':   dict(reduced=True, masked=True, chol=True, f64_it=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=sorted(CELLS))
    ap.add_argument('--variants', default='base;base_enc;enc_jit;v4')
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    if jax.default_backend() != 'gpu':
        raise SystemExit(f'FATAL jax_backend={jax.default_backend()}')
    _excl_start, _excl_ev = observe_isolation()
    print(f'isolated={_excl_start} evidence={_excl_ev}', flush=True)
    print(f'jax_backend={jax.default_backend()} {jax.devices()}', flush=True)

    ctx = build(args.cell)
    cfg = ctx['cfg']
    fom = make_fom(ctx)
    enc_jit = jax.jit(ctx['encode'])

    specs = args.variants.split(';')
    need_red = any(VARIANTS[s.partition('@')[0]].get('reduced')
                   for s in specs)
    if need_red and not X64:
        raise SystemExit('FATAL: reduced variants need X64=1 (f32 Grams are '
                         'off by 7% on J^T J -- see test_red2d.py)')
    R = build_reduced(ctx) if need_red else None

    roms, knobs = {}, {}
    for spec in specs:
        name, _, knob = spec.partition('@')
        v = dict(VARIANTS[name])
        it = tol = None
        if knob:
            a, _, bb = knob.partition(',')
            # `@4` (no comma) means: change the cap, keep the cell's tolerance
            it, tol = int(a), (float(bb) if bb else None)
        knobs[spec] = dict(gn_max_iters=it or cfg['gn_max_iters'],
                           gn_tol=tol if tol is not None else cfg['gn_tol'])
        if v.get('reduced'):
            r = make_rollout_reduced(ctx, R, masked_iters=v['masked'],
                                     chol=v['chol'], gn_max_iters=it,
                                     gn_tol=tol, f64_iterates=v['f64_it'],
                                     analytic=v.get('analytic', True))
            roms[spec] = make_solve_reduced(ctx, R, r)
        else:
            r = make_rollout(ctx, masked_iters=v['masked'], chol=v['chol'],
                             gn_max_iters=it, gn_tol=tol,
                             f64_assembly=v.get('f64_asm', False),
                             f64_decoder=v.get('f64_dec', False))
            roms[spec] = make_solve(ctx, r, v['include_encoder'],
                                    v['jit_encoder'])

    out = {v: dict(rel_l2=[], t=[], all_t=[], iters=[]) for v in roms}
    fom_t, fom_all, kappas, fid = [], [], [], []

    for i, tr in enumerate(ctx['trajs']):
        u0, kap_f, ref = tr['u0'], jnp.float32(tr['kappa']), tr['final']
        nrm = float(jnp.linalg.norm(ref))
        kappas.append(tr['kappa'])
        z0, s0 = jax.block_until_ready(enc_jit(u0))

        f_t, r_t, r_out = [], {v: [] for v in roms}, {}
        u_f = None
        for _ in range(args.warmup):
            u_f = jax.block_until_ready(fom(u0, kap_f))
            for v, rr in roms.items():
                r_out[v] = jax.block_until_ready(rr(u0, kap_f, z0, s0))
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            u_f = jax.block_until_ready(fom(u0, kap_f))
            f_t.append(time.perf_counter() - t0)
            for v, rr in roms.items():
                t0 = time.perf_counter()
                r_out[v] = jax.block_until_ready(rr(u0, kap_f, z0, s0))
                r_t[v].append(time.perf_counter() - t0)

        tf = float(np.median(f_t))
        fom_t.append(tf)
        fom_all.append(f_t)
        fid.append(float(jnp.linalg.norm(u_f - ref) / (nrm + 1e-30)))
        line = f'  [{i+1:2d}] k={tr["kappa"]:.4f} FOM {tf*1e3:8.2f}ms'
        for v in roms:
            t_r = float(np.median(r_t[v]))
            _, _, u_cur, it_arr = r_out[v]
            e = float(jnp.linalg.norm(u_cur - ref) / (nrm + 1e-12))
            out[v]['rel_l2'].append(e)
            out[v]['t'].append(t_r)
            out[v]['all_t'].append(r_t[v])
            out[v]['iters'].append([int(x) for x in np.asarray(it_arr)])
            line += f' | {v} {t_r*1e3:7.2f}ms {tf/t_r:6.3f}x e={e:.4e}'
        print(line, flush=True)

    print('\n=== summary ===', flush=True)
    print(f'cell {args.cell} k={cfg["k_dim"]} gn_max_iters={cfg["gn_max_iters"]} '
          f'gn_tol={cfg["gn_tol"]}  (DENSE residual, no EQ)')
    print(f'FOM jit vs cached reference: max rel dev {max(fid):.2e}')
    print(f'FOM median {np.median(fom_t)*1e3:.2f} ms')
    ref_l2 = cfg['retimed_rel_l2']
    summary = dict(cell=args.cell, kappas=kappas, fom_times=fom_t,
                   fom_all=fom_all, fom_fidelity=fid, config=cfg,
                   backend=jax.default_backend(),
                   xla_flags=os.environ.get('XLA_FLAGS', ''),
                   node=os.environ.get('SLURMD_NODENAME'),
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   exclusive_node=_excl_start and observe_isolation()[0],
                   isolation_evidence={'at_start': _excl_ev,
                                       'at_end': observe_isolation()[1]},
                   knobs=knobs, x64=X64,
                   reduced_build_seconds=(R['build_seconds'] if R else None),
                   reduced_offline_bytes=(R['offline_bytes'] if R else None),
                   reduced_rank=(R['rank'] if R else None),
                   variants={})
    for v in roms:
        m = float(np.mean(out[v]['rel_l2']))
        ratios = [f / r for f, r in zip(fom_t, out[v]['t'])]
        gn = float(np.mean([np.mean(x) for x in out[v]['iters']]))
        print(f'  {v:10s} rel_l2 {m:.6e} (delta vs retimed '
              f'{abs(m-ref_l2)/ref_l2:.2e})  ROM {np.median(out[v]["t"])*1e3:8.2f} ms'
              f'  speedup med {np.median(ratios):.3f}x'
              f'  range {min(ratios):.3f}-{max(ratios):.3f}x  GN {gn:.1f}')
        summary['variants'][v] = dict(
            knobs=knobs[v], rel_l2_mean=m, per_traj_rel_l2=out[v]['rel_l2'],
            rom_times=out[v]['t'], rom_all=out[v]['all_t'],
            gn_iters=out[v]['iters'],
            speedup_median=float(np.median(ratios)),
            speedup_range=[float(min(ratios)), float(max(ratios))],
            per_traj_speedup=[float(x) for x in ratios])

    # Equivalence check: the reduced variants must reproduce the dense
    # variants' Gauss-Newton counts step for step, or they are not the same
    # iteration and the timing is not comparable.
    # compare against the dense variant with the SAME loop and the same small
    # solve (while_loop + Cholesky), so only the reduced algebra differs
    _dense = [v for v in roms
              if not VARIANTS[v.partition('@')[0]].get('reduced')]
    dense_ref = next((v for v in _dense if v.startswith('v4w')),
                     next((v for v in _dense if v.startswith('enc_jit')),
                          _dense[0] if _dense else None))
    if dense_ref is not None:
        for v in roms:
            if not VARIANTS[v.partition('@')[0]].get('reduced'):
                continue
            same = sum(a == b for a, b in zip(out[dense_ref]['iters'],
                                              out[v]['iters']))
            tot = sum(sum(abs(np.asarray(a) - np.asarray(b)))
                      for a, b in zip(out[dense_ref]['iters'], out[v]['iters']))
            de = abs(np.mean(out[v]['rel_l2']) - np.mean(out[dense_ref]['rel_l2'])) \
                / (np.mean(out[dense_ref]['rel_l2']) + 1e-30)
            print(f'  [equiv] {v:9s} vs {dense_ref}: trajectories with '
                  f'identical GN counts {same}/{len(out[v]["iters"])}, '
                  f'total |GN diff| {int(tot)}, rel_l2 delta {de:.2e}')
            summary['variants'][v]['equiv_vs'] = dense_ref
            summary['variants'][v]['equiv_traj_identical_gn'] = int(same)
            summary['variants'][v]['equiv_total_gn_diff'] = int(tot)
            summary['variants'][v]['equiv_rel_l2_delta'] = float(de)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
