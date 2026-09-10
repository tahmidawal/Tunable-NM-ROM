"""
Heat-3D NM-ROM cost study (discussion period, NeurIPS 25242).

Standalone re-implementation of the *timed path* of the submission-era
Heat-3D driver (exp_best.py / exp_acc.py / exp_fast.py), so that the ROM
rollout can be restructured and re-measured without touching the FOM.

Correctness gate: every variant must reproduce the submission-era rel-L2 of
its cell. `base` must reproduce it to ~1e-8 (it is the same graph); algebraic
variants are allowed to differ at f32 re-association level and the deviation
is reported, never hidden.

The FOM is FROZEN: same operator, same CG tolerance (1e-6), same 1000-iter
cap, same 50 backward-Euler steps, jitted with kappa as a runtime argument,
warmed and median-aggregated exactly like the ROM.
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
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import flax.linen as nn
from scipy.optimize import nnls
from scipy.stats import qmc

import bench as _bench

ROOT = '/cluster/tufts/paralab/tawal01/NMROM-Apr8'
SHARED_DATA = {
    32:  f'{ROOT}/20260416-NeurIPS/Heat-3D/shared_data/training_data_32.pkl',
    64:  f'{ROOT}/20260416-NeurIPS/Heat-3D/shared_data/training_data_64.pkl',
    128: f'{ROOT}/20260416-NeurIPS/Heat-3D/shared_data/training_data_128.pkl',
}
RUNS = f'{ROOT}/20260423-NEURIPS/Autoresearch/Heat-3D/runs'

CELLS = {
    'h3d_n32': dict(
        N=32, k_dim=32, rank=512, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=6, max_iters=3, gn_tol=1e-3,
        n_eq_samples=16, patch_nnls=False, arm='best',
        ckpt=f'{RUNS}/N32_fixed_run10/checkpoint_vitcp.pkl',
        fom_cache=f'{RUNS}/N32_fixed_run10/fom_cache_32.pkl',
        posted_rel_l2=9.30e-2, retimed_rel_l2=0.09301180187612772),
    'h3d_n64_acc': dict(
        N=64, k_dim=40, rank=512, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=6, max_iters=12, gn_tol=1e-5,
        n_eq_samples=32, patch_nnls=True, arm='acc',
        ckpt=f'{RUNS}/N64_fixed_run3/checkpoint_vitcp.pkl',
        fom_cache=f'{RUNS}/N64_fixed_run3/fom_cache_64.pkl',
        posted_rel_l2=7.49e-2, retimed_rel_l2=0.07330456413328648),
    'h3d_n64_fast': dict(
        N=64, k_dim=40, rank=512, hidden_dim=256, patch_size=8, embed_dim=96,
        num_heads=4, num_enc_layers=6, max_iters=2, gn_tol=1e-2,
        n_eq_samples=32, patch_nnls=True, arm='fast',
        ckpt=f'{RUNS}/N64_fixed_run3/checkpoint_vitcp.pkl',
        fom_cache=f'{RUNS}/N64_fixed_run3/fom_cache_64.pkl',
        posted_rel_l2=7.85e-2, retimed_rel_l2=0.07719398587942124),
    'h3d_n128_acc': dict(
        N=128, k_dim=40, rank=512, hidden_dim=256, patch_size=16, embed_dim=64,
        num_heads=4, num_enc_layers=4, max_iters=12, gn_tol=1e-3,
        n_eq_samples=16, patch_nnls=True, arm='acc',
        ckpt=f'{RUNS}/N128_run10/checkpoint_vitcp.pkl',
        fom_cache=f'{RUNS}/N128_run10/fom_cache_128.pkl',
        posted_rel_l2=1.82e-1, retimed_rel_l2=0.18266838788986206),
    'h3d_n128_fast': dict(
        N=128, k_dim=40, rank=512, hidden_dim=256, patch_size=16, embed_dim=64,
        num_heads=4, num_enc_layers=4, max_iters=12, gn_tol=1e-3,
        n_eq_samples=16, patch_nnls=True, arm='fast',
        ckpt=f'{RUNS}/N128_run2/checkpoint_vitcp.pkl',
        fom_cache=f'{RUNS}/N128_run2/fom_cache_128.pkl',
        posted_rel_l2=1.90e-1, retimed_rel_l2=0.19028177112340927),
}

# Quadrature-budget variants.  n_eq comes out equal to k * n_eq_samples on every
# cell, so halving n_eq_samples halves the number of quadrature nodes and roughly
# halves the ROM's per-iteration cost.  NOTE: this is an OFFLINE change -- the
# NNLS has to be re-solved -- so it is NOT the deployment-time knob the paper
# describes.  Reported as an offline retuning.
CELLS['h3d_n64_acc_eq16'] = dict(CELLS['h3d_n64_acc'], n_eq_samples=16)
CELLS['h3d_n128_acc_eq8'] = dict(CELLS['h3d_n128_acc'], n_eq_samples=8)

NUM_STEPS = 50
DT = 0.005
AMP_EPS = 1e-6
S_MIN = jnp.float32(1e-10)
CG_TOL = 1e-6
CG_ITERS = 1000


# ---------------------------------------------------------------------------
# Model (verbatim from the submission-era driver)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(int(self.embed_dim * self.mlp_ratio))(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        return x + h


class ViTEncoder(nn.Module):
    latent_dim: int
    grid_n: int
    patch_size: int = 8
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4

    def setup(self):
        nps = self.grid_n // self.patch_size
        self._nps = nps
        self._np = nps ** 3
        self._pd = self.patch_size ** 3
        self.patch_embed = nn.Dense(self.embed_dim)
        self.enc_pos = self.param('enc_pos', nn.initializers.normal(0.02),
                                  (self._np, self.embed_dim))
        self.enc_blocks = [TransformerBlock(self.embed_dim, self.num_heads)
                           for _ in range(self.num_enc_layers)]
        self.enc_norm = nn.LayerNorm()
        self.enc_proj = nn.Dense(self.latent_dim)

    def _patchify(self, u):
        n = self._nps
        p = self.patch_size
        return u.reshape(n, p, n, p, n, p).transpose(0, 2, 4, 1, 3, 5).reshape(
            self._np, self._pd)

    def __call__(self, u):
        x = self.patch_embed(self._patchify(u)) + self.enc_pos
        for b in self.enc_blocks:
            x = b(x)
        return self.enc_proj(self.enc_norm(x).mean(axis=0))


class LinearCPDecoder(nn.Module):
    latent_dim: int
    rank: int = 512
    grid_size: int = 32
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
        self.W_z = self.param('W_z', init, (self.rank, Ng))
        self.bias = self.param('bias', nn.initializers.zeros, ())

    def __call__(self, z):
        h_nl = nn.swish(self.W1(z))
        h_nl = nn.swish(self.W2(h_nl))
        h_nl = self.W_rank(h_nl)
        h = self.W_direct(z) + h_nl
        return jnp.einsum('r,ri,rj,rk->ijk', h,
                          self.W_x, self.W_y, self.W_z).flatten() + self.bias


def make_ae(cfg, N):
    class ViTCPAutoencoder(nn.Module):
        latent_dim: int
        grid_size: int = 32
        patch_size: int = 8
        embed_dim: int = 64
        num_heads: int = 4
        num_enc_layers: int = 4
        rank: int = 512
        hidden_dim: int = 256

        def setup(self):
            self.encoder = ViTEncoder(
                latent_dim=self.latent_dim, grid_n=self.grid_size,
                patch_size=self.patch_size, embed_dim=self.embed_dim,
                num_heads=self.num_heads, num_enc_layers=self.num_enc_layers)
            self.decoder = LinearCPDecoder(
                latent_dim=self.latent_dim, rank=self.rank,
                grid_size=self.grid_size, hidden_dim=self.hidden_dim)

        def encode(self, u, training=False):
            scale = jnp.max(jnp.abs(u)) + AMP_EPS
            return self.encoder(u / scale), scale

        def decode(self, z, scale):
            return self.decoder(z) * scale

        def decode_normalised(self, z):
            return self.decoder(z)

    return ViTCPAutoencoder(
        latent_dim=cfg['k_dim'], grid_size=N, patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'], num_heads=cfg['num_heads'],
        num_enc_layers=cfg['num_enc_layers'], rank=cfg['rank'],
        hidden_dim=cfg['hidden_dim'])


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------
def build(cell, eq_cache_dir):
    cfg = CELLS[cell]
    N = cfg['N']
    num_nodes = N ** 3
    dx = 1.0 / (N - 1)

    with open(SHARED_DATA[N], 'rb') as f:
        data = pickle.load(f)

    model = make_ae(cfg, N)
    with open(cfg['ckpt'], 'rb') as f:
        params = pickle.load(f)['params']

    x_sp = jnp.linspace(0, 1.0, N, dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(x_sp, x_sp, x_sp, indexing='ij')

    def K_op_3d(u_flat):
        u = u_flat.reshape((N, N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            (6 * u[1:-1, 1:-1, 1:-1] - u[0:-2, 1:-1, 1:-1] - u[2:, 1:-1, 1:-1]
             - u[1:-1, 0:-2, 1:-1] - u[1:-1, 2:, 1:-1]
             - u[1:-1, 1:-1, 0:-2] - u[1:-1, 1:-1, 2:]) / dx ** 2)
        out = out.at[0, :, :].set(u[0, :, :])
        out = out.at[-1, :, :].set(u[-1, :, :])
        out = out.at[:, 0, :].set(u[:, 0, :])
        out = out.at[:, -1, :].set(u[:, -1, :])
        out = out.at[:, :, 0].set(u[:, :, 0])
        out = out.at[:, :, -1].set(u[:, :, -1])
        return out.flatten()

    def implicit_op(u_flat, kappa):
        return u_flat + DT * kappa * K_op_3d(u_flat)

    mask_3d = jnp.ones((N, N, N), dtype=jnp.float32)
    mask_3d = mask_3d.at[0, :, :].set(0.).at[-1, :, :].set(0.)
    mask_3d = mask_3d.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
    mask_3d = mask_3d.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
    mask = mask_3d.flatten()
    u_g = jnp.zeros(num_nodes, dtype=jnp.float32)

    def encode(u_flat):
        return model.apply({'params': params}, u_flat, training=False,
                           method=model.encode)

    def decode_normalised(z):
        return model.apply({'params': params}, z, method=model.decode_normalised)

    def decode_full(z, scale):
        return model.apply({'params': params}, z, scale, method=model.decode)

    def constrained_decode_normalised(z):
        return mask * decode_normalised(z) + u_g

    # ---- EQ (cached; the NNLS support is an offline artefact) --------------
    ckpt_tag = Path(cfg['ckpt']).parent.name
    cache = (Path(eq_cache_dir) /
             f'eq_N{N}_{ckpt_tag}_s{cfg["n_eq_samples"]}'
             f'_p{int(cfg["patch_nnls"])}.npz')
    if cache.exists():
        z = np.load(cache)
        eq_indices, eq_weights = z['idx'], z['w']
        print(f'[eq] loaded cached support: {len(eq_indices)} nodes '
              f'(sha {int(z["sha"]):d})', flush=True)
    else:
        all_pairs = []
        for i, traj in enumerate(data['all_snapshots']):
            kap = data['traj_kappas'][i]
            for step in range(NUM_STEPS):
                all_pairs.append((traj[step + 1], traj[step], kap))
        rng_eq = np.random.default_rng(123 + 0)
        sel = rng_eq.choice(len(all_pairs),
                            size=min(cfg['n_eq_samples'], len(all_pairs)),
                            replace=False)
        eq_pairs = [all_pairs[i] for i in sel]

        def get_integrand(z_next, u_prev_norm, kappa):
            def cd(zz):
                return constrained_decode_normalised(zz)
            u_pred = cd(z_next)
            R = u_pred - u_prev_norm + DT * kappa * K_op_3d(u_pred)
            J_D = jax.jacfwd(cd)(z_next)
            return J_D.T * R[None, :]

        # patch_nnls Fix 3: at N>=64 the jitted jacfwd is not compilable, so
        # the submission path ran this eagerly. Match that per cell.
        gi = get_integrand if cfg['patch_nnls'] else jax.jit(get_integrand)

        G_list = []
        for u_next, u_prev, kap in eq_pairs:
            z_next, _ = encode(jnp.asarray(u_next))
            sc = jnp.max(jnp.abs(jnp.asarray(u_prev))) + AMP_EPS
            u_prev_norm = jnp.asarray(u_prev) / sc
            G_list.append(np.asarray(gi(z_next, u_prev_norm, jnp.float32(kap))))
        G_np = np.concatenate(G_list, axis=0)
        G_np[:, np.asarray(mask) == 0] = 0.0
        b_np = np.sum(G_np, axis=1)
        # patch_nnls Fix 1: cap at 3*nrows (default 4*ncols hangs at N>=64)
        maxit = (3 * G_np.shape[0]) if cfg['patch_nnls'] else 4 * G_np.shape[1]
        t0 = time.perf_counter()
        w_eq, _ = nnls(G_np, b_np, maxiter=maxit)
        print(f'[eq] nnls maxiter={maxit} took {time.perf_counter()-t0:.1f}s',
              flush=True)
        eq_indices = np.where(w_eq > 1e-10)[0]
        eq_weights = w_eq[eq_indices]
        sha = int(np.sum(eq_indices.astype(np.int64)) % (1 << 62))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, idx=eq_indices, w=eq_weights, sha=np.int64(sha))
        del G_list, G_np
    print(f'[eq] n_eq = {len(eq_indices)} / {num_nodes}', flush=True)

    eq_idx_jnp = jnp.array(eq_indices)
    eq_w_jnp = jnp.array(eq_weights)
    n_eq = len(eq_indices)

    N2 = N * N
    stencil_offsets = jnp.array([0, -1, 1, -N, N, -N2, N2])
    gather_indices = (eq_idx_jnp[:, None] + stencil_offsets[None, :]).flatten()
    ix = gather_indices // N2
    iy = (gather_indices // N) % N
    iz = gather_indices % N
    W_x = params['decoder']['W_x']
    W_y = params['decoder']['W_y']
    W_z = params['decoder']['W_z']
    V_eq = W_x[:, ix] * W_y[:, iy] * W_z[:, iz]
    b_scalar = params['decoder']['bias']
    b_sparse = jnp.full(gather_indices.shape, b_scalar)
    mask_sp = mask[gather_indices]
    ug_sp = u_g[gather_indices]

    dec = params['decoder']
    mlp_w = (dec['W1']['kernel'], dec['W1']['bias'],
             dec['W2']['kernel'], dec['W2']['bias'],
             dec['W_rank']['kernel'], dec['W_rank']['bias'],
             dec['W_direct']['kernel'], dec['W_direct']['bias'])

    def make_gaussian_ic(centers, amplitudes, widths):
        u = jnp.zeros((N, N, N), dtype=jnp.float32)
        for (cx, cy, cz), A, sigma in zip(centers, amplitudes, widths):
            u = u + A * jnp.exp(
                -((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * sigma ** 2))
        u = u.at[0, :, :].set(0.).at[-1, :, :].set(0.)
        u = u.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        u = u.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        # f32 pinned: under X64=1 the meshgrid promotes and the whole solve --
        # including the frozen dense control -- would silently move to f64
        return u.flatten().astype(jnp.float32)

    return dict(
        cfg=cfg, N=N, dx=dx, num_nodes=num_nodes, data=data, params=params,
        model=model, mask=mask, u_g=u_g, encode=encode,
        decode_full=decode_full, implicit_op=implicit_op, K_op_3d=K_op_3d,
        V_eq=V_eq, b_sparse=b_sparse, mask_sp=mask_sp, ug_sp=ug_sp,
        eq_idx_jnp=eq_idx_jnp, eq_w_jnp=eq_w_jnp, n_eq=n_eq,
        mlp_w=mlp_w, make_gaussian_ic=make_gaussian_ic,
        eq_indices=eq_indices, eq_weights=eq_weights)


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


# ---------------------------------------------------------------------------
# FOM  -- FROZEN.  Same operator, same CG tol, same iteration cap, same steps.
# ---------------------------------------------------------------------------
def make_fom(ctx):
    implicit_op = ctx['implicit_op']

    @jax.jit
    def fom_rollout(u0_flat, kappa_rt):
        def _op(v):
            return implicit_op(v, kappa_rt)

        def _step(_i, u):
            u2, _ = jax_linalg.cg(_op, u, x0=u, tol=CG_TOL, maxiter=CG_ITERS)
            return u2

        return jax.lax.fori_loop(0, NUM_STEPS, _step, u0_flat)

    return fom_rollout


# ---------------------------------------------------------------------------
# ROM variants
# ---------------------------------------------------------------------------
def make_rom_base(ctx, masked_iters=False, chol=False):
    """Verbatim submission-era timed path (_solve_step_rt / _rollout_rt).

    `masked_iters` swaps the data-dependent while_loop for a counted loop that
    freezes the update once the stopping test is met.  Nothing else changes, so
    the iterates stay bit-identical (test_equiv.py checks this).
    `chol` swaps the LU solve of the SPD system for Cholesky."""
    cfg = ctx['cfg']
    k_dim = cfg['k_dim']
    GN_MAX_ITERS = cfg['max_iters']
    GN_REL_TOL = cfg['gn_tol']
    dx = ctx['dx']
    V_eq, b_sparse, mask_sp, ug_sp = ctx['V_eq'], ctx['b_sparse'], ctx['mask_sp'], ctx['ug_sp']
    eq_w_jnp = ctx['eq_w_jnp']
    W1k, b1k, W2k, b2k, Wrk, brk, Wdk, bdk = ctx['mlp_w']

    def _mlp_body(z):
        h_nl = nn.swish(z @ W1k + b1k)
        h_nl = nn.swish(h_nl @ W2k + b2k)
        h_nl = h_nl @ Wrk + brk
        h_lin = z @ Wdk + bdk
        return h_lin + h_nl

    def _solve_step_rt(z_init, s_init, u_prev_phys_eq_, kappa_f32):
        def _f_norm(z):
            h = _mlp_body(z)
            u_st = (mask_sp * (h @ V_eq + b_sparse) + ug_sp).reshape(-1, 7)
            lap = (6 * u_st[:, 0] - u_st[:, 1] - u_st[:, 2] - u_st[:, 3]
                   - u_st[:, 4] - u_st[:, 5] - u_st[:, 6]) / dx ** 2
            return u_st[:, 0] + DT * kappa_f32 * lap

        fn0 = _f_norm(z_init)
        R0 = s_init * fn0 - u_prev_phys_eq_
        J0 = jax.jacfwd(_f_norm)(z_init)
        WR0 = eq_w_jnp * R0
        g_z0 = s_init * (J0.T @ WR0)
        g_s0 = jnp.dot(fn0, WR0)
        gnorm0 = jnp.maximum(jnp.sqrt(jnp.dot(g_z0, g_z0) + g_s0 ** 2), 1e-30)

        def _body(carry):
            z, s, _, itr = carry
            fn = _f_norm(z)
            R = s * fn - u_prev_phys_eq_
            J = jax.jacfwd(_f_norm)(z)
            Wfn = eq_w_jnp * fn
            WR = eq_w_jnp * R
            WJ = eq_w_jnp[:, None] * J
            JtWJ = J.T @ WJ
            JtWfn = J.T @ Wfn
            fnWfn = jnp.dot(fn, Wfn)
            JtWR = J.T @ WR
            fnWR = jnp.dot(fn, WR)
            H_aug = jnp.block([[s ** 2 * JtWJ, (s * JtWfn)[:, None]],
                               [(s * JtWfn)[None, :], fnWfn[None, None]]])
            g_aug = jnp.append(s * JtWR, fnWR)
            lam = jnp.maximum(1e-3 * jnp.trace(H_aug) / (k_dim + 1), 1e-8)
            _A = H_aug + lam * jnp.eye(k_dim + 1, dtype=jnp.float32)
            if chol:
                _L = jnp.linalg.cholesky(_A)
                _y = jax.scipy.linalg.solve_triangular(_L, -g_aug, lower=True)
                delta = jax.scipy.linalg.solve_triangular(_L.T, _y, lower=False)
            else:
                delta = jnp.linalg.solve(_A, -g_aug)
            dz = delta[:k_dim]
            ds = delta[k_dim]
            f0v = jnp.dot(WR, R)

            def _f(a):
                fn_t = _f_norm(z + a * dz)
                R_t = (s + a * ds) * fn_t - u_prev_phys_eq_
                return jnp.dot(eq_w_jnp * R_t, R_t)

            f1, f2, f3, f4 = _f(1.), _f(.5), _f(.25), _f(.125)
            step = jnp.where(f1 < f0v, 1., jnp.where(
                f2 < f0v, .5, jnp.where(f3 < f0v, .25,
                                        jnp.where(f4 < f0v, .125, 0.))))
            gnorm = jnp.sqrt(jnp.dot(s * J.T @ WR, s * J.T @ WR)
                             + jnp.dot(fn, WR) ** 2)
            return z + step * dz, jnp.maximum(s + step * ds, S_MIN), gnorm, itr + 1

        def _cond(carry):
            _, _, gnorm, itr = carry
            return (gnorm > GN_REL_TOL * gnorm0) & (itr < GN_MAX_ITERS)

        _init = (z_init, s_init, jnp.array(jnp.inf, jnp.float32),
                 jnp.array(0, jnp.int32))
        if masked_iters:
            def _masked(_i, c):
                z, s, gnorm, itr = c
                done = ~_cond(c)
                zn, sn, gn, itn = _body(c)
                return (jnp.where(done, z, zn), jnp.where(done, s, sn),
                        jnp.where(done, gnorm, gn), jnp.where(done, itr, itn))
            z_f, s_f, gnorm_f, itr_f = jax.lax.fori_loop(
                0, GN_MAX_ITERS, _masked, _init)
        else:
            z_f, s_f, gnorm_f, itr_f = jax.lax.while_loop(_cond, _body, _init)
        fn_f = _f_norm(z_f)
        R_f = s_f * fn_f - u_prev_phys_eq_
        res_norm = jnp.sqrt(jnp.dot(eq_w_jnp * R_f, R_f))
        h_new = _mlp_body(z_f)
        u_new_eq_norm = (mask_sp * (h_new @ V_eq + b_sparse) + ug_sp).reshape(-1, 7)[:, 0]
        return z_f, s_f, gnorm_f, itr_f, res_norm, s_f * u_new_eq_norm

    @jax.jit
    def rollout(z0, s0, u_cur_eq0, kappa_f32):
        iters_arr = jnp.zeros(NUM_STEPS, dtype=jnp.int32)
        res_arr = jnp.zeros(NUM_STEPS, dtype=jnp.float32)

        def _step(i, carry):
            z_, s_, u_eq_, iters_, res_ = carry
            z_n, s_n, _, n_iters, res_norm, u_eq_n = _solve_step_rt(
                z_, s_, u_eq_, kappa_f32)
            return (z_n, s_n, u_eq_n, iters_.at[i].set(n_iters),
                    res_.at[i].set(res_norm))

        return jax.lax.fori_loop(0, NUM_STEPS, _step,
                                 (z0, s0, u_cur_eq0, iters_arr, res_arr))

    return rollout


def make_rom_v1(ctx, batched_ls=True, chol=True, linearize=True,
                fold_operator=True, keep_diag=True, fixed_iters=False,
                masked_iters=False):
    """Restructured rollout.  Same Gauss-Newton iteration, same line search,
    same stopping rule -- only the way the residual and its Jacobian are
    evaluated changes.

    fold_operator: the decoder is affine in the CP coefficient vector h(z) and
        the backward-Euler operator is linear, so (I + dt*kappa*K) can be
        contracted into the CP basis ONCE per kappa instead of being applied at
        every Gauss-Newton iteration.  The contraction happens inside the timed
        region.
    linearize: primal and Jacobian share one forward pass.
    batched_ls: the four line-search trial points evaluate as one batch.
    chol: the (k+1)x(k+1) regularised Gauss-Newton system is SPD, so Cholesky
        replaces the LU solve.
    """
    cfg = ctx['cfg']
    k_dim = cfg['k_dim']
    GN_MAX_ITERS = cfg['max_iters']
    GN_REL_TOL = cfg['gn_tol']
    dx = ctx['dx']
    n_eq = ctx['n_eq']
    eq_w_jnp = ctx['eq_w_jnp']
    W1k, b1k, W2k, b2k, Wrk, brk, Wdk, bdk = ctx['mlp_w']
    rank = cfg['rank']

    # --- constants folded once, offline w.r.t. kappa ---------------------
    Vm = (ctx['mask_sp'][None, :] * ctx['V_eq']).reshape(rank, n_eq, 7)
    cm = (ctx['mask_sp'] * ctx['b_sparse'] + ctx['ug_sp']).reshape(n_eq, 7)
    V_c = Vm[:, :, 0]
    V_lap = (6 * Vm[:, :, 0] - Vm[:, :, 1] - Vm[:, :, 2] - Vm[:, :, 3]
             - Vm[:, :, 4] - Vm[:, :, 5] - Vm[:, :, 6]) / dx ** 2
    c_c = cm[:, 0]
    c_lap = (6 * cm[:, 0] - cm[:, 1] - cm[:, 2] - cm[:, 3]
             - cm[:, 4] - cm[:, 5] - cm[:, 6]) / dx ** 2

    V_eq, b_sparse, mask_sp, ug_sp = ctx['V_eq'], ctx['b_sparse'], ctx['mask_sp'], ctx['ug_sp']

    def _mlp_body(z):
        h_nl = nn.swish(z @ W1k + b1k)
        h_nl = nn.swish(h_nl @ W2k + b2k)
        h_nl = h_nl @ Wrk + brk
        h_lin = z @ Wdk + bdk
        return h_lin + h_nl

    ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.125], dtype=jnp.float32)
    EYE = jnp.eye(k_dim + 1, dtype=jnp.float32)

    def _solve_step(z_init, s_init, u_prev, Vk, ck):
        if fold_operator:
            def _f_norm(z):
                return _mlp_body(z) @ Vk + ck
        else:
            kap = Vk  # scalar carrier when the fold is disabled

            def _f_norm(z):
                h = _mlp_body(z)
                u_st = (mask_sp * (h @ V_eq + b_sparse) + ug_sp).reshape(-1, 7)
                lap = (6 * u_st[:, 0] - u_st[:, 1] - u_st[:, 2] - u_st[:, 3]
                       - u_st[:, 4] - u_st[:, 5] - u_st[:, 6]) / dx ** 2
                return u_st[:, 0] + DT * kap * lap

        def _primal_and_jac(z):
            if linearize and fold_operator:
                # h and dh/dz share one pass; the basis contraction is a matmul
                Dh = jax.jacfwd(_mlp_body)(z)          # (rank, k)
                h = _mlp_body(z)
                return h @ Vk + ck, Vk.T @ Dh
            if linearize:
                fn, jvp = jax.linearize(_f_norm, z)
                return fn, jax.vmap(jvp, out_axes=1)(jnp.eye(k_dim, dtype=jnp.float32))
            return _f_norm(z), jax.jacfwd(_f_norm)(z)

        fn0, J0 = _primal_and_jac(z_init)
        R0 = s_init * fn0 - u_prev
        WR0 = eq_w_jnp * R0
        g_z0 = s_init * (J0.T @ WR0)
        g_s0 = jnp.dot(fn0, WR0)
        gnorm0 = jnp.maximum(jnp.sqrt(jnp.dot(g_z0, g_z0) + g_s0 ** 2), 1e-30)

        def _body(carry):
            z, s, _, itr = carry
            fn, J = _primal_and_jac(z)
            R = s * fn - u_prev
            Wfn = eq_w_jnp * fn
            WR = eq_w_jnp * R
            JtWJ = J.T @ (eq_w_jnp[:, None] * J)
            JtWfn = J.T @ Wfn
            fnWfn = jnp.dot(fn, Wfn)
            JtWR = J.T @ WR
            fnWR = jnp.dot(fn, WR)
            H_aug = jnp.block([[s ** 2 * JtWJ, (s * JtWfn)[:, None]],
                               [(s * JtWfn)[None, :], fnWfn[None, None]]])
            g_aug = jnp.append(s * JtWR, fnWR)
            lam = jnp.maximum(1e-3 * jnp.trace(H_aug) / (k_dim + 1), 1e-8)
            A = H_aug + lam * EYE
            if chol:
                Lc = jnp.linalg.cholesky(A)
                y = jax.scipy.linalg.solve_triangular(Lc, -g_aug, lower=True)
                delta = jax.scipy.linalg.solve_triangular(Lc.T, y, lower=False)
            else:
                delta = jnp.linalg.solve(A, -g_aug)
            dz = delta[:k_dim]
            ds = delta[k_dim]
            f0v = jnp.dot(WR, R)

            if batched_ls and fold_operator:
                hb = jax.vmap(_mlp_body)(z[None, :] + ALPHAS[:, None] * dz[None, :])
                fnb = hb @ Vk + ck[None, :]
                Rb = (s + ALPHAS * ds)[:, None] * fnb - u_prev[None, :]
                fv = jnp.sum((eq_w_jnp[None, :] * Rb) * Rb, axis=1)
            else:
                def _f(a):
                    fn_t = _f_norm(z + a * dz)
                    R_t = (s + a * ds) * fn_t - u_prev
                    return jnp.dot(eq_w_jnp * R_t, R_t)
                fv = jnp.stack([_f(1.), _f(.5), _f(.25), _f(.125)])

            step = jnp.where(fv[0] < f0v, 1., jnp.where(
                fv[1] < f0v, .5, jnp.where(fv[2] < f0v, .25,
                                           jnp.where(fv[3] < f0v, .125, 0.))))
            sJtWR = s * JtWR
            gnorm = jnp.sqrt(jnp.dot(sJtWR, sJtWR) + fnWR ** 2)
            return z + step * dz, jnp.maximum(s + step * ds, S_MIN), gnorm, itr + 1

        def _cond(carry):
            _, _, gnorm, itr = carry
            return (gnorm > GN_REL_TOL * gnorm0) & (itr < GN_MAX_ITERS)

        init = (z_init, s_init, jnp.array(jnp.inf, jnp.float32),
                jnp.array(0, jnp.int32))
        if fixed_iters:
            # Only exact where the tolerance never fires before the cap; the
            # driver prints the realised GN counts so that can be checked.
            z_f, s_f, gnorm_f, itr_f = jax.lax.fori_loop(
                0, GN_MAX_ITERS, lambda i, c: _body(c), init)
        elif masked_iters:
            # Same stopping rule as the while_loop, same iterates, but a fixed
            # trip count: once the tolerance is met the update is frozen.  A
            # data-dependent while predicate has to be resolved on the host
            # every iteration; a counted loop does not.  Trades arithmetic that
            # is thrown away for the removal of that synchronisation.
            def _masked(i, c):
                z, s, gnorm, itr = c
                done = ~((gnorm > GN_REL_TOL * gnorm0) & (itr < GN_MAX_ITERS))
                zn, sn, gn, itn = _body((z, s, gnorm, itr))
                return (jnp.where(done, z, zn), jnp.where(done, s, sn),
                        jnp.where(done, gnorm, gn), jnp.where(done, itr, itn))

            z_f, s_f, gnorm_f, itr_f = jax.lax.fori_loop(
                0, GN_MAX_ITERS, _masked, init)
        else:
            z_f, s_f, gnorm_f, itr_f = jax.lax.while_loop(_cond, _body, init)
        h_f = _mlp_body(z_f)
        if fold_operator:
            fn_f = h_f @ Vk + ck
            u_new_eq_norm = h_f @ V_c + c_c
        else:
            fn_f = _f_norm(z_f)
            u_new_eq_norm = (mask_sp * (h_f @ V_eq + b_sparse)
                             + ug_sp).reshape(-1, 7)[:, 0]
        R_f = s_f * fn_f - u_prev
        res_norm = jnp.sqrt(jnp.dot(eq_w_jnp * R_f, R_f))
        return z_f, s_f, gnorm_f, itr_f, res_norm, s_f * u_new_eq_norm

    @jax.jit
    def rollout(z0, s0, u_cur_eq0, kappa_f32):
        # kappa-dependent basis contraction: ONE (rank x 7 n_eq) -> (rank x n_eq)
        # reduction, inside the timed region, done once per solve.
        if fold_operator:
            Vk = V_c + DT * kappa_f32 * V_lap
            ck = c_c + DT * kappa_f32 * c_lap
        else:
            Vk, ck = kappa_f32, None

        iters_arr = jnp.zeros(NUM_STEPS, dtype=jnp.int32)
        res_arr = jnp.zeros(NUM_STEPS, dtype=jnp.float32)

        def _step(i, carry):
            z_, s_, u_eq_, iters_, res_ = carry
            z_n, s_n, _, n_it, rn, u_eq_n = _solve_step(z_, s_, u_eq_, Vk, ck)
            if keep_diag:
                return (z_n, s_n, u_eq_n, iters_.at[i].set(n_it),
                        res_.at[i].set(rn))
            return (z_n, s_n, u_eq_n, iters_, res_)

        return jax.lax.fori_loop(0, NUM_STEPS, _step,
                                 (z0, s0, u_cur_eq0, iters_arr, res_arr))

    return rollout


def make_rom_e2e(ctx, rollout, jit_encoder=True):
    """One compiled ROM solve: encode -> latent rollout -> decode.

    `jit_encoder=False` reproduces the submission-era structure, in which the
    ViT encoder ran one XLA op at a time from Python while only the rollout was
    compiled.  With it True the whole solve is a single compiled graph, which is
    the treatment the rollout already had.
    """
    encode = ctx['encode']
    decode_full = ctx['decode_full']
    eq_idx = ctx['eq_idx_jnp']

    if jit_encoder:
        @jax.jit
        def solve(u0, kappa_f32):
            z0, s0 = encode(u0)
            u_eq0 = decode_full(z0, s0)[eq_idx]
            z_f, s_f, _, it, rs = rollout(z0, s0, u_eq0, kappa_f32)
            return decode_full(z_f, s_f), it, rs
        return solve

    decode_jit = jax.jit(decode_full)

    def solve(u0, kappa_f32):
        z0, s0 = encode(u0)
        u_eq0 = decode_jit(z0, s0)[eq_idx]
        z_f, s_f, _, it, rs = rollout(z0, s0, u_eq0, kappa_f32)
        return decode_jit(z_f, s_f), it, rs
    return solve


VARIANTS = {
    'base':      lambda ctx: make_rom_base(ctx),
    # v5: the submission-era solver with ONLY the loop construct changed.
    # Bit-identical iterates -- see test_equiv.py.
    'v5':        lambda ctx: make_rom_base(ctx, masked_iters=True),
    # v5c: v5 plus Cholesky for the SPD (k+1)x(k+1) solve.
    'v5c':       lambda ctx: make_rom_base(ctx, masked_iters=True, chol=True),
    # v5w: the submission-era solver with ONLY the small dense solve changed
    # from LU to Cholesky.  Loop construct untouched.  This is the cleanest
    # variant to quote: everything else is byte-for-byte.
    'v5w':       lambda ctx: make_rom_base(ctx, masked_iters=False, chol=True),
    'v2':        lambda ctx: make_rom_v1(ctx, chol=True),
    'v2_fori':   lambda ctx: make_rom_v1(ctx, chol=True, fixed_iters=True),
    'v3':        lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=True),
    # v4w: like v4 but keeps the data-dependent while_loop, so cells whose
    # Gauss-Newton converges well before the cap do not pay for the cap.
    # v3w: the full algebraic restructuring but KEEPING the while_loop, so the
    # operator folding is not confounded with paying the iteration cap.
    'v3w':       lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=False,
                                         fold_operator=True, linearize=True,
                                         batched_ls=True),
    'v3w_nofold': lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=False,
                                          fold_operator=False, linearize=False,
                                          batched_ls=True),
    'v3w_nolin': lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=False,
                                         fold_operator=True, linearize=False,
                                         batched_ls=True),
    'v4w':       lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=False,
                                         fold_operator=False, linearize=False,
                                         batched_ls=False),
    # v4: submission-era residual/Jacobian algebra byte-for-byte, only the
    # control flow and the small dense solve change.
    'v4':        lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=True,
                                         fold_operator=False, linearize=False,
                                         batched_ls=False),
    'v3_lu':     lambda ctx: make_rom_v1(ctx, chol=False, masked_iters=True),
    'v3_nofold': lambda ctx: make_rom_v1(ctx, chol=True, masked_iters=True,
                                         fold_operator=False, linearize=False,
                                         batched_ls=False),
    'base_mask': lambda ctx: make_rom_v1(ctx, chol=False, masked_iters=True,
                                         fold_operator=False, linearize=False,
                                         batched_ls=False),
    'v1':        lambda ctx: make_rom_v1(ctx),
    'v1_nofold': lambda ctx: make_rom_v1(ctx, fold_operator=False),
    'v1_nochol': lambda ctx: make_rom_v1(ctx, chol=False),
    'v1_nols':   lambda ctx: make_rom_v1(ctx, batched_ls=False),
    'v1_nolin':  lambda ctx: make_rom_v1(ctx, linearize=False),
    'v1_nodiag': lambda ctx: make_rom_v1(ctx, keep_diag=False),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def sample_test_params(n_traj=10):
    sampler = qmc.LatinHypercube(d=17, seed=9999)
    samples = sampler.random(n=n_traj)
    trajs = []
    for s in samples:
        n_g = int(np.round(1 + 2 * s[0]))
        centers, amplitudes, widths = [], [], []
        for g in range(n_g):
            centers.append((0.15 + 0.70 * s[1 + g * 3],
                            0.15 + 0.70 * s[2 + g * 3],
                            0.15 + 0.70 * s[3 + g * 3]))
            amplitudes.append(1.0 + 9.0 * s[10 + g])
            widths.append(0.05 + 0.15 * s[13 + g])
        kappa = float(np.exp(np.log(0.01) + (np.log(0.5) - np.log(0.01)) * s[16]))
        trajs.append(dict(centers=centers, amplitudes=amplitudes,
                          widths=widths, kappa=kappa))
    return trajs


def fom_key(tp):
    return (round(tp['kappa'], 10),
            tuple(tuple(round(c, 10) for c in ctr) for ctr in tp['centers']),
            tuple(round(a, 10) for a in tp['amplitudes']),
            tuple(round(w, 10) for w in tp['widths']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=sorted(CELLS))
    ap.add_argument('--variants', default='base;v4')
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--eq-cache', default=os.environ.get(
        'EQ_CACHE', '/cluster/tufts/paralab/tawal01/heatopt/eqcache'))
    ap.add_argument('--out', default='')
    ap.add_argument('--batched', action='store_true',
                    help='also measure all 10 parameter instances solved as '
                         'one batch, the many-query setting a ROM is for; '
                         'the FOM is batched identically')
    args = ap.parse_args()

    if jax.default_backend() != 'gpu':
        raise SystemExit(f'FATAL jax_backend={jax.default_backend()} -- refusing '
                         'to publish CPU timings as GPU results')
    _excl_start, _excl_ev = observe_isolation()
    print(f'isolated={_excl_start} evidence={_excl_ev}', flush=True)
    print(f'jax_backend={jax.default_backend()} devices={jax.devices()}',
          flush=True)

    ctx = build(args.cell, args.eq_cache)
    cfg = ctx['cfg']
    fom = make_fom(ctx)
    tps = sample_test_params()

    with open(cfg['fom_cache'], 'rb') as f:
        fom_cache = pickle.load(f)

    roms = {}
    knobs = {}
    for v in args.variants.split(';'):
        spec, _, knob = v.partition('@')
        name, _, mode = spec.partition(':')
        sub = dict(ctx)
        if knob:
            it, _, tol = knob.partition(',')
            sub['cfg'] = dict(cfg, max_iters=int(it), gn_tol=float(tol))
            knobs[v] = dict(max_iters=int(it), gn_tol=float(tol))
        else:
            knobs[v] = dict(max_iters=cfg['max_iters'], gn_tol=cfg['gn_tol'])
        roms[v] = make_rom_e2e(ctx, VARIANTS[name](sub),
                               jit_encoder=(mode != 'eager'))

    out = {v: dict(rel_l2=[], t=[], all_t=[], iters=[]) for v in roms}
    fom_t_all, fom_all_all, kappas, fom_fid = [], [], [], []

    for i, tp in enumerate(tps):
        u0 = ctx['make_gaussian_ic'](tp['centers'], tp['amplitudes'], tp['widths'])
        kap = tp['kappa']
        kappas.append(float(kap))
        ref = jnp.asarray(fom_cache[fom_key(tp)]['U_fom'])[-1]
        nrm = float(jnp.linalg.norm(ref))
        kf = jnp.float32(kap)

        # Interleaved: FOM and ROM alternate inside one loop so clock and
        # thermal drift bias both sides equally.  Measuring them in separate
        # blocks let the device downclock during the slower side and inflated
        # the FOM by up to 5x (see ROUNDS.md, round 1).
        line = f'  [{i+1:2d}] k={kap:.4f}'
        f_times = []
        r_times = {v: [] for v in roms}
        u_fj = None
        r_out = {}
        for _ in range(args.warmup):
            u_fj = jax.block_until_ready(fom(u0, kf))
            for v, rom in roms.items():
                r_out[v] = jax.block_until_ready(rom(u0, kf))
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            u_fj = jax.block_until_ready(fom(u0, kf))
            f_times.append(time.perf_counter() - t0)
            for v, rom in roms.items():
                t0 = time.perf_counter()
                r_out[v] = jax.block_until_ready(rom(u0, kf))
                r_times[v].append(time.perf_counter() - t0)

        t_f = float(np.median(f_times))
        fom_t_all.append(t_f)
        fom_all_all.append(f_times)
        fom_fid.append(float(jnp.linalg.norm(u_fj - ref) / (nrm + 1e-30)))
        line += f' FOM {t_f*1e3:8.2f}ms'
        for v in roms:
            t_r = float(np.median(r_times[v]))
            u_r, it, _ = r_out[v]
            e = float(jnp.linalg.norm(u_r - ref) / (nrm + 1e-12))
            out[v]['rel_l2'].append(e)
            out[v]['t'].append(t_r)
            out[v]['all_t'].append(r_times[v])
            out[v]['iters'].append([int(x) for x in np.asarray(it)])
            line += f' | {v} {t_r*1e3:7.2f}ms {t_f/t_r:6.3f}x e={e:.4e}'
        print(line, flush=True)

    print('\n=== summary ===', flush=True)
    print(f'cell {args.cell}  n_eq={ctx["n_eq"]}  k={cfg["k_dim"]} '
          f'max_iters={cfg["max_iters"]} gn_tol={cfg["gn_tol"]}')
    print(f'FOM jit vs cached reference: max rel dev {max(fom_fid):.2e}')
    print(f'FOM median {np.median(fom_t_all)*1e3:.2f} ms')
    ref_l2 = cfg['retimed_rel_l2']
    summary = dict(cell=args.cell, n_eq=int(ctx['n_eq']), kappas=kappas,
                   fom_times=fom_t_all, fom_all=fom_all_all,
                   fom_fidelity=fom_fid, config=cfg,
                   backend=jax.default_backend(),
                   xla_flags=os.environ.get('XLA_FLAGS', ''),
                   node=os.environ.get('SLURMD_NODENAME'),
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   exclusive_node=_excl_start and observe_isolation()[0],
                   isolation_evidence={'at_start': _excl_ev,
                                       'at_end': observe_isolation()[1]},
                   variants={}, knobs=knobs)
    for v in roms:
        m = float(np.mean(out[v]['rel_l2']))
        ratios = [f / r for f, r in zip(fom_t_all, out[v]['t'])]
        gnmean = float(np.mean([np.mean(x) for x in out[v]['iters']]))
        print(f'  {v:11s} rel_l2 {m:.6e}  (delta vs retimed {abs(m-ref_l2)/ref_l2:.2e})'
              f'  ROM {np.median(out[v]["t"])*1e3:8.2f} ms'
              f'  speedup med {np.median(ratios):.3f}x'
              f'  range {min(ratios):.3f}-{max(ratios):.3f}x  GN {gnmean:.1f}')
        summary['variants'][v] = dict(
            knobs=knobs[v], rel_l2_mean=m, per_traj_rel_l2=out[v]['rel_l2'],
            rom_times=out[v]['t'], rom_all=out[v]['all_t'],
            gn_iters=out[v]['iters'],
            speedup_median=float(np.median(ratios)),
            speedup_range=[float(min(ratios)), float(max(ratios))],
            per_traj_speedup=[float(x) for x in ratios])

    if args.batched:
        print('\n=== batched over all 10 parameter instances ===', flush=True)
        U0 = jnp.stack([ctx['make_gaussian_ic'](t['centers'], t['amplitudes'],
                                                t['widths']) for t in tps])
        KP = jnp.asarray([t['kappa'] for t in tps], jnp.float32)
        REF = jnp.stack([jnp.asarray(fom_cache[fom_key(t)]['U_fom'])[-1]
                         for t in tps])
        nb = len(tps)
        fom_b = jax.jit(jax.vmap(fom, in_axes=(0, 0)))
        _, tfb, _ = _bench.timed_median(fom_b, U0, KP,
                                        repeats=args.repeats,
                                        warmup=args.warmup)
        print(f'  FOM batched : {tfb*1e3:8.2f} ms total, '
              f'{tfb/nb*1e3:7.2f} ms per instance')
        summary['batched'] = dict(n=nb, fom_total_s=tfb, variants={})
        for v, rom in roms.items():
            rb = jax.jit(jax.vmap(rom, in_axes=(0, 0)))
            try:
                (ub, _, _), trb, _ = _bench.timed_median(
                    rb, U0, KP, repeats=args.repeats, warmup=args.warmup)
            except Exception as exc:
                print(f'  {v:11s}: batching failed: {exc!r}')
                continue
            eb = float(jnp.mean(jnp.linalg.norm(ub - REF, axis=1)
                                / jnp.linalg.norm(REF, axis=1)))
            print(f'  {v:11s}: {trb*1e3:8.2f} ms total, '
                  f'{trb/nb*1e3:7.2f} ms per instance, '
                  f'speedup {tfb/trb:6.3f}x, rel_l2 {eb:.6e}')
            summary['batched']['variants'][v] = dict(
                rom_total_s=trb, speedup=float(tfb / trb), rel_l2_mean=eb)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
