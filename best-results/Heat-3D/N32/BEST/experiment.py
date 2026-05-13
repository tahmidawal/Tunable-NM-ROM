"""
nm_rom_heat_sweep.py
--------------------
Config-driven Heat-3D NM-ROM (joint z,s LM-GN + EQ hyper-reduction).

Loads a checkpoint trained by train_heat_sweep.py and benchmarks the ROM
against the FOM across a small LHS test set (10 trajectories).  All knobs
swept by the grid are CLI args.  Writes:
  - results.tsv   (one tab-separated row with summary metrics)
  - plots/*.png   (error-over-time, energy decay, speedup, slices)
  - nm_rom.log    (captured stdout)

FOM times are cached per (kappa, IC) key in <outdir>/fom_cache_{N}.pkl to
make speedups comparable across different GPU nodes.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import flax.linen as nn
import numpy as np
from scipy.optimize import nnls
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='heat3d_rom')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--ckpt-path', default=None,
                    help='default: <outdir>/checkpoint_vitcp.pkl')
    # ROM / EQ knobs
    ap.add_argument('--max-iters', type=int, default=10)
    ap.add_argument('--gn-tol', type=float, default=1e-3)
    ap.add_argument('--cg-tol', type=float, default=1e-6)
    ap.add_argument('--cg-iters', type=int, default=1000)
    ap.add_argument('--eq-mode', choices=['nnls', 'full'], default='nnls')
    ap.add_argument('--n-eq-samples', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    # Accept but don't use (shared config.json):
    ap.add_argument('--k-dim', type=int, default=0)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=0)
    ap.add_argument('--patch-size', type=int, default=0)
    ap.add_argument('--embed-dim', type=int, default=0)
    ap.add_argument('--num-heads', type=int, default=0)
    ap.add_argument('--num-enc-layers', type=int, default=0)
    ap.add_argument('--num-epochs', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=0)
    ap.add_argument('--peak-lr', type=float, default=0.)
    ap.add_argument('--weight-decay', type=float, default=0.)
    ap.add_argument('--n-train', type=int, default=0)
    ap.add_argument('--n-val', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / 'plots'; plot_dir.mkdir(exist_ok=True)

    log_f = open(outdir / 'nm_rom.log', 'w')
    class Tee:
        def write(self, m):
            sys.__stdout__.write(m); log_f.write(m); log_f.flush()
        def flush(self):
            sys.__stdout__.flush(); log_f.flush()
    sys.stdout = Tee()

    N = args.N
    num_nodes = N ** 3
    L = 1.0
    dx = L / (N - 1)
    dt = 0.005
    NUM_STEPS = 50
    AMP_EPS = 1e-6
    S_MIN = jnp.float32(1e-10)
    GN_MAX_ITERS = args.max_iters
    GN_REL_TOL = args.gn_tol

    print(f"[{args.label}] {N}^3 = {num_nodes:,} DOF | dt={dt} | T={dt*NUM_STEPS:.3f}s")
    print(f"JAX devices: {jax.devices()}")
    print(f"EQ: mode={args.eq_mode}  n_samples={args.n_eq_samples}  "
          f"max_iters={GN_MAX_ITERS}  gn_tol={GN_REL_TOL}")

    x_sp = jnp.linspace(0, L, N); y_sp = jnp.linspace(0, L, N); z_sp = jnp.linspace(0, L, N)
    X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

    def K_op_3d(u_flat):
        u = u_flat.reshape((N, N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            (6 * u[1:-1, 1:-1, 1:-1]
             - u[0:-2, 1:-1, 1:-1] - u[2:, 1:-1, 1:-1]
             - u[1:-1, 0:-2, 1:-1] - u[1:-1, 2:, 1:-1]
             - u[1:-1, 1:-1, 0:-2] - u[1:-1, 1:-1, 2:]) / dx ** 2
        )
        out = out.at[0, :, :].set(u[0, :, :]); out = out.at[-1, :, :].set(u[-1, :, :])
        out = out.at[:, 0, :].set(u[:, 0, :]); out = out.at[:, -1, :].set(u[:, -1, :])
        out = out.at[:, :, 0].set(u[:, :, 0]); out = out.at[:, :, -1].set(u[:, :, -1])
        return out.flatten()

    def implicit_op(u_flat, kappa):
        return u_flat + dt * kappa * K_op_3d(u_flat)

    def run_fom(u0_flat, kappa, steps):
        snapshots = [u0_flat]
        u = u0_flat
        op = lambda v: implicit_op(v, kappa)
        for _ in range(steps):
            u, _ = jax_linalg.cg(op, u, x0=u, tol=args.cg_tol,
                                  maxiter=args.cg_iters if args.cg_iters > 0 else 1000)
            snapshots.append(u)
        return jnp.stack(snapshots)

    def make_gaussian_ic(centers, amplitudes, widths):
        u = jnp.zeros((N, N, N))
        for (cx, cy, cz), A, sigma in zip(centers, amplitudes, widths):
            u = u + A * jnp.exp(
                -((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * sigma ** 2)
            )
        u = u.at[0, :, :].set(0.).at[-1, :, :].set(0.)
        u = u.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        u = u.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        return u.flatten()

    mask_3d = jnp.ones((N, N, N))
    mask_3d = mask_3d.at[0, :, :].set(0.).at[-1, :, :].set(0.)
    mask_3d = mask_3d.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
    mask_3d = mask_3d.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
    mask = mask_3d.flatten()
    u_g = jnp.zeros(num_nodes)

    def normalise(u_flat):
        scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
        return u_flat / scale, scale

    # Model (same defs as training)
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
            x = x + h
            return x

    class ViTEncoder(nn.Module):
        latent_dim: int
        grid_n: int
        patch_size: int = 8
        embed_dim: int = 64
        num_heads: int = 4
        num_enc_layers: int = 4

        def setup(self):
            n_per_side = self.grid_n // self.patch_size
            self._n_per_side = n_per_side
            self._num_patches = n_per_side ** 3
            self._patch_dim = self.patch_size ** 3
            self.patch_embed = nn.Dense(self.embed_dim)
            self.enc_pos = self.param(
                'enc_pos', nn.initializers.normal(stddev=0.02),
                (self._num_patches, self.embed_dim))
            self.enc_blocks = [
                TransformerBlock(self.embed_dim, self.num_heads)
                for _ in range(self.num_enc_layers)
            ]
            self.enc_norm = nn.LayerNorm()
            self.enc_proj = nn.Dense(self.latent_dim)

        def _patchify(self, u_flat):
            n = self._n_per_side; p = self.patch_size
            return (u_flat
                    .reshape(n, p, n, p, n, p)
                    .transpose(0, 2, 4, 1, 3, 5)
                    .reshape(self._num_patches, self._patch_dim))

        def __call__(self, u_flat):
            x = self._patchify(u_flat)
            x = self.patch_embed(x) + self.enc_pos
            for block in self.enc_blocks:
                x = block(x)
            x = self.enc_norm(x)
            z = x.mean(axis=0)
            return self.enc_proj(z)

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
            h_lin = self.W_direct(z)
            h = h_lin + h_nl
            u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
            return u_3d.flatten() + self.bias

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

        def encode(self, u_flat, training=False):
            u_norm, scale = normalise(u_flat)
            z = self.encoder(u_norm)
            return z, scale
        def decode(self, z, scale):
            return self.decoder(z) * scale
        def decode_normalised(self, z):
            return self.decoder(z)
        def __call__(self, u_flat, training=False):
            z, scale = self.encode(u_flat, training=training)
            return self.decode(z, scale)

    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else outdir / 'checkpoint_vitcp.pkl'
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    cfg = ckpt['model_cfg']
    meta = ckpt['train_meta']
    k_dim = cfg['latent_dim']
    model = ViTCPAutoencoder(**cfg)
    print(f"   latent_dim={k_dim}  rank={cfg['rank']}  grid={cfg['grid_size']}^3")

    def encode(u_flat):
        return model.apply({'params': params}, u_flat,
                           training=False, method=model.encode)
    def decode_normalised(z):
        return model.apply({'params': params}, z, method=model.decode_normalised)
    def decode_full(z, scale):
        return model.apply({'params': params}, z, scale, method=model.decode)
    def constrained_decode_normalised(z):
        return mask * decode_normalised(z) + u_g

    # EQ offline
    N_EQ_SAMPLES = args.n_eq_samples
    traj_kappas = meta['traj_kappas']

    print(f"Loading training snapshots for EQ from {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data_cache = pickle.load(f)
    all_snapshots_eq = data_cache['all_snapshots']
    train_params = data_cache['train_params']

    all_pairs = []
    for i, traj in enumerate(all_snapshots_eq):
        kap = traj_kappas[i]
        for step in range(NUM_STEPS):
            all_pairs.append((traj[step + 1], traj[step], kap))

    rng_eq = np.random.default_rng(seed=123 + args.seed)
    eq_indices_sample = rng_eq.choice(len(all_pairs),
                                       size=min(N_EQ_SAMPLES, len(all_pairs)),
                                       replace=False)
    eq_snapshot_pairs = [all_pairs[i] for i in eq_indices_sample]
    print(f"   EQ pairs: {len(eq_snapshot_pairs)} from {len(all_pairs)}")

    def compute_eq_weights():
        @jax.jit
        def get_integrand_heat(z_next, u_prev_norm, kappa):
            def cd(z):
                return constrained_decode_normalised(z)
            u_pred = cd(z_next)
            R = u_pred - u_prev_norm + dt * kappa * K_op_3d(u_pred)
            J_D = jax.jacfwd(cd)(z_next)
            return J_D.T * R[None, :]

        G_list = []
        for idx, (u_next, u_prev, kap) in enumerate(eq_snapshot_pairs):
            z_next, _ = encode(u_next)
            u_prev_norm, _ = normalise(u_prev)
            G_list.append(get_integrand_heat(z_next, u_prev_norm, jnp.float32(kap)))

        G_train = jnp.concatenate(G_list, axis=0)
        G_np = np.array(G_train)
        G_np[:, np.array(mask) == 0] = 0.0
        b_np = np.sum(G_np, axis=1)
        w_eq, _ = nnls(G_np, b_np, maxiter=4 * G_np.shape[1])
        eq_idx = np.where(w_eq > 1e-10)[0]
        eq_w = w_eq[eq_idx]
        return eq_idx, eq_w

    if args.eq_mode == 'full':
        # Use all interior nodes with unit weights (no reduction)
        interior_flat = np.where(np.array(mask) > 0)[0]
        eq_indices = interior_flat
        eq_weights = np.ones(len(interior_flat), dtype=np.float32)
        print(f"   eq_mode=full: using all {len(eq_indices)} interior nodes")
    else:
        print(f"   Computing EQ weights (nnls, n_samples={N_EQ_SAMPLES})...")
        eq_indices, eq_weights = compute_eq_weights()
        print(f"   EQ nodes: {len(eq_indices)} / {num_nodes} "
              f"({100 * len(eq_indices) / num_nodes:.4f}%)")

    eq_idx_jnp = jnp.array(eq_indices)
    eq_w_jnp = jnp.array(eq_weights)
    n_eq = len(eq_indices)

    # Precompute V_eq
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
    print(f"   V_eq: {V_eq.shape}")

    def make_heat_latent_solver(p, V_eq_, b_sp, mask_sp_, ug_sp_,
                                 eq_w, latent_dim, dx_, dt_):
        dec = p['decoder']
        W1k, b1k = dec['W1']['kernel'], dec['W1']['bias']
        W2k, b2k = dec['W2']['kernel'], dec['W2']['bias']
        Wrk, brk = dec['W_rank']['kernel'], dec['W_rank']['bias']
        Wdk, bdk = dec['W_direct']['kernel'], dec['W_direct']['bias']

        def _mlp_body(z):
            h_nl = nn.swish(z @ W1k + b1k)
            h_nl = nn.swish(h_nl @ W2k + b2k)
            h_nl = h_nl @ Wrk + brk
            h_lin = z @ Wdk + bdk
            return h_lin + h_nl

        def _f_norm(z, kappa_):
            h = _mlp_body(z)
            u_st = (mask_sp_ * (h @ V_eq_ + b_sp) + ug_sp_).reshape(-1, 7)
            u_centers = u_st[:, 0]
            lap = (6 * u_st[:, 0]
                   - u_st[:, 1] - u_st[:, 2]
                   - u_st[:, 3] - u_st[:, 4]
                   - u_st[:, 5] - u_st[:, 6]) / dx_ ** 2
            return u_centers + dt_ * kappa_ * lap

        def make_step(kappa_f32):
            @jax.jit
            def solve_step(z_init, s_init, u_prev_phys_eq_):
                fn0 = _f_norm(z_init, kappa_f32)
                R0 = s_init * fn0 - u_prev_phys_eq_
                J0 = jax.jacfwd(lambda zz: _f_norm(zz, kappa_f32))(z_init)
                WR0 = eq_w * R0
                g_z0 = s_init * (J0.T @ WR0)
                g_s0 = jnp.dot(fn0, WR0)
                gnorm0 = jnp.sqrt(jnp.dot(g_z0, g_z0) + g_s0 ** 2)
                gnorm0 = jnp.maximum(gnorm0, 1e-30)

                def _body(carry):
                    z, s, _, itr = carry
                    fn = _f_norm(z, kappa_f32)
                    R = s * fn - u_prev_phys_eq_
                    J = jax.jacfwd(lambda zz: _f_norm(zz, kappa_f32))(z)
                    Wfn = eq_w * fn
                    WR = eq_w * R
                    WJ = eq_w[:, None] * J
                    JtWJ = J.T @ WJ
                    JtWfn = J.T @ Wfn
                    fnWfn = jnp.dot(fn, Wfn)
                    JtWR = J.T @ WR
                    fnWR = jnp.dot(fn, WR)
                    H_zz = s ** 2 * JtWJ
                    H_zs = s * JtWfn
                    H_ss = fnWfn
                    g_z = s * JtWR
                    g_s = fnWR
                    gnorm = jnp.sqrt(jnp.dot(g_z, g_z) + g_s ** 2)
                    H_aug = jnp.block([
                        [H_zz, H_zs[:, None]],
                        [H_zs[None, :], H_ss[None, None]],
                    ])
                    g_aug = jnp.append(g_z, g_s)
                    lam = jnp.maximum(1e-3 * jnp.trace(H_aug) / (latent_dim + 1), 1e-8)
                    delta = jnp.linalg.solve(H_aug + lam * jnp.eye(latent_dim + 1), -g_aug)
                    dz = delta[:latent_dim]; ds = delta[latent_dim]
                    f0 = jnp.dot(WR, R)
                    def _f(alpha):
                        fn_t = _f_norm(z + alpha * dz, kappa_f32)
                        R_t = (s + alpha * ds) * fn_t - u_prev_phys_eq_
                        return jnp.dot(eq_w * R_t, R_t)
                    f1, f2, f3, f4 = _f(1.), _f(.5), _f(.25), _f(.125)
                    step = jnp.where(f1 < f0, 1.0,
                            jnp.where(f2 < f0, 0.5,
                             jnp.where(f3 < f0, 0.25,
                              jnp.where(f4 < f0, 0.125, 0.0))))
                    z_new = z + step * dz
                    s_new = jnp.maximum(s + step * ds, S_MIN)
                    return z_new, s_new, gnorm, itr + 1

                def _cond(carry):
                    _, _, gnorm, itr = carry
                    return (gnorm > GN_REL_TOL * gnorm0) & (itr < GN_MAX_ITERS)

                init = (z_init, s_init,
                        jnp.array(jnp.inf, dtype=jnp.float32),
                        jnp.array(0, dtype=jnp.int32))
                z_f, s_f, gnorm_f, itr_f = jax.lax.while_loop(_cond, _body, init)
                fn_f = _f_norm(z_f, kappa_f32)
                R_f = s_f * fn_f - u_prev_phys_eq_
                res_norm = jnp.sqrt(jnp.dot(eq_w * R_f, R_f))
                return z_f, s_f, gnorm_f, itr_f, res_norm

            return solve_step

        return make_step

    make_step_factory = make_heat_latent_solver(
        params, V_eq, b_sparse, mask_sp, ug_sp,
        eq_w_jnp, k_dim, dx, dt,
    )

    def run_rom(u0_flat, kappa, steps):
        kappa_f32 = jnp.float32(kappa)
        solve_step = make_step_factory(kappa_f32)
        z, s = encode(u0_flat)
        u_cur = decode_full(z, s)
        snapshots = [u_cur]
        gn_iters = []; res_norms = []
        for _ in range(steps):
            u_prev_phys_eq = u_cur[eq_idx_jnp]
            z, s, gnorm, n_iters, res_norm = solve_step(z, s, u_prev_phys_eq)
            u_cur = decode_full(z, s)
            snapshots.append(u_cur)
            gn_iters.append(int(n_iters))
            res_norms.append(float(res_norm))
        jax.block_until_ready(u_cur)
        return jnp.stack(snapshots), gn_iters, res_norms

    # Warm-up
    print("Warming up JIT...")
    _tp = train_params[0]
    _u0 = make_gaussian_ic(_tp['centers'], _tp['amplitudes'], _tp['widths'])
    _Uw, _, _ = run_rom(_u0, float(_tp['kappa']), NUM_STEPS); _Uw[-1].block_until_ready()
    _Uw, _, _ = run_rom(_u0, float(_tp['kappa']), NUM_STEPS); _Uw[-1].block_until_ready()
    print("   warm-up complete")

    def sample_test_params(rng, n_traj):
        sampler = qmc.LatinHypercube(d=17, seed=rng)
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

    test_params = sample_test_params(rng=9999, n_traj=10)

    FOM_CACHE_PATH = outdir / f'fom_cache_{N}.pkl'
    if FOM_CACHE_PATH.exists():
        with open(FOM_CACHE_PATH, 'rb') as f:
            fom_cache = pickle.load(f)
        print(f"   FOM cache: {len(fom_cache)} entries from {FOM_CACHE_PATH}")
    else:
        fom_cache = {}

    def _fom_key(tp):
        return (round(tp['kappa'], 10),
                tuple(tuple(round(c, 10) for c in ctr) for ctr in tp['centers']),
                tuple(round(a, 10) for a in tp['amplitudes']),
                tuple(round(w, 10) for w in tp['widths']))

    fom_times, rom_times, final_errors = [], [], []
    energy_fom_list, energy_rom_list = [], []
    stored = {}

    print("\n--- Benchmark ---")
    for i, tp in enumerate(test_params):
        u0 = make_gaussian_ic(tp['centers'], tp['amplitudes'], tp['widths'])
        kap = tp['kappa']; n_gauss = len(tp['centers'])
        _key = _fom_key(tp)
        if _key in fom_cache:
            U_fom = jnp.asarray(fom_cache[_key]['U_fom'])
            fom_t = float(fom_cache[_key]['fom_time'])
        else:
            t0 = time.perf_counter()
            U_fom = run_fom(u0, kap, NUM_STEPS); U_fom[-1].block_until_ready()
            fom_t = time.perf_counter() - t0
            fom_cache[_key] = dict(U_fom=np.array(U_fom), fom_time=fom_t)
            with open(FOM_CACHE_PATH, 'wb') as f:
                pickle.dump(fom_cache, f)
        fom_times.append(fom_t)

        t0 = time.perf_counter()
        U_rom, gn_iters, res_norms = run_rom(u0, kap, NUM_STEPS)
        rom_t = time.perf_counter() - t0
        rom_times.append(rom_t)

        norms_fom = jnp.linalg.norm(U_fom, axis=1)
        err_t = jnp.linalg.norm(U_rom - U_fom, axis=1) / (norms_fom + 1e-12)
        final_err = float(err_t[-1])
        final_errors.append(final_err)
        energy_fom_list.append(np.array(norms_fom))
        energy_rom_list.append(np.array(jnp.linalg.norm(U_rom, axis=1)))

        avg_gn = float(np.mean(gn_iters))
        print(f"  [{i+1:2d}/10] κ={kap:.4f} M={n_gauss} | "
              f"FOM {fom_t:.3f}s | ROM {rom_t:.3f}s | "
              f"RelL2(T) {final_err:.3e} | GN avg {avg_gn:.1f}")
        stored[i] = dict(U_fom=np.array(U_fom), U_rom=np.array(U_rom),
                         err_t=np.array(err_t), kappa=kap,
                         gn_iters=gn_iters, res_norms=res_norms, n_gauss=n_gauss)

    avg_fom = float(np.mean(fom_times))
    avg_rom = float(np.mean(rom_times))
    per_case_sp = [f / r for f, r in zip(fom_times, rom_times)]
    med_sp = float(np.median(per_case_sp))
    avg_err = float(np.mean(final_errors))

    print(f"\n{'='*60}")
    print(f"   {args.label}  Heat NM-ROM benchmark")
    print(f"{'='*60}")
    print(f"  Grid            : {N}³ = {num_nodes:,} DOF")
    print(f"  Latent / rank   : k={k_dim}  rank={cfg['rank']}")
    print(f"  EQ nodes        : {n_eq} / {num_nodes} ({100*n_eq/num_nodes:.4f}%)")
    print(f"  Avg FOM         : {avg_fom:.4f} s")
    print(f"  Avg ROM         : {avg_rom:.4f} s")
    print(f"  Median speedup  : {med_sp:.2f}×")
    print(f"  Avg rel-L2(T)   : {avg_err:.4e}")
    print(f"{'='*60}")

    # Write results.tsv (mirror Poisson-2D columns as closely as possible)
    header = ["label", "N", "k_dim", "rank", "eq_mode", "n_eq", "n_eq_samples",
              "max_iters", "gn_tol", "cg_tol", "speedup", "rel_l2",
              "fom_time", "rom_time"]
    row = [args.label, str(N), str(k_dim), str(cfg['rank']), args.eq_mode,
           str(n_eq), str(N_EQ_SAMPLES), str(GN_MAX_ITERS),
           f"{GN_REL_TOL:.0e}", f"{args.cg_tol:.0e}",
           f"{med_sp:.4f}", f"{avg_err:.4e}",
           f"{avg_fom:.5f}", f"{avg_rom:.5f}"]
    with open(outdir / 'results.tsv', 'w') as f:
        f.write('\t'.join(header) + '\n')
        f.write('\t'.join(row) + '\n')
    print(f"Wrote {outdir / 'results.tsv'}")

    # Plots
    t_axis = np.arange(NUM_STEPS + 1) * dt
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in stored.items():
        ax.semilogy(t_axis, s['err_t'],
                    label=f'κ={s["kappa"]:.3f} M={s["n_gauss"]}', lw=2)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Rel L2 (log)')
    ax.set_title(f'{args.label} — error vs time')
    ax.legend(fontsize=8); ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(plot_dir / 'error_over_time.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(1, 11), per_case_sp, color='#2ca02c', alpha=0.85, edgecolor='black')
    ax.axhline(1.0, color='red', ls='--', lw=1.5)
    ax.axhline(med_sp, color='orange', ls='--', lw=1.5, label=f'Median {med_sp:.1f}×')
    ax.set_xlabel('Test trajectory'); ax.set_ylabel('Speedup')
    ax.set_title(f'{args.label} — speedup')
    ax.legend(); ax.grid(True, axis='y', ls='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(plot_dir / 'speedup.png', dpi=150); plt.close()

    mid = N // 2
    time_checkpoints = [0, NUM_STEPS // 2, NUM_STEPS]
    for case_idx in range(min(3, 10)):
        s = stored[case_idx]; kap = s['kappa']
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for col, t_idx in enumerate(time_checkpoints):
            u_f = s['U_fom'][t_idx].reshape(N, N, N)[:, :, mid]
            u_r = s['U_rom'][t_idx].reshape(N, N, N)[:, :, mid]
            vmax = max(float(u_f.max()), 1e-8)
            kw = dict(origin='lower', cmap='magma', vmin=0, vmax=vmax,
                      extent=[0, L, 0, L], aspect='auto')
            axes[0, col].imshow(u_f.T, **kw)
            axes[0, col].set_title(f'FOM t={t_idx*dt:.3f}s')
            axes[1, col].imshow(u_r.T, **kw)
            err_c = float(np.linalg.norm(u_f - u_r) / (np.linalg.norm(u_f) + 1e-12))
            axes[1, col].set_title(f'ROM e={err_c:.2e}')
        fig.suptitle(f'{args.label} z=mid | κ={kap:.4f} M={s["n_gauss"]}')
        plt.tight_layout()
        plt.savefig(plot_dir / f'slice_case{case_idx+1}.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    print("\n=== NM-ROM Complete ===")


if __name__ == '__main__':
    main()


# =============================================================================
# AUTORESEARCH ENTRY POINT
# =============================================================================
# Shared training data (do NOT regenerate — reuse existing pkl files):
#   N=32:  /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Heat-3D/shared_data/training_data_32.pkl
#   N=64:  /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Heat-3D/shared_data/training_data_64.pkl
#   N=128: /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Heat-3D/shared_data/training_data_128.pkl
# Each pkl has keys: U_train, U_val, all_snapshots, val_snapshots,
#   train_params, val_params, traj_kappas, val_kappas, traj_starts, grid_config

SHARED_DATA = {
    32:  '/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Heat-3D/shared_data/training_data_32.pkl',
    64:  '/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Heat-3D/shared_data/training_data_64.pkl',
    128: '/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Heat-3D/shared_data/training_data_128.pkl',
}


def run_and_benchmark():
    import sys, pickle, time
    from pathlib import Path
    import jax, jax.numpy as jnp, jax.scipy.sparse.linalg as jax_linalg
    import flax.linen as nn, optax, numpy as np
    from scipy.optimize import nnls
    from scipy.stats import qmc

    # ── Hyperparameters (agent modifies this block) ────────────────────────
    N              = 32
    k_dim          = 32
    rank           = 512
    hidden_dim     = 256
    patch_size     = 8
    embed_dim      = 96
    num_heads      = 4
    num_enc_layers = 6
    num_epochs     = 80_000
    batch_size     = 32
    peak_lr        = 2e-3
    weight_decay   = 5e-4
    max_iters      = 3
    gn_tol         = 1e-3
    cg_tol         = 1e-6
    cg_iters       = 1000
    eq_mode        = 'nnls'   # 'nnls' or 'full'
    n_eq_samples   = 16       # Run10 setting (n_eq=512)
    outdir         = Path('/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Autoresearch/Heat-3D/runs/N32_fixed_run10')
    # AE reuse from Run10 ckpt (k=32 80k val=9.29e-2 spd=8.53x); set explicit path
    # because N32 ckpt-loading is gated on `ckpt_path is not None`, unlike N64.
    ckpt_path      = outdir / 'checkpoint_vitcp.pkl'
    seed           = 0
    # ───────────────────────────────────────────────────────────────────────

    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / 'plots'; plot_dir.mkdir(exist_ok=True)

    log_f = open(outdir / 'run.log', 'w')
    class Tee:
        def write(self, m): sys.__stdout__.write(m); log_f.write(m); log_f.flush()
        def flush(self): sys.__stdout__.flush(); log_f.flush()
    sys.stdout = Tee()

    num_nodes = N ** 3
    L = 1.0
    dx = L / (N - 1)
    dt = 0.005
    NUM_STEPS = 50
    AMP_EPS = 1e-6
    S_MIN = jnp.float32(1e-10)

    print(f"JAX devices: {jax.devices()}")
    print(f"Grid {N}^3 = {num_nodes:,} | k={k_dim} rank={rank} epochs={num_epochs}")

    x_sp = jnp.linspace(0, L, N); y_sp = jnp.linspace(0, L, N); z_sp = jnp.linspace(0, L, N)
    X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

    def K_op_3d(u_flat):
        u = u_flat.reshape((N, N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            (6*u[1:-1,1:-1,1:-1] - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
             - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
             - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2)
        out = out.at[0,:,:].set(u[0,:,:]); out = out.at[-1,:,:].set(u[-1,:,:])
        out = out.at[:,0,:].set(u[:,0,:]); out = out.at[:,-1,:].set(u[:,-1,:])
        out = out.at[:,:,0].set(u[:,:,0]); out = out.at[:,:,-1].set(u[:,:,-1])
        return out.flatten()

    def implicit_op(u_flat, kappa):
        return u_flat + dt * kappa * K_op_3d(u_flat)

    mask_3d = jnp.ones((N, N, N))
    mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    mask = mask_3d.flatten()
    u_g = jnp.zeros(num_nodes)

    def normalise(u_flat):
        scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
        return u_flat / scale, scale

    # ── Model ─────────────────────────────────────────────────────────────
    class TransformerBlock(nn.Module):
        embed_dim: int; num_heads: int; mlp_ratio: float = 4.0
        @nn.compact
        def __call__(self, x):
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(h, h)
            x = x + h; h = nn.LayerNorm()(x)
            h = nn.Dense(int(self.embed_dim * self.mlp_ratio))(h)
            h = nn.gelu(h); h = nn.Dense(self.embed_dim)(h)
            return x + h

    class ViTEncoder(nn.Module):
        latent_dim: int; grid_n: int; patch_size: int = 8; embed_dim: int = 64
        num_heads: int = 4; num_enc_layers: int = 4
        def setup(self):
            nps = self.grid_n // self.patch_size
            self._nps = nps; self._np = nps**3; self._pd = self.patch_size**3
            self.patch_embed = nn.Dense(self.embed_dim)
            self.enc_pos = self.param('enc_pos', nn.initializers.normal(0.02), (self._np, self.embed_dim))
            self.enc_blocks = [TransformerBlock(self.embed_dim, self.num_heads) for _ in range(self.num_enc_layers)]
            self.enc_norm = nn.LayerNorm(); self.enc_proj = nn.Dense(self.latent_dim)
        def _patchify(self, u):
            n = self._nps; p = self.patch_size
            return u.reshape(n,p,n,p,n,p).transpose(0,2,4,1,3,5).reshape(self._np, self._pd)
        def __call__(self, u):
            x = self.patch_embed(self._patchify(u)) + self.enc_pos
            for b in self.enc_blocks: x = b(x)
            return self.enc_proj(self.enc_norm(x).mean(axis=0))

    class LinearCPDecoder(nn.Module):
        latent_dim: int; rank: int = 512; grid_size: int = 32; hidden_dim: int = 256
        def setup(self):
            self.W1 = nn.Dense(self.hidden_dim); self.W2 = nn.Dense(self.hidden_dim)
            self.W_rank = nn.Dense(self.rank); self.W_direct = nn.Dense(self.rank)
            init = nn.initializers.normal(0.01); Ng = self.grid_size
            self.W_x = self.param('W_x', init, (self.rank, Ng))
            self.W_y = self.param('W_y', init, (self.rank, Ng))
            self.W_z = self.param('W_z', init, (self.rank, Ng))
            self.bias = self.param('bias', nn.initializers.zeros, ())
        def __call__(self, z):
            h_nl = nn.swish(self.W1(z)); h_nl = nn.swish(self.W2(h_nl)); h_nl = self.W_rank(h_nl)
            h = self.W_direct(z) + h_nl
            return jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z).flatten() + self.bias

    class ViTCPAutoencoder(nn.Module):
        latent_dim: int; grid_size: int = 32; patch_size: int = 8; embed_dim: int = 64
        num_heads: int = 4; num_enc_layers: int = 4; rank: int = 512; hidden_dim: int = 256
        def setup(self):
            self.encoder = ViTEncoder(latent_dim=self.latent_dim, grid_n=self.grid_size,
                patch_size=self.patch_size, embed_dim=self.embed_dim,
                num_heads=self.num_heads, num_enc_layers=self.num_enc_layers)
            self.decoder = LinearCPDecoder(latent_dim=self.latent_dim, rank=self.rank,
                grid_size=self.grid_size, hidden_dim=self.hidden_dim)
        def encode(self, u, training=False):
            u_norm, scale = normalise(u); return self.encoder(u_norm), scale
        def decode(self, z, scale): return self.decoder(z) * scale
        def decode_normalised(self, z): return self.decoder(z)
        def __call__(self, u, training=False):
            z, scale = self.encode(u, training=training); return self.decode(z, scale)

    # ── Load data ─────────────────────────────────────────────────────────
    data_path = Path(SHARED_DATA[N])
    print(f"Loading data: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    U_train = np.asarray(data['U_train'])
    U_val   = np.asarray(data['U_val'])
    n_train_data = len(U_train); n_val_data = len(U_val)
    print(f"   train {U_train.shape}  val {U_val.shape}")

    # ── Train or load checkpoint ───────────────────────────────────────────
    model = ViTCPAutoencoder(latent_dim=k_dim, grid_size=N, patch_size=patch_size,
        embed_dim=embed_dim, num_heads=num_heads, num_enc_layers=num_enc_layers,
        rank=rank, hidden_dim=hidden_dim)

    if ckpt_path is not None and ckpt_path.exists():
        print(f"AE reuse: loading checkpoint from {ckpt_path}")
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        params = ckpt['params']
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Model: {n_params:,} params (loaded)")
        training_seconds = 0.0
    else:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.asarray(U_train[0]), training=False)['params']
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Model: {n_params:,} params")

        REL_EPS = 1e-6
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0., peak_value=peak_lr,
            warmup_steps=min(2000, num_epochs // 10),
            decay_steps=num_epochs, end_value=1e-6)
        tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        opt_state = tx.init(params)

        def augment_3d(u_flat, aug_key):
            u3 = u_flat.reshape(N, N, N)
            k1, k2, k3, k4, k5, k6 = jax.random.split(aug_key, 6)
            u3 = jax.lax.cond(jax.random.uniform(k1) > 0.5, lambda x: jnp.flip(x, 0), lambda x: x, u3)
            u3 = jax.lax.cond(jax.random.uniform(k2) > 0.5, lambda x: jnp.flip(x, 1), lambda x: x, u3)
            u3 = jax.lax.cond(jax.random.uniform(k3) > 0.5, lambda x: jnp.flip(x, 2), lambda x: x, u3)
            return u3.flatten()

        @jax.jit
        def train_step(p, opt_st, batch, aug_key):
            aug_keys = jax.random.split(aug_key, batch.shape[0])
            batch_aug = jax.vmap(augment_3d)(batch, aug_keys)
            def loss_fn(w):
                preds = jax.vmap(lambda u: model.apply({'params': w}, u, training=False))(batch_aug)
                diffs = batch_aug - preds
                return jnp.mean(jnp.sum(diffs**2, axis=1) / (jnp.sum(batch_aug**2, axis=1) + REL_EPS))
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, new_opt = tx.update(grads, opt_st, p)
            return optax.apply_updates(p, updates), new_opt, loss

        LOG_EVERY = max(100, num_epochs // 100)
        print(f"Training {num_epochs} epochs batch={batch_size}")
        t_train_start = time.perf_counter()
        jkey = jax.random.PRNGKey(2 + seed); np_rng = np.random.default_rng(2 + seed)
        best_val = float('inf'); best_params = params

        for epoch in range(num_epochs + 1):
            jkey, ak = jax.random.split(jkey)
            idx = np_rng.choice(n_train_data, size=batch_size, replace=False)
            params, opt_state, loss = train_step(params, opt_state, jnp.asarray(U_train[idx]), ak)
            if epoch % LOG_EVERY == 0:
                v_sum = 0.0
                for s in range(0, n_val_data, 256):
                    vb = jnp.asarray(U_val[s:s+256])
                    preds = jax.vmap(lambda u: model.apply({'params': params}, u, training=False))(vb)
                    diffs = vb - preds
                    v_sum += float(jnp.mean(jnp.sum(diffs**2,1)/(jnp.sum(vb**2,1)+REL_EPS))) * len(vb)
                v_loss = v_sum / n_val_data
                print(f"  Epoch {epoch:6d} | val {v_loss:.4e} | {time.perf_counter()-t_train_start:.0f}s")
                if v_loss < best_val: best_val = v_loss; best_params = params

        params = best_params
        training_seconds = time.perf_counter() - t_train_start
        print(f"Best val: {best_val:.4e}  ({training_seconds:.0f}s)")

        ckpt = {
            'params': params,
            'model_cfg': dict(latent_dim=k_dim, grid_size=N, patch_size=patch_size,
                embed_dim=embed_dim, num_heads=num_heads, num_enc_layers=num_enc_layers,
                rank=rank, hidden_dim=hidden_dim),
            'train_meta': dict(n_train=n_train_data, num_steps=NUM_STEPS, dt=dt,
                traj_kappas=data['traj_kappas'], val_kappas=data['val_kappas'],
                traj_starts=data['traj_starts']),
        }
        with open(outdir / 'checkpoint_vitcp.pkl', 'wb') as f: pickle.dump(ckpt, f)

    # ── EQ construction ────────────────────────────────────────────────────
    def encode(u_flat):
        return model.apply({'params': params}, u_flat, training=False, method=model.encode)
    def decode_normalised(z):
        return model.apply({'params': params}, z, method=model.decode_normalised)
    def decode_full(z, scale):
        return model.apply({'params': params}, z, scale, method=model.decode)
    def constrained_decode_normalised(z):
        return mask * decode_normalised(z) + u_g

    all_pairs = []
    for i, traj in enumerate(data['all_snapshots']):
        kap = data['traj_kappas'][i]
        for step in range(NUM_STEPS):
            all_pairs.append((traj[step+1], traj[step], kap))

    rng_eq = np.random.default_rng(123 + seed)
    eq_idx_sample = rng_eq.choice(len(all_pairs), size=min(n_eq_samples, len(all_pairs)), replace=False)
    eq_pairs = [all_pairs[i] for i in eq_idx_sample]
    print(f"EQ pairs: {len(eq_pairs)}")

    if eq_mode == 'full':
        interior = np.where(np.array(mask) > 0)[0]
        eq_indices = interior; eq_weights = np.ones(len(interior), dtype=np.float32)
        print(f"eq_mode=full: {len(eq_indices)} nodes")
    else:
        @jax.jit
        def get_integrand(z_next, u_prev_norm, kappa):
            def cd(z): return constrained_decode_normalised(z)
            u_pred = cd(z_next)
            R = u_pred - u_prev_norm + dt * kappa * K_op_3d(u_pred)
            J_D = jax.jacfwd(cd)(z_next)
            return J_D.T * R[None, :]

        G_list = []
        for u_next, u_prev, kap in eq_pairs:
            z_next, _ = encode(u_next)
            u_prev_norm, _ = normalise(u_prev)
            G_list.append(get_integrand(z_next, u_prev_norm, jnp.float32(kap)))
        G_train = jnp.concatenate(G_list, axis=0)
        G_np = np.array(G_train); G_np[:, np.array(mask) == 0] = 0.0
        b_np = np.sum(G_np, axis=1)
        w_eq, _ = nnls(G_np, b_np, maxiter=4 * G_np.shape[1])
        eq_indices = np.where(w_eq > 1e-10)[0]; eq_weights = w_eq[eq_indices]
        print(f"EQ nodes: {len(eq_indices)} / {num_nodes}")

    eq_idx_jnp = jnp.array(eq_indices); eq_w_jnp = jnp.array(eq_weights)
    n_eq = len(eq_indices)

    # Precompute V_eq (CP basis at EQ node coordinates + stencil neighbours)
    N2 = N * N
    stencil_offsets = jnp.array([0, -1, 1, -N, N, -N2, N2])
    gather_indices = (eq_idx_jnp[:, None] + stencil_offsets[None, :]).flatten()
    ix = gather_indices // N2; iy = (gather_indices // N) % N; iz = gather_indices % N
    W_x = params['decoder']['W_x']; W_y = params['decoder']['W_y']; W_z = params['decoder']['W_z']
    V_eq = W_x[:, ix] * W_y[:, iy] * W_z[:, iz]
    b_scalar = params['decoder']['bias']
    b_sparse = jnp.full(gather_indices.shape, b_scalar)
    mask_sp = mask[gather_indices]; ug_sp = u_g[gather_indices]
    print(f"V_eq: {V_eq.shape}")

    # ── Heat latent solver (joint z,s LM-GN with EQ) ─────────────────────
    GN_MAX_ITERS = max_iters; GN_REL_TOL = gn_tol

    def make_heat_latent_solver(p, V_eq_, b_sp, mask_sp_, ug_sp_, eq_w, latent_dim, dx_, dt_):
        dec = p['decoder']
        W1k, b1k = dec['W1']['kernel'], dec['W1']['bias']
        W2k, b2k = dec['W2']['kernel'], dec['W2']['bias']
        Wrk, brk = dec['W_rank']['kernel'], dec['W_rank']['bias']
        Wdk, bdk = dec['W_direct']['kernel'], dec['W_direct']['bias']

        def _mlp_body(z):
            h_nl = nn.swish(z @ W1k + b1k); h_nl = nn.swish(h_nl @ W2k + b2k)
            h_nl = h_nl @ Wrk + brk; h_lin = z @ Wdk + bdk
            return h_lin + h_nl

        def _f_norm(z, kappa_):
            h = _mlp_body(z)
            u_st = (mask_sp_ * (h @ V_eq_ + b_sp) + ug_sp_).reshape(-1, 7)
            lap = (6*u_st[:,0] - u_st[:,1] - u_st[:,2] - u_st[:,3]
                   - u_st[:,4] - u_st[:,5] - u_st[:,6]) / dx_**2
            return u_st[:,0] + dt_ * kappa_ * lap

        def make_step(kappa_f32):
            @jax.jit
            def solve_step(z_init, s_init, u_prev_phys_eq_):
                fn0 = _f_norm(z_init, kappa_f32)
                R0 = s_init * fn0 - u_prev_phys_eq_
                J0 = jax.jacfwd(lambda zz: _f_norm(zz, kappa_f32))(z_init)
                WR0 = eq_w * R0; g_z0 = s_init * (J0.T @ WR0); g_s0 = jnp.dot(fn0, WR0)
                gnorm0 = jnp.maximum(jnp.sqrt(jnp.dot(g_z0,g_z0) + g_s0**2), 1e-30)

                def _body(carry):
                    z, s, _, itr = carry
                    fn = _f_norm(z, kappa_f32); R = s*fn - u_prev_phys_eq_
                    J = jax.jacfwd(lambda zz: _f_norm(zz, kappa_f32))(z)
                    Wfn = eq_w*fn; WR = eq_w*R; WJ = eq_w[:,None]*J
                    JtWJ = J.T @ WJ; JtWfn = J.T @ Wfn
                    fnWfn = jnp.dot(fn,Wfn); JtWR = J.T @ WR; fnWR = jnp.dot(fn,WR)
                    H_aug = jnp.block([[s**2*JtWJ, (s*JtWfn)[:,None]],
                                       [(s*JtWfn)[None,:], fnWfn[None,None]]])
                    g_aug = jnp.append(s*JtWR, fnWR)
                    lam = jnp.maximum(1e-3 * jnp.trace(H_aug) / (latent_dim+1), 1e-8)
                    delta = jnp.linalg.solve(H_aug + lam*jnp.eye(latent_dim+1), -g_aug)
                    dz = delta[:latent_dim]; ds = delta[latent_dim]
                    f0v = jnp.dot(WR, R)
                    def _f(a):
                        fn_t = _f_norm(z+a*dz, kappa_f32); R_t = (s+a*ds)*fn_t - u_prev_phys_eq_
                        return jnp.dot(eq_w*R_t, R_t)
                    f1,f2,f3,f4 = _f(1.),_f(.5),_f(.25),_f(.125)
                    step = jnp.where(f1<f0v,1.,jnp.where(f2<f0v,.5,jnp.where(f3<f0v,.25,jnp.where(f4<f0v,.125,0.))))
                    gnorm = jnp.sqrt(jnp.dot(s*J.T@WR, s*J.T@WR) + jnp.dot(fn,WR)**2)
                    return z+step*dz, jnp.maximum(s+step*ds, S_MIN), gnorm, itr+1

                def _cond(carry):
                    _, _, gnorm, itr = carry
                    return (gnorm > GN_REL_TOL * gnorm0) & (itr < GN_MAX_ITERS)

                z_f, s_f, gnorm_f, itr_f = jax.lax.while_loop(
                    _cond, _body,
                    (z_init, s_init, jnp.array(jnp.inf, jnp.float32), jnp.array(0, jnp.int32)))
                fn_f = _f_norm(z_f, kappa_f32); R_f = s_f*fn_f - u_prev_phys_eq_
                res_norm = jnp.sqrt(jnp.dot(eq_w*R_f, R_f))
                return z_f, s_f, gnorm_f, itr_f, res_norm
            return solve_step
        return make_step

    make_step_factory = make_heat_latent_solver(
        params, V_eq, b_sparse, mask_sp, ug_sp, eq_w_jnp, k_dim, dx, dt)

    def make_gaussian_ic(centers, amplitudes, widths):
        u = jnp.zeros((N, N, N))
        for (cx,cy,cz), A, sigma in zip(centers, amplitudes, widths):
            u = u + A * jnp.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*sigma**2))
        u = u.at[0,:,:].set(0.).at[-1,:,:].set(0.)
        u = u.at[:,0,:].set(0.).at[:,-1,:].set(0.)
        u = u.at[:,:,0].set(0.).at[:,:,-1].set(0.)
        return u.flatten()

    def run_fom(u0_flat, kappa, steps):
        snapshots = [u0_flat]; u = u0_flat
        op = lambda v: implicit_op(v, kappa)
        for _ in range(steps):
            u, _ = jax_linalg.cg(op, u, x0=u, tol=cg_tol, maxiter=cg_iters)
            snapshots.append(u)
        return jnp.stack(snapshots)

    # Fix: kappa as TRACED RUNTIME argument so JIT compiles ONCE and is reused
    # across all benchmark trajectories (closure-captured kappa previously
    # triggered a fresh trace+compile per kappa, polluting the ROM timing).
    decode_full_jit = jax.jit(decode_full)

    def _solve_step_rt(z_init, s_init, u_prev_phys_eq_, kappa_f32):
        dec = params['decoder']
        W1k, b1k = dec['W1']['kernel'], dec['W1']['bias']
        W2k, b2k = dec['W2']['kernel'], dec['W2']['bias']
        Wrk, brk = dec['W_rank']['kernel'], dec['W_rank']['bias']
        Wdk, bdk = dec['W_direct']['kernel'], dec['W_direct']['bias']
        def _mlp_body(z):
            h_nl = nn.swish(z @ W1k + b1k); h_nl = nn.swish(h_nl @ W2k + b2k)
            h_nl = h_nl @ Wrk + brk; h_lin = z @ Wdk + bdk
            return h_lin + h_nl
        def _f_norm(z):
            h = _mlp_body(z)
            u_st = (mask_sp * (h @ V_eq + b_sparse) + ug_sp).reshape(-1, 7)
            lap = (6*u_st[:,0] - u_st[:,1] - u_st[:,2] - u_st[:,3]
                   - u_st[:,4] - u_st[:,5] - u_st[:,6]) / dx**2
            return u_st[:,0] + dt * kappa_f32 * lap

        fn0 = _f_norm(z_init)
        R0 = s_init * fn0 - u_prev_phys_eq_
        J0 = jax.jacfwd(_f_norm)(z_init)
        WR0 = eq_w_jnp * R0; g_z0 = s_init * (J0.T @ WR0); g_s0 = jnp.dot(fn0, WR0)
        gnorm0 = jnp.maximum(jnp.sqrt(jnp.dot(g_z0,g_z0) + g_s0**2), 1e-30)

        def _body(carry):
            z, s, _, itr = carry
            fn = _f_norm(z); R = s*fn - u_prev_phys_eq_
            J = jax.jacfwd(_f_norm)(z)
            Wfn = eq_w_jnp*fn; WR = eq_w_jnp*R; WJ = eq_w_jnp[:,None]*J
            JtWJ = J.T @ WJ; JtWfn = J.T @ Wfn
            fnWfn = jnp.dot(fn,Wfn); JtWR = J.T @ WR; fnWR = jnp.dot(fn,WR)
            H_aug = jnp.block([[s**2*JtWJ, (s*JtWfn)[:,None]],
                               [(s*JtWfn)[None,:], fnWfn[None,None]]])
            g_aug = jnp.append(s*JtWR, fnWR)
            lam = jnp.maximum(1e-3 * jnp.trace(H_aug) / (k_dim+1), 1e-8)
            delta = jnp.linalg.solve(H_aug + lam*jnp.eye(k_dim+1), -g_aug)
            dz = delta[:k_dim]; ds = delta[k_dim]
            f0v = jnp.dot(WR, R)
            def _f(a):
                fn_t = _f_norm(z+a*dz); R_t = (s+a*ds)*fn_t - u_prev_phys_eq_
                return jnp.dot(eq_w_jnp*R_t, R_t)
            f1,f2,f3,f4 = _f(1.),_f(.5),_f(.25),_f(.125)
            step = jnp.where(f1<f0v,1.,jnp.where(f2<f0v,.5,jnp.where(f3<f0v,.25,jnp.where(f4<f0v,.125,0.))))
            gnorm = jnp.sqrt(jnp.dot(s*J.T@WR, s*J.T@WR) + jnp.dot(fn,WR)**2)
            return z+step*dz, jnp.maximum(s+step*ds, S_MIN), gnorm, itr+1

        def _cond(carry):
            _, _, gnorm, itr = carry
            return (gnorm > GN_REL_TOL * gnorm0) & (itr < GN_MAX_ITERS)

        z_f, s_f, gnorm_f, itr_f = jax.lax.while_loop(
            _cond, _body,
            (z_init, s_init, jnp.array(jnp.inf, jnp.float32), jnp.array(0, jnp.int32)))
        fn_f = _f_norm(z_f); R_f = s_f*fn_f - u_prev_phys_eq_
        res_norm = jnp.sqrt(jnp.dot(eq_w_jnp*R_f, R_f))
        h_new = _mlp_body(z_f)
        u_new_eq_norm = (mask_sp * (h_new @ V_eq + b_sparse) + ug_sp).reshape(-1, 7)[:, 0]
        u_new_phys_eq = s_f * u_new_eq_norm
        return z_f, s_f, gnorm_f, itr_f, res_norm, u_new_phys_eq

    @jax.jit
    def _rollout_rt(z0, s0, u_cur_eq0, kappa_f32):
        iters_arr = jnp.zeros(NUM_STEPS, dtype=jnp.int32)
        res_arr = jnp.zeros(NUM_STEPS, dtype=jnp.float32)
        def _step(i, carry):
            z_, s_, u_eq_, iters_, res_ = carry
            z_n, s_n, _, n_iters, res_norm, u_eq_n = _solve_step_rt(z_, s_, u_eq_, kappa_f32)
            iters_ = iters_.at[i].set(n_iters)
            res_ = res_.at[i].set(res_norm)
            return (z_n, s_n, u_eq_n, iters_, res_)
        return jax.lax.fori_loop(0, NUM_STEPS, _step, (z0, s0, u_cur_eq0, iters_arr, res_arr))

    def run_rom(u0_flat, kappa, steps):
        kappa_f32 = jnp.float32(kappa)
        z0, s0 = encode(u0_flat)
        u_cur_eq0 = decode_full_jit(z0, s0)[eq_idx_jnp]
        z_f, s_f, _, iters_arr, res_arr = _rollout_rt(z0, s0, u_cur_eq0, kappa_f32)
        u_final = decode_full_jit(z_f, s_f)
        jax.block_until_ready(u_final)
        snapshots = jnp.stack([u0_flat, u_final])
        gn_iters = [int(x) for x in np.array(iters_arr)]
        res_norms = [float(x) for x in np.array(res_arr)]
        return snapshots, gn_iters, res_norms

    # ── Test trajectories (fixed seed=9999, same as prior sweeps) ─────────
    def sample_test_params(rng, n_traj):
        sampler = qmc.LatinHypercube(d=17, seed=rng)
        samples = sampler.random(n=n_traj); trajs = []
        for s in samples:
            n_g = int(np.round(1 + 2*s[0]))
            centers, amplitudes, widths = [], [], []
            for g in range(n_g):
                centers.append((0.15+0.70*s[1+g*3], 0.15+0.70*s[2+g*3], 0.15+0.70*s[3+g*3]))
                amplitudes.append(1.0 + 9.0*s[10+g]); widths.append(0.05 + 0.15*s[13+g])
            kappa = float(np.exp(np.log(0.01)+(np.log(0.5)-np.log(0.01))*s[16]))
            trajs.append(dict(centers=centers, amplitudes=amplitudes, widths=widths, kappa=kappa))
        return trajs

    test_params = sample_test_params(rng=9999, n_traj=10)

    # Warmup JIT
    _tp = data['train_params'][0]
    _u0 = make_gaussian_ic(_tp['centers'], _tp['amplitudes'], _tp['widths'])
    run_rom(_u0, float(_tp['kappa']), NUM_STEPS)
    run_rom(_u0, float(_tp['kappa']), NUM_STEPS)
    print("Warmup complete")

    # FOM cache
    FOM_CACHE = outdir / f'fom_cache_{N}.pkl'
    fom_cache = {}
    if FOM_CACHE.exists():
        with open(FOM_CACHE, 'rb') as f: fom_cache = pickle.load(f)

    def _fom_key(tp):
        return (round(tp['kappa'],10),
                tuple(tuple(round(c,10) for c in ctr) for ctr in tp['centers']),
                tuple(round(a,10) for a in tp['amplitudes']),
                tuple(round(w,10) for w in tp['widths']))

    fom_times, rom_times, final_errors = [], [], []
    print("\n--- Benchmark ---")
    for i, tp in enumerate(test_params):
        u0 = make_gaussian_ic(tp['centers'], tp['amplitudes'], tp['widths'])
        kap = tp['kappa']; _key = _fom_key(tp)
        if _key in fom_cache:
            U_fom = jnp.asarray(fom_cache[_key]['U_fom']); fom_t = float(fom_cache[_key]['fom_time'])
        else:
            t0 = time.perf_counter()
            U_fom = run_fom(u0, kap, NUM_STEPS); U_fom[-1].block_until_ready()
            fom_t = time.perf_counter() - t0
            fom_cache[_key] = dict(U_fom=np.array(U_fom), fom_time=fom_t)
            with open(FOM_CACHE, 'wb') as f: pickle.dump(fom_cache, f)
        fom_times.append(fom_t)

        t0 = time.perf_counter()
        U_rom, gn_iters, _ = run_rom(u0, kap, NUM_STEPS)
        rom_t = time.perf_counter() - t0; rom_times.append(rom_t)

        norms_fom = jnp.linalg.norm(U_fom, axis=1)
        final_err = float(jnp.linalg.norm(U_rom[-1] - U_fom[-1]) / (norms_fom[-1] + 1e-12))
        final_errors.append(final_err)
        print(f"  [{i+1:2d}/10] κ={kap:.4f} | FOM {fom_t:.3f}s | ROM {rom_t:.3f}s | "
              f"RelL2(T) {final_err:.3e} | GN avg {np.mean(gn_iters):.1f}")

    speedup = float(np.median([f/r for f,r in zip(fom_times, rom_times)]))
    val_rel_l2 = float(np.mean(final_errors))
    print(f"\nMedian speedup {speedup:.2f}x  Avg rel-L2 {val_rel_l2:.4e}")

    return dict(
        val_rel_l2=val_rel_l2,
        speedup=speedup,
        fom_seconds=float(np.mean(fom_times)),
        rom_seconds=float(np.mean(rom_times)),
        gn_iters_mean=float(max_iters),
        training_seconds=training_seconds,
        peak_vram_mb=0.0,
        n_params=n_params,
        N=N, k_dim=k_dim, rank=rank, n_eq=n_eq,
    )
