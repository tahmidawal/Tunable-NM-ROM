"""
poisson3d_nmrom_sweep.py
------------------------
Config-driven Poisson-3D NM-ROM: combined data-gen / AE training / ROM
benchmark in one script.  Mirrors the shape of Poisson-2D's
poisson_nmrom_sweep.py but uses 3D kernels and CP-3 tensor decoder.

CLI knobs:
  --N                 grid resolution
  --k-dim             latent dim
  --rank              CP rank
  --hidden-dim, --patch-size, --embed-dim, --num-heads, --num-enc-layers
  --num-epochs, --batch-size, --peak-lr, --weight-decay
  --n-train, --n-val
  --max-iters         GN max iterations
  --gn-tol            GN rel tol
  --cg-tol            CG tol (inside GN)
  --eq-mode           nnls | full  (NM-ROM hyper-reduction mode)
  --min-eq-points     NNLS target number of EQ points
  --n-eq-snaps        EQ snapshot sampling count
  --outdir
"""

import argparse
import json
import sys
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import flax.linen as nn
import optax
import numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='poisson3d_run')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--k-dim', type=int, default=16)
    ap.add_argument('--rank', type=int, default=512)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--patch-size', type=int, default=8)
    ap.add_argument('--embed-dim', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=4)
    ap.add_argument('--num-enc-layers', type=int, default=4)
    ap.add_argument('--num-epochs', type=int, default=100_000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--peak-lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--n-train', type=int, default=500)
    ap.add_argument('--n-val', type=int, default=100)
    ap.add_argument('--max-iters', type=int, default=8)
    ap.add_argument('--gn-tol', type=float, default=1e-6)
    ap.add_argument('--cg-tol', type=float, default=1e-3)
    ap.add_argument('--cg-iters', type=int, default=8)
    ap.add_argument('--eq-mode', choices=['nnls', 'full'], default='nnls')
    ap.add_argument('--min-eq-points', type=int, default=300)
    ap.add_argument('--n-eq-snaps', type=int, default=80)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset-path', default=None,
                    help='Pre-generated dataset npz (U_train, U_val, train_freqs, val_freqs). '
                         'If set and exists, skips per-run generation.')
    ap.add_argument('--arch', choices=['full', 'no_linear', 'dense'], default='full',
                    help='Decoder architecture variant for the ablation study. '
                         '"full": linear branch + MLP + CP head (default). '
                         '"no_linear": MLP only into CP head (drops W_direct). '
                         '"dense": linear + MLP into a dense R->N^d readout (drops CP head).')
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / 'plots'; plot_dir.mkdir(exist_ok=True)

    log_f = open(outdir / 'run.log', 'w')
    class Tee:
        def write(self, m):
            sys.__stdout__.write(m); log_f.write(m); log_f.flush()
        def flush(self):
            sys.__stdout__.flush(); log_f.flush()
    sys.stdout = Tee()

    with open(outdir / 'config_used.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    N = args.N
    num_nodes = N ** 3
    L = 1.0
    dx = L / (N - 1)
    k_dim = args.k_dim

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

    def get_F_3d(k1, k2, k3):
        F = jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y) * jnp.sin(k3 * jnp.pi * Z) * 10.0
        F = F.at[0, :, :].set(0.).at[-1, :, :].set(0.)
        F = F.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        F = F.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        return F.flatten()

    mask_3d = jnp.ones((N, N, N))
    mask_3d = mask_3d.at[0, :, :].set(0.).at[-1, :, :].set(0.)
    mask_3d = mask_3d.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
    mask_3d = mask_3d.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
    mask = mask_3d.flatten()
    u_g = jnp.zeros(num_nodes)

    # ── Model ─────────────────────────────────────
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
                for _ in range(self.num_enc_layers)]
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

    # --- Ablation variant (B): no linear branch, MLP-only into CP head. --------
    class NoLinearCPDecoder(nn.Module):
        latent_dim: int
        rank: int = 512
        grid_size: int = 32
        hidden_dim: int = 256

        def setup(self):
            self.W1 = nn.Dense(self.hidden_dim)
            self.W2 = nn.Dense(self.hidden_dim)
            self.W_rank = nn.Dense(self.rank)
            init = nn.initializers.normal(0.01)
            Ng = self.grid_size
            self.W_x = self.param('W_x', init, (self.rank, Ng))
            self.W_y = self.param('W_y', init, (self.rank, Ng))
            self.W_z = self.param('W_z', init, (self.rank, Ng))
            self.bias = self.param('bias', nn.initializers.zeros, ())

        def __call__(self, z):
            h = nn.swish(self.W1(z))
            h = nn.swish(self.W2(h))
            h = self.W_rank(h)
            u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
            return u_3d.flatten() + self.bias

    # --- Ablation variant (C): both branches, CP head replaced by dense readout.
    # The rank-R channel vector h is mapped to the full flattened field by a
    # single Dense(N^d). Parameter count blows up as R * N_g^d.
    class LinearDenseDecoder(nn.Module):
        latent_dim: int
        rank: int = 512
        grid_size: int = 32
        hidden_dim: int = 256

        def setup(self):
            self.W1 = nn.Dense(self.hidden_dim)
            self.W2 = nn.Dense(self.hidden_dim)
            self.W_rank = nn.Dense(self.rank)
            self.W_direct = nn.Dense(self.rank)
            Ng = self.grid_size
            self.readout = nn.Dense(Ng * Ng * Ng)
            self.bias = self.param('bias', nn.initializers.zeros, ())

        def __call__(self, z):
            h_nl = nn.swish(self.W1(z))
            h_nl = nn.swish(self.W2(h_nl))
            h_nl = self.W_rank(h_nl)
            h_lin = self.W_direct(z)
            h = h_lin + h_nl
            u_flat = self.readout(h)
            return u_flat + self.bias

    class ViTLinearCPAutoencoder(nn.Module):
        latent_dim: int
        grid_size: int
        patch_size: int = 8
        embed_dim: int = 64
        num_heads: int = 4
        num_enc_layers: int = 4
        rank: int = 512
        hidden_dim: int = 256

        arch: str = 'full'

        def setup(self):
            self.encoder = ViTEncoder(
                latent_dim=self.latent_dim, grid_n=self.grid_size,
                patch_size=self.patch_size, embed_dim=self.embed_dim,
                num_heads=self.num_heads, num_enc_layers=self.num_enc_layers)
            if self.arch == 'full':
                dec_cls = LinearCPDecoder
            elif self.arch == 'no_linear':
                dec_cls = NoLinearCPDecoder
            elif self.arch == 'dense':
                dec_cls = LinearDenseDecoder
            else:
                raise ValueError(f'unknown arch: {self.arch}')
            self.decoder = dec_cls(
                latent_dim=self.latent_dim, rank=self.rank,
                grid_size=self.grid_size, hidden_dim=self.hidden_dim)

        def encode(self, u):
            return self.encoder(u)
        def decode(self, z):
            return self.decoder(z)
        def __call__(self, u):
            return self.decode(self.encode(u))

    # ── Training data ─────────────────────────────
    def full_order_fem_solver(F_vec, u_guess=None, tol=1e-6):
        if u_guess is None:
            u_guess = jnp.zeros(num_nodes)
        u, _ = jax_linalg.cg(K_op_3d, F_vec, x0=u_guess, tol=tol)
        return u

    print(f"JAX devices: {jax.devices()}")
    print(f"[{args.label}] Grid {N}^3 = {num_nodes:,} | k={k_dim} rank={args.rank}")

    rng = np.random.RandomState(args.seed)
    N_train, N_val = args.n_train, args.n_val
    train_freqs = rng.uniform(1.0, 3.0, size=(N_train, 3))
    val_freqs = rng.uniform(1.0, 3.0, size=(N_val, 3))
    u_guess = jnp.zeros(num_nodes)

    # If a shared dataset is provided, use it directly (skip per-run gen).
    # Shape is validated; freqs are replaced by the ones in the shared file
    # so the train/val splits match what was solved.
    SHARED_PATH = Path(args.dataset_path) if args.dataset_path else None
    DATASET_PATH = outdir / 'dataset.npz'
    _need_generate = True

    # Keep U_train/U_val as numpy (CPU) and ship per-batch to GPU below.
    # At N=512 the full dataset (107 GB for 200 train) won't fit in 80 GB GPU.
    if SHARED_PATH is not None and SHARED_PATH.exists():
        print(f"   loading shared dataset: {SHARED_PATH}")
        _ds = np.load(SHARED_PATH, mmap_mode='r')
        if _ds['U_train'].shape[0] < N_train or _ds['U_val'].shape[0] < N_val:
            raise ValueError(
                f"Shared dataset too small: has "
                f"{_ds['U_train'].shape[0]}/{_ds['U_val'].shape[0]} train/val, "
                f"requested {N_train}/{N_val}")
        U_train = np.asarray(_ds['U_train'][:N_train])
        U_val = np.asarray(_ds['U_val'][:N_val])
        train_freqs = np.asarray(_ds['train_freqs'][:N_train])
        val_freqs = np.asarray(_ds['val_freqs'][:N_val])
        print(f"   shared dataset: train {U_train.shape} val {U_val.shape}")
        _need_generate = False

    if _need_generate and DATASET_PATH.exists():
        _ds = np.load(DATASET_PATH)
        if _ds['U_train'].shape[0] == N_train and _ds['U_val'].shape[0] == N_val:
            U_train = np.asarray(_ds['U_train'])
            U_val = np.asarray(_ds['U_val'])
            print(f"   dataset cache hit: train {U_train.shape} val {U_val.shape}")
            _need_generate = False

    if _need_generate:
        print(f"   generating {N_train} train + {N_val} val snapshots...")
        U_train_list = []
        for i, (k1, k2, k3) in enumerate(train_freqs):
            U_train_list.append(np.asarray(full_order_fem_solver(get_F_3d(k1, k2, k3), u_guess)))
            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{N_train}")
        U_val_list = [np.asarray(full_order_fem_solver(get_F_3d(*f), u_guess)) for f in val_freqs]
        U_train = np.stack(U_train_list)
        U_val = np.stack(U_val_list)
        np.savez(DATASET_PATH, U_train=U_train, U_val=U_val)
        print(f"   saved {DATASET_PATH}")

    # ── Model init ────────────────────────────────
    model = ViTLinearCPAutoencoder(
        latent_dim=k_dim, grid_size=N,
        patch_size=args.patch_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_enc_layers=args.num_enc_layers,
        rank=args.rank, hidden_dim=args.hidden_dim,
        arch=args.arch,
    )
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones(num_nodes))['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model: {n_params:,} params")

    # ── Training ──────────────────────────────────
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=args.peak_lr,
        warmup_steps=500, decay_steps=args.num_epochs, end_value=1e-6,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    opt_state = tx.init(params)

    def rec_loss(p, u_batch):
        preds = jax.vmap(lambda u: model.apply({'params': p}, u))(u_batch)
        diffs = preds - u_batch
        norm_sq = jnp.sum(u_batch ** 2, axis=1) + 1e-6
        return jnp.mean(jnp.sum(diffs ** 2, axis=1) / norm_sq)

    @jax.jit
    def train_step(p, o, batch):
        loss, grads = jax.value_and_grad(rec_loss)(p, batch)
        upd, o = tx.update(grads, o, p)
        p = optax.apply_updates(p, upd)
        return p, o, loss

    LOG_EVERY = max(500, args.num_epochs // 100)
    print(f"Training {args.num_epochs} epochs batch={args.batch_size}")
    t0 = time.perf_counter()
    rng_np = np.random.default_rng(args.seed + 1)
    best_val = float('inf'); best_params = params
    train_curve = []; val_curve = []

    # Chunked val/train eval (keeps GPU memory bounded at high N)
    def chunked_rec_loss(p, U_np, chunk=32):
        total = 0.0; n = len(U_np)
        for s in range(0, n, chunk):
            b = jnp.asarray(U_np[s:s + chunk])
            total += float(rec_loss(p, b)) * len(b)
        return total / n

    for epoch in range(args.num_epochs + 1):
        idx = rng_np.choice(N_train, size=args.batch_size, replace=False)
        batch = jnp.asarray(U_train[idx])
        params, opt_state, loss = train_step(params, opt_state, batch)
        if epoch % LOG_EVERY == 0:
            v_loss = chunked_rec_loss(params, U_val)
            tr_err = chunked_rec_loss(params, U_train[:min(128, N_train)])
            train_curve.append((epoch, tr_err))
            val_curve.append((epoch, v_loss))
            print(f"  Epoch {epoch:6d} | train {tr_err:.4e} | val {v_loss:.4e} | "
                  f"{time.perf_counter()-t0:.0f}s")
            if v_loss < best_val:
                best_val = v_loss; best_params = params
    params = best_params
    print(f"Best val: {best_val:.4e}  ({time.perf_counter()-t0:.0f}s)")

    ckpt_path = outdir / 'checkpoint.pkl'
    with open(ckpt_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Checkpoint: {ckpt_path}")

    # Loss plot
    ep_t, lo_t = zip(*train_curve); ep_v, lo_v = zip(*val_curve)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(ep_t, lo_t, label='train'); ax.semilogy(ep_v, lo_v, label='val', ls='--')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend(); ax.grid(True, which='both', alpha=.3)
    plt.tight_layout(); plt.savefig(plot_dir / 'loss_curve.png', dpi=120); plt.close()

    # ── NM-ROM solver ─────────────────────────────
    def make_rom_solver(mdl, p, K_op, mask_vec, u_g_vec, max_iters, gn_tol, cg_tol, cg_iters):
        def constrained_decode(z):
            u_hat = mdl.apply({'params': p}, z, method=mdl.decode)
            return mask_vec * u_hat + u_g_vec

        @jax.jit
        def rom_solve(z_init, F_vec):
            def body_fn(state):
                z, _, i = state
                u_pred, vjp_fn = jax.vjp(constrained_decode, z)
                R = K_op(u_pred) - F_vec
                r_red = vjp_fn(R)[0]
                def gn_op(dz):
                    _, J_dz = jax.jvp(constrained_decode, (z,), (dz,))
                    return vjp_fn(K_op(J_dz))[0]
                delta_z, _ = jax_linalg.cg(gn_op, -r_red, tol=cg_tol, maxiter=cg_iters)
                z_new = z + delta_z
                return z_new, jnp.linalg.norm(r_red), i + 1

            def cond_fn(state):
                _, res_norm, i = state
                return (res_norm > gn_tol) & (i < max_iters)

            u0 = constrained_decode(z_init)
            R0 = K_op(u0) - F_vec
            _, vjp0 = jax.vjp(constrained_decode, z_init)
            r_red0 = vjp0(R0)[0]
            res0 = jnp.linalg.norm(r_red0)
            z_final, res_final, iters = jax.lax.while_loop(
                cond_fn, body_fn, (z_init, res0, 0))
            return z_final, constrained_decode(z_final), res0, res_final, iters

        return rom_solve

    rom_solve = make_rom_solver(
        model, params, K_op_3d, mask, u_g,
        args.max_iters, args.gn_tol, args.cg_tol, args.cg_iters,
    )

    # ── Benchmark ─────────────────────────────────
    TEST_CASES = [
        (1.5, 2.3, 1.8), (2.1, 1.4, 2.7),
        (1.2, 2.8, 1.3), (2.5, 1.7, 2.2),
    ]
    F_warm = get_F_3d(*TEST_CASES[0])
    _ = jax_linalg.cg(K_op_3d, F_warm, x0=u_guess, tol=1e-6)[0].block_until_ready()
    _, _, _, _, _ = rom_solve(jnp.zeros(k_dim), F_warm); jax.effects_barrier()

    fom_times, rom_times, rel_errors = [], [], []
    gn_iters_list, gn_res0_list, gn_resf_list = [], [], []
    for case_idx, (k1, k2, k3) in enumerate(TEST_CASES):
        F_test = get_F_3d(k1, k2, k3)
        z_init = jnp.zeros(k_dim)
        t0 = time.time()
        u_fom, _ = jax_linalg.cg(K_op_3d, F_test, x0=u_guess, tol=1e-6)
        u_fom.block_until_ready()
        fom_t = time.time() - t0
        t0 = time.time()
        z_final, u_pred, res0, res_final, iters = rom_solve(z_init, F_test)
        u_pred.block_until_ready()
        rom_t = time.time() - t0
        rel_l2 = float(jnp.linalg.norm(u_fom - u_pred) / jnp.linalg.norm(u_fom))
        fom_times.append(fom_t); rom_times.append(rom_t); rel_errors.append(rel_l2)
        gn_iters_list.append(int(iters))
        gn_res0_list.append(float(res0))
        gn_resf_list.append(float(res_final))
        print(f"   case {case_idx+1} k=({k1},{k2},{k3}) "
              f"FOM {fom_t:.4f}s ROM {rom_t:.4f}s "
              f"speedup {fom_t/rom_t:.2f}x relL2 {rel_l2:.4e}")

    avg_fom = float(np.mean(fom_times))
    avg_rom = float(np.mean(rom_times))
    per_case_sp = [f / r for f, r in zip(fom_times, rom_times)]
    med_sp = float(np.median(per_case_sp))
    avg_err = float(np.mean(rel_errors))

    print(f"\nAvg FOM {avg_fom:.5f}s  Avg ROM {avg_rom:.5f}s  Med speedup {med_sp:.2f}x  Rel-L2 {avg_err:.4e}")

    gn_converged_count = sum(
        1 for rf, r0, it in zip(gn_resf_list, gn_res0_list, gn_iters_list)
        if (rf <= args.gn_tol) or (it < args.max_iters))
    avg_gn_iters = float(np.mean(gn_iters_list))
    median_resf_over_res0 = float(np.median(
        [rf / max(r0, 1e-300) for rf, r0 in zip(gn_resf_list, gn_res0_list)]))

    header = ["label", "N", "k_dim", "rank", "arch", "n_params",
              "max_iters", "gn_tol", "cg_tol",
              "speedup", "rel_l2", "fom_time", "rom_time",
              "gn_converged_out_of", "avg_gn_iters", "median_resf_over_res0"]
    row = [args.label, str(N), str(k_dim), str(args.rank),
           args.arch, str(n_params),
           str(args.max_iters), f"{args.gn_tol:.0e}", f"{args.cg_tol:.0e}",
           f"{med_sp:.4f}", f"{avg_err:.4e}",
           f"{avg_fom:.5f}", f"{avg_rom:.5f}",
           f"{gn_converged_count}/{len(TEST_CASES)}",
           f"{avg_gn_iters:.2f}",
           f"{median_resf_over_res0:.3e}"]
    import json as _json
    with open(outdir / 'gn_per_case.json', 'w') as f:
        _json.dump({
            'label': args.label, 'arch': args.arch, 'k_dim': k_dim,
            'rank': args.rank, 'max_iters': args.max_iters,
            'gn_tol': args.gn_tol,
            'test_cases': TEST_CASES,
            'gn_iters': gn_iters_list,
            'gn_res0': gn_res0_list,
            'gn_resf': gn_resf_list,
            'rel_l2': rel_errors,
            'fom_time': fom_times,
            'rom_time': rom_times,
        }, f, indent=2)
    with open(outdir / 'results.tsv', 'w') as f:
        f.write('\t'.join(header) + '\n')
        f.write('\t'.join(row) + '\n')

    # Slice plot (last case)
    mid = N // 2
    k1, k2, k3 = TEST_CASES[-1]
    F_last = get_F_3d(k1, k2, k3)
    u_true = full_order_fem_solver(F_last, u_guess)
    _, u_pred_last, _, _, _ = rom_solve(jnp.zeros(k_dim), F_last)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    def plot_slice(ax, field, title):
        im = ax.imshow(np.array(field).reshape(N, N, N)[:, :, mid].T,
                       origin='lower', cmap='inferno',
                       extent=[0, L, 0, L], aspect='auto')
        ax.set_title(title); fig.colorbar(im, ax=ax)
    plot_slice(axs[0], u_true, f"FOM k=({k1:.1f},{k2:.1f},{k3:.1f})")
    plot_slice(axs[1], u_pred_last, f"NM-ROM")
    plot_slice(axs[2], jnp.abs(u_true - u_pred_last),
               f"|Error|  Rel L2={rel_errors[-1]:.2e}")
    plt.suptitle(f"{args.label} z=mid")
    plt.tight_layout(); plt.savefig(plot_dir / 'slice.png', dpi=120,
                                     bbox_inches='tight'); plt.close()

    print("\n=== Poisson-3D sweep run complete ===")


if __name__ == '__main__':
    main()


# =============================================================================
# AUTORESEARCH ENTRY POINT
# =============================================================================
# Shared CG-solved datasets (do NOT regenerate — use these paths):
#   N=32:  /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Poisson-3D/res-32/dataset.npz
#   N=64:  /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Poisson-3D/res-64/dataset.npz
#   N=128: /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Poisson-3D/res-128/dataset.npz
# These datasets have keys: U_train (700, N^3), U_val (140, N^3).
# They do NOT store train_freqs/val_freqs — those are reconstructed from seed=42
# using the same rng.uniform(1.0, 3.0) call as the original generate script.
# The freqs are only used for F-vector construction during training; the test
# cases are hardcoded below so this is irrelevant at benchmark time.

SHARED_DATASETS = {
    32:  '/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Poisson-3D/res-32/dataset.npz',
    64:  '/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Poisson-3D/res-64/dataset.npz',
    128: '/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260416-NeurIPS/Poisson-3D/res-128/dataset.npz',
}


def run_and_benchmark():
    import argparse, json, sys, pickle, time
    from pathlib import Path
    import jax, jax.numpy as jnp, jax.scipy.sparse.linalg as jax_linalg
    import flax.linen as nn, optax, numpy as np
    from scipy.optimize import nnls

    # ── Hyperparameters (agent modifies this block) ────────────────────────
    N             = 128
    k_dim         = 16
    rank          = 768
    hidden_dim    = 256
    patch_size    = 16
    embed_dim     = 64
    num_heads     = 4
    num_enc_layers = 4
    num_epochs    = 60_000
    batch_size    = 32
    peak_lr       = 1e-3  # fresh train: full-strength LR like Run26
    weight_decay  = 2e-3
    n_train       = 400
    n_val         = 80
    max_iters     = 25
    gn_tol        = 5e-2
    cg_tol        = 1e-3
    cg_iters      = 8
    eq_mode       = 'nnls'   # 'nnls' or 'full'
    min_eq_points = 2000
    n_eq_snaps    = 400
    outdir        = Path('/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Autoresearch/Poisson-3D/runs/N128_rank768')
    seed          = 42
    # Set to a checkpoint path to skip AE retraining (EQ-only run)
    load_ckpt     = None
    # Set to a checkpoint path to fine-tune from (training continues with current schedule)
    finetune_ckpt = None  # fresh training for capacity push
    # ───────────────────────────────────────────────────────────────────────

    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / 'plots'; plot_dir.mkdir(exist_ok=True)

    log_f = open(outdir / 'run.log', 'w')
    class Tee:
        def write(self, m):
            sys.__stdout__.write(m); log_f.write(m); log_f.flush()
        def flush(self):
            sys.__stdout__.flush(); log_f.flush()
    sys.stdout = Tee()

    num_nodes = N ** 3
    L = 1.0
    dx = L / (N - 1)

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

    def get_F_3d(k1, k2, k3):
        F = jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y) * jnp.sin(k3 * jnp.pi * Z) * 10.0
        F = F.at[0, :, :].set(0.).at[-1, :, :].set(0.)
        F = F.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        F = F.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        return F.flatten()

    mask_3d = jnp.ones((N, N, N))
    mask_3d = mask_3d.at[0, :, :].set(0.).at[-1, :, :].set(0.)
    mask_3d = mask_3d.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
    mask_3d = mask_3d.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
    mask = mask_3d.flatten()
    u_g = jnp.zeros(num_nodes)

    # ── Model ─────────────────────────────────────────────────────────────
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
        latent_dim: int; grid_n: int; patch_size: int = 8; embed_dim: int = 64
        num_heads: int = 4; num_enc_layers: int = 4
        def setup(self):
            n_per_side = self.grid_n // self.patch_size
            self._n_per_side = n_per_side
            self._num_patches = n_per_side ** 3
            self._patch_dim = self.patch_size ** 3
            self.patch_embed = nn.Dense(self.embed_dim)
            self.enc_pos = self.param('enc_pos', nn.initializers.normal(stddev=0.02),
                                      (self._num_patches, self.embed_dim))
            self.enc_blocks = [TransformerBlock(self.embed_dim, self.num_heads)
                               for _ in range(self.num_enc_layers)]
            self.enc_norm = nn.LayerNorm()
            self.enc_proj = nn.Dense(self.latent_dim)
        def _patchify(self, u_flat):
            n = self._n_per_side; p = self.patch_size
            return (u_flat.reshape(n, p, n, p, n, p)
                    .transpose(0, 2, 4, 1, 3, 5)
                    .reshape(self._num_patches, self._patch_dim))
        def __call__(self, u_flat):
            x = self._patchify(u_flat)
            x = self.patch_embed(x) + self.enc_pos
            for block in self.enc_blocks: x = block(x)
            x = self.enc_norm(x)
            return self.enc_proj(x.mean(axis=0))

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

    class ViTLinearCPAutoencoder(nn.Module):
        latent_dim: int; grid_size: int; patch_size: int = 8; embed_dim: int = 64
        num_heads: int = 4; num_enc_layers: int = 4; rank: int = 512; hidden_dim: int = 256
        def setup(self):
            self.encoder = ViTEncoder(latent_dim=self.latent_dim, grid_n=self.grid_size,
                patch_size=self.patch_size, embed_dim=self.embed_dim,
                num_heads=self.num_heads, num_enc_layers=self.num_enc_layers)
            self.decoder = LinearCPDecoder(latent_dim=self.latent_dim, rank=self.rank,
                grid_size=self.grid_size, hidden_dim=self.hidden_dim)
        def encode(self, u): return self.encoder(u)
        def decode(self, z): return self.decoder(z)
        def __call__(self, u): return self.decode(self.encode(u))

    # ── Dataset (reuse existing CG-solved data) ───────────────────────────
    def full_order_fem_solver(F_vec, u_guess=None, tol=1e-6):
        if u_guess is None: u_guess = jnp.zeros(num_nodes)
        u, _ = jax_linalg.cg(K_op_3d, F_vec, x0=u_guess, tol=tol)
        return u

    print(f"JAX devices: {jax.devices()}")
    print(f"Grid {N}^3 = {num_nodes:,} | k={k_dim} rank={rank}")

    # Reconstruct freqs from the same seed used to generate the shared dataset
    rng = np.random.RandomState(seed)
    train_freqs = rng.uniform(1.0, 3.0, size=(700, 3))  # full set; we slice below
    val_freqs   = rng.uniform(1.0, 3.0, size=(140, 3))

    shared_path = Path(SHARED_DATASETS[N])
    if not shared_path.exists():
        raise FileNotFoundError(f"Shared dataset missing: {shared_path}")
    _ds = np.load(shared_path, mmap_mode='r')
    U_train = np.asarray(_ds['U_train'][:n_train])
    U_val   = np.asarray(_ds['U_val'][:n_val])
    train_freqs = train_freqs[:n_train]
    val_freqs   = val_freqs[:n_val]
    print(f"Loaded shared dataset: train {U_train.shape} val {U_val.shape}")

    # ── Model init ────────────────────────────────────────────────────────
    model = ViTLinearCPAutoencoder(latent_dim=k_dim, grid_size=N, patch_size=patch_size,
        embed_dim=embed_dim, num_heads=num_heads, num_enc_layers=num_enc_layers,
        rank=rank, hidden_dim=hidden_dim)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones(num_nodes))['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model: {n_params:,} params")

    # ── Training ──────────────────────────────────────────────────────────
    # load_ckpt: skip training entirely (EQ-only)
    # finetune_ckpt: load ckpt as init then continue training with current schedule
    _load_ckpt = locals().get('load_ckpt', None)
    _finetune_ckpt = locals().get('finetune_ckpt', None)
    if _load_ckpt is not None and Path(_load_ckpt).exists():
        print(f"Loading checkpoint from {_load_ckpt} — skipping AE retraining")
        with open(_load_ckpt, 'rb') as f: params = pickle.load(f)
        training_seconds = 0.0
        print("Checkpoint loaded.")
    else:
        if _finetune_ckpt is not None and Path(_finetune_ckpt).exists():
            print(f"Loading checkpoint from {_finetune_ckpt} — fine-tuning for {num_epochs} epochs")
            with open(_finetune_ckpt, 'rb') as f: params = pickle.load(f)
            print("Checkpoint loaded as init.")
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0., peak_value=peak_lr, warmup_steps=500,
            decay_steps=num_epochs, end_value=1e-6)
        tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        opt_state = tx.init(params)

        def rec_loss(p, u_batch):
            preds = jax.vmap(lambda u: model.apply({'params': p}, u))(u_batch)
            diffs = preds - u_batch
            return jnp.mean(jnp.sum(diffs ** 2, axis=1) / (jnp.sum(u_batch ** 2, axis=1) + 1e-6))

        @jax.jit
        def train_step(p, o, batch):
            loss, grads = jax.value_and_grad(rec_loss)(p, batch)
            upd, o = tx.update(grads, o, p); p = optax.apply_updates(p, upd)
            return p, o, loss

        def chunked_rec_loss(p, U_np, chunk=32):
            total = 0.0; n = len(U_np)
            for s in range(0, n, chunk):
                b = jnp.asarray(U_np[s:s + chunk])
                total += float(rec_loss(p, b)) * len(b)
            return total / n

        LOG_EVERY = max(500, num_epochs // 100)
        print(f"Training {num_epochs} epochs batch={batch_size}")
        t_train_start = time.perf_counter()
        rng_np = np.random.default_rng(seed + 1)
        best_val = float('inf'); best_params = params

        for epoch in range(num_epochs + 1):
            idx = rng_np.choice(n_train, size=batch_size, replace=False)
            params, opt_state, _ = train_step(params, opt_state, jnp.asarray(U_train[idx]))
            if epoch % LOG_EVERY == 0:
                v_loss = chunked_rec_loss(params, U_val)
                print(f"  Epoch {epoch:6d} | val {v_loss:.4e} | {time.perf_counter()-t_train_start:.0f}s")
                if v_loss < best_val: best_val = v_loss; best_params = params

        params = best_params
        training_seconds = time.perf_counter() - t_train_start
        print(f"Best val: {best_val:.4e}  ({training_seconds:.0f}s)")

        ckpt_path = outdir / 'checkpoint.pkl'
        with open(ckpt_path, 'wb') as f: pickle.dump(params, f)

    # ── EQ construction (NNLS hyper-reduction) ────────────────────────────
    # Correct formulation: NNLS on PDE residuals of training snapshots.
    # G[s, i] = (K @ u_train[s])[i]^2  — squared PDE operator applied at node i.
    # This picks nodes that are important for integrating K*u accurately,
    # which is exactly what the GN inner solve needs.
    print(f"Building EQ (n_eq_snaps={n_eq_snaps}, min_eq_points={min_eq_points})")
    dec_p = params['decoder']
    W_x_np = np.asarray(dec_p['W_x'])  # (rank, N)
    W_y_np = np.asarray(dec_p['W_y'])
    W_z_np = np.asarray(dec_p['W_z'])
    N2 = N * N

    rng_eq = np.random.default_rng(seed + 99)
    eq_snap_idx = rng_eq.choice(len(U_train), size=min(n_eq_snaps, len(U_train)), replace=False)

    # Apply K to each training snapshot — gives the PDE operator output at all nodes
    K_jit = jax.jit(K_op_3d)
    Ku_snaps = np.stack([np.array(K_jit(jnp.array(U_train[i]))) for i in eq_snap_idx])
    # G[s, i] = |Ku[i]|  (absolute value, non-negative for NNLS)
    G_np = np.abs(Ku_snaps)   # (n_snaps, N^3)
    b_np = G_np.sum(axis=1)   # integrate = total PDE operator norm per snap
    # Restrict NNLS to strictly interior nodes — boundary nodes have known BC (u=0)
    # and their stencil wraps/clips incorrectly; only interior nodes have valid Laplacian
    interior_mask_np = np.zeros(num_nodes, dtype=bool)
    ix_all = np.arange(num_nodes) // N2
    iy_all = (np.arange(num_nodes) // N) % N
    iz_all = np.arange(num_nodes) % N
    interior_mask_np[(ix_all >= 1) & (ix_all <= N-2) &
                     (iy_all >= 1) & (iy_all <= N-2) &
                     (iz_all >= 1) & (iz_all <= N-2)] = True
    G_np_int = G_np[:, interior_mask_np]
    interior_indices = np.where(interior_mask_np)[0]

    w_eq_int, _ = nnls(G_np_int, b_np, maxiter=10_000)
    w_eq_full = np.zeros(num_nodes)
    w_eq_full[interior_indices] = w_eq_int

    eq_indices = np.where(w_eq_full > 1e-10)[0]
    if len(eq_indices) < min_eq_points:
        top_idx = np.argsort(w_eq_full)[::-1][:min_eq_points]
        eq_indices = np.unique(np.concatenate([eq_indices, top_idx]))
    eq_weights = w_eq_full[eq_indices]
    eq_weights = eq_weights / eq_weights.sum()  # normalise
    n_eq = len(eq_indices)
    print(f"EQ nodes: {n_eq} / {num_nodes}")

    eq_idx_jnp = jnp.array(eq_indices)
    eq_w_jnp   = jnp.array(eq_weights, dtype=jnp.float32)

    # Precompute V_eq: CP basis at EQ node coordinates
    ix_eq = eq_indices // N2
    iy_eq = (eq_indices // N) % N
    iz_eq = eq_indices % N
    V_eq = jnp.array(W_x_np[:, ix_eq] * W_y_np[:, iy_eq] * W_z_np[:, iz_eq])  # (rank, n_eq)
    bias_eq = jnp.full(n_eq, float(dec_p['bias']))
    mask_eq = mask[eq_idx_jnp]
    u_g_eq  = u_g[eq_idx_jnp]
    F_eq_cache = {}  # filled per test case below

    # Precompute stencil-neighbour EQ for Laplacian at EQ nodes
    # stencil: center + ±x ±y ±z  (7 points); all eq_indices are interior so no clipping needed
    stencil_offsets = np.array([0, -1, 1, -N, N, -N2, N2])
    st_indices = (eq_indices[:, None] + stencil_offsets[None, :]).flatten()  # (n_eq*7,)
    ix_st = st_indices // N2; iy_st = (st_indices // N) % N; iz_st = st_indices % N
    V_eq_st = jnp.array(
        W_x_np[:, np.asarray(ix_st)] * W_y_np[:, np.asarray(iy_st)] * W_z_np[:, np.asarray(iz_st)]
    )  # (rank, n_eq*7)
    bias_st  = jnp.full(st_indices.shape[0], float(dec_p['bias']))
    mask_st  = mask[st_indices]
    u_g_st   = u_g[st_indices]

    # Extract decoder MLP weights for jacfwd
    W1k = dec_p['W1']['kernel']; b1k = dec_p['W1']['bias']
    W2k = dec_p['W2']['kernel']; b2k = dec_p['W2']['bias']
    Wrk = dec_p['W_rank']['kernel']; brk = dec_p['W_rank']['bias']
    Wdk = dec_p['W_direct']['kernel']; bdk = dec_p['W_direct']['bias']

    def _h(z):
        h_nl = nn.swish(z @ W1k + b1k)
        h_nl = nn.swish(h_nl @ W2k + b2k)
        h_nl = h_nl @ Wrk + brk
        return z @ Wdk + bdk + h_nl  # (rank,)

    def _u_eq(z):
        # Decode at EQ node coordinates — O(k * rank * n_eq), N-independent
        h = _h(z)
        return mask_eq * (h @ V_eq + bias_eq) + u_g_eq  # (n_eq,)

    def _u_st(z):
        # Decode at stencil nodes — for Laplacian residual at EQ nodes
        h = _h(z)
        return (mask_st * (h @ V_eq_st + bias_st) + u_g_st).reshape(n_eq, 7)

    def _residual_eq(z, F_eq):
        # Poisson residual at EQ nodes: K*u - F  (finite-diff Laplacian)
        u_st = _u_st(z)  # (n_eq, 7): [center, -x, +x, -y, +y, -z, +z]
        lap = (6*u_st[:,0] - u_st[:,1] - u_st[:,2]
               - u_st[:,3] - u_st[:,4] - u_st[:,5] - u_st[:,6]) / dx**2
        return lap - F_eq  # residual = -∇²u - f = 0 at interior nodes

    # ── NM-ROM solver (EQ-sparse explicit Jacobian, no CG) ────────────────
    def make_rom_solver_eq(_max_iters, _gn_tol):
        @jax.jit
        def rom_solve(z_init, F_eq):
            R0 = _residual_eq(z_init, F_eq)
            norm0 = jnp.maximum(jnp.sqrt(jnp.dot(eq_w_jnp * R0, R0)), 1e-30)

            def body_fn(state):
                z, _, i = state
                R = _residual_eq(z, F_eq)
                J = jax.jacfwd(lambda zz: _residual_eq(zz, F_eq))(z)  # (n_eq, k_dim)
                WJ = eq_w_jnp[:, None] * J   # (n_eq, k_dim)
                JtWJ = J.T @ WJ              # (k_dim, k_dim)
                JtWR = J.T @ (eq_w_jnp * R)  # (k_dim,)
                lam = jnp.maximum(1e-4 * jnp.trace(JtWJ) / k_dim, 1e-8)
                dz = jnp.linalg.solve(JtWJ + lam * jnp.eye(k_dim), -JtWR)
                norm_r = jnp.sqrt(jnp.dot(eq_w_jnp * R, R))
                return z + dz, norm_r, i + 1

            def cond_fn(state):
                _, norm_r, i = state
                return (norm_r > _gn_tol * norm0) & (i < _max_iters)

            z_f, res_f, iters = jax.lax.while_loop(
                cond_fn, body_fn, (z_init, norm0, jnp.array(0, jnp.int32)))
            return z_f, iters

        return rom_solve

    rom_solve_eq = make_rom_solver_eq(max_iters, gn_tol)

    def decode_full(z):
        h = _h(z)
        u3 = jnp.einsum('r,ri,rj,rk->ijk', h, jnp.array(W_x_np), jnp.array(W_y_np), jnp.array(W_z_np))
        return mask * (u3.flatten() + float(dec_p['bias'])) + u_g

    # ── Benchmark ─────────────────────────────────────────────────────────
    # Same 10 test cases as N128_test10 — same seed=9999 uniform on [1,3]^3
    _rng_test = np.random.default_rng(9999)
    _new_freqs = _rng_test.uniform(1.0, 3.0, size=(6, 3))
    TEST_CASES = [(1.5, 2.3, 1.8), (2.1, 1.4, 2.7), (1.2, 2.8, 1.3), (2.5, 1.7, 2.2)] + \
                 [(float(f[0]), float(f[1]), float(f[2])) for f in _new_freqs]
    print(f"[retrain] Using {len(TEST_CASES)} test cases (4 original + 6 new)")
    u_guess = jnp.zeros(num_nodes)

    # Warmup — run both FOM and ROM twice to ensure full JIT compilation
    F_w = get_F_3d(*TEST_CASES[0])
    F_w_eq = F_w[eq_idx_jnp]
    jax_linalg.cg(K_op_3d, F_w, x0=u_guess, tol=1e-6)[0].block_until_ready()
    z_w, _ = rom_solve_eq(jnp.zeros(k_dim), F_w_eq)
    decode_full(z_w).block_until_ready()
    # second pass — ensures decode_full JIT is also compiled
    z_w, _ = rom_solve_eq(jnp.zeros(k_dim), F_w_eq)
    decode_full(z_w).block_until_ready()
    jax.effects_barrier()

    fom_times, rom_times, rel_errors, gn_iters_list = [], [], [], []
    for k1, k2, k3 in TEST_CASES:
        F_test = get_F_3d(k1, k2, k3)
        F_test_eq = F_test[eq_idx_jnp]

        t0 = time.time()
        u_fom, _ = jax_linalg.cg(K_op_3d, F_test, x0=u_guess, tol=1e-6)
        u_fom.block_until_ready(); fom_t = time.time() - t0

        t0 = time.time()
        z_f, iters = rom_solve_eq(jnp.zeros(k_dim), F_test_eq)
        u_pred = decode_full(z_f)
        u_pred.block_until_ready(); rom_t = time.time() - t0

        rel_l2 = float(jnp.linalg.norm(u_fom - u_pred) / jnp.linalg.norm(u_fom))
        fom_times.append(fom_t); rom_times.append(rom_t)
        rel_errors.append(rel_l2); gn_iters_list.append(int(iters))
        print(f"   case ({k1},{k2},{k3}) FOM {fom_t:.4f}s ROM {rom_t:.4f}s "
              f"speedup {fom_t/rom_t:.2f}x relL2 {rel_l2:.4e}")

    avg_fom = float(np.mean(fom_times))
    avg_rom = float(np.mean(rom_times))
    speedup = float(np.median([f/r for f, r in zip(fom_times, rom_times)]))
    val_rel_l2 = float(np.mean(rel_errors))
    median_rel_l2 = float(np.median(rel_errors))
    max_rel_l2 = float(np.max(rel_errors))
    min_rel_l2 = float(np.min(rel_errors))
    gn_iters_mean = float(np.mean(gn_iters_list))
    print(f"\n=== {len(rel_errors)}-case stats ===")
    print(f"Mean rel-L2:    {val_rel_l2:.4e}")
    print(f"Median rel-L2:  {median_rel_l2:.4e}")
    print(f"Min rel-L2:     {min_rel_l2:.4e}")
    print(f"Max rel-L2:     {max_rel_l2:.4e}")
    print(f"Sub-1e-2 count: {sum(1 for e in rel_errors if e < 1e-2)}/{len(rel_errors)}")
    print(f"Per-case errors:")
    for _i, (_tc, _e) in enumerate(zip(TEST_CASES, rel_errors)):
        print(f"  case {_i+1} ({_tc[0]:.2f},{_tc[1]:.2f},{_tc[2]:.2f}): {_e:.4e}")
    print(f"Median speedup {speedup:.2f}x")

    try:
        import jax._src.lib as jlib
        peak_vram_mb = 0.0
    except Exception:
        peak_vram_mb = 0.0

    return dict(
        val_rel_l2=val_rel_l2,
        speedup=speedup,
        fom_seconds=avg_fom,
        rom_seconds=avg_rom,
        gn_iters_mean=gn_iters_mean,
        training_seconds=training_seconds,
        peak_vram_mb=peak_vram_mb,
        n_params=n_params,
        N=N, k_dim=k_dim, rank=rank, n_eq=n_eq,
    )
