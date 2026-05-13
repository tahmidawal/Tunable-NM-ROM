"""
generate_poisson_data_cg.py
---------------------------
CG-based Poisson-3D datagen (diagnostic).

Solves the *discrete* FEM system -K u = F via CG, producing the u_FEM that
the ROM is compared against at test time. This is the original pipeline,
kept as a separate script for A/B diagnosis against the analytical version.

Writes to shared_data/poisson_data_cg_{N}.npz (distinct filename so both
datasets can coexist).
"""

import argparse
import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--n-train', type=int, default=500)
    ap.add_argument('--n-val', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cg-tol', type=float, default=1e-6)
    ap.add_argument('--cg-iters', type=int, default=5000)
    ap.add_argument('--outdir', default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).parent.resolve()
    outdir = Path(args.outdir) if args.outdir else script_dir / 'shared_data'
    outdir.mkdir(parents=True, exist_ok=True)

    N = args.N
    num_nodes = N ** 3
    L = 1.0
    dx = L / (N - 1)
    out_path = outdir / f'poisson_data_cg_{N}.npz'

    print(f"=== Poisson-3D CG datagen: N={N} ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"num_nodes = {num_nodes:,}  cg_tol={args.cg_tol}  iters={args.cg_iters}")
    print(f"output: {out_path}")

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

    # Warm-start CG from the analytical solution — essentially exact up to FEM
    # discretization error, so CG converges in a few iterations. Final u is
    # the FEM-consistent discrete solution (not the continuous one).
    def get_analytical(k1, k2, k3):
        c = 10.0 / ((k1 ** 2 + k2 ** 2 + k3 ** 2) * jnp.pi ** 2)
        u = (c * jnp.sin(k1 * jnp.pi * X)
                * jnp.sin(k2 * jnp.pi * Y)
                * jnp.sin(k3 * jnp.pi * Z))
        u = u.at[0, :, :].set(0.).at[-1, :, :].set(0.)
        u = u.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        u = u.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        return u.flatten()

    @jax.jit
    def fom_solve(k1, k2, k3):
        F = get_F_3d(k1, k2, k3)
        u0 = get_analytical(k1, k2, k3)   # warm start
        u, _ = jax_linalg.cg(K_op_3d, F, x0=u0,
                             tol=args.cg_tol, maxiter=args.cg_iters)
        return u

    rng = np.random.RandomState(args.seed)
    train_freqs = rng.uniform(1.0, 3.0, size=(args.n_train, 3)).astype(np.float32)
    val_freqs = rng.uniform(1.0, 3.0, size=(args.n_val, 3)).astype(np.float32)

    U_train = np.empty((args.n_train, num_nodes), dtype=np.float32)
    U_val = np.empty((args.n_val, num_nodes), dtype=np.float32)

    print("\n--- warm-up (JIT) ---")
    _u = fom_solve(*train_freqs[0]); _u.block_until_ready()
    print("warm-up done\n")

    t_global = time.perf_counter()
    print("--- train ---")
    for i, (k1, k2, k3) in enumerate(train_freqs):
        t0 = time.perf_counter()
        u = fom_solve(float(k1), float(k2), float(k3))
        u.block_until_ready()
        U_train[i] = np.asarray(u)
        if (i + 1) % 50 == 0 or i == 0:
            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_global
            eta = elapsed / (i + 1) * (args.n_train - i - 1)
            print(f"  [{i+1:4d}/{args.n_train}] last {dt*1000:.1f}ms  elapsed {elapsed:.1f}s  eta {eta:.0f}s")
            sys.stdout.flush()

    print("\n--- val ---")
    for i, (k1, k2, k3) in enumerate(val_freqs):
        u = fom_solve(float(k1), float(k2), float(k3))
        u.block_until_ready()
        U_val[i] = np.asarray(u)

    print(f"\nTotal: {time.perf_counter() - t_global:.0f}s")
    print(f"U_train {U_train.shape}  U_val {U_val.shape}")

    # Sanity: compute analytical-vs-FEM gap on a couple samples
    print("\n--- analytical vs FEM gap (sanity) ---")
    for idx in [0, 10, 100]:
        k1, k2, k3 = train_freqs[idx]
        u_ana = np.asarray(get_analytical(float(k1), float(k2), float(k3)))
        u_fem = U_train[idx]
        rel = np.linalg.norm(u_ana - u_fem) / np.linalg.norm(u_fem)
        print(f"  sample {idx}: relL2(analytical vs FEM) = {rel:.3e}")

    print(f"\nSaving to {out_path} ...")
    np.savez(out_path, U_train=U_train, U_val=U_val,
             train_freqs=train_freqs, val_freqs=val_freqs)
    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"Saved ({size_gb:.2f} GB)")


if __name__ == '__main__':
    main()
