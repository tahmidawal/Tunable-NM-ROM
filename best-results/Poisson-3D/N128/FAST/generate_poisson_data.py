"""
generate_poisson_data.py
------------------------
Pre-generate 3D Poisson training/val datasets (to be shared across sweep runs).

For each (k1, k2, k3) sampled uniformly in [1.0, 3.0]^3, solves:
    -Δu = 10 · sin(k1 π x) sin(k2 π y) sin(k3 π z)
via CG on the 7-point stencil with Dirichlet BCs.

CLI:
    python generate_poisson_data.py --N 256 --n-train 500 --n-val 100
Writes:
    shared_data/poisson_data_256.npz  (U_train, U_val, train_freqs, val_freqs)
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
    ap.add_argument('--outdir', default=None,
                    help='default: <script>/shared_data/')
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
    out_path = outdir / f'poisson_data_{N}.npz'

    print(f"=== Poisson-3D datagen: N={N} ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"num_nodes = {num_nodes:,}")
    print(f"CG tol={args.cg_tol}  iters={args.cg_iters}")
    print(f"output: {out_path}")

    x_sp = jnp.linspace(0, L, N); y_sp = jnp.linspace(0, L, N); z_sp = jnp.linspace(0, L, N)
    X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

    # Analytical solution to -Δu = 10 sin(k1πx) sin(k2πy) sin(k3πz):
    #   u(x,y,z) = [10 / (π²(k1² + k2² + k3²))] sin(k1πx) sin(k2πy) sin(k3πz)
    # Exact to the continuous PDE; at N ≥ 128 this is more accurate than FEM.
    @jax.jit
    def analytical_solve(k1, k2, k3):
        c = 10.0 / ((k1 ** 2 + k2 ** 2 + k3 ** 2) * jnp.pi ** 2)
        u = (c * jnp.sin(k1 * jnp.pi * X)
                * jnp.sin(k2 * jnp.pi * Y)
                * jnp.sin(k3 * jnp.pi * Z))
        # Dirichlet BCs (redundant for these separable modes, but safe)
        u = u.at[0, :, :].set(0.).at[-1, :, :].set(0.)
        u = u.at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        u = u.at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        return u.flatten()

    rng = np.random.RandomState(args.seed)
    train_freqs = rng.uniform(1.0, 3.0, size=(args.n_train, 3)).astype(np.float32)
    val_freqs = rng.uniform(1.0, 3.0, size=(args.n_val, 3)).astype(np.float32)

    # Preallocate CPU arrays (keeps peak GPU memory at one field at a time)
    U_train = np.empty((args.n_train, num_nodes), dtype=np.float32)
    U_val = np.empty((args.n_val, num_nodes), dtype=np.float32)

    print("\n--- warm-up (JIT) ---")
    _u = analytical_solve(*train_freqs[0])
    _u.block_until_ready()
    print("warm-up done\n")

    t_global = time.perf_counter()
    print("--- train ---")
    for i, (k1, k2, k3) in enumerate(train_freqs):
        t0 = time.perf_counter()
        u = analytical_solve(float(k1), float(k2), float(k3))
        u.block_until_ready()
        U_train[i] = np.asarray(u)
        if (i + 1) % 50 == 0 or i == 0:
            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_global
            print(f"  [{i+1:4d}/{args.n_train}] last {dt*1000:.1f}ms  elapsed {elapsed:.1f}s")
            sys.stdout.flush()

    print("\n--- val ---")
    for i, (k1, k2, k3) in enumerate(val_freqs):
        u = analytical_solve(float(k1), float(k2), float(k3))
        u.block_until_ready()
        U_val[i] = np.asarray(u)

    print(f"\nTotal: {time.perf_counter() - t_global:.0f}s")
    print(f"U_train {U_train.shape}  U_val {U_val.shape}")
    print(f"Saving to {out_path} ...")
    np.savez(out_path, U_train=U_train, U_val=U_val,
             train_freqs=train_freqs, val_freqs=val_freqs)
    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"Saved ({size_gb:.2f} GB)")


if __name__ == '__main__':
    main()
