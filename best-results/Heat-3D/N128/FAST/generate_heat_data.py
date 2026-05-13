"""
generate_heat_data.py
─────────────────────
Generate 3D Heat Equation training data for different grid resolutions.

Usage:
    python generate_heat_data.py --grid 64
    python generate_heat_data.py --grid 128
    python generate_heat_data.py --grid 64 128   # Both

Outputs:
    data/training_data_64.pkl
    data/training_data_128.pkl
"""

import argparse
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.stats import qmc
import pickle
import time
import sys
from pathlib import Path

# ─────────────────────────────────────────
# 0. Paths & Logging
# ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
OUT = SCRIPT_DIR / 'shared_data'
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = SCRIPT_DIR / 'data_generation.log'
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = TeeLogger(LOG_FILE)

# ─────────────────────────────────────────
# 1. Physics Parameters
# ─────────────────────────────────────────
L         = 1.0
dt        = 0.005
NUM_STEPS = 50
MAX_GAUSS = 3   # fixed max Gaussians per IC (pad unused ones with zero amplitude)

# Per-resolution sizing (memory-aware). At 128^3 the full 500-traj set ~440 GB
# won't fit in a 160 GB SLURM job; drop to 200 train / 20 val (~86 GB).
RES_DEFAULTS = {
    32:  dict(N_TRAIN=500, N_VAL=50, TRAJ_BATCH=20),
    64:  dict(N_TRAIN=500, N_VAL=50, TRAJ_BATCH=20),
    128: dict(N_TRAIN=100, N_VAL=10, TRAJ_BATCH=4),
}


def generate_data_for_grid(N: int):
    d = RES_DEFAULTS.get(N, dict(N_TRAIN=200, N_VAL=20, TRAJ_BATCH=4))
    N_TRAIN    = d['N_TRAIN']
    N_VAL      = d['N_VAL']
    TRAJ_BATCH = d['TRAJ_BATCH']
    num_nodes = N ** 3
    dx = L / (N - 1)

    print(f"\n{'='*60}")
    print(f"  Generating data for {N}³ = {num_nodes:,} grid")
    print(f"{'='*60}")
    print(f"  dt={dt}  T={dt*NUM_STEPS:.3f}s  dx={dx:.6f}")

    # ── Grid coordinates ──────────────────────────────────────────
    x_sp = jnp.linspace(0, L, N)
    y_sp = jnp.linspace(0, L, N)
    z_sp = jnp.linspace(0, L, N)
    X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

    # ── Stencil operator ─────────────────────────────────────────
    def K_op_3d(u_flat):
        u   = u_flat.reshape((N, N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1,1:-1,1:-1].set(
            (6*u[1:-1,1:-1,1:-1]
             - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
             - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
             - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
        )
        # Boundary rows stay zero (Dirichlet) — already zero from jnp.zeros_like
        return out.flatten()

    def implicit_op(u_flat, kappa):
        return u_flat + dt * kappa * K_op_3d(u_flat)

    # ── Vectorised Gaussian IC (fixed MAX_GAUSS slots, padded) ───
    # centers: (MAX_GAUSS, 3), amplitudes: (MAX_GAUSS,), widths: (MAX_GAUSS,)
    def make_gaussian_ic_vec(centers, amplitudes, widths):
        # centers (MAX_GAUSS,3), amplitudes (MAX_GAUSS,), widths (MAX_GAUSS,)
        # Compute all Gaussians at once via broadcasting
        cx = centers[:, 0][:, None, None, None]  # (G,1,1,1)
        cy = centers[:, 1][:, None, None, None]
        cz = centers[:, 2][:, None, None, None]
        A  = amplitudes[:, None, None, None]
        s  = widths[:, None, None, None]
        u  = jnp.sum(A * jnp.exp(
            -((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * s**2)
        ), axis=0)   # (N,N,N)
        # Zero out boundaries
        u = u.at[0,:,:].set(0.).at[-1,:,:].set(0.)
        u = u.at[:,0,:].set(0.).at[:,-1,:].set(0.)
        u = u.at[:,:,0].set(0.).at[:,:,-1].set(0.)
        return u.flatten()

    # ── JIT-compiled single-trajectory rollout via lax.scan ──────
    @jax.jit
    def run_fom(u0_flat, kappa):
        """
        Backward-Euler + CG rollout. Uses lax.scan so the whole
        time loop is compiled into a single XLA program.
        Returns (NUM_STEPS+1, num_nodes) snapshot array.
        """
        op = lambda v: implicit_op(v, kappa)

        def step(u, _):
            u_next, _ = jax_linalg.cg(op, u, x0=u, tol=1e-6, maxiter=300)
            return u_next, u_next

        _, snapshots = jax.lax.scan(step, u0_flat, None, length=NUM_STEPS)
        # snapshots: (NUM_STEPS, num_nodes) — prepend IC
        return jnp.concatenate([u0_flat[None], snapshots], axis=0)

    # ── vmap-able IC + rollout ────────────────────────────────────
    @jax.jit
    def run_batch(centers_b, amplitudes_b, widths_b, kappas_b):
        """
        centers_b:    (B, MAX_GAUSS, 3)
        amplitudes_b: (B, MAX_GAUSS)
        widths_b:     (B, MAX_GAUSS)
        kappas_b:     (B,)
        Returns:      (B, NUM_STEPS+1, num_nodes)
        """
        def single(centers, amplitudes, widths, kappa):
            u0 = make_gaussian_ic_vec(centers, amplitudes, widths)
            return run_fom(u0, kappa)
        return jax.vmap(single)(centers_b, amplitudes_b, widths_b, kappas_b)

    # ── LHS Parameter Sampling → fixed-size arrays ───────────────
    def sample_trajectory_params(rng, n_traj):
        sampler = qmc.LatinHypercube(d=13, seed=rng)
        samples = sampler.random(n=n_traj)

        centers_arr    = np.zeros((n_traj, MAX_GAUSS, 3))
        amplitudes_arr = np.zeros((n_traj, MAX_GAUSS))
        widths_arr     = np.zeros((n_traj, MAX_GAUSS))
        kappas_arr     = np.zeros(n_traj)
        params_list    = []

        for i, s in enumerate(samples):
            n_gauss = int(np.round(1 + 2 * s[0]))
            for g in range(n_gauss):
                centers_arr[i, g, 0] = 0.15 + 0.70 * s[1 + g*3]
                centers_arr[i, g, 1] = 0.15 + 0.70 * s[2 + g*3]
                centers_arr[i, g, 2] = 0.15 + 0.70 * s[3 + g*3]
                amplitudes_arr[i, g] = 1.0 + 9.0 * s[10]
                widths_arr[i, g]     = 0.05 + 0.15 * s[11]
            # Unused Gaussian slots: amplitude=0 → zero contribution
            kappas_arr[i] = float(np.exp(
                np.log(0.01) + (np.log(0.5) - np.log(0.01)) * s[12]
            ))
            params_list.append(dict(
                centers=[(centers_arr[i,g,0], centers_arr[i,g,1], centers_arr[i,g,2])
                         for g in range(n_gauss)],
                amplitudes=[amplitudes_arr[i,g] for g in range(n_gauss)],
                widths=[widths_arr[i,g] for g in range(n_gauss)],
                kappa=kappas_arr[i]
            ))

        return (jnp.array(centers_arr),
                jnp.array(amplitudes_arr),
                jnp.array(widths_arr),
                jnp.array(kappas_arr),
                params_list)

    # ── Generate Data ─────────────────────────────────────────────
    print(f"\n── Generating {N_TRAIN} training + {N_VAL} validation trajectories ──")
    print(f"   Each: {NUM_STEPS+1} snapshots  →  Total ≈ {N_TRAIN*(NUM_STEPS+1):,}")
    print(f"   Trajectory batch size: {TRAJ_BATCH}")

    (tr_centers, tr_amps, tr_widths,
     tr_kappas, train_params) = sample_trajectory_params(rng=42,   n_traj=N_TRAIN)
    (va_centers, va_amps, va_widths,
     va_kappas, val_params)   = sample_trajectory_params(rng=1337, n_traj=N_VAL)

    def run_all_batched(centers, amps, widths, kappas, label):
        n = len(kappas)
        all_trajs = []
        t0 = time.perf_counter()
        for start in range(0, n, TRAJ_BATCH):
            end = min(start + TRAJ_BATCH, n)
            batch = run_batch(
                centers[start:end], amps[start:end],
                widths[start:end],  kappas[start:end]
            )
            batch.block_until_ready()
            all_trajs.append(np.asarray(batch))  # offload to CPU; N=64 doesn't fit on GPU
            print(f"   {label} {end}/{n}  ({time.perf_counter()-t0:.0f}s)")
        return np.concatenate(all_trajs, axis=0)   # (n, T+1, nodes)

    print("\n  Warming up JIT (first batch compiles)...")
    all_train = run_all_batched(tr_centers, tr_amps, tr_widths, tr_kappas, "Train")
    all_val   = run_all_batched(va_centers, va_amps, va_widths, va_kappas, "Val  ")

    # all_train: (N_TRAIN, NUM_STEPS+1, num_nodes)
    U_train = all_train.reshape(N_TRAIN * (NUM_STEPS+1), num_nodes)
    U_val   = all_val.reshape(N_VAL   * (NUM_STEPS+1), num_nodes)
    print(f"   Training snapshots: {U_train.shape}")
    print(f"   Validation snapshots: {U_val.shape}")

    # per-trajectory snapshot lists (for backward-compat with train_heat_ae.py)
    all_snapshots = [np.array(all_train[i]) for i in range(N_TRAIN)]
    val_snapshots = [np.array(all_val[i])   for i in range(N_VAL)]
    traj_kappas   = [float(tr_kappas[i]) for i in range(N_TRAIN)]
    val_kappas    = [float(va_kappas[i]) for i in range(N_VAL)]
    traj_starts   = list(range(0, N_TRAIN * (NUM_STEPS+1), NUM_STEPS+1))

    # ── Save ─────────────────────────────────────────────────────
    data_file = OUT / f'training_data_{N}.pkl'
    data_to_save = {
        'U_train': np.array(U_train),
        'U_val':   np.array(U_val),
        'all_snapshots': all_snapshots,
        'val_snapshots': val_snapshots,
        'train_params':  train_params,
        'val_params':    val_params,
        'traj_kappas':   traj_kappas,
        'val_kappas':    val_kappas,
        'traj_starts':   traj_starts,
        'grid_config': {
            'N': N, 'L': L, 'dx': float(dx), 'dt': dt, 'NUM_STEPS': NUM_STEPS
        }
    }
    with open(data_file, 'wb') as f:
        pickle.dump(data_to_save, f)

    file_size_mb = data_file.stat().st_size / (1024 * 1024)
    print(f"\n   Saved: {data_file}  ({file_size_mb:.1f} MB)")
    return data_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D Heat Equation training data'
    )
    parser.add_argument('--grid', '-g', type=int, nargs='+', default=[64])
    args = parser.parse_args()

    print("="*60)
    print("  3D Heat Equation Data Generator")
    print("="*60)
    print(f"  Grid sizes: {args.grid}")
    print(f"  Time steps: {NUM_STEPS}  (dt={dt}, T={dt*NUM_STEPS:.3f}s)")
    print(f"  Per-resolution sizing: {RES_DEFAULTS}")

    for grid_size in args.grid:
        generate_data_for_grid(grid_size)

    print("\n" + "="*60)
    print("  Data generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
