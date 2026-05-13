"""
generate_heat_data_2d.py
─────────────────────────
Generate 2D Heat Equation training data on a 64² uniform FD grid.

PDE:  ∂u/∂t = κ∇²u,  u=0 on ∂Ω,  Ω = [0,1]²
ICs:  1–3 Gaussian blobs,  κ ∈ [0.01, 0.5] (log-uniform)

Output:  data/training_data_2d_64.pkl
"""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.stats import qmc
import pickle
import time
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUT        = SCRIPT_DIR / 'data'
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = SCRIPT_DIR / 'data_generation.log'
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log      = open(filename, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message); self.terminal.flush()
        self.log.write(message);      self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
sys.stdout = TeeLogger(LOG_FILE)

# ─────────────────────────────────────────
# Physics parameters
# ─────────────────────────────────────────
N          = 64
L          = 1.0
dt         = 0.005
NUM_STEPS  = 50
N_TRAIN    = 500
N_VAL      = 50
MAX_GAUSS  = 3
TRAJ_BATCH = 20

num_nodes = N * N
dx        = L / (N - 1)

print(f"JAX devices: {jax.devices()}")
print(f"2D Heat Data Generator")
print(f"Grid: {N}² = {num_nodes:,} nodes  |  dt={dt}  T={dt*NUM_STEPS:.3f}s")
print(f"Trajectories: {N_TRAIN} train + {N_VAL} val")

# ─────────────────────────────────────────
# Grid
# ─────────────────────────────────────────
x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
X, Y = jnp.meshgrid(x_sp, y_sp, indexing='ij')

# ─────────────────────────────────────────
# 5-point Laplacian + implicit operator
# ─────────────────────────────────────────
def K_op_2d(u_flat):
    u   = u_flat.reshape((N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1, 1:-1].set(
        (4*u[1:-1,1:-1]
         - u[0:-2,1:-1] - u[2:,1:-1]
         - u[1:-1,0:-2] - u[1:-1,2:]) / dx**2
    )
    return out.flatten()

def implicit_op(u_flat, kappa):
    return u_flat + dt * kappa * K_op_2d(u_flat)

# ─────────────────────────────────────────
# Vectorised Gaussian IC (padded MAX_GAUSS slots)
# ─────────────────────────────────────────
def make_gaussian_ic_vec(centers, amplitudes, widths):
    # centers: (MAX_GAUSS, 2), amplitudes: (MAX_GAUSS,), widths: (MAX_GAUSS,)
    cx = centers[:, 0][:, None, None]   # (G, 1, 1)
    cy = centers[:, 1][:, None, None]
    A  = amplitudes[:, None, None]
    s  = widths[:, None, None]
    u  = jnp.sum(A * jnp.exp(
        -((X - cx)**2 + (Y - cy)**2) / (2 * s**2)
    ), axis=0)   # (N, N)
    u  = u.at[0,:].set(0.).at[-1,:].set(0.)
    u  = u.at[:,0].set(0.).at[:,-1].set(0.)
    return u.flatten()

# ─────────────────────────────────────────
# JIT rollout
# ─────────────────────────────────────────
@jax.jit
def run_fom(u0_flat, kappa):
    op = lambda v: implicit_op(v, kappa)
    def step(u, _):
        u_next, _ = jax_linalg.cg(op, u, x0=u, tol=1e-6, maxiter=300)
        return u_next, u_next
    _, snapshots = jax.lax.scan(step, u0_flat, None, length=NUM_STEPS)
    return jnp.concatenate([u0_flat[None], snapshots], axis=0)

@jax.jit
def run_batch(centers_b, amplitudes_b, widths_b, kappas_b):
    def single(centers, amplitudes, widths, kappa):
        u0 = make_gaussian_ic_vec(centers, amplitudes, widths)
        return run_fom(u0, kappa)
    return jax.vmap(single)(centers_b, amplitudes_b, widths_b, kappas_b)

# ─────────────────────────────────────────
# LHS parameter sampling  (d=10)
# dims: [n_gauss_sel, cx0,cy0, cx1,cy1, cx2,cy2, amp, width, kappa]
# ─────────────────────────────────────────
def sample_trajectory_params(rng, n_traj):
    sampler = qmc.LatinHypercube(d=10, seed=rng)
    samples = sampler.random(n=n_traj)

    centers_arr    = np.zeros((n_traj, MAX_GAUSS, 2))
    amplitudes_arr = np.zeros((n_traj, MAX_GAUSS))
    widths_arr     = np.zeros((n_traj, MAX_GAUSS))
    kappas_arr     = np.zeros(n_traj)
    params_list    = []

    for i, s in enumerate(samples):
        n_gauss = int(np.round(1 + 2 * s[0]))
        for g in range(n_gauss):
            centers_arr[i, g, 0] = 0.15 + 0.70 * s[1 + g*2]   # cx
            centers_arr[i, g, 1] = 0.15 + 0.70 * s[2 + g*2]   # cy
            amplitudes_arr[i, g] = 1.0 + 9.0 * s[7]
            widths_arr[i, g]     = 0.05 + 0.15 * s[8]
        kappas_arr[i] = float(np.exp(
            np.log(0.01) + (np.log(0.5) - np.log(0.01)) * s[9]
        ))
        params_list.append(dict(
            centers=[(centers_arr[i,g,0], centers_arr[i,g,1]) for g in range(n_gauss)],
            amplitudes=[amplitudes_arr[i,g] for g in range(n_gauss)],
            widths=[widths_arr[i,g] for g in range(n_gauss)],
            kappa=kappas_arr[i]
        ))

    return (jnp.array(centers_arr), jnp.array(amplitudes_arr),
            jnp.array(widths_arr),  jnp.array(kappas_arr), params_list)

# ─────────────────────────────────────────
# Generate
# ─────────────────────────────────────────
print(f"\n── Generating {N_TRAIN} train + {N_VAL} val trajectories ──")

(tr_centers, tr_amps, tr_widths,
 tr_kappas, train_params) = sample_trajectory_params(rng=42,   n_traj=N_TRAIN)
(va_centers, va_amps, va_widths,
 va_kappas, val_params)   = sample_trajectory_params(rng=1337, n_traj=N_VAL)

def run_all_batched(centers, amps, widths, kappas, label):
    n = len(kappas)
    all_trajs = []
    t0 = time.perf_counter()
    for start in range(0, n, TRAJ_BATCH):
        end   = min(start + TRAJ_BATCH, n)
        batch = run_batch(centers[start:end], amps[start:end],
                          widths[start:end],  kappas[start:end])
        batch.block_until_ready()
        all_trajs.append(batch)
        print(f"   {label} {end}/{n}  ({time.perf_counter()-t0:.0f}s)")
    return jnp.concatenate(all_trajs, axis=0)   # (n, NUM_STEPS+1, num_nodes)

print("   Warming up JIT (first batch compiles)...")
all_train = run_all_batched(tr_centers, tr_amps, tr_widths, tr_kappas, "Train")
all_val   = run_all_batched(va_centers, va_amps, va_widths, va_kappas, "Val  ")

U_train = all_train.reshape(N_TRAIN * (NUM_STEPS+1), num_nodes)
U_val   = all_val.reshape(N_VAL   * (NUM_STEPS+1), num_nodes)

all_snapshots = [np.array(all_train[i]) for i in range(N_TRAIN)]
val_snapshots = [np.array(all_val[i])   for i in range(N_VAL)]
traj_kappas   = [float(tr_kappas[i]) for i in range(N_TRAIN)]
val_kappas_   = [float(va_kappas[i]) for i in range(N_VAL)]
traj_starts   = list(range(0, N_TRAIN * (NUM_STEPS+1), NUM_STEPS+1))

print(f"\nTraining snapshots : {U_train.shape}")
print(f"Validation snapshots: {U_val.shape}")

# ─────────────────────────────────────────
# Save
# ─────────────────────────────────────────
data_file = OUT / 'training_data_2d_64.pkl'
data_to_save = {
    'U_train':      np.array(U_train),
    'U_val':        np.array(U_val),
    'all_snapshots': all_snapshots,
    'val_snapshots': val_snapshots,
    'train_params':  train_params,
    'val_params':    val_params,
    'traj_kappas':   traj_kappas,
    'val_kappas':    val_kappas_,
    'traj_starts':   traj_starts,
    'grid_config': {
        'N': N, 'L': L, 'dx': float(dx), 'dt': dt, 'NUM_STEPS': NUM_STEPS
    }
}
with open(data_file, 'wb') as f:
    pickle.dump(data_to_save, f)

file_size_mb = data_file.stat().st_size / (1024 * 1024)
print(f"\nSaved: {data_file}  ({file_size_mb:.1f} MB)")
print("\n=== Data generation complete ===")
