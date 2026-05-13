"""
poisson_nmrom_sweep.py
----------------------
Unified Poisson-2D NM-ROM driver for sweep runs.

- EQ is the default code path; --eq-mode full reproduces the non-EQ baseline
  by selecting all interior nodes with unit weights (no NNLS).
- All outputs (dataset, checkpoint, logs, results.tsv, plots) live in --outdir.
- Knobs exposed as CLI flags: N, k_dim, rank, hidden_dim, patch_size,
  embed_dim, num_heads, num_enc_layers, num_epochs, batch_size,
  max_iters, gn_tol, cg_tol, cg_iters, eq_mode, n_eq_snaps, min_eq_points.
"""

import argparse
import hashlib
import json
import pickle
import sys
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


# ==========================================
# 0. CLI
# ==========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir',         type=str,   required=True)
    p.add_argument('--N',              type=int,   default=64)
    p.add_argument('--k-dim',          type=int,   default=12)
    p.add_argument('--rank',           type=int,   default=512)
    p.add_argument('--hidden-dim',     type=int,   default=256)
    p.add_argument('--patch-size',     type=int,   default=8)
    p.add_argument('--embed-dim',      type=int,   default=64)
    p.add_argument('--num-heads',      type=int,   default=4)
    p.add_argument('--num-enc-layers', type=int,   default=4)
    p.add_argument('--num-epochs',     type=int,   default=100_000)
    p.add_argument('--batch-size',     type=int,   default=32)
    p.add_argument('--peak-lr',        type=float, default=1e-3)
    p.add_argument('--weight-decay',   type=float, default=5e-4)
    p.add_argument('--n-train',        type=int,   default=700)
    p.add_argument('--n-val',          type=int,   default=140)
    # Solver knobs
    p.add_argument('--max-iters',      type=int,   default=8)
    p.add_argument('--gn-tol',         type=float, default=1e-6)
    p.add_argument('--cg-tol',         type=float, default=1e-3)
    p.add_argument('--cg-iters',       type=int,   default=0,
                   help='0 = unlimited; else cap on inner CG iterations')
    # EQ knobs
    p.add_argument('--eq-mode',        type=str,   default='nnls',
                   choices=['nnls', 'full'],
                   help='nnls = NNLS-selected EQ nodes; full = all interior nodes, uniform weights (non-EQ baseline)')
    p.add_argument('--n-eq-snaps',     type=int,   default=80)
    p.add_argument('--min-eq-points',  type=int,   default=200)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--label',          type=str,   default='unlabeled')
    p.add_argument('--use-checkpoint', action='store_true')
    p.add_argument('--skip-checkpoint',action='store_true')
    return p.parse_args()


args = parse_args()
OUTDIR = Path(args.outdir).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / 'plots').mkdir(exist_ok=True)

# ==========================================
# 1. Logging
# ==========================================
LOG_FILE = OUTDIR / 'training.log'
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
sys.stdout = TeeLogger(LOG_FILE)

print(f"=== Poisson-2D NM-ROM sweep run ===")
print(f"Label  : {args.label}")
print(f"Outdir : {OUTDIR}")
print(f"Args   : {json.dumps(vars(args), indent=2)}")

# Persist the actual config used
with open(OUTDIR / 'config_used.json', 'w') as f:
    json.dump(vars(args), f, indent=2)

# ==========================================
# 2. Domain
# ==========================================
N         = args.N
num_nodes = N ** 2
k_dim     = args.k_dim
L         = 1.0
dx        = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
X, Y = jnp.meshgrid(x_sp, y_sp, indexing='ij')

def K_op_2d(u_flat):
    u = u_flat.reshape((N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1, 1:-1].set(
        (4*u[1:-1,1:-1]
         - u[0:-2,1:-1] - u[2:,1:-1]
         - u[1:-1,0:-2] - u[1:-1,2:]) / dx**2
    )
    out = out.at[0,:].set(u[0,:])
    out = out.at[-1,:].set(u[-1,:])
    out = out.at[:,0].set(u[:,0])
    out = out.at[:,-1].set(u[:,-1])
    return out.flatten()

def get_F_2d(k1, k2):
    F = jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * 10.0
    F = F.at[0,:].set(0.).at[-1,:].set(0.)
    F = F.at[:,0].set(0.).at[:,-1].set(0.)
    return F.flatten()

mask_2d = jnp.ones((N, N))
mask_2d = mask_2d.at[0,:].set(0.).at[-1,:].set(0.)
mask_2d = mask_2d.at[:,0].set(0.).at[:,-1].set(0.)
mask    = mask_2d.flatten()
u_g     = jnp.zeros(num_nodes)

# ==========================================
# 3. Model
# ==========================================
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


class LinearCPDecoder(nn.Module):
    latent_dim: int
    rank:       int = 512
    grid_size:  int = 64
    hidden_dim: int = 256

    def setup(self):
        self.W1       = nn.Dense(self.hidden_dim)
        self.W2       = nn.Dense(self.hidden_dim)
        self.W_rank   = nn.Dense(self.rank)
        self.W_direct = nn.Dense(self.rank)
        init = nn.initializers.normal(0.01)
        self.W_x  = self.param('W_x',  init, (self.rank, self.grid_size))
        self.W_y  = self.param('W_y',  init, (self.rank, self.grid_size))
        self.bias = self.param('bias', nn.initializers.zeros, ())

    def __call__(self, z):
        h_nl  = nn.swish(self.W1(z))
        h_nl  = nn.swish(self.W2(h_nl))
        h_nl  = self.W_rank(h_nl)
        h_lin = self.W_direct(z)
        h     = h_lin + h_nl
        u_2d  = jnp.einsum('r,ri,rj->ij', h, self.W_x, self.W_y)
        return u_2d.flatten() + self.bias


class ViTLinearCPAutoencoder(nn.Module):
    latent_dim:     int
    num_nodes:      int
    patch_size:     int = 8
    embed_dim:      int = 64
    num_heads:      int = 4
    num_enc_layers: int = 4
    rank:           int = 512
    hidden_dim:     int = 64

    def setup(self):
        grid_n     = round(self.num_nodes ** (1/2))
        n_per_side = grid_n // self.patch_size
        patch_dim  = self.patch_size ** 2
        self._grid_n      = grid_n
        self._n_per_side  = n_per_side
        self._num_patches = n_per_side ** 2
        self._patch_dim   = patch_dim
        self.patch_embed = nn.Dense(self.embed_dim)
        self.enc_pos     = self.param('enc_pos',
            nn.initializers.normal(stddev=0.02),
            (self._num_patches, self.embed_dim))
        self.enc_blocks = [TransformerBlock(self.embed_dim, self.num_heads)
                           for _ in range(self.num_enc_layers)]
        self.enc_norm = nn.LayerNorm()
        self.enc_proj = nn.Dense(self.latent_dim)
        self.decoder = LinearCPDecoder(
            latent_dim=self.latent_dim, rank=self.rank,
            grid_size=self._grid_n, hidden_dim=self.hidden_dim,
        )

    def _patchify(self, u_flat):
        n, p = self._n_per_side, self.patch_size
        return (u_flat.reshape(n, p, n, p).transpose(0, 2, 1, 3)
                    .reshape(self._num_patches, self._patch_dim))

    def encode(self, u):
        x = self._patchify(u)
        x = self.patch_embed(x) + self.enc_pos
        for b in self.enc_blocks:
            x = b(x)
        x = self.enc_norm(x).mean(axis=0)
        return self.enc_proj(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, u):
        return self.decode(self.encode(u))

# ==========================================
# 4. Data
# ==========================================
def full_order_fem_solver(F_vec, u_guess=None, tol=1e-6):
    if u_guess is None:
        u_guess = jnp.zeros(num_nodes)
    u, _ = jax_linalg.cg(K_op_2d, F_vec, x0=u_guess, tol=tol)
    return u

print(f"\nJAX devices: {jax.devices()}")
print(f"Grid: {N}^2 = {num_nodes:,} nodes")

rng = np.random.RandomState(args.seed)
N_train, N_val = args.n_train, args.n_val
train_freqs = rng.uniform(1.0, 3.0, size=(N_train, 2))
val_freqs   = rng.uniform(1.0, 3.0, size=(N_val,   2))
u_guess = jnp.zeros(num_nodes)

DATASET_PATH = OUTDIR / 'dataset.npz'
need_generate = True
if DATASET_PATH.exists():
    ds = np.load(DATASET_PATH)
    if ds['U_train'].shape[0] == N_train and ds['U_val'].shape[0] == N_val \
            and ds['U_train'].shape[1] == num_nodes:
        print("   Loading cached dataset...")
        U_train = jnp.array(ds['U_train'])
        U_val   = jnp.array(ds['U_val'])
        need_generate = False

if need_generate:
    print("   Generating training snapshots...")
    U_train_list = []
    for i, (k1, k2) in enumerate(train_freqs):
        U_train_list.append(full_order_fem_solver(get_F_2d(k1, k2), u_guess))
        if (i+1) % 50 == 0:
            print(f"   Solved {i+1}/{N_train}")
    U_val_list = [full_order_fem_solver(get_F_2d(k1, k2), u_guess)
                  for k1, k2 in val_freqs]
    U_train = jnp.stack(U_train_list); U_val = jnp.stack(U_val_list)
    np.savez(DATASET_PATH, U_train=np.array(U_train), U_val=np.array(U_val))
print(f"   Train: {U_train.shape}  Val: {U_val.shape}")

# ==========================================
# 5. Model init + training
# ==========================================
model = ViTLinearCPAutoencoder(
    latent_dim=k_dim, num_nodes=num_nodes,
    patch_size=args.patch_size, embed_dim=args.embed_dim,
    num_heads=args.num_heads, num_enc_layers=args.num_enc_layers,
    rank=args.rank, hidden_dim=args.hidden_dim,
)
key    = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones(num_nodes))['params']
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"   Params: {n_params:,}")

warmup = min(500, max(1, args.num_epochs // 10))
decay  = max(warmup + 1, args.num_epochs)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., peak_value=args.peak_lr,
    warmup_steps=warmup, decay_steps=decay, end_value=1e-6,
)
tx = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)

# Checkpoint fingerprint: architecture + training config only (solver knobs excluded)
_arch_cfg = dict(
    N=N, k_dim=k_dim, rank=args.rank, hidden_dim=args.hidden_dim,
    patch_size=args.patch_size, embed_dim=args.embed_dim,
    num_heads=args.num_heads, num_enc_layers=args.num_enc_layers,
    num_epochs=args.num_epochs, batch_size=args.batch_size,
    peak_lr=args.peak_lr, weight_decay=args.weight_decay,
    n_train=N_train, n_val=N_val, seed=args.seed,
)
_fp = hashlib.md5(str(sorted(_arch_cfg.items())).encode()).hexdigest()[:12]
CKPT_PATH = OUTDIR / f'checkpoint_{_fp}.pkl'

def rel_l2_batch(u_true, u_pred):
    norms = jnp.linalg.norm(u_true - u_pred, axis=1)
    denom = jnp.linalg.norm(u_true, axis=1) + 1e-12
    return jnp.mean(norms / denom)

@jax.jit
def train_step(p, opt_st, batch):
    def loss_fn(w):
        preds = jax.vmap(lambda u: model.apply({'params': w}, u))(batch)
        return rel_l2_batch(batch, preds)
    loss, grads = jax.value_and_grad(loss_fn)(p)
    upd, new_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, upd), new_st, loss

@jax.jit
def eval_rec_err(p, batch):
    preds = jax.vmap(lambda u: model.apply({'params': p}, u))(batch)
    return rel_l2_batch(batch, preds)

if args.use_checkpoint and not args.skip_checkpoint and CKPT_PATH.exists():
    print(f"\n2. --use-checkpoint: loading {CKPT_PATH.name}")
    with open(CKPT_PATH, 'rb') as f:
        params = pickle.load(f)
    print(f"   train rec {float(eval_rec_err(params, U_train)):.4e} | "
          f"val rec {float(eval_rec_err(params, U_val)):.4e}")
else:
    print(f"\n2. Training ({args.num_epochs} epochs, batch={args.batch_size})...")
    opt_state = tx.init(params)
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(1)
    log_every = max(2_000, args.num_epochs // 50)
    for epoch in range(args.num_epochs + 1):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, len(U_train), shape=(args.batch_size,), replace=False)
        params, opt_state, loss = train_step(params, opt_state, U_train[idx])
        if epoch % log_every == 0:
            tr = float(eval_rec_err(params, U_train))
            vl = float(eval_rec_err(params, U_val))
            print(f"   ep {epoch:6d} | loss {float(loss):.4e} | "
                  f"train {tr:.4e} | val {vl:.4e} | {time.perf_counter()-t0:.0f}s")
    print(f"   Done in {time.perf_counter()-t0:.0f}s")
    with open(CKPT_PATH, 'wb') as f:
        pickle.dump(params, f)
    print(f"   Saved {CKPT_PATH.name}")

train_rec = float(eval_rec_err(params, U_train))
val_rec   = float(eval_rec_err(params, U_val))

# ==========================================
# 6. EQ offline phase
# ==========================================
print(f"\n3. EQ offline phase (mode={args.eq_mode})...")

def constrained_decode(z):
    u_hat = model.apply({'params': params}, z, method=model.decode)
    return mask * u_hat + u_g

N2              = N  # 2D: y-stride is N
stencil_offsets = jnp.array([0, -1, 1, -N2, N2])  # center, W, E, S, N
interior_idx_np = np.where(np.array(mask) > 0)[0]

if args.eq_mode == 'full':
    eq_indices    = interior_idx_np
    eq_weights_np = np.ones(len(eq_indices), dtype=np.float64)
    print(f"   Full-grid mode: {len(eq_indices)} interior nodes, uniform weights")
else:
    @jax.jit
    def get_integrand(z_val, F_vec):
        R_full = K_op_2d(constrained_decode(z_val)) - F_vec
        J_D    = jax.jacfwd(constrained_decode)(z_val)   # (N², k)
        return J_D.T * R_full[None, :]

    N_EQ_SNAPS = min(args.n_eq_snaps, len(U_train))
    eq_snap_idx = np.random.RandomState(0).choice(len(U_train), N_EQ_SNAPS, replace=False)
    print(f"   Computing G over {N_EQ_SNAPS} snapshots...")
    G_list = []
    for cnt, i in enumerate(eq_snap_idx):
        z_i = model.apply({'params': params}, U_train[i], method=model.encode)
        k1_i, k2_i = train_freqs[i]
        F_i = get_F_2d(k1_i, k2_i)
        G_list.append(get_integrand(z_i, F_i))
        if (cnt+1) % 20 == 0:
            print(f"   {cnt+1}/{N_EQ_SNAPS}")
    G_train_np = np.array(jnp.concatenate(G_list, axis=0))
    G_train_np[:, np.array(mask) == 0] = 0.0
    b_train_np = np.sum(G_train_np, axis=1)
    print(f"   G shape {G_train_np.shape}   G Frob {np.linalg.norm(G_train_np):.4e}")

    print("   NNLS...")
    w_eq, nnls_res = nnls(G_train_np, b_train_np)
    print(f"   NNLS residual {nnls_res:.4e}   max w {w_eq.max():.4e}   "
          f"nonzero(>1e-14) {np.sum(w_eq > 1e-14)}")

    eq_indices    = np.where(w_eq > 1e-14)[0]
    eq_weights_np = w_eq[eq_indices]

    MIN_EQ = args.min_eq_points
    if len(eq_indices) < MIN_EQ:
        print(f"   Only {len(eq_indices)} EQ pts — falling back to top-{MIN_EQ} by weight")
        top = np.argsort(w_eq)[::-1][:MIN_EQ]
        eq_indices = np.sort(top[w_eq[top] > 0])
        if len(eq_indices) < MIN_EQ:
            col_norms = np.linalg.norm(G_train_np, axis=0)
            sorted_int = interior_idx_np[np.argsort(col_norms[interior_idx_np])[::-1]]
            eq_indices = sorted_int[:MIN_EQ]
            eq_weights_np = np.ones(len(eq_indices))
        else:
            eq_weights_np = w_eq[eq_indices]
    print(f"   EQ nodes: {num_nodes:,} → {len(eq_indices)} "
          f"({100*len(eq_indices)/num_nodes:.3f}%)")

eq_indices_jnp = jnp.array(eq_indices)
eq_weights_jnp = jnp.array(eq_weights_np)
num_eq_points  = len(eq_indices)
eq_stencil_idx = (eq_indices_jnp[:, None] + stencil_offsets[None, :])  # (M, 5)
eq_stencil_flat = eq_stencil_idx.flatten()

# ==========================================
# 7. EQ-ROM solver (LM + backtracking, unified for both modes)
# ==========================================
print(f"\n4. Building EQ-ROM solver (max_iters={args.max_iters}, gn_tol={args.gn_tol})...")

def make_eq_rom_solver(latent_dim, eq_idx, eq_stencil, eq_w,
                       max_iters, tol):
    def _constrained_decode(z):
        u_hat = model.apply({'params': params}, z, method=model.decode)
        return mask * u_hat + u_g

    def _sparse_residual(z, F_eq):
        u_full    = _constrained_decode(z)
        u_stencil = u_full[eq_stencil].reshape(-1, 5)
        Ku_eq     = (4*u_stencil[:, 0]
                     - u_stencil[:, 1] - u_stencil[:, 2]
                     - u_stencil[:, 3] - u_stencil[:, 4]) / dx**2
        return Ku_eq - F_eq

    @jax.jit
    def solve(z_init, F_vec):
        F_eq = F_vec[eq_idx]
        def _body(carry):
            z, _, itr = carry
            R  = _sparse_residual(z, F_eq)
            J  = jax.jacfwd(lambda l: _sparse_residual(l, F_eq))(z)
            WJ   = eq_w[:, None] * J
            JtWJ = J.T @ WJ
            JtWr = J.T @ (eq_w * R)
            lam  = jnp.maximum(1e-5 * jnp.trace(JtWJ) / latent_dim, 1e-10)
            dz   = jnp.linalg.solve(JtWJ + lam * jnp.eye(latent_dim), -JtWr)
            f0 = jnp.dot(eq_w * R, R)
            def _f(a):
                Rt = _sparse_residual(z + a*dz, F_eq)
                return jnp.dot(eq_w * Rt, Rt)
            f1, f2, f3, f4 = _f(1.), _f(.5), _f(.25), _f(.125)
            step = jnp.where(f1 < f0, 1.0,
                   jnp.where(f2 < f0, 0.5,
                   jnp.where(f3 < f0, 0.25, 0.125)))
            return z + step*dz, jnp.linalg.norm(JtWr), itr + 1
        def _cond(carry):
            _, gnorm, itr = carry
            return jnp.logical_and(gnorm > tol, itr < max_iters)
        init = (z_init, jnp.array(jnp.inf, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32))
        z_final, _, n_iters = jax.lax.while_loop(_cond, _body, init)
        return z_final, _constrained_decode(z_final), n_iters
    return solve

eq_rom_solve = make_eq_rom_solver(
    k_dim, eq_indices_jnp, eq_stencil_flat, eq_weights_jnp,
    args.max_iters, args.gn_tol)

# ==========================================
# 8. Benchmark
# ==========================================
TEST_CASES = [(1.5, 2.3), (2.1, 1.4), (1.2, 2.8), (2.5, 1.7)]
print(f"\n5. Benchmark on {len(TEST_CASES)} unseen cases...")

# Warmup
F_warm = get_F_2d(*TEST_CASES[0])
_ = jax_linalg.cg(K_op_2d, F_warm, x0=u_guess, tol=1e-6)[0].block_until_ready()
_, _, _ = eq_rom_solve(jnp.zeros(k_dim), F_warm)
jax.effects_barrier()

fom_times, rom_times, rel_errs = [], [], []
for case_idx, (k1, k2) in enumerate(TEST_CASES):
    F_test = get_F_2d(k1, k2)
    u_true = full_order_fem_solver(F_test, u_guess)
    z_init = jnp.zeros(k_dim)

    t0 = time.time()
    u_fom, _ = jax_linalg.cg(K_op_2d, F_test, x0=u_guess, tol=1e-6)
    u_fom.block_until_ready()
    fom_t = time.time() - t0

    t0 = time.time()
    z_f, u_pred, n_it = eq_rom_solve(z_init, F_test)
    u_pred.block_until_ready()
    rom_t = time.time() - t0

    rel = float(jnp.linalg.norm(u_true - u_pred) / jnp.linalg.norm(u_true))
    fom_times.append(fom_t); rom_times.append(rom_t); rel_errs.append(rel)
    print(f"   Case {case_idx+1} k=({k1},{k2}): "
          f"FOM {fom_t:.4f}s  ROM {rom_t:.4f}s  speedup {fom_t/rom_t:.2f}x  "
          f"rel-L2 {rel:.4e}  iters {int(n_it)}")

avg_fom = float(np.mean(fom_times))
avg_rom = float(np.mean(rom_times))
avg_spup = avg_fom / avg_rom
avg_err = float(np.mean(rel_errs))

print(f"\nAvg FOM {avg_fom:.5f}s | Avg ROM {avg_rom:.5f}s | "
      f"speedup {avg_spup:.2f}x | rel-L2 {avg_err:.4e}")
print(f"EQ nodes: {num_eq_points}/{num_nodes} ({100*num_eq_points/num_nodes:.3f}%)")
print(f"Val rec err: {val_rec:.4e}")

# ==========================================
# 9. Results.tsv (append-style: one line)
# ==========================================
results_path = OUTDIR / 'results.tsv'
if not results_path.exists():
    with open(results_path, 'w') as f:
        f.write("label\tN\tk_dim\trank\teq_mode\tn_eq\tmax_iters\tgn_tol\t"
                "cg_tol\tspeedup\trel_l2\tfom_time\trom_time\tval_rec\ttrain_rec\n")
with open(results_path, 'a') as f:
    f.write(f"{args.label}\t{N}\t{k_dim}\t{args.rank}\t{args.eq_mode}\t"
            f"{num_eq_points}\t{args.max_iters}\t{args.gn_tol}\t{args.cg_tol}\t"
            f"{avg_spup:.4f}\t{avg_err:.4e}\t{avg_fom:.5f}\t{avg_rom:.5f}\t"
            f"{val_rec:.4e}\t{train_rec:.4e}\n")

# ==========================================
# 10. Plot last case
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
def plot_field(ax, field, title):
    im = ax.imshow(np.array(field).reshape(N,N).T, origin='lower', cmap='inferno',
                   extent=[0,L,0,L], aspect='auto')
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)

k1, k2 = TEST_CASES[-1]
F_last = get_F_2d(k1, k2)
u_true_last = full_order_fem_solver(F_last, u_guess)
_, u_pred_last, _ = eq_rom_solve(jnp.zeros(k_dim), F_last)
plot_field(axs[0], u_true_last, f"FOM k=({k1:.1f},{k2:.1f})")
plot_field(axs[1], u_pred_last, f"NM-ROM ({args.eq_mode})")
plot_field(axs[2], jnp.abs(u_true_last - u_pred_last),
           f"|err|  rel-L2={rel_errs[-1]:.2e}")
plt.suptitle(f"Poisson-2D N={N}  {args.label}  EQ={args.eq_mode}({num_eq_points})")
plt.tight_layout()
plt.savefig(OUTDIR / 'plots' / 'results.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {OUTDIR/'plots'/'results.png'}")
print("\n=== NM-ROM Complete ===")
