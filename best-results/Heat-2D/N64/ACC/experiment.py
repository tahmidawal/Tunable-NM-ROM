"""
experiment.py — THE file the agent modifies.

Defines the decoder architecture, training loop, and ROM solver for the
Heat-2D NM-ROM experiment. Everything in this file is fair game except
the public interface: the agent MUST keep `Hyperparams`, `train`, and
`benchmark` callable with the signatures below so that `run.py` can
invoke them unchanged.

Public contract:
    train(hp: Hyperparams) -> dict with keys {'params', 'model_cfg', 'peak_vram_mb'}
    benchmark(trained: dict, hp: Hyperparams, val_trajs: list[fixed.ValTrajectory])
        -> dict with keys {'val_rel_l2', 'speedup', 'fom_seconds', 'rom_seconds',
                           'gn_iters_mean'}

Baseline config mirrors the upstream Heat-2D reference (k=48, rank=256,
500 train trajectories, 40k epochs, relative-L2 reconstruction loss).
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.scipy.sparse.linalg import cg as jax_cg
from scipy.optimize import nnls

import fixed


# =====================================================================
# Hyperparameters — the agent tunes these + the architecture below.
# =====================================================================
@dataclass
class Hyperparams:
    # model
    N: int = 64
    k_dim: int = 64                # Run0 fix-verify: best-known N=64 config (Run1)
    rank: int = 256
    hidden_dim: int = 256
    patch_size: int = 8
    embed_dim: int = 96
    num_heads: int = 4
    num_enc_layers: int = 4
    # training
    num_epochs: int = 80_000
    batch_size: int = 32
    peak_lr: float = 2e-3
    weight_decay: float = 5e-4
    seed: int = 0
    # rom
    n_eq_samples: int = 100        # NNLS design-matrix snapshots
    # Run1: gn_max_iters 20→8 — speedup push (AE-reuse Run0 ckpt)
    gn_max_iters: int = 8
    gn_tol: float = 1e-3
    cg_tol: float = 1e-6
    cg_max_iters: int = 200


# =====================================================================
# Model: ViT encoder + LinearCPDecoder (direct linear branch + MLP + CP tensor)
# =====================================================================
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
        assert N % p == 0
        n_patches_per_side = N // p
        u = u_flat.reshape(N, N)
        # patchify
        patches = u.reshape(n_patches_per_side, p, n_patches_per_side, p)
        patches = jnp.transpose(patches, (0, 2, 1, 3)).reshape(n_patches_per_side**2, p * p)
        x = nn.Dense(self.embed_dim)(patches)
        pos = self.param('pos', nn.initializers.normal(0.02),
                         (n_patches_per_side**2, self.embed_dim))
        x = x + pos
        for _ in range(self.num_enc_layers):
            y = nn.LayerNorm()(x)
            y = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.embed_dim)(y)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.embed_dim * 4)(y); y = nn.gelu(y)
            y = nn.Dense(self.embed_dim)(y)
            x = x + y
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=0)
        return nn.Dense(self.latent_dim)(x)


class LinearCPDecoder(nn.Module):
    """Direct linear branch + shallow MLP → rank-R channel → CP tensor field."""
    latent_dim: int
    rank: int = 256
    grid_size: int = 64
    hidden_dim: int = 256

    def setup(self):
        self.W1 = nn.Dense(self.hidden_dim)
        self.W2 = nn.Dense(self.hidden_dim)
        self.W_rank   = nn.Dense(self.rank)
        self.W_direct = nn.Dense(self.rank)
        init = nn.initializers.normal(0.01)
        Ng = self.grid_size
        self.W_x  = self.param('W_x', init, (self.rank, Ng))
        self.W_y  = self.param('W_y', init, (self.rank, Ng))
        self.bias = self.param('bias', nn.initializers.zeros, ())

    def __call__(self, z):
        h_nl  = nn.swish(self.W1(z))
        h_nl  = nn.swish(self.W2(h_nl))
        h_nl  = self.W_rank(h_nl)
        h_lin = self.W_direct(z)
        h     = h_lin + h_nl
        u_2d  = jnp.einsum('r,ri,rj->ij', h, self.W_x, self.W_y)
        return u_2d.flatten() + self.bias


class Autoencoder(nn.Module):
    """Encoder → (z, scale); decoder outputs normalised field, scale applied outside."""
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
        scale = jnp.max(jnp.abs(u_flat)) + fixed.AMP_EPS
        u_norm = u_flat / scale
        z = self.encoder(u_norm)
        return z, scale

    def decode_normalised(self, z):
        return self.decoder(z)

    def __call__(self, u_flat):
        z, scale = self.encode(u_flat)
        return self.decoder(z) * scale


# =====================================================================
# Training
# =====================================================================
def _load_training_snapshots(hp):
    """Load the flat (n_snap, N^2) snapshot matrix used for AE training."""
    if hp.N != 64:
        raise NotImplementedError(
            f'baseline experiment.py supports N=64 only. '
            f'To extend, regenerate the upstream pickle at the desired N '
            f'and adjust the loader here.')
    with open(fixed.UPSTREAM_DATA_64, 'rb') as f:
        payload = pickle.load(f)
    U_train = np.asarray(payload['U_train'], dtype=np.float32)
    U_val   = np.asarray(payload['U_val'],   dtype=np.float32)
    return U_train, U_val


def train(hp: Hyperparams) -> dict:
    U_train, U_val = _load_training_snapshots(hp)
    N = hp.N
    num_nodes = N * N

    model = Autoencoder(
        latent_dim=hp.k_dim, grid_size=N, rank=hp.rank,
        hidden_dim=hp.hidden_dim, patch_size=hp.patch_size,
        embed_dim=hp.embed_dim, num_heads=hp.num_heads,
        num_enc_layers=hp.num_enc_layers)
    key = jax.random.PRNGKey(hp.seed)
    params = model.init(key, jnp.ones(num_nodes))['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'n_params: {n_params}')

    cfg = dict(
        latent_dim=hp.k_dim, grid_size=N, rank=hp.rank,
        hidden_dim=hp.hidden_dim, patch_size=hp.patch_size,
        embed_dim=hp.embed_dim, num_heads=hp.num_heads,
        num_enc_layers=hp.num_enc_layers)

    # ── Checkpoint reuse: keyed by AE config + epochs + seed ───────────────
    # If a ckpt already exists with matching cfg, load and skip training.
    ckpt_key = (
        f"k{hp.k_dim}_r{hp.rank}_h{hp.hidden_dim}_p{hp.patch_size}_"
        f"e{hp.embed_dim}_nh{hp.num_heads}_nl{hp.num_enc_layers}_"
        f"ep{hp.num_epochs}_bs{hp.batch_size}_lr{hp.peak_lr}_"
        f"wd{hp.weight_decay}_s{hp.seed}"
    )
    ckpt_path = Path(f"checkpoint_N{N}_{ckpt_key}.pkl")
    if ckpt_path.exists():
        print(f'[ckpt-skip] Loading {ckpt_path} — skipping training')
        with open(ckpt_path, 'rb') as f:
            saved = pickle.load(f)
        return dict(params=saved['params'], model_cfg=saved['model_cfg'],
                    peak_vram_mb=0.0, n_params=n_params)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=hp.peak_lr,
        warmup_steps=min(2000, hp.num_epochs // 10),
        decay_steps=hp.num_epochs, end_value=1e-6)
    tx = optax.adamw(learning_rate=schedule, weight_decay=hp.weight_decay)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, u_batch):
        def single(u):
            u_pred = model.apply({'params': params}, u)
            num = jnp.linalg.norm(u - u_pred)
            den = jnp.linalg.norm(u) + 1e-6
            return num / den
        return jnp.mean(jax.vmap(single)(u_batch))

    @jax.jit
    def train_step(params, opt_state, u_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, u_batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(hp.seed)
    for epoch in range(hp.num_epochs):
        idx = rng.integers(0, U_train.shape[0], size=hp.batch_size)
        batch = jnp.asarray(U_train[idx])
        params, opt_state, loss = train_step(params, opt_state, batch)
        if epoch % max(hp.num_epochs // 20, 1) == 0:
            val_loss = float(loss_fn(params, jnp.asarray(U_val[:256])))
            print(f'  epoch {epoch:6d} train {float(loss):.4e} val {val_loss:.4e}')

    try:
        peak_vram = float(jax.devices()[0].memory_stats()['peak_bytes_in_use']) / (1024**2)
    except Exception:
        peak_vram = 0.0

    # Save checkpoint for future ckpt-skip
    try:
        with open(ckpt_path, 'wb') as f:
            pickle.dump(dict(params=params, model_cfg=cfg), f)
        print(f'[ckpt-save] Saved {ckpt_path}')
    except Exception as exc:
        print(f'[ckpt-save] Skipped (write error): {exc}')

    return dict(params=params, model_cfg=cfg, peak_vram_mb=peak_vram,
                n_params=n_params)


# =====================================================================
# ROM solver + benchmark
# =====================================================================
def _build_eq_weights(params, model, K_op, mask, u_g, U_train, hp):
    """Assemble NNLS weights for EQ hyper-reduction."""
    N = hp.N
    num_nodes = N * N
    rng = np.random.default_rng(hp.seed + 1)
    idx = rng.choice(U_train.shape[0], size=min(hp.n_eq_samples, U_train.shape[0]),
                     replace=False)

    def constrained_decode(z, scale):
        u_hat = model.apply({'params': params}, z,
                            method=lambda m, z_: m.decode_normalised(z_))
        return mask * (u_hat * scale) + u_g

    G_rows = []
    b_rows = []
    for i in idx:
        u = jnp.asarray(U_train[i])
        scale = jnp.max(jnp.abs(u)) + fixed.AMP_EPS
        z = model.apply({'params': params}, u,
                        method=lambda m, u_: m.encode(u_))[0]
        u_pred = constrained_decode(z, scale)
        R = K_op(u_pred)
        J = jax.jacfwd(lambda zz: constrained_decode(zz, scale))(z)  # (N^2, k)
        Jnp = np.asarray(J)
        Rnp = np.asarray(R)
        G_rows.append(Jnp.T)        # (k, N^2)
        b_rows.append(Jnp.T @ Rnp)  # (k,)
    G = np.concatenate(G_rows, axis=0)  # (n_snap*k, N^2)
    b = np.concatenate(b_rows)          # (n_snap*k,)
    w, _ = nnls(G, b)                   # w: (N^2,) per-node EQ weights
    return w


def benchmark(trained, hp, val_trajs):
    """Run the ROM on each validation trajectory. Return metric dict."""
    N = hp.N
    num_nodes = N * N
    K_op, implicit_op, mask, u_g = fixed.make_fom_ops(N)

    cfg = trained['model_cfg']
    model = Autoencoder(**cfg)
    params = trained['params']

    # Use dense residual (no EQ sparsification) for baseline correctness.
    # EQ with too few nonzero weights produces a degenerate Jacobian that
    # prevents GN convergence. Dense is feasible because _gn_jacobian is
    # jit-compiled once per (shape, kappa) and reused across all steps.
    print('  using dense residual (no EQ)')

    def constrained_decode_normalised(z):
        u_hat = model.apply({'params': params}, z,
                            method=lambda m, z_: m.decode_normalised(z_))
        return mask * u_hat + u_g

    # ── KAPPA-RUNTIME + FORI-LOOP FIX (ported from Heat-3D N128 Run9) ──────
    # Original rom_step had Python loop with float() host-device sync each GN
    # iter, killing perf. New: pure-JAX while_loop for GN, fori_loop for the
    # 50-step rollout, kappa as traced runtime arg → single JIT compile reused
    # across all 10 benchmark kappas.
    GN_MAX_ITERS = int(hp.gn_max_iters)
    GN_TOL = float(hp.gn_tol)
    NUM_STEPS = fixed.N_STEPS

    def _residual_rt(zs, u_prev, kappa_f):
        z_, s_ = zs[:-1], zs[-1]
        u_n = constrained_decode_normalised(z_)
        return s_ * (u_n + fixed.DT * kappa_f * K_op(u_n)) - u_prev

    def _gn_step_rt(z_init, scale_init, u_prev, kappa_f):
        """One backward-Euler step via LM, all inside while_loop."""
        zs0 = jnp.concatenate([z_init, jnp.asarray([scale_init])])
        mu0 = jnp.float32(1e-4)
        ZERO = jnp.array(0, jnp.int32)

        def _body(carry):
            zs, mu, _, itr = carry
            J = jax.jacfwd(lambda zs_: _residual_rt(zs_, u_prev, kappa_f))(zs)
            r = _residual_rt(zs, u_prev, kappa_f)
            Jtr = J.T @ r
            JtJ = J.T @ J
            diag_scale = jnp.mean(jnp.diag(JtJ)) + 1e-8
            d = jnp.linalg.solve(JtJ + mu * diag_scale * jnp.eye(J.shape[1]), -Jtr)
            zs_new = zs + d
            r_new = _residual_rt(zs_new, u_prev, kappa_f)
            r_norm = jnp.linalg.norm(r)
            r_new_norm = jnp.linalg.norm(r_new)
            accept = r_new_norm < r_norm
            zs_next = jnp.where(accept, zs_new, zs)
            scale_floored = jnp.maximum(zs_next[-1], jnp.float32(1e-8))
            zs_next = zs_next.at[-1].set(scale_floored)
            mu_next = jnp.where(accept,
                                jnp.maximum(mu * 0.3, jnp.float32(1e-7)),
                                jnp.minimum(mu * 10.0, jnp.float32(1e2)))
            jtr_norm = jnp.linalg.norm(Jtr)
            return zs_next, mu_next, jtr_norm, itr + 1

        def _cond(carry):
            _, _, jtr_norm, itr = carry
            return (jtr_norm > GN_TOL) & (itr < GN_MAX_ITERS)

        zs_f, _, _, itr_f = jax.lax.while_loop(
            _cond, _body,
            (zs0, mu0, jnp.float32(jnp.inf), ZERO))
        z_f = zs_f[:-1]
        scale_f = zs_f[-1]
        return z_f, scale_f, itr_f

    @jax.jit
    def _rollout_rt(z0, scale0, u0_flat, kappa_f):
        iters_arr = jnp.zeros(NUM_STEPS, dtype=jnp.int32)
        def _step(i, carry):
            z_, sc_, u_prev, iters_ = carry
            z_n, sc_n, ni = _gn_step_rt(z_, sc_, u_prev, kappa_f)
            u_n = sc_n * constrained_decode_normalised(z_n)
            iters_ = iters_.at[i].set(ni)
            return (z_n, sc_n, u_n, iters_)
        return jax.lax.fori_loop(0, NUM_STEPS, _step, (z0, scale0, u0_flat, iters_arr))

    # Per-trajectory rollout — single JIT compile reused across kappas
    rom_finals = []
    fom_times = []
    rom_times = []
    gn_iters = []
    for tr in val_trajs:
        u0 = jnp.asarray(tr.u0)
        kappa = tr.kappa
        # FOM timing (fresh run; fair)
        t_fom, fom_final = fixed.time_fom(u0, kappa, N, fixed.N_STEPS, warmup=True)
        fom_times.append(t_fom)
        # ROM rollout — fully JIT'd, kappa as runtime arg
        scale0 = jnp.max(jnp.abs(u0)) + fixed.AMP_EPS
        z0 = model.apply({'params': params}, u0,
                         method=lambda m, u_: m.encode(u_))[0]
        kappa_f = jnp.float32(kappa)
        t0 = time.perf_counter()
        z_f, sc_f, u_cur, iters_arr = _rollout_rt(z0, scale0, u0, kappa_f)
        u_cur.block_until_ready()
        t_rom = time.perf_counter() - t0
        rom_times.append(t_rom)
        rom_finals.append(np.asarray(u_cur))
        gn_iters.append(float(jnp.mean(iters_arr)))

    val_rel_l2 = fixed.mean_rel_l2(rom_finals, [t.final_fom for t in val_trajs])
    speedup = float(np.mean(fom_times)) / max(float(np.mean(rom_times)), 1e-9)
    return dict(
        val_rel_l2=val_rel_l2, speedup=speedup,
        fom_seconds=float(np.mean(fom_times)),
        rom_seconds=float(np.mean(rom_times)),
        gn_iters_mean=float(np.mean(gn_iters)))
