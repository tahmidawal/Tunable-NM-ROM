"""Heat-2D coordinate-decoder testbed: does the Poisson result transfer to a
time-dependent problem?

Mirrors the 2D Poisson testbed (exp/2026-08-12-coord-decoder) on the repo's
heat FOM:  du/dt = kappa * lap(u) on [0,1]^2, u=0 on the boundary, backward
Euler dt=0.005 x 50 steps, CG per implicit step. Family fixed to ONE Gaussian
blob so the true parameter vector has fixed length:

    z = (cx, cy, width, amplitude, log kappa),  normalized to ~[-1, 1]

Three arms per training resolution N, all judged in-resolution here (the
cross-resolution eval vs the N=512 reference runs separately from the
checkpoints):

  - POD floors: SVD of the space-time training snapshot matrix (f64 Gram),
    projection error of val snapshots, rank sweep. The linear yardstick.
  - grid-tied decoder: MLP(z, t) -> rank-R coefficients over learned rank-1
    spatial factors (CP structure, parameters anchored to grid nodes). The
    disease control arm.
  - FiLM coord-net: u(x, y, t; z) — Fourier features in x, y AND t, trunk
    FiLM-modulated by (z, t). No grid-tied parameters anywhere.

Nyquist rule is enforced in-code: spatial n_freq <= (N-1)//2 (the aliasing
landmine from the Poisson round), time n_freq <= NUM_STEPS//2.

Usage:  N=64 [STEPS=120000] [GT_STEPS=40000] python heat2d_film.py [outdir]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

N = int(os.environ.get("N", "16"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
N_TRAIN = int(os.environ.get("N_TRAIN", "512"))    # trajectories
N_VAL = int(os.environ.get("N_VAL", "64"))
STEPS = int(os.environ.get("STEPS", "120000"))
GT_STEPS = int(os.environ.get("GT_STEPS", "40000"))
GT_RANK = int(os.environ.get("GT_RANK", "24"))
BATCH = 32
P_POINTS = int(os.environ.get("P_POINTS", "8192"))
PEAK_LR = 2e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05
SEED = int(os.environ.get("SEED", "0"))
N_FREQ = min(int(os.environ.get("N_FREQ", "32")), (N - 1) // 2)  # Nyquist cap
HIDDEN = int(os.environ.get("HIDDEN", "256"))
N_LAYERS = 5
EVAL_EVERY = 2000

DT = 0.005
NUM_STEPS = 50
T_FREQ = min(int(os.environ.get("T_FREQ", "8")), NUM_STEPS // 2)
CG_TOL = 1e-10
CG_MAXITER = 20_000
POD_TIME_STRIDE = 4
POD_RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
EVAL_TIMES = [0, 10, 20, 30, 40, 50]   # snapshot indices for reporting

F32 = jnp.float32


# ----------------------------- FOM (float64) -----------------------------

def boundary_mask(n):
    m = np.ones((n, n))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    return jnp.asarray(m)


def make_rollout(n):
    """Batched backward-Euler rollout, matching heat/src fom (dt, steps, BCs)."""
    mask = boundary_mask(n)
    mask_flat = mask.reshape(-1)
    dx = 1.0 / (n - 1)

    def implicit_op(v_flat, kap):
        u = v_flat.reshape(n, n)
        lap = jnp.zeros_like(u)
        lap = lap.at[1:-1, 1:-1].set(
            (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
             - 4.0 * u[1:-1, 1:-1]) / dx**2)
        Au = u - DT * kap * lap
        Au = jnp.where(mask > 0, Au, u)
        return Au.reshape(-1)

    def solve_one(u_prev, kap):
        A = lambda vv: implicit_op(vv, kap)
        u_next, _ = jax.scipy.sparse.linalg.cg(
            A, u_prev, x0=u_prev, tol=CG_TOL, maxiter=CG_MAXITER)
        return u_next * mask_flat

    step_batch = jax.vmap(solve_one)

    @jax.jit
    def rollout(U0_b, kappa_b):
        def body(u, _):
            u2 = step_batch(u, kappa_b)
            return u2, u2
        _, snaps = jax.lax.scan(body, U0_b, None, length=NUM_STEPS)
        return jnp.concatenate([U0_b[None], snaps], axis=0)  # (T+1, B, n*n)

    return rollout, implicit_op


def sample_params(seed=SEED, m=None):
    """One rng draw shared by every script; val = the LAST N_VAL entries."""
    rng = np.random.default_rng(seed)
    m = m or (N_TRAIN + N_VAL)
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = rng.uniform(0.05, 0.20, m)
    a = rng.uniform(1.0, 10.0, m)
    logk = rng.uniform(np.log(0.01), np.log(0.5), m)
    kappa = np.exp(logk)
    z = np.stack([
        (cx - 0.5) / 0.35,
        (cy - 0.5) / 0.35,
        (w - 0.125) / 0.075,
        (a - 5.5) / 4.5,
        (logk - np.log(np.sqrt(0.005))) / (0.5 * np.log(50.0)),
    ], axis=1).astype(np.float32)
    return cx, cy, w, a, kappa, z


def build_trajectories(n, chunk=64):
    cx, cy, w, a, kappa, z = sample_params()
    m = len(cx)
    rollout, implicit_op = make_rollout(n)
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.asarray(boundary_mask(n))
    t0 = time.time()
    U = np.zeros((m, NUM_STEPS + 1, n * n))
    res_max = 0.0
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        U0 = np.stack([
            (a[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2)
                           / (2 * w[i] ** 2)) * mask).reshape(-1)
            for i in range(s, e)])
        snaps = np.asarray(rollout(jnp.asarray(U0), jnp.asarray(kappa[s:e])))
        U[s:e] = snaps.transpose(1, 0, 2)
        # spot-check the implicit equation on the first chunk sample, last step
        if s == 0:
            r = np.asarray(implicit_op(jnp.asarray(U[0, -1]), kappa[0])) - U[0, -2]
            res_max = np.linalg.norm(r) / max(np.linalg.norm(U[0, -2]), 1e-300)
    print(f"  FOM: {m} trajectories ({NUM_STEPS} implicit steps each) in "
          f"{time.time()-t0:.0f}s, spot rel residual {res_max:.2e}", flush=True)
    return U, z, cx, cy, w, a, kappa


# ----------------------------- POD floors -----------------------------

def pod_floors(U_tr, U_va):
    """f64 Gram-based POD of training space-time snapshots; val projection."""
    S = U_tr[:, ::POD_TIME_STRIDE].reshape(-1, U_tr.shape[-1])       # (m_snap, n^2)
    Y = U_va[:, ::POD_TIME_STRIDE].reshape(-1, U_va.shape[-1])
    S_d, Y_d = jnp.asarray(S), jnp.asarray(Y)
    G = S_d @ S_d.T                                                   # f64 Gram
    evals, evecs = jnp.linalg.eigh(G)
    order = jnp.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    r_max = min(max(POD_RANKS), S.shape[0])
    V = (S_d.T @ evecs[:, :r_max]) / jnp.sqrt(jnp.maximum(evals[:r_max], 1e-300))
    ortho_dev = float(jnp.max(jnp.abs(V.T @ V - jnp.eye(r_max))))
    print(f"  POD: {S.shape[0]} snapshots, top-{r_max} ortho dev {ortho_dev:.1e}",
          flush=True)
    Y_norm = jnp.linalg.norm(Y_d, axis=1)
    C = Y_d @ V                                                       # (m_val_snap, r_max)
    floors = {}
    for r in POD_RANKS:
        if r > r_max:
            continue
        recon = C[:, :r] @ V[:, :r].T
        rel = jnp.linalg.norm(recon - Y_d, axis=1) / Y_norm
        floors[r] = float(jnp.mean(rel))
    sv = np.sqrt(np.maximum(np.asarray(evals[:r_max]), 0.0))
    return floors, ortho_dev, sv[:32].tolist()


# ----------------------------- shared bits -----------------------------

def rel_l2_sq(pred, true):
    return jnp.sum((pred - true) ** 2) / (jnp.sum(true**2) + 1e-12)


def init_dense(key, d_in, d_out):
    W = jax.random.normal(key, (d_in, d_out), dtype=F32) * np.sqrt(1.0 / d_in)
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F32)}


def count_params(params):
    return sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))


# ----------------------------- grid-tied arm -----------------------------
# u(z, t) = sum_r c_r(z, t) * (fx_r outer fy_r): coefficients from an MLP,
# factors are free parameters PER GRID NODE — the CP-style control.

GT_HIDDEN = 128


def init_grid_tied(key, n):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "fx": jax.random.normal(k1, (GT_RANK, n), dtype=F32) / np.sqrt(n),
        "fy": jax.random.normal(k2, (GT_RANK, n), dtype=F32) / np.sqrt(n),
        "l1": init_dense(k3, 6, GT_HIDDEN),
        "l2": init_dense(k4, GT_HIDDEN, GT_HIDDEN),
        "l3": init_dense(k5, GT_HIDDEN, GT_RANK),
    }


def grid_tied_apply(params, z, tau):
    """z (B,5), tau (B,) -> full-grid prediction (B, n*n)."""
    h = jnp.concatenate([z, 2.0 * tau[:, None] - 1.0], axis=1)
    h = jax.nn.swish(h @ params["l1"]["W"] + params["l1"]["b"])
    h = jax.nn.swish(h @ params["l2"]["W"] + params["l2"]["b"])
    c = h @ params["l3"]["W"] + params["l3"]["b"]                    # (B, R)
    U = jnp.einsum("br,rn,rm->bnm", c, params["fx"], params["fy"])
    return U.reshape(U.shape[0], -1)


# ----------------------------- FiLM coord net -----------------------------
# Fourier features in x, y (Nyquist-capped) and t; trunk modulated by (z, t).

COORD_FEATS = 2 * (2 * N_FREQ + 1) + (2 * T_FREQ + 1)


def init_film_net(key):
    keys = jax.random.split(key, N_LAYERS + 4)
    trunk = [init_dense(keys[0], COORD_FEATS, HIDDEN)]
    for i in range(1, N_LAYERS):
        trunk.append(init_dense(keys[i], HIDDEN, HIDDEN))
    out = init_dense(keys[N_LAYERS], HIDDEN, 1)
    z_embed = init_dense(keys[N_LAYERS + 1], 6, 64)   # (z, tau)
    film = init_dense(keys[N_LAYERS + 2], 64, N_LAYERS * 2 * HIDDEN)
    film["W"] = film["W"] * 0.01                       # ~identity at init
    return {"trunk": trunk, "out": out, "z_embed": z_embed, "film": film}


def coord_features(xy, tau):
    j = jnp.arange(1, N_FREQ + 1, dtype=F32)

    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)

    jt = jnp.arange(1, T_FREQ + 1, dtype=F32)
    t = jnp.full((xy.shape[0],), tau, dtype=F32)
    tf = jnp.concatenate(
        [t[:, None], jnp.sin(jnp.pi * jt * t[:, None]),
         jnp.cos(jnp.pi * jt * t[:, None])], axis=1)
    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1]), tf], axis=1)


def film_apply(params, z, tau, xy):
    zt = jnp.concatenate([z, jnp.array([2.0 * tau - 1.0], dtype=F32)])
    g = jax.nn.swish(zt @ params["z_embed"]["W"] + params["z_embed"]["b"])
    film = (g @ params["film"]["W"] + params["film"]["b"]).reshape(
        N_LAYERS, 2, HIDDEN)
    h = coord_features(xy, tau)
    for i, lyr in enumerate(params["trunk"]):
        h = h @ lyr["W"] + lyr["b"]
        h = h * (1.0 + film[i, 0]) + film[i, 1]
        h = jax.nn.swish(h)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]


# ----------------------------- training -----------------------------

def sample_points(np_rng, traj_idx, time_idx, cx, cy, w, kappa, n, P):
    """Half uniform grid nodes, half near the (diffusion-widened) blob."""
    B = len(traj_idx)
    Pu, Pb = P // 2, P - P // 2
    idx = np.empty((B, P), dtype=np.int64)
    idx[:, :Pu] = np_rng.integers(0, n * n, size=(B, Pu))
    for bi, (i, k) in enumerate(zip(traj_idx, time_idx)):
        w_eff = np.sqrt(w[i] ** 2 + 2.0 * kappa[i] * k * DT)
        px = cx[i] + np_rng.normal(0.0, 2.0 * w_eff, Pb)
        py = cy[i] + np_rng.normal(0.0, 2.0 * w_eff, Pb)
        ix = np.clip(np.round(px * (n - 1)), 0, n - 1).astype(np.int64)
        iy = np.clip(np.round(py * (n - 1)), 0, n - 1).astype(np.int64)
        idx[bi, Pu:] = ix * n + iy
    return idx


def main():
    backend = jax.default_backend()
    print(f"jax_backend={backend}", flush=True)
    n_nodes = N * N
    print(f"N={N} ({n_nodes} nodes x {NUM_STEPS+1} times), film {STEPS} steps / "
          f"grid-tied {GT_STEPS}, n_train={N_TRAIN} traj, n_freq={N_FREQ} "
          f"(Nyquist-capped), t_freq={T_FREQ}", flush=True)

    U, z, cx, cy, w, a, kappa = build_trajectories(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]

    floors, ortho_dev, sv = pod_floors(U_tr, U_va)
    print("  POD floors: " + "  ".join(f"r{r}={e:.3e}" for r, e in floors.items()),
          flush=True)

    U_tr32 = jnp.asarray(U_tr, dtype=F32)              # (n_tr, T+1, n^2)
    z_tr32 = jnp.asarray(z_tr, dtype=F32)
    z_va32 = jnp.asarray(z_va, dtype=F32)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    coords32 = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1),
                           dtype=F32)
    taus = np.arange(NUM_STEPS + 1, dtype=np.float64) / NUM_STEPS
    # f32 val copy on device for cheap periodic eval at stride-10 times
    U_va32_sub = jnp.asarray(U_va[:, ::10], dtype=F32)  # (n_val, 6, n^2)
    taus_sub = jnp.asarray(taus[::10], dtype=F32)

    np_rng = np.random.default_rng(SEED)
    P = min(P_POINTS, n_nodes)
    results = {"N": N, "backend": backend, "seed": SEED, "n_train": N_TRAIN,
               "n_freq": N_FREQ, "t_freq": T_FREQ, "hidden": HIDDEN,
               "gt_rank": GT_RANK, "steps": STEPS, "gt_steps": GT_STEPS,
               "pod": {str(r): e for r, e in floors.items()},
               "pod_ortho_dev": ortho_dev, "singular_values": sv}

    # ---------------- grid-tied arm (full-grid loss) ----------------
    gt_params = init_grid_tied(jax.random.PRNGKey(SEED + 1), N)
    n_par_gt = count_params(gt_params)
    warmup = max(1, int(GT_STEPS * WARMUP_FRAC))
    gt_opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, GT_STEPS - warmup, end_value=1e-7),
        weight_decay=WEIGHT_DECAY)
    gt_state = gt_opt.init(gt_params)

    def gt_loss(p, U_b, z_b, tau_b):
        pred = grid_tied_apply(p, z_b, tau_b)
        return jnp.mean(jax.vmap(rel_l2_sq)(pred, U_b))

    @jax.jit
    def gt_step(p, s, U_b, z_b, tau_b):
        loss, grads = jax.value_and_grad(gt_loss)(p, U_b, z_b, tau_b)
        upd, s = gt_opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, loss

    t0 = time.time()
    for it in range(GT_STEPS):
        bi = np_rng.integers(0, N_TRAIN, BATCH)
        bk = np_rng.integers(0, NUM_STEPS + 1, BATCH)
        gt_params, gt_state, _ = gt_step(
            gt_params, gt_state, U_tr32[bi, bk],
            z_tr32[bi], jnp.asarray(taus[bk], dtype=F32))
    # f64 full eval over all val trajs/times
    gt_errs = []
    for i in range(N_VAL):
        pred = np.asarray(grid_tied_apply(
            gt_params, jnp.tile(z_va32[i], (NUM_STEPS + 1, 1)),
            jnp.asarray(taus, dtype=F32)), dtype=np.float64)
        rel = (np.linalg.norm(pred - U_va[i], axis=1)
               / np.linalg.norm(U_va[i], axis=1))
        gt_errs.append(rel.mean())
    results["grid_tied"] = float(np.mean(gt_errs))
    results["params_grid"] = n_par_gt
    print(f"  grid-tied ({n_par_gt} params, {GT_STEPS} steps, "
          f"{time.time()-t0:.0f}s): {results['grid_tied']:.3e}", flush=True)

    # ---------------- FiLM coord-net arm ----------------
    params = init_film_net(jax.random.PRNGKey(SEED))
    n_par = count_params(params)
    print(f"  film params: {n_par}", flush=True)
    warmup = max(1, int(STEPS * WARMUP_FRAC))
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-7),
        weight_decay=WEIGHT_DECAY)
    opt_state = opt.init(params)

    def batched_loss(p, U_b, z_b, tau_b, idx):
        def one(z_i, tau_i, u_i, idx_i):
            pred = film_apply(p, z_i, tau_i, coords32[idx_i])
            return rel_l2_sq(pred, u_i[idx_i])
        return jnp.mean(jax.vmap(one)(z_b, tau_b, U_b, idx))

    @jax.jit
    def step(p, s, U_b, z_b, tau_b, idx):
        loss, grads = jax.value_and_grad(batched_loss)(p, U_b, z_b, tau_b, idx)
        upd, s = opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, loss

    @jax.jit
    def val_loss(p):
        def one_traj(zu):
            z_i, u_slices = zu
            def one_t(carry, kt):
                tau_i, u_i = kt
                pred = film_apply(p, z_i, tau_i, coords32)
                return carry + rel_l2_sq(pred, u_i), 0.0
            tot, _ = jax.lax.scan(one_t, 0.0, (taus_sub, u_slices))
            return tot / len(taus_sub)
        return jnp.mean(jax.lax.map(one_traj, (z_va32, U_va32_sub)))

    best_val, best_params = float("inf"), params
    t0 = time.time()
    for it in range(STEPS):
        bi = np_rng.integers(0, N_TRAIN, BATCH)
        bk = np_rng.integers(0, NUM_STEPS + 1, BATCH)
        idx = jnp.asarray(sample_points(np_rng, bi, bk, cx, cy, w, kappa, N, P))
        params, opt_state, _ = step(
            params, opt_state, U_tr32[bi, bk], z_tr32[bi],
            jnp.asarray(taus[bk], dtype=F32), idx)
        if it % EVAL_EVERY == 0 or it == STEPS - 1:
            v = float(val_loss(params))
            if v < best_val:
                best_val, best_params = v, params
            if it % (EVAL_EVERY * 10) == 0:
                print(f"  step {it:7d}  val {v:.3e}  [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"  trained {STEPS} steps in {time.time()-t0:.0f}s "
          f"(best val loss {best_val:.3e})", flush=True)

    # f64 full eval over ALL val trajs and times + per-time breakdown
    per_time = {k: [] for k in EVAL_TIMES}
    film_errs = []
    for i in range(N_VAL):
        rels = []
        for k in range(NUM_STEPS + 1):
            pred = np.asarray(film_apply(best_params, z_va32[i],
                                         jnp.asarray(taus[k], dtype=F32),
                                         coords32), dtype=np.float64)
            rel = (np.linalg.norm(pred - U_va[i, k])
                   / np.linalg.norm(U_va[i, k]))
            rels.append(rel)
            if k in per_time:
                per_time[k].append(rel)
        film_errs.append(np.mean(rels))
    results["film_coord"] = float(np.mean(film_errs))
    results["film_per_time"] = {str(k): float(np.mean(v))
                                for k, v in per_time.items()}
    results["params_film"] = n_par
    print(f"RESULT N={N}  film-coord={results['film_coord']:.3e}  "
          f"grid-tied={results['grid_tied']:.3e}  "
          f"POD-6={floors.get(6, float('nan')):.3e}  "
          f"POD-24={floors.get(24, float('nan')):.3e}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"heat2d_results_N{N}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTDIR, f"heat2d_film_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, best_params), f)
    with open(os.path.join(OUTDIR, f"heat2d_gt_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, gt_params), f)
    print("wrote results + checkpoints", flush=True)


if __name__ == "__main__":
    main()
