"""Burgers-2D coordinate-decoder testbed: does the heat-2D result survive a
NONLINEAR advection-dominated PDE?

Ports the heat-2D testbed (exp/2026-08-13-heat2d-coord-decoder) to scalar
viscous Burgers on [0,1]^2 with homogeneous Dirichlet walls:

    u_t + u * (u_x + u_y) = nu * lap(u),   u = 0 on the boundary

FOM: fully implicit backward Euler, dt=0.005 x 50 steps (T=0.25). Each step
solves the nonlinear system with Newton iterations; the (nonsymmetric)
Jacobian solves are matrix-free BiCGStab with J*v computed by jax.jvp of the
residual. Advection is FIRST-ORDER UPWIND (monotone, robust at cell Peclet
>> 1; spatial order ~1 in advection-dominated regions), diffusion is
second-order centered. Reference generation and data are f64.

Family fixed to ONE Gaussian blob so the true parameter vector has fixed
length:

    z = (cx, cy, width, amplitude, log nu), normalized to ~[-1, 1]
    cx,cy ~ U(0.15,0.85), w ~ U(0.05,0.20), a ~ U(0.5,2.0),
    nu ~ logU(0.01, 0.1)

Range reasoning (see README): viscous front width ~4*nu/a >= 0.02 = ~10 cells
on the 512 reference grid (resolved); cell Peclet a*h/nu on the N=16 grid
reaches ~13 (advection-dominated coarse grids); Re = a*L/nu up to 200.

Three arms per training resolution N, judged in-resolution here (the
cross-resolution eval vs the N=512 reference runs separately):

  - POD floors: SVD of the space-time training snapshot matrix (f64 Gram),
    fitted AND evaluated on ALL time slices (POD_TIME_STRIDE=1 -- the
    heat-round adversarial-review fix; POD and neural columns use identical
    slice sets).
  - grid-tied decoder: MLP(z, t) -> rank-R coefficients over learned rank-1
    spatial factors (CP structure). The disease control arm.
  - FiLM coord-net: u(x, y, t; z) -- Fourier features in x, y AND t, trunk
    FiLM-modulated by (z, t). No grid-tied parameters anywhere.

Nyquist rule enforced in-code: spatial n_freq <= (N-1)//2, time n_freq <=
NUM_STEPS//2.

Usage:  N=64 [STEPS=120000] [GT_STEPS=40000] python burgers2d_film.py [outdir]
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
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2000"))

DT = 0.005
NUM_STEPS = 50                                     # T = 0.25
T_FREQ = min(int(os.environ.get("T_FREQ", "8")), NUM_STEPS // 2)
NEWTON_ITERS = int(os.environ.get("NEWTON_ITERS", "8"))
LIN_TOL = float(os.environ.get("LIN_TOL", "1e-10"))
LIN_MAXITER = int(os.environ.get("LIN_MAXITER", "2000"))
POD_TIME_STRIDE = int(os.environ.get("POD_TIME_STRIDE", "1"))  # 1 = all slices
POD_RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
EVAL_TIMES = [0, 10, 20, 30, 40, 50]   # snapshot indices for reporting

F32 = jnp.float32


# ----------------------------- FOM (float64) -----------------------------

def boundary_mask(n):
    m = np.ones((n, n))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    return jnp.asarray(m)


def make_rollout(n):
    """Batched implicit (backward-Euler + Newton/BiCGStab) Burgers rollout.

    Returns (rollout, residual): rollout(U0_b, nu_b) -> (snaps, rel_res) with
    snaps (T+1, B, n*n) and rel_res (T, B) = ||R(u_new)|| / ||u_prev|| per
    step (Newton convergence audit); residual(u_flat, u_prev_flat, nu) is the
    backward-Euler residual with boundary rows R = u (enforces u=0).
    """
    dx = 1.0 / (n - 1)

    def residual(u_flat, u_prev_flat, nu):
        u = u_flat.reshape(n, n)
        up = u_prev_flat.reshape(n, n)
        uc = u[1:-1, 1:-1]
        lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
               - 4.0 * uc) / dx**2
        dxm = (uc - u[:-2, 1:-1]) / dx        # backward difference in x
        dxp = (u[2:, 1:-1] - uc) / dx         # forward difference in x
        dym = (uc - u[1:-1, :-2]) / dx
        dyp = (u[1:-1, 2:] - uc) / dx
        ux = jnp.where(uc > 0.0, dxm, dxp)    # upwind by sign(u)
        uy = jnp.where(uc > 0.0, dym, dyp)
        r_int = uc - up[1:-1, 1:-1] + DT * (uc * (ux + uy) - nu * lap)
        R = u                                  # boundary rows: R = u
        R = R.at[1:-1, 1:-1].set(r_int)
        return R.reshape(-1)

    def newton_step(u_prev, nu):
        def body(u, _):
            r = residual(u, u_prev, nu)
            Jv = lambda v: jax.jvp(
                lambda uu: residual(uu, u_prev, nu), (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER)
            return u + du, 0.0
        u_new, _ = jax.lax.scan(body, u_prev, None, length=NEWTON_ITERS)
        rel = (jnp.linalg.norm(residual(u_new, u_prev, nu))
               / jnp.maximum(jnp.linalg.norm(u_prev), 1e-300))
        return u_new, rel

    step_batch = jax.vmap(newton_step)

    @jax.jit
    def rollout(U0_b, nu_b):
        def body(u, _):
            u2, rr = step_batch(u, nu_b)
            return u2, (u2, rr)
        _, (snaps, res) = jax.lax.scan(body, U0_b, None, length=NUM_STEPS)
        snaps = jnp.concatenate([U0_b[None], snaps], axis=0)  # (T+1, B, n*n)
        return snaps, res

    return rollout, residual


def sample_params(seed=SEED, m=None):
    """One rng draw shared by every script; val = the LAST N_VAL entries."""
    rng = np.random.default_rng(seed)
    m = m or (N_TRAIN + N_VAL)
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = rng.uniform(0.05, 0.20, m)
    a = rng.uniform(0.5, 2.0, m)
    lognu = rng.uniform(np.log(0.01), np.log(0.1), m)
    nu = np.exp(lognu)
    z = np.stack([
        (cx - 0.5) / 0.35,
        (cy - 0.5) / 0.35,
        (w - 0.125) / 0.075,
        (a - 1.25) / 0.75,
        (lognu - np.log(np.sqrt(0.001))) / (0.5 * np.log(10.0)),
    ], axis=1).astype(np.float32)
    return cx, cy, w, a, nu, z


def blob_ic(n, cx, cy, w, a):
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.asarray(boundary_mask(n))
    return (a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2)
                       / (2 * w ** 2)) * mask).reshape(-1)


def build_trajectories(n, chunk=64):
    cx, cy, w, a, nu, z = sample_params()
    m = len(cx)
    rollout, _ = make_rollout(n)
    t0 = time.time()
    U = np.zeros((m, NUM_STEPS + 1, n * n))
    res_max = 0.0
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        U0 = np.stack([blob_ic(n, cx[i], cy[i], w[i], a[i])
                       for i in range(s, e)])
        snaps, res = rollout(jnp.asarray(U0), jnp.asarray(nu[s:e]))
        U[s:e] = np.asarray(snaps).transpose(1, 0, 2)
        res_max = max(res_max, float(jnp.max(res)))
    print(f"  FOM: {m} trajectories ({NUM_STEPS} Newton-implicit steps each) "
          f"in {time.time()-t0:.0f}s, max Newton rel residual {res_max:.2e}",
          flush=True)
    if not np.isfinite(res_max) or res_max > 1e-8:
        print(f"  WARNING: Newton residual {res_max:.2e} above 1e-8 -- "
              f"raise NEWTON_ITERS/LIN_MAXITER", flush=True)
    return U, z, cx, cy, w, a, nu


# ----------------------------- POD floors -----------------------------

def pod_floors(U_tr, U_va):
    """f64 Gram-based POD of training space-time snapshots; val projection.

    POD_TIME_STRIDE defaults to 1: fitted AND evaluated on ALL time slices,
    identical to the slice set the neural arms average over (the heat-round
    adversarial-review fix).
    """
    S = U_tr[:, ::POD_TIME_STRIDE].reshape(-1, U_tr.shape[-1])
    Y = U_va[:, ::POD_TIME_STRIDE].reshape(-1, U_va.shape[-1])
    S_d, Y_d = jnp.asarray(S), jnp.asarray(Y)
    G = S_d @ S_d.T                                                   # f64 Gram
    evals, evecs = jnp.linalg.eigh(G)
    order = jnp.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    r_max = min(max(POD_RANKS), S.shape[0])
    V = (S_d.T @ evecs[:, :r_max]) / jnp.sqrt(jnp.maximum(evals[:r_max], 1e-300))
    ortho_dev = float(jnp.max(jnp.abs(V.T @ V - jnp.eye(r_max))))
    print(f"  POD: {S.shape[0]} snapshots (stride {POD_TIME_STRIDE}), "
          f"top-{r_max} ortho dev {ortho_dev:.1e}", flush=True)
    Y_norm = jnp.linalg.norm(Y_d, axis=1)
    C = Y_d @ V
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
# factors are free parameters PER GRID NODE -- the CP-style control.

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

def sample_points(np_rng, traj_idx, time_idx, cx, cy, w, a, nu, n, P):
    """Half uniform grid nodes, half near the advected + diffused blob.

    The blob translates along +(1,1) at roughly speed a/2 per axis (viscous
    front of a decaying positive pulse) and widens by diffusion; the biased
    half covers original tail through leading front.
    """
    B = len(traj_idx)
    Pu, Pb = P // 2, P - P // 2
    idx = np.empty((B, P), dtype=np.int64)
    idx[:, :Pu] = np_rng.integers(0, n * n, size=(B, Pu))
    for bi, (i, k) in enumerate(zip(traj_idx, time_idx)):
        t = k * DT
        shift = 0.5 * a[i] * t
        w_eff = np.sqrt(w[i] ** 2 + 2.0 * nu[i] * t)
        sig = 2.0 * w_eff + 0.5 * shift
        px = cx[i] + 0.5 * shift + np_rng.normal(0.0, sig, Pb)
        py = cy[i] + 0.5 * shift + np_rng.normal(0.0, sig, Pb)
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
          f"(Nyquist-capped), t_freq={T_FREQ}, newton_iters={NEWTON_ITERS}",
          flush=True)

    U, z, cx, cy, w, a, nu = build_trajectories(N)
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
               "newton_iters": NEWTON_ITERS, "pod_time_stride": POD_TIME_STRIDE,
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
        idx = jnp.asarray(sample_points(np_rng, bi, bk, cx, cy, w, a, nu, N, P))
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
    with open(os.path.join(OUTDIR, f"burgers2d_results_N{N}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTDIR, f"burgers2d_film_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, best_params), f)
    with open(os.path.join(OUTDIR, f"burgers2d_gt_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, gt_params), f)
    print("wrote results + checkpoints", flush=True)


if __name__ == "__main__":
    main()
