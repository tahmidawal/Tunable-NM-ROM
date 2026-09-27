"""Wave-2D coordinate-decoder testbed: does the heat-2D result survive the jump
to a HYPERBOLIC, dissipation-free PDE?

Mirrors the heat-2D testbed (exp/2026-08-13-heat2d-coord-decoder) on

    u_tt = c^2 * lap(u)  on [0,1]^2,  u = 0 on the boundary,

written as the first-order system (u, v = u_t) and stepped with Crank-Nicolson
(trapezoidal): unconditionally stable, and it conserves the discrete energy
E = 0.5*||v||^2 + 0.5*c^2*u^T(-L)u exactly (up to CG solve tolerance), so the
solver dt is FIXED across every mesh resolution.  Each implicit step solves the
SPD system (I - (c*dt/2)^2 L) u^{n+1} = rhs with CG.  The solver takes SUBSTEPS
internal steps per stored snapshot so CN dispersion error can be pushed well
below the N=512 spatial error without inflating the stored snapshot count.

Family fixed to ONE Gaussian displacement blob, zero initial velocity:

    z = (cx, cy, width, amplitude, log c),  normalized to ~[-1, 1]
    cx,cy ~ U(0.15,0.85), w ~ U(0.05,0.20), a ~ U(1,10), c ~ logU(0.5, 2)

Three arms per training resolution N (in-resolution here; the cross-resolution
eval vs the N=512 reference runs separately from the checkpoints):

  - POD floors: SVD of the space-time training snapshot matrix (f64 Gram),
    fitted AND evaluated on ALL time slices (the heat adversarial-review fix).
  - grid-tied decoder: MLP(z, t) -> rank-R coefficients over learned rank-1
    spatial factors (CP structure).  The disease control arm.
  - FiLM coord-net: u(x, y, t; z) - Fourier features in x, y AND t, trunk
    FiLM-modulated by (z, t).  No grid-tied parameters anywhere.

METRICS (wave-specific): u(t) oscillates and a snapshot norm ||u(t)|| can pass
near zero (kinetic/potential energy exchange), which makes the heat-style
per-snapshot relative L2 spiky.  The PRIMARY metric here normalizes by the
trajectory-RMS norm  sqrt(mean_t ||u(t)||^2)  (bounded away from 0 by energy
conservation); the heat-style per-snapshot mean is recorded alongside as
*_snap for cross-PDE comparability.

Nyquist rule enforced in-code: spatial n_freq <= (N-1)//2.  The TIME bandwidth
defaults to the snapshot Nyquist NUM_STEPS//2 = 25 (NOT heat's 8): wave
solutions oscillate at temporal frequency c*k/(2*pi) and need the full stored
bandwidth (see README).

Usage:  N=64 [STEPS=120000] [GT_STEPS=40000] python wave2d_film.py [outdir]
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

# --- time discretization: 51 stored snapshots over T_FINAL, SUBSTEPS CN steps
# per snapshot interval.  DT_SUB = 2.5e-4 puts c_max*dt (= 5e-4) well below the
# N=512 grid spacing (1/511 = 1.96e-3): CN dispersion error sits at ~1/4 of the
# N=512 spatial error (checked explicitly in wave2d_selfconv.py; at
# SUBSTEPS=40 the two were comparable, which motivated 80).
T_FINAL = 1.0
NUM_STEPS = 50                                     # stored snapshot intervals
SUBSTEPS = int(os.environ.get("SUBSTEPS", "80"))
DT_SNAP = T_FINAL / NUM_STEPS
DT_SUB = DT_SNAP / SUBSTEPS
T_FREQ = min(int(os.environ.get("T_FREQ", str(NUM_STEPS // 2))), NUM_STEPS // 2)
CG_TOL = 1e-10
CG_MAXITER = 20_000
POD_TIME_STRIDE = int(os.environ.get("POD_TIME_STRIDE", "1"))  # fit stride;
# eval is ALWAYS on all slices (adversarial-review fix from the heat round)
POD_RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
EVAL_TIMES = [0, 10, 20, 30, 40, 50]   # snapshot indices for reporting

F32 = jnp.float32


# ----------------------------- FOM (float64) -----------------------------

def boundary_mask(n):
    m = np.ones((n, n))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    return jnp.asarray(m)


def make_rollout(n):
    """Batched Crank-Nicolson rollout of the (u, v) system, per-sample speed c.

    Returns rollout(U0_b, c_b) -> (snaps, energies):
      snaps    (NUM_STEPS+1, B, n*n)  stored u snapshots (U0 first)
      energies (NUM_STEPS+1, B)       discrete energy at each snapshot
    """
    mask = boundary_mask(n)
    mask_flat = mask.reshape(-1)
    dx = 1.0 / (n - 1)

    def lap_flat(v_flat):
        u = v_flat.reshape(n, n)
        lap = jnp.zeros_like(u)
        lap = lap.at[1:-1, 1:-1].set(
            (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
             - 4.0 * u[1:-1, 1:-1]) / dx**2)
        return lap.reshape(-1)

    def implicit_op(v_flat, c):
        alpha = (0.5 * DT_SUB * c) ** 2
        Au = v_flat - alpha * lap_flat(v_flat)
        Au = jnp.where(mask_flat > 0, Au, v_flat)
        return Au

    def substep_one(u, v, Lu, c):
        alpha = (0.5 * DT_SUB * c) ** 2
        rhs = u + DT_SUB * v + alpha * Lu          # boundary rows are 0 already
        A = lambda vv: implicit_op(vv, c)
        u1, _ = jax.scipy.sparse.linalg.cg(
            A, rhs, x0=u + DT_SUB * v, tol=CG_TOL, maxiter=CG_MAXITER)
        u1 = u1 * mask_flat
        Lu1 = lap_flat(u1)
        v1 = (v + 0.5 * DT_SUB * c**2 * (Lu + Lu1)) * mask_flat
        return u1, v1, Lu1

    substep_b = jax.vmap(substep_one)

    def energy_one(u_flat, v_flat, c):
        """E = 0.5||v||^2 + 0.5 c^2 u^T(-L)u — the CN-conserved invariant.
        u^T(-L)u = sum of squared forward differences / dx^2 (Dirichlet)."""
        u = u_flat.reshape(n, n)
        gx = (u[1:, :] - u[:-1, :]) / dx
        gy = (u[:, 1:] - u[:, :-1]) / dx
        return dx * dx * (0.5 * jnp.sum(v_flat**2)
                          + 0.5 * c**2 * (jnp.sum(gx**2) + jnp.sum(gy**2)))

    energy_b = jax.vmap(energy_one)

    @jax.jit
    def rollout(U0_b, c_b):
        V0 = jnp.zeros_like(U0_b)
        Lu0 = jax.vmap(lap_flat)(U0_b)

        def snap_body(carry, _):
            def sub(cc, __):
                u, v, Lu = cc
                return substep_b(u, v, Lu, c_b), None
            carry, _ = jax.lax.scan(sub, carry, None, length=SUBSTEPS)
            u, v, _ = carry
            return carry, (u, energy_b(u, v, c_b))

        _, (snaps, ens) = jax.lax.scan(
            snap_body, (U0_b, V0, Lu0), None, length=NUM_STEPS)
        snaps = jnp.concatenate([U0_b[None], snaps], axis=0)
        e0 = energy_b(U0_b, V0, c_b)
        ens = jnp.concatenate([e0[None], ens], axis=0)
        return snaps, ens

    return rollout, implicit_op


def sample_params(seed=SEED, m=None):
    """One rng draw shared by every script; val = the LAST N_VAL entries."""
    rng = np.random.default_rng(seed)
    m = m or (N_TRAIN + N_VAL)
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = rng.uniform(0.05, 0.20, m)
    a = rng.uniform(1.0, 10.0, m)
    logc = rng.uniform(np.log(0.5), np.log(2.0), m)
    c = np.exp(logc)
    z = np.stack([
        (cx - 0.5) / 0.35,
        (cy - 0.5) / 0.35,
        (w - 0.125) / 0.075,
        (a - 5.5) / 4.5,
        logc / np.log(2.0),
    ], axis=1).astype(np.float32)
    return cx, cy, w, a, c, z


def build_trajectories(n, chunk=64):
    cx, cy, w, a, c, z = sample_params()
    m = len(cx)
    rollout, _ = make_rollout(n)
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.asarray(boundary_mask(n))
    t0 = time.time()
    U = np.zeros((m, NUM_STEPS + 1, n * n))
    drift_max = 0.0
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        U0 = np.stack([
            (a[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2)
                           / (2 * w[i] ** 2)) * mask).reshape(-1)
            for i in range(s, e)])
        snaps, ens = rollout(jnp.asarray(U0), jnp.asarray(c[s:e]))
        U[s:e] = np.asarray(snaps).transpose(1, 0, 2)
        ens = np.asarray(ens)                       # (T+1, B)
        drift = np.max(np.abs(ens - ens[0]) / np.maximum(ens[0], 1e-300))
        dm = float(drift)
        # NaN-propagating accumulate: builtin max(x, nan) keeps x and would
        # mask a diverged chunk
        if not np.isfinite(dm) or dm > drift_max:
            drift_max = dm
    print(f"  FOM: {m} trajectories ({NUM_STEPS}x{SUBSTEPS} CN substeps, "
          f"dt={DT_SUB:g}) in {time.time()-t0:.0f}s, "
          f"max rel energy drift {drift_max:.2e}", flush=True)
    return U, z, cx, cy, w, a, c


def traj_norms(U):
    """Per-trajectory RMS snapshot norm sqrt(mean_t ||u(t)||^2) and per-node
    mean square (loss normalizer)."""
    sn = np.linalg.norm(U, axis=2)                       # (m, T+1)
    rms = np.sqrt(np.mean(sn**2, axis=1))                # (m,)
    msq = np.mean(U**2, axis=(1, 2))                     # (m,)
    return rms, msq


# ----------------------------- POD floors -----------------------------

def pod_floors(U_tr, U_va, rms_va):
    """f64 Gram POD of training space-time snapshots; val projection.
    Fit on ::POD_TIME_STRIDE slices (default 1 = all); ALWAYS evaluated on all
    slices.  Returns (floors_traj, floors_snap, ortho_dev, sv)."""
    n_val, n_t = U_va.shape[:2]
    S = U_tr[:, ::POD_TIME_STRIDE].reshape(-1, U_tr.shape[-1])
    Y = U_va.reshape(-1, U_va.shape[-1])                  # ALL slices
    # host f64: with all-slice fitting the 26112^2 Gram + cuSOLVER eigh
    # workspace OOM an 80 GB A100 at N>=128
    S_d = np.asarray(S, dtype=np.float64)
    Y_d = np.asarray(Y, dtype=np.float64)
    G = S_d @ S_d.T                                       # f64 Gram
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # cap by numerical rank so near-null Gram modes (possible at smoke scale,
    # where interior-node count < max rank) don't pollute V / the ortho check
    num_rank = int(np.sum(evals > 1e-12 * evals[0]))
    r_max = min(max(POD_RANKS), S.shape[0], num_rank)
    V = (S_d.T @ evecs[:, :r_max]) / np.sqrt(np.maximum(evals[:r_max], 1e-300))
    ortho_dev = float(np.max(np.abs(V.T @ V - np.eye(r_max))))
    print(f"  POD: {S.shape[0]} snapshots (fit stride {POD_TIME_STRIDE}), "
          f"top-{r_max} ortho dev {ortho_dev:.1e}", flush=True)
    sv = np.sqrt(np.maximum(evals[:r_max], 0.0))
    del G, evecs
    Y_norm = np.linalg.norm(Y_d, axis=1)                  # (n_val*(T+1),)
    rms_rep = np.repeat(rms_va, n_t)
    C = Y_d @ V
    floors_traj, floors_snap = {}, {}
    for r in POD_RANKS:
        if r > r_max:
            continue
        recon = C[:, :r] @ V[:, :r].T
        diff = np.linalg.norm(recon - Y_d, axis=1)
        floors_traj[r] = float(np.mean(diff / rms_rep))
        floors_snap[r] = float(np.mean(diff / np.maximum(Y_norm, 1e-300)))
    return floors_traj, floors_snap, ortho_dev, sv[:32].tolist()


# ----------------------------- shared bits -----------------------------

def init_dense(key, d_in, d_out):
    W = jax.random.normal(key, (d_in, d_out), dtype=F32) * np.sqrt(1.0 / d_in)
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F32)}


def count_params(params):
    return sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))


# ----------------------------- grid-tied arm -----------------------------

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

def sample_points(np_rng, traj_idx, time_idx, cx, cy, w, c, n, P):
    """Half uniform grid nodes, half concentrated on the propagating wavefront:
    the disturbance is an annulus of radius ~ c*t around the blob center, width
    ~ the blob width (reflections beyond r ~ 0.5 are covered by the uniform
    half + clipping)."""
    B = len(traj_idx)
    Pu, Pb = P // 2, P - P // 2
    idx = np.empty((B, P), dtype=np.int64)
    idx[:, :Pu] = np_rng.integers(0, n * n, size=(B, Pu))
    for bi, (i, k) in enumerate(zip(traj_idx, time_idx)):
        t_phys = k * DT_SNAP
        r = np.abs(np_rng.normal(c[i] * t_phys, 2.0 * w[i], Pb))
        th = np_rng.uniform(0.0, 2.0 * np.pi, Pb)
        px = np.clip(cx[i] + r * np.cos(th), 0.0, 1.0)
        py = np.clip(cy[i] + r * np.sin(th), 0.0, 1.0)
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
          f"(Nyquist-capped), t_freq={T_FREQ}, substeps={SUBSTEPS}", flush=True)

    U, z, cx, cy, w, a, c = build_trajectories(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]
    rms_tr, msq_tr = traj_norms(U_tr)
    rms_va, msq_va = traj_norms(U_va)

    floors_traj, floors_snap, ortho_dev, sv = pod_floors(U_tr, U_va, rms_va)
    print("  POD floors (traj-RMS metric): "
          + "  ".join(f"r{r}={e:.3e}" for r, e in floors_traj.items()),
          flush=True)
    print("  POD floors (per-snap metric): "
          + "  ".join(f"r{r}={e:.3e}" for r, e in floors_snap.items()),
          flush=True)

    U_tr32 = jnp.asarray(U_tr, dtype=F32)              # (n_tr, T+1, n^2)
    z_tr32 = jnp.asarray(z_tr, dtype=F32)
    z_va32 = jnp.asarray(z_va, dtype=F32)
    msq_tr32 = jnp.asarray(msq_tr, dtype=F32)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    coords32 = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1),
                           dtype=F32)
    taus = np.arange(NUM_STEPS + 1, dtype=np.float64) / NUM_STEPS
    # f32 val copy on device for cheap periodic eval at stride-10 times
    U_va32_sub = jnp.asarray(U_va[:, ::10], dtype=F32)  # (n_val, 6, n^2)
    taus_sub = jnp.asarray(taus[::10], dtype=F32)
    msq_va32 = jnp.asarray(msq_va, dtype=F32)

    np_rng = np.random.default_rng(SEED)
    P = min(P_POINTS, n_nodes)
    results = {"N": N, "backend": backend, "seed": SEED, "n_train": N_TRAIN,
               "n_freq": N_FREQ, "t_freq": T_FREQ, "hidden": HIDDEN,
               "gt_rank": GT_RANK, "steps": STEPS, "gt_steps": GT_STEPS,
               "substeps": SUBSTEPS, "t_final": T_FINAL,
               "pod_time_stride": POD_TIME_STRIDE,
               "pod": {str(r): e for r, e in floors_traj.items()},
               "pod_snap": {str(r): e for r, e in floors_snap.items()},
               "pod_ortho_dev": ortho_dev, "singular_values": sv}

    # ---------------- grid-tied arm (full-grid loss) ----------------
    gt_params = init_grid_tied(jax.random.PRNGKey(SEED + 1), N)
    n_par_gt = count_params(gt_params)
    warmup = max(1, int(GT_STEPS * WARMUP_FRAC))
    gt_opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, GT_STEPS - warmup, end_value=1e-7),
        weight_decay=WEIGHT_DECAY)
    gt_state = gt_opt.init(gt_params)

    def gt_loss(p, U_b, z_b, tau_b, msq_b):
        pred = grid_tied_apply(p, z_b, tau_b)
        se = jnp.mean((pred - U_b) ** 2, axis=1)         # (B,)
        return jnp.mean(se / jnp.maximum(msq_b, 1e-30))

    @jax.jit
    def gt_step(p, s, U_b, z_b, tau_b, msq_b):
        loss, grads = jax.value_and_grad(gt_loss)(p, U_b, z_b, tau_b, msq_b)
        upd, s = gt_opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, loss

    t0 = time.time()
    for it in range(GT_STEPS):
        bi = np_rng.integers(0, N_TRAIN, BATCH)
        bk = np_rng.integers(0, NUM_STEPS + 1, BATCH)
        gt_params, gt_state, _ = gt_step(
            gt_params, gt_state, U_tr32[bi, bk],
            z_tr32[bi], jnp.asarray(taus[bk], dtype=F32), msq_tr32[bi])
    # f64 full eval over all val trajs/times, both metrics
    gt_traj, gt_snap = [], []
    for i in range(N_VAL):
        pred = np.asarray(grid_tied_apply(
            gt_params, jnp.tile(z_va32[i], (NUM_STEPS + 1, 1)),
            jnp.asarray(taus, dtype=F32)), dtype=np.float64)
        diff = np.linalg.norm(pred - U_va[i], axis=1)
        gt_traj.append(np.mean(diff / rms_va[i]))
        gt_snap.append(np.mean(diff / np.maximum(
            np.linalg.norm(U_va[i], axis=1), 1e-300)))
    results["grid_tied"] = float(np.mean(gt_traj))
    results["grid_tied_snap"] = float(np.mean(gt_snap))
    results["params_grid"] = n_par_gt
    print(f"  grid-tied ({n_par_gt} params, {GT_STEPS} steps, "
          f"{time.time()-t0:.0f}s): traj={results['grid_tied']:.3e} "
          f"snap={results['grid_tied_snap']:.3e}", flush=True)

    # ---------------- FiLM coord-net arm ----------------
    params = init_film_net(jax.random.PRNGKey(SEED))
    n_par = count_params(params)
    print(f"  film params: {n_par}", flush=True)
    warmup = max(1, int(STEPS * WARMUP_FRAC))
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, warmup, STEPS - warmup, end_value=1e-7),
        weight_decay=WEIGHT_DECAY)
    opt_state = opt.init(params)

    def batched_loss(p, U_b, z_b, tau_b, idx, msq_b):
        def one(z_i, tau_i, u_i, idx_i, msq_i):
            pred = film_apply(p, z_i, tau_i, coords32[idx_i])
            return jnp.mean((pred - u_i[idx_i]) ** 2) / jnp.maximum(msq_i, 1e-30)
        return jnp.mean(jax.vmap(one)(z_b, tau_b, U_b, idx, msq_b))

    @jax.jit
    def step(p, s, U_b, z_b, tau_b, idx, msq_b):
        loss, grads = jax.value_and_grad(batched_loss)(
            p, U_b, z_b, tau_b, idx, msq_b)
        upd, s = opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, loss

    @jax.jit
    def val_loss(p):
        def one_traj(zu):
            z_i, u_slices, msq_i = zu
            def one_t(carry, kt):
                tau_i, u_i = kt
                pred = film_apply(p, z_i, tau_i, coords32)
                return carry + jnp.mean((pred - u_i) ** 2), 0.0
            tot, _ = jax.lax.scan(one_t, 0.0, (taus_sub, u_slices))
            return tot / (len(taus_sub) * jnp.maximum(msq_i, 1e-30))
        return jnp.mean(jax.lax.map(one_traj, (z_va32, U_va32_sub, msq_va32)))

    best_val, best_params = float("inf"), params
    t0 = time.time()
    for it in range(STEPS):
        bi = np_rng.integers(0, N_TRAIN, BATCH)
        bk = np_rng.integers(0, NUM_STEPS + 1, BATCH)
        idx = jnp.asarray(sample_points(np_rng, bi, bk, cx, cy, w, c, N, P))
        params, opt_state, _ = step(
            params, opt_state, U_tr32[bi, bk], z_tr32[bi],
            jnp.asarray(taus[bk], dtype=F32), idx, msq_tr32[bi])
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
    per_time_t = {k: [] for k in EVAL_TIMES}
    per_time_s = {k: [] for k in EVAL_TIMES}
    film_traj, film_snap = [], []
    for i in range(N_VAL):
        rels_t, rels_s = [], []
        for k in range(NUM_STEPS + 1):
            pred = np.asarray(film_apply(best_params, z_va32[i],
                                         jnp.asarray(taus[k], dtype=F32),
                                         coords32), dtype=np.float64)
            d = np.linalg.norm(pred - U_va[i, k])
            rt = d / rms_va[i]
            rs = d / max(np.linalg.norm(U_va[i, k]), 1e-300)
            rels_t.append(rt)
            rels_s.append(rs)
            if k in per_time_t:
                per_time_t[k].append(rt)
                per_time_s[k].append(rs)
        film_traj.append(np.mean(rels_t))
        film_snap.append(np.mean(rels_s))
    results["film_coord"] = float(np.mean(film_traj))
    results["film_coord_snap"] = float(np.mean(film_snap))
    results["film_per_time"] = {str(k): float(np.mean(v))
                                for k, v in per_time_t.items()}
    results["film_per_time_snap"] = {str(k): float(np.mean(v))
                                     for k, v in per_time_s.items()}
    results["params_film"] = n_par
    print(f"RESULT N={N}  film-coord={results['film_coord']:.3e}  "
          f"grid-tied={results['grid_tied']:.3e}  "
          f"POD-6={floors_traj.get(6, float('nan')):.3e}  "
          f"POD-24={floors_traj.get(24, float('nan')):.3e}  "
          f"(traj-RMS metric; snap: film={results['film_coord_snap']:.3e})",
          flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"wave2d_results_N{N}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTDIR, f"wave2d_film_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, best_params), f)
    with open(os.path.join(OUTDIR, f"wave2d_gt_N{N}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, gt_params), f)
    print("wrote results + checkpoints", flush=True)


if __name__ == "__main__":
    main()
