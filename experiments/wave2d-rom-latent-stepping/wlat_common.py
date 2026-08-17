"""Wave-2D INR-decoder LATENT-STEPPING ROM -- shared machinery (Agent D).

Port of the Poisson/Burgers recipe (exp/2026-08-16-poisson2d-rom-objective,
exp/2026-08-16-burgers2d-rom-latent-stepping) to the HYPERBOLIC, dissipation-
free wave equation of the wave2d testbed (exp/2026-08-14-wave2d-coord-rom):

    u_tt = c^2 lap u  on [0,1]^2,  u = 0 on the walls,  u(.,0) = blob, u_t(.,0) = 0.

FOM      : wave2d_film.make_rollout imported verbatim -- Crank-Nicolson on the
           (u, v) system, SUBSTEPS=80 CN sub-steps per stored snapshot
           (dt_FOM = 2.5e-4), CG per step, energy drift <= 1e-11.  Its 51 stored
           u-snapshots per trajectory are the data and "the FOM" every ROM error
           is measured against.
ROM step : DECODER FOR u ONLY; the velocity is eliminated by the trapezoidal
           rule v_{n+1} = 2(u_{n+1}-u_n)/dt - v_n, so CN becomes the three-level
           Newmark (average-acceleration) form in u:
             R_n(u) = (u - 2u_n + u_{n-1}) - alpha L (u + 2u_n + u_{n-1}),  alpha = (c dt/2)^2
           first step (v_0 = 0):  R_0(u) = (u - u_0) - alpha L (u + u_0).
           The ROM steps at dt = DT_SNAP / RS (RS = ROM_SUBSTEPS, default 20 ->
           dt = 1e-3, 1000 latent steps per trajectory).  RS = 80 IS the FOM's
           operator; smaller RS adds a CN time-discretisation error which is
           measured separately by a same-dt u-only FOM (`newmark_fom`).
decoders : (i) hard-BC FiLM coordinate auto-decoder u(x;z) = eps b(x) film(x;z)
           (blat_common.CoordDecoder), one latent per (trajectory, time) snapshot;
           (ii) POD basis (linear control), same solver.
residuals: strong 'fd' -- point-local Newmark residual at m interior nodes from
           the decoder at the m 5-point stencils; carried state = stencil values
           of u_n and u_{n-1} (m,5) each;  'offgrid' -- continuum strong form with
           autodiff Laplacians (control).
           weak 'weak<M>' -- Galerkin against the M lowest discrete sine modes
           phi_i (exact eigenvectors of the FOM's ghost-zero Laplacian,
           -L phi_i = lam_i phi_i):
             R_i = w_i [ (1+alpha lam_i)(phi_i^T u + p_{n-1,i}) - 2 (1-alpha lam_i) p_n,i ],
           p_n = Phi^T u_n carried as M numbers; hyper-reduction = quadrature of
           phi_i^T u at m nodes (grid 'eq<m>' or MESHFREE pool 'eqoff<m>') with
           capped Lawson-Hanson NNLS weights fitted to reproduce the exact grid
           projections of DECODER-OUTPUT snapshots (Agent A recipe).  For the
           linear wave operator no derivatives of the decoder are ever needed and
           grid / meshfree pools use the same (discrete) lam_i because the
           quadrature targets the FOM's grid sums.  Weight w_i =
           (1+alpha lam_i)^-WEAK_ALPHA * (lam_1/lam_i)^(WEAK_BETA/2)
           ('weakl<M>' sets beta=1: an energy-type 1/sqrt(lam) weighting).
solvers  : lspg -- LM on ||R||^2 fully on device (lax.while_loop step, lax.scan
           rollout); galerkin -- damped Newton on J_D^T W R = 0 (python loop).
           Warm start z_n; cold start z_0 by a JITTED LM fit of the decoder to the
           KNOWN u0 (best of mean-t0 / nearest-IC training latents).  The held-out
           trajectory never touches the ROM path.
energy   : v reconstructed by the trapezoidal recursion on the decoded fields;
           E_n = dx^2 [ 0.5||v_n||^2 + 0.5 c^2 u_n^T(-L)u_n ] at the stored
           snapshots; drift = max_n |E_n - E_0| / E_0 (the FOM conserves it to
           1e-11; the ROM does not, and the drift is the stability diagnostic).
metrics  : traj-RMS (PRIMARY, wave2d testbed):  mean_t ||u_ROM - u||_t /
           sqrt(mean_t ||u(t)||^2) over the 51 stored snapshots; per-snapshot
           'snap' metric alongside.

Import order matters (env-var collisions): wave2d_film first (HIDDEN default 256
= the sweep checkpoints), then blat_common (which sets HIDDEN/N_LAYERS for
ms_parametric = the auto-decoder architecture AD_HIDDEN x AD_LAYERS).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# FROZEN copies of the validated building blocks live in ./deps (see deps/PROVENANCE.md);
# they are preferred over the live sibling worktrees so concurrent edits there cannot
# change the code under a running experiment, and cluster staging ships exactly this.
DEPS = os.path.join(HERE, "deps")
WAVE_DIR = os.path.join(DEPS, "wave2d-coord-rom")
BLAT_DIR = os.path.join(DEPS, "burgers2d-rom-latent-stepping")
if not os.path.isdir(WAVE_DIR):
    WAVE_DIR = os.path.join(_WT, "2026-08-14-wave2d-coord-rom", "experiments", "wave2d-coord-rom")
if not os.path.isdir(BLAT_DIR):
    BLAT_DIR = os.path.join(_WT, "2026-08-16-burgers2d-rom-latent-stepping", "experiments",
                            "burgers2d-rom-latent-stepping")
# blat_common imports burgers2d_film / ms_parametric / ms_autodecoder by name; put our
# frozen copies on sys.path FIRST so those imports resolve here regardless of the
# directory guesses inside blat_common.
for d in (WAVE_DIR, BLAT_DIR, os.path.join(DEPS, "burgers2d-coord-rom"),
          os.path.join(DEPS, "multistage-precision")):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import wave2d_film as wf                                # noqa: E402  (HIDDEN 256 sweep arch)
assert wf.HIDDEN == 256, "wave2d_film must keep the sweep architecture"
import blat_common as bc                                # noqa: E402
from blat_common import (mp, lm_solve, log, grid_coords, interior_indices,   # noqa: E402,F401
                         pod_basis, bc_factor, CoordDecoder, PODDecoder,
                         stencil_indices, test_modes, modes_at, nnls_capped, time_fn)

F64 = jnp.float64
N = wf.N
NUM_STEPS = wf.NUM_STEPS               # 50 stored snapshot intervals
DT_SNAP = wf.DT_SNAP                   # 0.02
FOM_SUBSTEPS = wf.SUBSTEPS             # 80
N_TRAIN, N_VAL, SEED = wf.N_TRAIN, wf.N_VAL, wf.SEED
RS = int(os.environ.get("ROM_SUBSTEPS", "20"))     # ROM sub-steps per snapshot interval
DT = DT_SNAP / RS
BC_MODE = bc.BC_MODE
K_LAT = bc.K_LAT
N_TEST = int(os.environ.get("N_TEST", "16"))
TEST_SEED = int(os.environ.get("TEST_SEED", str(SEED + 1)))
GN_BUDGET = int(os.environ.get("GN_BUDGET", "30"))
GN_TOL = float(os.environ.get("GN_TOL", "1e-9"))
IC_BUDGET = int(os.environ.get("IC_BUDGET", "100"))
WEAK_ALPHA = float(os.environ.get("WEAK_ALPHA", "1.0"))
WEAK_BETA = float(os.environ.get("WEAK_BETA", "0.0"))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_POOL = int(os.environ.get("EQ_POOL", "4096"))
AD_HIDDEN, AD_LAYERS = bc.AD_HIDDEN, bc.AD_LAYERS
EVAL_TIMES = wf.EVAL_TIMES

CONFIG = dict(pde="wave2d", N=N, dt_snap=DT_SNAP, num_steps=NUM_STEPS,
              fom_substeps=FOM_SUBSTEPS, rom_substeps=RS, dt_rom=DT,
              n_train=N_TRAIN, n_val=N_VAL, seed=SEED, test_seed=TEST_SEED,
              bc_mode=BC_MODE, k_lat=K_LAT, n_test=N_TEST, gn_budget=GN_BUDGET,
              gn_tol=GN_TOL, ic_budget=IC_BUDGET, weak_alpha=WEAK_ALPHA,
              weak_beta=WEAK_BETA, eq_snaps=EQ_SNAPS, eq_pool=EQ_POOL,
              ad_hidden=AD_HIDDEN, ad_layers=AD_LAYERS, x64=True)


# --------------------------- grid helpers ---------------------------

def lap_interior_field(u_int, n):
    """5-point Laplacian of an interior field (n_i^2,), ghost zeros on the walls."""
    ni = n - 2
    dx = 1.0 / (n - 1)
    U = jnp.pad(u_int.reshape(ni, ni), 1)
    return ((U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2]
             - 4.0 * U[1:-1, 1:-1]) / dx**2).reshape(-1)


def lap_full_field(u_flat, n):
    """5-point Laplacian of a full-grid field, zero on the walls (FOM convention)."""
    dx = 1.0 / (n - 1)
    U = u_flat.reshape(n, n)
    L = jnp.zeros_like(U)
    L = L.at[1:-1, 1:-1].set((U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2]
                              - 4.0 * U[1:-1, 1:-1]) / dx**2)
    return L.reshape(-1)


def energy_full(u_flat, v_flat, c, n):
    """Discrete energy dx^2 [0.5||v||^2 + 0.5 c^2 u^T(-L)u] -- identical to the
    FOM's energy_one (sum of squared forward differences)."""
    dx = 1.0 / (n - 1)
    U = u_flat.reshape(n, n)
    gx = (U[1:, :] - U[:-1, :]) / dx
    gy = (U[:, 1:] - U[:, :-1]) / dx
    return dx * dx * (0.5 * jnp.sum(v_flat**2) + 0.5 * c**2 * (jnp.sum(gx**2) + jnp.sum(gy**2)))


def traj_metrics(F, Ut):
    """F, Ut (T1, n^2).  Returns per-snapshot |diff|/rms (traj metric rows),
    per-snapshot |diff|/||u_t|| (snap metric rows), traj mean, snap mean."""
    d = np.linalg.norm(F - Ut, axis=1)
    rms = np.sqrt(np.mean(np.sum(Ut**2, axis=1)))
    per_t = d / rms
    per_s = d / np.maximum(np.linalg.norm(Ut, axis=1), 1e-300)
    return per_t, per_s, float(per_t.mean()), float(per_s.mean())


# --------------------------- data ---------------------------

def blob_ic(n, cx, cy, w, a):
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.asarray(wf.boundary_mask(n))
    return (a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)) * mask).reshape(-1)


def train_trajectories(n, chunk=64):
    """The wave2d sweep's TRAIN/VAL draw, reproduced verbatim from
    wave2d_film.build_trajectories but RETURNING the per-chunk max relative
    energy drift (the frozen function only prints it, so build_data could not
    threshold it -- Codex MUST).  wlat_verify asserts this reproduces
    wf.build_trajectories bit-for-bit."""
    cx, cy, w, a, c, z = wf.sample_params()
    m = len(cx)
    rollout, _ = wf.make_rollout(n)
    U = np.zeros((m, NUM_STEPS + 1, n * n))
    drift_max = 0.0
    for s_ in range(0, m, chunk):
        e_ = min(s_ + chunk, m)
        U0 = np.stack([blob_ic(n, cx[i], cy[i], w[i], a[i]) for i in range(s_, e_)])
        snaps, ens = rollout(jnp.asarray(U0), jnp.asarray(c[s_:e_]))
        U[s_:e_] = np.asarray(snaps).transpose(1, 0, 2)
        ens = np.asarray(ens)
        dm = float(np.max(np.abs(ens - ens[0]) / np.maximum(ens[0], 1e-300)))
        if not np.isfinite(dm) or dm > drift_max:      # NaN-propagating accumulate
            drift_max = dm
    return U, z, cx, cy, w, a, c, drift_max


def build_data(n=N):
    """TRAIN/VAL from SEED (identical draw to the wave2d sweep; the sweep
    checkpoints were model-selected on VAL, so VAL is NOT a test set).  TEST =
    N_TEST fresh trajectories from TEST_SEED.  ABORTS if the FOM's relative
    energy drift exceeds 1e-9 on EITHER split, or if anything is non-finite
    (a diverged CN/CG solve would otherwise silently become 'truth')."""
    U, z, cx, cy, w, a, c, drift_tr = train_trajectories(n)
    log(f"  TRAIN/VAL FOM: {U.shape[0]} trajectories, max rel energy drift {drift_tr:.2e}")
    if not np.isfinite(drift_tr) or drift_tr > 1e-9:
        raise SystemExit(f"TRAIN FOM energy drift {drift_tr:.2e} > 1e-9")
    if not np.all(np.isfinite(U)):
        raise SystemExit("non-finite training data")
    cxt, cyt, wt, at, ct, zt = wf.sample_params(seed=TEST_SEED, m=N_TEST)
    rollout, _ = wf.make_rollout(n)
    U0 = np.stack([blob_ic(n, cxt[i], cyt[i], wt[i], at[i]) for i in range(N_TEST)])
    snaps, ens = rollout(jnp.asarray(U0), jnp.asarray(ct))
    Ut = np.asarray(snaps).transpose(1, 0, 2)
    ens = np.asarray(ens)
    drift = np.abs(ens - ens[0]) / np.maximum(ens[0], 1e-300)
    dm = float(np.max(drift)) if np.all(np.isfinite(drift)) else float("nan")
    log(f"  TEST FOM: {N_TEST} trajectories, max rel energy drift {dm:.2e}")
    if not np.isfinite(dm) or dm > 1e-9:
        raise SystemExit(f"TEST FOM energy drift {dm:.2e} > 1e-9")
    if not np.all(np.isfinite(Ut)):
        raise SystemExit("non-finite test data")
    return dict(U=U, z=z, cx=cx, cy=cy, w=w, a=a, c=c, U_test=Ut, z_test=zt, c_test=ct,
                cx_test=cxt, cy_test=cyt, w_test=wt, a_test=at, test_energy_drift=dm,
                train_energy_drift=drift_tr)


def data_fingerprint(U):
    """Global moments PLUS a sha256 of the exact bytes and per-trajectory
    checksums (Codex SHOULD: two global moments at 1e-6 could pass a local
    corruption).  The sha256 is bit-exact and is the check that actually binds."""
    A = np.ascontiguousarray(np.asarray(U, dtype=np.float64))
    return dict(sum=float(np.sum(A)), sumsq=float(np.sum(A * A)), shape=list(A.shape),
                sha256=hashlib.sha256(A.tobytes()).hexdigest(),
                traj_sumsq_first8=[float(x) for x in np.sum(A[:8] ** 2, axis=(1, 2))])


# --------------------------- u-only Newmark FOM (same-dt reference + verification) ---------------------------

def make_newmark_fom(n, rs):
    """u-only CN/Newmark rollout at dt = DT_SNAP/rs on the full grid (masked
    walls), CG on (I - alpha L).  Returns fields at the 51 stored snapshots
    (T1, n^2) and the energies (T1,) via the trapezoidal v recursion.  With
    rs = 80 this must reproduce wave2d_film's (u,v) CN rollout to CG tolerance
    (checked in the smoke test): the two schemes are algebraically identical."""
    mask = wf.boundary_mask(n).reshape(-1)
    dt = DT_SNAP / rs

    def lap(u):
        return lap_full_field(u, n)

    def solve(rhs, alpha, x0):
        A = lambda v: jnp.where(mask > 0, v - alpha * lap(v), v)
        x, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=x0, tol=1e-12, maxiter=20000)
        return x * mask

    @jax.jit
    def rollout(u0, c):
        alpha = (0.5 * dt * c) ** 2
        # first step: (I - aL) u1 = (I + aL) u0
        u1 = solve(u0 + alpha * lap(u0), alpha, u0)
        v1 = 2.0 * (u1 - u0) / dt                       # v0 = 0

        def substep(carry, _):
            um, u, v = carry                          # u_{n-1}, u_n, v_n
            rhs = 2.0 * u - um + alpha * lap(2.0 * u + um)
            up = solve(rhs, alpha, 2.0 * u - um)
            vp = 2.0 * (up - u) / dt - v
            return (u, up, vp), None

        def snap_body(carry, k):
            # k = 0: from (u0,u1) take rs-1 more substeps to reach snapshot 1
            def one_interval(cc):
                cc, _ = jax.lax.scan(substep, cc, None, length=rs)
                return cc
            def first_interval(cc):
                cc, _ = jax.lax.scan(substep, cc, None, length=rs - 1)
                return cc
            carry = jax.lax.cond(k == 0, first_interval, one_interval, carry)
            um, u, v = carry
            return carry, (u, energy_full(u, v, c, n))

        _, (snaps, ens) = jax.lax.scan(snap_body, (u0, u1, v1), jnp.arange(NUM_STEPS))
        snaps = jnp.concatenate([u0[None], snaps], axis=0)
        e0 = energy_full(u0, jnp.zeros_like(u0), c, n)
        return snaps, jnp.concatenate([e0[None], ens])

    return rollout


# --------------------------- weak-form pieces (wave: linear) ---------------------------

def check_M(n, M):
    """blat_common.test_modes silently returns fewer than M modes when
    M > (n-2)^2, while the report would still say M (Codex MUST).  Fail loudly."""
    avail = (n - 2) ** 2
    if not (1 <= M <= avail):
        raise SystemExit(f"weak form asks for M={M} test modes but only {avail} "
                         f"exist on the interior of the N={n} grid")
    return M


def weak_weights(lam, c, alpha_w=WEAK_ALPHA, beta=WEAK_BETA):
    a = (0.5 * DT * c) ** 2
    w = (1.0 + a * lam) ** (-alpha_w)
    if beta != 0.0:
        w = w * (lam[0] / lam) ** (0.5 * beta)
    return w


def fit_eq_weights(dec, n, M, m, Z_snap, pool="grid", rng=None):
    """NNLS-EQ quadrature for the linear weak form: reproduce the exact grid
    projections Phi^T u_s of decoder-output snapshots u_s = D(z_s) with m
    weighted nodes from a candidate pool (interior grid nodes, or EQ_POOL random
    interior points = meshfree).  Returns dict(kind, idx|xy, w, info)."""
    rng = rng or np.random.default_rng(0)
    t0 = time.time()
    check_M(n, M)
    kx, ky, Phi, lam, _ = test_modes(n, M)
    coords = jnp.asarray(grid_coords(n))
    interior = interior_indices(n)
    xy_int = coords[jnp.asarray(interior)]
    kxj, kyj = jnp.asarray(kx, dtype=F64), jnp.asarray(ky, dtype=F64)
    if pool == "grid":
        cand_xy = xy_int
        Phi_c = np.asarray(Phi)
    else:
        assert dec.kind == "coord", "meshfree pool needs the coord decoder"
        cand_xy = jnp.asarray(rng.uniform(1.0 / (n - 1), 1.0 - 1.0 / (n - 1), size=(EQ_POOL, 2)))
        Phi_c = np.asarray(modes_at(cand_xy, kxj, kyj, n)[0])
    n_c = cand_xy.shape[0]
    if dec.kind == "coord":
        u_full = jax.jit(lambda z: dec(z, xy_int))
        u_cand = jax.jit(lambda z: dec(z, cand_xy)) if pool != "grid" else u_full
    else:
        u_full = jax.jit(lambda z: dec.rows(z, jnp.asarray(interior)))
        u_cand = u_full
    Phi_np = np.asarray(Phi)
    Gs, bs = [], []
    for z in Z_snap:
        z = jnp.asarray(z, dtype=F64)
        uf, uc = np.asarray(u_full(z)), np.asarray(u_cand(z))
        bs.append(Phi_np.T @ uf)
        Gs.append(Phi_c.T * uc[None, :])
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    sc = np.linalg.norm(G, axis=1) + 1e-300
    G, b = G / sc[:, None], b / sc
    wts, rnorm, n_outer = nnls_capped(G, b, max_support=m)
    supp = np.nonzero(wts > 0)[0]
    padded = 0
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]
    else:
        rest = np.setdiff1d(np.arange(n_c), supp)
        score = np.abs(G).mean(0)
        pad = rest[np.argsort(-score[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad]); padded = len(pad)
    wq, _, _ = nnls_capped(G[:, keep], b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    rnorm_final = float(np.linalg.norm(G[:, keep] @ wq - b))
    info = dict(support=int(len(supp)), padded=int(padded), rnorm_capped=float(rnorm),
                rnorm_final=rnorm_final, b_norm=float(np.linalg.norm(b)),
                rel_fit=rnorm_final / float(np.linalg.norm(b)), n_rows=int(G.shape[0]),
                n_cand=int(n_c), secs=time.time() - t0, M=int(M), m=int(len(keep)), pool=pool)
    log(f"  NNLS-EQ {pool} M={M} m={m}: support {len(supp)} (+{padded} pad), "
        f"rel fit {info['rel_fit']:.2e} [{info['secs']:.0f}s]")
    out = dict(kind="grid" if pool == "grid" else "offgrid", w=wq, info=info)
    if pool == "grid":
        out["idx"] = interior[keep]
    else:
        out["xy"] = np.asarray(cand_xy)[keep]
    return out


# --------------------------- ROM step operators ---------------------------
#
# Every ops dict exposes:
#   r_w(z, prev, c)      weighted residual vector for the GENERAL step
#   r_w0(z, prev, c)     ... for the FIRST step (u_{-1} eliminated by v_0 = 0)
#   state_of(z)          carried per-latent state S(z) (stencil values / mode
#                        projections / point values)
#   prev = (S(z_n), S(z_{n-1}))  stacked as one array (2, ...) ; the first step
#   uses prev[0] only.
#   full(z), m, M, tol_scale, step_jit, rollout_jit, solver

def _make_ops_from(r_gen, r_first, state_of, full, m, solver, tol_scale, colloc_info=None, M=None):
    def r_w(z, prev, c):
        return r_gen(z, prev[0], prev[1], c)

    def r_w0(z, prev, c):
        return r_first(z, prev[0], c)

    ops = dict(r_w=jax.jit(r_w), r_w0=jax.jit(r_w0), state_of=jax.jit(state_of),
               full=jax.jit(full), full_batch=jax.jit(jax.vmap(full)), m=m, M=M,
               solver=solver, tol_scale=tol_scale, colloc_info=colloc_info)
    ops.update(_device_solvers(r_w, r_w0, state_of))
    return ops


def _lm_while(res_fn, z0, args, tol_abs, budget):
    """Generic on-device LM (lax.while_loop) on ||res_fn(z, *args)||^2, same
    acceptance rule as ms_autodecoder.lm_solve.  Returns z, rn, n_jac, accepted,
    reason (0 budget, 1 tol, 2 stalled, 3 lambda_max, 4 tol_at_init, 5 nan_at_init), attempts."""
    def rJ(z):
        return res_fn(z, *args), jax.jacfwd(lambda zz: res_fn(zz, *args))(z)
    def rn_fn(z):
        return jnp.linalg.norm(res_fn(z, *args))
    r0, J0 = rJ(z0)
    rn0 = jnp.linalg.norm(r0)
    K = z0.shape[0]
    init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                            jnp.where(rn0 <= tol_abs, jnp.int32(4), jnp.int32(0)))
    init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
            jnp.int32(1), init_reason)

    def cond(s):
        return (s[8] == 0) & (s[5] < budget)

    def body(s):
        z, r, J, rn, lam, att, acc, nJ, _ = s
        H = J.T @ J
        g = J.T @ r
        D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
        dz = jnp.linalg.solve(H + lam * D, -g)
        finite = jnp.all(jnp.isfinite(dz))
        tiny = finite & (jnp.linalg.norm(dz) <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
        z_new = z + jnp.where(finite, dz, 0.0)
        rn_new = rn_fn(z_new)
        accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
        r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new), lambda: (r, J))
        rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
        z = jnp.where(accept, z_new, z)
        rn = jnp.where(accept, rn_new, rn)
        lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 10.0, 1e12))
        acc = acc + accept.astype(jnp.int32)
        nJ = nJ + accept.astype(jnp.int32)
        reason = jnp.where(accept & (rn <= tol_abs), 1,
                  jnp.where((accept & (rel_dec < 1e-12)) | tiny, 2,
                   jnp.where((~accept) & (lam >= 1e12), 3, 0))).astype(jnp.int32)
        return (z, r2, J2, rn, lam, att + 1, acc, nJ, reason)

    z, r, J, rn, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
    return z, rn, nJ, acc, reason, att


def _device_solvers(r_w, r_w0, state_of):
    def step_gen(z0, prev, c, tol_abs, budget):
        return _lm_while(r_w, z0, (prev, c), tol_abs, budget)

    def step_first(z0, prev, c, tol_abs, budget):
        return _lm_while(r_w0, z0, (prev, c), tol_abs, budget)

    step_gen_j = jax.jit(step_gen, static_argnums=(4,))
    step_first_j = jax.jit(step_first, static_argnums=(4,))

    def rollout_fn(z0, c, tol_abs, budget, n_steps):
        """Full latent rollout on device: first step, then n_steps-1 general
        steps (lax.scan).  Returns latents (n_steps+1, K) incl. z0, rn, nJ, reason."""
        S0 = state_of(z0)
        prev0 = jnp.stack([S0, S0])
        z1, rn1, nJ1, acc1, re1, att1 = _lm_while(r_w0, z0, (prev0, c), tol_abs, budget)
        S1 = state_of(z1)

        def body(carry, _):
            z, Sn, Snm = carry
            prev = jnp.stack([Sn, Snm])
            z2, rn, nJ, acc, re, att = _lm_while(r_w, z, (prev, c), tol_abs, budget)
            S2 = state_of(z2)
            return (z2, S2, Sn), (z2, rn, nJ, re, att)

        (_, _, _), (Z, rns, nJs, res, atts) = jax.lax.scan(body, (z1, S1, S0), None,
                                                           length=n_steps - 1)
        Z = jnp.concatenate([z0[None], z1[None], Z], axis=0)
        rns = jnp.concatenate([rn1[None], rns]); nJs = jnp.concatenate([nJ1[None], nJs])
        res = jnp.concatenate([re1[None], res]); atts = jnp.concatenate([att1[None], atts])
        return Z, rns, nJs, res, atts

    return dict(step_gen=step_gen_j, step_first=step_first_j,
                rollout_jit=jax.jit(rollout_fn, static_argnums=(3, 4)))


def make_strong_ops(dec, n, colloc, solver="lspg"):
    """Strong-form ('fd') Newmark residual.  colloc: dict(kind='grid', idx) ->
    point-local at the m 5-point stencils (exact FOM interior operator);
    dict(kind='offgrid', xy) -> continuum strong form with autodiff Laplacians
    (control; coord decoder only)."""
    coords = jnp.asarray(grid_coords(n))
    dx = 1.0 / (n - 1)
    if colloc["kind"] == "grid":
        idx = np.asarray(colloc["idx"])
        st = jnp.asarray(stencil_indices(idx, n))
        m = idx.size
        xy_st = coords[st.reshape(-1)]
        if dec.kind == "coord":
            vals_st = lambda z: dec(z, xy_st).reshape(m, 5)
        else:
            vals_st = lambda z: dec.rows(z, st.reshape(-1)).reshape(m, 5)

        def lap_of(us):
            return (us[:, 1] + us[:, 2] + us[:, 3] + us[:, 4] - 4.0 * us[:, 0]) / dx**2

        def state_of(z):
            return vals_st(z)                                     # (m,5)

        def r_gen(z, Sn, Snm, c):
            a = (0.5 * DT * c) ** 2
            us = vals_st(z)
            return ((us[:, 0] - 2.0 * Sn[:, 0] + Snm[:, 0])
                    - a * (lap_of(us) + 2.0 * lap_of(Sn) + lap_of(Snm)))

        def r_first(z, S0, c):
            a = (0.5 * DT * c) ** 2
            us = vals_st(z)
            return (us[:, 0] - S0[:, 0]) - a * (lap_of(us) + lap_of(S0))
    else:
        assert dec.kind == "coord"
        xy = jnp.asarray(colloc["xy"])
        m = xy.shape[0]

        def u_lap(z):
            def f(p):
                return dec(z, p[None, :])[0]
            def one(p):
                H = jax.hessian(f)(p)
                return jnp.stack([f(p), H[0, 0] + H[1, 1]])
            return jax.vmap(one)(xy)                              # (m,2)

        def state_of(z):
            return u_lap(z)

        def r_gen(z, Sn, Snm, c):
            a = (0.5 * DT * c) ** 2
            S = u_lap(z)
            return (S[:, 0] - 2.0 * Sn[:, 0] + Snm[:, 0]) - a * (S[:, 1] + 2.0 * Sn[:, 1] + Snm[:, 1])

        def r_first(z, S0, c):
            a = (0.5 * DT * c) ** 2
            S = u_lap(z)
            return (S[:, 0] - S0[:, 0]) - a * (S[:, 1] + S0[:, 1])

    def full(z):
        return dec(z, coords) if dec.kind == "coord" else dec.V @ z

    return _make_ops_from(r_gen, r_first, state_of, full, m, solver, float(np.sqrt(m)))


def make_weak_ops(dec, n, colloc, M=64, solver="lspg", beta=None, alpha_w=None):
    """Weak-form Galerkin against M discrete sine modes.  colloc: full grid
    (idx=interior, w=None -> exact grid sums), 'grid' with NNLS weights, or
    'offgrid' meshfree points with NNLS weights.  Carried state = the M mode
    projections of the decoded field."""
    check_M(n, M)
    kx, ky, Phi, lam, _ = test_modes(n, M)
    assert Phi.shape[1] == M, f"test_modes returned {Phi.shape[1]} of {M} modes"
    coords = jnp.asarray(grid_coords(n))
    interior = interior_indices(n)
    lam_j = jnp.asarray(lam, dtype=F64)
    kxj, kyj = jnp.asarray(kx, dtype=F64), jnp.asarray(ky, dtype=F64)
    beta = WEAK_BETA if beta is None else beta
    alpha_w = WEAK_ALPHA if alpha_w is None else alpha_w
    if colloc["kind"] == "grid":
        idx = np.asarray(colloc["idx"])
        m = idx.size
        w = jnp.asarray(colloc.get("w") if colloc.get("w") is not None else np.ones(m), dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx)
        Phi_q = jnp.asarray(Phi[pos]) * w[:, None]                     # (m,M)
        xy_q = coords[jnp.asarray(idx)]
        if dec.kind == "coord":
            u_q = lambda z: dec(z, xy_q)
        else:
            u_q = lambda z: dec.rows(z, jnp.asarray(idx))
    else:
        assert dec.kind == "coord"
        xy_q = jnp.asarray(colloc["xy"])
        m = xy_q.shape[0]
        w = jnp.asarray(colloc["w"], dtype=F64)
        Phi_q = modes_at(xy_q, kxj, kyj, n)[0] * w[:, None]
        u_q = lambda z: dec(z, xy_q)

    def state_of(z):
        return Phi_q.T @ u_q(z)                                        # (M,)

    def r_gen(z, pn, pnm, c):
        a = (0.5 * DT * c) ** 2
        wt = weak_weights(lam_j, c, alpha_w, beta)
        p = state_of(z)
        # phi^T[(u - 2u_n + u_{n-1}) - a L (u + 2u_n + u_{n-1})], phi^T L v = -lam phi^T v
        return wt * ((1.0 + a * lam_j) * (p + pnm) - 2.0 * (1.0 - a * lam_j) * pn)

    def r_first(z, p0, c):
        a = (0.5 * DT * c) ** 2
        wt = weak_weights(lam_j, c, alpha_w, beta)
        p = state_of(z)
        return wt * ((1.0 + a * lam_j) * p - (1.0 - a * lam_j) * p0)

    def full(z):
        return dec(z, coords) if dec.kind == "coord" else dec.V @ z

    ops = _make_ops_from(r_gen, r_first, state_of, full, m, solver, float(np.sqrt(interior.size)),
                         colloc_info=colloc.get("info"), M=M)
    ops["Phi_q"] = Phi_q
    ops["u_q"] = jax.jit(u_q)
    ops["weak_alpha"] = float(alpha_w)
    ops["weak_beta"] = float(beta)
    return ops


def make_collocation(name, n, rng, u0=None):
    """full | rand<m> | biased<m> | offgrid<m> (strong form).  biased: half
    uniform, half by the KNOWN |u0| as density (no held-out information)."""
    interior = interior_indices(n)
    if name == "full":
        return dict(kind="grid", idx=interior)
    if name.startswith("rand"):
        m = min(int(name[4:]), interior.size)
        return dict(kind="grid", idx=np.sort(rng.choice(interior, m, replace=False)))
    if name.startswith("biased"):
        m = min(int(name[6:]), interior.size)
        p = np.abs(np.asarray(u0).reshape(n, n)[1:-1, 1:-1].reshape(-1))
        p = p / p.sum()
        P = 0.5 / p.size + 0.5 * p
        pick = rng.choice(interior.size, m, replace=False, p=P)
        return dict(kind="grid", idx=np.sort(interior[pick]))
    if name.startswith("offgrid"):
        m = int(name[7:])
        return dict(kind="offgrid", xy=rng.uniform(1.0 / (n - 1), 1.0 - 1.0 / (n - 1), size=(m, 2)))
    raise ValueError(name)


# --------------------------- galerkin (python) step ---------------------------

def galerkin_step(ops, z0, prev, c, first, tol_abs, budget=GN_BUDGET, lam0=1e-6):
    """Damped Newton on g = JD^T r_w = 0 with the GN Jacobian JD^T J (JD = the
    Jacobian of the carried state = mode projections / point values of the
    tangent basis), backtracking on ||g||.  Python loop (not timed)."""
    r_fn = ops["r_w0"] if first else ops["r_w"]
    S_fn = ops["state_of"]
    def rJ(z):
        r = r_fn(z, prev, c)
        J = jax.jacfwd(lambda zz: r_fn(zz, prev, c))(z)
        JD = jax.jacfwd(S_fn)(z)
        JD = JD.reshape(-1, JD.shape[-1])
        return r, J, JD
    rJ = jax.jit(rJ)

    def tangent(JD, r):
        # strong form: state = (m,5) stencils / (m,2) point values -> the residual
        # rows live at the CENTRES: use column block 0 -> (m,K)
        if JD.shape[0] != r.shape[0]:
            JD = JD.reshape(r.shape[0], -1, JD.shape[-1])[:, 0, :]
        return JD

    z = z0
    r, J, JD = rJ(z)
    JD = tangent(JD, r)
    g = JD.T @ r
    gn = float(jnp.linalg.norm(g))
    lam = lam0
    n_J = 1
    acc = rej = 0
    reason = "budget"
    for _ in range(budget):
        A = JD.T @ J
        dz = jnp.linalg.solve(A + lam * jnp.eye(A.shape[0], dtype=F64), -g)
        if not bool(jnp.all(jnp.isfinite(dz))):
            lam = min(lam * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "nan_step"; break
            continue
        alpha, ok = 1.0, False
        for _ in range(8):
            z_new = z + alpha * dz
            r2, J2, JD2 = rJ(z_new); n_J += 1
            JD2 = tangent(JD2, r2)
            g2 = JD2.T @ r2
            gn2 = float(jnp.linalg.norm(g2))
            if np.isfinite(gn2) and gn2 < gn:
                ok = True
                break
            alpha *= 0.5
        if not ok:
            lam = min(max(lam, 1e-8) * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "no_descent"; break
            continue
        rel_dec = (gn - gn2) / gn
        z, r, J, JD, g, gn = z_new, r2, J2, JD2, g2, gn2
        acc += 1
        lam = max(lam / 3.0, 0.0)
        rn = float(jnp.linalg.norm(r))
        if gn <= tol_abs * 1e-2 or rn <= tol_abs:
            reason = "tol"; break
        if rel_dec < 1e-12:
            reason = "stalled"; break
    return z, dict(rn=float(jnp.linalg.norm(r)), n_jac=n_J, accepted=acc, rejected=rej, reason=reason)


# --------------------------- cold start ---------------------------

def make_ic_solver(dec, n):
    """JITTED LM fit of the decoder to a known u0 on the full grid (coord decoder);
    POD: projection.  Returns fit(u0, z_inits (S,K)) -> (z_best, rel_best, idx_best)."""
    coords = jnp.asarray(grid_coords(n))
    if dec.kind == "pod":
        def fit(u0, z_inits):
            z = dec.V.T @ u0
            rel = jnp.linalg.norm(dec.V @ z - u0) / jnp.linalg.norm(u0)
            return z, rel, jnp.int32(-1)
        return jax.jit(fit)

    def res(z, u0):
        return dec(z, coords) - u0

    def fit(u0, z_inits):
        def one(z0):
            z, rn, nJ, acc, re, att = _lm_while(res, z0, (u0,), jnp.asarray(0.0, F64), IC_BUDGET)
            return z, rn / jnp.linalg.norm(u0), nJ
        Zs, rels, nJs = jax.vmap(one)(z_inits)
        i = jnp.argmin(rels)
        return Zs[i], rels[i], i
    return jax.jit(fit)


# --------------------------- rollout (python driver) ---------------------------

def energy_curves(Uall, c, n, rs):
    """Two energy estimates for a decoded sub-step trajectory Uall (n_steps+1, n^2).

    KINEMATIC: v_k = 2(u_k - u_{k-1})/dt - v_{k-1}, v_0 = 0 -- the CN trapezoidal
      relation.  Exact for an exact CN trajectory, but the recursion has a (-1)^k
      mode: a state error e injected at one step leaves a persistent O(e/dt)
      alternating velocity, and random per-step decoder error accumulates like
      sqrt(k)/dt.  So on a ROM trajectory it measures the decoder's grid-scale
      noise as much as the ROM's dynamics (Codex MUST).
    DYNAMIC:  v_k = v_{k-1} + (dt c^2 / 2)(L u_{k-1} + L u_k) -- the FOM's own
      velocity update.  This one integrates the error rather than differentiating
      it, so it is the fair long-horizon stability diagnostic; the DEFECT between
      the two velocities is reported as the consistency measure.

    Returns dict with both energy curves at the stored snapshots, their drifts,
    and the max relative kinematic-vs-dynamic velocity defect.
    """
    dt = DT_SNAP / rs
    e_fn = jax.jit(lambda u, v: energy_full(u, v, float(c), n))
    lap_j = jax.jit(lambda u: lap_full_field(u, n))
    v_kin = jnp.zeros((n * n,), F64)
    v_dyn = jnp.zeros((n * n,), F64)
    u_prev = jnp.asarray(Uall[0])
    Lp = lap_j(u_prev)
    Ek = [float(e_fn(u_prev, v_kin))]
    Ed = [float(e_fn(u_prev, v_dyn))]
    defect = 0.0
    vscale = 0.0
    for k in range(1, Uall.shape[0]):
        u = jnp.asarray(Uall[k])
        Lu = lap_j(u)
        v_kin = 2.0 * (u - u_prev) / dt - v_kin
        v_dyn = v_dyn + 0.5 * dt * float(c) ** 2 * (Lp + Lu)
        if k % rs == 0:
            Ek.append(float(e_fn(u, v_kin)))
            Ed.append(float(e_fn(u, v_dyn)))
            d = float(jnp.linalg.norm(v_kin - v_dyn))
            sc = float(jnp.linalg.norm(v_dyn))
            defect = max(defect, d / max(sc, 1e-300))
            vscale = max(vscale, sc)
        u_prev, Lp = u, Lu
    Ek = np.asarray(Ek); Ed = np.asarray(Ed)
    out = dict(energy_kin=Ek.tolist(), energy_dyn=Ed.tolist(),
               energy_drift_max=float(np.max(np.abs(Ek - Ek[0]) / Ek[0])),
               energy_final_ratio=float(Ek[-1] / Ek[0]),
               energy_dyn_drift_max=float(np.max(np.abs(Ed - Ed[0]) / Ed[0])),
               energy_dyn_final_ratio=float(Ed[-1] / Ed[0]),
               v_kin_dyn_defect_max=float(defect))
    if not all(np.isfinite(x) for x in (out["energy_drift_max"], out["energy_dyn_drift_max"])):
        out["energy_nonfinite"] = True
    return out


REASONS = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max", 4: "tol_at_init", 5: "nan_at_init"}


def rollout(dec, n, ops, z0, c, u0_rms, U_true=None, budget=GN_BUDGET, tol=GN_TOL,
            energies=True):
    """Latent time stepping over NUM_STEPS*RS Newmark steps.

    A run counts as COMPLETE only if every step finished (n_done == n_steps), every
    snapshot latent is finite, the decoded fields are finite, and -- when energies
    were requested -- the energy curves are finite (Codex MUST: `complete` used to
    depend on the snapshot latents alone, so a step that returned its warm start
    unchanged after a NaN could enter the headline averages).  Incomplete runs get
    NaN errors and are counted, never averaged.
    """
    z0 = jnp.asarray(z0, dtype=F64)
    n_steps = NUM_STEPS * RS
    tol_abs = tol * float(u0_rms) * ops["tol_scale"]
    t0 = time.perf_counter()
    n_attempted = n_steps
    if ops["solver"] == "lspg":
        Z, rns, nJs, res, atts = ops["rollout_jit"](z0, float(c), tol_abs, budget, n_steps)
        Z.block_until_ready()
        wall = time.perf_counter() - t0
        Z = np.asarray(Z); rns = np.asarray(rns); nJs = np.asarray(nJs)
        res = np.asarray(res); atts = np.asarray(atts)
        reasons = [REASONS[int(r)] for r in res]
    else:
        z = z0
        S0 = ops["state_of"](z)
        prev = jnp.stack([S0, S0])
        Zl, rns, nJs, atts, reasons = [np.asarray(z)], [], [], [], []
        Sn, Snm = S0, S0
        n_attempted = 0
        for k in range(n_steps):
            z, info = galerkin_step(ops, z, prev, float(c), k == 0, tol_abs, budget)
            n_attempted += 1
            S2 = ops["state_of"](z)
            Snm, Sn = Sn, S2
            prev = jnp.stack([Sn, Snm])
            Zl.append(np.asarray(z)); rns.append(info["rn"]); nJs.append(info["n_jac"])
            atts.append(info["accepted"] + info["rejected"]); reasons.append(info["reason"])
            if not np.all(np.isfinite(Zl[-1])):
                reasons[-1] = "blowup"; break
        wall = time.perf_counter() - t0
        Z = np.stack(Zl)
        rns = np.asarray(rns, dtype=float); nJs = np.asarray(nJs, dtype=float)
        atts = np.asarray(atts, dtype=float)
        res = np.array([5 if r_ in ("blowup", "nan_step") else 0 for r_ in reasons])
        if Z.shape[0] < n_steps + 1:                  # truncated: pad with NaN
            Z = np.concatenate([Z, np.full((n_steps + 1 - Z.shape[0], Z.shape[1]), np.nan)])
            pad = n_steps - rns.size
            rns = np.concatenate([rns, np.full(pad, np.nan)])
            nJs = np.concatenate([nJs, np.full(pad, np.nan)])
            atts = np.concatenate([atts, np.full(pad, np.nan)])
            res = np.concatenate([res, np.full(pad, 5)])
            reasons = reasons + ["blowup"] * pad
    # a step "failed" if its residual is non-finite or the LM saw NaN at init
    bad = (~np.isfinite(rns)) | (res == 5) | (~np.all(np.isfinite(Z[1:]), axis=1))
    n_done = int(np.argmax(bad)) if bad.any() else n_steps
    out = dict(n_done=n_done, n_steps=n_steps, n_attempted=int(n_attempted),
               iters=nJs[: max(n_done, 0)].tolist(), res=rns[: max(n_done, 0)].tolist(),
               wall_s=wall,
               # Codex MUST: divide by the steps actually ATTEMPTED, not requested
               step_time_ms=1e3 * wall / max(n_attempted if ops["solver"] != "lspg" else n_steps, 1),
               reasons={r: int(sum(1 for x in reasons[:n_done] if x == r)) for r in set(reasons[:n_done])})
    out["iters_cold"] = float(nJs[0]) if n_done >= 1 else float("nan")
    out["iters_warm_mean"] = float(np.mean(nJs[1:n_done])) if n_done >= 2 else float("nan")
    out["iters_warm_max"] = float(np.max(nJs[1:n_done])) if n_done >= 2 else float("nan")
    out["lm_attempts_warm_mean"] = float(np.mean(atts[1:n_done])) if n_done >= 2 else float("nan")
    # decoded fields at the stored snapshots (every RS-th latent)
    idx_snap = np.arange(0, n_steps + 1, RS)
    Zs = Z[idx_snap]
    ok = np.all(np.isfinite(Zs), axis=1) & (idx_snap <= n_done)
    F = np.full((NUM_STEPS + 1, n * n), np.nan)
    dec_all = ops["full_batch"]
    if ok.any():
        F[ok] = np.asarray(dec_all(jnp.asarray(Zs[ok])))
    out["Z_snap"] = Zs
    complete = bool(n_done == n_steps) and bool(ok.all()) and bool(np.all(np.isfinite(F)))
    if U_true is not None:
        Ut = np.asarray(U_true)
        per_t, per_s, tr, sn = traj_metrics(F, Ut)
        per_t = np.where(ok, per_t, np.nan); per_s = np.where(ok, per_s, np.nan)
        complete = complete and bool(np.all(np.isfinite(per_t)))
        out["per_time"] = per_t.tolist(); out["per_time_snap"] = per_s.tolist()
        out["traj_rel"] = float(np.mean(per_t)) if complete else float("nan")
        out["snap_rel"] = float(np.mean(per_s)) if complete else float("nan")
    if energies and complete:
        CH = 256
        Uall = np.concatenate([np.asarray(dec_all(jnp.asarray(Z[s_:s_ + CH])))
                               for s_ in range(0, n_steps + 1, CH)])
        if np.all(np.isfinite(Uall)):
            out.update(energy_curves(Uall, c, n, RS))
        if out.pop("energy_nonfinite", False) or "energy_drift_max" not in out:
            complete = False                      # Codex MUST: never hide an energy NaN
            out["traj_rel"] = float("nan"); out["snap_rel"] = float("nan")
    out["complete"] = complete
    out["fields"] = F
    return out


# --------------------------- residual verification ---------------------------

def newmark_substeps(n, rs, u0, c, n_steps=None):
    """EVERY u-only Newmark sub-step (n_steps+1, n^2) at dt = DT_SNAP/rs, CG tol
    1e-13 -- the exact-trajectory calibration input for energy_curves."""
    n_steps = NUM_STEPS * rs if n_steps is None else n_steps
    mask = wf.boundary_mask(n).reshape(-1)
    dt = DT_SNAP / rs
    alpha = (0.5 * dt * c) ** 2
    lap = lambda u: lap_full_field(u, n)
    A = lambda v: jnp.where(mask > 0, v - alpha * lap(v), v)
    solve = jax.jit(lambda rhs, x0: jax.scipy.sparse.linalg.cg(
        A, rhs, x0=x0, tol=1e-13, maxiter=50000)[0] * mask)
    u0 = jnp.asarray(u0, F64)
    u = solve(u0 + alpha * lap(u0), u0)
    um = u0
    out = [np.asarray(u0), np.asarray(u)]
    for _ in range(n_steps - 1):
        up = solve(2.0 * u - um + alpha * lap(2.0 * u + um), 2.0 * u - um)
        um, u = u, up
        out.append(np.asarray(u))
    return np.stack(out)


def newmark_first_states(n, rs, u0, c, k=3):
    """The first k+1 u-only Newmark states (u0, u1, ..., uk) at dt = DT_SNAP/rs
    on the full grid, CG tol 1e-13 (for verifying the ROM residual operators)."""
    mask = wf.boundary_mask(n).reshape(-1)
    dt = DT_SNAP / rs
    alpha = (0.5 * dt * c) ** 2
    lap = lambda u: lap_full_field(u, n)
    A = lambda v: jnp.where(mask > 0, v - alpha * lap(v), v)
    def solve(rhs, x0):
        x, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=x0, tol=1e-13, maxiter=50000)
        return x * mask
    u0 = jnp.asarray(u0, F64)
    us = [u0, solve(u0 + alpha * lap(u0), u0)]
    for _ in range(k - 1):
        um, u = us[-2], us[-1]
        us.append(solve(2.0 * u - um + alpha * lap(2.0 * u + um), 2.0 * u - um))
    return [np.asarray(u) for u in us]


def verify_residual_ops(n, c=1.3, M=32, m=None):
    """Feed exact Newmark states through the ROM residual operators via an
    identity 'decoder' (V = I): every residual must vanish to CG tolerance.
    Returns dict of max |r| / ||u|| for strong-full, strong-subset, weak-full."""
    u0 = blob_ic(n, 0.4, 0.55, 0.1, 3.0)
    st = newmark_first_states(n, RS, u0, c, 3)
    dec = PODDecoder(np.eye(n * n))
    out = {}
    interior = interior_indices(n)
    rng = np.random.default_rng(0)
    for name, col in (("strong_full", dict(kind="grid", idx=interior)),
                      ("strong_rand", dict(kind="grid", idx=np.sort(rng.choice(interior, m or min(64, interior.size), replace=False))))):
        ops = make_strong_ops(dec, n, col)
        S = [ops["state_of"](jnp.asarray(u)) for u in st]
        r1 = ops["r_w0"](jnp.asarray(st[1]), jnp.stack([S[0], S[0]]), c)
        r2 = ops["r_w"](jnp.asarray(st[2]), jnp.stack([S[1], S[0]]), c)
        r3 = ops["r_w"](jnp.asarray(st[3]), jnp.stack([S[2], S[1]]), c)
        out[name] = float(max(jnp.max(jnp.abs(r)) for r in (r1, r2, r3)) / np.linalg.norm(u0))
    ops = make_weak_ops(dec, n, dict(kind="grid", idx=interior, w=None), M=M)
    S = [ops["state_of"](jnp.asarray(u)) for u in st]
    r1 = ops["r_w0"](jnp.asarray(st[1]), jnp.stack([S[0], S[0]]), c)
    r2 = ops["r_w"](jnp.asarray(st[2]), jnp.stack([S[1], S[0]]), c)
    out["weak_full"] = float(max(jnp.max(jnp.abs(r)) for r in (r1, r2)) / np.linalg.norm(u0))
    # and a NON-solution must NOT vanish (guards against a trivially-zero residual)
    r_bad = ops["r_w"](jnp.asarray(st[1]), jnp.stack([S[1], S[0]]), c)
    out["weak_nonsolution"] = float(jnp.max(jnp.abs(r_bad)) / np.linalg.norm(u0))
    return out
