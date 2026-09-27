"""1D viscous Burgers testbed for the separable EQ-decoder (2026-08-27).

New 1D port of the 2D machinery (blat_common.py + sep_common.py) for the
frozen-decoder node-learning screening at N in {128, 256, 512}.  The FOM
mirrors the 2D testbed's discretization one dimension down: sign-upwind
advection, centered diffusion, backward Euler dt=0.005 x 50 steps, Newton
with a DENSE interior Jacobian (n_i <= 510: direct solve, no BiCGStab and
none of its breakdown landmines).  Weak form on the M lowest discrete sine
modes, which are EXACT eigenvectors of the 1D Dirichlet Laplacian, so all
linear terms are exact through A = Phi^T G (the exlin rule) and only the
advection term u*u_x is sampled.

Self-contained on purpose: importing blat_common drags in the 2D FiLM stack.
nnls_capped is copied verbatim from blat_common.py (same algorithm, same
tolerances); test_modes_1d / upwind_adv_field_1d are the 1D analogs of the
blat_common functions with the same normalization rules; the decoder is the
sep_common two-track model with a 1D coordinate and bc(x) = 4 x (1-x).
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

F64 = jnp.float64

DT = 0.005
NUM_STEPS = 50
NEWTON_ITERS = 8
WEAK_ALPHA = 1.0


def log(*a):
    print(*a, flush=True)


# ------------------------------- grid ---------------------------------------

def grid_coords_1d(n):
    return np.linspace(0.0, 1.0, n)[:, None]                  # (n, 1)


def interior_indices_1d(n):
    return np.arange(1, n - 1)


# ------------------------------- model --------------------------------------

def bc_poly_1d(x):
    return 4.0 * x[:, 0] * (1.0 - x[:, 0])


def init_mlp(key, sizes):
    params = []
    for i in range(len(sizes) - 1):
        key, k1 = jax.random.split(key)
        w = jax.random.normal(k1, (sizes[i], sizes[i + 1]), dtype=F64) \
            * jnp.sqrt(2.0 / sizes[i])
        params.append((w, jnp.zeros((sizes[i + 1],), dtype=F64)))
    return params


def apply_mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def init_separable_1d(key, k_lat, r_feat, n_ff=64, ff_scale=4.0,
                      g_hidden=128, g_layers=2, h_hidden=128, h_layers=2,
                      out_scale=1.0):
    kb, kg, kh, kl = jax.random.split(key, 4)
    B = jax.random.normal(kb, (1, n_ff), dtype=F64) * ff_scale
    g_mlp = init_mlp(kg, [2 * n_ff] + [g_hidden] * g_layers + [r_feat])
    h_mlp = init_mlp(kh, [k_lat] + [h_hidden] * h_layers + [r_feat])
    h_lin = jax.random.normal(kl, (k_lat, r_feat), dtype=F64) * 0.3
    return dict(B=B, g=g_mlp, h=h_mlp, h_lin=h_lin,
                out_scale=jnp.asarray(float(out_scale), dtype=F64))


def features(params, x):
    """bc(x) * g~(x): (n_pts, r).  ALL x-dependence of the decoder."""
    ang = 2.0 * jnp.pi * (x @ params["B"])
    ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return (params["out_scale"] * bc_poly_1d(x))[..., None] \
        * apply_mlp(params["g"], ff)


def head(params, z):
    return apply_mlp(params["h"], z) + z @ params["h_lin"]


# ------------------------------- family / FOM --------------------------------

def sample_params_1d(seed, m):
    """c ~ U(0.3,0.7), w ~ U(0.05,0.15), a ~ U(0.5,1.5), nu ~ logU(0.01,0.1)
    (the 1D analog of the 2D blob family; nu range identical to 2D)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.3, 0.7, m)
    w = rng.uniform(0.05, 0.15, m)
    a = rng.uniform(0.5, 1.5, m)
    nu = np.exp(rng.uniform(np.log(0.01), np.log(0.1), m))
    return c, w, a, nu


def blob_ic_1d(n, c, w, a):
    x = np.linspace(0.0, 1.0, n)
    u0 = a * np.exp(-((x - c) ** 2) / (2.0 * w * w))
    u0[0] = 0.0
    u0[-1] = 0.0
    return u0


def upwind_adv_field_1d(u_int, n):
    """N(u) = u * u_x with the FOM's sign-upwind stencil on the interior field
    u_int (n_i,), ghost zeros on the walls."""
    dx = 1.0 / (n - 1)
    U = jnp.pad(u_int, 1)
    c = U[1:-1]
    ux = jnp.where(c > 0, (c - U[:-2]) / dx, (U[2:] - c) / dx)
    return c * ux


def fom_residual_int(u_int, up_int, nu, n):
    """Backward-Euler interior residual: u - u_prev + dt*(N(u) - nu*lap(u))."""
    dx = 1.0 / (n - 1)
    U = jnp.pad(u_int, 1)
    lap = (U[2:] - 2.0 * U[1:-1] + U[:-2]) / dx ** 2
    return u_int - up_int + DT * (upwind_adv_field_1d(u_int, n) - nu * lap)


def make_rollout_1d(n):
    """Vmapped fixed-iteration Newton rollout (the truth generator).  The
    2D lesson set is applied: fixed NEWTON_ITERS with a skip-guard on
    already-converged residuals, non-finite steps rejected, NaN-PROPAGATING
    residual audit (max accumulate via jnp.maximum, never python max)."""
    interior = interior_indices_1d(n)

    def newton_step(u_int, up_int, nu):
        def body(u, _):
            r = fom_residual_int(u, up_int, nu, n)
            J = jax.jacfwd(lambda v: fom_residual_int(v, up_int, nu, n))(u)
            dz = jnp.linalg.solve(J, -r)
            ok = jnp.all(jnp.isfinite(dz)) & \
                (jnp.linalg.norm(r) > 1e-13 * (1.0 + jnp.linalg.norm(u)))
            return jnp.where(ok, u + dz, u), None
        u, _ = jax.lax.scan(body, u_int, None, length=NEWTON_ITERS)
        rfin = jnp.linalg.norm(fom_residual_int(u, up_int, nu, n)) \
            / (jnp.linalg.norm(up_int) + 1e-300)
        return u, rfin

    def rollout(U0, nu):
        """U0 (B, n) full-grid ICs, nu (B,).  Returns snaps (B, T+1, n) and
        the max relative Newton residual over all steps (NaN-propagating)."""
        u0_int = jnp.asarray(U0)[:, interior]

        def body(carry, _):
            u, worst = carry
            u2, r = jax.vmap(newton_step, in_axes=(0, 0, 0))(u, u, nu)
            return (u2, jnp.maximum(worst, jnp.max(r))), u2

        (uT, worst), traj = jax.lax.scan(
            body, (u0_int, jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        B = U0.shape[0]
        full = jnp.zeros((NUM_STEPS, B, n), dtype=F64)
        full = full.at[:, :, 1:-1].set(traj)
        snaps = jnp.concatenate([jnp.asarray(U0)[None], full], axis=0)
        return jnp.transpose(snaps, (1, 0, 2)), worst

    return jax.jit(rollout)


def tridiag_jac(u_int, nu, n):
    """Analytic tridiagonal Jacobian of fom_residual_int w.r.t. u_int
    (the sign-upwind `where` treated as locally constant, the standard
    subgradient convention — identical to what jacfwd produces).  Returns
    (dl, d, du) in the jax.lax.linalg.tridiagonal_solve convention
    (dl[0] and du[-1] unused/zero)."""
    dx = 1.0 / (n - 1)
    U = jnp.pad(u_int, 1)
    c = U[1:-1]
    s = (c > 0).astype(u_int.dtype)                           # upwind switch
    ux = jnp.where(c > 0, (c - U[:-2]) / dx, (U[2:] - c) / dx)
    dl = DT * (c * (-s / dx) - nu / dx ** 2)
    d = 1.0 + DT * (ux + c * (2.0 * s - 1.0) / dx + 2.0 * nu / dx ** 2)
    du = DT * (c * (1.0 - s) / dx - nu / dx ** 2)
    dl = dl.at[0].set(0.0)
    du = du.at[-1].set(0.0)
    return dl, d, du


def tri_solve(dl, d, du, b):
    from jax.lax.linalg import tridiagonal_solve
    return tridiagonal_solve(dl, d, du, b[:, None])[:, 0]


def make_rollout_1d_tri(n):
    """The truth generator with the O(n) tridiagonal Newton solve instead of
    the dense Jacobian — same residual, same fixed-iteration Newton with the
    same skip/finite guards, same NaN-propagating audit.  Used for data
    generation at large n (the dense path is kept for cross-gating)."""
    interior = interior_indices_1d(n)

    def newton_step(u_int, up_int, nu):
        def body(u, _):
            r = fom_residual_int(u, up_int, nu, n)
            dl, d, du = tridiag_jac(u, nu, n)
            dz = tri_solve(dl, d, du, -r)
            ok = jnp.all(jnp.isfinite(dz)) & \
                (jnp.linalg.norm(r) > 1e-13 * (1.0 + jnp.linalg.norm(u)))
            return jnp.where(ok, u + dz, u), None
        u, _ = jax.lax.scan(body, u_int, None, length=NEWTON_ITERS)
        rfin = jnp.linalg.norm(fom_residual_int(u, up_int, nu, n)) \
            / (jnp.linalg.norm(up_int) + 1e-300)
        return u, rfin

    def rollout(U0, nu):
        u0_int = jnp.asarray(U0)[:, interior]

        def body(carry, _):
            u, worst = carry
            u2, r = jax.vmap(newton_step, in_axes=(0, 0, 0))(u, u, nu)
            return (u2, jnp.maximum(worst, jnp.max(r))), u2

        (uT, worst), traj = jax.lax.scan(
            body, (u0_int, jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        B = U0.shape[0]
        full = jnp.zeros((NUM_STEPS, B, n), dtype=F64)
        full = full.at[:, :, 1:-1].set(traj)
        snaps = jnp.concatenate([jnp.asarray(U0)[None], full], axis=0)
        return jnp.transpose(snaps, (1, 0, 2)), worst

    return jax.jit(rollout)


def make_fom_tol_rollout(n, max_newton=8):
    """Timed FOM baseline: tolerance-terminated Newton (stop when the
    relative residual <= ntol), tridiagonal solve — the algorithm a real 1D
    production FOM would use, so its cost is not inflated by a dense
    factorization.  Whole 50-step rollout on device (lax.scan over steps,
    lax.while_loop Newton inside).  Single trajectory (batch=1 single-query
    cost).  Returns (traj (T, n_i), total Newton iters, worst rel residual)."""
    interior = interior_indices_1d(n)

    def step(u_int, nu, ntol):
        up = u_int
        upn = jnp.linalg.norm(up) + 1e-300

        def cond(s):
            u, it = s
            r = fom_residual_int(u, up, nu, n)
            return (jnp.linalg.norm(r) > ntol * upn) & (it < max_newton)

        def body(s):
            u, it = s
            r = fom_residual_int(u, up, nu, n)
            dl, d, du = tridiag_jac(u, nu, n)
            dz = tri_solve(dl, d, du, -r)
            u2 = jnp.where(jnp.all(jnp.isfinite(dz)), u + dz, u)
            return (u2, it + 1)

        u, it = jax.lax.while_loop(cond, body, (u_int, jnp.int32(0)))
        rrel = jnp.linalg.norm(fom_residual_int(u, up, nu, n)) / upn
        return u, it, rrel

    def rollout(u0_full, nu, ntol):
        def body(carry, _):
            u, tot, worst = carry
            u2, it, rrel = step(u, nu, ntol)
            return (u2, tot + it, jnp.maximum(worst, rrel)), u2
        (uT, tot, worst), traj = jax.lax.scan(
            body, (jnp.asarray(u0_full)[interior], jnp.int32(0),
                   jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        return traj, tot, worst

    return jax.jit(rollout)


def build_data_1d(n, n_train, n_test, seed, test_seed, chunk=128,
                  solver="dense"):
    """Regenerate train + fresh-test trajectories from seeds.  Aborts if any
    trajectory's FOM residual audit exceeds 1e-8 (unconverged Newton must
    never become 'truth').  solver='dense' (the screening runs' generator)
    or 'tri' (tridiagonal Newton, O(n), for large n; cross-gated in the
    scale driver)."""
    rollout = make_rollout_1d(n) if solver == "dense" \
        else make_rollout_1d_tri(n)
    out = {}
    for name, sd, m in (("train", seed, n_train), ("test", test_seed, n_test)):
        c, w, a, nu = sample_params_1d(sd, m)
        U0 = np.stack([blob_ic_1d(n, c[i], w[i], a[i]) for i in range(m)])
        snaps, worsts = [], []
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            sn, wr = rollout(jnp.asarray(U0[s:e]), jnp.asarray(nu[s:e]))
            snaps.append(np.asarray(sn))
            worsts.append(float(wr))
        U = np.concatenate(snaps, axis=0)
        worst = float(np.max(worsts))
        if not np.isfinite(worst) or worst > 1e-8:
            raise SystemExit(f"{name} FOM residual {worst:.2e} > 1e-8: "
                             "data not converged")
        out[name] = dict(U=U, nu=nu, c=c, w=w, a=a, worst_res=worst)
        log(f"  data[{name}] N={n} m={m}: max FOM rel residual {worst:.2e}")
    return out


# ------------------------------- weak form -----------------------------------

def test_modes_1d(n, M):
    """M lowest discrete sine modes on the interior grid.  Exact eigenvectors
    of the 1D ghost-zero Laplacian.  Returns kx (M,), Phi (n_i, M) unit-2-norm
    columns, lam_disc (M,)."""
    dx = 1.0 / (n - 1)
    kk = np.arange(1, n - 1)
    lam = (4.0 / dx ** 2) * np.sin(np.pi * kk / (2 * (n - 1))) ** 2
    order = np.argsort(lam, kind="stable")[:M]
    kx = kk[order]
    xi = kk / (n - 1)
    Phi = np.sin(np.pi * np.outer(xi, kx))                    # (n_i, M)
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
    return kx, Phi, lam[order]


def modes_at_1d(x, kx, n):
    """Continuous sine modes at positions x (m,) with the grid columns'
    normalization (||sin(pi k x_i)||_2^2 over the interior = (n-1)/2)."""
    nrm = np.sqrt((n - 1) / 2.0)
    return jnp.sin(jnp.pi * kx[None, :] * x[:, None]) / nrm   # (m, M)


# ------------------------------- NNLS ----------------------------------------

def nnls_capped(G, b, max_support, tol=1e-10, inner_max=200):
    """Lawson-Hanson active-set NNLS min_{w>=0} ||G w - b|| that STOPS when the
    support reaches max_support (ECSW-style) or at optimality.  Copied
    verbatim from blat_common.nnls_capped."""
    n = G.shape[1]
    w = np.zeros(n)
    P = np.zeros(n, bool)
    r = b - G @ w
    outer = 0
    while outer < 5 * max_support + 10:
        grad = G.T @ r
        cand = np.where(~P)[0]
        if cand.size == 0 or P.sum() >= max_support:
            break
        j = cand[np.argmax(grad[cand])]
        if grad[j] <= tol * (np.linalg.norm(b) + 1e-300):
            break
        P[j] = True
        outer += 1
        for _ in range(inner_max):
            idx = np.where(P)[0]
            s_, *_ = np.linalg.lstsq(G[:, idx], b, rcond=None)
            if np.all(s_ > 0):
                w[:] = 0.0
                w[idx] = s_
                break
            neg = s_ <= 0
            alpha = np.min(w[idx][neg] / (w[idx][neg] - s_[neg] + 1e-300))
            w[idx] = w[idx] + alpha * (s_ - w[idx])
            P[idx[w[idx] <= 1e-14]] = False
            w[~P] = 0.0
        r = b - G @ w
    return w, float(np.linalg.norm(r)), outer


def eq_fit_adv_1d(u_full, Phi, cand_pos, Z_snap, m, n, label):
    """Advection-only NNLS node fit, the exlin_common.eq_fit_burgers_adv
    recipe in 1D: rows Phi_c^T * N(u)|cand per snapshot, targets the exact
    full-grid projections Phi^T N(u); row-normalized; capped Lawson-Hanson;
    support padded by mean |N| if short; nonnegative refit on the kept
    columns.  Returns keep (positions into cand_pos), weights, info."""
    t0 = time.time()
    Phi_c = Phi[cand_pos]
    Gs, bs, snap_c = [], [], []
    for z in Z_snap:
        uf = np.asarray(u_full(jnp.asarray(z)))
        Nf = np.asarray(upwind_adv_field_1d(jnp.asarray(uf), n))
        bs.append(Phi.T @ Nf)
        Gs.append(Phi_c.T * Nf[cand_pos][None, :])
        snap_c.append(Nf[cand_pos])
    pad_score = np.abs(np.stack(snap_c)).mean(0)
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    sc = np.linalg.norm(G, axis=1) + 1e-300
    Gn, bn = G / sc[:, None], b / sc
    wts, _, _ = nnls_capped(Gn, bn, max_support=m)
    supp = np.nonzero(wts > 0)[0]
    padded = 0
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]
    else:
        rest = np.setdiff1d(np.arange(G.shape[1]), supp)
        pad = rest[np.argsort(-pad_score[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad])
        padded = len(pad)
    keep = np.sort(keep)
    wq, _, _ = nnls_capped(Gn[:, keep], bn, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    resid = Gn[:, keep] @ wq - bn
    rel_rows = np.abs(resid) / (np.abs(bn) + 1e-300)
    info = dict(support=int(len(supp)), padded=int(padded),
                rel_fit=float(np.linalg.norm(resid) / np.linalg.norm(bn)),
                row_rel_median=float(np.median(rel_rows)),
                row_rel_p95=float(np.quantile(rel_rows, 0.95)),
                row_rel_max=float(np.max(rel_rows)),
                n_rows=int(G.shape[0]), m=int(len(keep)),
                secs=time.time() - t0)
    log(f"  NNLS-EQ {label}: support {len(supp)} (+{padded} pad), rel fit "
        f"{info['rel_fit']:.2e} (row p95 {info['row_rel_p95']:.1e}, max "
        f"{info['row_rel_max']:.1e}) [{info['secs']:.0f}s]")
    return keep, wq, info


# ------------------------------- training ------------------------------------

def train_autodecoder_1d(key, coords, U, k_lat, r_feat, steps=40000, lr=1e-3,
                         lam_orth=1e-4, log_every=5000, tag="", **arch):
    """Joint Adam over (g, h, per-snapshot codes Z), the sep_common recipe in
    1D.  U (S, n_pts) f64 snapshot values at coords (n_pts, 1)."""
    coords = jnp.asarray(coords, dtype=F64)
    U = jnp.asarray(U, dtype=F64)
    S = U.shape[0]
    key, kz, kp = jax.random.split(key, 3)
    u_rms = float(jnp.sqrt(jnp.mean(U * U)))
    params = init_separable_1d(kp, k_lat, r_feat, out_scale=u_rms, **arch)
    Z = 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64)
    u_ms = jnp.mean(U * U)

    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    state = opt.init((params, Z))

    def loss_fn(pz, Uc):
        p, z = pz
        G = features(p, coords)
        H = head(p, z)
        err = H @ G.T - Uc
        rel = jnp.mean(err * err) / u_ms
        C = (G.T @ G) / (G.shape[0] * p["out_scale"] ** 2)
        orth = jnp.mean((C - jnp.eye(C.shape[0], dtype=F64)) ** 2)
        return rel + lam_orth * orth, rel

    @jax.jit
    def step(pz, st, Uc):
        (val, rel_), grads = jax.value_and_grad(loss_fn, has_aux=True)(pz, Uc)
        grads[0]["out_scale"] = jnp.zeros_like(grads[0]["out_scale"])
        upd, st = opt.update(grads, st)
        return optax.apply_updates(pz, upd), st, val, rel_

    pz = (params, Z)
    t0 = time.time()
    rel_v = jnp.inf
    for i in range(steps):
        pz, state, val, rel_v = step(pz, state, U)
        if (i + 1) % log_every == 0 or i == 0:
            log(f"   train[{tag}] step {i+1:6d}/{steps}  rel-MSE "
                f"{float(rel_v):.3e}  [{time.time()-t0:.0f}s]")
    params, Z = pz
    G = features(params, coords)
    Uh = head(params, Z) @ G.T
    per = jnp.linalg.norm(Uh - U, axis=1) / jnp.linalg.norm(U, axis=1)
    info = dict(final_rel_mse=float(rel_v), steps=steps, lr=lr,
                lam_orth=lam_orth, seconds=time.time() - t0,
                recon_rel_l2_mean=float(jnp.mean(per)),
                recon_rel_l2_max=float(jnp.max(per)),
                n_snapshots=int(S), n_points=int(coords.shape[0]))
    log(f"   train[{tag}] done: recon rel-L2 mean "
        f"{info['recon_rel_l2_mean']:.3e} max {info['recon_rel_l2_max']:.3e} "
        f"[{info['seconds']:.0f}s]")
    return params, np.asarray(Z), info


# ------------------------------- io ------------------------------------------

def save_pkl(path, params, Z_tr, cfg):
    host = jax.tree_util.tree_map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(dict(params=host, Z_tr=np.asarray(Z_tr), cfg=cfg), f)


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    return params, d["Z_tr"], d["cfg"]
