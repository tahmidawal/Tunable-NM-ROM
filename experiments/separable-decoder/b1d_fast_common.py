"""Shared machinery for the sep_b1d_fast.py speed study (2026-08-28).

Contains, side by side:
  * loaders that rebuild the EXACT arm state of a committed sep_b1d_scale.py
    run from its artifacts (checkpoint .pkl + _nodes.npz) plus a test-only
    data regeneration (same seeds, same tridiagonal truth generator) — no
    retraining, no NNLS refit, no node training;
  * REFERENCE implementations of the on-device LM rollout and the Gram-space
    IC fit, copied VERBATIM (modulo closure plumbing) from sep_b1d_scale.py
    make_device / ic_fit_dev — these define baseline numbers and parity;
  * OPTIMIZED implementations behind explicit opt flags.  The winning
    composition (all defaults in sep_b1d_fast.py):
      solver='gj'    broadcast Gauss-Jordan for the (K,K) normal equations
                     (no cuSOLVER custom call; ~28 vs ~120 µs/LM-iteration)
      onepass        one-pass residual+Jacobian via jax.linearize
      nocond         no lax.cond re-evaluation (unconditional r+J, selected)
      hoist/lean     per-rollout folding of every nu-constant into
                     premultiplied matrices; head last-layer merged with
                     h_lin; A-projection and node features in ONE matmul;
                     H,g as one (K,K+1) dot; Lt folded into the IC head
      nodot          matvecs as broadcast-reduce (no cublas custom calls)
      scan_unroll=5  unroll the fixed 50-step outer scan (bitwise identical)
    Measured/rejected: scalar-unrolled Cholesky (slow serial fusions), XLA
    command buffers (no effect / slower), masked LM-loop unroll (pays full
    body for masked iterations), reduced-init IC (breaks parity — the
    9-init multistart is load-bearing, see OPTIM-NOTES.md).

Nothing here changes the algorithm: budgets, tolerances, trust region, stall
rule, accept/reject logic, and init sets are identical between ref and fast.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

import b1d_common as b1

import jax
import jax.numpy as jnp

F64 = jnp.float64

# ---- config constants (defaults identical to sep_b1d_scale.py) --------------
K = int(os.environ.get("K", "8"))
R = int(os.environ.get("R", "32"))
M = int(os.environ.get("M", "32"))
SEED0 = int(os.environ.get("SEED0", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", "1"))
N_TEST = int(os.environ.get("N_TEST", "8"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
STALL = float(os.environ.get("STALL", "1e-3"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
GN_BUDGET = int(os.environ.get("GN_BUDGET", "30"))
IC_BUDGET = int(os.environ.get("IC_BUDGET", "200"))


def gen_test(N):
    """Test trajectories exactly as build_data_1d's test branch (tri solver)."""
    c, w, a, nu = b1.sample_params_1d(TEST_SEED, N_TEST)
    U0 = np.stack([b1.blob_ic_1d(N, c[i], w[i], a[i]) for i in range(N_TEST)])
    rollout = b1.make_rollout_1d_tri(N)
    sn, wr = rollout(jnp.asarray(U0), jnp.asarray(nu))
    worst = float(wr)
    assert np.isfinite(worst) and worst <= 1e-8, \
        f"test FOM residual {worst:.2e} > 1e-8"
    return np.asarray(sn), nu


class Setup:
    """Everything sep_b1d_scale.py derives from (params, Z_tr, N) that the
    rollout/IC machinery closes over — reproduced verbatim."""

    def __init__(self, ckpt_path, N):
        params, Z_tr, ck_cfg = b1.load_pkl(ckpt_path)
        assert int(ck_cfg["N"]) == N and int(ck_cfg["k"]) == K
        self.params, self.Z_tr, self.N = params, Z_tr, N
        self.dx = 1.0 / (N - 1)
        self.interior = b1.interior_indices_1d(N)
        self.n_i = self.interior.size
        self.coords_int = b1.grid_coords_1d(N)[self.interior]

        self.h_fn = jax.jit(lambda z: b1.head(params, z))
        self.G_int = jnp.asarray(b1.features(params,
                                             jnp.asarray(self.coords_int)))
        kx, Phi_np, lam_np = b1.test_modes_1d(N, M)
        self.Phi_j = jnp.asarray(Phi_np)
        self.lam_j = jnp.asarray(lam_np, dtype=F64)
        self.kx_f = jnp.asarray(kx, dtype=F64)
        self.A_j = self.Phi_j.T @ self.G_int
        train_radius = float(np.max(np.linalg.norm(
            Z_tr - Z_tr.mean(0), axis=1)))
        self.TR_DELTA = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
        self.zbar = Z_tr.mean(0)
        self.OFF = jnp.asarray(np.array([0.0, self.dx, -self.dx]))

        # Gram-space IC fit constants (verbatim)
        Gram = np.asarray(self.G_int).T @ np.asarray(self.G_int)
        Lch = np.linalg.cholesky(Gram + 1e-30 * np.eye(R))
        self.Lt_j = jnp.asarray(Lch.T)
        self.Gram_j = jnp.asarray(Gram)
        orng = np.random.default_rng(SEED0 + 11)
        self.Z0S = jnp.asarray(np.stack(
            [self.zbar] + [Z_tr[orng.integers(len(Z_tr))] for _ in range(8)]))

        self.prev_of = jax.jit(lambda z: self.A_j @ self.h_fn(z))
        self.decode_all = jax.jit(
            lambda Z: b1.head(params, Z) @ self.G_int.T)

    # ---- continuous-node machinery (verbatim from sep_b1d_scale.py) ---------
    def phi_at(self, X):
        nrm = np.sqrt((self.N - 1) / 2.0)
        return jnp.sin(jnp.pi * self.kx_f[None, :] * X[:, None]) / nrm

    def g3_of(self, X):
        Xs = (X[:, None] + self.OFF[None, :]).reshape(-1, 1)
        return b1.features(self.params, Xs).reshape(-1, 3, R)

    def adv_nodes(self, G3, Hb):
        U3 = jnp.einsum("mfr,sr->smf", G3, Hb)
        c, xp, xm = U3[..., 0], U3[..., 1], U3[..., 2]
        ux = jnp.where(c > 0.0, (c - xm) / self.dx, (xp - c) / self.dx)
        return c * ux

    def make_sampled_rw(self, X_v, w_v):
        """Verbatim make_sampled_rw."""
        G3 = self.g3_of(jnp.asarray(X_v))
        Phi_q = self.phi_at(jnp.asarray(X_v)) * jnp.asarray(w_v)[:, None]
        A_j, lam_j, h_fn = self.A_j, self.lam_j, self.h_fn
        adv_nodes = self.adv_nodes

        def r_w(z, prev_m, nu):
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            hz = h_fn(z)
            Ah = A_j @ hz
            Nu = adv_nodes(G3, hz[None])[0]
            lin = (Ah - prev_m) + b1.DT * nu * lam_j * Ah
            return wt * (lin + b1.DT * (Phi_q.T @ Nu))
        return r_w

    def make_full_rw(self):
        """Verbatim full_r_w (oracle arm)."""
        A_j, lam_j, h_fn = self.A_j, self.lam_j, self.h_fn
        G_int, Phi_j, N = self.G_int, self.Phi_j, self.N

        def r_w(z, prev_m, nu):
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            hz = h_fn(z)
            Ah = A_j @ hz
            Nu = b1.upwind_adv_field_1d(G_int @ hz, N)
            lin = (Ah - prev_m) + b1.DT * nu * lam_j * Ah
            return wt * (lin + b1.DT * (Phi_j.T @ Nu))
        return r_w

    def make_tensor_rw(self, Q):
        """Tensor arm (2026-08-29): the oracle residual with the full-grid
        advection projection Phi^T upwind(G h) replaced by the precomputed
        quadratic 0.5 * h^T Q h (b1d_tensor_common).  Everything else — exact
        linear terms, weights, dt, nu — is verbatim the oracle's."""
        A_j, lam_j, h_fn = self.A_j, self.lam_j, self.h_fn
        Qj = jnp.asarray(Q, dtype=F64)

        def r_w(z, prev_m, nu):
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            hz = h_fn(z)
            Ah = A_j @ hz
            Pq = 0.5 * ((Qj @ hz) @ hz)
            lin = (Ah - prev_m) + b1.DT * nu * lam_j * Ah
            return wt * (lin + b1.DT * Pq)
        return r_w


def load_arms(nodes_npz):
    d = np.load(nodes_npz)
    return {
        "base_tight": (d["X0_tight"], d["w0_tight"]),
        "nodes_tight": (d["X_tight"], d["w_tight"]),
        "base_half": (d["X0_half"], d["w0_half"]),
        "nodes_half": (d["X_half"], d["w_half"]),
    }


# =============================================================================
# REFERENCE implementations — verbatim copies of sep_b1d_scale.py make_device
# and ic_fit_dev (only closure plumbing differs).
# =============================================================================

def make_device_ref(su, r_w):
    TR_DELTA = su.TR_DELTA
    prev_of = su.prev_of

    rn_fn = lambda z, p, nu: jnp.linalg.norm(r_w(z, p, nu))
    rJ_f = lambda z, p, nu: (r_w(z, p, nu), jax.jacfwd(r_w)(z, p, nu))

    def lm_step(z0, prev_c, nu, tol_abs, budget):
        r0, J0 = rJ_f(z0, prev_c, nu)
        rn0 = jnp.linalg.norm(r0)
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where(rn0 <= tol_abs, jnp.int32(4),
                                          jnp.int32(0)))
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(1), init_reason)

        def cond(s):
            return (s[7] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, nJ, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            within = jnp.linalg.norm(dz) <= TR_DELTA
            tiny = finite & (jnp.linalg.norm(dz)
                             <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
            z_new = z + jnp.where(finite & within, dz, 0.0)
            rn_new = rn_fn(z_new, prev_c, nu)
            accept = finite & within & jnp.isfinite(rn_new) & \
                (rn_new < rn) & ~tiny
            r2, J2 = jax.lax.cond(
                accept, lambda: rJ_f(z_new, prev_c, nu), lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(
                accept & (rn <= tol_abs), 1,
                jnp.where((accept & (rel_dec < STALL)) | tiny, 2,
                          jnp.where((~accept) & (lam >= 1e12), 3,
                                    0))).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, nJ, reason)

        z, r, J, rn, lam, att, nJ, reason = jax.lax.while_loop(
            cond, body, init)
        return z, rn, nJ, reason, att

    def rollout_fn(z0, nu, tol_abs, budget):
        def body(carry, _):
            z_km1, z_k = carry
            z_init = z_k + EXTRAP * (z_k - z_km1)
            prev_c = prev_of(z_k)
            z2, rn, nJ, reason, att = lm_step(z_init, prev_c, nu,
                                              tol_abs, budget)
            return (z_k, z2), (z2, rn, nJ, reason)
        (_, zT), (Z, rns, nJs, reasons) = jax.lax.scan(
            body, (z0, z0), None, length=b1.NUM_STEPS)
        return Z, rns, nJs, reasons

    return dict(rollout=jax.jit(rollout_fn, static_argnums=(3,)),
                rJ=jax.jit(rJ_f), rn=jax.jit(rn_fn))


def make_ic_ref(su):
    Gram_j, G_int, Lt_j, Z0S, params = \
        su.Gram_j, su.G_int, su.Lt_j, su.Z0S, su.params

    def cstar_of(u0_int):
        return jnp.linalg.solve(Gram_j, G_int.T @ u0_int)

    def ic_lm_one(z0, cstar):
        def r_of(z):
            return Lt_j @ (b1.head(params, z) - cstar)
        rJ_f = lambda z: (r_of(z), jax.jacfwd(r_of)(z))
        r0, J0 = rJ_f(z0)
        init = (z0, r0, J0, jnp.linalg.norm(r0), jnp.asarray(1e-6, F64),
                jnp.int32(0), jnp.int32(0))

        def cond(s):
            return (s[6] == 0) & (s[5] < IC_BUDGET)

        def body(s):
            z, r, J, rn, lam, att, done = s
            H = J.T @ J
            g = J.T @ r
            dz = jnp.linalg.solve(
                H + lam * (jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K)), -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            rn_new = jnp.linalg.norm(r_of(z_new))
            accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
            r2, J2 = jax.lax.cond(accept, lambda: rJ_f(z_new),
                                  lambda: (r, J))
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            done = jnp.where((~accept) & (lam >= 1e12), 1, 0).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, done)

        z, r, J, rn, lam, att, done = jax.lax.while_loop(cond, body, init)
        return z, rn

    @jax.jit
    def ic_fit_dev(u0_int):
        cstar = cstar_of(u0_int)
        zs, rns = jax.vmap(ic_lm_one, in_axes=(0, None))(Z0S, cstar)
        i = jnp.argmin(rns)
        return zs[i], rns[i]

    return ic_fit_dev


# =============================================================================
# OPTIMIZED implementations
# =============================================================================

def solve_spd8(Amat, bvec):
    """Unrolled Cholesky solve for a KxK SPD system, pure jnp scalar ops
    (fuses into a handful of kernels; no cuSOLVER custom call).  NaN on a
    non-SPD matrix, which the callers' `finite` guard rejects exactly like a
    failed LU solve."""
    n = Amat.shape[0]
    a = [[Amat[i, j] for j in range(n)] for i in range(n)]
    bl = [bvec[i] for i in range(n)]
    l = [[None] * n for _ in range(n)]
    for j in range(n):
        s = a[j][j]
        for p in range(j):
            s = s - l[j][p] * l[j][p]
        d = jnp.sqrt(s)
        l[j][j] = d
        inv = 1.0 / d
        for i in range(j + 1, n):
            s2 = a[i][j]
            for p in range(j):
                s2 = s2 - l[i][p] * l[j][p]
            l[i][j] = s2 * inv
    y = [None] * n
    for i in range(n):
        s = bl[i]
        for p in range(i):
            s = s - l[i][p] * y[p]
        y[i] = s / l[i][i]
    x = [None] * n
    for i in reversed(range(n)):
        s = y[i]
        for p in range(i + 1, n):
            s = s - l[p][i] * x[p]
        x[i] = s / l[i][i]
    return jnp.stack(x)


def gj_solve(A, b):
    """Gauss-Jordan solve, no pivoting (safe for the SPD LM normal equations),
    broadcast elementwise ops ONLY — no dots, no custom calls, fuses into a
    handful of kernels.  Measured 28 µs/iteration in-loop on GB10 vs 120 µs
    for the cuSOLVER LU behind jnp.linalg.solve.  Accuracy vs LU ~4e-16.
    A singular/indefinite system yields inf/nan, which the callers' `finite`
    guard rejects exactly as it does for a failed LU."""
    n = A.shape[0]
    M = jnp.concatenate([A, b[:, None]], axis=1)
    rows = jnp.arange(n)
    for k_ in range(n):
        piv = M[k_, k_]
        row = M[k_] / piv
        fac = jnp.where(rows == k_, 0.0, M[:, k_])
        M = M - fac[:, None] * row[None, :]
        M = jnp.where((rows == k_)[:, None], row[None, :], M)
    return M[:, n]


def pick_solver(opt):
    s = opt.get("solver", "gj")
    if s == "gj":
        return gj_solve
    if s == "spd8":
        return solve_spd8
    return lambda A, b: jnp.linalg.solve(A, b)


def make_device_fast(su, X_v, w_v, opt, Q=None):
    """Optimized rollout.  opt keys: solver ('gj'|'spd8'|'lu'), onepass,
    hoist, nocond, lean, unroll.  X_v None => oracle (full weak residual;
    the lean restructure applies only to sampled arms).  Q not None (with
    X_v None) => TENSOR arm: advection term 0.5 h^T Q h (2026-08-29); the
    lean restructure applies to it too (A-projection and h from one stacked
    matmul, then the (M,R,R) contraction)."""
    TR_DELTA = su.TR_DELTA
    A_j, lam_j, params = su.A_j, su.lam_j, su.params
    solver_fn = pick_solver(opt)
    use_onepass = opt.get("onepass", True)
    use_hoist = opt.get("hoist", True)
    use_nocond = opt.get("nocond", True)
    tensor = Q is not None
    assert not (tensor and X_v is not None)
    use_lean = opt.get("lean", False) and (X_v is not None or tensor)
    use_nodot = opt.get("nodot", False)
    unroll = int(opt.get("unroll", 1))
    scan_unroll = int(opt.get("scan_unroll", 1))
    with_att = bool(opt.get("with_att", False))   # also return LM attempts
    Qj = jnp.asarray(Q, dtype=F64) if tensor else None

    if use_lean and tensor:
        (W1, bh1), (W2, bh2), (W3, bh3) = params["h"]
        W3aug = jnp.concatenate([W3, params["h_lin"]], axis=0)  # (h+K, R)
        eyeR = jnp.eye(R, dtype=F64)
    elif use_lean:
        # static pieces of the folded residual
        (W1, bh1), (W2, bh2), (W3, bh3) = params["h"]
        W3aug = jnp.concatenate([W3, params["h_lin"]], axis=0)  # (h+K, R)
        G3l = su.g3_of(jnp.asarray(X_v))
        m_n = G3l.shape[0]
        G3f = G3l.reshape(m_n * 3, R)                           # (3m, R)
        PhiqTl = (su.phi_at(jnp.asarray(X_v))
                  * jnp.asarray(w_v)[:, None]).T                # (M, m)

    if X_v is None:
        G3 = None
        PhiqT = None
        G_int, Phi_j, Ngrid = su.G_int, su.Phi_j, su.N
    else:
        G3 = su.g3_of(jnp.asarray(X_v))
        PhiqT = (su.phi_at(jnp.asarray(X_v))
                 * jnp.asarray(w_v)[:, None]).T          # (M, m)
    dx = su.dx
    eyeK = jnp.eye(K, dtype=F64)

    def make_rw(nu):
        """Returns r_w(z, prev_m) with the nu-dependent constants hoisted
        (bitwise-identical values: wt and DT*nu*lam_j computed exactly as in
        the reference expression)."""
        wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
        cvec = b1.DT * nu * lam_j

        def r_w(z, prev_m):
            hz = b1.head(params, z)
            Ah = A_j @ hz
            if tensor:
                Pq = 0.5 * ((Qj @ hz) @ hz)
            elif X_v is None:
                Nu = b1.upwind_adv_field_1d(G_int @ hz, Ngrid)
                Pq = Phi_j.T @ Nu
            else:
                U3 = jnp.einsum("mfr,r->mf", G3, hz)
                c, xp, xm = U3[..., 0], U3[..., 1], U3[..., 2]
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                Pq = PhiqT @ (c * ux)
            lin = (Ah - prev_m) + cvec * Ah
            return wt * (lin + b1.DT * Pq)
        return r_w

    def make_rw_ref_style(nu_unused):
        # non-hoisted fallback (nu recomputed in-call), for A/B of hoisting
        def outer(nu):
            def r_w(z, prev_m):
                wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
                hz = b1.head(params, z)
                Ah = A_j @ hz
                if tensor:
                    Pq = 0.5 * ((Qj @ hz) @ hz)
                elif X_v is None:
                    Nu = b1.upwind_adv_field_1d(G_int @ hz, Ngrid)
                    Pq = Phi_j.T @ Nu
                else:
                    U3 = jnp.einsum("mfr,r->mf", G3, hz)
                    c, xp, xm = U3[..., 0], U3[..., 1], U3[..., 2]
                    ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                    Pq = PhiqT @ (c * ux)
                lin = (Ah - prev_m) + b1.DT * nu * lam_j * Ah
                return wt * (lin + b1.DT * Pq)
            return r_w
        return outer

    def rollout_fn(z0, nu, tol_abs, budget):
        if use_lean and tensor:
            # Tensor arm, lean: r = wt*(1+cvec)*(A h) - wt*prev_m
            #                       + wt*DT*0.5*h^T Q h
            # A-projection and h itself from ONE stacked matmul over the
            # merged head last layer, then the (M,R,R) contraction.
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            cvec = b1.DT * nu * lam_j
            scale = wt * (1.0 + cvec)
            A2 = scale[:, None] * A_j                       # (M, R)
            Mstack = jnp.concatenate([A2, eyeR], axis=0)    # (M+R, R)
            Mfold = Mstack @ W3aug.T                        # (M+R, h+K)
            yb = Mstack @ bh3
            P2 = wt * b1.DT                                 # (M,)
            Aw = wt[:, None] * A_j
            Afold = Aw @ W3aug.T
            ab = Aw @ bh3
            Mloc = A_j.shape[0]

            if use_nodot:
                def cc_of(z):
                    a1 = jax.nn.silu(jnp.sum(z[:, None] * W1, axis=0) + bh1)
                    a2 = jax.nn.silu(jnp.sum(a1[:, None] * W2, axis=0) + bh2)
                    return jnp.concatenate([a2, z])

                def r_w(z, pm2):
                    cc = cc_of(z)
                    y = jnp.sum(Mfold * cc[None, :], axis=1) + yb
                    Ah2 = y[:Mloc]
                    hz = y[Mloc:]
                    v = jnp.sum(Qj * hz[None, None, :], axis=2)   # (M, R)
                    q = 0.5 * jnp.sum(v * hz[None, :], axis=1)    # (M,)
                    return Ah2 - pm2 + P2 * q

                def prev_fn(z):
                    cc = cc_of(z)
                    return jnp.sum(Afold * cc[None, :], axis=1) + ab
            else:
                def cc_of(z):
                    a1 = jax.nn.silu(z @ W1 + bh1)
                    a2 = jax.nn.silu(a1 @ W2 + bh2)
                    return jnp.concatenate([a2, z])

                def r_w(z, pm2):
                    y = Mfold @ cc_of(z) + yb
                    Ah2 = y[:Mloc]
                    hz = y[Mloc:]
                    q = 0.5 * ((Qj @ hz) @ hz)
                    return Ah2 - pm2 + P2 * q

                def prev_fn(z):
                    return Afold @ cc_of(z) + ab
        elif use_lean:
            # Fold every per-trajectory constant once per rollout call.
            # Mathematically identical residual (associativity-only changes):
            #   r = wt*(1+cvec)*(A h) - wt*prev_m + wt*DT*(Phi_q^T Nu)
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            cvec = b1.DT * nu * lam_j
            scale = wt * (1.0 + cvec)
            A2 = scale[:, None] * A_j                       # (M, R)
            P2 = (wt * b1.DT)[:, None] * PhiqTl             # (M, m)
            Mstack = jnp.concatenate([A2, G3f], axis=0)     # (M+3m, R)
            Mfold = Mstack @ W3aug.T                        # (M+3m, h+K)
            yb = Mstack @ bh3
            Aw = wt[:, None] * A_j
            Afold = Aw @ W3aug.T                            # (M, h+K)
            ab = Aw @ bh3
            Mloc = A_j.shape[0]

            if use_nodot:
                # broadcast-reduce matvecs: no cublas custom calls, fuse as
                # reduce fusions (helps the batched one-pass tangent path)
                def cc_of(z):
                    a1 = jax.nn.silu(jnp.sum(z[:, None] * W1, axis=0) + bh1)
                    a2 = jax.nn.silu(jnp.sum(a1[:, None] * W2, axis=0) + bh2)
                    return jnp.concatenate([a2, z])

                def r_w(z, pm2):
                    cc = cc_of(z)
                    y = jnp.sum(Mfold * cc[None, :], axis=1) + yb
                    Ah2 = y[:Mloc]
                    U3 = y[Mloc:].reshape(m_n, 3)
                    c, xp, xm = U3[:, 0], U3[:, 1], U3[:, 2]
                    ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                    v = c * ux
                    return Ah2 - pm2 + jnp.sum(P2 * v[None, :], axis=1)

                def prev_fn(z):
                    cc = cc_of(z)
                    return jnp.sum(Afold * cc[None, :], axis=1) + ab
            else:
                def cc_of(z):
                    a1 = jax.nn.silu(z @ W1 + bh1)
                    a2 = jax.nn.silu(a1 @ W2 + bh2)
                    return jnp.concatenate([a2, z])

                def r_w(z, pm2):
                    y = Mfold @ cc_of(z) + yb
                    Ah2 = y[:Mloc]
                    U3 = y[Mloc:].reshape(m_n, 3)
                    c, xp, xm = U3[:, 0], U3[:, 1], U3[:, 2]
                    ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                    return Ah2 - pm2 + P2 @ (c * ux)

                def prev_fn(z):
                    return Afold @ cc_of(z) + ab            # = wt * A h(z)
        else:
            r_w = make_rw(nu) if use_hoist else make_rw_ref_style(None)(nu)

            def prev_fn(z):
                return A_j @ b1.head(params, z)

        if use_onepass:
            def rJT(z, prev_m):
                r, lin = jax.linearize(lambda zz: r_w(zz, prev_m), z)
                JT = jax.vmap(lin)(eyeK)                  # (K, M) = J^T
                return r, JT
        else:
            def rJT(z, prev_m):
                r = r_w(z, prev_m)
                J = jax.jacfwd(r_w)(z, prev_m)
                return r, J.T

        solve = solver_fn

        def lm_step(z0_, prev_c):
            r0, JT0 = rJT(z0_, prev_c)
            rn0 = jnp.linalg.norm(r0)
            init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                    jnp.where(rn0 <= tol_abs, jnp.int32(4),
                                              jnp.int32(0)))
            init = (z0_, r0, JT0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0),
                    jnp.int32(1), init_reason)

            def cond(s):
                return (s[7] == 0) & (s[5] < budget)

            def body(s):
                z, r, JT, rn, lam, att, nJ, _ = s
                if use_lean:
                    Hg = JT @ jnp.concatenate([JT.T, r[:, None]], axis=1)
                    H = Hg[:, :K]
                    g = Hg[:, K]
                else:
                    H = JT @ JT.T
                    g = JT @ r
                dH = jnp.diag(H)
                Amat = H + lam * (jnp.diag(dH) + 1e-30 * eyeK)
                dz = solve(Amat, -g)
                finite = jnp.all(jnp.isfinite(dz))
                ndz = jnp.linalg.norm(dz)
                within = ndz <= TR_DELTA
                tiny = finite & (ndz <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
                z_new = z + jnp.where(finite & within, dz, 0.0)
                if use_nocond:
                    r_new, JT_new = rJT(z_new, prev_c)
                    rn_new = jnp.linalg.norm(r_new)
                    accept = finite & within & jnp.isfinite(rn_new) & \
                        (rn_new < rn) & ~tiny
                    r2 = jnp.where(accept, r_new, r)
                    JT2 = jnp.where(accept, JT_new, JT)
                else:
                    rn_new = jnp.linalg.norm(r_w(z_new, prev_c))
                    accept = finite & within & jnp.isfinite(rn_new) & \
                        (rn_new < rn) & ~tiny
                    r2, JT2 = jax.lax.cond(
                        accept, lambda: rJT(z_new, prev_c),
                        lambda: (r, JT))
                rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
                z = jnp.where(accept, z_new, z)
                rn = jnp.where(accept, rn_new, rn)
                lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                                jnp.minimum(lam * 10.0, 1e12))
                nJ = nJ + accept.astype(jnp.int32)
                reason = jnp.where(
                    accept & (rn <= tol_abs), 1,
                    jnp.where((accept & (rel_dec < STALL)) | tiny, 2,
                              jnp.where((~accept) & (lam >= 1e12), 3,
                                        0))).astype(jnp.int32)
                return (z, r2, JT2, rn, lam, att + 1, nJ, reason)

            if unroll > 1:
                # U masked micro-iterations per while trip.  A micro-iteration
                # whose entry state is already terminal (reason != 0 or budget
                # exhausted) leaves the state bit-identical, so the sequence of
                # applied updates — and every output — is EXACTLY that of the
                # unroll=1 loop; only loop-control overhead is amortized.
                inner_body = body

                def body(s):
                    for _ in range(unroll):
                        live = (s[7] == 0) & (s[5] < budget)
                        s_new = inner_body(s)
                        s = tuple(jnp.where(live, a_, b_)
                                  for a_, b_ in zip(s_new, s))
                    return s

            z, r, JT, rn, lam, att, nJ, reason = jax.lax.while_loop(
                cond, body, init)
            return z, rn, nJ, reason, att

        def body(carry, _):
            z_km1, z_k = carry
            z_init = z_k + EXTRAP * (z_k - z_km1)
            prev_c = prev_fn(z_k)
            z2, rn, nJ, reason, att = lm_step(z_init, prev_c)
            return (z_k, z2), (z2, rn, nJ, reason, att)

        (_, zT), (Z, rns, nJs, reasons, atts) = jax.lax.scan(
            body, (z0, z0), None, length=b1.NUM_STEPS, unroll=scan_unroll)
        if with_att:
            return Z, rns, nJs, reasons, atts
        return Z, rns, nJs, reasons

    return dict(rollout=jax.jit(rollout_fn, static_argnums=(3,)))


def make_ic_fast(su, opt, z0s=None, budget=None):
    """Optimized Gram-space IC fit.  Identical algorithm; chol + one-pass +
    no-cond restructure.  z0s/budget override only for clearly-labeled
    ALGORITHMIC arms (default = reference Z0S / IC_BUDGET)."""
    Gram_j, G_int, Lt_j, params = su.Gram_j, su.G_int, su.Lt_j, su.params
    Z0S = su.Z0S if z0s is None else jnp.asarray(z0s)
    B = IC_BUDGET if budget is None else int(budget)
    solve = pick_solver(opt)
    use_onepass = opt.get("onepass", True)
    use_nocond = opt.get("nocond", True)
    use_lean = opt.get("lean", False)
    use_nodot = opt.get("nodot", False)
    ic_unroll = int(opt.get("ic_unroll", 1))
    eyeK = jnp.eye(K, dtype=F64)
    if use_lean:
        (W1, bh1), (W2, bh2), (W3, bh3) = params["h"]
        W3aug = jnp.concatenate([W3, params["h_lin"]], axis=0)
        Wic = Lt_j @ W3aug.T                                # (R, h+K)

    def ic_lm_one(z0, cstar):
        if use_lean and use_nodot:
            bvec = Lt_j @ (bh3 - cstar)

            def r_of(z):
                a1 = jax.nn.silu(jnp.sum(z[:, None] * W1, axis=0) + bh1)
                a2 = jax.nn.silu(jnp.sum(a1[:, None] * W2, axis=0) + bh2)
                cc = jnp.concatenate([a2, z])
                return jnp.sum(Wic * cc[None, :], axis=1) + bvec
        elif use_lean:
            bvec = Lt_j @ (bh3 - cstar)

            def r_of(z):
                a1 = jax.nn.silu(z @ W1 + bh1)
                a2 = jax.nn.silu(a1 @ W2 + bh2)
                return Wic @ jnp.concatenate([a2, z]) + bvec
        else:
            def r_of(z):
                return Lt_j @ (b1.head(params, z) - cstar)

        if use_onepass:
            def rJT(z):
                r, lin = jax.linearize(r_of, z)
                JT = jax.vmap(lin)(eyeK)
                return r, JT
        else:
            def rJT(z):
                return r_of(z), jax.jacfwd(r_of)(z).T

        r0, JT0 = rJT(z0)
        init = (z0, r0, JT0, jnp.linalg.norm(r0), jnp.asarray(1e-6, F64),
                jnp.int32(0), jnp.int32(0))

        def cond(s):
            return (s[6] == 0) & (s[5] < B)

        def body(s):
            z, r, JT, rn, lam, att, done = s
            H = JT @ JT.T
            g = JT @ r
            dz = solve(H + lam * (jnp.diag(jnp.diag(H)) + 1e-30 * eyeK), -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            if use_nocond:
                r_new, JT_new = rJT(z_new)
                rn_new = jnp.linalg.norm(r_new)
                accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
                r2 = jnp.where(accept, r_new, r)
                JT2 = jnp.where(accept, JT_new, JT)
            else:
                rn_new = jnp.linalg.norm(r_of(z_new))
                accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
                r2, JT2 = jax.lax.cond(accept, lambda: rJT(z_new),
                                       lambda: (r, JT))
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            done = jnp.where((~accept) & (lam >= 1e12), 1, 0).astype(jnp.int32)
            return (z, r2, JT2, rn, lam, att + 1, done)

        if ic_unroll > 1:
            inner_body = body

            def body(s):
                for _ in range(ic_unroll):
                    live = (s[6] == 0) & (s[5] < B)
                    s_new = inner_body(s)
                    s = tuple(jnp.where(live, a_, b_)
                              for a_, b_ in zip(s_new, s))
                return s

        z, r, JT, rn, lam, att, done = jax.lax.while_loop(cond, body, init)
        return z, rn

    @jax.jit
    def ic_fit(u0_int):
        cstar = jnp.linalg.solve(Gram_j, G_int.T @ u0_int)
        zs, rns = jax.vmap(ic_lm_one, in_axes=(0, None))(Z0S, cstar)
        i = jnp.argmin(rns)
        return zs[i], rns[i]

    return ic_fit
