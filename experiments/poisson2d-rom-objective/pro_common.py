"""Shared machinery for the POISSON-2D ROM-OBJECTIVE study (2026-08-16).

Question: the FiLM auto-decoder (multistage-precision worktree, K=8) has a
held-out inferred-latent error of ~8e-3 but the ROM solve on the plain FD
residual lands at ~6e-2.  Which OBJECTIVE / collocation / init closes the gap?

The FD residual r(z) = A u(z) - f (A = ghost-zero-Dirichlet 5-point -Laplacian
on the interior) amplifies grid-scale decoder error by up to ~lambda_max ~ 8/dx^2.
For SPD A the natural family of objectives is a SPECTRAL re-weighting of r:

    obj_{alpha,M}(z) = || Lambda_M^{-alpha} Phi_M^T r(z) ||_2

with Phi the (orthonormal, separable) sine eigenbasis of A, Lambda its
eigenvalues, M = number of lowest modes kept.  Special cases:
    alpha=0,   M=all : plain LSPG / FD residual (the control)
    alpha=0,   M<all : Galerkin projection onto smooth test functions
    alpha=1/2, M=all : energy norm ||u(z)-u*||_A  ==  Ritz / Galerkin NM-ROM
                       (stationarity J^T r = 0 of the energy functional)
    alpha=1,   M=all : ||A^{-1} r|| = ||u(z) - u*||_2 = the DATA MISFIT itself
                       (exact H^-1; an upper bound at FOM-solve cost)
Also: 'ritz' = GN on the energy functional E(z) = 1/2 u^T A u - f^T u with
H = J^T A J (matrix-free, no eigenbasis; must agree with alpha=1/2, M=all),
'cgK' = K CG iterations as an approximate A^{-1} (nonlinear in r), and
'lowpass' = spectral Gaussian smoothing of r (cheap low-pass proxy).

Everything is f64.  Validated pieces are IMPORTED from the multistage-precision
worktree (ms_parametric: family/FOM/FiLM nets/metrics; ms_autodecoder:
lm_solve, infer_latents_lm) — never re-implemented.  On the cluster the same
two files are staged under ./deps/.
"""
from __future__ import annotations

import os
import sys
import time
import pickle

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_CANDS = [os.path.join(HERE, "deps"),
          os.path.join(_WT, "2026-08-14-multistage-precision", "experiments",
                       "multistage-precision")]
for d in _CANDS:
    if os.path.isfile(os.path.join(d, "ms_parametric.py")):
        if d not in sys.path:
            sys.path.insert(0, d)
        break
else:
    raise ImportError("ms_parametric.py not found in deps/ or sibling worktree")

import ms_parametric as mp                    # noqa: E402
from ms_autodecoder import lm_solve, infer_latents_lm   # noqa: E402,F401

F64 = jnp.float64
SEED = mp.SEED


# ------------------------------ checkpoints ------------------------------

def load_pkl(path):
    d = pickle.load(open(path, "rb"))
    cfg = d["config"]
    for k in ("N", "n_train", "n_val", "seed", "hidden", "n_layers"):
        assert cfg[k] == mp.CONFIG[k], f"env/pkl config mismatch on {k}: {cfg[k]} vs {mp.CONFIG[k]}"
    stages = mp.stages_from_np(d["stages"])
    return d, cfg, stages, np.asarray(d["z_tr"])


def make_decoder(stages, hard_bc=False):
    """dec(z, xy) -> values at xy (P,2).  hard_bc multiplies by b(x,y) =
    16 x(1-x) y(1-y) (== 1 at the centre, 0 on the walls)."""
    if not hard_bc:
        return lambda z, xy: mp.combined_apply(stages, z, xy)

    def dec(z, xy):
        b = 16.0 * xy[:, 0] * (1 - xy[:, 0]) * xy[:, 1] * (1 - xy[:, 1])
        return b * mp.combined_apply(stages, z, xy)
    return dec


# ------------------------------ grid / spectral tools ------------------------------

class Grid:
    """Interior grid of an N-point Dirichlet FD mesh + sine eigenbasis of A."""

    def __init__(self, N):
        self.N = N
        self.n_i = N - 2
        self.dx = 1.0 / (N - 1)
        x = np.linspace(0.0, 1.0, N)
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.coords = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], 1))
        self.coords_int = jnp.asarray(np.stack([X[1:-1, 1:-1].reshape(-1),
                                                Y[1:-1, 1:-1].reshape(-1)], 1))
        bmask = np.zeros((N, N), bool)
        bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
        self.bpts = jnp.asarray(np.stack([X[bmask], Y[bmask]], 1))
        ii, jj = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1), indexing="ij")
        self.ix_full, self.iy_full = ii.reshape(-1), jj.reshape(-1)
        # orthonormal sine matrix S (n_i x n_i): S[p, k] = sqrt(2/(N-1)) sin(pi k p/(N-1))
        p = np.arange(1, N - 1)
        S = np.sqrt(2.0 / (N - 1)) * np.sin(np.pi * np.outer(p, p) / (N - 1))
        self.S = jnp.asarray(S)
        lam1 = (4.0 / self.dx ** 2) * np.sin(np.pi * p / (2 * (N - 1))) ** 2
        self.lam = jnp.asarray(lam1[:, None] + lam1[None, :])       # (n_i, n_i)
        self.lam_sorted = np.sort(np.asarray(self.lam).reshape(-1))

    def op(self, u_int2d):
        return mp.neg_lap_interior(u_int2d, self.N)

    def spec(self, R2d):
        """Coefficients Phi^T r for a 2-D interior field (n_i, n_i)."""
        return self.S.T @ R2d @ self.S

    def ispec(self, C2d):
        return self.S @ C2d @ self.S.T

    def mode_mask(self, M):
        """0/1 mask keeping the M lowest eigenmodes (M=None or >= n_i^2: all)."""
        if M is None or M >= self.n_i ** 2:
            return jnp.ones_like(self.lam)
        thr = self.lam_sorted[M - 1]
        return (self.lam <= thr).astype(F64)

    def stencil(self, ix, iy):
        """5 point sets (centre,+x,-x,+y,-y) and a 0/1 keep-mask that zeroes
        neighbours ON the wall (ghost-zero Dirichlet) — identical to the
        multistage-precision phase C."""
        offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        pts, keep = [], []
        for ox, oy in offs:
            jx, jy = ix + ox, iy + oy
            pts.append(np.stack([jx * self.dx, jy * self.dx], 1))
            keep.append(~((jx == 0) | (jx == self.N - 1) | (jy == 0) | (jy == self.N - 1)))
        return jnp.asarray(np.stack(pts)), jnp.asarray(np.stack(keep).astype(float))


def source_at(cx, cy, w, a, xs, ys):
    return a * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * w ** 2))


# ------------------------------ generic LM ------------------------------

def lm_generic(HgV, V, z0, budget, lam0=1e-6):
    """Levenberg-Marquardt on a generic objective given (H, g, val) at z and a
    cheap val(z).  Same damping schedule / acceptance / stopping / accounting
    as ms_autodecoder.lm_solve (for residual objectives H=J^T J, g=J^T r,
    val=||r|| it IS that solver)."""
    z = z0
    lam = lam0
    H, g, val = HgV(z)
    n_r, n_J = 1, 1
    val = float(val)
    acc = rej = 0
    reason = "budget"
    attempt = 0
    if not np.isfinite(val):
        return z, val, dict(accepted=0, rejected=0, n_resid_evals=1, n_jac_evals=1,
                            final_lambda=lam, reason="nan_at_init", attempts=0)
    for attempt in range(1, budget + 1):
        D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(H.shape[0], dtype=F64)
        dz = jnp.linalg.solve(H + lam * D, -g)
        if not bool(jnp.all(jnp.isfinite(dz))):
            lam = min(lam * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "nan_step_lambda_max"; break
            continue
        z_new = z + dz
        val_new = float(V(z_new)); n_r += 1
        if np.isfinite(val_new) and val_new < val:
            rel_dec = (val - val_new) / (abs(val) + 1e-300)
            step = float(jnp.linalg.norm(dz)) / (1.0 + float(jnp.linalg.norm(z)))
            z, val = z_new, val_new
            H, g, _ = HgV(z); n_J += 1; n_r += 1
            lam = max(lam / 3.0, 1e-12); acc += 1
            if rel_dec < 1e-12 or step < 1e-13:
                reason = "converged"; break
        else:
            lam = min(lam * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "lambda_max"; break
    return z, val, dict(accepted=acc, rejected=rej, n_resid_evals=n_r, n_jac_evals=n_J,
                        final_lambda=float(lam), reason=reason, attempts=attempt)


# ------------------------------ objectives (full interior grid) ------------------------------

def parse_objective(name):
    """'fd' | 'spec_a{alpha}_M{M|all}' | 'ritz' | 'cg{K}' | 'lowpass{sigma_cells}'"""
    if name == "fd":
        return dict(kind="fd")
    if name == "ritz":
        return dict(kind="ritz")
    if name.startswith("spec_a"):
        a_s, m_s = name[len("spec_a"):].split("_M")
        return dict(kind="spec", alpha=float(a_s), M=None if m_s == "all" else int(m_s))
    if name.startswith("cg"):
        return dict(kind="cg", K=int(name[2:]))
    if name.startswith("lowpass"):
        return dict(kind="lowpass", sigma=float(name[len("lowpass"):]))
    raise ValueError(name)


def make_full_objective(dec, grid, spec):
    """Objective on the FULL interior grid.  Returns (HgV(z,f2d), V(z,f2d),
    diag(z,f2d)) where diag returns (fd_resid_norm, obj_value).  f2d: source on
    the interior grid (n_i, n_i)."""
    n_i = grid.n_i
    kind = spec["kind"]

    def u_int(z):
        return dec(z, grid.coords_int).reshape(n_i, n_i)

    def fd_res(z, f2d):
        return grid.op(u_int(z)) - f2d

    if kind in ("fd", "spec", "cg", "lowpass"):
        if kind == "fd":
            wres = lambda z, f2d: fd_res(z, f2d).reshape(-1)
        elif kind == "spec":
            W = grid.mode_mask(spec["M"]) * grid.lam ** (-spec["alpha"])
            wres = lambda z, f2d: (W * grid.spec(fd_res(z, f2d))).reshape(-1)
        elif kind == "lowpass":
            sig = spec["sigma"] * grid.dx
            W = jnp.exp(-0.5 * sig ** 2 * grid.lam)
            wres = lambda z, f2d: (W * grid.spec(fd_res(z, f2d))).reshape(-1)
        else:                                       # cg K
            K = spec["K"]
            def wres(z, f2d):
                r = fd_res(z, f2d)
                return jax.scipy.sparse.linalg.cg(grid.op, r, tol=0.0, maxiter=K)[0].reshape(-1)

        def HgV(z, f2d):
            r = wres(z, f2d)
            J = jax.jacfwd(wres)(z, f2d)
            return J.T @ J, J.T @ r, jnp.linalg.norm(r)
        V = lambda z, f2d: jnp.linalg.norm(wres(z, f2d))
    elif kind == "ritz":
        # E(z) = 1/2 u^T A u - f^T u ; grad = J^T (A u - f) ; GN Hessian J^T A J
        def E(z, f2d):
            u = u_int(z)
            return 0.5 * jnp.sum(u * grid.op(u)) - jnp.sum(f2d * u)

        def HgV(z, f2d):
            r = fd_res(z, f2d).reshape(-1)
            Ju = jax.jacfwd(u_int)(z)                # (n_i, n_i, K)
            AJ = jax.vmap(grid.op, in_axes=2, out_axes=2)(Ju)
            Jf = Ju.reshape(-1, Ju.shape[-1]); AJf = AJ.reshape(-1, Ju.shape[-1])
            return Jf.T @ AJf, Jf.T @ r, E(z, f2d)
        V = E
    else:
        raise ValueError(kind)

    def diag(z, f2d):
        return jnp.linalg.norm(fd_res(z, f2d)), V(z, f2d)

    return jax.jit(HgV), jax.jit(V), jax.jit(diag)


# ------------------------------ objectives on COLLOCATION subsets ------------------------------

def make_colloc_objective(dec, grid, spec, pts_kind, pts, wq, keep=None, M=None):
    """Objective from residual values at m collocation points.

    pts_kind 'grid': pts = stencil point sets (5, m, 2) with keep mask (5, m);
                     residual = FD stencil at the m nodes (exactly the FOM rows).
    pts_kind 'offgrid': pts = (m, 2) interior points; residual = strong-form
                     -Laplace(dec)(x) - f(x) via autodiff (meshfree).
    wq: (m,) quadrature weights approximating the integral / grid sum
        (uniform: |Omega|/m for offgrid, n_i^2/m for grid nodes).
    spec: 'fd'  -> ||sqrt(wq) * r_S||  (weighted LSPG on the subset; with
                    uniform weights == plain subset residual up to a constant)
          'spec_a{alpha}_M{M}' -> || Lambda_M^{-alpha} Phi_M^T_quad r ||, with
                    Phi_M^T_quad r = sum_p wq_p phi_i(x_p) r_p over the M
                    lowest continuous modes phi_ij(x,y) = 2 sin(i pi x) sin(j pi y)
                    (continuous eigenvalues pi^2(i^2+j^2); on-grid we use the
                    discrete sine eigenpairs so the full-grid limit is exact).
    f_at: callable giving f at the centre points (m,) — passed at solve time.
    """
    kind = spec["kind"]
    if pts_kind == "grid":
        centre = pts[0]
        def r_pts(z, f_m):
            u = dec(z, pts.reshape(-1, 2)).reshape(5, -1) * keep
            lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (grid.dx ** 2)
            return -lap - f_m
    else:
        centre = pts
        def lap_one(z, x):
            h = jax.hessian(lambda xx: dec(z, xx[None, :])[0])(x)
            return jnp.trace(h)
        def r_pts(z, f_m):
            lap = jax.vmap(lambda x: lap_one(z, x))(centre)
            return -lap - f_m

    if kind == "fd":
        wres = lambda z, f_m: jnp.sqrt(wq) * r_pts(z, f_m)
    elif kind == "spec":
        alpha, M = spec["alpha"], spec["M"]
        # continuous mode table for the M lowest modes
        n_i = grid.n_i
        if pts_kind == "grid":
            # discrete eigenpairs at the collocation nodes: phi_ij(p) = S[px,i] S[py,j]
            mask = np.asarray(grid.mode_mask(M)).astype(bool)
            I, Jm = np.nonzero(mask)
            lam = np.asarray(grid.lam)[I, Jm]
            S = np.asarray(grid.S)
            px = np.rint(np.asarray(centre[:, 0]) / grid.dx).astype(int) - 1
            py = np.rint(np.asarray(centre[:, 1]) / grid.dx).astype(int) - 1
            PhiT = jnp.asarray(S[px][:, I] * S[py][:, Jm]).T          # (M, m)
        else:
            kk = np.arange(1, 64)
            II, JJ = np.meshgrid(kk, kk, indexing="ij")
            lam_c = (np.pi ** 2) * (II ** 2 + JJ ** 2)
            order = np.argsort(lam_c.reshape(-1))[:M]
            I, Jm = II.reshape(-1)[order], JJ.reshape(-1)[order]
            lam = lam_c.reshape(-1)[order]
            xs, ys = np.asarray(centre[:, 0]), np.asarray(centre[:, 1])
            PhiT = jnp.asarray(2.0 * np.sin(np.pi * np.outer(I, xs)) * np.sin(np.pi * np.outer(Jm, ys)))
        Wl = jnp.asarray(lam) ** (-alpha)
        wres = lambda z, f_m: Wl * (PhiT @ (wq * r_pts(z, f_m)))
    else:
        raise ValueError("collocation objectives: fd or spec only")

    def HgV(z, f_m):
        r = wres(z, f_m)
        J = jax.jacfwd(wres)(z, f_m)
        return J.T @ J, J.T @ r, jnp.linalg.norm(r)
    V = lambda z, f_m: jnp.linalg.norm(wres(z, f_m))
    return jax.jit(HgV), jax.jit(V), centre


# ------------------------------ encoder E(f) -> z ------------------------------

def lattice_source(cx, cy, w, a, m=16):
    t = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(t, t, indexing="ij")
    return source_at(cx[:, None], cy[:, None], w[:, None], a[:, None],
                     X.reshape(-1)[None], Y.reshape(-1)[None])


def fit_encoder(key, F_lat_tr, Z_tr, steps=3000, hidden=256, layers=3, lr=1e-3):
    """Small MLP from standardized lattice source to the learned training
    latents (MSE).  Returns (apply(F_lat)->Z, params, train_mse)."""
    mu = F_lat_tr.mean(0, keepdims=True); sd = F_lat_tr.std(0, keepdims=True) + 1e-8
    X = jnp.asarray((F_lat_tr - mu) / sd); Y = jnp.asarray(Z_tr)
    keys = jax.random.split(key, layers + 1)
    ps = [mp.init_dense(keys[0], X.shape[1], hidden)]
    for i in range(1, layers):
        ps.append(mp.init_dense(keys[i], hidden, hidden))
    out = mp.init_dense(keys[layers], hidden, Y.shape[1]); out["W"] = out["W"] * 0.1
    params = {"layers": ps, "out": out}

    def apply(p, x):
        h = x
        for l in p["layers"]:
            h = jax.nn.swish(h @ l["W"] + l["b"])
        return h @ p["out"]["W"] + p["out"]["b"]

    loss = lambda p: jnp.mean((apply(p, X) - Y) ** 2)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, 1e-6), 1e-5)
    st = opt.init(params)

    @jax.jit
    def step(p, s):
        v, g = jax.value_and_grad(loss)(p)
        up, s = opt.update(g, s, p)
        return optax.apply_updates(p, up), s, v
    for _ in range(steps):
        params, st, v = step(params, st)
    enc = lambda F_lat: np.asarray(apply(params, jnp.asarray((F_lat - mu) / sd)))
    return enc, params, float(v)


# ------------------------------ auto-decoder training (stage 0, optional hard BC) ------------------------------

def train_autodecoder_stage0(key, np_rng, coords, U_tr, k_lat, hard_bc, steps,
                             batch, p_sub, lat_lr=5e-3, lat_reg=1e-4, tag="A"):
    """Stage-0 auto-decoder training identical to ms_autodecoder phase (A)
    (lazy per-row latent Adam, inverse-energy weights, warmup-cosine AdamW),
    with an optional hard-BC multiplier b(x) on the decoder output.  Returns
    stages(list of 1), Z_tr(np), eps0, n_freq, adam_loss."""
    N = int(round(np.sqrt(coords.shape[0])))
    eps0 = float(jnp.sqrt(jnp.mean(U_tr ** 2)))
    n_freq = mp.freq_schedule(mp.dominant_radial_freq(U_tr, N), 0, N)
    target = U_tr / eps0
    weights = mp.sample_weights(U_tr)
    n_s, n_pts = target.shape
    key, kz, k0 = jax.random.split(key, 3)
    Z = 0.1 * jax.random.normal(kz, (n_s, k_lat), dtype=F64)
    params = mp.init_film_net(k0, n_freq, k_lat, 0)
    bfun = (lambda xy: 16.0 * xy[:, 0] * (1 - xy[:, 0]) * xy[:, 1] * (1 - xy[:, 1])) if hard_bc \
        else (lambda xy: jnp.ones((xy.shape[0],), F64))
    opt = optax.adamw(mp.make_lr_schedule(steps), weight_decay=1e-6)
    state = opt.init(params)

    def loss_fn(ps, z_b, t_b, w_b, pidx):
        xy = coords[pidx]
        pred = jax.vmap(lambda zi: bfun(xy) * mp.film_apply(ps, zi, xy, n_freq, 0))(z_b)
        se = jnp.mean((pred - t_b[:, pidx]) ** 2, axis=1)
        return jnp.mean(w_b * se)

    b1, b2, eps_a = 0.9, 0.999, 1e-8
    lat_sched = optax.warmup_cosine_decay_schedule(0.0, lat_lr, max(1, steps // 20), steps,
                                                   end_value=1e-9)

    @jax.jit
    def step_wz(ps, st, Z, m, v, cnt, gstep, bi, t_b, w_b, pidx):
        z_b = Z[bi]
        def lz(ps_, z_b_):
            return loss_fn(ps_, z_b_, t_b, w_b, pidx) + lat_reg * jnp.mean(z_b_ ** 2)
        val, (gp, gz) = jax.value_and_grad(lz, argnums=(0, 1))(ps, z_b)
        up, st = opt.update(gp, st, ps)
        ps = optax.apply_updates(ps, up)
        m_b = b1 * m[bi] + (1 - b1) * gz
        v_b = b2 * v[bi] + (1 - b2) * gz ** 2
        c_b = cnt[bi] + 1.0
        mhat = m_b / (1 - b1 ** c_b[:, None]); vhat = v_b / (1 - b2 ** c_b[:, None])
        z_new = z_b - lat_sched(gstep) * mhat / (jnp.sqrt(vhat) + eps_a)
        return (ps, st, Z.at[bi].set(z_new), m.at[bi].set(m_b), v.at[bi].set(v_b),
                cnt.at[bi].set(c_b), val)

    m = jnp.zeros_like(Z); v = jnp.zeros_like(Z); cnt = jnp.zeros((n_s,), F64)
    all_pts = jnp.arange(n_pts)
    t0 = time.time()
    for it in range(steps):
        bi = np.arange(n_s) if batch >= n_s else np_rng.choice(n_s, size=batch, replace=False)
        pidx = all_pts if p_sub <= 0 or p_sub >= n_pts else \
            jnp.asarray(np_rng.choice(n_pts, size=p_sub, replace=False))
        params, state, Z, m, v, cnt, val = step_wz(params, state, Z, m, v, cnt, it,
                                                   jnp.asarray(bi), target[bi], weights[bi], pidx)
        if it % 5000 == 0:
            print(f"  [{tag}] step {it:6d} loss {float(val):.3e} [{time.time()-t0:.0f}s]", flush=True)
    print(f"  [{tag}] Adam {steps} steps in {time.time()-t0:.0f}s (final batch loss {float(val):.3e})",
          flush=True)
    stages = [{"params": params, "n_freq": n_freq, "eps": eps0, "z_ff": 0}]
    return stages, np.asarray(Z), eps0, n_freq, float(val)


def stages_to_np(stages):
    return mp.stages_to_np(stages)
