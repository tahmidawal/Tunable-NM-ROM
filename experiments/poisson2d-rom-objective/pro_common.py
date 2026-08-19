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

_ARCH_CANDS = [os.path.join(HERE, "deps", "nonlinear-decoder-architecture"),
               os.path.join(os.path.dirname(HERE), "nonlinear-decoder-architecture")]
for d in _ARCH_CANDS:
    if os.path.isfile(os.path.join(d, "nda_arch.py")):
        if d not in sys.path:
            sys.path.insert(0, d)
        break
else:
    raise ImportError("nda_arch.py not found in deps/ or sibling experiment")

import ms_parametric as mp                    # noqa: E402
from ms_autodecoder import lm_solve, infer_latents_lm   # noqa: E402,F401
import nda_arch as nda                         # noqa: E402

F64 = jnp.float64
SEED = mp.SEED


def decoder_config_from_env():
    """Explicit architecture manifest written into every new checkpoint."""
    name = os.environ.get("DECODER_ARCH", "film")
    if name == "film":
        return dict(name="film", hidden=mp.HIDDEN, n_layers=mp.N_LAYERS,
                    z_ff=mp.Z_FF)
    return nda.config(
        name, mp.HIDDEN, mp.N_LAYERS,
        group_size=int(os.environ.get("FILM_GROUP_SIZE", "8")),
        film_start=int(os.environ.get("FILM_START", "1")),
        z_width=int(os.environ.get("Z_WIDTH", "64")),
        residual_scale=(None if "RESIDUAL_SCALE" not in os.environ
                        else float(os.environ["RESIDUAL_SCALE"])),
        warp_max_shift=float(os.environ.get("WARP_MAX_SHIFT", "0.15")),
        warp_max_log_scale=float(os.environ.get("WARP_MAX_LOG_SCALE", "0.25"))) | {
                            "z_ff": mp.Z_FF}


def init_decoder(key, n_freq, k_lat, decoder_cfg):
    if decoder_cfg["name"] == "film":
        return mp.init_film_net(key, n_freq, k_lat, decoder_cfg.get("z_ff", 0))
    return nda.init(key, n_freq, k_lat, decoder_cfg, decoder_cfg.get("z_ff", 0))


def apply_decoder(params, z, xy, n_freq, decoder_cfg):
    if decoder_cfg.get("name", "film") == "film":
        return mp.film_apply(params, z, xy, n_freq, decoder_cfg.get("z_ff", 0))
    return nda.apply(params, z, xy, n_freq, decoder_cfg,
                     decoder_cfg.get("z_ff", 0))


def parameter_count(params):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def combined_apply(stages, z, xy):
    total = 0.0
    for stage in stages:
        cfg = stage.get("decoder_config", dict(name="film", z_ff=stage.get("z_ff", 0)))
        total = total + stage["eps"] * apply_decoder(
            stage["params"], z, xy, stage["n_freq"], cfg)
    return total


def stages_from_np(raw):
    return [{"params": jax.tree_util.tree_map(jnp.asarray, s["params"]),
             "n_freq": int(s["n_freq"]), "eps": float(s["eps"]),
             "z_ff": int(s.get("z_ff", 0)),
             "decoder_config": s.get("decoder_config",
                                       dict(name="film", z_ff=int(s.get("z_ff", 0))))}
            for s in raw]


# ------------------------------ checkpoints ------------------------------

def load_pkl(path):
    """Load a multistage-format auto-decoder pkl; asserts the env family config
    matches; the hard-BC flag is taken FROM THE PKL (env HARD_BC, if set, must
    agree — a silent mismatch would change every prediction)."""
    d = pickle.load(open(path, "rb"))
    cfg = d["config"]
    for k in ("N", "n_train", "n_val", "seed", "hidden", "n_layers"):
        assert cfg[k] == mp.CONFIG[k], f"env/pkl config mismatch on {k}: {cfg[k]} vs {mp.CONFIG[k]}"
    hb = int(cfg.get("hard_bc", 0))
    env_hb = os.environ.get("HARD_BC")
    if env_hb is not None and int(env_hb) != hb:
        raise ValueError(f"HARD_BC env={env_hb} conflicts with pkl hard_bc={hb}")
    stages = stages_from_np(d["stages"])
    return d, cfg, stages, np.asarray(d["z_tr"]), hb


def make_decoder(stages, hard_bc=False):
    """dec(z, xy) -> values at xy (P,2).  hard_bc multiplies by b(x,y) =
    16 x(1-x) y(1-y) (== 1 at the centre, 0 on the walls)."""
    if not hard_bc:
        return lambda z, xy: combined_apply(stages, z, xy)

    def dec(z, xy):
        b = 16.0 * xy[:, 0] * (1 - xy[:, 0]) * xy[:, 1] * (1 - xy[:, 1])
        return b * combined_apply(stages, z, xy)
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
        """0/1 mask keeping the M lowest eigenmodes, COMPLETE eigenshells (ties
        at the cutoff are kept, so the retained count can exceed M; use
        n_modes(M) for the actual count).  M=None or >= n_i^2: all."""
        if M is None or M >= self.n_i ** 2:
            return jnp.ones_like(self.lam)
        thr = self.lam_sorted[M - 1]
        return (self.lam <= thr).astype(F64)

    def n_modes(self, M):
        return int(np.asarray(self.mode_mask(M)).sum())

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

def lm_generic(HgV, V, z0, budget, lam0=1e-6, use_rel_dec=True,
               trust_delta=np.inf):
    """Levenberg-Marquardt on a generic objective given (H, g, val) at z and a
    cheap val(z).  Same damping schedule / acceptance / stopping / accounting
    as ms_autodecoder.lm_solve (for residual objectives H=J^T J, g=J^T r,
    val=||r|| it IS that solver).  use_rel_dec=False disables the relative-
    decrease stop (needed for objectives with an unknown additive constant,
    e.g. the Ritz energy) — then only the step-size stop / budget apply."""
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
        within_trust = float(jnp.linalg.norm(dz)) <= trust_delta
        z_new = z + dz
        val_new = float(V(z_new)) if within_trust else float("inf")
        n_r += int(within_trust)
        if within_trust and np.isfinite(val_new) and val_new < val:
            rel_dec = (val - val_new) / (abs(val) + 1e-300)
            step = float(jnp.linalg.norm(dz)) / (1.0 + float(jnp.linalg.norm(z)))
            z, val = z_new, val_new
            H, g, _ = HgV(z); n_J += 1; n_r += 1
            lam = max(lam / 3.0, 1e-12); acc += 1
            if (use_rel_dec and rel_dec < 1e-12) or step < 1e-13:
                reason = "converged"; break
        else:
            lam = min(lam * 10.0, 1e12); rej += 1
            if lam >= 1e12:
                reason = "lambda_max"; break
    return z, val, dict(accepted=acc, rejected=rej, n_resid_evals=n_r, n_jac_evals=n_J,
                        final_lambda=float(lam), reason=reason, attempts=attempt,
                        trust_delta=float(trust_delta))


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
    if name.startswith("weak_a"):
        a_s, m_s = name[len("weak_a"):].split("_M")
        return dict(kind="weak", alpha=float(a_s), M=None if m_s == "all" else int(m_s))
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
        else:                                       # cg K: EXPLICITLY unrolled K CG steps
            K = spec["K"]                           # (jax.scipy cg would use implicit diff = wrong J)
            def cg_unrolled(b):
                x = jnp.zeros_like(b); r = b; p = b; rs = jnp.sum(r * r)
                for _ in range(K):
                    Ap = grid.op(p)
                    alpha = rs / jnp.sum(p * Ap)
                    x = x + alpha * p; r = r - alpha * Ap
                    rs_new = jnp.sum(r * r)
                    p = r + (rs_new / rs) * p; rs = rs_new
                return x
            def wres(z, f2d):
                return cg_unrolled(fd_res(z, f2d)).reshape(-1)

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

def colloc_mode_table(grid, spec, pts_kind, centre_np):
    """(PhiT (M',m), Wl (M',)) for a spectral collocation objective at the given
    centre points.  on-grid -> discrete sine eigenpairs of A (exact full-grid
    limit); off-grid -> TRUNCATED CONTINUUM modes 2 sin(i pi x) sin(j pi y),
    i,j <= N-2 (same count as the discrete basis), lam = pi^2(i^2+j^2).  Both
    keep COMPLETE eigenshells (all modes with lam <= the M-th smallest), so
    M' >= M; the off-grid limit m->inf is a continuum strong-form objective,
    NOT the discrete FD one (labelled as such in reports)."""
    alpha, M = spec["alpha"], spec["M"]
    if pts_kind == "grid":
        mask = np.asarray(grid.mode_mask(M)).astype(bool)
        I, Jm = np.nonzero(mask)
        lam = np.asarray(grid.lam)[I, Jm]
        S = np.asarray(grid.S)
        px = np.rint(centre_np[:, 0] / grid.dx).astype(int) - 1
        py = np.rint(centre_np[:, 1] / grid.dx).astype(int) - 1
        PhiT = (S[px][:, I] * S[py][:, Jm]).T
    else:
        kk = np.arange(1, grid.N - 1)
        II, JJ = np.meshgrid(kk, kk, indexing="ij")
        lam_c = (np.pi ** 2) * (II ** 2 + JJ ** 2)
        lam_flat = lam_c.reshape(-1)
        M_eff = min(M if M is not None else lam_flat.size, lam_flat.size)
        thr = np.sort(lam_flat)[M_eff - 1]
        sel = lam_flat <= thr
        I, Jm = II.reshape(-1)[sel], JJ.reshape(-1)[sel]
        lam = lam_flat[sel]
        xs, ys = centre_np[:, 0], centre_np[:, 1]
        PhiT = 2.0 * np.sin(np.pi * np.outer(I, xs)) * np.sin(np.pi * np.outer(Jm, ys))
    if spec["kind"] == "weak":
        # weak form: Phi^T A u = Lambda Phi^T u  ->  weight Lambda^{1-alpha} on the
        # quadrature of the SMOOTH decoder output; the source side Lambda^{-alpha} Phi^T f
        # is precomputed once per query (see weak_source_term).
        return jnp.asarray(PhiT), jnp.asarray(lam ** (1.0 - alpha))
    return jnp.asarray(PhiT), jnp.asarray(lam ** (-alpha))


def weak_source_term(grid, spec, pts_kind, f_int2d):
    """Lambda^{-alpha} Phi^T f for the M lowest modes (same shell policy as
    colloc_mode_table).  on-grid: exact discrete projection of the interior
    source; off-grid: continuum modes integrated with the grid's dx^2 rule."""
    alpha, M = spec["alpha"], spec["M"]
    if pts_kind == "grid":
        mask = np.asarray(grid.mode_mask(M)).astype(bool)
        C = np.asarray(grid.spec(jnp.asarray(f_int2d)))
        I, Jm = np.nonzero(mask)
        lam = np.asarray(grid.lam)[I, Jm]
        return jnp.asarray(C[I, Jm] * lam ** (-alpha))
    kk = np.arange(1, grid.N - 1)
    II, JJ = np.meshgrid(kk, kk, indexing="ij")
    lam_c = (np.pi ** 2) * (II ** 2 + JJ ** 2)
    lam_flat = lam_c.reshape(-1)
    M_eff = min(M if M is not None else lam_flat.size, lam_flat.size)
    thr = np.sort(lam_flat)[M_eff - 1]
    sel = lam_flat <= thr
    I, Jm = II.reshape(-1)[sel], JJ.reshape(-1)[sel]
    lam = lam_flat[sel]
    X = np.asarray(grid.coords_int)
    Phi = 2.0 * np.sin(np.pi * np.outer(I, X[:, 0])) * np.sin(np.pi * np.outer(Jm, X[:, 1]))  # (M, n_i^2)
    return jnp.asarray((Phi @ np.asarray(f_int2d).reshape(-1)) * grid.dx ** 2 * lam ** (-alpha))


def boundary_points(m_b, rng):
    """m_b points uniformly on the perimeter of the unit square (for the
    off-grid soft-BC penalty)."""
    t = rng.uniform(0.0, 4.0, size=m_b)
    x = np.where(t < 1, t, np.where(t < 2, 1.0, np.where(t < 3, 3.0 - t, 0.0)))
    y = np.where(t < 1, 0.0, np.where(t < 2, t - 1.0, np.where(t < 3, 1.0, 4.0 - t)))
    return np.stack([x, y], 1)


def make_colloc_objective(dec, grid, spec, pts_kind, bc_beta=0.0):
    """Objective from residual values at m collocation points; ONE jit per
    (objective, pts_kind, m) — points/weights/mode tables are ARGUMENTS.

    pts_kind 'grid': pts = stencil point sets (5, m, 2), keep (5, m) 0/1 mask;
                     residual = FD stencil at the m nodes (exactly FOM rows;
                     the ghost-zero stencil enforces the Dirichlet BC).
    pts_kind 'offgrid': pts = (m, 2) interior points; residual = strong-form
                     -Laplace(dec)(x) - f(x) via autodiff (meshfree).  Since
                     point collocation alone cannot see harmonic components,
                     a soft-BC penalty block sqrt(bc_beta * 4/m_b) * u(z, x_b)
                     over m_b perimeter points (`keep` carries them, (m_b,2))
                     is APPENDED unless bc_beta == 0 (hard-BC decoders).
    wq: (m,) quadrature weights approximating the grid sum / integral.
    spec 'fd'   : || sqrt(wq) * r_S ||   (weighted LSPG on the subset)
    spec 'spec' : || Wl * (PhiT @ (wq * r_S)) ||  with (PhiT, Wl) from
                  colloc_mode_table (pass zeros-shaped dummies for 'fd').
    Returns (HgV, V) with signature (z, pts, keep, wq, PhiT, Wl, f_m)."""
    kind = spec["kind"]
    if pts_kind == "grid":
        def r_pts(z, pts, keep, f_m):
            u = dec(z, pts.reshape(-1, 2)).reshape(5, -1) * keep
            lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (grid.dx ** 2)
            return -lap - f_m
    else:
        def lap_one(z, x):
            return jnp.trace(jax.hessian(lambda xx: dec(z, xx[None, :])[0])(x))
        def r_pts(z, pts, keep, f_m):
            return -jax.vmap(lambda x: lap_one(z, x))(pts) - f_m

    if kind == "fd":
        core = lambda z, pts, keep, wq, PhiT, Wl, f_m: jnp.sqrt(wq) * r_pts(z, pts, keep, f_m)
    elif kind == "spec":
        core = lambda z, pts, keep, wq, PhiT, Wl, f_m: Wl * (PhiT @ (wq * r_pts(z, pts, keep, f_m)))
    elif kind == "weak":
        # WEAK FORM: || Lambda^{1-a} Phi^T_quad u(z) - Lambda^{-a} Phi^T f ||; pts = (m,2)
        # centre points only (no stencil, no decoder derivatives); f_m = weak_source_term.
        core = lambda z, pts, keep, wq, PhiT, Wl, f_m: Wl * (PhiT @ (wq * dec(z, pts))) - f_m
    else:
        raise ValueError("collocation objectives: fd, spec or weak")
    if pts_kind == "offgrid" and bc_beta > 0 and kind != "weak":
        def wres(z, pts, keep, wq, PhiT, Wl, f_m):
            ub = dec(z, keep) * jnp.sqrt(bc_beta * 4.0 / keep.shape[0])
            return jnp.concatenate([core(z, pts, keep, wq, PhiT, Wl, f_m), ub])
    else:
        wres = core

    def HgV(z, pts, keep, wq, PhiT, Wl, f_m):
        r = wres(z, pts, keep, wq, PhiT, Wl, f_m)
        J = jax.jacfwd(wres)(z, pts, keep, wq, PhiT, Wl, f_m)
        return J.T @ J, J.T @ r, jnp.linalg.norm(r)
    V = lambda z, pts, keep, wq, PhiT, Wl, f_m: jnp.linalg.norm(wres(z, pts, keep, wq, PhiT, Wl, f_m))
    return jax.jit(HgV), jax.jit(V)


# ------------------------------ capped Lawson-Hanson NNLS (EQ weights) ------------------------------

def nnls_capped(G, b, max_support, tol=1e-10, inner_max=200):
    """Lawson-Hanson active-set NNLS  min_{w>=0} ||G w - b||  that STOPS when the
    support reaches max_support (ECSW-style early termination) or at
    optimality.  Returns (w, ||Gw-b||, n_outer)."""
    n = G.shape[1]
    w = np.zeros(n)
    P = np.zeros(n, bool)
    r = b - G @ w
    outer = 0
    while outer < 5 * max_support + 10:            # safety cap on outer iterations
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
            s, *_ = np.linalg.lstsq(G[:, idx], b, rcond=None)
            if np.all(s > 0):
                w[:] = 0.0; w[idx] = s
                break
            neg = s <= 0
            alpha = np.min(w[idx][neg] / (w[idx][neg] - s[neg] + 1e-300))
            w[idx] = w[idx] + alpha * (s - w[idx])
            P[idx[w[idx] <= 1e-14]] = False
            w[~P] = 0.0
        r = b - G @ w
    return w, float(np.linalg.norm(r)), outer


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
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, max(1, min(100, steps // 10)), steps, 1e-6), 1e-5)
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
    decoder_cfg = decoder_config_from_env()
    params = init_decoder(k0, n_freq, k_lat, decoder_cfg)
    bfun = (lambda xy: 16.0 * xy[:, 0] * (1 - xy[:, 0]) * xy[:, 1] * (1 - xy[:, 1])) if hard_bc \
        else (lambda xy: jnp.ones((xy.shape[0],), F64))
    opt = optax.adamw(mp.make_lr_schedule(steps), weight_decay=1e-6)
    state = opt.init(params)

    def loss_fn(ps, z_b, t_b, w_b, pidx):
        xy = coords[pidx]
        pred = jax.vmap(lambda zi: bfun(xy) * apply_decoder(
            ps, zi, xy, n_freq, decoder_cfg))(z_b)
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
    stages = [{"params": params, "n_freq": n_freq, "eps": eps0,
               "z_ff": decoder_cfg.get("z_ff", 0),
               "decoder_config": decoder_cfg}]
    return stages, np.asarray(Z), eps0, n_freq, float(val)


def stages_to_np(stages):
    return [{"params": jax.tree_util.tree_map(np.asarray, s["params"]),
             "n_freq": int(s["n_freq"]), "eps": float(s["eps"]),
             "z_ff": int(s.get("z_ff", 0)),
             "decoder_config": s.get("decoder_config",
                                       dict(name="film", z_ff=int(s.get("z_ff", 0))))}
            for s in stages]
