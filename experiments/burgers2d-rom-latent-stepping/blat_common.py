"""Burgers-2D INR-decoder LATENT-STEPPING ROM -- shared machinery.

Design (2026-08-16):
  FOM      : the burgers2d testbed's implicit Burgers solver, imported verbatim
             (upwind advection, centered diffusion, backward Euler dt=0.005 x 50,
             Newton/BiCGStab).  Its `residual(u, u_prev, nu)` IS the discrete
             operator the ROM steps -- no re-implementation.
  decoder  : (i) FiLM coordinate net u(x; z) with a HARD Dirichlet factor
                 b(x,y) = 16 x(1-x) y(1-y) (BC_MODE=poly, default; binary grid
                 mask available as BC_MODE=binary), trained as an AUTO-DECODER
                 (one latent per (trajectory, time) snapshot);
             (ii) POD basis V (linear control), same solver.
  residual : POINT-LOCAL.  For a collocation set of m interior nodes the
             backward-Euler residual needs the decoder only at the m 5-point
             stencils (m x 5 evaluations, n-free); OFF-GRID collocation uses the
             strong form with autodiff derivatives of the decoder (meshfree).
  objective: pluggable weighting of the residual field (fd | lowpass | ihelm),
             see `make_objective`.
  solver   : per time step, z_{n+1} from z_n by damped LM on ||W R_n(D(z))||
             (LSPG) or by damped Newton on J_D^T W R_n(D(z)) = 0 (Galerkin
             root), warm-started; cold-start z_0 by LM data-misfit to the KNOWN
             initial condition u0.  The held-out trajectory never touches the
             ROM path.

Environment-variable collision note: burgers2d_film reads N/N_TRAIN/N_VAL/
HIDDEN/N_FREQ at import; ms_parametric reads N/HIDDEN/N_LAYERS/STEPS/BATCH/
P_SUB.  We import burgers2d_film FIRST (its defaults = the sweep checkpoints'
architecture: HIDDEN 256, 5 layers), then set the auto-decoder architecture
env (AD_HIDDEN/AD_LAYERS -> HIDDEN/N_LAYERS) and import ms_parametric.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BURGERS_DIR = os.path.join(_WT, "2026-08-14-burgers2d-coord-rom", "experiments",
                           "burgers2d-coord-rom")
MSP_DIR = os.path.join(_WT, "2026-08-14-multistage-precision", "experiments",
                       "multistage-precision")
# on the cluster the two source dirs are staged as siblings under ./deps/
for cand in (BURGERS_DIR, os.path.join(HERE, "deps", "burgers2d-coord-rom")):
    if os.path.isdir(cand):
        BURGERS_DIR = cand
        break
for cand in (MSP_DIR, os.path.join(HERE, "deps", "multistage-precision")):
    if os.path.isdir(cand):
        MSP_DIR = cand
        break
for d in (BURGERS_DIR, MSP_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

# --- burgers2d testbed first (sweep-checkpoint architecture from its defaults)
os.environ.setdefault("HIDDEN", "256")
import burgers2d_film as bf                       # noqa: E402

# --- auto-decoder architecture for ms_parametric (read at ITS import)
AD_HIDDEN = int(os.environ.get("AD_HIDDEN", "256"))
AD_LAYERS = int(os.environ.get("AD_LAYERS", "5"))
os.environ["HIDDEN"] = str(AD_HIDDEN)
os.environ["N_LAYERS"] = str(AD_LAYERS)
import ms_parametric as mp                        # noqa: E402
from ms_autodecoder import lm_solve               # noqa: E402,F401

assert mp.HIDDEN == AD_HIDDEN and mp.N_LAYERS == AD_LAYERS
assert bf.HIDDEN == 256, "burgers2d_film must keep the sweep architecture"

F64 = jnp.float64
N = bf.N
DT = bf.DT
NUM_STEPS = bf.NUM_STEPS
N_TRAIN, N_VAL = bf.N_TRAIN, bf.N_VAL
SEED = bf.SEED
BC_MODE = os.environ.get("BC_MODE", "poly")      # poly | binary
K_LAT = int(os.environ.get("K_LAT", "8"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_BUDGET = int(os.environ.get("GN_BUDGET", "30"))   # attempts per time step
GN_TOL = float(os.environ.get("GN_TOL", "1e-9"))     # ||r|| <= tol * ||u_prev||
IC_BUDGET = int(os.environ.get("IC_BUDGET", "100"))  # LM attempts for z_0

CONFIG = dict(N=N, dt=DT, num_steps=NUM_STEPS, n_train=N_TRAIN, n_val=N_VAL,
              seed=SEED, bc_mode=BC_MODE, k_lat=K_LAT, n_test=N_TEST,
              gn_budget=GN_BUDGET, gn_tol=GN_TOL, ic_budget=IC_BUDGET,
              ad_hidden=AD_HIDDEN, ad_layers=AD_LAYERS, x64=True)


def log(*a):
    print(*a, flush=True)


# --------------------------- grid / data ---------------------------

def grid_coords(n):
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)     # flat = i*n+j


def interior_indices(n):
    ii, jj = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1), indexing="ij")
    return (ii * n + jj).reshape(-1)


TEST_SEED = int(os.environ.get("TEST_SEED", str(SEED + 1)))


def max_rel_residual(U, nu, n, chunk=64):
    """Max over trajectories/steps of ||R(u_{k+1}, u_k)|| / ||u_k|| through the
    FOM's own residual (independent check that the data is a converged
    implicit solution; NaN-propagating)."""
    _, res = bf.make_rollout(n)
    f = jax.jit(jax.vmap(lambda u1, u0, nu_: jnp.linalg.norm(res(u1, u0, nu_))
                         / jnp.linalg.norm(u0)))
    worst = 0.0
    for s in range(0, U.shape[0], chunk):
        e = min(s + chunk, U.shape[0])
        for k in range(NUM_STEPS):
            r = float(jnp.max(f(jnp.asarray(U[s:e, k + 1]), jnp.asarray(U[s:e, k]),
                                jnp.asarray(nu[s:e]))))
            if not np.isfinite(r) or r > worst:
                worst = r
    return worst


def build_data(n=N, check=True):
    """TRAIN/VAL: regenerated from SEED (identical draw to the sweep; the sweep
    checkpoints were model-selected on VAL, so VAL is NOT used as a test set
    here).  TEST: N_TEST fresh trajectories from TEST_SEED (= SEED+1), never
    seen by any training or model selection.  Aborts if any trajectory's FOM
    residual exceeds 1e-8 (unconverged Newton would otherwise be 'truth')."""
    U, z, cx, cy, w, a, nu = bf.build_trajectories(n)
    cxt, cyt, wt, at, nut, zt = bf.sample_params(seed=TEST_SEED, m=N_TEST)
    rollout, _ = bf.make_rollout(n)
    U0 = np.stack([bf.blob_ic(n, cxt[i], cyt[i], wt[i], at[i]) for i in range(N_TEST)])
    snaps, res = rollout(jnp.asarray(U0), jnp.asarray(nut))
    Ut = np.asarray(snaps).transpose(1, 0, 2)
    rt = float(jnp.max(res))
    if not np.isfinite(rt) or rt > 1e-8:
        raise SystemExit(f"TEST FOM Newton residual {rt:.2e} > 1e-8")
    d = dict(U=U, z=z, cx=cx, cy=cy, w=w, a=a, nu=nu, U_test=Ut, z_test=zt,
             nu_test=nut, cx_test=cxt, cy_test=cyt, w_test=wt, a_test=at)
    if check:
        worst = max(max_rel_residual(U, nu, n), max_rel_residual(Ut, nut, n))
        log(f"  data check: max FOM rel residual over all trajectories {worst:.2e}")
        if not np.isfinite(worst) or worst > 1e-8:
            raise SystemExit(f"FOM residual {worst:.2e} > 1e-8: data not converged")
        d["max_fom_rel_residual"] = worst
    return d


def data_fingerprint(U):
    """Cheap reproducibility check across machines (f64 sums)."""
    return dict(sum=float(np.sum(U)), sumsq=float(np.sum(U * U)),
                shape=list(U.shape))


# --------------------------- POD (spatial Gram) ---------------------------

def pod_basis(S, kmax=64):
    """SVD-optimal basis of the snapshot rows S (n_s, n^2) via the SMALLER
    Gram (spatial n^2 x n^2 when n^2 < n_s), f64 on the HOST (the 26112^2
    device Gram OOMs an A100 -- burgers2d landmine).  Returns V (n^2, kmax),
    singular values."""
    S = np.asarray(S, dtype=np.float64)
    n_s, n2 = S.shape
    if n2 <= n_s:
        G = S.T @ S
        ev, EV = np.linalg.eigh(G)
        o = np.argsort(ev)[::-1]
        ev, EV = ev[o], EV[:, o]
        V = EV[:, :kmax]
        sv = np.sqrt(np.maximum(ev[:kmax], 0.0))
    else:
        G = S @ S.T
        ev, EV = np.linalg.eigh(G)
        o = np.argsort(ev)[::-1]
        ev, EV = ev[o], EV[:, o]
        sv = np.sqrt(np.maximum(ev[:kmax], 0.0))
        V = (S.T @ EV[:, :kmax]) / np.maximum(sv, 1e-300)
    dev = float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))
    return V, sv, dev


# --------------------------- decoders ---------------------------

def bc_factor(xy):
    """Hard Dirichlet factor.  poly: 16 x(1-x) y(1-y) (smooth, exact zero on the
    walls, defined off-grid); binary: 1 in the open square, 0 on the walls."""
    x, y = xy[:, 0], xy[:, 1]
    if BC_MODE == "binary":
        return jnp.where((x > 0) & (x < 1) & (y > 0) & (y < 1), 1.0, 0.0)
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


class CoordDecoder:
    """u(x; z) = eps * b(x) * film(x; z).  Meshfree: evaluate anywhere."""

    def __init__(self, params, n_freq, eps, k_lat):
        self.params = params
        self.n_freq = int(n_freq)
        self.eps = float(eps)
        self.k = int(k_lat)
        self.kind = "coord"

    def __call__(self, z, xy):
        return self.eps * bc_factor(xy) * mp.film_apply(self.params, z, xy,
                                                        self.n_freq)


class PODDecoder:
    """u = V c on the grid; V rows are exactly zero on the walls (snapshots are).
    Only defined at grid nodes (rows), so off-grid collocation is unavailable."""

    def __init__(self, V):
        self.V = jnp.asarray(V, dtype=F64)
        self.k = int(V.shape[1])
        self.kind = "pod"

    def rows(self, z, idx):
        return self.V[idx] @ z


# --------------------------- residuals (point-local) ---------------------------

def stencil_indices(idx, n):
    """(m,5) flat indices [c, x+, x-, y+, y-] for interior nodes idx."""
    return np.stack([idx, idx + n, idx - n, idx + 1, idx - 1], axis=1)


def be_residual_from_stencil(us, up_c, nu, n):
    """Backward-Euler Burgers residual at m interior nodes from the decoder
    values us (m,5) at the stencil [c,x+,x-,y+,y-] and up_c (m,) = previous
    state at the centers.  Bit-for-bit the burgers2d_film FOM interior residual
    (checked in the smoke test)."""
    dx = 1.0 / (n - 1)
    c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
    lap = (xp + xm + yp + ym - 4.0 * c) / dx**2
    ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
    uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
    return c - up_c + DT * (c * (ux + uy) - nu * lap)


def strong_form_residual(dec, z, up_pts, nu, xy):
    """Off-grid strong-form BE residual at points xy (m,2) with autodiff
    derivatives of the coord decoder: u - u_prev + dt (u (u_x+u_y) - nu lap u).
    Advection is CENTERED here (the continuous operator), unlike the FOM's
    upwind stencil -- the meshfree variant discretizes the PDE, not the FOM."""
    def f(p):
        return dec(z, p[None, :])[0]
    def one(p):
        u = f(p)
        g = jax.grad(f)(p)
        H = jax.hessian(f)(p)
        return u + DT * (u * (g[0] + g[1]) - nu * (H[0, 0] + H[1, 1]))
    return jax.vmap(one)(xy) - up_pts


# --------------------------- objectives ---------------------------

def make_objective(name, n, idx_full):
    """Returns weight(r_field_interior) -> weighted residual vector.  Full-grid
    only for lowpass/ihelm (they need the field); fd is the identity."""
    m = n - 2
    if name == "fd":
        return lambda r: r
    if name.startswith("lowpass"):
        # separable Gaussian blur, sigma in cells (lowpass2 -> sigma 2)
        sig = float(name[len("lowpass"):] or 2.0)
        rad = int(3 * sig)
        k1 = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sig) ** 2)
        k1 = jnp.asarray(k1 / k1.sum())
        def w(r):
            R = r.reshape(m, m)
            R = jax.vmap(lambda row: jnp.convolve(row, k1, mode="same"))(R)
            R = jax.vmap(lambda col: jnp.convolve(col, k1, mode="same"),
                         in_axes=1, out_axes=1)(R)
            return R.reshape(-1)
        return w
    if name.startswith("ihelm"):
        # K CG iterations of (I - dt*nu_ref*Lap)^{-1} on the interior field:
        # the linear (diffusive) BE operator as an approximate H^-1 metric.
        K = int(name[len("ihelm"):] or 20)
        dx = 1.0 / (n - 1)
        def apply_A(R, nu):
            Rp = jnp.pad(R, 1)
            lap = (Rp[2:, 1:-1] + Rp[:-2, 1:-1] + Rp[1:-1, 2:] + Rp[1:-1, :-2]
                   - 4.0 * R) / dx**2
            return R - DT * nu * lap
        def w(r, nu):
            R = r.reshape(m, m)
            sol, _ = jax.scipy.sparse.linalg.cg(lambda X: apply_A(X, nu), R,
                                                maxiter=K, tol=0.0)
            return sol.reshape(-1)
        w.needs_nu = True
        return w
    raise ValueError(name)


# --------------------------- collocation ---------------------------

def make_collocation(name, n, rng, data_row=None):
    """name: full | rand<m> | biased<m> | offgrid<m>.  Returns dict with either
    'idx' (interior flat indices) or 'xy' (off-grid points).  biased<m>: half
    uniform interior, half near the blob (advected front) using the KNOWN
    initial condition u0 (>= 0) as the sampling density -- no held-out
    trajectory information."""
    interior = interior_indices(n)
    if name == "full":
        return dict(kind="grid", idx=interior)
    if name.startswith("rand"):
        m = min(int(name[4:]), interior.size)
        return dict(kind="grid", idx=np.sort(rng.choice(interior, m, replace=False)))
    if name.startswith("biased"):
        m = min(int(name[6:]), interior.size)
        u0 = np.asarray(data_row["u0"]).reshape(n, n)[1:-1, 1:-1].reshape(-1)
        p = u0 / u0.sum()
        # blend: 0.5 uniform + 0.5 IC-weighted (the uniform half covers the
        # advected front, which moves ~a/2 per axis over T)
        pu = np.full_like(p, 1.0 / p.size)
        P = 0.5 * pu + 0.5 * p
        pick = rng.choice(interior.size, m, replace=False, p=P)
        return dict(kind="grid", idx=np.sort(interior[pick]))
    if name.startswith("offgrid"):
        m = int(name[7:])
        xy = rng.uniform(1.0 / (n - 1), 1.0 - 1.0 / (n - 1), size=(m, 2))
        return dict(kind="offgrid", xy=xy)
    raise ValueError(name)


# --------------------------- WEAK-form Galerkin (Agent A recipe) ---------------------------
#
# Test modes phi_i = discrete sine modes on the interior grid (exact eigenvectors of
# the FOM's ghost-zero 5-point Laplacian, -L phi_i = lam_i phi_i).  Weak BE residual:
#   R_i(z) = w_i [ phi_i^T (u - u_n) + dt ( phi_i^T N(u) + nu lam_i phi_i^T u ) ]
# with u = D(z) on the interior, N = the FOM's upwind advection ('weak', exact FOM
# operator; needs u at the 5-point stencils of the quadrature nodes) or, for the
# meshfree variant 'weakc', the continuum advection integrated by parts,
#   phi_i^T N(u) -> -1/2 sum_q om_q u_q^2 (dphi_i/dx + dphi_i/dy)(x_q),  lam_i^c = pi^2(kx^2+ky^2)
# (no decoder derivatives, points anywhere; it targets the continuum PDE, not the
# FOM's upwind discretization -- O(h) apart at N=64).  Weighting w_i =
# (1 + dt nu lam_i)^-alpha (alpha=WEAK_ALPHA, default 1).  Hyper-reduction: quadrature
# weights om_q on m nodes from capped Lawson-Hanson NNLS fitted to reproduce the mode
# projections of DECODER-OUTPUT snapshots (u and N(u) at training latents).

WEAK_ALPHA = float(os.environ.get("WEAK_ALPHA", "1.0"))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_POOL = int(os.environ.get("EQ_POOL", "4096"))       # meshfree candidate pool size


def test_modes(n, M):
    """M lowest discrete sine modes on the interior (n-2)^2 grid.  Returns
    kx, ky (M,), Phi (n_i^2, M) unit-2-norm columns, lam_disc (M,), lam_cont (M,)."""
    dx = 1.0 / (n - 1)
    kk = np.arange(1, n - 1)
    KX, KY = np.meshgrid(kk, kk, indexing="ij")
    lam = (4.0 / dx**2) * (np.sin(np.pi * KX / (2 * (n - 1)))**2
                           + np.sin(np.pi * KY / (2 * (n - 1)))**2)
    order = np.argsort(lam.reshape(-1), kind="stable")[:M]
    kx, ky = KX.reshape(-1)[order], KY.reshape(-1)[order]
    xi = kk / (n - 1)
    Sx = np.sin(np.pi * np.outer(xi, kx))            # (n_i, M)
    Sy = np.sin(np.pi * np.outer(xi, ky))
    Phi = (Sx[:, None, :] * Sy[None, :, :]).reshape(-1, M)   # interior flat i*(n-2)+j
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
    return kx, ky, Phi, lam.reshape(-1)[order], (np.pi**2) * (kx**2 + ky**2)


def modes_at(xy, kx, ky, n):
    """Continuous sine modes at points xy (m,2) with the SAME normalisation as
    the grid columns (unit 2-norm over the interior grid): values (m,M) and
    gradients (m,M,2)."""
    nrm = np.sqrt(((n - 1) / 2.0) ** 2)      # ||sin(pi k x_i)||_2^2 over i=1..n-2 = (n-1)/2
    x, y = xy[:, 0:1], xy[:, 1:2]
    sx, sy = jnp.sin(jnp.pi * kx * x), jnp.sin(jnp.pi * ky * y)
    cx, cy = jnp.cos(jnp.pi * kx * x), jnp.cos(jnp.pi * ky * y)
    val = sx * sy / nrm
    gx = jnp.pi * kx * cx * sy / nrm
    gy = jnp.pi * ky * sx * cy / nrm
    return val, jnp.stack([gx, gy], axis=-1)


def upwind_adv_field(u_int, n):
    """N(u) = u (u_x + u_y) with the FOM's sign-upwind stencil on the interior
    field u_int (n_i^2,), ghost zeros on the walls."""
    ni = n - 2
    dx = 1.0 / (n - 1)
    U = jnp.pad(u_int.reshape(ni, ni), 1)
    c = U[1:-1, 1:-1]
    ux = jnp.where(c > 0, (c - U[:-2, 1:-1]) / dx, (U[2:, 1:-1] - c) / dx)
    uy = jnp.where(c > 0, (c - U[1:-1, :-2]) / dx, (U[1:-1, 2:] - c) / dx)
    return (c * (ux + uy)).reshape(-1)


def nnls_capped(G, b, max_support, tol=1e-10, inner_max=200):
    """Lawson-Hanson active-set NNLS min_{w>=0} ||G w - b|| that STOPS when the
    support reaches max_support (ECSW-style) or at optimality (copied from the
    Poisson study's pro_common.nnls_capped).  Returns (w, ||Gw-b||, n_outer)."""
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
                w[:] = 0.0; w[idx] = s_
                break
            neg = s_ <= 0
            alpha = np.min(w[idx][neg] / (w[idx][neg] - s_[neg] + 1e-300))
            w[idx] = w[idx] + alpha * (s_ - w[idx])
            P[idx[w[idx] <= 1e-14]] = False
            w[~P] = 0.0
        r = b - G @ w
    return w, float(np.linalg.norm(r)), outer


def fit_eq_weights(dec, n, M, m, Z_snap, kind="weak", pool="grid", rng=None):
    """NNLS-EQ quadrature for the weak form.  Candidates: interior grid nodes
    (pool='grid') or EQ_POOL random interior points (pool='off', 'weakc' only).
    Snapshots: decoder outputs u_s (and the upwind field N(u_s) for 'weak',
    u_s^2 for 'weakc') at the latents Z_snap (K x n_snap).  Targets: exact
    full-grid projections (grid rule).  Returns dict(idx|xy, w (m,), info)."""
    rng = rng or np.random.default_rng(0)
    t0 = time.time()
    kx, ky, Phi, lam, lamc = test_modes(n, M)
    coords = jnp.asarray(grid_coords(n))
    interior = interior_indices(n)
    xy_int = coords[jnp.asarray(interior)]
    n_i2 = interior.size
    kxj, kyj = jnp.asarray(kx, dtype=F64), jnp.asarray(ky, dtype=F64)
    if pool == "grid":
        cand_xy = xy_int
        Phi_c = jnp.asarray(Phi)                                    # (n_c, M)
        dPhi_c = modes_at(xy_int, kxj, kyj, n)[1] if kind == "weakc" else None
    else:
        assert kind == "weakc", "the FOM upwind stencil needs grid nodes"
        cand_xy = jnp.asarray(rng.uniform(1.0 / (n - 1), 1.0 - 1.0 / (n - 1), size=(EQ_POOL, 2)))
        Phi_c, dPhi_c = modes_at(cand_xy, kxj, kyj, n)
    n_c = cand_xy.shape[0]
    u_full = jax.jit(lambda z: dec(z, xy_int) if dec.kind == "coord" else dec.rows(z, jnp.asarray(interior)))
    u_cand = jax.jit(lambda z: dec(z, cand_xy)) if pool == "off" else u_full
    Gs, bs = [], []
    Phi_np = np.asarray(Phi)
    for z in Z_snap:
        z = jnp.asarray(z, dtype=F64)
        uf = u_full(z)                                              # (n_i2,)
        uc = u_cand(z)                                              # (n_c,)
        if kind == "weak":
            Nf = upwind_adv_field(uf, n)
            for v_f, v_c in ((np.asarray(uf), np.asarray(uc)), (np.asarray(Nf), np.asarray(Nf))):
                bs.append(Phi_np.T @ v_f)                           # (M,)
                Gs.append(np.asarray(Phi_c).T * v_c[None, :])       # (M, n_c)
        else:
            gsum = np.asarray(dPhi_c[..., 0] + dPhi_c[..., 1])      # (n_c, M)
            dPhi_f = np.asarray(modes_at(xy_int, kxj, kyj, n)[1])
            gsum_f = dPhi_f[..., 0] + dPhi_f[..., 1]
            uf_np, uc_np = np.asarray(uf), np.asarray(uc)
            bs.append(Phi_np.T @ uf_np);          Gs.append(np.asarray(Phi_c).T * uc_np[None, :])
            bs.append(gsum_f.T @ (uf_np ** 2));    Gs.append(gsum.T * (uc_np ** 2)[None, :])
    G = np.concatenate(Gs, axis=0)                                  # (rows, n_c)
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
    wq, rnorm_final, _ = nnls_capped(G[:, keep], b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    rnorm_final = float(np.linalg.norm(G[:, keep] @ wq - b))
    info = dict(support=int(len(supp)), padded=int(padded), rnorm_capped=float(rnorm),
                rnorm_final=rnorm_final, b_norm=float(np.linalg.norm(b)),
                rel_fit=rnorm_final / float(np.linalg.norm(b)), n_rows=int(G.shape[0]),
                n_cand=int(n_c), secs=time.time() - t0, M=int(M), m=int(len(keep)),
                kind=kind, pool=pool)
    log(f"  NNLS-EQ {kind}/{pool} M={M} m={m}: support {len(supp)} (+{padded} pad), "
        f"rel fit {info['rel_fit']:.2e} [{info['secs']:.0f}s]")
    out = dict(kind="grid" if pool == "grid" else "offgrid", w=wq, info=info)
    if pool == "grid":
        out["idx"] = interior[keep]
    else:
        out["xy"] = np.asarray(cand_xy)[keep]
    return out


def make_weak_ops(dec, n, colloc, kind="weak", M=64, alpha=WEAK_ALPHA, solver="lspg"):
    """Weak-form step operators.  colloc: dict(kind='grid', idx, w) or
    dict(kind='offgrid', xy, w) (weights w = quadrature weights; for the full
    grid pass idx=interior and w=None -> ones = exact grid sums)."""
    kx, ky, Phi, lam, lamc = test_modes(n, M)
    coords = jnp.asarray(grid_coords(n))
    interior = interior_indices(n)
    kxj, kyj = jnp.asarray(kx, dtype=F64), jnp.asarray(ky, dtype=F64)
    lam_j = jnp.asarray(lam if kind == "weak" else lamc, dtype=F64)
    if colloc["kind"] == "grid":
        idx = np.asarray(colloc["idx"])
        m = idx.size
        w = jnp.asarray(colloc.get("w") if colloc.get("w") is not None else np.ones(m), dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx), "weak grid collocation must be interior nodes"
        Phi_q = jnp.asarray(Phi[pos]) * w[:, None]                  # (m, M) weighted
        xy_q = coords[jnp.asarray(idx)]
        if kind == "weak":
            st = jnp.asarray(stencil_indices(idx, n))               # (m,5)
            xy_st = coords[st.reshape(-1)]
            if dec.kind == "coord":
                vals_st = lambda z: dec(z, xy_st).reshape(m, 5)
            else:
                vals_st = lambda z: dec.rows(z, st.reshape(-1)).reshape(m, 5)
            def u_and_N(z):
                us = vals_st(z)
                c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
                dx = 1.0 / (n - 1)
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
                return c, c * (ux + uy)
            def prev_of(z):
                return vals_st(z)[:, 0]
        else:
            gsum_q = (modes_at(xy_q, kxj, kyj, n)[1].sum(-1)) * w[:, None]   # (m,M)
            if dec.kind == "coord":
                u_q = lambda z: dec(z, xy_q)
            else:
                u_q = lambda z: dec.rows(z, jnp.asarray(idx))
            prev_of = u_q
    else:
        assert kind == "weakc" and dec.kind == "coord"
        xy_q = jnp.asarray(colloc["xy"])
        m = xy_q.shape[0]
        w = jnp.asarray(colloc["w"], dtype=F64)
        val, grad = modes_at(xy_q, kxj, kyj, n)
        Phi_q = val * w[:, None]
        gsum_q = grad.sum(-1) * w[:, None]
        u_q = lambda z: dec(z, xy_q)
        prev_of = u_q

    def r_w(z, prev_c, nu):
        wt = (1.0 + DT * nu * lam_j) ** (-alpha)
        if kind == "weak":
            u, Nu = u_and_N(z)
            adv = Phi_q.T @ Nu
        else:
            u = u_q(z)
            adv = -0.5 * (gsum_q.T @ (u ** 2))
        pu = Phi_q.T @ u
        return wt * (Phi_q.T @ (u - prev_c) + DT * (adv + nu * lam_j * pu))

    def d_c(z):
        return u_and_N(z)[0] if kind == "weak" else u_q(z)

    def rJ(z, prev_c, nu):
        # Galerkin variant: test functions = mode projections of the tangent
        # basis, JD_M = Phi_q^T dD/dz (M x K)  ->  root of JD_M^T r_w = 0
        return (r_w(z, prev_c, nu), jax.jacfwd(r_w)(z, prev_c, nu),
                Phi_q.T @ jax.jacfwd(d_c)(z))

    def full(z):
        return dec(z, coords) if dec.kind == "coord" else dec.V @ z

    ops = _finish_ops(rJ, r_w, prev_of, full, m, solver)
    ops["M"] = M
    ops["tol_scale"] = float(np.sqrt(interior.size))   # mode projections of a field of RMS rho: |phi^T v| <= rho*sqrt(n_i^2)
    ops["colloc_info"] = colloc.get("info")
    return ops


# --------------------------- ROM step operators ---------------------------

def make_step_ops(dec, n, colloc, objective="fd", solver="lspg"):
    """Build jitted residual/Jacobian closures for ONE (decoder, collocation,
    objective) combination.

    Returns dict(rJ(z, prev_c, nu) -> (r_w, J_w, JD_w), rn(z, prev_c, nu),
    prev_of(z) -> prev_c [decoder at the collocation centers], full(z) -> grid
    field, m)."""
    coords = jnp.asarray(grid_coords(n))
    if objective != "fd" and not (colloc["kind"] == "grid"
                                  and colloc["idx"].size == (n - 2) ** 2):
        raise ValueError(f"objective {objective} needs colloc=full")
    obj = make_objective(objective, n, interior_indices(n))
    needs_nu = getattr(obj, "needs_nu", False)
    if colloc["kind"] == "grid":
        idx = np.asarray(colloc["idx"])
        st = jnp.asarray(stencil_indices(idx, n))               # (m,5)
        m = idx.shape[0]
        if dec.kind == "coord":
            xy_st = coords[st.reshape(-1)]                        # (5m,2)
            xy_c = coords[jnp.asarray(idx)]
            def vals_st(z):
                return dec(z, xy_st).reshape(m, 5)
            def prev_of(z):
                return dec(z, xy_c)
        else:
            def vals_st(z):
                return dec.rows(z, st.reshape(-1)).reshape(m, 5)
            def prev_of(z):
                return dec.rows(z, jnp.asarray(idx))
        def r_raw(z, prev_c, nu):
            return be_residual_from_stencil(vals_st(z), prev_c, nu, n)
        def d_c(z):
            return vals_st(z)[:, 0]
    else:
        assert dec.kind == "coord", "off-grid collocation needs the coord decoder"
        xy = jnp.asarray(colloc["xy"])
        m = xy.shape[0]
        def prev_of(z):
            return dec(z, xy)
        def r_raw(z, prev_c, nu):
            return strong_form_residual(dec, z, prev_c, nu, xy)
        def d_c(z):
            return dec(z, xy)
        assert objective == "fd", "field objectives need the full grid"

    def r_w(z, prev_c, nu):
        r = r_raw(z, prev_c, nu)
        return obj(r, nu) if needs_nu else obj(r)

    def rJ(z, prev_c, nu):
        r, J = r_w(z, prev_c, nu), jax.jacfwd(r_w)(z, prev_c, nu)
        JD = jax.jacfwd(d_c)(z)                                   # (m,K)
        if needs_nu:
            JD = jax.vmap(lambda col: obj(col, nu), in_axes=1, out_axes=1)(JD)
        elif objective != "fd":
            JD = jax.vmap(obj, in_axes=1, out_axes=1)(JD)
        return r, J, JD

    def full(z):
        if dec.kind == "coord":
            return dec(z, coords)
        return dec.V @ z

    return _finish_ops(rJ, r_w, prev_of, full, m, solver)


def _finish_ops(rJ, r_w, prev_of, full, m, solver):
    """Shared: jitted closures + on-device LM step / scan rollout."""
    rn_fn = lambda z, p, nu: jnp.linalg.norm(r_w(z, p, nu))
    rJ_lspg = lambda z, p, nu: (r_w(z, p, nu), jax.jacfwd(r_w)(z, p, nu))

    def lm_step_jit(z0, prev_c, nu, tol_abs, budget):
        """Whole LSPG time step ON DEVICE (lax.while_loop LM, same acceptance
        rule as solve_step).  Returns z, rn, n_jac, accepted, reason code
        (0 budget, 1 tol, 2 stalled, 3 lambda_max/nan, 4 tol_at_init)."""
        r0, J0 = rJ_lspg(z0, prev_c, nu)
        rn0 = jnp.linalg.norm(r0)
        K = z0.shape[0]
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where(rn0 <= tol_abs, jnp.int32(4), jnp.int32(0)))
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), jnp.int32(0), init_reason)

        def cond(s):
            return (s[9] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, nJ, _, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            tiny = finite & (jnp.linalg.norm(dz) <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
            z_new = z + jnp.where(finite, dz, 0.0)
            rn_new = rn_fn(z_new, prev_c, nu)
            accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
            r2, J2 = jax.lax.cond(accept, lambda: rJ_lspg(z_new, prev_c, nu),
                                  lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (rn <= tol_abs), 1,
                      jnp.where((accept & (rel_dec < 1e-12)) | tiny, 2,
                       jnp.where((~accept) & (lam >= 1e12), 3, 0))).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, acc, nJ, jnp.int32(0), reason)

        z, r, J, rn, lam, att, acc, nJ, _, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, nJ, acc, reason, att

    step_jit = jax.jit(lm_step_jit, static_argnums=(4,))

    def rollout_jit_fn(z0, nu, u_scales, budget):
        """Fully on-device rollout (lax.scan over the 50 steps).  u_scales
        (NUM_STEPS,) = ABSOLUTE residual tolerances per step (the caller passes
        GN_TOL * rms(u0) * sqrt(m), the same rule as rollout())."""
        def body(carry, us):
            z, prev_c = carry
            z2, rn, nJ, acc, reason, att = lm_step_jit(z, prev_c, nu, us, budget)
            return (z2, prev_of(z2)), (z2, rn, nJ, reason)
        (zT, _), (Z, rns, nJs, reasons) = jax.lax.scan(
            body, (z0, prev_of(z0)), u_scales)
        return Z, rns, nJs, reasons

    return dict(rJ=jax.jit(rJ), rn=jax.jit(rn_fn), prev_of=jax.jit(prev_of),
                full=jax.jit(full), m=m, solver=solver, step_jit=step_jit,
                rollout_jit=jax.jit(rollout_jit_fn, static_argnums=(3,)))


def solve_step(ops, z0, prev_c, nu, u_scale, budget=GN_BUDGET, tol=GN_TOL,
               lam0=1e-6):
    """One implicit time step on the manifold.  LSPG: damped LM on
    ||r_w||^2.  Galerkin: damped Newton on g = JD^T r_w = 0 with the
    Gauss-Newton Jacobian JD^T J (second derivatives of D dropped), backtracking
    on ||g||.  Returns (z, info)."""
    z = z0
    r, J, JD = ops["rJ"](z, prev_c, nu)
    rn = float(jnp.linalg.norm(r))
    n_J = 1
    n_r = 1
    lam = lam0
    acc = rej = 0
    reason = "budget"
    tol_abs = tol * float(u_scale)
    if not np.isfinite(rn):
        return z, dict(rn=rn, reason="nan_at_init", n_jac=1, n_res=1, accepted=0)
    if rn <= tol_abs:
        return z, dict(rn=rn, reason="tol_at_init", n_jac=1, n_res=1, accepted=0)
    if ops["solver"] == "lspg":
        for attempt in range(1, budget + 1):
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(H.shape[0], dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            if not bool(jnp.all(jnp.isfinite(dz))):
                lam = min(lam * 10.0, 1e12); rej += 1
                if lam >= 1e12:
                    reason = "nan_step"; break
                continue
            if float(jnp.linalg.norm(dz)) <= 1e-12 * (1.0 + float(jnp.linalg.norm(z))):
                reason = "stalled"; break
            z_new = z + dz
            rn_new = float(ops["rn"](z_new, prev_c, nu)); n_r += 1
            if np.isfinite(rn_new) and rn_new < rn:
                rel_dec = (rn - rn_new) / rn
                z, rn = z_new, rn_new
                acc += 1
                if rn <= tol_abs:
                    reason = "tol"; break
                r, J, JD = ops["rJ"](z, prev_c, nu); n_J += 1; n_r += 1
                lam = max(lam / 3.0, 1e-12)
                if rel_dec < 1e-12:
                    reason = "stalled"; break
            else:
                lam = min(lam * 10.0, 1e12); rej += 1
                if lam >= 1e12:
                    reason = "lambda_max"; break
    else:  # galerkin root
        g = JD.T @ r
        gn = float(jnp.linalg.norm(g))
        for attempt in range(1, budget + 1):
            A = JD.T @ J
            dz = jnp.linalg.solve(A + lam * jnp.eye(A.shape[0], dtype=F64), -g)
            if not bool(jnp.all(jnp.isfinite(dz))):
                lam = min(lam * 10.0, 1e12); rej += 1
                if lam >= 1e12:
                    reason = "nan_step"; break
                continue
            # backtracking on ||g||
            alpha, ok = 1.0, False
            for _ in range(8):
                z_new = z + alpha * dz
                r2, J2, JD2 = ops["rJ"](z_new, prev_c, nu); n_J += 1; n_r += 1
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
            rn = float(jnp.linalg.norm(r)); acc += 1
            lam = max(lam / 3.0, 0.0)
            if gn <= tol_abs * 1e-2 or rn <= tol_abs:
                reason = "tol"; break
            if rel_dec < 1e-12:
                reason = "stalled"; break
    return z, dict(rn=rn, reason=reason, n_jac=n_J, n_res=n_r, accepted=acc,
                   rejected=rej, final_lambda=float(lam))


def fit_ic(dec, n, u0, inits, budget=IC_BUDGET, coords=None):
    """Cold start: LM on the data misfit to the KNOWN initial condition, best of
    the given inits.  Returns (z0, rel_misfit, info)."""
    coords = jnp.asarray(grid_coords(n)) if coords is None else coords
    u0 = jnp.asarray(u0, dtype=F64)
    if dec.kind == "coord":
        f = lambda z: dec(z, coords) - u0
    else:
        return jnp.asarray(dec.V.T @ u0), float(
            jnp.linalg.norm(dec.V @ (dec.V.T @ u0) - u0) / jnp.linalg.norm(u0)), {}
    rJ = jax.jit(lambda z: (f(z), jax.jacfwd(f)(z)))
    rn = jax.jit(lambda z: jnp.linalg.norm(f(z)))
    best = None
    for name, z0 in inits.items():
        z, r, info = lm_solve(rJ, rn, jnp.asarray(z0, dtype=F64), budget)
        rel = r / float(jnp.linalg.norm(u0))
        if best is None or rel < best[1]:
            best = (z, rel, dict(info, init=name))
    return best


def rollout(dec, n, ops, z0, nu, u_scale, U_true=None, budget=GN_BUDGET,
            tol=GN_TOL):
    """Latent time stepping.  u_scale = RMS of the known u0 over the interior
    (tolerance scale).  Returns dict with latents, per-time rel-L2 (if U_true
    given), iteration counts, residual norms, wall time per step; a blow-up
    truncates the rollout (n_done < NUM_STEPS) and traj_rel is NaN."""
    z = jnp.asarray(z0, dtype=F64)
    Z = [np.asarray(z)]
    prev_c = ops["prev_of"](z)
    fields = [np.asarray(ops["full"](z))]
    iters, ress, reasons, times = [], [], [], []
    REASONS = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max", 4: "tol_at_init",
               5: "nan_at_init"}
    attempts = []
    for k in range(NUM_STEPS):
        # stopping rule: RMS residual over the m collocation points <= tol * RMS(u0)
        # (u_scale = rms of the known IC over the interior); ||r|| <= tol*rms*sqrt(m)
        # (weak form: mode projections, scale sqrt(n_i^2) -- see make_weak_ops)
        us = u_scale * ops.get("tol_scale", np.sqrt(ops["m"]))
        t0 = time.perf_counter()
        if ops["solver"] == "lspg":
            z, rn_, nJ, acc, reason, att = ops["step_jit"](z, prev_c, nu, tol * us, budget)
            info = dict(rn=float(rn_), n_jac=int(nJ), accepted=int(acc),
                        reason=REASONS[int(reason)], attempts=int(att))
        else:
            z, info = solve_step(ops, z, prev_c, nu, us, budget=budget, tol=tol)
            info["attempts"] = info.get("accepted", 0) + info.get("rejected", 0)
        prev_c = ops["prev_of"](z)
        prev_c.block_until_ready()
        times.append(time.perf_counter() - t0)
        Z.append(np.asarray(z))
        fields.append(np.asarray(ops["full"](z)))
        iters.append(info["n_jac"]); ress.append(info["rn"]); reasons.append(info["reason"])
        attempts.append(info["attempts"])
        if not np.all(np.isfinite(fields[-1])) or info["reason"] == "nan_at_init":
            reasons[-1] = "blowup"
            fields.pop(); Z.pop()
            break
    out = dict(Z=np.stack(Z), iters=iters, attempts=attempts, res=ress, reasons=reasons,
               step_time=times, n_done=len(fields) - 1)
    F = np.stack(fields)
    if U_true is not None:
        T = F.shape[0]
        Ut = np.asarray(U_true)[:T]
        per = np.linalg.norm(F - Ut, axis=1) / np.linalg.norm(Ut, axis=1)
        if T < NUM_STEPS + 1:
            per = np.concatenate([per, np.full(NUM_STEPS + 1 - T, np.nan)])
        out["per_time"] = per
        out["traj_rel"] = float(np.nanmean(per)) if T == NUM_STEPS + 1 else float("nan")
        out["traj_rel_frob"] = (float(np.linalg.norm(F - Ut) / np.linalg.norm(Ut))
                                if T == NUM_STEPS + 1 else float("nan"))
    out["fields"] = F
    return out


def time_fn(fn, reps=7, warm=2):
    """Median wall time of fn() (which must block) after `warm` warm-ups."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]
