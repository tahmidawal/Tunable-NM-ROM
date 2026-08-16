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


def build_data(n=N):
    """Regenerated from seed (identical draw to the sweep).  Returns dict."""
    U, z, cx, cy, w, a, nu = bf.build_trajectories(n)
    return dict(U=U, z=z, cx=cx, cy=cy, w=w, a=a, nu=nu)


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
        # blend: 0.5 uniform + 0.5 IC-weighted (dilated by a box blur to cover
        # the advected front which moves ~a/2 per axis over T)
        pu = np.full_like(p, 1.0 / p.size)
        P = 0.5 * pu + 0.5 * p
        pick = rng.choice(interior.size, m, replace=False, p=P)
        return dict(kind="grid", idx=np.sort(interior[pick]))
    if name.startswith("offgrid"):
        m = int(name[7:])
        xy = rng.uniform(1.0 / (n - 1), 1.0 - 1.0 / (n - 1), size=(m, 2))
        return dict(kind="offgrid", xy=xy)
    raise ValueError(name)


# --------------------------- ROM step operators ---------------------------

def make_step_ops(dec, n, colloc, objective="fd", solver="lspg"):
    """Build jitted residual/Jacobian closures for ONE (decoder, collocation,
    objective) combination.

    Returns dict(rJ(z, prev_c, nu) -> (r_w, J_w, JD_w), rn(z, prev_c, nu),
    prev_of(z) -> prev_c [decoder at the collocation centers], full(z) -> grid
    field, m)."""
    coords = jnp.asarray(grid_coords(n))
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

    rn_fn = lambda z, p, nu: jnp.linalg.norm(r_w(z, p, nu))
    rJ_lspg = lambda z, p, nu: (r_w(z, p, nu), jax.jacfwd(r_w)(z, p, nu))

    def lm_step_jit(z0, prev_c, nu, tol_abs, budget):
        """Whole LSPG time step ON DEVICE (lax.while_loop LM, same acceptance
        rule as solve_step).  Returns z, rn, n_jac, accepted, reason code
        (0 budget, 1 tol, 2 stalled, 3 lambda_max/nan, 4 tol_at_init)."""
        r0, J0 = rJ_lspg(z0, prev_c, nu)
        rn0 = jnp.linalg.norm(r0)
        K = z0.shape[0]
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), jnp.int32(0),
                jnp.where(rn0 <= tol_abs, jnp.int32(4), jnp.int32(0)))

        def cond(s):
            return (s[9] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, nJ, _, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
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
                      jnp.where(accept & (rel_dec < 1e-12), 2,
                       jnp.where((~accept) & (lam >= 1e12), 3, 0))).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, acc, nJ, jnp.int32(0), reason)

        z, r, J, rn, lam, att, acc, nJ, _, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, nJ, acc, reason

    step_jit = jax.jit(lm_step_jit, static_argnums=(4,))

    def rollout_jit_fn(z0, nu, u_scales, budget):
        """Fully on-device rollout (lax.scan over the 50 steps).  u_scales
        (NUM_STEPS,) = tolerance scales per step (we use ||u0|| for all steps
        so the scan has no data dependence on the decoded field)."""
        def body(carry, us):
            z, prev_c = carry
            z2, rn, nJ, acc, reason = lm_step_jit(z, prev_c, nu, GN_TOL * us, budget)
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


def rollout(dec, n, ops, z0, nu, U_true=None, u_scale=None, budget=GN_BUDGET,
            tol=GN_TOL):
    """Latent time stepping.  Returns dict with latents, per-time rel-L2 (if
    U_true given), iteration counts, residual norms, wall time per step."""
    z = jnp.asarray(z0, dtype=F64)
    Z = [np.asarray(z)]
    prev_c = ops["prev_of"](z)
    fields = [np.asarray(ops["full"](z))]
    iters, ress, reasons, times = [], [], [], []
    REASONS = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max", 4: "tol_at_init"}
    for k in range(NUM_STEPS):
        us = float(jnp.linalg.norm(fields[-1])) if u_scale is None else u_scale
        t0 = time.perf_counter()
        if ops["solver"] == "lspg":
            z, rn_, nJ, acc, reason = ops["step_jit"](z, prev_c, nu, tol * us, budget)
            info = dict(rn=float(rn_), n_jac=int(nJ), accepted=int(acc),
                        reason=REASONS[int(reason)])
        else:
            z, info = solve_step(ops, z, prev_c, nu, us, budget=budget, tol=tol)
        prev_c = ops["prev_of"](z)
        prev_c.block_until_ready()
        times.append(time.perf_counter() - t0)
        Z.append(np.asarray(z))
        fields.append(np.asarray(ops["full"](z)))
        iters.append(info["n_jac"]); ress.append(info["rn"]); reasons.append(info["reason"])
        if not np.all(np.isfinite(fields[-1])):
            reasons[-1] = "blowup"
            break
    out = dict(Z=np.stack(Z), iters=iters, res=ress, reasons=reasons,
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
