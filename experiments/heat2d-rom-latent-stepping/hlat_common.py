"""Heat-2D INR-decoder LATENT-STEPPING ROM -- shared machinery.

Port of the Burgers-2D reference (exp/2026-08-16-burgers2d-rom-latent-stepping,
`blat_common.py`) to the LINEAR heat testbed (exp/2026-08-13-heat2d-coord-decoder):

  FOM      : heat2d_film.make_rollout imported verbatim: du/dt = kappa lap u on
             [0,1]^2, u=0 walls, backward Euler dt=0.005 x 50, CG per implicit
             step (tol 1e-10).  Its implicit operator A_kappa u = u - dt kappa L u
             (interior; identity on the walls) defines the residual the ROM
             steps: R_n(u) = A_kappa u - u_n.
  decoder  : (i) hard-BC FiLM coordinate net u(x;z) = eps b(x) film(x;z), trained
             as an AUTO-DECODER (one latent per (trajectory, time) snapshot);
             (ii) POD basis V (linear control), same solver.
  residual : strong form, POINT-LOCAL (5-point stencil at m interior nodes:
             m x 5 decoder evaluations, n-free) or off-grid (autodiff Laplacian);
             WEAK form against the M lowest discrete sine modes phi_i (exact
             eigenvectors of the ghost-zero 5-point Laplacian, -L phi_i =
             lam_i phi_i):
                 R_i(z) = w_i [ phi_i^T (u - u_n) + dt kappa lam_i phi_i^T u ],
                 w_i = (1 + dt kappa lam_i)^-alpha
             which is EXACT (no advection: nothing but the decoder OUTPUT is
             needed -- no stencil, no derivatives), so hyper-reduction only has
             to quadrature phi_i^T u: NNLS-EQ weights on grid nodes (`eq<m>`) or
             on a meshfree random pool (`eqoff<m>`), fitted to decoder-output
             snapshots.  `weakc` = the same with the continuum eigenvalues
             pi^2(kx^2+ky^2) (O(h^2) from the discrete ones).
  solver   : per step, z_{n+1} from z_n by damped LM on ||R|| (LSPG, on-device
             lax.while_loop / lax.scan rollout) or damped Newton on the Galerkin
             root J_D^T R = 0; warm-started; cold start z_0 by LM data-misfit to
             the KNOWN u0 (python LM and a jitted on-device LM, both reported).
             The held-out trajectory never touches the ROM path.

PDE-agnostic pieces (POD, decoders, hard-BC factor, sine test modes, capped
Lawson-Hanson NNLS, LM step / device rollout, Galerkin root solver, cold-start
fit, rollout driver, timing helper) are IMPORTED from blat_common (validated by
the Burgers round and its Codex review); only the heat operator, data, weak
operators and EQ fitting live here.  dt and NUM_STEPS are asserted identical
between the two testbeds (both 0.005 x 50).

Env collisions: heat2d_film reads N/N_TRAIN/N_VAL/STEPS/N_FREQ/T_FREQ/HIDDEN at
import (its defaults = the sweep checkpoints' architecture); blat_common then
imports burgers2d_film (HIDDEN 256) and sets HIDDEN/N_LAYERS for ms_parametric
from AD_HIDDEN/AD_LAYERS.  Import order below is load-bearing.
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
_DEPS = os.path.join(HERE, "deps")           # cluster staging: all deps as siblings
DEP_DIRS = {
    "heat2d-coord-decoder": os.path.join(_WT, "2026-08-13-heat2d-coord-decoder", "experiments",
                                         "heat2d-coord-decoder"),
    "burgers2d-rom-latent-stepping": os.path.join(_WT, "2026-08-16-burgers2d-rom-latent-stepping",
                                                  "experiments", "burgers2d-rom-latent-stepping"),
    "burgers2d-coord-rom": os.path.join(_WT, "2026-08-14-burgers2d-coord-rom", "experiments",
                                        "burgers2d-coord-rom"),
    "multistage-precision": os.path.join(_WT, "2026-08-14-multistage-precision", "experiments",
                                         "multistage-precision"),
}
for name, cand in list(DEP_DIRS.items()):
    staged = os.path.join(_DEPS, name)
    d = cand if os.path.isdir(cand) else staged
    if not os.path.isdir(d):
        raise ImportError(f"dependency dir for {name} not found ({cand} / {staged})")
    DEP_DIRS[name] = d
    if d not in sys.path:
        sys.path.insert(0, d)
HEAT_DIR = DEP_DIRS["heat2d-coord-decoder"]

os.environ.setdefault("HIDDEN", "256")        # heat sweep + burgers sweep architecture
import heat2d_film as hf                       # noqa: E402  (reads env at import)
import blat_common as bc                       # noqa: E402  (burgers2d_film + ms_parametric)
from blat_common import (                      # noqa: E402,F401  PDE-agnostic, reused verbatim
    mp, lm_solve, F64, log, grid_coords, interior_indices, stencil_indices, pod_basis,
    bc_factor, CoordDecoder, PODDecoder, test_modes, modes_at, nnls_capped, _finish_ops,
    solve_step, fit_ic, rollout, time_fn, data_fingerprint, make_collocation,
    WEAK_ALPHA, EQ_SNAPS, EQ_POOL, K_LAT, N_TEST, GN_BUDGET, GN_TOL, IC_BUDGET, BC_MODE,
    AD_HIDDEN, AD_LAYERS, TEST_SEED)

assert hf.DT == bc.DT and hf.NUM_STEPS == bc.NUM_STEPS, "heat/burgers dt or steps differ"
assert hf.N == bc.N and hf.N_TRAIN == bc.N_TRAIN and hf.SEED == bc.SEED
assert hf.HIDDEN == 256, "heat2d_film must keep the sweep architecture"

N = hf.N
DT = hf.DT
NUM_STEPS = hf.NUM_STEPS
N_TRAIN, N_VAL = hf.N_TRAIN, hf.N_VAL
SEED = hf.SEED
CONFIG = dict(bc.CONFIG, pde="heat2d", n_train=N_TRAIN, n_val=N_VAL, seed=SEED)


# --------------------------- heat FOM pieces ---------------------------

def blob_ic(n, cx, cy, w, a):
    """The testbed's Gaussian-bump initial condition (masked to zero walls)."""
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.asarray(hf.boundary_mask(n))
    return (a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)) * mask).reshape(-1)


def make_fom(n):
    """(rollout(U0_b, kappa_b) -> (T+1,B,n^2), residual(u1,u0,kappa) -> (n^2,))
    residual = A_kappa u1 - u0 through the FOM's own implicit operator (interior
    BE rows; walls: u1 - u0)."""
    rollout_, implicit_op = hf.make_rollout(n)
    def residual(u1, u0, kap):
        return implicit_op(u1, kap) - u0
    return rollout_, residual


def max_rel_residual(U, kap, n, chunk=64):
    """Max over trajectories/steps of ||R(u_{k+1},u_k)||/||u_k|| through the FOM's
    own operator (NaN-propagating)."""
    _, res = make_fom(n)
    f = jax.jit(jax.vmap(lambda u1, u0, k_: jnp.linalg.norm(res(u1, u0, k_))
                         / jnp.linalg.norm(u0)))
    worst = 0.0
    for s in range(0, U.shape[0], chunk):
        e = min(s + chunk, U.shape[0])
        for k in range(NUM_STEPS):
            r = float(jnp.max(f(jnp.asarray(U[s:e, k + 1]), jnp.asarray(U[s:e, k]),
                                jnp.asarray(kap[s:e]))))
            if not np.isfinite(r) or r > worst:
                worst = r
    return worst


def build_data(n=N, check=True):
    """TRAIN/VAL regenerated from SEED (identical draw to the heat sweep; the sweep
    checkpoints were model-selected on VAL, so VAL is not a test set here); TEST =
    N_TEST fresh trajectories from TEST_SEED.  Aborts if any FOM residual > 1e-8
    (CG tol is 1e-10 relative)."""
    U, z, cx, cy, w, a, kap = hf.build_trajectories(n)
    cxt, cyt, wt, at, kapt, zt = hf.sample_params(seed=TEST_SEED, m=N_TEST)
    rollout_, _ = make_fom(n)
    U0 = np.stack([blob_ic(n, cxt[i], cyt[i], wt[i], at[i]) for i in range(N_TEST)])
    Ut = np.asarray(rollout_(jnp.asarray(U0), jnp.asarray(kapt))).transpose(1, 0, 2)
    d = dict(U=U, z=z, cx=cx, cy=cy, w=w, a=a, kappa=kap, U_test=Ut, z_test=zt,
             kappa_test=kapt, cx_test=cxt, cy_test=cyt, w_test=wt, a_test=at)
    if check:
        worst = max(max_rel_residual(U, kap, n), max_rel_residual(Ut, kapt, n))
        log(f"  data check: max FOM rel residual over all trajectories {worst:.2e}")
        if not np.isfinite(worst) or worst > 1e-8:
            raise SystemExit(f"FOM residual {worst:.2e} > 1e-8: data not converged")
        d["max_fom_rel_residual"] = worst
    return d


# --------------------------- residuals ---------------------------

def be_residual_from_stencil(us, up_c, kap, n):
    """Backward-Euler heat residual at m interior nodes from decoder values us
    (m,5) at the stencil [c,x+,x-,y+,y-] and up_c (m,) = previous state at the
    centers.  Equals the FOM interior residual (asserted in every run)."""
    dx = 1.0 / (n - 1)
    c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
    lap = (xp + xm + yp + ym - 4.0 * c) / dx**2
    return c - up_c - DT * kap * lap


def strong_form_residual(dec, z, up_pts, kap, xy):
    """Off-grid strong-form BE residual u - u_prev - dt kappa lap u with autodiff
    derivatives of the coord decoder (continuum operator, meshfree)."""
    def f(p):
        return dec(z, p[None, :])[0]
    def one(p):
        H = jax.hessian(f)(p)
        return f(p) - DT * kap * (H[0, 0] + H[1, 1])
    return jax.vmap(one)(xy) - up_pts


# --------------------------- WEAK form + NNLS-EQ ---------------------------

def fit_eq_weights(dec, n, M, m, Z_snap, kind="weak", pool="grid", rng=None):
    """NNLS-EQ quadrature for the heat weak form: only phi_i^T u is needed, so the
    fit rows are the mode projections of DECODER-OUTPUT snapshots u_s at the
    latents Z_snap; candidates = interior grid nodes ('grid') or EQ_POOL random
    interior points ('off', meshfree).  Returns dict(idx|xy, w, info)."""
    rng = rng or np.random.default_rng(0)
    t0 = time.time()
    kx, ky, Phi, lam, lamc = test_modes(n, M)
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
    u_full = jax.jit(lambda z: dec(z, xy_int) if dec.kind == "coord" else dec.rows(z, jnp.asarray(interior)))
    u_cand = jax.jit(lambda z: dec(z, cand_xy)) if pool == "off" else u_full
    Gs, bs = [], []
    Phi_np = np.asarray(Phi)
    for z in Z_snap:
        z = jnp.asarray(z, dtype=F64)
        uf, uc = np.asarray(u_full(z)), np.asarray(u_cand(z))
        bs.append(Phi_np.T @ uf)                     # exact grid-rule projections (M,)
        Gs.append(Phi_c.T * uc[None, :])            # (M, n_c)
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
    dict(kind='offgrid', xy, w); w = quadrature weights (None on the full grid ->
    ones = exact grid sums).  kind 'weak' uses the discrete eigenvalues (exact for
    the FOM), 'weakc' the continuum ones pi^2(kx^2+ky^2)."""
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
        if dec.kind == "coord":
            u_q = lambda z: dec(z, xy_q)
        else:
            u_q = lambda z: dec.rows(z, jnp.asarray(idx))
    else:
        assert dec.kind == "coord", "off-grid quadrature needs the coord decoder"
        xy_q = jnp.asarray(colloc["xy"])
        m = xy_q.shape[0]
        w = jnp.asarray(colloc["w"], dtype=F64)
        Phi_q = modes_at(xy_q, kxj, kyj, n)[0] * w[:, None]
        u_q = lambda z: dec(z, xy_q)

    def r_w(z, prev_c, kap):
        wt = (1.0 + DT * kap * lam_j) ** (-alpha)
        u = u_q(z)
        pu = Phi_q.T @ u
        return wt * (Phi_q.T @ (u - prev_c) + DT * kap * lam_j * pu)

    def rJ(z, prev_c, kap):
        return (r_w(z, prev_c, kap), jax.jacfwd(r_w)(z, prev_c, kap),
                Phi_q.T @ jax.jacfwd(u_q)(z))

    def full(z):
        return dec(z, coords) if dec.kind == "coord" else dec.V @ z

    ops = _finish_ops(rJ, r_w, u_q, full, m, solver)
    ops["M"] = M
    ops["tol_scale"] = float(np.sqrt(interior.size))
    ops["colloc_info"] = colloc.get("info")
    return ops


# --------------------------- strong-form step operators ---------------------------

def make_step_ops(dec, n, colloc, objective="fd", solver="lspg"):
    """Strong-form (FD residual) step operators; colloc from make_collocation."""
    assert objective == "fd", "heat port: strong form is the fd control only"
    coords = jnp.asarray(grid_coords(n))
    if colloc["kind"] == "grid":
        idx = np.asarray(colloc["idx"])
        st = jnp.asarray(stencil_indices(idx, n))
        m = idx.shape[0]
        if dec.kind == "coord":
            xy_st = coords[st.reshape(-1)]
            xy_c = coords[jnp.asarray(idx)]
            vals_st = lambda z: dec(z, xy_st).reshape(m, 5)
            prev_of = lambda z: dec(z, xy_c)
        else:
            vals_st = lambda z: dec.rows(z, st.reshape(-1)).reshape(m, 5)
            prev_of = lambda z: dec.rows(z, jnp.asarray(idx))
        r_w = lambda z, prev_c, kap: be_residual_from_stencil(vals_st(z), prev_c, kap, n)
        d_c = lambda z: vals_st(z)[:, 0]
    else:
        assert dec.kind == "coord"
        xy = jnp.asarray(colloc["xy"])
        m = xy.shape[0]
        prev_of = lambda z: dec(z, xy)
        r_w = lambda z, prev_c, kap: strong_form_residual(dec, z, prev_c, kap, xy)
        d_c = prev_of

    def rJ(z, prev_c, kap):
        return r_w(z, prev_c, kap), jax.jacfwd(r_w)(z, prev_c, kap), jax.jacfwd(d_c)(z)

    def full(z):
        return dec(z, coords) if dec.kind == "coord" else dec.V @ z

    return _finish_ops(rJ, r_w, prev_of, full, m, solver)


# --------------------------- jitted cold start ---------------------------

def make_ic_solver_jit(dec, n, coords=None):
    """On-device LM (lax.while_loop, same acceptance rule as ms_autodecoder.lm_solve)
    for the cold start z_0 = argmin ||D(z) - u0||.  Returns fn(z0, u0, budget) ->
    (z, rn, n_jac, accepted, reason, attempts); reason 1 = converged (rel
    decrease < 1e-12 or step < 1e-13), 3 = lambda_max, 0 = budget."""
    coords = jnp.asarray(grid_coords(n)) if coords is None else coords
    f = lambda z, u0: dec(z, coords) - u0
    rJ = lambda z, u0: (f(z, u0), jax.jacfwd(f)(z, u0))
    rn_fn = lambda z, u0: jnp.linalg.norm(f(z, u0))

    def solve(z0, u0, budget):
        r0, J0 = rJ(z0, u0)
        rn0 = jnp.linalg.norm(r0)
        K = z0.shape[0]
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(1), jnp.where(jnp.isfinite(rn0), jnp.int32(0), jnp.int32(5)))

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, nJ, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            rn_new = rn_fn(z_new, u0)
            accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, u0), lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), 1,
                               jnp.where((~accept) & (lam >= 1e12), 3, 0)).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, acc, nJ, reason)

        z, r, J, rn, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, nJ, acc, reason, att

    return jax.jit(solve, static_argnums=(2,))
