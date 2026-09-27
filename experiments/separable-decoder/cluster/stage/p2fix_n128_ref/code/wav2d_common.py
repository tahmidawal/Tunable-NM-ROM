"""Wave 2D — shared FOM machinery for the 2026-09-03 mechanism cell (design: WAVE2D-DESIGN.md r3).

    u_tt = c^2 lap u  on [0,1]^2,   u(.,0) = blob,  u_t(.,0) = v0 (= 0 for the family).

Two boundary conditions, treated as two CONFIGURATIONS (not one with a flag):

  bc="ref"  reflective, u = 0 on the walls.  Unknowns on the (N-2)^2 interior nodes, ghost-zero
            5-point Laplacian L_D, mass M = dx^2 I.  The (u,v) Crank-Nicolson FOM is the 08-14
            `wave2d_film.make_rollout`, reproduced here op-for-op (gate V0 checks it against the
            frozen copy in wav2d_refs/).
  bc="abs"  absorbing, first-order Engquist-Majda  u_t + c du/dn = 0, ghost-eliminated:
            u_{J+1} = u_{J-1} - 2 dx v_J / c  =>  damped ODE  u_t = v,  v_t = c^2 L_N u - c D_B v,
            L_N the ghost-reflected (Neumann-closed) Laplacian on ALL N^2 nodes, D_B = 2/dx on a
            face node, 4/dx at a corner, 0 inside.  Trapezoid mass M = M_x (x) M_y, M_x = dx *
            diag(1/2, 1, ..., 1, 1/2).  M L_N is symmetric negative semidefinite (constants in the
            kernel), and E = 1/2 v^T M v - c^2/2 u^T M L_N u satisfies  dE/dt = -c v^T M D_B v.

Conventions (binding, from the design and its audit):
  * positive-Lambda:  L Phi = - Phi Lambda   =>   A = Phi^T M L G = - Lambda B,  B = Phi^T M G.
  * every operator that the SOLVER uses (lap_D / lap_N as stencil functions) has an INDEPENDENT
    row-by-row sparse assembly (`assemble_L_independent`) used only by the gates.
  * CN on (u,v) with damping, s = c dt / 2, a = s^2:
        (I + s D_B - a L) u1 = (I + s D_B) u + dt v + a L u
        v1 = (I + s D_B)^{-1} [ (I - s D_B) v + (dt c^2 / 2) L (u + u1) ]
    (reflective: D_B = 0).  Solved by CG on the M^{1/2}-similarity-scaled operator, which is
    SPD when M(I + s D_B - a L) is (gate F0d).
  * u-only three-level (damped Newmark) recurrence, algebraically identical to the above:
        (I + s D_B - a L) u^{n+1} = 2u^n - u^{n-1} + a L (2u^n + u^{n-1}) + s D_B u^{n-1}
    first step:  (I + s D_B - a L) u^1 = (I + s D_B) u^0 + dt v0 + a L u^0.
  * energies in the M-weighted quadratic forms; the reflective one equals the 08-14 forward-
    difference form exactly (D_e^T D_e = -L_D), which gate F1a also checks.

Everything is f64.  Data as explicit jit ARGUMENTS, never closed over.
"""
from __future__ import annotations

import functools
import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

F64 = jnp.float64

# ----------------------------- frozen contract -----------------------------
T_FINAL = 1.0
NUM_STEPS = 50                      # stored snapshot intervals
SUBSTEPS = 80                       # CN substeps per stored snapshot (dt_FOM = 2.5e-4)
DT_SNAP = T_FINAL / NUM_STEPS
DT_SUB = DT_SNAP / SUBSTEPS
CG_TOL = 1e-10
CG_MAXITER = 20_000
SEED = 0
TEST_SEED = 1
N_TRAIN = 512
N_VAL = 64                          # the 08-14 draw is N_TRAIN + N_VAL; TRAIN = the first 512
N_TEST = 16


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def precond(ok, msg):
    """Precondition that survives `python -O` (assert does not)."""
    if not ok:
        raise RuntimeError("PRECOND failed: " + msg)


# ----------------------------- grid -----------------------------

@dataclass(frozen=True)
class Grid:
    N: int            # nodes per side (x_i = i dx, dx = 1/(N-1))
    bc: str           # "ref" | "abs"

    @property
    def dx(self):
        return 1.0 / (self.N - 1)

    @property
    def n(self):
        """number of unknowns per field"""
        return (self.N - 2) ** 2 if self.bc == "ref" else self.N ** 2

    @property
    def side(self):
        return self.N - 2 if self.bc == "ref" else self.N

    def coords(self):
        x = np.linspace(0.0, 1.0, self.N)
        if self.bc == "ref":
            x = x[1:-1]
        X, Y = np.meshgrid(x, x, indexing="ij")
        return X.reshape(-1), Y.reshape(-1)

    def mass_diag(self):
        """diag(M) as a (n,) array"""
        if self.bc == "ref":
            return np.full(self.n, self.dx ** 2)
        w = np.ones(self.N); w[0] = w[-1] = 0.5
        return (self.dx ** 2) * np.outer(w, w).reshape(-1)

    def damping_diag(self):
        """diag(D_B): 2/dx on a face, 4/dx at a corner, 0 inside; all zero for 'ref'."""
        if self.bc == "ref":
            return np.zeros(self.n)
        f = np.zeros(self.N); f[0] = f[-1] = 1.0
        d = (np.outer(f, np.ones(self.N)) + np.outer(np.ones(self.N), f)) * (2.0 / self.dx)
        return d.reshape(-1)

    def full_to_state(self, U_full):
        """(N,N) full-grid field -> (n,) unknown vector"""
        return (U_full[1:-1, 1:-1] if self.bc == "ref" else U_full).reshape(-1)

    def state_to_full(self, u):
        s = self.side
        U = u.reshape(s, s)
        if self.bc == "ref":
            return np.pad(np.asarray(U), 1)
        return np.asarray(U)


def _stencil_ref(u, N, dx):
    """ghost-zero 5-point Laplacian on the interior field (n_i^2,) -- jnp"""
    ni = N - 2
    U = jnp.pad(u.reshape(ni, ni), 1)
    return ((U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2]
             - 4.0 * U[1:-1, 1:-1]) / dx ** 2).reshape(-1)


def _stencil_abs(u, N, dx):
    """ghost-reflected (Neumann-closed) 5-point Laplacian on the full field (N^2,) -- jnp.
    Reflected padding: U[-1] = U[1], U[N] = U[N-2]  (the v-dependent part of the ghost is the
    D_B damping term, applied separately)."""
    U = u.reshape(N, N)
    P = jnp.pad(U, 1, mode="reflect")            # numpy 'reflect' = mirror about the edge node
    return ((P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2]
             - 4.0 * P[1:-1, 1:-1]) / dx ** 2).reshape(-1)


def lap_fn(g: Grid):
    """the SOLVER's stencil; dx is read from the grid object (so a perturbed test grid perturbs it)"""
    N, dx = g.N, g.dx
    if g.bc == "ref":
        return lambda u: _stencil_ref(u, N, dx)
    return lambda u: _stencil_abs(u, N, dx)


def assemble_L_independent(g: Grid):
    """Row-by-row sparse assembly of L_D / L_N by explicit index loops.  Used ONLY by the gates,
    as the independent path against which the stencil functions are checked (F0)."""
    N, dx, s = g.N, g.dx, g.side
    rows, cols, vals = [], [], []

    def idx(i, j):
        return i * s + j

    if g.bc == "ref":
        for i in range(s):
            for j in range(s):
                r = idx(i, j)
                rows.append(r); cols.append(r); vals.append(-4.0 / dx ** 2)
                for (ii, jj) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= ii < s and 0 <= jj < s:          # ghost-zero: outside contributes 0
                        rows.append(r); cols.append(idx(ii, jj)); vals.append(1.0 / dx ** 2)
    else:
        for i in range(N):
            for j in range(N):
                r = idx(i, j)
                rows.append(r); cols.append(r); vals.append(-4.0 / dx ** 2)
                for (ii, jj) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    # ghost reflection: an out-of-range neighbour is replaced by the mirror
                    # node across the boundary (i=-1 -> 1, i=N -> N-2)
                    if ii < 0: ii = 1
                    if ii >= N: ii = N - 2
                    if jj < 0: jj = 1
                    if jj >= N: jj = N - 2
                    rows.append(r); cols.append(idx(ii, jj)); vals.append(1.0 / dx ** 2)
    n = g.n
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ----------------------------- test modes -----------------------------

def mode_table(g: Grid, kmax: int):
    """The kmax^2 lowest-index separable modes and their positive eigenvalues, sorted by
    eigenvalue.  'ref': sin(k pi x) sin(l pi y), k,l >= 1, on the interior nodes.
    'abs': cos(k pi x) cos(l pi y), k,l >= 0, on all nodes ((0,0) has lambda = 0).
    lambda_kl = (2/dx^2) [ (1 - cos(k pi dx)) + (1 - cos(l pi dx)) ]  -- exact for both closures.
    Returns Phi (n, kmax^2) with columns of unit M-norm, lam (kmax^2,), (k,l) list."""
    N, dx = g.N, g.dx
    X, Y = g.coords()
    ks = range(1, kmax + 1) if g.bc == "ref" else range(0, kmax)
    cols, lams, kl = [], [], []
    for k in ks:
        for l in ks:
            if g.bc == "ref":
                phi = np.sin(k * np.pi * X) * np.sin(l * np.pi * Y)
            else:
                phi = np.cos(k * np.pi * X) * np.cos(l * np.pi * Y)
            lam = (2.0 / dx ** 2) * ((1 - np.cos(k * np.pi * dx)) + (1 - np.cos(l * np.pi * dx)))
            cols.append(phi); lams.append(lam); kl.append((k, l))
    Phi = np.stack(cols, axis=1)
    lam = np.array(lams)
    order = np.argsort(lam, kind="stable")
    Phi, lam = Phi[:, order], lam[order]
    kl = [kl[i] for i in order]
    m = g.mass_diag()
    Phi = Phi / np.sqrt(np.sum(m[:, None] * Phi ** 2, axis=0))[None, :]
    return Phi, lam, kl


# ----------------------------- energies -----------------------------

def energy_quadratic(g: Grid, u, v, c, lap):
    """E = 1/2 v^T M v - c^2/2 u^T M L u  (jnp; lap = the stencil function)."""
    m = jnp.asarray(g.mass_diag())
    return 0.5 * jnp.sum(m * v * v) - 0.5 * c ** 2 * jnp.sum(m * u * lap(u))


def energy_fwd_diff_ref(g: Grid, u, v, c):
    """The 08-14 form for 'ref': dx^2 [ 1/2 ||v||^2 + c^2/2 sum of squared forward differences
    over ALL edges incl. the boundary ones ].  Equals energy_quadratic exactly (D_e^T D_e = -L_D)."""
    precond(g.bc == "ref", "fwd-diff energy is the reflective form")
    N, dx = g.N, g.dx
    U = jnp.pad(u.reshape(N - 2, N - 2), 1)
    gx = (U[1:, :] - U[:-1, :]) / dx
    gy = (U[:, 1:] - U[:, :-1]) / dx
    return dx * dx * (0.5 * jnp.sum(v ** 2) + 0.5 * c ** 2 * (jnp.sum(gx ** 2) + jnp.sum(gy ** 2)))


def boundary_flux(g: Grid, vbar, c):
    """-c vbar^T M D_B vbar  (<= 0): the exact per-unit-time energy loss rate for 'abs'."""
    m = jnp.asarray(g.mass_diag()); d = jnp.asarray(g.damping_diag())
    return -c * jnp.sum(m * d * vbar * vbar)


# ----------------------------- CN FOM on (u,v), both BCs -----------------------------

def make_cn_fom(g: Grid, substeps=SUBSTEPS, cg_tol=CG_TOL, store_v=False, num_steps=NUM_STEPS):
    """Batched Crank-Nicolson rollout of the (u,v) system with per-sample c.

    rollout(U0_b, V0_b, c_b) -> (snaps, energies[, vsnaps])
      snaps    (NUM_STEPS+1, B, n)   u at the stored snapshots (U0 first)
      energies (NUM_STEPS+1, B)      E at the stored snapshots
      vsnaps   (NUM_STEPS+1, B, n)   v at the stored snapshots (if store_v)

    For 'ref' this is op-for-op the 08-14 `make_rollout` restricted to the interior unknowns
    (same rhs, same CG x0, same tolerance) -- gate V0 checks bit-agreement to CG tolerance.
    For 'abs' the damped CN system is solved by CG on the M^{1/2}-scaled operator."""
    N = g.N
    dt = DT_SNAP / substeps
    lap = lap_fn(g)
    mdiag = jnp.asarray(g.mass_diag())
    dB = jnp.asarray(g.damping_diag())
    sq = jnp.sqrt(mdiag)
    isq = 1.0 / sq
    is_abs = g.bc == "abs"

    def op(u, c):
        """(I + s D_B - a L) u"""
        s = 0.5 * dt * c
        a = s * s
        return u + s * dB * u - a * lap(u)

    def solve(rhs, c, x0):
        if not is_abs:
            A = lambda w: op(w, c)
            x, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=x0, tol=cg_tol, maxiter=CG_MAXITER)
            return x
        # symmetric form: (M^{1/2} A M^{-1/2}) y = M^{1/2} rhs,  u = M^{-1/2} y
        As = lambda y: sq * op(isq * y, c)
        y, _ = jax.scipy.sparse.linalg.cg(As, sq * rhs, x0=sq * x0, tol=cg_tol, maxiter=CG_MAXITER)
        return isq * y

    def substep_one(u, v, Lu, c):
        s = 0.5 * dt * c
        a = s * s
        rhs = u + s * dB * u + dt * v + a * Lu
        u1 = solve(rhs, c, u + dt * v)
        Lu1 = lap(u1)
        v1 = ((1.0 - s * dB) * v + 0.5 * dt * c ** 2 * (Lu + Lu1)) / (1.0 + s * dB)
        return u1, v1, Lu1

    substep_b = jax.vmap(substep_one)

    def energy_one(u, v, c):
        return energy_quadratic(g, u, v, c, lap)

    energy_b = jax.vmap(energy_one)

    @jax.jit
    def rollout(U0_b, V0_b, c_b):
        Lu0 = jax.vmap(lap)(U0_b)

        def snap_body(carry, _):
            def sub(cc, __):
                u, v, Lu = cc
                return substep_b(u, v, Lu, c_b), None
            carry, _ = jax.lax.scan(sub, carry, None, length=substeps)
            u, v, _ = carry
            out = (u, energy_b(u, v, c_b)) + ((v,) if store_v else ())
            return carry, out

        _, outs = jax.lax.scan(snap_body, (U0_b, V0_b, Lu0), None, length=num_steps)
        snaps = jnp.concatenate([U0_b[None], outs[0]], axis=0)
        e0 = energy_b(U0_b, V0_b, c_b)
        ens = jnp.concatenate([e0[None], outs[1]], axis=0)
        if store_v:
            vs = jnp.concatenate([V0_b[None], outs[2]], axis=0)
            return snaps, ens, vs
        return snaps, ens

    return rollout, op


def make_cn_fom_stepwise(g: Grid, substeps=SUBSTEPS, cg_tol=CG_TOL):
    """Single-trajectory CN rollout returning EVERY substep's (u, v) -- for the per-step energy
    identity gates (F1b, F4).  Memory: (substeps*NUM_STEPS+1, n) x 2; fine at N <= 128."""
    N = g.N
    dt = DT_SNAP / substeps
    lap = lap_fn(g)
    mdiag = jnp.asarray(g.mass_diag()); dB = jnp.asarray(g.damping_diag())
    sq = jnp.sqrt(mdiag); isq = 1.0 / sq
    is_abs = g.bc == "abs"

    def op(u, c):
        s = 0.5 * dt * c
        return u + s * dB * u - (s * s) * lap(u)

    def solve(rhs, c, x0):
        if not is_abs:
            x, _ = jax.scipy.sparse.linalg.cg(lambda w: op(w, c), rhs, x0=x0, tol=cg_tol, maxiter=CG_MAXITER)
            return x
        y, _ = jax.scipy.sparse.linalg.cg(lambda y: sq * op(isq * y, c), sq * rhs, x0=sq * x0,
                                          tol=cg_tol, maxiter=CG_MAXITER)
        return isq * y

    @functools.partial(jax.jit, static_argnums=3)
    def rollout(u0, v0, c, n_steps):
        def step(carry, _):
            u, v, Lu = carry
            s = 0.5 * dt * c; a = s * s
            rhs = u + s * dB * u + dt * v + a * Lu
            u1 = solve(rhs, c, u + dt * v)
            Lu1 = lap(u1)
            v1 = ((1.0 - s * dB) * v + 0.5 * dt * c ** 2 * (Lu + Lu1)) / (1.0 + s * dB)
            return (u1, v1, Lu1), (u1, v1)
        _, (us, vs) = jax.lax.scan(step, (u0, v0, lap(u0)), None, length=n_steps)
        return jnp.concatenate([u0[None], us]), jnp.concatenate([v0[None], vs])

    return rollout


def make_newmark_fom(g: Grid, rs, cg_tol=1e-12, num_steps=NUM_STEPS):
    """u-only three-level recurrence at dt = DT_SNAP/rs (the ROM's operator), both BCs.
    rollout(u0, v0, c) -> (snaps (NUM_STEPS+1, n), energies (NUM_STEPS+1,)) with the energy from
    the CN-CONSISTENT dynamic velocity  v_{k} = (I+sD)^{-1}[(I-sD) v_{k-1} + (dt c^2/2) L(u_{k-1}+u_k)]
    (never the kinematic recursion -- design, 'Energy reporting')."""
    dt = DT_SNAP / rs
    lap = lap_fn(g)
    mdiag = jnp.asarray(g.mass_diag()); dB = jnp.asarray(g.damping_diag())
    sq = jnp.sqrt(mdiag); isq = 1.0 / sq
    is_abs = g.bc == "abs"

    def op(u, c):
        s = 0.5 * dt * c
        return u + s * dB * u - (s * s) * lap(u)

    def solve(rhs, c, x0):
        if not is_abs:
            x, _ = jax.scipy.sparse.linalg.cg(lambda w: op(w, c), rhs, x0=x0, tol=cg_tol, maxiter=CG_MAXITER)
            return x
        y, _ = jax.scipy.sparse.linalg.cg(lambda y: sq * op(isq * y, c), sq * rhs, x0=sq * x0,
                                          tol=cg_tol, maxiter=CG_MAXITER)
        return isq * y

    def dyn_v(v, u, up, c):
        s = 0.5 * dt * c
        return ((1.0 - s * dB) * v + 0.5 * dt * c ** 2 * (lap(u) + lap(up))) / (1.0 + s * dB)

    @jax.jit
    def rollout(u0, v0, c):
        s = 0.5 * dt * c; a = s * s
        u1 = solve(u0 + s * dB * u0 + dt * v0 + a * lap(u0), c, u0 + dt * v0)
        v1 = dyn_v(v0, u0, u1, c)

        def substep(carry, _):
            um, u, v = carry
            rhs = 2.0 * u - um + a * lap(2.0 * u + um) + s * dB * um
            up = solve(rhs, c, 2.0 * u - um)
            return (u, up, dyn_v(v, u, up, c)), None

        def snap_body(carry, k):
            def full(cc):
                cc, _ = jax.lax.scan(substep, cc, None, length=rs); return cc
            def first(cc):
                cc, _ = jax.lax.scan(substep, cc, None, length=rs - 1); return cc
            carry = jax.lax.cond(k == 0, first, full, carry)
            um, u, v = carry
            return carry, (u, energy_quadratic(g, u, v, c, lap))

        _, (snaps, ens) = jax.lax.scan(snap_body, (u0, u1, v1), jnp.arange(num_steps))
        snaps = jnp.concatenate([u0[None], snaps], axis=0)
        e0 = energy_quadratic(g, u0, v0, c, lap)
        return snaps, jnp.concatenate([e0[None], ens])

    return rollout


# ----------------------------- family & data -----------------------------

def sample_params(seed=SEED, m=None):
    """The 08-14 / 08-16 draw, verbatim (same rng stream, same order of draws)."""
    rng = np.random.default_rng(seed)
    m = m or (N_TRAIN + N_VAL)
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = rng.uniform(0.05, 0.20, m)
    a = rng.uniform(1.0, 10.0, m)
    logc = rng.uniform(np.log(0.5), np.log(2.0), m)
    c = np.exp(logc)
    mu = np.stack([(cx - 0.5) / 0.35, (cy - 0.5) / 0.35, (w - 0.125) / 0.075,
                   (a - 5.5) / 4.5, logc / np.log(2.0)], axis=1)      # in [-1,1]^5 (f64 here)
    return cx, cy, w, a, c, mu


def blob_full(N, cx, cy, w, a, masked):
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    U = a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2))
    if masked:
        m = np.ones((N, N)); m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
        U = U * m
    return U


def blob_ic(g: Grid, cx, cy, w, a):
    """'ref': the 08-16 `blob_ic` (hard Dirichlet mask) restricted to the interior unknowns.
    'abs': the same blob on all nodes, unmasked."""
    return g.full_to_state(blob_full(g.N, cx, cy, w, a, masked=(g.bc == "ref")))


def compat_defect(g: Grid, u0, v0, c):
    """'abs': || v0 + c du0/dn ||_M / ||u0||_M on the faces (one-sided 2nd-order normal derivative)
    -- the startup-wave size the design says to record.  0 for 'ref'."""
    if g.bc == "ref":
        return 0.0
    N, dx = g.N, g.dx
    U = np.asarray(u0).reshape(N, N); V = np.asarray(v0).reshape(N, N)
    dn = np.zeros((N, N))
    dn[-1, :] = (3 * U[-1, :] - 4 * U[-2, :] + U[-3, :]) / (2 * dx)
    dn[0, :] = (3 * U[0, :] - 4 * U[1, :] + U[2, :]) / (2 * dx)
    dn[:, -1] += (3 * U[:, -1] - 4 * U[:, -2] + U[:, -3]) / (2 * dx)
    dn[:, 0] += (3 * U[:, 0] - 4 * U[:, 1] + U[:, 2]) / (2 * dx)
    face = np.zeros((N, N), bool); face[0, :] = face[-1, :] = face[:, 0] = face[:, -1] = True
    r = (V + c * dn)[face]
    m = g.mass_diag().reshape(N, N)[face]
    return float(np.sqrt(np.sum(m * r ** 2)) / np.sqrt(np.sum(g.mass_diag() * np.asarray(u0) ** 2)))


def build_data(g: Grid, chunk=64, n_train=N_TRAIN, n_test=N_TEST, drift_tol=1e-9):
    """TRAIN = the first n_train trajectories of the seed-0 draw; TEST = n_test fresh from
    TEST_SEED.  'ref' aborts if the relative energy drift exceeds drift_tol (the invariant is
    exact); 'abs' aborts if the energy ever INCREASES by more than drift_tol relative (it must be
    monotone non-increasing).  Any non-finite value aborts.  Returns a dict with u snapshots
    (m, T+1, n), the FOM velocities at the snapshots (needed by the auto+vc head and by G0b),
    energies, parameters and the fingerprint."""
    rollout, _ = make_cn_fom(g, store_v=True)

    def run(cx, cy, w, a, c):
        m = len(cx)
        U = np.zeros((m, NUM_STEPS + 1, g.n)); V = np.zeros_like(U); E = np.zeros((m, NUM_STEPS + 1))
        worst = 0.0
        for s_ in range(0, m, chunk):
            e_ = min(s_ + chunk, m)
            U0 = np.stack([blob_ic(g, cx[i], cy[i], w[i], a[i]) for i in range(s_, e_)])
            V0 = np.zeros_like(U0)
            snaps, ens, vs = rollout(jnp.asarray(U0), jnp.asarray(V0), jnp.asarray(c[s_:e_]))
            U[s_:e_] = np.asarray(snaps).transpose(1, 0, 2)
            V[s_:e_] = np.asarray(vs).transpose(1, 0, 2)
            ens = np.asarray(ens); E[s_:e_] = ens.T
            if g.bc == "ref":
                dm = np.abs(ens - ens[0]) / np.maximum(ens[0], 1e-300)
            else:
                dm = np.maximum(np.diff(ens, axis=0), 0.0) / np.maximum(ens[0], 1e-300)
            dmv = float(np.max(dm)) if np.all(np.isfinite(dm)) else float("nan")
            if not np.isfinite(dmv) or dmv > worst:            # NaN-propagating accumulate
                worst = dmv
        return U, V, E, worst

    cx, cy, w, a, c, mu = sample_params(SEED)
    cx, cy, w, a, c, mu = (x[:n_train] for x in (cx, cy, w, a, c, mu))
    t0 = time.time()
    U, V, E, drift_tr = run(cx, cy, w, a, c)
    log(f"  TRAIN FOM {g.bc} N={g.N}: {n_train} traj in {time.time()-t0:.0f}s, worst drift/growth {drift_tr:.2e}")
    if not np.isfinite(drift_tr) or drift_tr > drift_tol:
        raise SystemExit(f"TRAIN FOM energy check {drift_tr:.2e} > {drift_tol:.0e}")
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V))):
        raise SystemExit("non-finite training data")
    cxt, cyt, wt, at, ct, mut = sample_params(TEST_SEED, m=n_test)
    Ut, Vt, Et, drift_te = run(cxt, cyt, wt, at, ct)
    log(f"  TEST  FOM {g.bc} N={g.N}: {n_test} traj, worst drift/growth {drift_te:.2e}")
    if not np.isfinite(drift_te) or drift_te > drift_tol:
        raise SystemExit(f"TEST FOM energy check {drift_te:.2e} > {drift_tol:.0e}")
    if not (np.all(np.isfinite(Ut)) and np.all(np.isfinite(Vt))):
        raise SystemExit("non-finite test data")
    return dict(U=U, V=V, E=E, mu=mu, cx=cx, cy=cy, w=w, a=a, c=c,
                U_test=Ut, V_test=Vt, E_test=Et, mu_test=mut, cx_test=cxt, cy_test=cyt,
                w_test=wt, a_test=at, c_test=ct,
                train_energy_check=drift_tr, test_energy_check=drift_te,
                fingerprint=data_fingerprint(U), fingerprint_test=data_fingerprint(Ut))


def data_fingerprint(U):
    A = np.ascontiguousarray(np.asarray(U, dtype=np.float64))
    return dict(sum=float(np.sum(A)), sumsq=float(np.sum(A * A)), shape=list(A.shape),
                sha256=hashlib.sha256(A.tobytes()).hexdigest())


# ----------------------------- metrics -----------------------------

def traj_rms(g: Grid, F, Ut):
    """08-16 traj-RMS in the M-norm:  sqrt(mean_t ||F_t - U_t||_M^2) / sqrt(mean_t ||U_t||_M^2).
    NaN anywhere -> NaN (never silently finite)."""
    F = np.asarray(F); Ut = np.asarray(Ut)
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(Ut))):
        return float("nan")
    m = g.mass_diag()
    d2 = np.sum(m[None, :] * (F - Ut) ** 2, axis=1)
    r2 = np.sum(m[None, :] * Ut ** 2, axis=1)
    return float(np.sqrt(np.mean(d2)) / np.sqrt(np.mean(r2)))


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".",
                                       text=True).strip()
    except Exception:                      # pragma: no cover
        return "unknown"


def provenance():
    return dict(git_commit=git_commit(), jax_backend=jax.default_backend(), jax_version=jax.__version__,
                matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
                x64=bool(jax.config.jax_enable_x64))
