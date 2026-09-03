"""Wave 2D phase-1 FOM gates (WAVE2D-DESIGN.md r2, 'Phase 1'), both boundary conditions.

Every gate records value, threshold, pass, and its NEGATIVE CONTROL (value, must-fire threshold,
fired).  A gate whose control does not fire is FAIL.  NaN anywhere is FAIL.  Output: one JSON
per (N) with both BCs at runs/wav2d/wav2d_fom_gates_N{N}.json.

Usage (local GB10, sub-minute at N=64):
  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY wav2d_fom_gates.py N=64 [F2_REF=256] [F3_NS=64,128,256]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import jax
import jax.numpy as jnp

import wav2d_common as wc
from wav2d_common import Grid, log, precond

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
N = int(ARGS.get("N", "64"))
F2_REF = int(ARGS.get("F2_REF", "256"))                 # spatial reference resolution
F2_NS = [int(x) for x in ARGS.get("F2_NS", "32,64,128").split(",")]
F2_SUBS = [int(x) for x in ARGS.get("F2_SUBS", "10,20,40").split(",")]
F2_SUB_REF = int(ARGS.get("F2_SUB_REF", "320"))
F3_NS = [int(x) for x in ARGS.get("F3_NS", "64,128,256,512").split(",")]
OUT = ARGS.get("OUT", "runs/wav2d")
os.makedirs(OUT, exist_ok=True)


def finite(*xs):
    return all(np.all(np.isfinite(np.asarray(x))) for x in xs)


def gate(name, value, thr, control=None, control_thr=None, control_dir="ge", note="", aggregate="max"):
    """value must be <= thr; control must satisfy (>= control_thr if 'ge', <= if 'le')."""
    v = float(value) if value is not None else float("nan")
    ok = bool(np.isfinite(v) and v <= thr)
    rec = dict(value=v, threshold=thr, aggregate=aggregate, passed=ok, note=note)
    if control is not None:
        cv = float(control)
        fired = bool(np.isfinite(cv) and (cv >= control_thr if control_dir == "ge" else cv <= control_thr))
        rec.update(control_value=cv, control_threshold=control_thr, control_dir=control_dir, control_fired=fired)
        rec["passed"] = ok and fired
    status = "PASS" if rec["passed"] else "FAIL"
    log(f"  {name:6s} {status}  value={v:.3e} thr={thr:.0e}" +
        (f"  control={rec['control_value']:.3e} fired={rec['control_fired']}" if control is not None else "") +
        (f"  [{note}]" if note else ""))
    return rec


# ----------------------------- operators (F0) -----------------------------

def ghost_row_closed_form(g: Grid, U, V, c, i, j):
    """Closed-form ghost-eliminated Laplacian row at node (i,j) of an 'abs' grid, written out
    longhand per face/corner (the independent path for F0c).  Returns (L_N u - D_B v / c)_ij."""
    N, dx = g.N, g.dx
    tot = 0.0
    for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = i + di, j + dj
        if 0 <= ii < N and 0 <= jj < N:
            tot += (U[ii, jj] - U[i, j]) / dx ** 2
        else:                                     # ghost: mirror node minus the BC term
            mi, mj = i - di, j - dj
            tot += (U[mi, mj] - 2.0 * dx * V[i, j] / c - U[i, j]) / dx ** 2
    return tot


def gates_F0(g: Grid, c=1.3):
    out = {}
    L = wc.assemble_L_independent(g)
    lap = wc.lap_fn(g)
    rng = np.random.default_rng(0)
    u = rng.normal(size=g.n)
    # stencil (solver path) vs independent assembly
    d = np.max(np.abs(np.asarray(lap(jnp.asarray(u))) - L @ u)) / np.max(np.abs(L @ u))
    Lp = L.copy().tolil(); Lp[0, 0] *= 1 + 1e-6; Lp = Lp.tocsr()
    dp = np.max(np.abs(np.asarray(lap(jnp.asarray(u))) - Lp @ u)) / np.max(np.abs(L @ u))
    out["F0-stencil"] = gate("F0s", d, 1e-13, control=dp, control_thr=1e-9,
                             note="solver stencil vs independent row assembly; control: one coefficient perturbed 1e-6")
    # eigenvectors
    Phi, lam, kl = wc.mode_table(g, 4)
    res = np.linalg.norm(L @ Phi + Phi * lam[None, :]) / np.linalg.norm(Phi * lam[None, :])
    lam_p = lam.copy(); lam_p[-1] *= 1.01
    res_p = np.linalg.norm(L @ Phi + Phi * lam_p[None, :]) / np.linalg.norm(Phi * lam[None, :])
    if g.bc == "abs":                              # (0,0) mode: ||L phi|| directly
        j0 = kl.index((0, 0))
        res00 = np.linalg.norm(L @ Phi[:, j0]) / (np.linalg.norm(Phi[:, j0]) * lam[-1])
        out["F0b-zero-mode"] = gate("F0b0", res00, 1e-13, note="||L_N phi_00|| / (||phi_00|| lam_max)")
    out["F0a" if g.bc == "ref" else "F0b"] = gate("F0a" if g.bc == "ref" else "F0b", res, 1e-13,
                                                   control=res_p, control_thr=1e-4,
                                                   note="16 modes, closed-form lambda; control: one lambda perturbed 1%")
    m = g.mass_diag()
    ML = sp.diags(m) @ L
    sym = sp.linalg.norm(ML - ML.T) / sp.linalg.norm(ML)
    sym_I = sp.linalg.norm(L - L.T) / sp.linalg.norm(L)
    out["F0d-sym"] = gate("F0d", sym, 1e-15, control=sym_I if g.bc == "abs" else None,
                          control_thr=1e-3, note="||ML-(ML)^T||/||ML||; control (abs): M=I")
    if g.bc == "abs":
        # SPD of the absorbing step matrix at this N (dense eig at N<=64, else Lanczos min-eig)
        dt = wc.DT_SUB; s = 0.5 * dt * c
        A = sp.diags(m) @ (sp.eye(g.n) + s * sp.diags(g.damping_diag()) - (s * s) * L)
        if g.n <= 4096:
            mn = float(np.min(np.linalg.eigvalsh(A.toarray())))
        else:
            mn = float(sp.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
        out["F0d-spd"] = dict(min_eig=mn, min_eig_over_maxM=float(mn / np.max(m)), passed=bool(np.isfinite(mn) and mn > 0),
                              note="min eigenvalue of M(I + sD_B - aL_N) at dt_FOM must be > 0")
        log(f"  F0dS {'PASS' if out['F0d-spd']['passed'] else 'FAIL'}  min eig / max(M) = {mn/np.max(m):.3e}")
        # F0c: manufactured field, prescribed v, row-by-row vs closed-form ghost rows
        x = np.linspace(0, 1, g.N); X, Y = np.meshgrid(x, x, indexing="ij")
        U = X ** 2 + 2 * X * Y + 3 * Y ** 2 + 0.5 * np.sin(3 * X) * np.cos(2 * Y)
        V = np.cos(4 * X + Y) + 0.3 * X * Y
        row_solver = np.asarray(lap(jnp.asarray(U.reshape(-1)))) - g.damping_diag() * V.reshape(-1) / c
        row_solver = row_solver.reshape(g.N, g.N)
        ref = np.zeros_like(U)
        for i in range(g.N):
            for j in range(g.N):
                ref[i, j] = ghost_row_closed_form(g, U, V, c, i, j)
        err = np.max(np.abs(row_solver - ref)) / np.max(np.abs(ref))
        # control: corner coefficient 4/dx -> 2/dx
        dwrong = g.damping_diag().copy().reshape(g.N, g.N)
        for (i, j) in ((0, 0), (0, g.N - 1), (g.N - 1, 0), (g.N - 1, g.N - 1)):
            dwrong[i, j] = 2.0 / g.dx
        row_wrong = (np.asarray(lap(jnp.asarray(U.reshape(-1)))) - dwrong.reshape(-1) * V.reshape(-1) / c).reshape(g.N, g.N)
        err_w = np.max(np.abs(row_wrong - ref)) / np.max(np.abs(ref))
        out["F0c"] = gate("F0c", err, 1e-12, control=err_w, control_thr=1e-6,
                          note="face/corner ghost rows vs closed form; control: corner 4/dx -> 2/dx")
    return out


# ----------------------------- energy identities (F1, F4) -----------------------------

def make_be_fom_stepwise(g: Grid, substeps=wc.SUBSTEPS, cg_tol=1e-12):
    """Backward Euler on (u,v) -- the F1a negative control (must show drift)."""
    dt = wc.DT_SNAP / substeps
    lap = wc.lap_fn(g)

    @jax.jit
    def rollout(u0, v0, c, n_steps):
        def step(carry, _):
            u, v = carry
            A = lambda w: w - (dt * c) ** 2 * lap(w)
            u1, _ = jax.scipy.sparse.linalg.cg(A, u + dt * v, x0=u + dt * v, tol=cg_tol, maxiter=wc.CG_MAXITER)
            v1 = (u1 - u) / dt
            return (u1, v1), wc.energy_quadratic(g, u1, v1, c, lap)
        _, E = jax.lax.scan(step, (u0, v0), None, length=n_steps)
        return jnp.concatenate([wc.energy_quadratic(g, u0, v0, c, lap)[None], E])
    return rollout


def energy_trace(g: Grid, u0, v0, c, n_steps, cg_tol=1e-12, substeps=wc.SUBSTEPS):
    """CN on (u,v) returning per-step scalars only: E_n (n+1,), flux_n = -c dt vbar^T M D_B vbar (n,),
    and the control flux with v^{n+1} in place of vbar; plus the final (u,v).  Memory-light."""
    dt = wc.DT_SNAP / substeps
    lap = wc.lap_fn(g)
    mdiag = jnp.asarray(g.mass_diag()); dB = jnp.asarray(g.damping_diag())
    sq = jnp.sqrt(mdiag); isq = 1.0 / sq
    is_abs = g.bc == "abs"

    def op(u, c):
        s = 0.5 * dt * c
        return u + s * dB * u - (s * s) * lap(u)

    def solve(rhs, c, x0):
        if not is_abs:
            x, _ = jax.scipy.sparse.linalg.cg(lambda w: op(w, c), rhs, x0=x0, tol=cg_tol, maxiter=wc.CG_MAXITER)
            return x
        y, _ = jax.scipy.sparse.linalg.cg(lambda y: sq * op(isq * y, c), sq * rhs, x0=sq * x0,
                                          tol=cg_tol, maxiter=wc.CG_MAXITER)
        return isq * y

    @jax.jit
    def run(u0, v0, c, n_steps):
        def step(carry, _):
            u, v, Lu = carry
            s = 0.5 * dt * c; a = s * s
            u1 = solve(u + s * dB * u + dt * v + a * Lu, c, u + dt * v)
            Lu1 = lap(u1)
            v1 = ((1.0 - s * dB) * v + 0.5 * dt * c ** 2 * (Lu + Lu1)) / (1.0 + s * dB)
            vbar = 0.5 * (v + v1)
            E1 = wc.energy_quadratic(g, u1, v1, c, lap)
            fl = -c * dt * jnp.sum(mdiag * dB * vbar * vbar)
            fl_end = -c * dt * jnp.sum(mdiag * dB * v1 * v1)
            return (u1, v1, Lu1), (E1, fl, fl_end)
        (u, v, _), (E, fl, fle) = jax.lax.scan(step, (u0, v0, lap(u0)), None, length=n_steps)
        E0 = wc.energy_quadratic(g, u0, v0, c, lap)
        return jnp.concatenate([E0[None], E]), fl, fle, u, v
    return run


def gates_F1_F4(g: Grid, horizon_T=4.0):
    out = {}
    cx, cy, w, a, c, _ = wc.sample_params(m=3)
    i = 0                                          # one trajectory; the invariant is exact so one suffices
    u0 = jnp.asarray(wc.blob_ic(g, cx[i], cy[i], w[i], a[i])); v0 = jnp.zeros_like(u0); ci = float(c[i])
    n_steps = int(round(horizon_T / wc.DT_SUB))
    t0 = time.time()
    E, flux, flux_end, u_end, v_end = energy_trace(g, u0, v0, ci, n_steps)
    E = np.asarray(E); flux = np.asarray(flux); flux_end = np.asarray(flux_end)
    lap = wc.lap_fn(g)
    log(f"  CN trace {g.bc} N={g.N}: {n_steps} steps in {time.time()-t0:.0f}s, E0={E[0]:.4e}, E_end/E0={E[-1]/E[0]:.6f}")
    precond(finite(E, flux), "non-finite energies")
    dt = wc.DT_SUB
    if g.bc == "ref":
        drift = np.max(np.abs(E - E[0])) / E[0]
        be = make_be_fom_stepwise(g)
        Eb = np.asarray(be(u0, v0, ci, int(round(1.0 / dt))))
        drift_be = np.max(np.abs(Eb - Eb[0])) / Eb[0]
        out["F1a"] = gate("F1a", drift, 1e-10, control=drift_be, control_thr=1e-4,
                          note=f"CN relative energy drift over {horizon_T}T (CG 1e-12); control: backward Euler over T")
        e_fd = float(wc.energy_fwd_diff_ref(g, u_end, v_end, ci))
        out["F1a-form"] = gate("F1af", abs(e_fd - E[-1]) / E[-1], 1e-14, note="fwd-difference energy == quadratic form (D_e^T D_e = -L_D)")
        growth = np.max(E / E[0]) - 1.0
        out["F4"] = gate("F4", growth, 1e-10, note=f"max E^n/E^0 - 1 over {horizon_T}T")
    else:
        ident = (E[1:] - E[:-1]) - flux
        active = np.abs(flux) >= 1e-3 * E[0] * dt
        precond(active.sum() > 0, "no active-flux steps found")
        val = np.max(np.abs(ident[active])) / E[0]
        ctrl = np.max(np.abs((E[1:] - E[:-1]) - flux_end)[active]) / E[0]
        out["F1b"] = gate("F1b", val, 1e-10, control=ctrl, control_thr=1e-6,
                          note=f"E^{{n+1}}-E^n + c dt vbar^T M D_B vbar, rel E0, {int(active.sum())} active steps; control: v^{{n+1}} for vbar")
        growth = np.max(np.maximum(np.diff(E), 0.0)) / E[0]
        Ea = np.asarray(energy_trace(_AntiGrid(g), u0, v0, ci, 400)[0])
        growth_a = (np.max(Ea) - Ea[0]) / Ea[0] if finite(Ea) else float("inf")
        out["F4"] = gate("F4", growth, 1e-10, control=min(growth_a, 1e300), control_thr=1e-6,
                         note=f"max positive energy increment per step, rel E0, over {horizon_T}T; control: D_B -> -D_B (400 steps)")
        out["absorbing_energy_ratio_4T"] = float(E[-1] / E[0])
        out["absorbing_energy_ratio_T"] = float(E[int(round(1.0 / dt))] / E[0])
    return out


class _AntiGrid(Grid):
    """Grid with the damping sign flipped (negative control for F4/F5)."""
    def __new__(cls, g):
        obj = object.__new__(cls)
        object.__setattr__(obj, "N", g.N); object.__setattr__(obj, "bc", g.bc)
        return obj
    def __init__(self, g):
        pass
    def damping_diag(self):
        return -Grid.damping_diag(self)


# ----------------------------- generator spectrum (F5) -----------------------------

def gate_F5(N5=32, c=1.3):
    g = Grid(N5, "abs")
    L = wc.assemble_L_independent(g).toarray()
    D = np.diag(g.damping_diag())
    n = g.n
    Gm = np.block([[np.zeros((n, n)), np.eye(n)], [c ** 2 * L, -c * D]])
    ev = np.linalg.eigvals(Gm)
    val = float(np.max(ev.real)) / (c / g.dx)
    Ga = np.block([[np.zeros((n, n)), np.eye(n)], [c ** 2 * L, +c * D]])
    eva = np.linalg.eigvals(Ga)
    ctrl = float(np.max(eva.real)) / (c / g.dx)
    return {"F5": gate("F5", val, 1e-12, control=ctrl, control_thr=1e-3,
                       note=f"max Re(lambda)/(c/dx) of the 2n block generator at N={N5}; control: anti-damping")}


# ----------------------------- V0: reflective FOM reproduces the frozen 08-14 rollout -----------------------------

def gate_V0(N0, n_traj=4):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wav2d_refs"))
    os.environ.setdefault("N", str(N0))
    import wave2d_film_frozen_2026_08_14 as wf   # noqa
    g = Grid(N0, "ref")
    cx, cy, w, a, c, _ = wc.sample_params(m=n_traj)
    U0f = np.stack([wc.blob_full(N0, cx[i], cy[i], w[i], a[i], masked=True).reshape(-1) for i in range(n_traj)])
    wf.CG_TOL = 1e-13
    roll_f, _ = wf.make_rollout(N0)
    Sf, Ef = roll_f(jnp.asarray(U0f), jnp.asarray(c))
    Sf = np.asarray(Sf); Ef = np.asarray(Ef)
    roll, _ = wc.make_cn_fom(g, cg_tol=1e-13)
    U0 = np.stack([g.full_to_state(U0f[i].reshape(N0, N0)) for i in range(n_traj)])
    S, E = roll(jnp.asarray(U0), jnp.zeros_like(jnp.asarray(U0)), jnp.asarray(c))
    S = np.asarray(S); E = np.asarray(E)
    Sf_int = np.stack([np.stack([g.full_to_state(Sf[t, b].reshape(N0, N0)) for b in range(n_traj)]) for t in range(Sf.shape[0])])
    d = np.max(np.abs(S - Sf_int)) / np.max(np.abs(Sf_int))
    dE = np.max(np.abs(E - Ef) / Ef)
    # control: perturb one stencil coefficient by 1e-6 -> the two rollouts must separate
    g2 = _PerturbedGrid(g)
    roll2, _ = wc.make_cn_fom(g2, cg_tol=1e-13)
    S2, _ = roll2(jnp.asarray(U0), jnp.zeros_like(jnp.asarray(U0)), jnp.asarray(c))
    d2 = np.max(np.abs(np.asarray(S2) - Sf_int)) / np.max(np.abs(Sf_int))
    return {"V0": gate("V0", d, 1e-13, control=d2, control_thr=1e-7,
                       note=f"new 'ref' CN vs frozen 08-14 make_rollout, {n_traj} traj, CG 1e-13 both; energies agree to {dE:.1e}; control: dx perturbed 1e-6"),
            "V0_energy_reldiff": float(dE)}


class _PerturbedGrid(Grid):
    def __new__(cls, g):
        obj = object.__new__(cls)
        object.__setattr__(obj, "N", g.N); object.__setattr__(obj, "bc", g.bc)
        return obj
    def __init__(self, g):
        pass
    @property
    def dx(self):
        return (1.0 / (self.N - 1)) * (1 + 1e-6)




# ----------------------------- V1: u-only recurrence vs independently assembled block CN -----------------------------

def gate_V1(g: Grid, rs=wc.SUBSTEPS):
    """Independent path: dense/sparse block CN with the assembled L (scipy splu), not the stencil."""
    cx, cy, w, a, c, _ = wc.sample_params(m=2)
    i = 1
    u0 = wc.blob_ic(g, cx[i], cy[i], w[i], a[i]); v0 = np.zeros_like(u0); ci = float(c[i])
    L = wc.assemble_L_independent(g); D = sp.diags(g.damping_diag()); I = sp.eye(g.n)
    dt = wc.DT_SNAP / rs; s = 0.5 * dt * ci; aa = s * s
    A = (I + s * D - aa * L).tocsc()
    lu = sp.linalg.splu(A)
    Pinv = sp.diags(1.0 / (1.0 + s * g.damping_diag()))
    u, v = u0.copy(), v0.copy(); S_ref = [u0.copy()]
    n_int = wc.NUM_STEPS * rs
    for k in range(n_int):
        u1 = lu.solve(u + s * (g.damping_diag() * u) + dt * v + aa * (L @ u))
        v = Pinv @ ((1 - s * g.damping_diag()) * v + 0.5 * dt * ci ** 2 * (L @ (u + u1)))
        u = u1
        if (k + 1) % rs == 0:
            S_ref.append(u.copy())
    S_ref = np.array(S_ref)
    nm = wc.make_newmark_fom(g, rs, cg_tol=1e-13)
    S, E = nm(jnp.asarray(u0), jnp.asarray(v0), ci)
    S = np.asarray(S)
    d = np.max(np.abs(S - S_ref)) / np.max(np.abs(S_ref))
    if g.bc == "abs":
        ga = _AntiGrid(g)
        S2, _ = wc.make_newmark_fom(ga, rs, cg_tol=1e-13)(jnp.asarray(u0), jnp.asarray(v0), ci)
        note = "control: damping sign flipped"
    else:
        S2, _ = wc.make_newmark_fom(g, rs, cg_tol=1e-13)(jnp.asarray(u0), jnp.asarray(v0), ci * np.sqrt(1.01))
        note = "control: a -> 1.01 a (c -> 1.005 c)"
    d2 = np.max(np.abs(np.asarray(S2) - S_ref)) / np.max(np.abs(S_ref))
    return {"V1": gate("V1", d, 1e-11, control=d2, control_thr=1e-3 if g.bc == "ref" else 1e-2,
                       note=f"u-only damped Newmark (RS={rs}, CG 1e-13) vs splu block CN with assembled L; {note}")}


# ----------------------------- F2: self-convergence -----------------------------

def _restrict(Ufull_fine, Nf, Nc):
    """full-grid (Nf,Nf) field -> the coincident coarse nodes (Nc,Nc); requires (Nf-1) % (Nc-1) == 0"""
    st = (Nf - 1) // (Nc - 1)
    precond((Nf - 1) % (Nc - 1) == 0, "non-nested grids")
    return Ufull_fine[::st, ::st]


def gate_F2(bc):
    out = {}
    cx, cy, w, a, c, _ = wc.sample_params(m=2)
    i = 1
    # spatial: dt frozen at the FOM's dt, N in F2_NS vs F2_REF
    gref = Grid(F2_REF, bc)
    u0 = wc.blob_ic(gref, cx[i], cy[i], w[i], a[i])
    t0 = time.time()
    roll, _ = wc.make_cn_fom(gref, cg_tol=1e-12)
    Sref, _ = roll(jnp.asarray(u0)[None], jnp.zeros((1, gref.n)), jnp.asarray([c[i]]))
    Sref = np.asarray(Sref)[:, 0]
    log(f"  F2 {bc}: reference N={F2_REF} in {time.time()-t0:.0f}s")
    errs = []
    for Nc in F2_NS:
        g = Grid(Nc, bc)
        S, _ = wc.make_cn_fom(g, cg_tol=1e-12)[0](jnp.asarray(wc.blob_ic(g, cx[i], cy[i], w[i], a[i]))[None],
                                                  jnp.zeros((1, g.n)), jnp.asarray([c[i]]))
        S = np.asarray(S)[:, 0]
        m = g.mass_diag()
        num = den = 0.0
        for t in range(S.shape[0]):
            R = g.full_to_state(_restrict(gref.state_to_full(Sref[t]), F2_REF, Nc))
            num += np.sum(m * (S[t] - R) ** 2); den += np.sum(m * R ** 2)
        errs.append(np.sqrt(num / den))
    errs = np.array(errs)
    hs = 1.0 / (np.array(F2_NS) - 1)
    p_sp = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    out["F2-spatial"] = dict(N=F2_NS, ref=F2_REF, errors=[float(e) for e in errs], order=float(p_sp),
                             passed=bool(np.isfinite(p_sp) and abs(p_sp - 2.0) <= 0.3))
    log(f"  F2sp {bc}: errors {errs} order {p_sp:.3f} -> {'PASS' if out['F2-spatial']['passed'] else 'FAIL'}")
    # temporal: N frozen at N, SUBSTEPS in F2_SUBS vs F2_SUB_REF
    g = Grid(N, bc)
    u0 = wc.blob_ic(g, cx[i], cy[i], w[i], a[i])
    Sref, _ = wc.make_cn_fom(g, substeps=F2_SUB_REF, cg_tol=1e-12)[0](jnp.asarray(u0)[None], jnp.zeros((1, g.n)), jnp.asarray([c[i]]))
    Sref = np.asarray(Sref)[:, 0]
    errs_t = []
    for ss in F2_SUBS:
        S, _ = wc.make_cn_fom(g, substeps=ss, cg_tol=1e-12)[0](jnp.asarray(u0)[None], jnp.zeros((1, g.n)), jnp.asarray([c[i]]))
        errs_t.append(wc.traj_rms(g, np.asarray(S)[:, 0], Sref))
    errs_t = np.array(errs_t)
    dts = wc.DT_SNAP / np.array(F2_SUBS)
    p_t = np.polyfit(np.log(dts), np.log(errs_t), 1)[0]
    # control: backward Euler must read order ~1
    errs_be = []
    for ss in F2_SUBS:
        be = _be_snapshots(g, ss)
        Sb = be(jnp.asarray(u0), jnp.zeros(g.n), float(c[i]))
        errs_be.append(wc.traj_rms(g, np.asarray(Sb), Sref))
    p_be = np.polyfit(np.log(dts), np.log(np.array(errs_be)), 1)[0]
    out["F2-temporal"] = dict(SUBSTEPS=F2_SUBS, ref=F2_SUB_REF, N=N, errors=[float(e) for e in errs_t], order=float(p_t),
                              control_order_BE=float(p_be), control_fired=bool(abs(p_be - 1.0) <= 0.3),
                              passed=bool(np.isfinite(p_t) and abs(p_t - 2.0) <= 0.3 and abs(p_be - 1.0) <= 0.3))
    log(f"  F2t  {bc}: errors {errs_t} order {p_t:.3f}; BE control order {p_be:.3f} -> {'PASS' if out['F2-temporal']['passed'] else 'FAIL'}")
    return out


def _be_snapshots(g: Grid, substeps):
    dt = wc.DT_SNAP / substeps
    lap = wc.lap_fn(g)
    dB = jnp.asarray(g.damping_diag())

    @jax.jit
    def run(u0, v0, c):
        def step(carry, _):
            u, v = carry
            # BE on the damped system: u1 = u + dt v1,  v1 = v + dt (c^2 L u1 - c D v1)
            A = lambda w: (1 + dt * c * dB) * w - (dt * c) ** 2 * lap(w)
            rhs = (1 + dt * c * dB) * u + dt * v
            u1, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=u + dt * v, tol=1e-12, maxiter=wc.CG_MAXITER)
            v1 = (u1 - u) / dt
            return (u1, v1), None
        def snap(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=substeps)
            return carry, carry[0]
        _, S = jax.lax.scan(snap, (u0, v0), None, length=wc.NUM_STEPS)
        return jnp.concatenate([u0[None], S])
    return run


# ----------------------------- F3: absorber on an isolated quasi-1D pulse -----------------------------

class _XFaceGrid(Grid):
    """'abs' grid whose y-faces carry NO damping: a y-uniform pulse is then exactly compatible with
    the y-face closure (Neumann), isolating the x-face absorber.  Used only by F3."""
    def __new__(cls, N):
        obj = object.__new__(cls)
        object.__setattr__(obj, "N", N); object.__setattr__(obj, "bc", "abs")
        return obj
    def __init__(self, N):
        pass
    def damping_diag(self):
        N = self.N
        f = np.zeros(N); f[0] = f[-1] = 1.0
        return (np.outer(f, np.ones(N)) * (2.0 / self.dx)).reshape(-1)


def _pulse(N, x0, w, c):
    x = np.linspace(0, 1, N); X, Y = np.meshgrid(x, x, indexing="ij")
    U0 = np.exp(-((X - x0) ** 2) / (2 * w ** 2))
    V0 = -c * (-(X - x0) / w ** 2) * U0                     # right-going: v0 = -c du0/dx
    return U0, V0


def gate_F3(c=1.0, x0=0.45, w=0.05):
    """Reflected-energy fraction of an isolated y-uniform right-going Gaussian pulse after it has
    exited through the x=1 face (y-faces undamped so the pulse is exactly compatible with them),
    vs the discrete prediction |R|^2 ~ (15/1024)(dx/w)^4.  Energies at the stored snapshot after
    t_exit = (1 - x0 + 6w)/c."""
    out = dict(N=F3_NS, w=w, x0=x0, c=c)
    t_exit = (1.0 - x0 + 6 * w) / c
    k_exit = int(np.ceil(t_exit / wc.DT_SNAP))
    precond(k_exit <= wc.NUM_STEPS, "pulse does not exit within T")
    fracs = []; preds = []
    for Nn in F3_NS:
        g = _XFaceGrid(Nn)
        U0, V0 = _pulse(Nn, x0, w, c)
        roll, _ = wc.make_cn_fom(g, cg_tol=1e-12)
        t0 = time.time()
        _, ens = roll(jnp.asarray(U0.reshape(-1))[None], jnp.asarray(V0.reshape(-1))[None], jnp.asarray([c]))
        ens = np.asarray(ens)[:, 0]
        frac = float(ens[k_exit] / ens[0]) if finite(ens) else float("nan")
        pred = (15.0 / 1024.0) * (g.dx / w) ** 4
        fracs.append(frac); preds.append(pred)
        log(f"  F3 N={Nn}: reflected fraction {frac:.3e}, prediction {pred:.3e}  ({time.time()-t0:.0f}s)")
    fracs = np.array(fracs); preds = np.array(preds)
    hs = 1.0 / (np.array(F3_NS) - 1)
    slope = np.polyfit(np.log(hs), np.log(fracs), 1)[0] if finite(fracs) and np.all(fracs > 0) else float("nan")
    ratio = fracs / preds
    ok_slope = bool(np.isfinite(slope) and abs(slope - 4.0) <= 0.5)
    ok_pred = bool(finite(ratio) and np.all(ratio <= 2.0) and np.all(ratio >= 0.5))
    # control: reflective walls retain ~all the energy
    gR = Grid(F3_NS[0], "ref")
    U0, V0 = _pulse(gR.N, x0, w, c)
    m = np.ones((gR.N, gR.N)); m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    rollR, _ = wc.make_cn_fom(gR, cg_tol=1e-12)
    _, ensR = rollR(jnp.asarray(gR.full_to_state(U0 * m))[None], jnp.asarray(gR.full_to_state(V0 * m))[None], jnp.asarray([c]))
    ensR = np.asarray(ensR)[:, 0]
    fracR = float(ensR[k_exit] / ensR[0])
    out.update(reflected_fraction=[float(f) for f in fracs], prediction=[float(p) for p in preds],
               ratio_to_prediction=[float(r) for r in ratio], slope=float(slope), t_exit=t_exit,
               control_reflective_fraction=fracR, control_fired=bool(fracR > 0.9),
               passed=bool((ok_slope or ok_pred) and fracR > 0.9))
    log(f"  F3: slope {slope:.3f} (target 4+-0.5), ratio to prediction {ratio}, reflective control {fracR:.3f} -> {'PASS' if out['passed'] else 'FAIL'}")
    return {"F3": out}


# ----------------------------- driver -----------------------------

def main():
    t0 = time.time()
    res = dict(N=N, provenance=wc.provenance(), args=ARGS, gates={})
    log(f"phase-1 FOM gates, N={N}, backend={res['provenance']['jax_backend']}")
    for bc in ("ref", "abs"):
        g = Grid(N, bc)
        log(f"== {bc} ==")
        G = {}
        G.update(gates_F0(g))
        G.update(gates_F1_F4(g))
        G.update(gate_V1(g, rs=wc.SUBSTEPS))
        G.update(gate_F2(bc))
        res["gates"][bc] = G
    res["gates"]["ref"].update(gate_V0(N))
    res["gates"]["abs"].update(gate_F5())
    res["gates"]["abs"].update(gate_F3())
    # verdict
    allp = []
    for bc, G in res["gates"].items():
        for k, v in G.items():
            if isinstance(v, dict) and "passed" in v:
                allp.append((bc, k, v["passed"]))
    res["all_passed"] = bool(all(p for _, _, p in allp))
    res["failed"] = [f"{bc}:{k}" for bc, k, p in allp if not p]
    res["wall_s"] = time.time() - t0
    path = os.path.join(OUT, f"wav2d_fom_gates_N{N}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"phase 1 {'ALL PASS' if res['all_passed'] else 'FAILED: ' + ', '.join(res['failed'])}  ({res['wall_s']:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
