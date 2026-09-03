"""Wave 2D phase-1 FOM gates (WAVE2D-DESIGN.md r3, 'Phase 1'), both boundary conditions.

Every gate records value, threshold, pass, and its NEGATIVE CONTROL (value, must-fire threshold,
fired).  A gate whose control does not fire is FAIL.  NaN anywhere is FAIL.  Output: one JSON
per (N) with both BCs at runs/wav2d/wav2d_fom_gates_N{N}.json.

Usage (local GB10, sub-minute at N=64):
  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY wav2d_fom_gates.py N=64 [F2_REF=257] [F3_NS=64,128,256,512]
"""
from __future__ import annotations

import functools
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
F2_REF = int(ARGS.get("F2_REF", "257"))                 # spatial reference resolution (NESTED: N-1 multiples)
F2_NS = [int(x) for x in ARGS.get("F2_NS", "33,65,129").split(",")]
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
    # RETRACTION 2: backward-error normalisation ||L Phi + Phi Lam||_F / (||L||_2 ||Phi||_F) with
    # ||L||_2 <= 8/dx^2 -- the stencil cancels O(1) terms to produce O(lam), so the raw residual
    # relative to ||Phi Lam|| amplifies roundoff by ~1/(k pi dx)^2 and scales with N
    # mesh scale for the backward-error normalisation: sqrt(||L||_1 ||L||_inf), a valid 2-norm upper bound for the
    # non-symmetric L_N as well (verification round 2: the plain row sum is ||L||_inf, not a 2-norm bound)
    Lnorm = float(np.sqrt(np.max(np.asarray(abs(L).sum(axis=1)).ravel()) * np.max(np.asarray(abs(L).sum(axis=0)).ravel())))
    res = np.linalg.norm(L @ Phi + Phi * lam[None, :]) / (Lnorm * np.linalg.norm(Phi))
    lam_p = lam.copy(); lam_p[-1] *= 1.01
    # the control keeps the lambda-normalised form, which is N-independent (reads ~5e-3 at every N)
    res_p = np.linalg.norm(L @ Phi + Phi * lam_p[None, :]) / np.linalg.norm(Phi * lam[None, :])
    if g.bc == "abs":                              # (0,0) mode: ||L phi|| directly
        j0 = kl.index((0, 0))
        res00 = np.linalg.norm(L @ Phi[:, j0]) / (np.linalg.norm(Phi[:, j0]) * Lnorm)
        j1 = kl.index((0, 1)); phi_mix = Phi[:, j0] + 1e-3 * Phi[:, j1]
        res00_c = np.linalg.norm(L @ phi_mix) / (np.linalg.norm(phi_mix) * Lnorm)
        out["F0b-zero-mode"] = gate("F0b0", res00, 1e-13, control=res00_c, control_thr=1e-9,
                                    note="||L_N phi_00|| / (||phi_00|| sqrt(||L||_1 ||L||_inf)); control: phi_00 + 1e-3 phi_01 (decays like h^2: certified over N=64/128 only)")
    out["F0a" if g.bc == "ref" else "F0b"] = gate("F0a" if g.bc == "ref" else "F0b", res, 1e-13,
                                                   control=res_p, control_thr=1e-4,
                                                   note="16 modes, closed-form lambda, backward-error normalised by sqrt(||L||_1 ||L||_inf) >= ||L||_2 (retraction 2); control: one lambda perturbed 1%, lambda-normalised (N-independent)")
    m = g.mass_diag()
    ML = sp.diags(m) @ L
    sym = sp.linalg.norm(ML - ML.T) / sp.linalg.norm(ML)
    sym_I = sp.linalg.norm(L - L.T) / sp.linalg.norm(L)
    Lo = L.copy().tolil(); Lo[0, 1] *= 1.5; Lo = Lo.tocsr()                # one asymmetric coefficient
    sym_o = sp.linalg.norm(sp.diags(m) @ Lo - (sp.diags(m) @ Lo).T) / sp.linalg.norm(ML)
    out["F0d-sym"] = gate("F0d", sym, 1e-15, control=(sym_I if g.bc == "abs" else sym_o),
                          control_thr=1e-6, note="||ML-(ML)^T||/||ML||; control: M=I (abs) / one asymmetric coefficient (ref)")
    if g.bc == "abs":
        # SPD of the absorbing step matrix at this N (dense eig at N<=64, else Lanczos min-eig)
        dt = wc.DT_SUB; s = 0.5 * dt * c
        A = sp.diags(m) @ (sp.eye(g.n) + s * sp.diags(g.damping_diag()) - (s * s) * L)
        if g.n <= 4096:
            mn = float(np.min(np.linalg.eigvalsh(A.toarray())))
        else:
            mn = float(sp.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
        g32 = Grid(32, "abs"); m32 = g32.mass_diag(); L32 = wc.assemble_L_independent(g32)
        A32 = sp.diags(m32) @ (sp.eye(g32.n) + s * sp.diags(g32.damping_diag()) - (s * s) * L32)
        mn32 = float(np.min(np.linalg.eigvalsh(A32.toarray())))
        out["F0d-spd"] = dict(min_eig=mn, min_eig_over_maxM=float(mn / np.max(m)), min_eig_over_maxM_N32=float(mn32 / np.max(m32)),
                              passed=bool(np.isfinite(mn) and mn > 0 and np.isfinite(mn32) and mn32 > 0),
                              note="min eigenvalue of M(I + sD_B - aL_N) at dt_FOM must be > 0 (this N and the design's N=32); an eigenvalue fact, no mutation control")
        log(f"  F0dS {'PASS' if out['F0d-spd']['passed'] else 'FAIL'}  min eig / max(M) = {mn/np.max(m):.3e} (N=32: {mn32/np.max(m32):.3e})")
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
        # second control: one face L_N coefficient 2/dx^2 -> 1/dx^2 (the mirror-node weight at node (N-1, 3))
        Lm = wc.assemble_L_independent(g).tolil()
        r_ = (g.N - 1) * g.N + 3; Lm[r_, (g.N - 2) * g.N + 3] = 1.0 / g.dx ** 2
        row_wrong2 = (Lm.tocsr() @ U.reshape(-1) - g.damping_diag() * V.reshape(-1) / c).reshape(g.N, g.N)
        err_w2 = np.max(np.abs(row_wrong2 - ref)) / np.max(np.abs(ref))
        out["F0c"] = gate("F0c", err, 1e-12, control=min(err_w, err_w2), control_thr=1e-6,
                          note=f"full ghost row L_N u - D_B v/c vs longhand closed form; controls: corner 4/dx->2/dx ({err_w:.1e}), face 2/dx^2->1/dx^2 ({err_w2:.1e}), both must fire")
    return out


# ----------------------------- energy identities (F1, F4) -----------------------------

def make_be_fom_stepwise(g: Grid, substeps=wc.SUBSTEPS, cg_tol=1e-12):
    """Backward Euler on (u,v) -- the F1a negative control (must show drift)."""
    dt = wc.DT_SNAP / substeps
    lap = wc.lap_fn(g)

    @functools.partial(jax.jit, static_argnums=3)
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
    def run(u0, v0, c):
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
    return run(jnp.asarray(u0), jnp.asarray(v0), c)


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
    precond(finite(E, flux), "non-finite energies in the MAIN trace (the gate cannot be read)")
    log(f"  CN trace {g.bc} N={g.N}: {n_steps} steps in {time.time()-t0:.0f}s, E0={E[0]:.4e}, E_end/E0={E[-1]/E[0]:.6f}")
    dt = wc.DT_SUB
    if g.bc == "ref":
        drift = np.max(np.abs(E - E[0])) / E[0]
        be = make_be_fom_stepwise(g)
        Eb = np.asarray(be(u0, v0, ci, int(round(1.0 / dt))))
        drift_be = np.max(np.abs(Eb - Eb[0])) / Eb[0]
        out["F1a"] = gate("F1a", drift, 1e-10, control=drift_be, control_thr=1e-4,
                          note=f"CN relative energy drift over {horizon_T}T (CG 1e-12); control: backward Euler over T")
        e_fd = float(wc.energy_fwd_diff_ref(g, u_end, v_end, ci))
        # control: the forward-difference sum WITHOUT the boundary edges (D_e^T D_e != -L_D then)
        Uf = np.pad(np.asarray(u_end).reshape(g.N - 2, g.N - 2), 1); dxg = g.dx
        gx = (Uf[1:, :] - Uf[:-1, :])[1:-1, :] / dxg; gy = (Uf[:, 1:] - Uf[:, :-1])[:, 1:-1] / dxg
        e_fd_c = dxg * dxg * (0.5 * np.sum(np.asarray(v_end) ** 2) + 0.5 * ci ** 2 * (np.sum(gx ** 2) + np.sum(gy ** 2)))
        out["F1a-form"] = gate("F1af", abs(e_fd - E[-1]) / E[-1], 1e-14, control=abs(e_fd_c - E[-1]) / E[-1], control_thr=1e-6,
                               note="fwd-difference energy == quadratic form (D_e^T D_e = -L_D); control: boundary edges dropped from the sum")
        growth = np.max(E / E[0]) - 1.0
        out["F4"] = gate("F4", growth, 1e-10, control=drift_be, control_thr=1e-4,
                         note=f"max E^n/E^0 - 1 over {horizon_T}T (implied by F1a; same BE control)")
    else:
        ident = (E[1:] - E[:-1]) - flux
        active = np.abs(flux) >= 1e-3 * E[0] * dt
        precond(active.sum() > 0, "no active-flux steps found")
        val = np.max(np.abs(ident[active])) / E[0]
        ctrl = np.max(np.abs((E[1:] - E[:-1]) - flux_end)[active]) / E[0]
        out["F1b"] = gate("F1b", val, 1e-10, control=ctrl, control_thr=1e-6,
                          note=f"E^{{n+1}}-E^n + c dt vbar^T M D_B vbar, rel E0, {int(active.sum())} active steps; control: v^{{n+1}} for vbar")
        growth = np.max(np.maximum(np.diff(E), 0.0)) / E[0]
        Ea = np.asarray(energy_trace(_AntiGrid(g), u0, v0, ci, 400)[0])       # diagnostic run: no finiteness precondition
        # the anti-damped run may overflow: the control is read from its FINITE PREFIX only (max growth over the finite
        # steps), and a nonfinite tail is recorded, never converted into a value
        fin = np.isfinite(Ea); n_fin = int(np.argmin(fin)) if not fin.all() else len(Ea)
        growth_a = float((np.max(Ea[:n_fin]) - Ea[0]) / Ea[0]) if n_fin > 1 else float("nan")
        out["F4"] = gate("F4", growth, 1e-10, control=growth_a, control_thr=1e-6,
                         note=f"max positive energy increment per step, rel E0, over {horizon_T}T; control: D_B -> -D_B, max growth over its first {n_fin} finite steps of 400 (nonfinite tail: {not fin.all()})")
        out["F4"]["control_nonfinite_tail"] = bool(not fin.all()); out["F4"]["control_finite_steps"] = n_fin
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
    import importlib.util
    os.environ.setdefault("N", str(N0))
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wav2d_refs", "wave2d_film_frozen_2026-08-14.py")
    spec = importlib.util.spec_from_file_location("wave2d_film_frozen", fpath)
    wf = importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
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
                       note=f"new 'ref' CN vs frozen 08-14 make_rollout, {n_traj} traj, CG 1e-13 both; energies agree to {dE:.1e}; control: dx perturbed 1e-6 in the stencil (every Laplacian coefficient; the mass also changes but only the state is compared)"),
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
    """Three paths, two comparisons:
      (i)  BLOCK CN: the 2n x 2n system  [[I, -(dt/2)I],[-(dt/2)c^2 L, I+(dt/2)cD]] w1 = [[I,(dt/2)I],[(dt/2)c^2 L, I-(dt/2)cD]] w0
           solved by splu on the assembled L -- the (u,v) scheme with NO elimination;
      (ii) the u-only Newmark recurrence solved by LU (no iterative error): must agree with (i) to roundoff
           -> certifies the ELIMINATION ALGEBRA, tolerance-free;
      (iii) the JAX/CG recurrence (the solver path) at CG tol 1e-9, 1e-11, 1e-13: must converge monotonically
           toward (ii) -> certifies the implementation is the same recurrence, solver-limited.
    Controls: damping sign flipped (abs) / a -> 1.01a (ref) on path (ii) vs (i) must separate."""
    cx, cy, w, a, c, _ = wc.sample_params(m=2)
    i = 1
    u0 = wc.blob_ic(g, cx[i], cy[i], w[i], a[i]); v0 = np.zeros_like(u0); ci = float(c[i])
    n = g.n
    L = wc.assemble_L_independent(g).tocsc(); dB = g.damping_diag(); D = sp.diags(dB); I = sp.eye(n)
    dt = wc.DT_SNAP / rs; s_ = 0.5 * dt * ci; aa = s_ * s_
    n_int = wc.NUM_STEPS * rs
    # (i) block CN
    Ablk = sp.bmat([[I, -(dt / 2) * I], [-(dt / 2) * ci ** 2 * L, I + (dt / 2) * ci * D]]).tocsc()
    Bblk = sp.bmat([[I, (dt / 2) * I], [(dt / 2) * ci ** 2 * L, I - (dt / 2) * ci * D]]).tocsr()
    lu_blk = sp.linalg.splu(Ablk)
    wv = np.concatenate([u0, v0]); S_blk = [u0.copy()]; S_blk10 = [u0.copy()]
    for k in range(n_int):
        wv = lu_blk.solve(Bblk @ wv)
        if k < 10: S_blk10.append(wv[:n].copy())
        if (k + 1) % rs == 0:
            S_blk.append(wv[:n].copy())
    S_blk = np.array(S_blk); S_blk10 = np.array(S_blk10)
    # (ii) Newmark recurrence by LU
    def newmark_lu(Lm, dBm, cc):
        s2 = 0.5 * dt * cc; a2 = s2 * s2
        A = (I + s2 * sp.diags(dBm) - a2 * Lm).tocsc(); lu = sp.linalg.splu(A)
        um = u0.copy()
        u = lu.solve(um + s2 * dBm * um + dt * v0 + a2 * (Lm @ um))
        out = [u0.copy()]; out10 = [u0.copy(), u.copy()]
        k = 1
        if k % rs == 0: out.append(u.copy())
        while k < n_int:
            up = lu.solve(2.0 * u - um + a2 * (Lm @ (2.0 * u + um)) + s2 * dBm * um)
            um, u = u, up; k += 1
            if k <= 10: out10.append(u.copy())
            if k % rs == 0: out.append(u.copy())
        return np.array(out), np.array(out10)
    S_lu, S_lu10 = newmark_lu(L, dB, ci)
    scale = np.max(np.abs(S_blk))
    d_alg_full = np.max(np.abs(S_lu - S_blk)) / scale                  # reported: LU roundoff over 4000 solves
    d_alg = np.max(np.abs(S_lu10 - S_blk10)) / np.max(np.abs(S_blk10))  # GATED: first 10 steps, roundoff ~1e-14
    # control on the algebra comparison
    if g.bc == "abs":
        _, S_ctrl10 = newmark_lu(L, -dB, ci); note = "control: damping sign flipped in the recurrence"
    else:
        _, S_ctrl10 = newmark_lu(L, dB, ci * np.sqrt(1.01)); note = "control: a -> 1.01 a in the recurrence"
    d_ctrl = np.max(np.abs(S_ctrl10 - S_blk10)) / np.max(np.abs(S_blk10))
    rec_alg = gate("V1alg", d_alg, 1e-13, control=d_ctrl, control_thr=1e-7,
                   note=f"u-only Newmark by LU vs the 2n x 2n block CN by LU over the FIRST 10 STEPS (empirical LU-roundoff certification; the mutation controls show what an algebraic error reads); "
                        f"full {n_int}-step LU-vs-LU discrepancy {d_alg_full:.2e} reported (LU roundoff accumulation, retraction 4); {note}")
    rec_alg["value_full_horizon"] = float(d_alg_full)
    # (iii) CG ladder on the solver path
    ladder = {}
    for tol in (1e-9, 1e-11, 1e-13):
        S_cg, _ = wc.make_newmark_fom(g, rs, cg_tol=tol)(jnp.asarray(u0), jnp.asarray(v0), ci)
        ladder[tol] = float(np.max(np.abs(np.asarray(S_cg) - S_lu)) / scale)
    vals = [ladder[t] for t in (1e-9, 1e-11, 1e-13)]
    monotone = bool(vals[0] > vals[1] > vals[2])
    # achieved CG residual at one representative solve (the last stored state as rhs surrogate)
    lap = wc.lap_fn(g)
    A_op = lambda x: np.asarray(x + s_ * dB * x - aa * np.asarray(lap(jnp.asarray(x))))
    rhs = S_lu[-1] + s_ * dB * S_lu[-1] + dt * 0.0 + aa * np.asarray(lap(jnp.asarray(S_lu[-1])))
    sq = np.sqrt(g.mass_diag()); isq = 1 / sq
    Aj = (lambda y: jnp.sqrt(jnp.asarray(g.mass_diag())) * (isq * y + s_ * dB * (isq * y) - aa * lap(isq * y))) if g.bc == "abs" else \
         (lambda y: y + s_ * dB * y - aa * lap(y))
    yy, _ = jax.scipy.sparse.linalg.cg(Aj, jnp.asarray(sq * rhs if g.bc == "abs" else rhs), tol=1e-13, maxiter=wc.CG_MAXITER)
    x_cg = np.asarray(isq * yy) if g.bc == "abs" else np.asarray(yy)
    cg_resid = float(np.linalg.norm(A_op(x_cg) - rhs) / np.linalg.norm(rhs))
    rec_cg = dict(value=vals[2], threshold=1e-8, ladder={str(k): v for k, v in ladder.items()}, monotone=monotone,
                  achieved_cg_relresid_1e13=cg_resid,
                  passed=bool(np.isfinite(vals[2]) and vals[2] <= 1e-8 and monotone and cg_resid <= 1e-12),
                  note="JAX/CG recurrence vs the LU recurrence at CG tol 1e-9/1e-11/1e-13: must decrease monotonically and reach <= 1e-8; "
                       "achieved relative CG residual of ONE representative surrogate solve at tol 1e-13 recorded (<= 1e-12; not per-step monitoring). RETRACTIONS 1+3 superseded: the algebra is certified by V1alg.")
    log(f"  V1cg  {'PASS' if rec_cg['passed'] else 'FAIL'}  ladder {vals} monotone={monotone} achieved CG resid {cg_resid:.1e}")
    return {"V1alg": rec_alg, "V1cg": rec_cg}


# ----------------------------- F2: self-convergence -----------------------------

def _restrict(Ufull_fine, Nf, Nc):
    """full-grid (Nf,Nf) field -> the coincident coarse nodes (Nc,Nc); requires (Nf-1) % (Nc-1) == 0"""
    st = (Nf - 1) // (Nc - 1)
    precond((Nf - 1) % (Nc - 1) == 0, "non-nested grids")
    return Ufull_fine[::st, ::st]


def _spatial_order(bc, ic_fn, ci, label):
    """errors of N in F2_NS vs the F2_REF reference on the coincident (nested) nodes, M-norm traj-RMS"""
    gref = Grid(F2_REF, bc)
    Sref, _ = wc.make_cn_fom(gref, cg_tol=1e-12)[0](jnp.asarray(ic_fn(gref))[None], jnp.zeros((1, gref.n)), jnp.asarray([ci]))
    Sref = np.asarray(Sref)[:, 0]
    errs = []
    for Nc in F2_NS:
        g = Grid(Nc, bc)
        S, _ = wc.make_cn_fom(g, cg_tol=1e-12)[0](jnp.asarray(ic_fn(g))[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))
        S = np.asarray(S)[:, 0]
        m = g.mass_diag(); num = den = 0.0
        for t in range(S.shape[0]):
            R = g.full_to_state(_restrict(gref.state_to_full(Sref[t]), F2_REF, Nc))
            num += np.sum(m * (S[t] - R) ** 2); den += np.sum(m * R ** 2)
        errs.append(np.sqrt(num / den))
    errs = np.array(errs)
    hs = 1.0 / (np.array(F2_NS) - 1)
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0] if finite(errs) and np.all(errs > 0) else float("nan")
    log(f"  F2sp {bc} [{label}]: errors {errs} order {p:.3f}")
    return errs, float(p)


def _spatial_order_wrongref(bc, ic_fn, ci):
    """negative control for the spatial study: coarse solutions vs a reference run at 1.01 c"""
    gref = Grid(F2_REF, bc)
    Sref, _ = wc.make_cn_fom(gref, cg_tol=1e-12)[0](jnp.asarray(ic_fn(gref))[None], jnp.zeros((1, gref.n)), jnp.asarray([1.01 * ci]))
    Sref = np.asarray(Sref)[:, 0]
    errs = []
    for Nc in F2_NS:
        g = Grid(Nc, bc)
        S, _ = wc.make_cn_fom(g, cg_tol=1e-12)[0](jnp.asarray(ic_fn(g))[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))
        S = np.asarray(S)[:, 0]; m = g.mass_diag(); num = den = 0.0
        for t in range(S.shape[0]):
            Rr = g.full_to_state(_restrict(gref.state_to_full(Sref[t]), F2_REF, Nc))
            num += np.sum(m * (S[t] - Rr) ** 2); den += np.sum(m * Rr ** 2)
        errs.append(np.sqrt(num / den))
    errs = np.array(errs); hs = 1.0 / (np.array(F2_NS) - 1)
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0] if finite(errs) and np.all(errs > 0) else float("nan")
    return errs, float(p)


def gate_F2(bc):
    out = {}
    cx, cy, w, a, c, _ = wc.sample_params(m=16)
    i = int(np.argmax(w))
    ci = float(c[i])
    # (a) the GATE: a smooth, wall-compatible centred bump (w=0.1 -> wall values ~4e-6 of the peak),
    #     so the spatial study measures the scheme's order, not the family's wall discontinuity
    smooth = lambda g: wc.blob_ic(g, 0.5, 0.5, 0.1, 1.0)
    errs, p_sp = _spatial_order(bc, smooth, ci, "smooth centred bump w=0.1")
    # an EXACTLY compatible initial datum: a sum of two eigenmodes of the closure (v0 = 0 and, for
    # 'abs', d u0/dn = 0 exactly) -- its spatial error is the eigenvalue's O(dx^2) phase error
    def mode_ic(g):
        Phi, lam, kl = wc.mode_table(g, 4)
        j1 = kl.index((1, 1)); j2 = kl.index((2, 3)) if (2, 3) in kl else kl.index((2, 1))
        return Phi[:, j1] + 0.5 * Phi[:, j2]
    errs_m, p_m = _spatial_order(bc, mode_ic, ci, "exactly compatible two-mode sum")
    # control: the SAME coarse solutions against a WRONG reference (c -> 1.01 c): the 'error' saturates
    errs_w, p_w = _spatial_order_wrongref(bc, smooth, ci)
    out["F2-spatial"] = dict(N=F2_NS, ref=F2_REF, ic="centred bump w=0.1 (wall-compatible) AND exactly compatible two-mode sum", c=ci,
                             errors=[float(e) for e in errs], order=p_sp, errors_modes=[float(e) for e in errs_m], order_modes=p_m,
                             control_errors_wrongref=[float(e) for e in errs_w], control_order_wrongref=p_w,
                             control_fired=bool(np.isfinite(p_w) and abs(p_w - 2.0) > 0.7),
                             passed=bool(np.isfinite(p_sp) and abs(p_sp - 2.0) <= 0.3 and np.isfinite(p_m) and abs(p_m - 2.0) <= 0.3
                                         and np.isfinite(p_w) and abs(p_w - 2.0) > 0.7))
    log(f"  F2sp {bc}: bump order {p_sp:.3f}, modes order {p_m:.3f}, wrong-reference control order {p_w:.3f} -> {'PASS' if out['F2-spatial']['passed'] else 'FAIL'}")
    # (b) REPORTED, not gated: the family's own widest blob.  The inherited family multiplies the
    #     blob by a hard wall mask (ref) or leaves du/dn != 0 with v0 = 0 (abs), so a wide near-wall
    #     blob carries a 1-cell wall discontinuity / a startup wave -> reduced order is a DATA
    #     property, recorded here so the manifold discussion sees it
    fam = lambda g: wc.blob_ic(g, cx[i], cy[i], w[i], a[i])
    errs_f, p_f = _spatial_order(bc, fam, ci, f"family widest blob w={w[i]:.3f} cx={cx[i]:.2f} cy={cy[i]:.2f}")
    out["F2-spatial-family-reported"] = dict(index=i, w=float(w[i]), cx=float(cx[i]), cy=float(cy[i]), a=float(a[i]), c=ci,
                                             errors=[float(e) for e in errs_f], order=p_f,
                                             note="not gated: the family's wall treatment limits the order; see WAVE2D-NOTES")
    # temporal: N frozen at N, SUBSTEPS in F2_SUBS vs F2_SUB_REF, on the SMOOTH bump (amendment 4);
    # the family blob's wall discontinuity excites the high modes whose CN phase error dominates at
    # coarse dt, and more so at larger N, so the family reads pre-asymptotic (reported below)
    g = Grid(N, bc)
    u0 = smooth(g)
    Sref, _ = wc.make_cn_fom(g, substeps=F2_SUB_REF, cg_tol=1e-12)[0](jnp.asarray(u0)[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))
    Sref = np.asarray(Sref)[:, 0]
    errs_t = []
    for ss in F2_SUBS:
        S, _ = wc.make_cn_fom(g, substeps=ss, cg_tol=1e-12)[0](jnp.asarray(u0)[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))
        errs_t.append(wc.traj_rms(g, np.asarray(S)[:, 0], Sref))
    errs_t = np.array(errs_t)
    dts = wc.DT_SNAP / np.array(F2_SUBS)
    p_t = np.polyfit(np.log(dts), np.log(errs_t), 1)[0]
    # control: backward Euler must read order ~1
    errs_be = []
    for ss in F2_SUBS:
        be = _be_snapshots(g, ss)
        Sb = be(jnp.asarray(u0), jnp.zeros(g.n), ci)
        errs_be.append(wc.traj_rms(g, np.asarray(Sb), Sref))
    p_be = np.polyfit(np.log(dts), np.log(np.array(errs_be)), 1)[0]
    sep = float(errs_be[-1] / errs_t[-1])          # BE must be far worse than CN at the finest dt
    ctrl_ok = bool(np.isfinite(sep) and sep >= 10.0 and np.isfinite(p_be) and abs(p_be - 1.0) <= 0.3)
    out["F2-temporal"] = dict(SUBSTEPS=F2_SUBS, ref=F2_SUB_REF, N=N, errors=[float(e) for e in errs_t], order=float(p_t),
                              control_errors_BE=[float(e) for e in errs_be], control_order_BE=float(p_be),
                              control_separation=sep, control_fired=ctrl_ok,
                              passed=bool(np.isfinite(p_t) and abs(p_t - 2.0) <= 0.3 and ctrl_ok),
                              note="control: backward Euler must read order 1 +- 0.3 AND be >= 10x worse at the finest step (amendment 3 restored to both)")
    log(f"  F2t  {bc}: errors {errs_t} order {p_t:.3f}; BE control errors {np.array(errs_be)} (order {p_be:.2f}), separation {sep:.1f}x -> {'PASS' if out['F2-temporal']['passed'] else 'FAIL'}")
    if bc == "abs":
        _, ensb = wc.make_cn_fom(g, cg_tol=1e-12)[0](jnp.asarray(u0)[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))
        ensb = np.asarray(ensb)[:, 0]
        out["F2-absorbing-energy-ratio-T"] = float(ensb[-1] / ensb[0])
        log(f"  F2 abs: the smooth-bump study trajectory retains E(T)/E0 = {ensb[-1]/ensb[0]:.3e} (the absorber IS exercised iff this is << 1)")
    # reported: the family blob's temporal order at this N
    u0f = fam(g)
    Sref_f, _ = wc.make_cn_fom(g, substeps=F2_SUB_REF, cg_tol=1e-12)[0](jnp.asarray(u0f)[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))
    errs_tf = [wc.traj_rms(g, np.asarray(wc.make_cn_fom(g, substeps=ss, cg_tol=1e-12)[0](jnp.asarray(u0f)[None], jnp.zeros((1, g.n)), jnp.asarray([ci]))[0])[:, 0], np.asarray(Sref_f)[:, 0]) for ss in F2_SUBS]
    p_tf = np.polyfit(np.log(dts), np.log(np.array(errs_tf)), 1)[0]
    out["F2-temporal-family-reported"] = dict(errors=[float(e) for e in errs_tf], order=float(p_tf), note="not gated: family widest blob")
    log(f"  F2t  {bc} [family widest blob, reported]: errors {np.array(errs_tf)} order {p_tf:.3f}")
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
    fracs = []; preds = []; diag = []
    for Nn in F3_NS:
        g = _XFaceGrid(Nn)
        U0, V0 = _pulse(Nn, x0, w, c)
        roll, _ = wc.make_cn_fom(g, cg_tol=1e-12)
        t0 = time.time()
        snaps, ens = roll(jnp.asarray(U0.reshape(-1))[None], jnp.asarray(V0.reshape(-1))[None], jnp.asarray([c]))
        ens = np.asarray(ens)[:, 0]; snaps = np.asarray(snaps)[:, 0]
        frac = float(ens[k_exit] / ens[0]) if finite(ens) else float("nan")
        pred = (15.0 / 1024.0) * (g.dx / w) ** 4
        # corroboration: plateau (fraction two snapshots later), y-invariance of the final field, mean-field remainder
        plateau = float(ens[min(k_exit + 2, wc.NUM_STEPS)] / ens[0])
        Uf = snaps[k_exit].reshape(Nn, Nn); yvar = float(np.max(np.abs(Uf - Uf[:, :1])) / max(np.max(np.abs(snaps[0])), 1e-300))
        meanfield = float(np.mean(Uf))
        diag.append(dict(N=Nn, plateau_fraction=plateau, y_invariance=yvar, mean_field_remainder=meanfield))
        fracs.append(frac); preds.append(pred)
        log(f"  F3 N={Nn}: reflected fraction {frac:.3e}, prediction {pred:.3e}, plateau {plateau:.3e}, y-var {yvar:.1e}, mean {meanfield:.1e}  ({time.time()-t0:.0f}s)")
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
               ratio_to_prediction=[float(r) for r in ratio], slope=float(slope), t_exit=t_exit, diagnostics=diag,
               control_reflective_fraction=fracR, control_fired=bool(fracR > 0.9),
               passed=bool(ok_slope and ok_pred and fracR > 0.9),
               note="pass requires BOTH slope 4 +- 0.5 AND coefficient agreement within a factor 2 (an 'or' could pass a wrong coefficient)")
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
