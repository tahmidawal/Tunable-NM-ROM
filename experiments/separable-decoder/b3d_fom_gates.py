"""Burgers 3D — phase 0: the FOM gates of B3D-DESIGN.md r3 (F1..F11), each with
its negative control, at one resolution N (F7 spans F7_NS).

    N=33 OUT=runs/b3dtensor/fom_gates_n33.json TABLE_DIR=runs/b3dtensor/tables \
        python b3d_fom_gates.py

Every gate value is RECORDED in the JSON before it is asserted; the controls
are asserted to FIRE (a control that does not cross its threshold is a FAIL
of the gate design, not a pass).  NaN anywhere is FAIL.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import b3d_common as b3

F64 = jnp.float64
HERE = os.path.dirname(os.path.abspath(__file__))

N = int(os.environ.get("N", "33"))
OUT = os.environ.get("OUT", f"/tmp/b3d_fom_gates_n{N}.json")
TABLE_DIR = os.environ.get("TABLE_DIR", os.path.join(HERE, "runs", "b3dtensor", "tables"))
N_TRAIN_TABLE = int(os.environ.get("N_TRAIN_TABLE", "576"))
N_TEST = int(os.environ.get("N_TEST", "8"))
SEED = int(os.environ.get("SEED", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", "1"))
F7_NS = [int(v) for v in os.environ.get("F7_NS", "33,65,129").split(",") if v]
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "8"))
F10_TRAJ = int(os.environ.get("F10_TRAJ", "8"))
_FILM_CANDS = [os.path.join(HERE, "deps", "burgers2d-coord-rom", "burgers2d_film.py"),          # staged
               os.path.join(HERE, "..", "wave2d-rom-latent-stepping", "deps", "burgers2d-coord-rom",
                            "burgers2d_film.py")]                                                  # worktree
FILM_PATH = os.environ.get("FILM_PATH", next((c for c in _FILM_CANDS if os.path.exists(c)), _FILM_CANDS[0]))

log = b3.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def bwd(a, b, opnorm, x):
    """backward-error normalisation: ||a - b|| / (||Op||_inf ||x|| + ||b||)"""
    return float(np.linalg.norm(a - b) / (opnorm * np.linalg.norm(x) + np.linalg.norm(b) + 1e-300))


def git_commit():
    c = os.environ.get("COMMIT")
    if c:
        return c
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL, cwd=HERE).strip()
    except Exception:
        return None


def tables():
    os.makedirs(TABLE_DIR, exist_ok=True)
    out = {}
    for name, sd, m in (("train", SEED, N_TRAIN_TABLE), ("test", TEST_SEED, N_TEST)):
        path = os.path.join(TABLE_DIR, f"b3d_params_{name}_seed{sd}_m{m}.npz")
        if os.path.exists(path):
            tab = b3.load_param_table(path)
        else:
            t0 = time.time()
            tab = b3.build_param_table(sd, m, path)
            log(f"  table[{name}] built: {m} rows, seed {sd}, s* on {b3.N_REF}^3 "
                f"[{time.time()-t0:.0f}s] -> {path}")
        tab["path"] = path
        out[name] = tab
    return out


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B3D-FOM-GATES N={N}")
    t_all = time.time()
    n = N
    ni = n - 2
    dx = 1.0 / (n - 1)
    interior = b3.interior_indices_3d(n)
    coords = b3.grid_coords_3d(n)
    tabs = tables()
    tt = tabs["test"]
    report = dict(config=dict(N=N, n_interior=int(ni ** 3), dx=dx, dt=b3.DT,
                              num_steps=b3.NUM_STEPS, newton_iters=b3.NEWTON_ITERS,
                              lin_tol=b3.LIN_TOL, lin_maxiter=b3.LIN_MAXITER, seed=SEED,
                              test_seed=TEST_SEED, n_test=N_TEST, f7_ns=F7_NS,
                              table_train=dict(path=tabs["train"]["path"], sha256=tabs["train"]["sha256"],
                                               m=int(tabs["train"]["m"])),
                              table_test=dict(path=tt["path"], sha256=tt["sha256"], m=int(tt["m"])),
                              x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                              backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
                              jax_version=jax.__version__, commit=git_commit(),
                              slurm_job=os.environ.get("SLURM_JOB_ID"),
                              node=os.environ.get("SLURMD_NODENAME", "local")),
                  gates={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    def gate(name, value, passed, control=None, control_fired=None, **extra):
        d = dict(value=value, passed=bool(passed), **extra)
        if control is not None:
            d["control"] = control
            d["control_fired"] = bool(control_fired)
        report["gates"][name] = d
        tag = "PASS" if passed else "FAIL"
        ctl = "" if control is None else f"  | control {control:.3e} fired={bool(control_fired)}"
        log(f"  GATE {name}: {value if isinstance(value, str) else f'{value:.3e}'} {tag}{ctl}")
        save()

    rng = np.random.default_rng(SEED + 900)
    L = b3.assemble_L_3d(n)
    Lnorm = float(np.abs(L).sum(axis=1).max())

    # ---------------- F9: two DST implementations ---------------------------
    v = rng.standard_normal(ni ** 3)
    nu9 = 0.03
    h_mm = b3.make_helmholtz_inv(n, "mm")
    h_ff = b3.make_helmholtz_inv(n, "fft")
    y_mm = np.asarray(jax.block_until_ready(h_mm(jnp.asarray(v), nu9)))
    y_ff = np.asarray(jax.block_until_ready(h_ff(jnp.asarray(v), nu9)))
    tm = {}
    for nm, fn in (("mm", h_mm), ("fft", h_ff)):
        f = jax.jit(fn)
        jax.block_until_ready(f(jnp.asarray(v), nu9))
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            jax.block_until_ready(f(jnp.asarray(v), nu9))
            ts.append(time.perf_counter() - t0)
        tm[nm] = float(np.median(ts) * 1e3)
    f9 = rel(y_ff, y_mm)
    # control: the FFT path with its orthonormal factor dropped on the forward transform
    V = jnp.asarray(v).reshape(ni, ni, ni)
    C_bad = b3.dst3_fft(V, n) / np.sqrt(2.0 / (n - 1)) ** 3
    y_bad = np.asarray(b3.dst3_fft(C_bad / (1.0 + b3.DT * nu9 * jnp.asarray(b3.lam_3d(n))), n)).reshape(-1)
    f9c = rel(y_bad, y_mm)
    gate("F9_dst_mm_vs_fft", f9, f9 <= 1e-13, control=f9c, control_fired=f9c > 1e-3,
         helmholtz_ms=tm, faster="mm" if tm["mm"] <= tm["fft"] else "fft",
         control_note="FFT forward transform without its orthonormal factor")
    dst = "mm" if tm["mm"] <= tm["fft"] else "fft"
    hinv = b3.make_helmholtz_inv(n, dst)
    report["config"]["dst"] = dst

    # ---------------- F4: preconditioner exactness (backward error) ---------
    Hn = 1.0 + b3.DT * nu9 * Lnorm
    y = np.asarray(hinv(jnp.asarray(v), nu9))
    Hy = y + b3.DT * nu9 * (-(L @ y))
    f4 = bwd(Hy, v, Hn, y)
    y2 = np.asarray(hinv(jnp.asarray(v), nu9, nu_scale=2.0))
    f4c = bwd(y2 + b3.DT * nu9 * (-(L @ y2)), v, Hn, y2)
    gate("F4_helmholtz_backward_error", f4, f4 <= 1e-14, control=f4c, control_fired=f4c > 1e-2)

    # ---------------- F2: modes are eigenvectors (backward error) -----------
    M2 = 256 if ni ** 3 >= 256 else ni ** 3
    kx, ky, kz, Phi, lam = b3.test_modes_3d(n, M2)
    PL = Phi * lam[None, :]
    f2 = float(np.linalg.norm(L @ Phi + PL) / (Lnorm * np.linalg.norm(Phi) + np.linalg.norm(PL)))
    lam_cont = np.pi ** 2 * (kx ** 2 + ky ** 2 + kz ** 2).astype(float)
    PLc = Phi * lam_cont[None, :]
    f2c = float(np.linalg.norm(L @ Phi + PLc) / (Lnorm * np.linalg.norm(Phi) + np.linalg.norm(PLc)))
    colnorm = float(np.max(np.abs(np.linalg.norm(Phi, axis=0) - 1.0)))
    gate("F2_modes_eigenvectors", f2, f2 <= 1e-14 and colnorm <= 1e-12, control=f2c,
         control_fired=f2c > 1e-3, M=M2, column_norm_dev=colnorm)

    # ---------------- test truth: F3, F5 (+ controls), F10 -----------------
    roll = b3.make_truth_rollout(n, dst)
    U0 = np.stack([b3.blob_ic_3d(n, tt, j, coords)[interior] for j in range(N_TEST)])
    t0 = time.time()
    U, worst, umin, umax, frac_le0, secs = b3.build_truth(n, tt, np.arange(N_TEST), GEN_CHUNK, roll, coords)
    report["data"] = dict(test=dict(n_test=N_TEST, max_fom_rel_residual=worst, min_u=umin, max_u=umax,
                                    frac_points_le0=frac_le0, gen_secs=secs,
                                    u0_max=[float(v_) for v_ in U0.max(axis=1)],
                                    nu=[float(v_) for v_ in tt["nu"][:N_TEST]],
                                    B=[int(v_) for v_ in tt["B"][:N_TEST]]))
    # F3 control: 2-iteration generator on the first trajectory
    roll1 = b3.make_truth_rollout_iters(n, 1, dst)
    _, w1 = roll1(jnp.asarray(U0[:1]), jnp.asarray(tt["nu"][:1]))
    gate("F3_truth_acceptance", worst, np.isfinite(worst) and worst <= 1e-8, control=float(w1),
         control_fired=float(w1) > 1e-8,
         control_note="1 Newton iteration per step (2 iterations converged to 2.9e-10 at N=33)")
    # F5 control (design r4, [A55]): a DETERMINISTIC solver-output mutation -- one interior node of an
    # accepted k>=1 state set to -1e-3 -- run through the same check; guaranteed to fire.  The
    # downwind (anti-diffusive) rollout is RECORDED as a diagnostic with its finiteness and residual.
    Umut = U.copy(); Umut[0, 25, Umut.shape[2] // 2] = -2e-3
    f5c = float(np.min(Umut[:, 1:]))
    rd = b3.make_control_rollout_adv(n, "downwind", dst)
    sd, wd = rd(jnp.asarray(U0[:1]), jnp.asarray([0.01]))
    sd = np.asarray(sd)
    down = dict(finite=bool(np.all(np.isfinite(sd))), worst_res=float(wd),
                min_u=float(np.min(sd[:, 1:])) if np.all(np.isfinite(sd)) else None,
                converged=bool(np.isfinite(wd) and wd <= 1e-8),
                note="anti-diffusive scheme: Newton does not converge on it; recorded, not the control")
    gate("F5_nonnegativity_test", umin, np.isfinite(umin) and umin >= -1e-9, control=f5c,
         control_fired=f5c < -1e-3, control_note="output mutation: one node of state k=25 set to -2e-3 (bar: < -1e-3)",
         downwind_diagnostic=down)
    # F10: generator cost per trajectory
    t0 = time.time()
    sn_, _ = roll(jnp.asarray(U0[:F10_TRAJ]), jnp.asarray(tt["nu"][:F10_TRAJ]))
    jax.block_until_ready(sn_)
    per_traj = (time.time() - t0) / F10_TRAJ
    del sn_
    report["gates"]["F10_generator_secs_per_traj"] = dict(
        value=per_traj, chunk=F10_TRAJ, projected_hours_512=per_traj * 512 / 3600,
        projected_hours_256=per_traj * 256 / 3600, projected_hours_128=per_traj * 128 / 3600)
    log(f"  F10: {per_traj:.2f} s/traj (chunk {F10_TRAJ}); 512 traj = {per_traj*512/3600:.2f} h")
    save()

    # ---------------- F6: stencil vs assembled operator ---------------------
    D = b3.assemble_Dminus_3d(n)
    u = U[0, 25]
    up = U[0, 24]
    nu0 = float(tt["nu"][0])
    R_st = np.asarray(b3.fom_residual_int(jnp.asarray(u), jnp.asarray(up), nu0, n))
    R_as = u - up + b3.DT * (u * (D @ u) - nu0 * (L @ u))
    opn = 1.0 + b3.DT * (float(np.abs(D).sum(1).max()) * float(np.abs(u).max()) + nu0 * Lnorm)
    f6 = bwd(R_st, R_as, opn, u)
    # control: a sign-changing state (blob minus displaced blob)
    us = U0[0] - np.roll(U0[0].reshape(ni, ni, ni), 3, axis=0).reshape(-1)
    R_st_c = np.asarray(b3.fom_residual_int(jnp.asarray(us), jnp.asarray(up), nu0, n))
    R_as_c = us - up + b3.DT * (us * (D @ us) - nu0 * (L @ us))
    f6c = bwd(R_st_c, R_as_c, opn, us)
    gate("F6_stencil_vs_assembled", f6, f6 <= 1e-13, control=f6c, control_fired=f6c > 1e-6,
         control_note="sign-changing state: the upwind switch is live")

    # ---------------- F1: axis symmetry ------------------------------------
    sym_tab = dict(B=np.array([3]), c=np.array([[[0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]]]),
                   w=np.array([[0.12, 0.15, 0.12]]), rho=np.array([[1.0, 0.8, 1.0]]),
                   A=np.array([1.5]), nu=np.array([0.02]), s_star=np.array([1.0]), m=1)
    u0s = b3.blob_ic_3d(n, sym_tab, 0, coords)
    u0s = u0s / u0s.max() * 1.5
    sn_s, _ = roll(jnp.asarray(u0s[interior][None]), jnp.asarray([0.02]))
    Us = np.asarray(sn_s[0])                                   # (T+1, n_i)
    perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    f1 = 0.0
    for t_ in range(1, b3.NUM_STEPS + 1):
        V = Us[t_].reshape(ni, ni, ni)
        for p in perms:
            f1 = max(f1, rel(np.transpose(V, p).reshape(-1), Us[t_]))
    rz = b3.make_control_rollout_zadv(n, 1.01, dst)
    sn_c, _ = rz(jnp.asarray(u0s[interior][None]), jnp.asarray([0.02]))
    Uc = np.asarray(sn_c[0])
    f1c = 0.0
    for t_ in range(1, b3.NUM_STEPS + 1):
        V = Uc[t_].reshape(ni, ni, ni)
        for p in perms:
            f1c = max(f1c, rel(np.transpose(V, p).reshape(-1), Uc[t_]))
    gate("F1_axis_symmetry", f1, f1 <= 1e-12, control=f1c, control_fired=f1c > 1e-4,
         control_note="z-advection coefficient x1.01")

    # ---------------- F8: 2D-vs-3D consistency on a plateau-in-z state -------
    spec = importlib.util.spec_from_file_location("burgers2d_film", FILM_PATH)
    bf = importlib.util.module_from_spec(spec)
    os.environ.setdefault("N", str(n))
    spec.loader.exec_module(bf)
    _, res2d = bf.make_rollout(n)
    assert abs(bf.DT - b3.DT) < 1e-15 and bf.NUM_STEPS == b3.NUM_STEPS
    # 2D fields v, v_prev on the full n x n grid (walls zero), from a 2D blob
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    v2 = 1.3 * np.exp(-((X - 0.45) ** 2 + (Y - 0.55) ** 2) / (2 * 0.12 ** 2)) * 16 * X * (1 - X) * Y * (1 - Y)
    vp2 = 1.1 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.14 ** 2)) * 16 * X * (1 - X) * Y * (1 - Y)
    # add a sign change so the upwind switch is exercised on the plane
    v2 = v2 - 0.6 * np.exp(-((X - 0.7) ** 2 + (Y - 0.3) ** 2) / (2 * 0.1 ** 2)) * 16 * X * (1 - X) * Y * (1 - Y)
    kmid = n // 2
    prof = np.zeros(n)
    prof[kmid - 1:kmid + 2] = 1.0                               # three central planes
    for k_ in range(1, kmid - 1):                                # linear taper to the faces
        prof[k_] = k_ / (kmid - 1)
        prof[n - 1 - k_] = k_ / (kmid - 1)
    U3 = v2[:, :, None] * prof[None, None, :]
    UP3 = vp2[:, :, None] * prof[None, None, :]
    u3 = U3.reshape(-1)[interior]
    up3 = UP3.reshape(-1)[interior]
    nu8 = 0.02
    R3 = np.asarray(b3.fom_residual_int(jnp.asarray(u3), jnp.asarray(up3), nu8, n)).reshape(ni, ni, ni)
    R2 = np.asarray(res2d(jnp.asarray(v2.reshape(-1)), jnp.asarray(vp2.reshape(-1)), nu8)).reshape(n, n)
    plane3 = R3[:, :, kmid - 1]                                  # interior index kmid-1 = full index kmid
    plane2 = R2[1:-1, 1:-1]
    opn8 = 1.0 + b3.DT * (3 * np.abs(u3).max() / dx + nu8 * Lnorm)
    f8r = bwd(plane3.reshape(-1), plane2.reshape(-1), opn8, u3)
    # JVP on the plane: random tangent with the same plateau structure
    dv = rng.standard_normal((n, n)) * 16 * X * (1 - X) * Y * (1 - Y)
    dU3 = (dv[:, :, None] * prof[None, None, :]).reshape(-1)[interior]
    J3 = np.asarray(jax.jvp(lambda uu: b3.fom_residual_int(uu, jnp.asarray(up3), nu8, n),
                            (jnp.asarray(u3),), (jnp.asarray(dU3),))[1]).reshape(ni, ni, ni)
    J2 = np.asarray(jax.jvp(lambda uu: res2d(uu, jnp.asarray(vp2.reshape(-1)), nu8),
                            (jnp.asarray(v2.reshape(-1)),), (jnp.asarray(dv.reshape(-1)),))[1]).reshape(n, n)
    f8j = bwd(J3[:, :, kmid - 1].reshape(-1), J2[1:-1, 1:-1].reshape(-1), opn8, dU3)
    R3c = np.asarray(b3.fom_residual_mutated(jnp.asarray(u3), jnp.asarray(up3), nu8, n, xadv=2.0)).reshape(ni, ni, ni)
    f8c = bwd(R3c[:, :, kmid - 1].reshape(-1), plane2.reshape(-1), opn8, u3)
    # the z terms really vanish on the plane (the reason the r2 control was inert): record it
    R3z = np.asarray(b3.fom_residual_mutated(jnp.asarray(u3), jnp.asarray(up3), nu8, n, zscale=1.01, zadv=1.01)).reshape(ni, ni, ni)
    f8z = bwd(R3z[:, :, kmid - 1].reshape(-1), plane2.reshape(-1), opn8, u3)
    gate("F8_2d_vs_3d_plateau", max(f8r, f8j), max(f8r, f8j) <= 1e-13, control=f8c,
         control_fired=f8c > 1e-4, residual=f8r, jvp=f8j, control_note="x-advection coefficient doubled (x1.01 gave 1.9e-5 under the backward-error normalisation, below the 1e-4 bar)",
         z_terms_mutated_on_plane=f8z, film_sha256=b3.sha256_file(FILM_PATH),
         sign_changing_plane=bool(v2.min() < 0))

    # ---------------- F11: manufactured solution -> the order band ------------
    p_mms = None
    if len(F7_NS) >= 3 and N == F7_NS[0]:
        cm, wm, num = np.array([0.5, 0.5, 0.5]), 0.2, 0.03

        def u_ex(x, t):                                              # (P,3), scalar t -> (P,)
            mask = 64.0 * x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1]) * x[:, 2] * (1 - x[:, 2])
            return (1.0 + 0.5 * jnp.sin(2 * jnp.pi * t)) * mask * jnp.exp(
                -jnp.sum((x - cm[None, :]) ** 2, axis=1) / (2 * wm ** 2))

        def forcing_of(x):
            """f = u_t + u (u_x+u_y+u_z) - nu lap u at the points x, by autodiff of
            the closed form (continuum operators; the discrete scheme's error
            against u_ex is then the discretisation error)."""
            def f(t):
                ut = (u_ex(x, t) - u_ex(x, t - b3.DT)) / b3.DT      # the DISCRETE BE time quotient [A52]
                def grad_u(xx):
                    return jax.grad(lambda q: u_ex(q[None, :], t)[0])(xx)
                g = jax.vmap(grad_u)(x)                             # (P,3)
                def lap_u(xx):
                    Hm = jax.hessian(lambda q: u_ex(q[None, :], t)[0])(xx)
                    return jnp.trace(Hm)
                lp = jax.vmap(lap_u)(x)
                uu = u_ex(x, t)
                return ut + uu * jnp.sum(g, axis=1) - num * lp
            return jax.jit(f)
        errs, errs_c, res11 = {}, {}, {}
        for n7 in F7_NS:
            c7 = b3.grid_coords_3d(n7)
            i7 = b3.interior_indices_3d(n7)
            xi = jnp.asarray(c7[i7])
            ff = forcing_of(xi)
            r11 = b3.make_mms_rollout(n7, ff, dst)
            u0 = np.asarray(u_ex(xi, 0.0))
            assert float(jnp.min(jax.vmap(lambda t_: jnp.min(u_ex(xi, t_)))(jnp.arange(51) * b3.DT))) > 0.0, "MMS field not positive"
            t0 = time.time()
            s11, w11 = r11(jnp.asarray(u0), num)
            uT = np.asarray(u_ex(xi, b3.NUM_STEPS * b3.DT))
            st = (n7 - 1) // (F7_NS[0] - 1)
            full = np.zeros(n7 ** 3); full[i7] = np.asarray(s11[-1]); fullT = np.zeros(n7 ** 3); fullT[i7] = uT
            A_ = full.reshape(n7, n7, n7)[::st, ::st, ::st]; B_ = fullT.reshape(n7, n7, n7)[::st, ::st, ::st]
            errs[n7] = float(np.linalg.norm(A_ - B_) / np.linalg.norm(B_))
            res11[n7] = float(w11)
            if n7 == F7_NS[0]:
                r11c = b3.make_mms_rollout(n7, jax.jit(lambda t, _f=ff: -_f(t)), dst)
                s11c, _ = r11c(jnp.asarray(u0), num)
                fc = np.zeros(n7 ** 3); fc[i7] = np.asarray(s11c[-1])
                errs_c[n7] = float(np.linalg.norm(fc.reshape(n7, n7, n7) - B_) / np.linalg.norm(B_))
            log(f"    F11 N={n7}: MMS error {errs[n7]:.3e} worst res {float(w11):.2e} [{time.time()-t0:.0f}s]")
        p1 = float(np.log2(errs[F7_NS[0]] / errs[F7_NS[1]])); p2 = float(np.log2(errs[F7_NS[1]] / errs[F7_NS[2]]))
        p_mms = p2
        gate("F11_manufactured_solution_order", p_mms, 0.7 <= p_mms <= 1.3, control=errs_c[F7_NS[0]],
             control_fired=errs_c[F7_NS[0]] > 1e-1, errors={str(k): v for k, v in errs.items()},
             order_coarse=p1, order_fine=p2, worst_res=res11, control_note="forcing sign flipped: O(1) error")

    # ---------------- F7: spatial consistency on nested grids ---------------
    if len(F7_NS) >= 3 and N == F7_NS[0]:
        sm_tab = dict(B=np.array([1]), c=np.array([[[0.45, 0.5, 0.55]] * 3]), w=np.array([[0.2] * 3]),
                      rho=np.array([[1.0] * 3]), A=np.array([1.5]), nu=np.array([0.03]),
                      s_star=np.array([1.0]), m=1)
        sols, sols_c = {}, {}
        for n7 in F7_NS:
            c7 = b3.grid_coords_3d(n7)
            i7 = b3.interior_indices_3d(n7)
            u07 = b3.blob_ic_3d(n7, sm_tab, 0, c7)
            u07 = u07 / u07.max() * 1.5
            r7 = b3.make_truth_rollout(n7, dst)
            t0 = time.time()
            s7, w7 = r7(jnp.asarray(u07[i7][None]), jnp.asarray([0.03]))
            full = np.zeros(n7 ** 3); full[i7] = np.asarray(s7[0, -1])
            sols[n7] = (full.reshape(n7, n7, n7), float(w7), time.time() - t0)
            log(f"    F7 N={n7}: worst res {float(w7):.2e} [{time.time()-t0:.0f}s]")
        n0 = F7_NS[0]
        def on_common(A_, n7, shift=0):
            st = (n7 - 1) // (n0 - 1)
            A2 = np.roll(A_, shift, axis=0) if shift else A_
            return A2[::st, ::st, ::st]
        u_a, u_b, u_c = (on_common(sols[F7_NS[0]][0], F7_NS[0]), on_common(sols[F7_NS[1]][0], F7_NS[1]),
                         on_common(sols[F7_NS[2]][0], F7_NS[2]))
        d1 = np.linalg.norm(u_a - u_b); d2 = np.linalg.norm(u_b - u_c)
        order = float(np.log2(d1 / d2))
        # control: index mutation -- the fine solution sampled one fine cell off
        u_cs = on_common(sols[F7_NS[2]][0], F7_NS[2], shift=1)
        order_c = float(np.log2(d1 / np.linalg.norm(u_b - u_cs)))
        lo, hi = 0.7, 1.3                                     # frozen theoretically (first-order upwind) [A53]
        gate("F7_spatial_order", order, lo <= order <= hi, control=order_c,
             control_fired=not (lo <= order_c <= hi), d_coarse_mid=float(d1), d_mid_fine=float(d2),
             band=[lo, hi], band_source="frozen [0.7, 1.3]; F11 reported separately", p_mms=p_mms,
             worst_res={str(k): v[1] for k, v in sols.items()},
             secs={str(k): v[2] for k, v in sols.items()},
             control_note="fine solution sampled one fine cell off (index mutation)")

    report["complete"] = all(g.get("passed", True) and g.get("control_fired", True)
                             for g in report["gates"].values())
    report["secs_total"] = time.time() - t_all
    save()
    log(f"DONE -> {OUT}  complete={report['complete']} [{time.time()-t_all:.0f}s]")
    for k_, g in report["gates"].items():
        if not g.get("passed", True):
            raise SystemExit(f"GATE {k_} FAILED: {g}")
        if "control_fired" in g and not g["control_fired"]:
            raise SystemExit(f"GATE {k_} CONTROL DID NOT FIRE: {g}")


if __name__ == "__main__":
    main()
