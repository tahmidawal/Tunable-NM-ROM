"""Wave 2D phase 4 — cost ladder vs the FOM across resolution.  RUN ONLY IF W3 PASSED (design r3).

  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY wav2d_ladder.py N=256 BC=ref HEAD=sup ARM=C RS=20 [R=64 M=64]
        [SRC_N=128] [REPS=5 BURN=2] [OUT=runs/wav2d] [CACHE=cache/wav2d]

Arms, timed in ONE process, balanced AB/BA order (reps outermost), warm-up discarded, all 16 held-out cases
must complete, raw repetitions retained:
  FOM-ref   CN, 80 substeps, CG 1e-10, JAX on the GPU (the reference solution)
  FOM-tol   the FASTEST configuration on the predeclared grid SUBSTEPS in {1,2,4,8,20,40,80} x CG tol in
            {1e-4,1e-6,1e-8,1e-10} whose traj-RMS vs FOM-ref (same 16 cases, same metric, accuracy and time
            from the same invocation) is <= the ROM's error; solver: Jacobi-preconditioned CG, and for the
            reflective BC also the exact DST fast solve (L_D is diagonalised by the sine modes) -- the faster
            of the two per configuration is used (both reported)
  POD-K, POD-R   Galerkin CN recurrences (numpy, K x K / R x R dense)
  ours      the latent-stepping arm (A or C) at the RS that passed W3, numpy
Reported: (i) latent-only kernel time per trajectory and per step; (ii) END-TO-END time = cold start (LM fit
of the head to the known u0) + stepping + decode at the 51 output times (the headline, since W3 judges full
fields); ratios ours/FOM-tol and ours/POD-R on the end-to-end figure.

BANK ACROSS THE LADDER (amendment 7): K, R, M are fixed; at a resolution N != SRC_N the bank is the SRC_N bank
PROLONGED by bilinear interpolation of its R modes and re-orthonormalised in the fine M-metric, and the head
is the SRC_N head (it maps z to coefficients in that basis).  The ROM's accuracy at N is measured against the
N-resolution FOM and FOM-tol is matched to it, so nothing is assumed about the prolongation's quality.
Where SRC_N == N the phase-2 bank and head are used directly.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.ndimage as ndi

import jax
import jax.numpy as jnp

import wav2d_common as wc
import wav2d_bank as wb
import wav2d_head as wh
import wav2d_rom as wr
from wav2d_common import Grid, log, precond

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
N = int(ARGS.get("N", "64")); BC = ARGS.get("BC", "ref"); R = int(ARGS.get("R", "64")); MM = int(ARGS.get("M", "64"))
HEAD = ARGS.get("HEAD", "sup"); ARM = ARGS.get("ARM", "C"); RS = int(ARGS.get("RS", "20"))
STEPS = int(ARGS.get("STEPS", "40000")); SRC_N = int(ARGS.get("SRC_N", str(N))); SMOKE = int(ARGS.get("SMOKE", "0"))
REPS = int(ARGS.get("REPS", "5")); BURN = int(ARGS.get("BURN", "2"))
OUT = ARGS.get("OUT", "runs/wav2d"); CACHE = ARGS.get("CACHE", "cache/wav2d")
N_TRAIN = int(ARGS.get("N_TRAIN", str(wc.N_TRAIN))); N_TEST = int(ARGS.get("N_TEST", str(wc.N_TEST)))
SUBS_GRID = [int(x) for x in ARGS.get("SUBS_GRID", "1,2,4,8,20,40,80").split(",")]
TOL_GRID = [float(x) for x in ARGS.get("TOL_GRID", "1e-4,1e-6,1e-8,1e-10").split(",")]
K_OF = {"sup": 6, "auto": 8, "auto+vc": 8}
os.makedirs(OUT, exist_ok=True)


# ----------------------------- bank prolongation -----------------------------

def prolong_bank(g_src: Grid, g_dst: Grid, G_src):
    """bilinear prolongation of each mode from the SRC grid to the DST grid, then M-orthonormalisation
    (QR in the fine M-metric); returns G_dst (n_dst, R) with G^T M G = I."""
    R_ = G_src.shape[1]
    out = np.zeros((g_dst.n, R_))
    xs = np.linspace(0, 1, g_src.N); xd = np.linspace(0, 1, g_dst.N)
    # map fine node coordinates to fractional source indices
    fi = xd * (g_src.N - 1)
    FI, FJ = np.meshgrid(fi, fi, indexing="ij")
    for r in range(R_):
        Us = g_src.state_to_full(G_src[:, r])
        Ud = ndi.map_coordinates(Us, [FI.reshape(-1), FJ.reshape(-1)], order=1, mode="nearest").reshape(g_dst.N, g_dst.N)
        out[:, r] = g_dst.full_to_state(Ud)
    sq = np.sqrt(g_dst.mass_diag())
    Q, _ = np.linalg.qr(out * sq[:, None])
    # keep the sign convention of the prolonged modes (QR may flip)
    sgn = np.sign(np.sum(Q * (out * sq[:, None]), axis=0)); sgn[sgn == 0] = 1
    return (Q * sgn[None, :]) / sq[:, None]


# ----------------------------- FOM solvers for the ladder -----------------------------

def make_fom_timed(g: Grid, substeps, cg_tol, solver="pcg"):
    """CN rollout of ONE trajectory returning the 51 stored snapshots; solver 'pcg' (Jacobi-preconditioned CG on
    the M^{1/2}-scaled operator) or 'dst' (exact sine-transform solve, reflective only)."""
    N_ = g.N; dt = wc.DT_SNAP / substeps
    lap = wc.lap_fn(g)
    mdiag = jnp.asarray(g.mass_diag()); dB = jnp.asarray(g.damping_diag()); sq = jnp.sqrt(mdiag); isq = 1.0 / sq
    is_abs = g.bc == "abs"
    ni = N_ - 2
    if solver == "dst":
        precond(not is_abs, "DST solve is reflective-only")
        k = jnp.arange(1, ni + 1)
        lam1 = (2.0 / g.dx ** 2) * (1 - jnp.cos(k * jnp.pi * g.dx))
        LAM = lam1[:, None] + lam1[None, :]

        def dst2(U):          # DST-I via odd extension FFT, both axes
            def dst1(x, axis):
                n_ = x.shape[axis]
                z = jnp.zeros_like(jnp.take(x, jnp.array([0]), axis=axis))
                ext = jnp.concatenate([z, x, z, -jnp.flip(x, axis=axis)], axis=axis)
                return -0.5 * jnp.imag(jnp.fft.fft(ext, axis=axis)).take(jnp.arange(1, n_ + 1), axis=axis)
            return dst1(dst1(U, 0), 1)

        def solve(rhs, c, x0):
            a = (0.5 * dt * c) ** 2
            Y = dst2(rhs.reshape(ni, ni)) / (1.0 + a * LAM)
            return (dst2(Y) * (4.0 / (N_ - 1) ** 2)).reshape(-1)
    else:
        def solve(rhs, c, x0):
            s = 0.5 * dt * c; a = s * s
            diag = 1.0 + s * dB + 4.0 * a / g.dx ** 2            # Jacobi
            if is_abs:
                As = lambda y: sq * ((isq * y) + s * dB * (isq * y) - a * lap(isq * y))
                y, _ = jax.scipy.sparse.linalg.cg(As, sq * rhs, x0=sq * x0, tol=cg_tol, maxiter=wc.CG_MAXITER, M=lambda y: y / diag)
                return isq * y
            A = lambda w: w + s * dB * w - a * lap(w)
            x, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=x0, tol=cg_tol, maxiter=wc.CG_MAXITER, M=lambda y: y / diag)
            return x

    @jax.jit
    def rollout(u0, v0, c):
        def sub(carry, _):
            u, v, Lu = carry
            s = 0.5 * dt * c; a = s * s
            u1 = solve(u + s * dB * u + dt * v + a * Lu, c, u + dt * v)
            Lu1 = lap(u1)
            v1 = ((1.0 - s * dB) * v + 0.5 * dt * c ** 2 * (Lu + Lu1)) / (1.0 + s * dB)
            return (u1, v1, Lu1), None

        def snap(carry, _):
            carry, _ = jax.lax.scan(sub, carry, None, length=substeps)
            return carry, carry[0]
        _, S = jax.lax.scan(snap, (u0, v0, lap(u0)), None, length=wc.NUM_STEPS)
        return jnp.concatenate([u0[None], S])
    return rollout


def time_fn(fn, reps, burn):
    """wall time per call, raw reps retained"""
    ts = []
    for r in range(burn + reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts[burn:]


def main():
    t_all = time.time()
    g = Grid(N, BC); gs = Grid(SRC_N, BC)
    prov = wc.provenance()
    log(f"phase-4 ladder N={N} BC={BC} head={HEAD} arm={ARM} RS={RS} SRC_N={SRC_N} backend={prov['jax_backend']}")
    K = K_OF[HEAD]
    # data at N: 16 held-out trajectories regenerated (the FOM-ref arm computes them again, timed)
    cxt, cyt, wt, at, ct, mut = wc.sample_params(wc.TEST_SEED, m=N_TEST)
    U0 = np.stack([wc.blob_ic(g, cxt[i], cyt[i], wt[i], at[i]) for i in range(N_TEST)])
    # bank and head from SRC_N
    Gs = np.load(os.path.join(CACHE, f"bank_{BC}_N{SRC_N}_R{R}.npz"))["G"]
    spec = wh.load_spec(os.path.join(CACHE, f"head_{BC}_N{SRC_N}_R{R}_{HEAD.replace('+','')}_s{STEPS}{'_SMOKE' if SMOKE else ''}.npz"),
                        expect=dict(mode=HEAD, K=K, R=R, steps=STEPS, smoke=bool(SMOKE), bc=BC, N=SRC_N))
    precond(spec is not None, "certified head not found in the cache")
    head = wr.HeadNP(spec)
    t0 = time.time()
    G = Gs if SRC_N == N else prolong_bank(gs, g, Gs)
    T = wr.build_tables(g, G, MM)
    setup_rom = time.time() - t0
    log(f"  bank {'direct' if SRC_N == N else 'prolonged from N=' + str(SRC_N)} + tables in {setup_rom:.1f}s; Mr-I {np.linalg.norm(T['Mr'] - np.eye(R)):.1e}")
    m = g.mass_diag()

    # ---------------- FOM-ref (timed, also the reference solution) ----------------
    fom_ref = make_fom_timed(g, wc.SUBSTEPS, 1e-10, "pcg")
    def run_ref():
        out = []
        for i in range(N_TEST):
            S = fom_ref(jnp.asarray(U0[i]), jnp.zeros(g.n), float(ct[i])); S.block_until_ready(); out.append(np.asarray(S))
        return out
    Uref = run_ref()
    t_ref = time_fn(run_ref, REPS, BURN)
    log(f"  FOM-ref: {np.median(t_ref)/N_TEST:.4f} s/traj (median of {REPS})")

    # ---------------- ours: cold start + stepping + decode (end-to-end) and latent-only kernel ----------------
    Ct0 = wb.coefficients(g, G, U0)
    def cold_start(i):
        Z, _ = wh.oracle(spec, Ct0[i:i + 1], n_starts=4, iters=200)
        return Z[0]
    dt = wc.DT_SNAP / RS; n_steps = wc.NUM_STEPS * RS
    PhiM = T["PhiM"]
    def run_ours(i, z0, decode=True):
        c_i = float(ct[i])
        if ARM == "A":
            Zs, _, _, ok = wr.ArmA(T, head, c_i, dt).rollout(z0, PhiM @ np.zeros(g.n), n_steps, RS)
        else:
            arm = wr.ArmC(T, head, c_i, dt); Zs, _, _, _, ok = arm.rollout(z0, np.zeros(K), n_steps, RS)
        precond(ok, f"ours arm {ARM} did not complete on case {i}")
        H = np.array([head.h(z) for z in Zs])
        return (H @ G.T) if decode else H
    z0s = [cold_start(i) for i in range(N_TEST)]
    Uours = [run_ours(i, z0s[i]) for i in range(N_TEST)]
    err_ours = [wc.traj_rms(g, Uours[i], Uref[i]) for i in range(N_TEST)]
    log(f"  ours arm {ARM}: traj-RMS median {np.median(err_ours):.4f} vs FOM-ref at N={N}")
    def run_ours_e2e():
        for i in range(N_TEST):
            run_ours(i, cold_start(i), decode=True)
    def run_ours_kernel():
        for i in range(N_TEST):
            run_ours(i, z0s[i], decode=False)

    # ---------------- POD-K / POD-R CN ----------------
    def run_pod(K_):
        outs = []
        for i in range(N_TEST):
            P = wr.PodCN(T, K_, float(ct[i]), dt); Q, _ = P.rollout(Ct0[i][:K_], np.zeros(K_), n_steps, RS)
            outs.append(Q @ G[:, :K_].T)
        return outs
    err_podK = [wc.traj_rms(g, u, Uref[i]) for i, u in enumerate(run_pod(K))]
    err_podR = [wc.traj_rms(g, u, Uref[i]) for i, u in enumerate(run_pod(R))]

    # ---------------- FOM-tol: fastest configuration meeting the ROM's error ----------------
    target = float(np.median(err_ours))
    grid_res = []
    for subs in SUBS_GRID:
        for tol in TOL_GRID:
            for solver in (["pcg", "dst"] if BC == "ref" else ["pcg"]):
                if solver == "dst" and tol != TOL_GRID[-1]:
                    continue                                     # the exact solve has no tolerance
                fom = make_fom_timed(g, subs, tol, solver)
                def run_c(fom=fom):
                    out = []
                    for i in range(N_TEST):
                        S = fom(jnp.asarray(U0[i]), jnp.zeros(g.n), float(ct[i])); S.block_until_ready(); out.append(np.asarray(S))
                    return out
                Uc = run_c()
                errs = [wc.traj_rms(g, Uc[i], Uref[i]) for i in range(N_TEST)]
                ok = all(np.isfinite(errs)) and float(np.median(errs)) <= target
                ts = time_fn(run_c, REPS if ok else 1, BURN if ok else 1)
                grid_res.append(dict(substeps=subs, cg_tol=tol, solver=solver, err_median=float(np.median(errs)), meets=bool(ok),
                                     t_per_traj=float(np.median(ts)) / N_TEST, times=ts))
                log(f"  FOM cfg subs={subs} tol={tol:.0e} {solver}: err {np.median(errs):.3e} {'MEETS' if ok else 'no'} {np.median(ts)/N_TEST:.4f} s/traj")
    if BC == "ref":
        # the exact DST solve must agree with PCG at the tightest tolerance (a solver-correctness gate for the ladder)
        S_d = np.asarray(make_fom_timed(g, wc.SUBSTEPS, 1e-10, "dst")(jnp.asarray(U0[0]), jnp.zeros(g.n), float(ct[0])))
        dst_vs_pcg = float(np.max(np.abs(S_d - Uref[0])) / np.max(np.abs(Uref[0])))
        log(f"  DST solve vs PCG (CG 1e-10) at N={N}: {dst_vs_pcg:.2e}")
        precond(dst_vs_pcg <= 1e-7, "DST fast solve disagrees with PCG")
    else:
        dst_vs_pcg = None
    meeting = [r for r in grid_res if r["meets"]]
    fom_tol = min(meeting, key=lambda r: r["t_per_traj"]) if meeting else None

    # ---------------- balanced timing of the final arms ----------------
    arms = dict(fom_ref=run_ref, ours_e2e=run_ours_e2e, ours_kernel=run_ours_kernel,
                podK=lambda: run_pod(K), podR=lambda: run_pod(R))
    if fom_tol is not None:
        fomt = make_fom_timed(g, fom_tol["substeps"], fom_tol["cg_tol"], fom_tol["solver"])
        def run_tol():
            for i in range(N_TEST):
                fomt(jnp.asarray(U0[i]), jnp.zeros(g.n), float(ct[i])).block_until_ready()
        arms["fom_tol"] = run_tol
    names = list(arms); times = {k: [] for k in names}
    for r in range(BURN + REPS):
        order = names if r % 2 == 0 else names[::-1]           # AB / BA
        for k in order:
            t0 = time.perf_counter(); arms[k](); dtm = time.perf_counter() - t0
            if r >= BURN:
                times[k].append(dtm)
    summary = {k: dict(median_s_per_traj=float(np.median(v)) / N_TEST, raw=v) for k, v in times.items()}
    res = dict(N=N, bc=BC, head=HEAD, arm=ARM, RS=RS, R=R, M=MM, K=K, src_N=SRC_N, provenance=prov, args=ARGS,
               err_ours=err_ours, err_podK=err_podK, err_podR=err_podR, target_err=target,
               fom_grid=grid_res, fom_tol=fom_tol, timing=summary, setup_rom_s=setup_rom, dst_vs_pcg=dst_vs_pcg, smoke=bool(SMOKE),
               latent_steps_per_traj=n_steps,
               ratios=dict(ours_e2e_over_fom_tol=(summary["ours_e2e"]["median_s_per_traj"] / summary["fom_tol"]["median_s_per_traj"]) if fom_tol else None,
                           ours_e2e_over_podR=summary["ours_e2e"]["median_s_per_traj"] / summary["podR"]["median_s_per_traj"],
                           ours_kernel_over_fom_tol=(summary["ours_kernel"]["median_s_per_traj"] / summary["fom_tol"]["median_s_per_traj"]) if fom_tol else None,
                           fom_ref_over_ours_e2e=summary["fom_ref"]["median_s_per_traj"] / summary["ours_e2e"]["median_s_per_traj"]),
               where=dict(fom="JAX on " + prov["jax_backend"], rom="numpy on CPU (latent kernel), decode numpy"),
               wall_s=time.time() - t_all)
    path = os.path.join(OUT, f"wav2d_ladder_{BC}_N{N}_{HEAD.replace('+','')}_{ARM}{'_SMOKE' if SMOKE else ''}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"  timing (s/traj): " + ", ".join(f"{k} {v['median_s_per_traj']:.4f}" for k, v in summary.items()))
    log(f"phase 4 done ({res['wall_s']:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
