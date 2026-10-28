#!/usr/bin/env python
"""Burgers ROM on the POD+MLP nonlinear chart -- solve, gate, and time.

Mirrors `scripts/run_rom_pod.py` (same residual, same damping, same line
search, same stopping rule, same `--fast-jac --lean-step` cost path, same
convergence gating, same bench harness) with ONE thing changed: the chart is
`mean + V z + MLP(z_std)` instead of `mean + V z`.

FOUR gates run before anything is timed; each aborts rather than warns:

  1. solver equivalence -- the POD+MLP solver, handed a chart whose MLP output
     layer is ZEROED, must reproduce the production `PODLSGNSolver`. That is
     the frozen chart exactly, so this gates the solver transcription.
  2. decoder equivalence -- `PODMLPChart.decode` is an INDEPENDENT
     re-implementation of the trained decoder (it splits the MLP into trunk and
     output projection so the output layer can be row-gathered). It is checked
     against a reference decode stored by the trainer, computed by podmlp.py's
     own `make_decode`, in a different process.
  3. row-gather -- `at_stencil(z)` vs `decode(z)[stencil]`. A previous version
     of the chart ablation silently dropped an amplitude factor here and nearly
     manufactured a false result; this catches that class of bug.
  4. analytic Jacobian vs `jacfwd` of the residual.

Denominator: the shipped FOM and every requested `--strong-fom` variant are
timed in this same job on this same device, each gated on the FOM's own
per-step Newton criterion, and the FASTEST GATE-PASSING one is recorded by
name as the denominator. Ratios against every variant are shipped too.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(os.environ.get("ROOT", "/cluster/home/tawal01/nmrom/Tunable-NM-ROM"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("_bench", REPO_ROOT / "bench.py")
fa = _load("_fom_strong", REPO_ROOT / "burgers" / "scripts" / "fom_strong.py")

from tunable_rom_burgers.fom.burgers import (  # noqa: E402
    BurgersFOM, NUM_STEPS, NEWTON_RTOL, NEWTON_MAX_ITERS, KRYLOV_TOL,
    KRYLOV_MAXITER, DT)
from tunable_rom_burgers.solver.pod_solver import (  # noqa: E402
    PODLSGNSolver, interior_flat, stencil_indices_for)
from tunable_rom_burgers.solver.podmlp_chart import (  # noqa: E402
    PODMLPChart, PODMLPLSGNSolver)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _spread(xs):
    a = np.asarray(xs, dtype=float)
    med = float(np.median(a))
    return {"median": med, "min": float(a.min()), "max": float(a.max()),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "max_over_median": float(a.max() / med) if med > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--pod", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--n-eq", type=int, default=0)
    ap.add_argument("--eq-seed", type=int, default=0)
    ap.add_argument("--gn-max-iters", type=int, default=20)
    ap.add_argument("--gn-rel-tol", type=float, default=1e-6)
    ap.add_argument("--lm-damping", type=float, default=1e-3)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--num-trajectories", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--strong-fom", default="incr_1e-2",
                    help="comma-separated FOM variants to time as candidate "
                         "denominators alongside the shipped one")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--skip-gates", action="store_true",
                    help="ONLY for re-timing an arm whose gates already passed "
                         "in this same job; never for a first run")
    args = ap.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    wdt = jnp.float64 if args.x64 else jnp.float32
    prec = "float64" if args.x64 else "float32"

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    k = int(ck["k"])
    data = np.load(args.data)
    N, d = int(data["N"]), int(data["spatial_dim"])
    U_val, val_nu = data["U_val"], data["val_nu"]
    pod = np.load(args.pod)
    V = pod["V"][:, :k]
    mean = pod["mean"]
    pod_hash = sha256(args.pod)
    if ck.get("pod_sha256") and ck["pod_sha256"] != pod_hash:
        raise SystemExit("ABORT: checkpoint was trained on a different POD file "
                         f"({ck['pod_sha256']} vs {pod_hash}).")

    fom = BurgersFOM(N=N, spatial_dim=d)
    int_flat = interior_flat(N, d)
    if args.n_eq and args.n_eq < int_flat.size:
        rng = np.random.default_rng(args.eq_seed)
        centers = np.sort(rng.choice(int_flat, size=args.n_eq, replace=False))
        eq_mode = f"random-{args.n_eq}"
    else:
        centers = int_flat
        eq_mode = "full-grid"
    weights = np.full(centers.shape[0], 1.0 / centers.shape[0], dtype=np.float64)
    st_flat = stencil_indices_for(centers, N, d).reshape(-1)

    common = dict(N=N, spatial_dim=d, dx=fom.dx, eq_flat_indices=centers,
                  eq_weights=weights, gn_max_iters=args.gn_max_iters,
                  gn_rel_tol=args.gn_rel_tol, lm_damping=args.lm_damping,
                  x64=args.x64)

    chart = PODMLPChart(V, mean, ck["params"], ck["znorm"], st_flat, x64=args.x64)
    solver = PODMLPLSGNSolver(chart=chart, fast_jac=True, lean_step=True, **common)
    print(f"POD+MLP LS-GN: k={k} W={ck['width']} collocation={eq_mode} "
          f"({centers.shape[0]}) gn_max={args.gn_max_iters} tol={args.gn_rel_tol:.1e} "
          f"x64={args.x64}", flush=True)
    print(f"pod sha256 {pod_hash}", flush=True)

    val_starts = np.arange(0, U_val.shape[0], NUM_STEPS + 1)[: args.num_trajectories]
    u_g = jnp.asarray(U_val[val_starts[0]], dtype=wdt)
    nu_g = jnp.asarray(float(val_nu[0]), dtype=wdt)

    gates = {}
    if not args.skip_gates:
        # ---- gate 1: zeroed correction == production frozen-POD solver -----
        zp = dict(ck["params"])
        Wo, bo = zp["out"]
        zp["out"] = (np.zeros_like(np.asarray(Wo)), np.zeros_like(np.asarray(bo)))
        chart0 = PODMLPChart(V, mean, zp, ck["znorm"], st_flat, x64=args.x64)
        sol0 = PODMLPLSGNSolver(chart=chart0, fast_jac=True, lean_step=True, **common)
        prod = PODLSGNSolver(V=V, mean=mean, N=N, spatial_dim=d, dx=fom.dx,
                             eq_flat_indices=centers, eq_weights=weights,
                             gn_max_iters=args.gn_max_iters,
                             gn_rel_tol=args.gn_rel_tol,
                             lm_damping=args.lm_damping, x64=args.x64,
                             fast_jac=True, lean_step=True)
        u_a, _ = jax.jit(lambda u, nu: sol0.rollout(u, nu, NUM_STEPS))(u_g, nu_g)
        u_b, _ = jax.jit(lambda u, nu: prod.rollout(u, nu, NUM_STEPS))(u_g, nu_g)
        g1 = float(jnp.linalg.norm(u_a - u_b) / (jnp.linalg.norm(u_b) + 1e-30))
        print(f"gate 1 zeroed-MLP solver vs production PODLSGNSolver: {g1:.3e}",
              flush=True)
        if not np.isfinite(g1) or g1 > 1e-10:
            raise SystemExit(f"ABORT gate 1: {g1:.3e} > 1e-10")
        del sol0, prod, chart0

        # ---- gate 2: chart decode vs the trainer's own decode --------------
        z_ref = jnp.asarray(ck["z_ref"], dtype=wdt)
        xh = jax.vmap(chart.decode)(z_ref)
        ref = jnp.asarray(ck["xhat_ref"], dtype=wdt)
        g2 = float(jnp.max(jnp.linalg.norm(xh - ref, axis=1)
                           / (jnp.linalg.norm(ref, axis=1) + 1e-30)))
        print(f"gate 2 chart decode vs trained decoder (podmlp.make_decode, "
              f"f32 reference): {g2:.3e}", flush=True)
        if not np.isfinite(g2) or g2 > 1e-4:
            raise SystemExit(f"ABORT gate 2: {g2:.3e} > 1e-4 -- the chart is not "
                             "the trained model.")

        # ---- gate 3: row-gather vs full decode ------------------------------
        z0 = chart.encode(u_g)
        s_val = chart.at_stencil(z0)
        s_ref = chart.decode(z0)[jnp.asarray(st_flat)]
        g3 = float(jnp.max(jnp.abs(s_val - s_ref))
                   / (jnp.max(jnp.abs(s_ref)) + 1e-30))
        print(f"gate 3 at_stencil row-gather vs full decode: {g3:.3e}", flush=True)
        tol3 = 1e-10 if args.x64 else 1e-5
        if not np.isfinite(g3) or g3 > tol3:
            raise SystemExit(f"ABORT gate 3: {g3:.3e} > {tol3:.1e}")

        # ---- gate 4: analytic Jacobian vs jacfwd ---------------------------
        u_ref = jnp.asarray(U_val[val_starts[0] + 10], dtype=wdt)
        z_r = chart.encode(u_ref)
        up = u_ref[solver.eq_flat_j]
        J_ad = jax.jit(jax.jacfwd(lambda zz: solver.residual(zz, up, nu_g)))(z_r)
        J_an = jax.jit(lambda zz: solver.jacobian(zz, nu_g))(z_r)
        scale = float(jnp.max(jnp.abs(J_ad)))
        g4 = float(jnp.max(jnp.abs(J_an - J_ad)) / max(scale, 1e-300))
        print(f"gate 4 analytic vs jacfwd Jacobian: {g4:.3e} (shape {tuple(J_ad.shape)})",
              flush=True)
        tol4 = 1e-10 if args.x64 else 1e-4
        if not np.isfinite(g4) or g4 > tol4:
            raise SystemExit(f"ABORT gate 4: {g4:.3e} > {tol4:.1e}")
        gates = {"gate1_zeroed_vs_production_solver": g1,
                 "gate2_chart_vs_trained_decoder": g2,
                 "gate3_rowgather_vs_full_decode": g3,
                 "gate4_analytic_vs_jacfwd_jacobian": g4}
        del J_ad, J_an

    # ---------------- FOM variants ---------------------------------------
    rollout_jit = jax.jit(lambda u0, nu: solver.rollout(u0, nu, NUM_STEPS))
    fom_rollout = jax.jit(lambda u, nu: fom.rollout(u, nu, NUM_STEPS))
    fom_step_jit = jax.jit(fom.step)
    fom_res_jit = jax.jit(lambda u, up, nu: fom.be_residual(u, up, nu))

    variants = {}
    specs = fa.variant_specs(KRYLOV_TOL)
    for name in [s for s in args.strong_fom.split(",") if s.strip()]:
        if name not in specs:
            raise SystemExit(f"unknown --strong-fom {name!r}")
        st = fa.make_step(fom, DT, NEWTON_MAX_ITERS, NEWTON_RTOL,
                          maxiter=KRYLOV_MAXITER, **specs[name])
        variants[name] = (jax.jit(st), jax.jit(fa.rollout_of(st, NUM_STEPS)))

    # frozen-POD reference floor at this k (the accuracy the MLP must beat)
    V_j, m_j = jnp.asarray(np.asarray(V, dtype=np.float64 if args.x64 else np.float32)), \
        jnp.asarray(np.asarray(mean, dtype=np.float64 if args.x64 else np.float32))

    rel_l2s, pod_floors, mlp_floors = [], [], []
    rom_times, per_traj = [], []
    fom_time_by_variant = {"shipped": []}
    sp_by_variant = {"shipped": []}
    conv_by_variant = {"shipped": []}
    for nm in variants:
        fom_time_by_variant[nm] = []
        sp_by_variant[nm] = []
        conv_by_variant[nm] = []

    for i, s in enumerate(val_starts):
        u0 = jnp.asarray(U_val[s], dtype=wdt)
        nu = jnp.asarray(float(val_nu[i]), dtype=wdt)
        uT = jnp.asarray(U_val[s + NUM_STEPS], dtype=wdt)

        u_rom, iters = rollout_jit(u0, nu)
        u_rom.block_until_ready()
        it_arr = np.asarray(iters)
        n_cap = int((it_arr >= args.gn_max_iters).sum())
        rom_conv = bool(n_cap == 0)
        rel = float(jnp.linalg.norm(u_rom - uT) / jnp.linalg.norm(uT))
        a_pod = V_j.T @ (uT - m_j)
        pf = float(jnp.linalg.norm(uT - (m_j + V_j @ a_pod)) / jnp.linalg.norm(uT))
        mf = float(jnp.linalg.norm(uT - chart.decode(chart.encode(uT)))
                   / jnp.linalg.norm(uT))

        ratios, nfail = [], 0
        u = u0
        for _ in range(NUM_STEPS):
            r0 = float(jnp.linalg.norm(fom_res_jit(u, u, nu)))
            un = fom_step_jit(u, nu)
            rf = float(jnp.linalg.norm(fom_res_jit(un, u, nu)))
            ratios.append(rf / max(r0, 1e-300))
            u = un
        nfail = int(sum(1 for r in ratios if not (r <= NEWTON_RTOL)))
        fom_conv = bool(nfail == 0)

        rom_fn = lambda: rollout_jit(u0, nu)
        entry = {"traj": i, "nu": float(nu), "rel_l2": rel,
                 "pod_proj_floor": pf, "podmlp_recon_floor": mf,
                 "rom_converged": rom_conv, "rom_gn_steps_at_cap": n_cap,
                 "rom_gn_iters_mean": float(it_arr.mean()),
                 "rom_gn_iters_max": int(it_arr.max()),
                 "fom_variants": {}}

        # shipped
        try:
            r = bench.speedup(lambda: fom_rollout(u0, nu), rom_fn,
                              repeats=args.repeats, warmup=args.warmup,
                              fom_converged=fom_conv, rom_converged=rom_conv)
            t_f, t_r, sp = r["fom_time_s"], r["rom_time_s"], r["speedup"]
        except bench.BenchError as e:
            _, t_f, _ = bench.timed_median(lambda: fom_rollout(u0, nu),
                                           repeats=args.repeats, warmup=args.warmup)
            _, t_r, _ = bench.timed_median(rom_fn, repeats=args.repeats,
                                           warmup=args.warmup)
            sp = None
            entry["shipped_gate_error"] = str(e)
        rom_times.append(t_r)
        fom_time_by_variant["shipped"].append(t_f)
        conv_by_variant["shipped"].append(fom_conv)
        if sp is not None:
            sp_by_variant["shipped"].append(sp)
        entry["fom_variants"]["shipped"] = {
            "time_s": t_f, "converged": fom_conv,
            "steps_failing_newton_rtol": nfail, "speedup": sp}

        u_shipped_end = fom_rollout(u0, nu)
        for nm, (stj, rlj) in variants.items():
            _, vfail, _, u_end = fa.gate(stj, fom_res_jit, u0, nu, NUM_STEPS,
                                         NEWTON_RTOL)
            vconv = bool(vfail == 0)
            drift = float(jnp.linalg.norm(u_end - u_shipped_end)
                          / jnp.maximum(jnp.linalg.norm(u_shipped_end), 1e-300))
            try:
                rv = bench.speedup(lambda: rlj(u0, nu), rom_fn,
                                   repeats=args.repeats, warmup=args.warmup,
                                   fom_converged=vconv, rom_converged=rom_conv)
                t_v, sp_v = rv["fom_time_s"], rv["speedup"]
            except bench.BenchError:
                _, t_v, _ = bench.timed_median(lambda: rlj(u0, nu),
                                               repeats=args.repeats,
                                               warmup=args.warmup)
                sp_v = None
            fom_time_by_variant[nm].append(t_v)
            conv_by_variant[nm].append(vconv)
            if sp_v is not None:
                sp_by_variant[nm].append(sp_v)
            entry["fom_variants"][nm] = {
                "time_s": t_v, "converged": vconv,
                "steps_failing_newton_rtol": vfail, "speedup": sp_v,
                "rel_drift_vs_shipped": drift}

        rel_l2s.append(rel); pod_floors.append(pf); mlp_floors.append(mf)
        per_traj.append(entry)
        print(f"  traj {i}: nu={float(nu):.4e} rel_l2={rel:.4e} "
              f"pod_floor={pf:.4e} podmlp_floor={mf:.4e} "
              f"GN iters={it_arr.mean():.1f} cap={n_cap}/{NUM_STEPS} "
              f"t_rom={t_r:.4f}s " +
              " ".join(f"{nm}:{v['time_s']:.4f}s/"
                       f"{'%.2fx' % v['speedup'] if v['speedup'] else 'GATED'}"
                       for nm, v in entry["fom_variants"].items()), flush=True)

    n_traj = len(per_traj)
    rom_all_conv = all(p["rom_converged"] for p in per_traj)

    # ---- denominator selection: fastest GATE-PASSING variant, measured ----
    cand = {}
    for nm in fom_time_by_variant:
        if len(conv_by_variant[nm]) == n_traj and all(conv_by_variant[nm]):
            cand[nm] = float(np.median(fom_time_by_variant[nm]))
    denom = min(cand, key=cand.get) if cand else None
    speedup_final = None
    if denom is not None and rom_all_conv and len(sp_by_variant[denom]) == n_traj:
        speedup_final = float(np.median(sp_by_variant[denom]))

    print(f"\nmean rel_l2 {np.mean(rel_l2s):.4e} | mean POD floor "
          f"{np.mean(pod_floors):.4e} | mean POD+MLP floor {np.mean(mlp_floors):.4e}")
    print(f"FOM candidate denominators (gate-passing, median s): {cand}")
    print(f"DENOMINATOR USED: {denom}  ->  speedup "
          f"{'%.2fx' % speedup_final if speedup_final else 'GATED(no ratio)'}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec = {
        "pde": f"burgers{d}d", "N": N, "method": "nm_rom",
        "variant": args.variant,
        "rel_l2": float(np.mean(rel_l2s)),
        "solve_time_s": float(np.median(rom_times)),
        "fom_time_s": (cand[denom] if denom else None),
        "speedup": speedup_final,
        "fom_converged": bool(denom is not None),
        "rom_converged": bool(rom_all_conv),
        "precision": prec, "fom_precision": prec, "rom_precision": prec,
        "config": {"chart": "POD+MLP (frozen V, tiny-init MLP correction)",
                   "solver": "LS-GN collocation (fast-jac, lean-step)",
                   "k": k, "width": int(ck["width"]), "depth": int(ck["depth"]),
                   "eq_mode": eq_mode, "n_centres": int(centers.shape[0]),
                   "gn_max_iters": args.gn_max_iters,
                   "gn_rel_tol": args.gn_rel_tol, "lm_damping": args.lm_damping,
                   "x64": bool(args.x64), "pod_file": str(args.pod),
                   "data_file": str(args.data), "ckpt": str(args.ckpt)},
        "protocol": bench.protocol_stamp(repeats=args.repeats, warmup=args.warmup,
                                         interleaved=True,
                                         converged=bool(speedup_final is not None)),
        "seed": args.eq_seed, "timestamp": stamp, "notes": args.notes,
        "extra": {
            "pod_sha256": pod_hash, "ckpt_sha256": sha256(args.ckpt),
            "data_path": str(args.data), "k": k, "width": int(ck["width"]),
            "n_eq": int(centers.shape[0]), "precision": prec,
            "knob_setting": args.variant,
            "gn_max_iters": args.gn_max_iters, "gn_rel_tol": args.gn_rel_tol,
            "gates": gates,
            "denominator_variant": denom,
            "denominator_median_time_s": (cand[denom] if denom else None),
            "denominator_candidates_gate_passing_median_s": cand,
            "fom_median_time_by_variant_s": {kk: float(np.median(v))
                                             for kk, v in fom_time_by_variant.items()},
            "speedup_by_variant": {kk: (float(np.median(v)) if len(v) == n_traj else None)
                                   for kk, v in sp_by_variant.items()},
            "per_traj_rel_l2": rel_l2s,
            "per_traj_pod_proj_floor": pod_floors,
            "per_traj_podmlp_recon_floor": mlp_floors,
            "per_traj_rom_time_s": rom_times,
            "across_traj_rom_time_spread": _spread(rom_times),
            "mean_pod_proj_floor": float(np.mean(pod_floors)),
            "mean_podmlp_recon_floor": float(np.mean(mlp_floors)),
            "decoder_training_metrics": ck.get("metrics"),
            "n_train_snapshots": ck.get("n_train_snapshots"),
            "train_wall_s": ck.get("train_wall_s"),
            "per_traj_detail": per_traj,
            "newton_rtol": float(NEWTON_RTOL),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": os.environ.get("PODMLP_GPU"),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp = args.out_dir / f"burgers{d}d_n{N}_nm_rom_{args.variant}_{stamp}.json"
    fp.write_text(json.dumps(rec, indent=2))
    print(f"Wrote {fp}", flush=True)


if __name__ == "__main__":
    main()
