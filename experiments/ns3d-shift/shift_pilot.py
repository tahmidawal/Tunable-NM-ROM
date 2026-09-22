"""Development pilot for the co-moving shift ROM: floors, checks, arms, cost.

Stages, in order, each gated by the previous one:
  1. floors     A0 fixed uncentred POD (must fail), A1 centered POD + oracle shift
  2. checks     operator validation, delta==0 parity with ns3d_rom.make_run,
                complete-query equivariance, translation-tangent rank, Jacobian
                conditioning including the deflated shift block
  3. arms       B0 (frozen frame), B1 (solved delta), B2 (solved delta + gauge)
                on a base setting plus one-factor variations
  4. cost       complete-query paired timing against the centroid tracker and
                CNAB2, errors taken from the timed outputs

The final cohort is never drawn here. Development seed only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ("experiments/ns3d", "experiments/ns2d", "experiments/separable-decoder",
              "experiments/ns3d-grok"):
    sys.path.insert(0, str(ROOT / extra))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import ns3d_fom as F  # noqa: E402
import ns3d_rom as R  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402

log = D.log


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def case_errors(pred, truth):
    """(cases, times) relative L2 against each case's own initial field."""
    out = np.empty(truth.shape[:2], dtype=np.float64)
    for case in range(len(truth)):
        out[case] = D.rel_rows(pred[case], truth[case], truth[case, 0])
        check = D.sumsq_rows(pred[case], truth[case], truth[case, 0])
        if float(np.max(np.abs(out[case] - check))) > 1e-12:
            raise RuntimeError("error reductions disagree")
    return out


def paired_times(callables, reps, burn=2):
    """Interleaved repetitions so a drifting device affects every arm alike."""
    for fn in callables.values():
        for _ in range(burn):
            jax.block_until_ready(fn())
    samples = {name: [] for name in callables}
    last = {}
    for _ in range(reps):
        for name, fn in callables.items():
            start = time.perf_counter()
            value = fn()
            jax.block_until_ready(value)
            samples[name].append((time.perf_counter() - start) * 1e3)
            last[name] = value
    return ({name: dict(median_ms=float(np.median(v)), min_ms=float(np.min(v)),
                        max_ms=float(np.max(v)), repetitions=v)
             for name, v in samples.items()}, last)


# --------------------------------------------------------------------- stages --

def build_arm(cfg, basis, ops, nsteps, out_every, n, r, mode, gauge, budget):
    return SR.make_shift_run(cfg["rom_dt"], nsteps, out_every, n, r, mode=mode,
                             gauge=gauge, budget=budget, gtol=float(cfg["gtol"]))


def arm_arguments(basis, ops, nsteps):
    return (jnp.asarray(basis), jnp.asarray(ops["A"]), jnp.asarray(ops["T"]),
            jnp.asarray(ops["lam"]), jnp.asarray(ops["Dd"]), jnp.asarray(ops["S"]),
            jnp.zeros((nsteps, 3)))


def run_arm(runner, args, dev, viscosities, truth_shape):
    fields = np.empty(truth_shape, dtype=np.float64)
    centers = []
    diagnostics = []
    for case in range(len(dev)):
        out, cen, info = runner(jnp.asarray(dev[case, 0]), float(viscosities[case]), *args)
        out = np.asarray(out)
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"nonfinite ROM field on case {case}")
        fields[case] = out.reshape(truth_shape[1:])
        centers.append(np.asarray(cen))
        rn, it, reason, gn, delta = (np.asarray(x) for x in info)
        diagnostics.append(dict(residual=rn, iterations=it, reason=reason,
                                gradient=gn, delta=delta))
    return fields, np.stack(centers), diagnostics


def summarize_diagnostics(diagnostics, budget):
    it = np.concatenate([d["iterations"] for d in diagnostics])
    reason = np.concatenate([d["reason"] for d in diagnostics])
    res = np.concatenate([d["residual"] for d in diagnostics])
    delta = np.concatenate([d["delta"] for d in diagnostics])
    names = {0: "budget", 1: "tolerance", 2: "tiny_step", 3: "rejected", 4: "stationary"}
    counts = {names[k]: int(np.sum(reason == k)) for k in names}
    return dict(median_iterations=float(np.median(it)), max_iterations=int(it.max()),
                budget=int(budget), stopping_reasons=counts,
                fraction_on_budget=float(counts["budget"] / max(len(reason), 1)),
                median_residual=float(np.median(res)), max_residual=float(res.max()),
                max_abs_delta=float(np.max(np.abs(delta))),
                median_abs_delta=float(np.median(np.abs(delta))))


def centroid_diagnostic(fields, truth):
    """centroid(u_ROM) against centroid(u_true). Never c against a truth centroid."""
    gaps = []
    for case in range(len(truth)):
        for instant in range(truth.shape[1]):
            gaps.append(np.max(D.torus_delta(D.energy_centroid(fields[case, instant]),
                                             D.energy_centroid(truth[case, instant]))))
    gaps = np.asarray(gaps)
    return dict(median=float(np.median(gaps)), worst=float(gaps.max()))


def projection_inequality(fields, truth, basis, centers, n):
    """At the ROM's own predicted shift, the orthogonal projection of the truth
    into the shifted bank cannot be worse than the ROM field."""
    worst = -np.inf
    G = np.asarray(basis)
    for case in range(len(truth)):
        for instant in range(truth.shape[1]):
            c = centers[case, instant]
            centered = D.fourier_shift(truth[case, instant], -c * n)
            best = D.fourier_shift((G @ (G.T @ centered.ravel())).reshape(3, n, n, n), c * n)
            num = np.linalg.norm(best - truth[case, instant])
            rom = np.linalg.norm(fields[case, instant] - truth[case, instant])
            den = max(np.linalg.norm(truth[case, 0]), 1e-300)
            worst = max(worst, (num - rom) / den)
    return float(worst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(n=8, dt_truth=0.005, horizon=0.05, rom_dt=0.01, train_cases=8,
                   dev_cases=2, ranks=[8], base_rank=8, base_modes=32,
                   modes_ladder=[32], dt_ladder=[0.01], gauge_ladder=[0.0, 1.0],
                   budget_ladder=[20], timing_repetitions=2, gram_block=8,
                   fom_dt_ladder=[0.005, 0.01])
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["train_seed"]), int(cfg["dev_seed"])):
            raise RuntimeError("refusing to read a closed seed")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 2**30
    log(f"disk_free_gb={free_gb:.1f} devices={jax.devices()}")
    if free_gb < (2.0 if args.smoke else 30.0):
        raise RuntimeError(f"not enough free disk: {free_gb:.1f} GB")
    if jax.default_backend() != "gpu" and not args.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    if jnp.zeros((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError("float64 is not enabled")

    job_start = time.time()
    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    report = dict(schema="ns3d-shift-pilot-v1", config=cfg, smoke=bool(args.smoke),
                  final_cohort_opened=False, device=str(jax.devices()),
                  source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-shift/shift_rom.py",
                      "experiments/ns3d-shift/shift_pilot.py",
                      "experiments/ns3d/ns3d_fom.py",
                      "experiments/ns3d/ns3d_rom.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"

    train_par = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_par = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["train_parameter_sha256"] = D.sha256_array(train_par)
    report["dev_parameter_sha256"] = D.sha256_array(dev_par)
    log("generating trajectories")
    train, steps, every = D.generate(train_par, n, float(cfg["dt_truth"]), horizon)
    dev, _, _ = D.generate(dev_par, n, float(cfg["dt_truth"]), horizon)
    D.require_finite(train, "training")
    D.require_finite(dev, "development")
    np.save(out / "dev_truth.npy", dev)
    report["truth"] = dict(dt=float(cfg["dt_truth"]), steps=steps, out_every=every,
                           frames=int(dev.shape[1]),
                           dev_truth_sha256=sha256_file(out / "dev_truth.npy"))
    report["shift_self_check"] = D.shift_self_check(dev[0, 0])
    if not report["shift_self_check"]["passed"]:
        raise RuntimeError("Fourier shift / centroid identity check failed")
    viscosities = dev_par[:, -1]
    target = float(cfg["target_relative"])
    block = int(cfg.get("gram_block", 512))
    ranks = [int(x) for x in cfg["ranks"]]

    # ---- stage 1: floors -----------------------------------------------------
    log("A0: fixed uncentred POD (must-fail control)")
    plain_snaps = train.reshape(len(train) * train.shape[1], -1)
    plain_basis, _, _ = D.pod_basis(plain_snaps, max(ranks), block)
    a0 = {}
    for rank in [x for x in ranks if x <= plain_basis.shape[1]]:
        q, _ = D.orthonormalize_prefix(plain_basis, rank)
        errors, _ = D.project_errors(q, dev)
        a0[str(rank)] = D.stats_from_cases(errors, target)
        log(f"A0 rank {rank} evolved worst {a0[str(rank)]['evolved_worst']:.6f}")
    del plain_basis, plain_snaps
    gc.collect()
    report["A0_fixed_bank_floor"] = a0
    base_rank = int(cfg["base_rank"])
    if a0[str(base_rank)]["cases_evolved_over_target"] == 0 and not args.smoke:
        raise RuntimeError("must-fail control A0 passed the target; harness suspect")

    log("A1: centered POD with an oracle per-time shift")
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    centered_snaps = centered.reshape(len(centered) * centered.shape[1], -1)
    cbasis, cenergy, corth = D.pod_basis(centered_snaps, max(ranks), block)
    del centered, centered_snaps
    gc.collect()
    available = int(cbasis.shape[1])
    log(f"centered POD available rank {available}")
    a1 = {}
    for rank in sorted({x for x in ranks if x <= available} | {available}):
        q, _ = D.orthonormalize_prefix(cbasis, rank)
        errors, _, travel = D.oracle_shift_errors(q, dev)
        a1[str(rank)] = dict(stats=D.stats_from_cases(errors, target), travel=travel)
        log(f"A1 rank {rank} evolved worst {a1[str(rank)]['stats']['evolved_worst']:.6f}")
    report["A1_centered_oracle_floor"] = dict(
        available_rank=available, orthogonality=corth,
        spectrum_head=[float(x) for x in cenergy[:8]], ranks=a1)
    if a1[str(base_rank)]["stats"]["evolved_worst"] >= a0[str(base_rank)]["evolved_worst"]:
        report["stopped"] = "A1 did not collapse relative to A0; stop rule 1"
        D.dump_json(out / "summary.json", report)
        return

    # ---- stage 2: operators and checks ---------------------------------------
    base_modes = int(cfg["base_modes"])
    base_dt = float(cfg["rom_dt"])
    base_budget = int(cfg["budget_ladder"][-1])
    basis_cache = {}

    def get_basis(rank):
        if rank not in basis_cache:
            q, _ = D.orthonormalize_prefix(cbasis, rank)
            basis_cache[rank] = np.ascontiguousarray(q)
        return basis_cache[rank]

    ops_cache = {}

    def get_ops(rank, modes):
        key = (rank, modes)
        if key not in ops_cache:
            log(f"building operators rank={rank} modes={modes}")
            ops_cache[key] = SR.build_operators(get_basis(rank), n, modes)
        return ops_cache[key]

    base_basis = get_basis(base_rank)
    ops, ops_report = get_ops(base_rank, base_modes)
    report["operator_checks"] = ops_report
    log(f"operator checks {ops_report}")

    nsteps = D.nsteps_for(base_dt, horizon)
    out_every = nsteps // 5
    args_base = arm_arguments(base_basis, ops, nsteps)

    # delta == 0 must reproduce the existing weak ROM in the centered frame.
    zero_run = SR.make_shift_run(base_dt, nsteps, out_every, n, base_rank, mode="zero",
                                 budget=base_budget, gtol=float(cfg["gtol"]))
    existing = R.make_run(base_dt, nsteps, out_every, 0, base_rank, linear=True,
                          budget=base_budget, gtol=float(cfg["gtol"]))
    u0 = jnp.asarray(dev[0, 0])
    nu0 = float(viscosities[0])
    c0 = np.asarray(SR.grid_centroid(u0))
    v0 = jnp.asarray(D.fourier_shift(np.asarray(u0), -c0 * n))
    mine, centers0, _ = zero_run(u0, nu0, *args_base)
    mine_centered = np.stack([D.fourier_shift(np.asarray(f), -c0 * n) for f in np.asarray(mine)])
    theirs = np.asarray(existing(v0, nu0, jnp.asarray(base_basis), jnp.asarray(base_basis),
                                 jnp.eye(base_rank), jnp.asarray(ops["A"]),
                                 jnp.asarray(ops["T"]), jnp.asarray(ops["lam"]),
                                 jnp.zeros((base_rank, 0)), {}, jnp.zeros((1, 0)))[0])
    parity = float(np.linalg.norm(mine_centered.reshape(len(theirs), -1) - theirs)
                   / max(np.linalg.norm(theirs), 1e-300))
    report["zero_delta_parity_vs_ns3d_rom"] = parity
    log(f"delta==0 parity vs ns3d_rom.make_run: {parity:.3e}")
    if parity > 1e-10:
        raise RuntimeError(f"delta==0 does not reproduce the existing weak ROM: {parity}")

    # complete-query equivariance, with a torus-boundary crossing
    free_run = SR.make_shift_run(base_dt, nsteps, out_every, n, base_rank, mode="free",
                                 gauge=0.0, budget=base_budget, gtol=float(cfg["gtol"]))
    offset = np.array([0.37, -0.61, 0.94])
    plain, _, _ = free_run(u0, nu0, *args_base)
    shifted, _, _ = free_run(jnp.asarray(D.fourier_shift(np.asarray(u0), offset * n)),
                             nu0, *args_base)
    want = np.stack([D.fourier_shift(np.asarray(f), offset * n) for f in np.asarray(plain)])
    equivariance = float(np.linalg.norm(np.asarray(shifted) - want)
                         / max(np.linalg.norm(want), 1e-300))
    report["query_equivariance"] = equivariance
    log(f"complete-query equivariance: {equivariance:.3e}")

    # conditioning, at a mid-trajectory development state
    mid_field = dev[0, 2]
    cmid = D.energy_centroid(mid_field)
    amid = base_basis.T @ D.fourier_shift(mid_field, -cmid * n).ravel()
    nxt_field = dev[0, 3]
    cnxt = D.energy_centroid(nxt_field)
    anxt = base_basis.T @ D.fourier_shift(nxt_field, -cnxt * n).ravel()
    report["conditioning"] = SR.jacobian_report(ops, n, base_rank, base_dt, nu0,
                                                jnp.asarray(amid), jnp.asarray(anxt),
                                                float(cfg["base_gauge"]))
    report["translation_tangents"] = SR.translation_tangent_report(base_basis, n, amid)
    log(f"conditioning {json.dumps(report['conditioning']['no_gauge'], default=float)[:400]}")
    log(f"tangents {report['translation_tangents']}")

    # ---- stage 3: arms -------------------------------------------------------
    specs = [dict(id="B0", mode="zero", rank=base_rank, modes=base_modes, dt=base_dt,
                  gauge=0.0, budget=base_budget),
             dict(id="B1", mode="free", rank=base_rank, modes=base_modes, dt=base_dt,
                  gauge=0.0, budget=base_budget),
             dict(id="B2", mode="free", rank=base_rank, modes=base_modes, dt=base_dt,
                  gauge=float(cfg["base_gauge"]), budget=base_budget)]
    for rank in cfg["ranks"]:
        if int(rank) != base_rank and int(rank) <= available:
            specs.append(dict(id=f"B2_r{rank}", mode="free", rank=int(rank),
                              modes=max(base_modes, int(rank) + 16 + (-(int(rank) + 16) % 4)),
                              dt=base_dt, gauge=float(cfg["base_gauge"]), budget=base_budget))
    for modes in cfg["modes_ladder"]:
        if int(modes) != base_modes and int(modes) >= base_rank + 16:
            specs.append(dict(id=f"B2_M{modes}", mode="free", rank=base_rank,
                              modes=int(modes), dt=base_dt,
                              gauge=float(cfg["base_gauge"]), budget=base_budget))
    for dtv in cfg["dt_ladder"]:
        if float(dtv) != base_dt:
            specs.append(dict(id=f"B2_dt{dtv}", mode="free", rank=base_rank,
                              modes=base_modes, dt=float(dtv),
                              gauge=float(cfg["base_gauge"]), budget=base_budget))
    for gauge in cfg["gauge_ladder"]:
        if float(gauge) not in (0.0, float(cfg["base_gauge"])):
            specs.append(dict(id=f"B2_w{gauge}", mode="free", rank=base_rank,
                              modes=base_modes, dt=base_dt, gauge=float(gauge),
                              budget=base_budget))
    for budget in cfg["budget_ladder"]:
        if int(budget) != base_budget:
            specs.append(dict(id=f"B2_b{budget}", mode="free", rank=base_rank,
                              modes=base_modes, dt=base_dt,
                              gauge=float(cfg["base_gauge"]), budget=int(budget)))

    arms = {}
    kept_fields = {}
    for spec in specs:
        label = spec["id"]
        log(f"arm {label}: {spec}")
        ops_s, _ = get_ops(spec["rank"], spec["modes"])
        basis_s = get_basis(spec["rank"])
        steps_s = D.nsteps_for(spec["dt"], horizon)
        runner = SR.make_shift_run(spec["dt"], steps_s, steps_s // 5, n, spec["rank"],
                                   mode=spec["mode"], gauge=spec["gauge"],
                                   budget=spec["budget"], gtol=float(cfg["gtol"]))
        args_s = arm_arguments(basis_s, ops_s, steps_s)
        started = time.time()
        try:
            fields, centers, diag = run_arm(runner, args_s, dev, viscosities, dev.shape)
        except Exception as exc:                      # noqa: BLE001
            arms[label] = dict(spec=spec, failed=f"{type(exc).__name__}: {exc}")
            log(f"arm {label} failed: {exc}")
            continue
        errors = case_errors(fields, dev)
        entry = dict(spec=spec, stats=D.stats_from_cases(errors, target),
                     errors=errors.tolist(), seconds=time.time() - started,
                     solver=summarize_diagnostics(diag, spec["budget"]),
                     centroid_gap=centroid_diagnostic(fields, dev),
                     projection_inequality=projection_inequality(
                         fields, dev, basis_s, centers, n))
        if entry["projection_inequality"] > 1e-9:
            raise RuntimeError(f"{label} beats the projection of truth into its own "
                               f"shifted bank by {entry['projection_inequality']}")
        arms[label] = entry
        kept_fields[label] = (fields, spec)
        log(f"arm {label} evolved worst {entry['stats']['evolved_worst']:.6f} "
            f"over target {entry['stats']['cases_evolved_over_target']}/{len(dev)} "
            f"median iters {entry['solver']['median_iterations']}")
    report["arms"] = arms

    # ---- comparators ---------------------------------------------------------
    log("arm C: centroid tracker")
    geom = F.geometry(n)
    linear = R.dense_galerkin_linear(base_basis, n)
    tracker = SR.make_tracker_run(base_dt, nsteps, out_every, n)
    tracker_args = (jnp.asarray(base_basis), jnp.asarray(linear), geom)
    tfields = np.empty_like(dev)
    for case in range(len(dev)):
        tfields[case] = np.asarray(tracker(jnp.asarray(dev[case, 0]),
                                           float(viscosities[case]), *tracker_args))
    terrors = case_errors(tfields, dev)
    report["C_tracker"] = dict(stats=D.stats_from_cases(terrors, target),
                               errors=terrors.tolist(), rank=base_rank, dt=base_dt)
    log(f"tracker evolved worst {report['C_tracker']['stats']['evolved_worst']:.6f}")

    log("arm E: CNAB2 comparators")
    fom = {}
    for dtv in cfg["fom_dt_ladder"]:
        steps_f = D.nsteps_for(float(dtv), horizon)
        solver = F.make_solver(float(dtv), steps_f, steps_f // 5)
        pred = np.empty_like(dev)
        for case in range(len(dev)):
            pred[case] = np.asarray(solver(jnp.asarray(dev[case, 0]),
                                           float(viscosities[case]), geom))
        errors = case_errors(pred, dev)
        fom[str(dtv)] = D.stats_from_cases(errors, target)
        log(f"CNAB2 dt={dtv} evolved worst {fom[str(dtv)]['evolved_worst']:.6f}")
    report["E_cnab2"] = fom

    # ---- stage 4: complete-query paired timing -------------------------------
    log("timing complete queries")
    case = int(cfg.get("timing_case", 0))
    u0t, nut = jnp.asarray(dev[case, 0]), float(viscosities[case])
    timed = {}
    shortlist = [s for s in specs if s["id"] in cfg.get(
        "timing_arms", ["B0", "B1", "B2"])]
    for spec in shortlist:
        if spec["id"] not in kept_fields:
            continue
        ops_s, _ = get_ops(spec["rank"], spec["modes"])
        basis_s = get_basis(spec["rank"])
        steps_s = D.nsteps_for(spec["dt"], horizon)
        runner = SR.make_shift_run(spec["dt"], steps_s, steps_s // 5, n, spec["rank"],
                                   mode=spec["mode"], gauge=spec["gauge"],
                                   budget=spec["budget"], gtol=float(cfg["gtol"]))
        args_s = arm_arguments(basis_s, ops_s, steps_s)
        timed[spec["id"]] = (lambda rn=runner, aa=args_s: rn(u0t, nut, *aa))
    timed["C_tracker"] = lambda: tracker(u0t, nut, *tracker_args)
    for dtv in cfg["fom_dt_ladder"]:
        steps_f = D.nsteps_for(float(dtv), horizon)
        solver = F.make_solver(float(dtv), steps_f, steps_f // 5)
        timed[f"CNAB2_dt{dtv}"] = (lambda s=solver: s(u0t, nut, geom))
    timings, last = paired_times(timed, int(cfg["timing_repetitions"]))
    report["timing"] = dict(case=case, note="complete queries; initial centering, "
                            "projection and every output reconstruction are inside "
                            "the timed call; interleaved repetitions after burn-in",
                            arms=timings)
    # errors taken from the outputs of the timed calls
    timed_errors = {}
    for name, value in last.items():
        fields = np.asarray(value[0] if isinstance(value, tuple) else value)
        timed_errors[name] = [float(x) for x in D.rel_rows(
            fields.reshape(dev.shape[1:]), dev[case], dev[case, 0])]
    report["timing"]["errors_from_timed_outputs"] = timed_errors
    log(json.dumps(timings, indent=None, default=float)[:900])

    np.save(out / "tracker_fields.npy", tfields)
    for label, (fields, spec) in kept_fields.items():
        if label in ("B0", "B1", "B2"):
            np.save(out / f"fields_{label}.npy", fields)
    report["saved_fields"] = sorted(p.name for p in out.glob("*.npy"))
    report["elapsed_seconds"] = time.time() - job_start
    D.dump_json(out / "summary.json", report)
    log("done")


if __name__ == "__main__":
    main()
