"""One mesh of the resolution ladder: accuracy-cost frontier, paired against CNAB2.

Per mesh, in order:
  1. floors        centered-POD oracle-shift floor at each rank, so a miss can be
                   attributed to representation rather than to the solver
  2. operators     Phi-free build (build_operators_fast), validated per mesh
  3. reference     the pilot's LM arm, for the same-job before/after of the driver fix
  4. frontier      rank x dt x Gauss-Newton iteration count, errors on the cohort,
                   each parity-checked against the LM reference where they coincide
  5. baselines     centroid tracker, CNAB2 at a dt ladder with stability flags
  6. cost          one interleaved timing block over every arm, complete queries,
                   plus the isolated grid-sized pieces

Comparator rule: the fastest tested **stable** CNAB2 setting whose evolved worst is
no larger than the ROM's. A CNAB2 setting is unstable if any field is non-finite or
its evolved worst exceeds 100 %; unstable settings are never comparators, and where
one is cheaper than the comparator that fact is recorded rather than used.
"""
from __future__ import annotations

import argparse
import gc
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
from shift_pilot import paired_times, case_errors, sha256_file  # noqa: E402

log = D.log
UNSTABLE = 1.0          # evolved worst above this counts as a blown-up FOM


def rel_case(pred, truth, case):
    return D.rel_rows(pred, truth[case], truth[case, 0])


def run_cases(call, dev, viscosities, keep=False):
    """Errors over the cohort, holding at most one trajectory of fields at a time."""
    errors = np.empty(dev.shape[:2], dtype=np.float64)
    fields = np.empty_like(dev) if keep else None
    for case in range(len(dev)):
        out = np.asarray(call(jnp.asarray(dev[case, 0]), float(viscosities[case])))
        out = out.reshape(dev.shape[1:])
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"nonfinite field on case {case}")
        errors[case] = rel_case(out, dev, case)
        check = D.sumsq_rows(out, dev[case], dev[case, 0])
        if float(np.max(np.abs(errors[case] - check))) > 1e-12:
            raise RuntimeError("error reductions disagree")
        if keep:
            fields[case] = out
    return errors, fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(n=8, dt_truth=0.005, horizon=0.05, train_cases=8, dev_cases=2,
                   modes=32, ranks=[8], rom_dt_ladder=[0.01], iters_ladder=[3],
                   fom_dt_ladder=[0.005, 0.01], timing_repetitions=2, gram_block=8,
                   reference_rank=8, reference_dt=0.01)
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["train_seed"]), int(cfg["dev_seed"])):
            raise RuntimeError("refusing to read a closed seed")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 2**30
    log(f"disk_free_gb={free_gb:.1f} devices={jax.devices()}")
    if jax.default_backend() != "gpu" and not args.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")

    job_start = time.time()
    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    modes = int(cfg["modes"])
    ranks = [int(x) for x in cfg["ranks"]]
    target = float(cfg["target_relative"])
    report = dict(schema="ns3d-shift-ladder-v1", config=cfg, smoke=bool(args.smoke),
                  final_cohort_opened=False, device=str(jax.devices()),
                  source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-shift/shift_rom.py",
                      "experiments/ns3d-shift/shift_ladder.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"

    train_par = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_par = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["dev_parameter_sha256"] = D.sha256_array(dev_par)
    log("generating trajectories")
    train, _, _ = D.generate(train_par, n, float(cfg["dt_truth"]), horizon)
    dev, _, _ = D.generate(dev_par, n, float(cfg["dt_truth"]), horizon)
    D.require_finite(train, "training")
    D.require_finite(dev, "development")
    np.save(out / "dev_truth.npy", dev)
    viscosities = dev_par[:, -1]
    report["shift_self_check"] = D.shift_self_check(dev[0, 0])
    if not report["shift_self_check"]["passed"]:
        raise RuntimeError("Fourier shift / centroid identity check failed")

    log("centering the training snapshots in place (GPU), cross-checked against numpy")
    centre_jit = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))
    cross = 0.0
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            field = train[case, instant]
            if case < 2 and instant < 2:
                want, _ = D.center_field(field)
                got = np.asarray(centre_jit(jnp.asarray(field)))
                cross = max(cross, float(np.linalg.norm(got - want)
                                         / max(np.linalg.norm(want), 1e-300)))
            train[case, instant] = np.asarray(centre_jit(jnp.asarray(field)))
    report["gpu_centering_vs_numpy"] = cross
    log(f"gpu centering vs numpy: {cross:.3e}")
    if cross > 1e-10:
        raise RuntimeError(f"GPU centering disagrees with the numpy path: {cross}")
    basis_all, energy, _ = D.pod_basis(
        train.reshape(len(train) * train.shape[1], -1), max(ranks), int(cfg["gram_block"]))
    del train
    gc.collect()
    available = int(basis_all.shape[1])
    log(f"centered POD available rank {available}")

    banks, ops = {}, {}
    floors = {}
    for rank in [r for r in ranks if r <= available]:
        q, _ = D.orthonormalize_prefix(basis_all, rank)
        banks[rank] = np.ascontiguousarray(q)
        errors, _, travel = D.oracle_shift_errors(banks[rank], dev)
        floors[str(rank)] = dict(stats=D.stats_from_cases(errors, target), travel=travel)
        log(f"floor rank {rank} evolved worst "
            f"{floors[str(rank)]['stats']['evolved_worst']:.6f}")
        ops[rank], rep = SR.build_operators_fast(banks[rank], n, modes,
                                                 check_modes=int(cfg["check_modes"]))
        report.setdefault("operator_checks", {})[str(rank)] = rep
        log(f"operators rank {rank}: {rep}")
    del basis_all
    gc.collect()
    report["floors"] = dict(available_rank=available, ranks=floors,
                            spectrum_head=[float(x) for x in energy[:8]])

    arg_cache, runner_cache = {}, {}

    def frozen_args(rank):
        if rank not in arg_cache:
            o = ops[rank]
            arg_cache[rank] = (jnp.asarray(banks[rank]), jnp.asarray(o["A"]),
                               jnp.asarray(o["T"]), jnp.asarray(o["lam"]),
                               jnp.asarray(o["Dd"]))
        return arg_cache[rank]

    def frozen_runner(rank, dtv, iters, outputs=5):
        """One compile per distinct configuration, shared by accuracy and timing."""
        key = (rank, float(dtv), int(iters), int(outputs))
        if key not in runner_cache:
            steps = D.nsteps_for(float(dtv), horizon)
            every = steps // outputs
            runner_cache[key] = SR.make_frozen_run(
                float(dtv), steps, every, n, rank, iters=int(iters),
                damping=float(cfg["damping"]), extrapolate=bool(cfg["extrapolate"]),
                diagnose=False)
        return runner_cache[key]

    # ---- reference LM arm, for the same-job before/after of the driver fix -----
    ref_rank, ref_dt = int(cfg["reference_rank"]), float(cfg["reference_dt"])
    ref_steps = D.nsteps_for(ref_dt, horizon)
    reference = SR.make_shift_run(ref_dt, ref_steps, ref_steps // 5, n, ref_rank,
                                  mode="free", gauge=0.0, budget=int(cfg["lm_budget"]),
                                  gtol=float(cfg["lm_gtol"]))
    ref_args = frozen_args(ref_rank) + (jnp.zeros((ref_rank, ref_rank, 3)).transpose(2, 0, 1),
                                        jnp.zeros((ref_steps, 3)))
    ref_errors, ref_fields = run_cases(
        lambda u, nu: reference(u, nu, *ref_args)[0], dev, viscosities, keep=True)
    report["reference_lm"] = dict(rank=ref_rank, dt=ref_dt,
                                  stats=D.stats_from_cases(ref_errors, target),
                                  errors=ref_errors.tolist())
    log(f"reference LM rank {ref_rank} dt {ref_dt} evolved worst "
        f"{report['reference_lm']['stats']['evolved_worst']:.6f}")

    # ---- frontier -------------------------------------------------------------
    frontier = {}
    best_fields, best_key, best_error = None, None, np.inf
    for rank in [r for r in ranks if r <= available]:
        argv = frozen_args(rank)
        for dtv in cfg["rom_dt_ladder"]:
            steps = D.nsteps_for(float(dtv), horizon)
            if steps % 5:
                log(f"skipping dt={dtv}: {steps} steps is not five equal blocks")
                continue
            for iters in cfg["iters_ladder"]:
                key = f"r{rank}_dt{dtv}_it{iters}"
                runner = frozen_runner(rank, dtv, iters)
                keep = True
                errors, fields = run_cases(lambda u, nu: runner(u, nu, *argv),
                                           dev, viscosities, keep=keep)
                stats = D.stats_from_cases(errors, target)
                entry = dict(rank=rank, dt=float(dtv), iters=int(iters), steps=int(steps),
                             stats=stats, errors=errors.tolist())
                if rank == ref_rank and float(dtv) == ref_dt:
                    gap = float(np.linalg.norm(fields - ref_fields)
                                / max(np.linalg.norm(ref_fields), 1e-300))
                    entry["parity_vs_reference_lm"] = gap
                    log(f"{key}: parity vs LM {gap:.3e}")
                frontier[key] = entry
                log(f"{key}: evolved worst {stats['evolved_worst']:.6f} "
                    f"({stats['cases_evolved_over_target']}/{stats['cases']} over target)")
                if stats["evolved_worst"] < best_error:
                    best_error, best_key = stats["evolved_worst"], key
                    best_fields = fields
    report["frontier"] = frontier
    if best_fields is not None and cfg.get("save_fields", True):
        np.save(out / f"fields_{best_key}.npy", best_fields)
        report["saved_field_setting"] = best_key
    del best_fields, ref_fields
    gc.collect()

    # ---- baselines ------------------------------------------------------------
    geom = F.geometry(n)
    tracker = tracker_args = None
    if cfg.get("include_tracker", True):
        linear = R.dense_galerkin_linear(banks[ref_rank], n)
        tracker = SR.make_tracker_run(ref_dt, ref_steps, ref_steps // 5, n)
        tracker_args = (jnp.asarray(banks[ref_rank]), jnp.asarray(linear), geom)
        terrors, _ = run_cases(lambda u, nu: tracker(u, nu, *tracker_args),
                               dev, viscosities)
        report["tracker"] = dict(rank=ref_rank, dt=ref_dt,
                                 stats=D.stats_from_cases(terrors, target),
                                 errors=terrors.tolist())
        log(f"tracker evolved worst {report['tracker']['stats']['evolved_worst']:.6f}")

    fom = {}
    for dtv in cfg["fom_dt_ladder"]:
        steps = D.nsteps_for(float(dtv), horizon)
        solver = F.make_solver(float(dtv), steps, steps // 5)
        try:
            errors, _ = run_cases(lambda u, nu: solver(u, nu, geom), dev, viscosities)
            stats = D.stats_from_cases(errors, target)
            unstable = bool(stats["evolved_worst"] > UNSTABLE)
        except RuntimeError as exc:
            log(f"CNAB2 dt={dtv} produced nonfinite fields: {exc}")
            stats, unstable = None, True
        fom[str(dtv)] = dict(stats=stats, unstable=unstable, steps=int(steps))
        log(f"CNAB2 dt={dtv} evolved worst "
            f"{stats['evolved_worst'] if stats else float('nan'):.6f} unstable={unstable}")
    report["cnab2"] = fom

    # ---- one interleaved timing block over every arm --------------------------
    case = int(cfg["timing_case"])
    u0, nu = jnp.asarray(dev[case, 0]), float(viscosities[case])
    timed = {}
    for rank in banks:
        argv = frozen_args(rank)
        bank_j = argv[0]

        @jax.jit
        def initial_piece(u0, bank, nn=n):
            c0 = SR.grid_centroid(u0)
            return bank.T @ SR.shift_field(u0, -c0 * nn).ravel(), c0

        a0, c0 = initial_piece(u0, bank_j)

        @jax.jit
        def output_piece(a, c, bank, nn=n):
            return SR.shift_field((bank @ a).reshape(3, nn, nn, nn), c * nn)

        timed[f"piece_initial_r{rank}"] = (
            lambda fn=initial_piece, b=bank_j: fn(u0, b))
        timed[f"piece_output_r{rank}"] = (
            lambda fn=output_piece, aa=a0, cc=c0, b=bank_j: fn(aa, cc, b))
    for key, entry in frontier.items():
        runner = frozen_runner(entry["rank"], entry["dt"], entry["iters"])
        argv = frozen_args(entry["rank"])
        timed[f"query_{key}"] = (lambda rn=runner, aa=argv: rn(u0, nu, *aa))
        if entry["iters"] == min(cfg["iters_ladder"]):
            # one output instead of five isolates the reconstruction cost
            single = frozen_runner(entry["rank"], entry["dt"], entry["iters"], outputs=1)
            timed[f"one_output_{key}"] = (lambda rn=single, aa=argv: rn(u0, nu, *aa))
    timed["reference_lm"] = lambda: reference(u0, nu, *ref_args)
    if tracker is not None:
        timed["tracker"] = lambda: tracker(u0, nu, *tracker_args)
    for dtv in cfg["fom_dt_ladder"]:
        steps = D.nsteps_for(float(dtv), horizon)
        solver = F.make_solver(float(dtv), steps, steps // 5)
        timed[f"CNAB2_dt{dtv}"] = (lambda s=solver: s(u0, nu, geom))
    log(f"timing {len(timed)} arms, {cfg['timing_repetitions']} interleaved repetitions")
    timings, last = paired_times(timed, int(cfg["timing_repetitions"]))
    report["timing"] = dict(case=case, arms=timings, note=(
        "one interleaved block for every arm on one allocation; complete queries with "
        "the initial centering, the solve and every output reconstruction inside the "
        "timed call; the timed reduced arms use diagnose=False, which produces "
        "bit-identical fields and is what a deployment runs"))
    errs = {}
    for name, value in last.items():
        if not (name.startswith("query_") or name.startswith("CNAB2")
                or name in ("reference_lm", "tracker")):
            continue
        fields = np.asarray(value[0] if isinstance(value, tuple) else value)
        errs[name] = [float(x) for x in rel_case(fields.reshape(dev.shape[1:]), dev, case)]
    report["timing"]["errors_from_timed_outputs"] = errs
    report["elapsed_seconds"] = time.time() - job_start
    D.dump_json(out / "summary.json", report)
    log("done")


if __name__ == "__main__":
    main()
