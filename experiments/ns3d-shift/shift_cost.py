"""Where the co-moving query's time goes, and whether a bigger ROM step buys it back.

pilot01 (job 4176514) settled accuracy: the solved frame with no gauge reaches
0.449 % evolved worst on development, 0/16 over 5 %, but the complete query costs
23.9 ms against a 5.5 ms comparator. The reduced model is not accuracy-limited,
so the only lever left is cost. This job sweeps the test count M and the ROM step
dt at gauge 0, times every setting as a complete query paired with CNAB2 in one
allocation, and splits out the grid-sized pieces the ROM cannot avoid.

Separately compiled pieces do not add up exactly to the compiled whole-trajectory
query; the whole query is timed too and is the number that counts.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
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
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402
from shift_pilot import (paired_times, case_errors, sha256_file,  # noqa: E402
                         summarize_diagnostics, run_arm, arm_arguments)

log = D.log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(n=8, dt_truth=0.005, horizon=0.05, train_cases=8, dev_cases=2,
                   rank=8, modes_ladder=[32], rom_dt_ladder=[0.01],
                   fom_dt_ladder=[0.005, 0.01], timing_repetitions=2, gram_block=8)
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["train_seed"]), int(cfg["dev_seed"])):
            raise RuntimeError("refusing to read a closed seed")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu" and not args.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    log(f"disk_free_gb={shutil.disk_usage(out).free / 2**30:.1f} devices={jax.devices()}")

    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    rank = int(cfg["rank"])
    target = float(cfg["target_relative"])
    report = dict(schema="ns3d-shift-cost-v1", config=cfg, smoke=bool(args.smoke),
                  final_cohort_opened=False, device=str(jax.devices()),
                  source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-shift/shift_rom.py",
                      "experiments/ns3d-shift/shift_cost.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"

    train_par = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_par = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["dev_parameter_sha256"] = D.sha256_array(dev_par)
    train, _, _ = D.generate(train_par, n, float(cfg["dt_truth"]), horizon)
    dev, _, _ = D.generate(dev_par, n, float(cfg["dt_truth"]), horizon)
    np.save(out / "dev_truth.npy", dev)
    viscosities = dev_par[:, -1]

    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    snaps = centered.reshape(len(centered) * centered.shape[1], -1)
    basis, _, _ = D.pod_basis(snaps, rank, int(cfg.get("gram_block", 512)))
    del centered, snaps, train
    gc.collect()
    basis, _ = D.orthonormalize_prefix(basis, rank)
    basis = np.ascontiguousarray(basis)
    Gj = jnp.asarray(basis)

    geom = F.geometry(n)
    case = int(cfg.get("timing_case", 0))
    u0 = jnp.asarray(dev[case, 0])
    nu = float(viscosities[case])
    reps = int(cfg["timing_repetitions"])

    @jax.jit
    def initial_piece(u0, basis):
        c0 = SR.grid_centroid(u0)
        return basis.T @ SR.shift_field(u0, -c0 * n).ravel(), c0

    @jax.jit
    def output_piece(a, c, basis):
        return SR.shift_field((basis @ a).reshape(3, n, n, n), c * n)

    a0, c0 = initial_piece(u0, Gj)
    common = {"piece_initial_centering_projection": lambda: initial_piece(u0, Gj),
              "piece_one_output_reconstruction": lambda: output_piece(a0, c0, Gj)}
    for fdt in cfg["fom_dt_ladder"]:
        fsteps = D.nsteps_for(float(fdt), horizon)
        solver = F.make_solver(float(fdt), fsteps, fsteps // 5)
        common[f"CNAB2_dt{fdt}"] = (lambda s=solver: s(u0, nu, geom))

    log("CNAB2 comparator accuracy on the full development cohort")
    fom = {}
    for fdt in cfg["fom_dt_ladder"]:
        fsteps = D.nsteps_for(float(fdt), horizon)
        solver = F.make_solver(float(fdt), fsteps, fsteps // 5)
        pred = np.empty_like(dev)
        for c in range(len(dev)):
            pred[c] = np.asarray(solver(jnp.asarray(dev[c, 0]), float(viscosities[c]), geom))
        fom[str(fdt)] = D.stats_from_cases(case_errors(pred, dev), target)
        log(f"CNAB2 dt={fdt} evolved worst {fom[str(fdt)]['evolved_worst']:.6f}")
    report["cnab2"] = fom

    results = {}
    for modes in cfg["modes_ladder"]:
        ops, ops_report = SR.build_operators(basis, n, int(modes))
        for dtv in cfg["rom_dt_ladder"]:
            steps = D.nsteps_for(float(dtv), horizon)
            if steps % 5:
                log(f"skipping dt={dtv}: {steps} steps is not five equal blocks")
                continue
            key = f"M{modes}_dt{dtv}"
            runner = SR.make_shift_run(float(dtv), steps, steps // 5, n, rank,
                                       mode="free", gauge=float(cfg["gauge"]),
                                       budget=int(cfg["budget"]), gtol=float(cfg["gtol"]))
            argv = arm_arguments(basis, ops, steps)
            try:
                fields, centers, diag = run_arm(runner, argv, dev, viscosities, dev.shape)
            except Exception as exc:                          # noqa: BLE001
                results[key] = dict(modes=int(modes), dt=float(dtv),
                                    failed=f"{type(exc).__name__}: {exc}")
                log(f"{key} failed: {exc}")
                continue
            errors = case_errors(fields, dev)
            stats = D.stats_from_cases(errors, target)
            # one output instead of five isolates the reconstruction cost
            single = SR.make_shift_run(float(dtv), steps, steps, n, rank, mode="free",
                                       gauge=float(cfg["gauge"]), budget=int(cfg["budget"]),
                                       gtol=float(cfg["gtol"]))
            timed = dict(common)
            timed[f"query_{key}"] = (lambda rn=runner, aa=argv: rn(u0, nu, *aa))
            timed[f"query_one_output_{key}"] = (lambda rn=single, aa=argv: rn(u0, nu, *aa))
            timings, last = paired_times(timed, reps)
            timed_fields = np.asarray(last[f"query_{key}"][0])
            timed_err = D.rel_rows(timed_fields.reshape(dev.shape[1:]), dev[case], dev[case, 0])
            results[key] = dict(modes=int(modes), dt=float(dtv), rank=rank, steps=int(steps),
                                operator_checks=ops_report, stats=stats,
                                errors=errors.tolist(), timing=timings,
                                solver=summarize_diagnostics(diag, int(cfg["budget"])),
                                timed_query_errors=[float(x) for x in timed_err])
            log(f"{key}: evolved worst {stats['evolved_worst']:.6f} "
                f"({stats['cases_evolved_over_target']}/{stats['cases']} over target), "
                f"query {timings[f'query_{key}']['median_ms']:.3f} ms, "
                f"one-output {timings[f'query_one_output_{key}']['median_ms']:.3f} ms, "
                f"median iters {results[key]['solver']['median_iterations']}")
            if stats["evolved_worst"] <= target:
                np.save(out / f"fields_{key}.npy", fields)
        del ops
        gc.collect()
    report["settings"] = results
    D.dump_json(out / "summary.json", report)
    log("done")


if __name__ == "__main__":
    main()
