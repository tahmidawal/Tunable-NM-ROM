"""One sealed evaluation of the development-selected center tracker.

Frozen before this file is allowed to read seed 202609203: centered POD rank
64, ROM-centroid re-centering every startup step, dt=0.01. Nothing in this
script chooses a rank or a time step from the sealed errors.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

import diag_floor as D
import diag04
import jax
import jax.numpy as jnp
import ns3d_fom as F

FROZEN_RANK = 64
FROZEN_DT = 0.01
FOM_DTS = (0.004, 0.005, 0.01, 0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(n=8, dt_truth=0.01, horizon=0.05, train_cases=4, eval_cases=2,
                   eval_seed=int(cfg["dev_seed"]))
    if args.final and args.smoke:
        raise RuntimeError("smoke cannot open the final cohort")
    if args.final:
        if int(cfg["eval_seed"]) != int(cfg["final_seed_closed"]):
            raise RuntimeError("final mode requires the closed seed")
        if int(cfg["frozen_rank"]) != FROZEN_RANK or float(cfg["frozen_dt"]) != FROZEN_DT:
            raise RuntimeError("refusing to change the frozen rank or dt")
    elif int(cfg["eval_seed"]) == int(cfg["final_seed_closed"]):
        raise RuntimeError("refusing the closed final seed without --final")
    if jax.default_backend() != "gpu" or jnp.zeros((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError(f"need a float64 GPU backend, got {jax.default_backend()}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = int(cfg["n"])
    dt = float(cfg["dt_truth"])
    horizon = float(cfg["horizon"])
    step = FROZEN_DT if not args.smoke else 0.01
    rank = FROZEN_RANK if not args.smoke else 4
    started = time.time()
    report = dict(
        schema="ns3d-grok-diag05-v1", config=cfg, smoke=bool(args.smoke),
        final_cohort_opened=bool(args.final), frozen_rank=rank, frozen_dt=step,
        source_commit=os.environ.get("SOURCE_COMMIT"), job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()),
        selection="Fastest diag04 development setting with zero cases over 5%: rank 64, dt=0.01.",
    )
    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    eval_parameters = F.parameters(int(cfg["eval_seed"]), int(cfg["eval_cases"]))
    report["train_parameter_sha256"] = D.sha256_array(train_parameters)
    report["eval_parameter_sha256"] = D.sha256_array(eval_parameters)
    D.log("generating training POD data and evaluation truth")
    train, _, _ = D.generate(train_parameters, n, dt, horizon)
    truth, _, _ = D.generate(eval_parameters, n, dt, horizon)
    np.save(out / "eval_truth.npy", truth)
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    del train
    gc.collect()
    snapshots = np.ascontiguousarray(centered.reshape(-1, int(np.prod(centered.shape[2:]))))
    del centered
    basis, _, orth = D.pod_basis(snapshots, rank, int(cfg.get("gram_block", 512)))
    del snapshots
    gc.collect()
    if basis.shape[1] < rank:
        raise RuntimeError(f"centered spectrum has {basis.shape[1]} modes, frozen rank is {rank}")
    q, _ = D.orthonormalize_prefix(basis, rank)
    linear = jax.device_put(jnp.asarray(D.linear_operator(q, n)))
    basis_j = jax.device_put(jnp.asarray(q))
    steps = D.nsteps_for(step, horizon)
    runner = diag04.make_tracked_run(step, steps, steps // 5, n)
    geom = F.geometry(n)
    viscosities = eval_parameters[:, -1]
    repetitions = 2 if args.smoke else 5
    predicted = np.empty_like(truth)
    rom_times = []
    jax.block_until_ready(runner(jnp.asarray(truth[0, 0]), float(viscosities[0]), basis_j, linear, geom))
    for case in range(len(truth)):
        state = jnp.asarray(truth[case, 0])
        viscosity = float(viscosities[case])
        samples = []
        outputs = []
        for _ in range(repetitions):
            start = time.perf_counter()
            value = runner(state, viscosity, basis_j, linear, geom)
            jax.block_until_ready(value)
            samples.append((time.perf_counter() - start) * 1e3)
            outputs.append(np.asarray(value))
        if float(np.max(np.abs(outputs[0] - outputs[-1]))) > 1e-12:
            raise RuntimeError(f"ROM repetitions differ on case {case}")
        predicted[case] = outputs[0]
        rom_times.append(samples)
    np.save(out / "tracked.npy", predicted)
    errors = np.empty((len(truth), truth.shape[1]), dtype=np.float64)
    for case in range(len(truth)):
        errors[case] = D.rel_rows(predicted[case], truth[case], truth[case, 0])
    D.require_finite(errors, "rom errors")
    report["rom"] = dict(
        stats=D.stats_from_cases(errors, 0.05), errors=errors.tolist(),
        repetition_ms=rom_times, median_ms=float(np.median(np.asarray(rom_times))),
    )
    D.log(f"rom evolved worst {report['rom']['stats']['evolved_worst']:.6f}")
    report["fom"] = {}
    for fom_dt in (() if args.smoke else FOM_DTS):
        if abs(round(horizon / fom_dt) * fom_dt - horizon) > 1e-12:
            raise RuntimeError(f"dt {fom_dt} does not divide the horizon")
        fom_steps = D.nsteps_for(fom_dt, horizon)
        if fom_steps % 5:
            raise RuntimeError(f"dt {fom_dt} misses the output times")
        solver = F.make_solver(fom_dt, fom_steps, fom_steps // 5)
        fields = np.empty_like(truth)
        samples_all = []
        jax.block_until_ready(solver(jnp.asarray(truth[0, 0]), float(viscosities[0]), geom))
        for case in range(len(truth)):
            state = jnp.asarray(truth[case, 0])
            viscosity = float(viscosities[case])
            samples = []
            outputs = []
            for _ in range(repetitions):
                start = time.perf_counter()
                value = solver(state, viscosity, geom)
                jax.block_until_ready(value)
                samples.append((time.perf_counter() - start) * 1e3)
                outputs.append(np.asarray(value))
            if float(np.max(np.abs(outputs[0] - outputs[-1]))) > 1e-12:
                raise RuntimeError(f"FOM repetitions differ on case {case} dt {fom_dt}")
            fields[case] = outputs[0]
            samples_all.append(samples)
        fom_errors = np.empty_like(errors)
        for case in range(len(truth)):
            fom_errors[case] = D.rel_rows(fields[case], truth[case], truth[case, 0])
        tag = D.file_dt(fom_dt)
        np.save(out / f"fom_dt{tag}.npy", fields)
        report["fom"][tag] = dict(
            dt=fom_dt, stats=D.stats_from_cases(fom_errors, 0.05), errors=fom_errors.tolist(),
            repetition_ms=samples_all, median_ms=float(np.median(np.asarray(samples_all))),
        )
        D.log(f"fom dt={fom_dt} worst={report['fom'][tag]['stats']['evolved_worst']:.6f} "
              f"median_ms={report['fom'][tag]['median_ms']:.3f}")
    report["pod_orth_error"] = orth
    report["elapsed_seconds"] = time.time() - started
    D.dump_json(out / "summary.json", report)
    D.log(f"wrote {out / 'summary.json'} final_opened={args.final}")


if __name__ == "__main__":
    main()
