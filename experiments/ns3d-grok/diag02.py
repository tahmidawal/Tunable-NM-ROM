"""Shift-frame POD: oracle center versus a center taken only from the initial field.

The per-time center uses the truth at that instant and is a representation floor.
The solved arm shifts by the centroid of u0, integrates CNAB2 Galerkin in that
frame, and shifts back. The final cohort is not drawn.
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
import jax
import jax.numpy as jnp
import ns3d_fom as F


def load_config(path, smoke):
    cfg = json.loads(Path(path).read_text())
    if smoke:
        cfg.update(n=8, dt_truth=0.01, horizon=0.05, train_cases=4, dev_cases=2,
                   ranks=[4], rollout_dts=[0.01], timing_cases=1, timing_repetitions=2,
                   gram_block=4)
    return cfg


def restore(frames, center):
    out = np.empty_like(frames)
    offset = np.asarray(center) * frames.shape[-1]
    for instant in range(len(frames)):
        out[instant] = D.fourier_shift(frames[instant], offset)
    return out


def frozen_projection(basis, frames):
    """Project each frame after shifting by the centroid of that case's initial field."""
    cases, times = frames.shape[:2]
    errors = np.empty((cases, times), dtype=np.float64)
    recon = np.empty_like(frames)
    centers = np.empty((cases, 3), dtype=np.float64)
    for case in range(cases):
        center = D.energy_centroid(frames[case, 0])
        centers[case] = center
        offset = -center * frames.shape[-1]
        for instant in range(times):
            shifted = D.fourier_shift(frames[case, instant], offset)
            coeff = basis.T @ shifted.ravel()
            recon[case, instant] = D.fourier_shift((basis @ coeff).reshape(shifted.shape), -offset)
        errors[case] = D.rel_rows(recon[case], frames[case], frames[case, 0])
    return errors, recon, centers


def shifted_galerkin(runner, frames, viscosities, basis, linear, geom):
    cases = len(frames)
    errors = np.empty((cases, frames.shape[1]), dtype=np.float64)
    predicted = np.empty_like(frames)
    gate = []
    basis_j = jax.device_put(jnp.asarray(basis))
    linear_j = jax.device_put(jnp.asarray(linear))
    for case in range(cases):
        center = D.energy_centroid(frames[case, 0])
        initial = jnp.asarray(D.fourier_shift(frames[case, 0], -center * frames.shape[-1]))
        out = np.asarray(runner(initial, float(viscosities[case]), basis_j, linear_j, geom))
        predicted[case] = restore(out.reshape(frames.shape[1:]), center)
        projection = np.asarray(basis_j.T @ initial.ravel())
        frame0 = D.fourier_shift((basis @ projection).reshape(frames.shape[2:]), center * frames.shape[-1])
        gap = float(np.linalg.norm(predicted[case, 0] - frame0) / np.linalg.norm(frame0))
        gate.append(gap)
        errors[case] = D.rel_rows(predicted[case], frames[case], frames[case, 0])
        D.require_finite(predicted[case], f"shifted galerkin case {case}")
    if max(gate) > 1e-8:
        raise RuntimeError(f"shifted frame 0 is not the frozen projection: {max(gate)}")
    return errors, predicted, float(max(gate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)
    if int(cfg["dev_seed"]) == int(cfg["final_seed_closed"]) or int(cfg["train_seed"]) == int(cfg["final_seed_closed"]):
        raise RuntimeError("refusing the closed final seed")
    if jax.default_backend() != "gpu" or jnp.zeros((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError(f"need a float64 GPU backend, got {jax.default_backend()}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = int(cfg["n"])
    dt = float(cfg["dt_truth"])
    horizon = float(cfg["horizon"])
    started = time.time()
    report = dict(
        schema="ns3d-grok-diag02-v1",
        config=cfg,
        smoke=bool(args.smoke),
        final_cohort_opened=False,
        source_commit=os.environ.get("SOURCE_COMMIT"),
        job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()),
        note="Solved trajectories use only the centroid of the initial field. "
             "per_time_oracle recenters every saved truth and is not a model.",
    )
    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_parameters = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["train_parameter_sha256"] = D.sha256_array(train_parameters)
    report["dev_parameter_sha256"] = D.sha256_array(dev_parameters)
    D.log("generating training and development trajectories")
    train, _, _ = D.generate(train_parameters, n, dt, horizon)
    dev, _, _ = D.generate(dev_parameters, n, dt, horizon)
    D.require_finite(train, "training trajectories")
    D.require_finite(dev, "development trajectories")
    np.save(out / "dev_truth.npy", dev)
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    del train
    gc.collect()
    snapshots = centered.reshape(len(centered) * centered.shape[1], -1)
    del centered
    requested = [int(rank) for rank in cfg["ranks"]]
    basis, energy, orth = D.pod_basis(snapshots, max(requested), int(cfg.get("gram_block", 512)))
    del snapshots
    gc.collect()
    max_rank = int(basis.shape[1])
    ranks = [rank for rank in requested if rank <= max_rank]
    if max_rank not in ranks:
        ranks.append(max_rank)
    ranks = sorted(set(ranks))
    report["pod"] = dict(available_rank=max_rank, orth_error=orth,
                         spectrum_head=[float(x) for x in energy[:8]], snapshots=int(cfg["train_cases"]) * 6)
    rom = D.import_rom()
    dts = [float(value) for value in cfg["rollout_dts"]]
    runners = {}
    for step in dts:
        steps = D.nsteps_for(step, horizon)
        if steps % 5:
            raise RuntimeError(f"dt {step} does not hit five output intervals")
        runners[step] = rom.make_dense_galerkin_run(step, steps, steps // 5, n)
    viscosities = dev_parameters[:, -1]
    geom = F.geometry(n)
    report["ranks"] = {}
    for rank in ranks:
        q, _ = D.orthonormalize_prefix(basis, rank)
        oracle_errors, oracle_fields, travel = D.oracle_shift_errors(q, dev)
        np.save(out / f"projection_oracle_r{rank}.npy", oracle_fields)
        frozen_errors, frozen_fields, _ = frozen_projection(q, dev)
        np.save(out / f"projection_frozen_r{rank}.npy", frozen_fields)
        D.require_finite(frozen_fields, f"frozen projection {rank}")
        row = dict(
            per_time_oracle=dict(stats=D.stats_from_cases(oracle_errors, float(cfg["target_relative"])),
                                 errors=oracle_errors.tolist(), centroid_travel=travel),
            frozen_c0=dict(stats=D.stats_from_cases(frozen_errors, float(cfg["target_relative"])),
                           errors=frozen_errors.tolist()),
            galerkin={},
        )
        linear = D.linear_operator(q, n)
        for step in dts:
            errors, predicted, gap = shifted_galerkin(
                runners[step], dev, viscosities, q, linear, geom)
            np.save(out / f"galerkin_r{rank}_dt{D.file_dt(step)}.npy", predicted)
            row["galerkin"][D.file_dt(step)] = dict(
                dt=step, frame0_projection_gap=gap,
                stats=D.stats_from_cases(errors, float(cfg["target_relative"])),
                errors=errors.tolist())
        report["ranks"][str(rank)] = row
        D.log(
            f"rank {rank} oracle_worst={row['per_time_oracle']['stats']['evolved_worst']:.6f} "
            f"frozen_worst={row['frozen_c0']['stats']['evolved_worst']:.6f} "
            + " ".join(
                f"gal_{step}={row['galerkin'][D.file_dt(step)]['stats']['evolved_worst']:.6f}"
                for step in dts))
        D.dump_json(out / "summary_partial.json", report)
        del q, linear, oracle_fields, frozen_fields
        gc.collect()
    if not args.smoke and "64" in report["ranks"]:
        oracle64 = report["ranks"]["64"]["per_time_oracle"]["stats"]["evolved_worst"]
        if oracle64 > 0.01:
            raise RuntimeError(f"rank-64 per-time floor {oracle64} is above the diag01 sanity bar 0.01")

    timing_cases = min(int(cfg["timing_cases"]), len(dev))
    repetitions = int(cfg["timing_repetitions"])
    report["timing"] = dict(fom={}, shifted_galerkin={}, note=(
        "Warmup discarded. Shifted times include the initial centroid, the shift in, "
        "the CNAB2 rollout, and the shift back. Same job as the errors."))
    for step in dts:
        steps = D.nsteps_for(step, horizon)
        solver = F.make_solver(step, steps, steps // 5)
        jax.block_until_ready(solver(jnp.asarray(dev[0, 0]), float(viscosities[0]), geom))
        samples = []
        for case in range(timing_cases):
            state = jnp.asarray(dev[case, 0])
            viscosity = float(viscosities[case])
            samples.extend(D.time_calls(lambda state=state, viscosity=viscosity: solver(state, viscosity, geom), repetitions))
        report["timing"]["fom"][D.file_dt(step)] = dict(**D.median_summary(samples), dt=step)
    for rank in ranks:
        q, _ = D.orthonormalize_prefix(basis, rank)
        linear = jax.device_put(jnp.asarray(D.linear_operator(q, n)))
        basis_j = jax.device_put(jnp.asarray(q))
        for step, runner in runners.items():
            def query(case, runner=runner, basis_j=basis_j, linear=linear):
                center = D.energy_centroid(dev[case, 0])
                initial = jnp.asarray(D.fourier_shift(dev[case, 0], -center * n))
                out = runner(initial, float(viscosities[case]), basis_j, linear, geom)
                jax.block_until_ready(out)
                return restore(np.asarray(out).reshape(dev.shape[1:]), center)
            query(0)
            samples = []
            for case in range(timing_cases):
                samples.extend(D.time_calls(lambda case=case: query(case), repetitions))
            report["timing"]["shifted_galerkin"][f"r{rank}_dt{D.file_dt(step)}"] = dict(
                **D.median_summary(samples), rank=rank, dt=step)
    report["elapsed_seconds"] = time.time() - started
    D.dump_json(out / "summary.json", report)
    D.log(f"wrote {out / 'summary.json'} elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
