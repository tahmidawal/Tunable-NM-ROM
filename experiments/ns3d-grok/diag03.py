"""Online center tracking in a centered POD, development only.

Frozen initial centers miss the 5% bar by the first saved time. This job
re-centers the ROM field every step using either the ROM field's own energy
centroid or the true field's centroid. The true centroid is a control, not a
model. The final cohort is not drawn.
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


def composition_check(field):
    n = field.shape[-1]
    center = D.energy_centroid(field)
    centered = D.fourier_shift(field, -center * n)
    shift = np.array([0.2, -0.15, 0.3])
    moved = D.fourier_shift(centered, shift * n)
    got = D.energy_centroid(moved)
    expect = (D.energy_centroid(centered) + shift) % 1.0
    gap = float(np.max(D.torus_delta(got, expect)))
    if gap > 1e-5:
        raise RuntimeError(f"centroid does not compose with a Fourier shift: {gap}")
    return gap


def generate_full(params, n, dt, horizon):
    steps = D.nsteps_for(dt, horizon)
    solver = F.make_solver(dt, steps, 1)
    geom = F.geometry(n)
    frames = np.empty((len(params), steps + 1, 3, n, n, n), dtype=np.float64)
    for index, parameter in enumerate(params):
        frames[index] = np.asarray(solver(
            jnp.asarray(F.initial(n, parameter)), float(parameter[-1]), geom))
        if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(params):
            D.log(f"full trajectory {index + 1}/{len(params)}")
    return frames


def lab_field(field, center):
    return D.fourier_shift(field, np.asarray(center) * field.shape[-1])


def track_case(runner, truth, viscosity, basis_j, linear_j, geom, dt, truth_dt, mode):
    """Return lab-frame fields at the six output times. `truth` is every fine step."""
    n = truth.shape[-1]
    ratio = int(round(dt / truth_dt))
    if abs(ratio * truth_dt - dt) > 1e-12:
        raise RuntimeError("rollout dt is not a multiple of the truth dt")
    steps = truth.shape[0] - 1
    if steps % ratio:
        raise RuntimeError("truth length does not match the rollout dt")
    nsteps = steps // ratio
    out_every = nsteps // 5
    spatial = (3, n, n, n)
    center = D.energy_centroid(truth[0])
    initial = jnp.asarray(D.fourier_shift(truth[0], -center * n))
    out = np.asarray(runner(initial, float(viscosity), basis_j, linear_j, geom))
    saved = [lab_field(out[0].reshape(spatial), center)]
    field = out[1].reshape(spatial)
    max_delta = 0.0
    for step in range(nsteps):
        if mode == "rom":
            delta = D.energy_centroid(field)
            max_delta = max(max_delta, float(np.max(D.torus_delta(delta, 0.0))))
            center = (center + delta) % 1.0
            field = D.fourier_shift(field, -delta * n)
        elif mode == "truth":
            true_center = D.energy_centroid(truth[(step + 1) * ratio])
            field = D.fourier_shift(field, (center - true_center) * n)
            center = true_center
        elif mode == "interval":
            if (step + 1) % out_every == 0:
                delta = D.energy_centroid(field)
                max_delta = max(max_delta, float(np.max(D.torus_delta(delta, 0.0))))
                center = (center + delta) % 1.0
                field = D.fourier_shift(field, -delta * n)
        else:
            raise RuntimeError(mode)
        if (step + 1) % out_every == 0 or step + 1 == nsteps:
            saved.append(lab_field(field, center))
        if step + 1 == nsteps:
            break
        field = np.asarray(runner(
            jnp.asarray(field), float(viscosity), basis_j, linear_j, geom))[1].reshape(spatial)
    if len(saved) != 6:
        raise RuntimeError(f"{mode} saved {len(saved)} frames, expected 6")
    D.require_finite(np.stack(saved), mode)
    return np.stack(saved), max_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)
    if int(cfg["dev_seed"]) == int(cfg["final_seed_closed"]):
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
        schema="ns3d-grok-diag03-v1",
        config=cfg, smoke=bool(args.smoke), final_cohort_opened=False,
        source_commit=os.environ.get("SOURCE_COMMIT"), job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()),
        scheme="Each substep is the startup step of make_dense_galerkin_run, then the field is "
               "re-centered. This is not multistep CNAB2. rom uses the ROM centroid; truth uses "
               "the true centroid; interval re-centers only at the saved output times.",
    )
    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_parameters = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["shift_composition_gap"] = composition_check(
        np.asarray(F.initial(n, dev_parameters[0])))
    D.log("generating centered-POD training trajectories and full development trajectories")
    train, _, _ = D.generate(train_parameters, n, dt, horizon)
    dev = generate_full(dev_parameters, n, dt, horizon)
    D.require_finite(dev, "development trajectories")
    output_stride = D.nsteps_for(dt, horizon) // 5
    np.save(out / "dev_output.npy", dev[:, ::output_stride])
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    del train
    gc.collect()
    snapshots = centered.reshape(-1, int(np.prod(centered.shape[2:])))
    del centered
    requested = [int(rank) for rank in cfg["ranks"]]
    basis, energy, orth = D.pod_basis(snapshots, max(requested), int(cfg.get("gram_block", 512)))
    del snapshots
    gc.collect()
    max_rank = int(basis.shape[1])
    ranks = sorted(set(rank for rank in requested if rank <= max_rank) | {max_rank})
    report["pod"] = dict(available_rank=max_rank, orth_error=orth,
                         spectrum_head=[float(x) for x in energy[:8]])
    rom = D.import_rom()
    dts = [float(value) for value in cfg["rollout_dts"]]
    runners = {}
    for step in dts:
        runners[step] = rom.make_dense_galerkin_run(step, 1, 1, n)
    viscosities = dev_parameters[:, -1]
    geom = F.geometry(n)
    modes = ("rom", "truth", "interval")
    report["ranks"] = {}
    for rank in ranks:
        q, _ = D.orthonormalize_prefix(basis, rank)
        linear = D.linear_operator(q, n)
        basis_j = jax.device_put(jnp.asarray(q))
        linear_j = jax.device_put(jnp.asarray(linear))
        row = {}
        for mode in modes:
            row[mode] = {}
            for step in dts:
                predicted = np.empty((len(dev), 6, 3, n, n, n), dtype=np.float64)
                deltas = []
                for case in range(len(dev)):
                    predicted[case], delta = track_case(
                        runners[step], dev[case], float(viscosities[case]),
                        basis_j, linear_j, geom, step, dt, mode)
                    deltas.append(delta)
                reference = dev[:, ::output_stride]
                errors = np.empty((len(dev), 6), dtype=np.float64)
                for case in range(len(dev)):
                    errors[case] = D.rel_rows(predicted[case], reference[case], reference[case, 0])
                tag = D.file_dt(step)
                np.save(out / f"{mode}_r{rank}_dt{tag}.npy", predicted)
                row[mode][tag] = dict(
                    dt=step, max_recentering_shift=float(np.max(deltas)),
                    stats=D.stats_from_cases(errors, float(cfg["target_relative"])),
                    errors=errors.tolist())
                D.log(f"rank {rank} {mode} dt={step} worst={row[mode][tag]['stats']['evolved_worst']:.6f}")
        report["ranks"][str(rank)] = row
        D.dump_json(out / "summary_partial.json", report)
        del q, linear
        gc.collect()

    timing_cases = min(int(cfg["timing_cases"]), len(dev))
    repetitions = int(cfg["timing_repetitions"])
    rank = ranks[0]
    q, _ = D.orthonormalize_prefix(basis, rank)
    basis_j = jax.device_put(jnp.asarray(q))
    linear_j = jax.device_put(jnp.asarray(D.linear_operator(q, n)))
    report["timing"] = {}
    for step in dts:
        def query(case, step=step):
            return track_case(runners[step], dev[case], float(viscosities[case]),
                              basis_j, linear_j, geom, step, dt, "rom")[0]
        query(0)
        samples = []
        for case in range(timing_cases):
            samples.extend(D.time_calls(lambda case=case: query(case), repetitions))
        report["timing"][D.file_dt(step)] = dict(**D.median_summary(samples), rank=rank, dt=step, mode="rom")
    report["elapsed_seconds"] = time.time() - started
    D.dump_json(out / "summary.json", report)
    D.log(f"wrote {out / 'summary.json'} elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
