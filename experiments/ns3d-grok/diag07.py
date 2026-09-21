"""One sealed evaluation of the truncation selected by diag06.

The setting is frozen: centered POD rank 64, startup step, dt=0.01, Fourier
tail 1e-6. There is no second selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import coeff_shift as C
import diag_floor as D
import diag04
import diag06
import jax
import jax.numpy as jnp
import ns3d_fom as F

FROZEN_RANK = 64
FROZEN_DT = 0.01
FROZEN_TAIL = 1e-6
FROZEN_N_FREQ = 9222
FROZEN_ENERGY = 0.9999990109711092
FROZEN_SEED = 202609211
CLOSED_SEED = 202609203


def load_config(path, smoke):
    cfg = json.loads(Path(path).read_text())
    if smoke:
        cfg.update(n=8, dt_truth=0.01, horizon=0.05, train_cases=4, eval_cases=2,
                   eval_seed=int(cfg["dev_seed"]), rank=4, tail=1e-3, gram_block=4,
                   timing_repetitions=2, fom_dts=[])
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.final and args.smoke:
        raise RuntimeError("a smoke run cannot open the sealed seed")
    cfg = load_config(args.config, args.smoke)
    eval_seed = int(cfg["eval_seed"])
    if args.final:
        mismatches = []
        if eval_seed != FROZEN_SEED or int(cfg["eval_cases"]) != 32:
            mismatches.append("cohort")
        if int(cfg["rank"]) != FROZEN_RANK or float(cfg["dt"]) != FROZEN_DT:
            mismatches.append("rank or dt")
        if float(cfg["tail"]) != FROZEN_TAIL:
            mismatches.append("tail")
        if mismatches:
            raise RuntimeError(f"frozen setting mismatch: {mismatches}")
    elif eval_seed in (FROZEN_SEED, CLOSED_SEED):
        raise RuntimeError("refusing a sealed seed without --final")
    if jax.default_backend() != "gpu" or jnp.zeros((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError(f"need a float64 GPU backend, got {jax.default_backend()}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    step = float(cfg["dt"])
    steps = D.nsteps_for(step, horizon)
    out_every = steps // 5
    started = time.time()
    report = dict(
        schema="ns3d-grok-diag07-v1", config=cfg, smoke=bool(args.smoke),
        final_cohort_opened=bool(args.final), sealed_seed_opened=bool(args.final),
        source_commit=os.environ.get("SOURCE_COMMIT"), job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()),
        frozen=dict(rank=FROZEN_RANK, dt=FROZEN_DT, tail=FROZEN_TAIL, n_freq=FROZEN_N_FREQ),
    )
    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    D.log("generating the training POD before the evaluation seed")
    train, _, _ = D.generate(train_parameters, n, float(cfg["dt_truth"]), horizon)
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    del train
    gc.collect()
    snapshots = np.ascontiguousarray(centered.reshape(-1, int(np.prod(centered.shape[2:]))))
    del centered
    gc.collect()
    basis, _, orth = D.pod_basis(snapshots, int(cfg["rank"]), int(cfg.get("gram_block", 512)))
    del snapshots
    gc.collect()
    if int(basis.shape[1]) < int(cfg["rank"]):
        raise RuntimeError("POD rank is below the frozen rank")
    q, _ = D.orthonormalize_prefix(basis, int(cfg["rank"]))
    del basis
    linear = D.linear_operator(q, n)
    geom = F.geometry(n)
    tensor, tensor_gap = C.assemble_advection_tensor(q, n, geom)
    sin_m, cos_m, centroid_gap = C.centroid_matrices(q, n)
    packed, kx, ky, kz = C.fourier_pack(q, n)
    cut, cut_x, cut_y, cut_z, count, energy = C.truncate_pack(packed, kx, ky, kz, float(cfg["tail"]))
    report["operators"] = dict(
        pod_orth_error=orth, tensor_probe_gap=tensor_gap, centroid_probe_gap=centroid_gap,
        n_freq=count, energy_kept=energy,
    )
    if not args.smoke:
        if count != FROZEN_N_FREQ or abs(energy - FROZEN_ENERGY) > 1e-12:
            raise RuntimeError(f"truncation fingerprint changed: freq {count}, energy {energy}")
    D.log(f"truncation freq={count} energy={energy:.12f}")
    eval_parameters = F.parameters(eval_seed, int(cfg["eval_cases"]))
    report["eval_parameter_sha256"] = hashlib.sha256(
        np.ascontiguousarray(eval_parameters).tobytes()).hexdigest()
    D.log(f"generating evaluation seed {eval_seed}")
    truth, _, _ = D.generate(eval_parameters, n, float(cfg["dt_truth"]), horizon)
    np.save(out / "eval_truth.npy", truth)
    viscosities = eval_parameters[:, -1]
    repetitions = int(cfg["timing_repetitions"])
    target = float(cfg["target_relative"])
    basis_j = jax.device_put(jnp.asarray(q))
    linear_j = jax.device_put(jnp.asarray(linear))
    tensor_j = jax.device_put(jnp.asarray(tensor))
    arguments = (basis_j, linear_j, tensor_j, jnp.asarray(cut), jnp.asarray(cut_x),
                 jnp.asarray(cut_y), jnp.asarray(cut_z), jnp.asarray(sin_m), jnp.asarray(cos_m),
                 jnp.zeros((1, 3)))
    coeff = C.make_coeff_run(step, steps, out_every, n, "pod")
    grid = diag04.make_tracked_run(step, steps, out_every, n)
    coeff_fields, coeff_times = diag06.paired_rollout(
        coeff, arguments, truth, viscosities, repetitions, "coeff")
    np.save(out / "coeff.npy", coeff_fields)
    grid_fields, grid_times = diag06.paired_rollout(
        grid, (basis_j, linear_j, geom), truth, viscosities, repetitions, "tracker")
    np.save(out / "tracker.npy", grid_fields)
    report["coeff"] = diag06.arm_record("coeff", coeff_fields, truth, coeff_times, grid_fields, target)
    report["tracker"] = diag06.arm_record("tracker", grid_fields, truth, grid_times, grid_fields, target)
    D.log(f"coeff worst={report['coeff']['stats']['evolved_worst']:.6f} "
          f"parity={report['coeff']['parity_worst']:.3e} median_ms={report['coeff']['median_ms']:.3f}")
    report["fom"] = {}
    for fom_dt in cfg["fom_dts"]:
        fom_dt = float(fom_dt)
        fom_steps = D.nsteps_for(fom_dt, horizon)
        solver = F.make_solver(fom_dt, fom_steps, fom_steps // 5)
        fields, samples = diag06.paired_rollout(
            solver, (geom,), truth, viscosities, repetitions, f"fom-{fom_dt}")
        tag = D.file_dt(fom_dt)
        np.save(out / f"fom_dt{tag}.npy", fields)
        errors = diag06.case_errors(fields, truth)
        report["fom"][tag] = dict(
            dt=fom_dt, stats=D.stats_from_cases(errors, target), errors=errors.tolist(),
            median_ms=float(np.median(samples)), repetition_ms=samples)
        D.log(f"fom dt={fom_dt} worst={report['fom'][tag]['stats']['evolved_worst']:.6f} "
              f"median_ms={report['fom'][tag]['median_ms']:.3f}")
    report["elapsed_seconds"] = time.time() - started
    D.dump_json(out / "summary.json", report)
    D.log(f"wrote {out / 'summary.json'} final_opened={args.final}")


if __name__ == "__main__":
    main()
