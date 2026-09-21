"""Fused every-step center tracking, timed against CNAB2 in the same job.

The substep is the startup step of make_dense_galerkin_run, then the decoded
field is re-centered by its own energy centroid. A local smoke checks that
this scan matches diag03.track_case. The final cohort is not drawn.
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
import diag03
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


def shift_field(field, offset_samples):
    spec = jnp.fft.fftn(field, axes=(-3, -2, -1))
    n = field.shape[-1]
    for axis in range(3):
        modes = jnp.fft.fftfreq(n) * n
        phase = jnp.exp(-2j * jnp.pi * modes * (offset_samples[axis] / n))
        shape = [1, 1, 1]
        shape[axis] = n
        spec = spec * phase.reshape((1, *shape))
    return jnp.fft.ifftn(spec, axes=(-3, -2, -1)).real


def energy_centroid(field):
    weight = jnp.sum(field * field, axis=0)
    n = field.shape[-1]
    coord = jnp.arange(n, dtype=field.dtype) / n
    angle = 2 * jnp.pi * coord
    centers = []
    for axis in range(3):
        marginal = weight.sum(axis=tuple(i for i in range(3) if i != axis))
        centers.append(jnp.arctan2(jnp.sum(marginal * jnp.sin(angle)),
                                   jnp.sum(marginal * jnp.cos(angle))) / (2 * jnp.pi) % 1)
    return jnp.stack(centers)


def make_tracked_run(dt, nsteps, out_every, n):
    assert nsteps % out_every == 0

    @jax.jit
    def run(u0, nu, G, L, geom):
        identity = jnp.eye(L.shape[0], dtype=u0.dtype)
        half = jnp.linalg.cholesky(identity - 0.5 * dt * nu * L)
        full = jnp.linalg.cholesky(identity - dt * nu * L)

        def solve(factor, rhs):
            return jax.scipy.linalg.cho_solve((factor, True), rhs)

        def advection(coeff):
            field = (G @ coeff).reshape(3, n, n, n)
            return G.T @ F.ifft(F.nonlinear(F.fft(field), geom)).ravel()

        def step(carry, _index):
            coeff, center = carry
            current = advection(coeff)
            predicted = solve(full, coeff + dt * current)
            updated = solve(half, coeff + 0.5 * dt * (
                nu * (L @ coeff) + current + advection(predicted)))
            field = (G @ updated).reshape(3, n, n, n)
            delta = energy_centroid(field)
            recentered = shift_field(field, -delta * n)
            center = (center + delta) % 1
            return (G.T @ recentered.ravel(), center), shift_field(recentered, center * n)

        center0 = energy_centroid(u0)
        shifted0 = shift_field(u0, -center0 * n)
        coeff0 = G.T @ shifted0.ravel()
        frame0 = shift_field((G @ coeff0).reshape(3, n, n, n), center0 * n)

        def block(carry, _index):
            new, frames = jax.lax.scan(step, carry, jnp.arange(out_every))
            return new, frames[-1]

        _, frames = jax.lax.scan(block, (coeff0, center0), jnp.arange(nsteps // out_every))
        return jnp.concatenate((frame0[None], frames))

    return run


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
        schema="ns3d-grok-diag04-v1", config=cfg, smoke=bool(args.smoke),
        final_cohort_opened=False, source_commit=os.environ.get("SOURCE_COMMIT"),
        job_id=os.environ.get("SLURM_JOB_ID"), device=str(jax.devices()),
        scheme="Fused scan of the Galerkin startup step plus a ROM-centroid shift. Not multistep CNAB2.",
    )
    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_parameters = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    D.log("generating trajectories")
    train, _, _ = D.generate(train_parameters, n, dt, horizon)
    if args.smoke:
        dev_full = diag03.generate_full(dev_parameters, n, dt, horizon)
        dev = dev_full[:, ::(D.nsteps_for(dt, horizon) // 5)]
    else:
        dev_full = None
        dev, _, _ = D.generate(dev_parameters, n, dt, horizon)
    np.save(out / "dev_truth.npy", dev)
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    del train
    gc.collect()
    snapshots = np.ascontiguousarray(centered.reshape(centered.shape[0] * centered.shape[1], -1))
    del centered
    requested = [int(rank) for rank in cfg["ranks"]]
    basis, energy, orth = D.pod_basis(snapshots, max(requested), int(cfg.get("gram_block", 512)))
    del snapshots
    gc.collect()
    max_rank = int(basis.shape[1])
    ranks = sorted({rank for rank in requested if rank <= max_rank})
    report["pod"] = dict(available_rank=max_rank, orth_error=orth)
    rom = D.import_rom()
    dts = [float(value) for value in cfg["rollout_dts"]]
    viscosities = dev_parameters[:, -1]
    geom = F.geometry(n)
    report["ranks"] = {}
    runners = {}
    for step in dts:
        steps = D.nsteps_for(step, horizon)
        runners[step] = make_tracked_run(step, steps, steps // 5, n)
    if args.smoke:
        reference_runner = rom.make_dense_galerkin_run(dts[0], 1, 1, n)
        q, _ = D.orthonormalize_prefix(basis, ranks[0])
        linear = D.linear_operator(q, n)
        basis_j = jax.device_put(jnp.asarray(q))
        linear_j = jax.device_put(jnp.asarray(linear))
        fused = np.asarray(runners[dts[0]](
            jnp.asarray(dev_full[0, 0]), float(viscosities[0]), basis_j, linear_j, geom))
        python, _ = diag03.track_case(
            reference_runner, dev_full[0], float(viscosities[0]), basis_j, linear_j, geom,
            dts[0], dt, "rom")
        gap = float(np.linalg.norm(fused - python) / np.linalg.norm(python))
        report["smoke_python_gap"] = gap
        D.log(f"smoke python gap {gap:.3e}")
        if gap > 1e-6:
            raise RuntimeError(f"fused tracker disagrees with the Python tracker: {gap}")
    for rank in ranks:
        q, _ = D.orthonormalize_prefix(basis, rank)
        linear = D.linear_operator(q, n)
        basis_j = jax.device_put(jnp.asarray(q))
        linear_j = jax.device_put(jnp.asarray(linear))
        row = {}
        for step, runner in runners.items():
            predicted = np.empty_like(dev)
            for case in range(len(dev)):
                predicted[case] = np.asarray(runner(
                    jnp.asarray(dev[case, 0]), float(viscosities[case]), basis_j, linear_j, geom))
            errors = np.empty((len(dev), dev.shape[1]), dtype=np.float64)
            for case in range(len(dev)):
                errors[case] = D.rel_rows(predicted[case], dev[case], dev[case, 0])
            D.require_finite(predicted, f"rank {rank} dt {step}")
            tag = D.file_dt(step)
            np.save(out / f"tracked_r{rank}_dt{tag}.npy", predicted)
            row[tag] = dict(dt=step, stats=D.stats_from_cases(errors, float(cfg["target_relative"])),
                            errors=errors.tolist())
            D.log(f"rank {rank} dt={step} worst={row[tag]['stats']['evolved_worst']:.6f}")
        report["ranks"][str(rank)] = row
        del q, linear
        gc.collect()

    timing_cases = min(int(cfg["timing_cases"]), len(dev))
    repetitions = int(cfg["timing_repetitions"])
    report["timing"] = dict(fom={}, tracked={})
    for step in dts:
        steps = D.nsteps_for(step, horizon)
        solver = F.make_solver(step, steps, steps // 5)
        jax.block_until_ready(solver(jnp.asarray(dev[0, 0]), float(viscosities[0]), geom))
        samples = []
        for case in range(timing_cases):
            state = jnp.asarray(dev[case, 0])
            viscosity = float(viscosities[case])
            samples.extend(D.time_calls(lambda state=state, viscosity=viscosity, solver=solver: solver(state, viscosity, geom), repetitions))
        report["timing"]["fom"][D.file_dt(step)] = dict(**D.median_summary(samples), dt=step)
        for rank in ranks:
            q, _ = D.orthonormalize_prefix(basis, rank)
            basis_j = jax.device_put(jnp.asarray(q))
            linear_j = jax.device_put(jnp.asarray(D.linear_operator(q, n)))
            runner = runners[step]
            jax.block_until_ready(runner(jnp.asarray(dev[0, 0]), float(viscosities[0]), basis_j, linear_j, geom))
            samples = []
            for case in range(timing_cases):
                state = jnp.asarray(dev[case, 0])
                viscosity = float(viscosities[case])
                samples.extend(D.time_calls(
                    lambda state=state, viscosity=viscosity, basis_j=basis_j, linear_j=linear_j, runner=runner:
                    runner(state, viscosity, basis_j, linear_j, geom), repetitions))
            report["timing"]["tracked"][f"r{rank}_{D.file_dt(step)}"] = dict(
                **D.median_summary(samples), rank=rank, dt=step)
    report["elapsed_seconds"] = time.time() - started
    D.dump_json(out / "summary.json", report)
    D.log(f"wrote {out / 'summary.json'} elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
