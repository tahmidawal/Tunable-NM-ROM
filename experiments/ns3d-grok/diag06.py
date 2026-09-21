"""Development test of a coefficient-space center tracker.

The grid tracker is the parity reference. Seed 202609203 and the reserved
sealed seed are not read.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

import coeff_shift as C
import diag_floor as D
import diag04
import jax
import jax.numpy as jnp
import ns3d_fom as F


def load_config(path, smoke):
    cfg = json.loads(Path(path).read_text())
    if smoke:
        cfg.update(n=8, dt_truth=0.01, horizon=0.05, train_cases=4, dev_cases=2,
                   rank=4, gram_block=4, truncation_tails=[1e-3], fourier_ranks=[4],
                   timing_repetitions=2, fom_dts=[])
    return cfg


def reject_seeds(cfg):
    dev = int(cfg["dev_seed"])
    closed = {int(cfg["final_seed_closed"]), int(cfg["sealed_seed_reserved"])}
    if dev in closed or int(cfg["train_seed"]) in closed:
        raise RuntimeError("refusing a closed or reserved seed")


def field_gaps(predicted, reference):
    gaps = []
    for case in range(len(reference)):
        denominator = np.linalg.norm(reference[case])
        gaps.append(float(np.linalg.norm(predicted[case] - reference[case]) / max(denominator, 1e-300)))
    return gaps


def case_errors(predicted, truth):
    errors = np.empty((len(truth), truth.shape[1]), dtype=np.float64)
    for case in range(len(truth)):
        errors[case] = D.rel_rows(predicted[case], truth[case], truth[case, 0])
    return errors


def paired_rollout(runner, arguments, truth, viscosities, repetitions, label):
    first = (jnp.asarray(truth[0, 0]), float(viscosities[0]), *arguments)
    jax.block_until_ready(runner(*first))
    jax.block_until_ready(runner(*first))
    predicted = np.empty_like(truth)
    samples = []
    for case in range(len(truth)):
        outputs = []
        for _ in range(repetitions):
            start = time.perf_counter()
            value = runner(jnp.asarray(truth[case, 0]), float(viscosities[case]), *arguments)
            jax.block_until_ready(value)
            samples.append((time.perf_counter() - start) * 1e3)
            outputs.append(np.asarray(value, dtype=np.float64))
        disagreement = float(np.max(np.abs(outputs[0] - outputs[-1])))
        if disagreement > 1e-12:
            raise RuntimeError(f"{label} repetitions differ on case {case}: {disagreement}")
        predicted[case] = outputs[0]
    return predicted, samples


def time_only(function, repetitions):
    jax.block_until_ready(function())
    jax.block_until_ready(function())
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        jax.block_until_ready(function())
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def arm_record(name, predicted, truth, samples, tracker, target):
    errors = case_errors(predicted, truth)
    D.require_finite(errors, name)
    gaps = field_gaps(predicted, tracker)
    stats = D.stats_from_cases(errors, target)
    return dict(
        name=name, stats=stats, errors=errors.tolist(), parity_gaps=gaps,
        parity_worst=float(np.max(gaps)), median_ms=float(np.median(samples)),
        repetition_ms=samples,
        under_target=bool(stats["evolved_worst"] <= target and stats["cases_evolved_over_target"] == 0),
    )


def eligible(record, gate):
    return bool(record["under_target"] and record["parity_worst"] <= gate)


def profile_pieces(basis, linear, tensor, packed, kx, ky, kz, geom, n, nsteps, repetitions):
    basis_j = jnp.asarray(basis)
    tensor_j = jnp.asarray(tensor)
    packed_j = jnp.asarray(packed)
    kx_j, ky_j, kz_j = jnp.asarray(kx), jnp.asarray(ky), jnp.asarray(kz)
    coeff = jnp.asarray(np.random.default_rng(5).normal(size=basis.shape[1]))
    center = jnp.zeros(3)

    @jax.jit
    def grid_advect(_coeff, _basis, _geom):
        def body(carry, _index):
            field = (_basis @ carry).reshape(3, n, n, n)
            advected = _basis.T @ F.ifft(F.nonlinear(F.fft(field), _geom)).ravel()
            return carry, advected
        _, out = jax.lax.scan(body, _coeff, jnp.arange(2 * nsteps))
        return out[-1]

    @jax.jit
    def grid_shift(_coeff, _center, _basis):
        def body(carry, _index):
            vector, origin = carry
            field = (_basis @ vector).reshape(3, n, n, n)
            weight = jnp.sum(field * field, axis=0)
            angle = 2 * jnp.pi * jnp.arange(n, dtype=field.dtype) / n
            delta = []
            for axis in range(3):
                marginal = weight.sum(axis=tuple(i for i in range(3) if i != axis))
                delta.append(jnp.arctan2(jnp.sum(marginal * jnp.sin(angle)),
                                         jnp.sum(marginal * jnp.cos(angle))) / (2 * jnp.pi) % 1)
            delta = jnp.stack(delta)
            recentered = C._shift_field(field, -delta * n)
            return (_basis.T @ recentered.ravel(), (origin + delta) % 1), delta
        final, _ = jax.lax.scan(body, (_coeff, _center), jnp.arange(nsteps))
        return final[0]

    @jax.jit
    def coeff_advect(_coeff, _tensor):
        def body(carry, _index):
            advected = jnp.einsum("mjk,j,k->m", _tensor, carry, carry)
            return carry, advected
        _, out = jax.lax.scan(body, _coeff, jnp.arange(2 * nsteps))
        return out[-1]

    @jax.jit
    def coeff_shift(_coeff, _packed, _kx, _ky, _kz):
        delta = jnp.array([0.002, -0.001, 0.0015])

        def body(carry, _index):
            mixed = jnp.einsum("fck,k->fc", _packed, carry)
            phase = jnp.exp(-2j * jnp.pi * (
                _kx * (-delta[0]) + _ky * (-delta[1]) + _kz * (-delta[2])))
            mixed = mixed * phase[:, None]
            updated = jnp.real(jnp.einsum("fck,fc->k", jnp.conj(_packed), mixed)) / n ** 3
            return updated, updated
        final, _ = jax.lax.scan(body, _coeff, jnp.arange(nsteps))
        return final

    @jax.jit
    def outputs(_coeff, _basis):
        def body(carry, origin):
            field = (_basis @ carry).reshape(3, n, n, n)
            return carry, C._shift_field(field, origin * n)
        _, frames = jax.lax.scan(body, _coeff, jnp.zeros((6, 3)))
        return frames

    pieces = {
        "grid_advection_twice_per_step": lambda: grid_advect(coeff, basis_j, geom),
        "grid_centroid_shift_reproject": lambda: grid_shift(coeff, center, basis_j),
        "coeff_advection_twice_per_step": lambda: coeff_advect(coeff, tensor_j),
        "coeff_shift_reproject": lambda: coeff_shift(coeff, packed_j, kx_j, ky_j, kz_j),
        "six_output_shifts": lambda: outputs(coeff, basis_j),
    }
    report = {}
    for name, function in pieces.items():
        samples = time_only(function, repetitions)
        report[name] = dict(**D.median_summary(samples), nsteps=int(nsteps))
        D.log(f"profile {name} median_ms={report[name]['median_ms']:.3f}")
    return report


def fourier_floor(dev, ranks, n):
    centered = np.empty_like(dev)
    for case in range(len(dev)):
        for instant in range(dev.shape[1]):
            centered[case, instant], _ = D.center_field(dev[case, instant])
    rom = D.import_rom()
    largest = max(int(rank) for rank in ranks)
    basis, _, ids = rom.test_modes(n, largest)
    flat = centered.reshape(len(dev) * dev.shape[1], -1)
    report = {}
    chosen = None
    for rank in sorted(int(rank) for rank in ranks):
        phi = basis[:, :rank]
        recon = ((flat @ phi) @ phi.T).reshape(dev.shape)
        errors = np.empty((len(dev), dev.shape[1]), dtype=np.float64)
        for case in range(len(dev)):
            errors[case] = D.rel_rows(recon[case], centered[case], dev[case, 0])
        stats = D.stats_from_cases(errors, 0.05)
        report[str(rank)] = dict(evolved_worst=stats["evolved_worst"],
                                 cases_over=stats["cases_evolved_over_target"])
        D.log(f"fourier floor rank {rank} worst={stats['evolved_worst']:.6f}")
        if chosen is None and stats["evolved_worst"] <= 0.05:
            chosen = rank
    if chosen is None:
        return report, None, None, None
    return report, basis[:, :chosen], ids[:chosen], chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)
    reject_seeds(cfg)
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
        schema="ns3d-grok-diag06-v1", config=cfg, smoke=bool(args.smoke),
        final_cohort_opened=False, sealed_seed_opened=False,
        source_commit=os.environ.get("SOURCE_COMMIT"), job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()),
        scheme="Coefficient form of the diag04 startup tracker. Not multistep CNAB2.",
    )
    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_parameters = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    D.log("generating centered-POD training data and development truth")
    train, _, _ = D.generate(train_parameters, n, float(cfg["dt_truth"]), horizon)
    dev, _, _ = D.generate(dev_parameters, n, float(cfg["dt_truth"]), horizon)
    np.save(out / "dev_truth.npy", dev)
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = D.center_field(train[case, instant])
    del train
    gc.collect()
    snapshots = np.ascontiguousarray(centered.reshape(centered.shape[0] * centered.shape[1], -1))
    del centered
    gc.collect()
    rank = int(cfg["rank"])
    basis, _, orth = D.pod_basis(snapshots, rank, int(cfg.get("gram_block", 512)))
    del snapshots
    gc.collect()
    if int(basis.shape[1]) < rank:
        raise RuntimeError(f"POD rank {basis.shape[1]} is below the frozen rank {rank}")
    q, _ = D.orthonormalize_prefix(basis, rank)
    del basis
    linear = D.linear_operator(q, n)
    geom = F.geometry(n)
    D.log("assembling the advection tensor and centroid forms")
    tensor, tensor_gap = C.assemble_advection_tensor(q, n, geom)
    sin_m, cos_m, centroid_gap = C.centroid_matrices(q, n)
    packed, kx, ky, kz = C.fourier_pack(q, n)
    report["operators"] = dict(
        pod_orth_error=orth, tensor_probe_gap=tensor_gap, centroid_probe_gap=centroid_gap,
        **C.check_shift_identity(q, packed, kx, ky, kz, n), n_freq_full=int(packed.shape[0]),
    )
    D.log(f"operator gaps tensor={tensor_gap:.3e} centroid={centroid_gap:.3e} "
          f"shift={report['operators']['shift_gap']:.3e}")
    probe_rank = min(int(value) for value in cfg["fourier_ranks"])
    probe_basis, _, probe_ids = D.import_rom().test_modes(n, probe_rank)
    report["operators"]["phase_gap"] = C.check_phase_rotation(probe_basis, probe_ids, n)
    D.log(f"phase rotation gap {report['operators']['phase_gap']:.3e}")
    viscosities = dev_parameters[:, -1]
    repetitions = int(cfg["timing_repetitions"])
    target = float(cfg["target_relative"])
    parity_gate = float(cfg["parity_relative"])
    basis_j = jax.device_put(jnp.asarray(q))
    linear_j = jax.device_put(jnp.asarray(linear))
    tensor_j = jax.device_put(jnp.asarray(tensor))
    sin_j = jax.device_put(jnp.asarray(sin_m))
    cos_j = jax.device_put(jnp.asarray(cos_m))
    dummy_wave = jnp.zeros((1, 3))
    grid = diag04.make_tracked_run(step, steps, out_every, n)
    coeff = C.make_coeff_run(step, steps, out_every, n, "pod")
    tracker_fields, tracker_times = paired_rollout(
        grid, (basis_j, linear_j, geom), dev, viscosities, repetitions, "tracker")
    np.save(out / "tracker.npy", tracker_fields)
    report["tracker"] = arm_record("tracker", tracker_fields, dev, tracker_times, tracker_fields, target)
    report["tracker"]["parity_worst"] = 0.0
    D.log(f"tracker worst={report['tracker']['stats']['evolved_worst']:.6f} "
          f"median_ms={report['tracker']['median_ms']:.3f}")
    full_args = (basis_j, linear_j, tensor_j, jnp.asarray(packed), jnp.asarray(kx),
                 jnp.asarray(ky), jnp.asarray(kz), sin_j, cos_j, dummy_wave)
    coeff_fields, coeff_times = paired_rollout(
        coeff, full_args, dev, viscosities, repetitions, "coeff_full")
    np.save(out / "coeff_full.npy", coeff_fields)
    report["coeff_full"] = arm_record("coeff_full", coeff_fields, dev, coeff_times, tracker_fields, target)
    report["coeff_full"]["kept"] = eligible(report["coeff_full"], parity_gate)
    D.log(f"coeff_full parity={report['coeff_full']['parity_worst']:.3e} "
          f"median_ms={report['coeff_full']['median_ms']:.3f}")

    report["truncations"] = []
    for tail in cfg["truncation_tails"]:
        cut, cut_x, cut_y, cut_z, count, energy = C.truncate_pack(packed, kx, ky, kz, float(tail))
        if count == packed.shape[0]:
            D.log(f"truncation tail {tail} keeps every frequency")
            continue
        arguments = (basis_j, linear_j, tensor_j, jnp.asarray(cut), jnp.asarray(cut_x),
                     jnp.asarray(cut_y), jnp.asarray(cut_z), sin_j, cos_j, dummy_wave)
        fields, samples = paired_rollout(
            coeff, arguments, dev, viscosities, repetitions, f"trunc-{tail}")
        tag = f"{float(tail):.0e}".replace("+", "").replace("-", "m")
        np.save(out / f"trunc_{tag}.npy", fields)
        record = arm_record(f"trunc_{tag}", fields, dev, samples, tracker_fields, target)
        record.update(tail=float(tail), n_freq=count, energy_kept=energy, file=f"trunc_{tag}.npy")
        record["kept"] = eligible(record, parity_gate)
        report["truncations"].append(record)
        D.log(f"trunc tail {tail} freq={count} parity={record['parity_worst']:.3e} "
              f"median_ms={record['median_ms']:.3f}")

    def as_f32(value):
        array = jnp.asarray(value)
        if jnp.iscomplexobj(array):
            return array.astype(jnp.complex64)
        return array.astype(jnp.float32)

    f32_args = tuple(as_f32(value) for value in full_args)
    f32_fields, f32_times = paired_rollout(
        coeff, f32_args, dev, viscosities, repetitions, "f32")
    np.save(out / "coeff_f32.npy", f32_fields)
    report["f32"] = arm_record("f32", f32_fields, dev, f32_times, coeff_fields, target)
    report["f32"]["parity_reference"] = "coeff_full"
    report["f32"]["kept"] = bool(
        report["f32"]["under_target"]
        and report["f32"]["parity_worst"] <= float(cfg["f32_parity_relative"])
        and report["f32"]["median_ms"] <= float(cfg["f32_speed_factor"]) * report["coeff_full"]["median_ms"])
    D.log(f"f32 parity_vs_f64={report['f32']['parity_worst']:.3e} median_ms={report['f32']['median_ms']:.3f}")

    report["profile"] = profile_pieces(
        q, linear, tensor, packed, kx, ky, kz, geom, n, steps, repetitions)
    report["fourier_floor"], fourier_basis, fourier_ids, fourier_rank = fourier_floor(
        dev, cfg["fourier_ranks"], n)
    report["fourier"] = None
    if fourier_basis is not None:
        fourier_linear = D.linear_operator(fourier_basis, n)
        fourier_tensor, fourier_tensor_gap = C.assemble_advection_tensor(fourier_basis, n, geom)
        fourier_sin, fourier_cos, fourier_centroid_gap = C.centroid_matrices(fourier_basis, n)
        phase_gap = C.check_phase_rotation(fourier_basis, fourier_ids, n)
        wave = C.fourier_pair_waves(fourier_ids)
        fourier_grid = diag04.make_tracked_run(step, steps, out_every, n)
        fourier_basis_j = jax.device_put(jnp.asarray(fourier_basis))
        fourier_linear_j = jax.device_put(jnp.asarray(fourier_linear))
        grid_fields, _grid_times = paired_rollout(
            fourier_grid, (fourier_basis_j, fourier_linear_j, geom), dev, viscosities,
            repetitions, "fourier-grid")
        np.save(out / "fourier_grid.npy", grid_fields)
        fourier_run = C.make_coeff_run(step, steps, out_every, n, "fourier")
        dummy_pack = jnp.zeros((1, 3, fourier_basis.shape[1]), dtype=jnp.float64)
        dummy_freq = jnp.zeros((1,))
        arguments = (fourier_basis_j, fourier_linear_j, jnp.asarray(fourier_tensor),
                     dummy_pack, dummy_freq, dummy_freq, dummy_freq,
                     jnp.asarray(fourier_sin), jnp.asarray(fourier_cos), jnp.asarray(wave))
        fields, samples = paired_rollout(
            fourier_run, arguments, dev, viscosities, repetitions, "fourier")
        np.save(out / "fourier.npy", fields)
        record = arm_record("fourier", fields, dev, samples, grid_fields, target)
        record.update(rank=int(fourier_rank), tensor_probe_gap=fourier_tensor_gap,
                      centroid_probe_gap=fourier_centroid_gap, phase_gap=phase_gap,
                      parity_reference="fourier_grid")
        record["kept"] = eligible(record, parity_gate)
        report["fourier"] = record
        D.log(f"fourier rank {fourier_rank} parity={record['parity_worst']:.3e} "
              f"worst={record['stats']['evolved_worst']:.6f} median_ms={record['median_ms']:.3f}")

    report["fom"] = {}
    for fom_dt in cfg["fom_dts"]:
        fom_dt = float(fom_dt)
        fom_steps = D.nsteps_for(fom_dt, horizon)
        solver = F.make_solver(fom_dt, fom_steps, fom_steps // 5)
        fields, samples = paired_rollout(solver, (geom,), dev, viscosities, repetitions, f"fom-{fom_dt}")
        tag = D.file_dt(fom_dt)
        np.save(out / f"fom_dt{tag}.npy", fields)
        errors = case_errors(fields, dev)
        report["fom"][tag] = dict(
            dt=fom_dt, stats=D.stats_from_cases(errors, target), errors=errors.tolist(),
            median_ms=float(np.median(samples)), repetition_ms=samples)
        D.log(f"fom dt={fom_dt} worst={report['fom'][tag]['stats']['evolved_worst']:.6f} "
              f"median_ms={report['fom'][tag]['median_ms']:.3f}")

    candidates = []
    if report["coeff_full"]["kept"]:
        candidates.append(report["coeff_full"])
    candidates.extend(record for record in report["truncations"] if record["kept"])
    if report["f32"]["kept"]:
        candidates.append(report["f32"])
    if report["fourier"] is not None and report["fourier"]["kept"]:
        candidates.append(report["fourier"])
    if candidates:
        chosen = min(candidates, key=lambda record: record["median_ms"])
        report["selection"] = dict(name=chosen["name"], median_ms=chosen["median_ms"],
                                   evolved_worst=chosen["stats"]["evolved_worst"])
    else:
        report["selection"] = None
    report["elapsed_seconds"] = time.time() - started
    D.dump_json(out / "summary.json", report)
    D.log(f"wrote {out / 'summary.json'} selection={report['selection']}")
    if args.smoke and report["coeff_full"]["parity_worst"] > float(cfg["parity_relative"]):
        raise RuntimeError("smoke coefficient tracker missed the parity gate")


if __name__ == "__main__":
    main()
