"""Development diagnosis: POD snapshot floor versus CNAB2 Galerkin rollout.

The final cohort is not drawn. POD is fit on the training seed only.
Errors are relative L2 against each case's initial field, matching ns3d_fom.relative_errors.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "ns3d"))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import ns3d_fom as F  # noqa: E402
import translation_cohort as TC  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def log(message):
    print(message, flush=True)


def require_finite(array, label):
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"nonfinite values in {label}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rel_rows(pred, ref, u0):
    """Per-time relative L2. pred/ref (T, ...), u0 (...)."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    num = np.linalg.norm(diff.reshape(len(diff), -1), axis=1)
    den = np.linalg.norm(np.asarray(u0, dtype=np.float64))
    return num / max(den, 1e-300)


def sumsq_rows(pred, ref, u0):
    """Independent reduction of the same relative error, for the in-process check."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    num = np.sqrt(np.sum(diff * diff, axis=tuple(range(1, diff.ndim))))
    den = np.sqrt(np.sum(np.asarray(u0, dtype=np.float64) ** 2))
    return num / max(float(den), 1e-300)


def stats_from_cases(errors, target):
    """errors is (cases, times). Evolved statistics ignore time 0."""
    errors = np.asarray(errors, dtype=np.float64)
    evolved = errors[:, 1:]
    worst = evolved.max(axis=1)
    return dict(
        per_time_median=[float(x) for x in np.median(errors, axis=0)],
        per_time_worst=[float(x) for x in errors.max(axis=0)],
        initial_median=float(np.median(errors[:, 0])),
        initial_worst=float(errors[:, 0].max()),
        evolved_median=float(np.median(worst)),
        evolved_worst=float(worst.max()),
        evolved_mean=float(worst.mean()),
        all_times_worst=float(errors.max()),
        all_times_median_of_case_max=float(np.median(errors.max(axis=1))),
        cases_initial_over_target=int(np.sum(errors[:, 0] > target)),
        cases_evolved_over_target=int(np.sum(worst > target)),
        cases_grew=int(np.sum(worst > errors[:, 0] + 1e-12)),
        median_evolved_minus_initial=float(np.median(worst - errors[:, 0])),
        max_evolved_minus_initial=float(np.max(worst - errors[:, 0])),
        cases=int(errors.shape[0]),
        times=int(errors.shape[1]),
    )


def fourier_shift(field, offset_samples):
    """Fractional periodic shift. Positive offset matches np.roll toward +index."""
    spec = np.fft.fftn(np.asarray(field, dtype=np.float64), axes=(-3, -2, -1))
    n = field.shape[-1]
    for axis, shift in enumerate(np.asarray(offset_samples, dtype=np.float64)):
        modes = np.fft.fftfreq(n) * n
        phase = np.exp(-2j * np.pi * modes * (shift / n))
        shape = [1, 1, 1]
        shape[axis] = n
        spec = spec * phase.reshape((1, *shape))
    return np.fft.ifftn(spec, axes=(-3, -2, -1)).real


def energy_centroid(field):
    """Circular mean of |u|^2 on the unit torus, in [0, 1)^3."""
    weight = np.sum(np.asarray(field, dtype=np.float64) ** 2, axis=0)
    n = field.shape[-1]
    coord = np.arange(n, dtype=np.float64) / n
    center = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        marginal = weight.sum(axis=tuple(i for i in range(3) if i != axis))
        angle = 2 * np.pi * coord
        center[axis] = np.arctan2(np.sum(marginal * np.sin(angle)),
                                  np.sum(marginal * np.cos(angle))) / (2 * np.pi) % 1.0
    return center


def torus_delta(a, b):
    raw = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(raw, 1.0 - raw)


def center_field(field):
    center = energy_centroid(field)
    return fourier_shift(field, -center * field.shape[-1]), center


def shift_self_check(field):
    rolled = np.roll(field, (2, -1, 3), axis=(-3, -2, -1))
    spectral = fourier_shift(field, (2, -1, 3))
    roll_error = float(np.linalg.norm(spectral - rolled) / np.linalg.norm(rolled))
    c0 = energy_centroid(field)
    c1 = energy_centroid(rolled)
    expected = (c0 + np.array([2, -1, 3]) / field.shape[-1]) % 1.0
    centroid_error = float(np.max(torus_delta(c1, expected)))
    centered, center = center_field(field)
    recentered, _ = center_field(rolled)
    match = float(np.linalg.norm(centered - recentered) / np.linalg.norm(centered))
    residual = float(np.max(torus_delta(energy_centroid(centered), np.zeros(3))))
    return dict(integer_shift_relative=roll_error, centroid_shift_absolute=centroid_error,
                centered_translate_relative=match, centered_residual=residual,
                passed=bool(roll_error < 1e-8 and centroid_error < 1e-8
                            and match < 1e-6 and residual < 1e-4))


def nsteps_for(dt, horizon):
    steps = int(round(horizon / dt))
    if abs(steps * dt - horizon) > 1e-12:
        raise RuntimeError(f"horizon {horizon} is not an integer multiple of dt {dt}")
    return steps


def generate(params, n, dt, horizon):
    steps = nsteps_for(dt, horizon)
    out_every = steps // 5
    if out_every < 1 or steps % 5:
        raise RuntimeError("need five equal output intervals")
    solver = F.make_solver(dt, steps, out_every)
    geom = F.geometry(n)
    frames = np.empty((len(params), steps // out_every + 1, 3, n, n, n), dtype=np.float64)
    for index, parameter in enumerate(params):
        initial = jnp.asarray(F.initial(n, parameter))
        frames[index] = np.asarray(solver(initial, float(parameter[-1]), geom))
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(params):
            log(f"generated {index + 1}/{len(params)} dt={dt}")
    return frames, steps, out_every


def write_augmented(base, table):
    """Materialize integer translates in memory. Callers must not also keep `base`."""
    frames = base.shape[1]
    spatial = int(np.prod(base.shape[2:]))
    snapshots = np.empty((len(table) * frames, spatial), dtype=np.float64)
    for index, row in enumerate(table):
        base_index, ox, oy, oz = (int(v) for v in row)
        rolled = np.roll(base[base_index], (ox, oy, oz), axis=(-3, -2, -1))
        snapshots[index * frames:(index + 1) * frames] = np.ascontiguousarray(rolled).reshape(frames, spatial)
        if index == 0 or (index + 1) % 256 == 0 or index + 1 == len(table):
            log(f"augmented {index + 1}/{len(table)}")
    return snapshots


def gram_matrix(snapshots, block):
    count = snapshots.shape[0]
    gram = np.empty((count, count), dtype=np.float64)
    for i0 in range(0, count, block):
        left = jnp.asarray(np.array(snapshots[i0:i0 + block], dtype=np.float64, copy=True))
        for j0 in range(0, count, block):
            right = jnp.asarray(np.array(snapshots[j0:j0 + block], dtype=np.float64, copy=True))
            gram[i0:i0 + block, j0:j0 + block] = np.asarray(left @ right.T)
        log(f"gram rows {min(i0 + block, count)}/{count}")
    return 0.5 * (gram + gram.T)


def pod_basis(snapshots, max_rank, block):
    gram = gram_matrix(snapshots, block)
    log(f"eigh dimension {gram.shape[0]}")
    try:
        evals, evecs = jnp.linalg.eigh(jnp.asarray(gram))
        evals = np.asarray(evals)
        evecs = np.asarray(evecs)
    except Exception as exc:
        log(f"gpu eigh failed ({type(exc).__name__}: {exc}); falling back to numpy")
        evals, evecs = np.linalg.eigh(gram)
    del gram
    gc.collect()
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    keep = int(np.sum(evals > max(evals[0], 0.0) * 1e-12))
    rank = min(max_rank, keep)
    if rank < 1:
        raise RuntimeError("POD spectrum has no positive eigenvalue")
    weights = evecs[:, :rank] / np.sqrt(evals[:rank])
    spatial = snapshots.shape[1]
    basis = jnp.zeros((spatial, rank), dtype=jnp.float64)
    for start in range(0, snapshots.shape[0], block):
        block_x = jnp.asarray(np.array(snapshots[start:start + block], dtype=np.float64, copy=True))
        basis = basis + block_x.T @ jnp.asarray(weights[start:start + block])
    basis = np.asarray(basis)
    gram_basis = np.asarray(jnp.asarray(basis).T @ jnp.asarray(basis))
    orth = float(np.max(np.abs(gram_basis - np.eye(rank))))
    log(f"pod rank={rank} orth={orth:.3e} energy_tail={float(evals[rank - 1] / evals[0]):.3e}")
    return basis, evals[:rank], orth


def orthonormalize_prefix(basis, rank):
    """QR of a POD prefix. The column span is unchanged; R maps raw coefficients."""
    raw = np.ascontiguousarray(basis[:, :rank])
    q, factor = np.linalg.qr(raw, mode="reduced")
    return np.ascontiguousarray(q), np.ascontiguousarray(factor)


def project_errors(basis, frames):
    """Orthogonal projection error of every saved field. basis is (dof, rank)."""
    cases, times = frames.shape[:2]
    flat = frames.reshape(cases * times, -1)
    coeff = flat @ basis
    recon = (coeff @ basis.T).reshape(frames.shape)
    errors = np.empty((cases, times), dtype=np.float64)
    for case in range(cases):
        errors[case] = rel_rows(recon[case], frames[case], frames[case, 0])
        check = sumsq_rows(recon[case], frames[case], frames[case, 0])
        if float(np.max(np.abs(errors[case] - check))) > 1e-12:
            raise RuntimeError("projection error reductions disagree")
    return errors, recon


def oracle_shift_errors(basis, frames):
    cases, times = frames.shape[:2]
    errors = np.empty((cases, times), dtype=np.float64)
    recon = np.empty_like(frames)
    centroids = np.empty((cases, times, 3), dtype=np.float64)
    n = frames.shape[-1]
    for case in range(cases):
        for instant in range(times):
            field = frames[case, instant]
            center = energy_centroid(field)
            centroids[case, instant] = center
            centered = fourier_shift(field, -center * n)
            coeff = basis.T @ centered.ravel()
            restored = fourier_shift((basis @ coeff).reshape(field.shape), center * n)
            recon[case, instant] = restored
        errors[case] = rel_rows(recon[case], frames[case], frames[case, 0])
    travel = np.max(torus_delta(centroids, centroids[:, :1, :]), axis=(1, 2))
    return errors, recon, dict(
        centroid_travel_median=float(np.median(travel)),
        centroid_travel_worst=float(np.max(travel)),
    )


def coefficient_pca_errors(train_coeff, dev_frames, basis, components):
    """Best affine K-dimensional reconstruction inside a fixed orthonormal bank."""
    mean = train_coeff.mean(axis=0)
    centered = train_coeff - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dev_flat = dev_frames.reshape(len(dev_frames) * dev_frames.shape[1], -1)
    dev_coeff = dev_flat @ basis
    reports = {}
    saved = {}
    for width in components:
        if width > vt.shape[0]:
            continue
        axes = vt[:width]
        hat = (dev_coeff - mean) @ axes.T @ axes + mean
        recon = (hat @ basis.T).reshape(dev_frames.shape)
        errors = np.empty((len(dev_frames), dev_frames.shape[1]), dtype=np.float64)
        for case in range(len(dev_frames)):
            errors[case] = rel_rows(recon[case], dev_frames[case], dev_frames[case, 0])
        reports[str(width)] = errors
        saved[str(width)] = recon
    return reports, saved


def galerkin_errors(runner, frames, viscosities, basis, linear, geom):
    cases = len(frames)
    errors = np.empty((cases, frames.shape[1]), dtype=np.float64)
    predicted = np.empty_like(frames)
    gate = []
    basis_j = jax.device_put(jnp.asarray(basis))
    linear_j = jax.device_put(jnp.asarray(linear))
    for case in range(cases):
        initial = jnp.asarray(frames[case, 0])
        out = np.asarray(runner(initial, float(viscosities[case]), basis_j, linear_j, geom))
        predicted[case] = out.reshape(frames.shape[1:])
        projection = np.asarray(basis_j.T @ initial.ravel())
        frame0 = basis @ projection
        gap = float(np.linalg.norm(predicted[case, 0].ravel() - frame0) / np.linalg.norm(frame0))
        gate.append(gap)
        errors[case] = rel_rows(predicted[case], frames[case], frames[case, 0])
        check = sumsq_rows(predicted[case], frames[case], frames[case, 0])
        if float(np.max(np.abs(errors[case] - check))) > 1e-12:
            raise RuntimeError("rollout error reductions disagree")
    if max(gate) > 1e-8:
        raise RuntimeError(f"Galerkin frame 0 is not the orthogonal projection: {max(gate)}")
    return errors, predicted, float(max(gate))


def one_interval_errors(runner, frames, viscosities, basis, linear, geom):
    """Restart from the truth at each saved time and advance one output interval."""
    cases, times = frames.shape[:2]
    errors = np.empty((cases, times - 1), dtype=np.float64)
    predicted = np.empty((cases, times - 1, *frames.shape[2:]), dtype=np.float64)
    basis_j = jax.device_put(jnp.asarray(basis))
    linear_j = jax.device_put(jnp.asarray(linear))
    for case in range(cases):
        for instant in range(times - 1):
            initial = jnp.asarray(frames[case, instant])
            out = np.asarray(runner(initial, float(viscosities[case]), basis_j, linear_j, geom))
            predicted[case, instant] = out[-1].reshape(frames.shape[2:])
        for instant in range(times - 1):
            num = np.linalg.norm(predicted[case, instant] - frames[case, instant + 1])
            errors[case, instant] = num / np.linalg.norm(frames[case, 0])
    return errors, predicted


def import_rom():
    import ns3d_rom as rom
    return rom


def linear_operator(basis, n):
    rom = import_rom()
    return rom.dense_galerkin_linear(basis, n)


def time_calls(function, repetitions):
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        value = function()
        jax.block_until_ready(value)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def profile_pieces(basis, geom, n, repetitions):
    """Median device time of the four pieces of one dense Galerkin stage."""
    rom_basis = jax.device_put(jnp.asarray(basis))
    coeff = jax.device_put(jnp.ones(basis.shape[1], dtype=jnp.float64))
    eye = jnp.eye(basis.shape[1], dtype=jnp.float64)

    @jax.jit
    def decode(basis, coeff):
        return (basis @ coeff).reshape(3, n, n, n)

    @jax.jit
    def nonlinear_only(field, geom):
        return F.ifft(F.nonlinear(F.fft(field), geom))

    @jax.jit
    def project_back(basis, field):
        return basis.T @ field.ravel()

    @jax.jit
    def solve_only(factor, rhs):
        return jax.scipy.linalg.cho_solve((factor, True), rhs)

    field = decode(rom_basis, coeff)
    projected = project_back(rom_basis, field)
    @jax.jit
    def factor_only(matrix):
        return jnp.linalg.cholesky(matrix)

    matrix = eye
    factor = factor_only(matrix)
    for fn, args in ((decode, (rom_basis, coeff)), (nonlinear_only, (field, geom)),
                     (project_back, (rom_basis, field)), (solve_only, (factor, projected)),
                     (factor_only, (matrix,))):
        jax.block_until_ready(fn(*args))
    return dict(
        decode_ms=time_calls(lambda: decode(rom_basis, coeff), repetitions),
        nonlinear_ms=time_calls(lambda: nonlinear_only(field, geom), repetitions),
        project_ms=time_calls(lambda: project_back(rom_basis, field), repetitions),
        cholesky_factor_ms=time_calls(lambda: factor_only(matrix), repetitions),
        cholesky_solve_ms=time_calls(lambda: solve_only(factor, projected), repetitions),
    )


def weak_profile(basis, frames, viscosity, geom, n, dt, horizon, modes, budget, repetitions):
    """One production dense weak POD query, for the cost split against Galerkin."""
    rom = import_rom()
    steps = nsteps_for(dt, horizon)
    out_every = steps // 5
    phi, lam, ids = rom.test_modes(n, modes)
    selector = (np.asarray([row["wave"] for row in ids]) % n,
                np.asarray([row["polarization"] for row in ids]),
                np.asarray([row["kind"] == "cos" for row in ids]), geom)
    rank = basis.shape[1]
    arguments = (basis, basis, np.eye(rank), phi.T @ basis, selector, lam,
                 np.empty((rank, 0)), {}, np.empty((1, 0)))
    arguments = jax.tree_util.tree_map(jax.device_put, arguments)
    runner = rom.make_run(dt, steps, out_every, 0, rank, linear=True, budget=budget,
                          gtol=1e-7, retain_states=False, dense_n=n)
    initial = jnp.asarray(frames[0])
    warmup = runner(initial, float(viscosity), *arguments)
    jax.block_until_ready(warmup)
    samples = time_calls(lambda: runner(initial, float(viscosity), *arguments), repetitions)
    info = warmup[2]
    iterations = np.asarray(info[1])
    reasons = np.asarray(info[2])
    return dict(repetition_ms=samples, median_ms=float(np.median(samples)),
                iterations_per_step=iterations.tolist(),
                median_iterations=float(np.median(iterations)),
                stopping_reasons=[int(x) for x in reasons],
                test_modes=int(modes), budget=int(budget), rank=int(rank), dt=float(dt))


def median_summary(samples):
    array = np.asarray(samples, dtype=np.float64)
    return dict(median_ms=float(np.median(array)), min_ms=float(array.min()),
                max_ms=float(array.max()), repetitions=array.tolist())


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_dt(value):
    return f"{float(value):.4f}".replace(".", "p")


def apply_smoke(cfg):
    cfg = dict(cfg)
    cfg.update(n=8, dt_truth=0.01, horizon=0.05, rom_dt=0.01,
               rollout_dt_extra_on_largest_rank=0.01, train_cases=4, dev_cases=2,
               augmentation_copies=1, ranks=[4, 8], pca_components=[2], timing_cases=1,
               timing_repetitions=2, weak_profile_rank=4, weak_profile_modes=8,
               weak_profile_budget=4, gram_block=4)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg = apply_smoke(cfg)
    if int(cfg["train_seed"]) == int(cfg["final_seed_closed"]) or int(cfg["dev_seed"]) == int(cfg["final_seed_closed"]):
        raise RuntimeError("refusing to draw the closed final seed")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 2**30
    log(f"disk_free_gb={free_gb:.1f} devices={jax.devices()}")
    if free_gb < (2.0 if args.smoke else 40.0):
        raise RuntimeError(f"not enough free disk: {free_gb:.1f} GB")

    if jax.default_backend() != "gpu" or jnp.zeros((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError(f"need a float64 GPU backend, got {jax.default_backend()}")
    n = int(cfg["n"])
    dt = float(cfg["dt_truth"])
    horizon = float(cfg["horizon"])
    started = time.time()
    report = dict(
        schema="ns3d-grok-diag01-v1",
        config=cfg,
        smoke=bool(args.smoke),
        final_cohort_opened=False,
        source_commit=os.environ.get("SOURCE_COMMIT"),
        job_id=os.environ.get("SLURM_JOB_ID"),
        device=str(jax.devices()),
        files={name: sha256_file(ROOT / name) for name in (
            "experiments/ns3d-grok/diag_floor.py",
            "experiments/ns3d/ns3d_fom.py",
            "experiments/ns3d/ns3d_rom.py",
            "experiments/ns3d/translation_cohort.py",
        )},
    )
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu_name = "unavailable"
    report["gpu"] = gpu_name

    train_parameters = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_parameters = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["train_parameter_sha256"] = sha256_array(train_parameters)
    report["dev_parameter_sha256"] = sha256_array(dev_parameters)
    log("generating training and development trajectories")
    train, steps, every = generate(train_parameters, n, dt, horizon)
    dev, _, _ = generate(dev_parameters, n, dt, horizon)
    require_finite(train, "training trajectories")
    require_finite(dev, "development trajectories")
    np.save(out / "dev_truth.npy", dev)
    report["truth"] = dict(dt=dt, steps=steps, out_every=every, frames=int(dev.shape[1]),
                           dev_truth_sha256=sha256_file(out / "dev_truth.npy"))
    report["shift_self_check"] = shift_self_check(dev[0, 0])
    log(f"shift_self_check {report['shift_self_check']}")
    if not report["shift_self_check"]["passed"]:
        raise RuntimeError("Fourier shift or energy centroid failed its identity check")

    requested = [int(rank) for rank in cfg["ranks"]]
    components = [int(width) for width in cfg["pca_components"]]
    block = int(cfg.get("gram_block", 512))
    viscosities = dev_parameters[:, -1]
    geom = F.geometry(n)

    log("oracle-shift POD on centered training snapshots")
    centered = np.empty_like(train)
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            centered[case, instant], _ = center_field(train[case, instant])
    shift_snapshots = centered.reshape(len(centered) * centered.shape[1], -1)
    shift_basis, shift_energy, shift_orth = pod_basis(shift_snapshots, max(requested), block)
    del centered, shift_snapshots
    gc.collect()
    shift_rows = {}
    shift_ranks = [rank for rank in requested if rank <= shift_basis.shape[1]]
    if int(shift_basis.shape[1]) not in shift_ranks:
        shift_ranks.append(int(shift_basis.shape[1]))
    for rank in shift_ranks:
        q, _ = orthonormalize_prefix(shift_basis, rank)
        errors, recon, travel = oracle_shift_errors(q, dev)
        shift_rows[str(rank)] = dict(stats=stats_from_cases(errors, float(cfg["target_relative"])),
                                      errors=errors.tolist(), centroid_travel=travel)
        if rank in (shift_ranks[0], shift_ranks[-1]):
            np.save(out / f"oracle_shift_r{rank}.npy", recon)
        log(f"oracle shift rank {rank} evolved worst {shift_rows[str(rank)]['stats']['evolved_worst']:.6f}")
    report["oracle_shift"] = dict(
        note="Centroid taken from the true field. This is a representation floor, not a solved model. "
             "Training snapshots are centered base trajectories; integer translates would duplicate them.",
        snapshots=int(cfg["train_cases"]) * int(dev.shape[1]),
        orth_error_before_prefix_qr=shift_orth,
        spectrum_head=[float(x) for x in shift_energy[:8]],
        ranks=shift_rows,
    )
    del shift_basis
    gc.collect()
    dump_json(out / "summary_partial.json", report)

    log("plain POD on integer-translated training trajectories")
    table = TC.membership(len(train), n, int(cfg["augmentation_copies"]), int(cfg["augmentation_seed"]))
    snapshots = write_augmented(train, table)
    del train
    gc.collect()
    basis, energy, orth = pod_basis(snapshots, max(requested), block)
    max_rank = basis.shape[1]
    train_coeff = np.empty((snapshots.shape[0], max_rank), dtype=np.float64)
    coeff_block = min(block, 256)
    for start in range(0, snapshots.shape[0], coeff_block):
        block_x = jnp.asarray(np.array(snapshots[start:start + coeff_block], dtype=np.float64, copy=True))
        train_coeff[start:start + coeff_block] = np.asarray(block_x @ jnp.asarray(basis))
    del snapshots
    gc.collect()
    report["plain_pod"] = dict(
        augmented_snapshots=int(train_coeff.shape[0]),
        membership_sha256=sha256_array(table),
        orth_error=orth,
        spectrum_head=[float(x) for x in energy[:8]],
        available_rank=int(max_rank),
    )

    rom = import_rom()
    ranks = [rank for rank in requested if rank <= max_rank]
    if int(max_rank) not in ranks:
        ranks.append(int(max_rank))
    ranks = sorted(set(ranks))
    if not ranks:
        raise RuntimeError(f"no requested rank fits the POD spectrum of size {max_rank}")
    report["ranks"] = {}
    runners = {}
    interval_steps = every
    runners["truth"] = rom.make_dense_galerkin_run(dt, steps, every, n)
    runners["interval"] = rom.make_dense_galerkin_run(dt, interval_steps, interval_steps, n)
    extra_dt = float(cfg["rollout_dt_extra_on_largest_rank"])
    extra_steps = nsteps_for(extra_dt, horizon)
    runners["extra"] = rom.make_dense_galerkin_run(extra_dt, extra_steps, extra_steps // 5, n)

    previous_floor = None
    for rank in ranks:
        q, factor = orthonormalize_prefix(basis, rank)
        coeff_q = np.linalg.solve(factor.T, train_coeff[:, :rank].T).T
        linear = linear_operator(q, n)
        floor, recon = project_errors(q, dev)
        floor_all = float(floor.max())
        if previous_floor is not None and floor_all > previous_floor + 1e-4:
            raise RuntimeError(f"projection floor rose from {previous_floor} to {floor_all} at rank {rank}")
        previous_floor = floor_all
        np.save(out / f"projection_r{rank}.npy", recon)
        rollout, predicted, frame0 = galerkin_errors(
            runners["truth"], dev, viscosities, q, linear, geom)
        np.save(out / f"galerkin_r{rank}_dt{file_dt(dt)}.npy", predicted)
        local, local_fields = one_interval_errors(
            runners["interval"], dev, viscosities, q, linear, geom)
        np.save(out / f"onestep_r{rank}.npy", local_fields)
        require_finite(predicted, f"galerkin rank {rank}")
        require_finite(local_fields, f"one-interval rank {rank}")
        pca = {}
        pca_errors, pca_fields = coefficient_pca_errors(coeff_q, dev, q, components)
        for width, matrix in pca_errors.items():
            pca[width] = dict(stats=stats_from_cases(matrix, float(cfg["target_relative"])),
                              errors=matrix.tolist())
        if rank == ranks[-1] and pca_fields:
            keep_width = "64" if "64" in pca_fields else next(iter(pca_fields))
            np.save(out / f"pca_k{keep_width}_r{rank}.npy", pca_fields[keep_width])
            row_pca_saved = keep_width
        else:
            row_pca_saved = None
        row = dict(
            floor=dict(stats=stats_from_cases(floor, float(cfg["target_relative"])), errors=floor.tolist()),
            galerkin_dt_truth=dict(stats=stats_from_cases(rollout, float(cfg["target_relative"])),
                                   errors=rollout.tolist(), frame0_projection_gap=frame0, dt=dt),
            one_interval=dict(
                per_interval_worst=[float(x) for x in local.max(axis=0)],
                per_interval_median=[float(x) for x in np.median(local, axis=0)],
                worst=float(local.max()), median_of_case_worst=float(np.median(local.max(axis=1))),
                errors=local.tolist(), dt=dt, steps=int(interval_steps)),
            affine_pca_head=pca,
            saved_pca_width=row_pca_saved,
        )
        if rank == ranks[-1] and abs(extra_dt - dt) > 1e-15:
            extra_errors, extra_fields, extra_gap = galerkin_errors(
                runners["extra"], dev, viscosities, q, linear, geom)
            np.save(out / f"galerkin_r{rank}_dt{file_dt(extra_dt)}.npy", extra_fields)
            row["galerkin_dt_rom"] = dict(
                stats=stats_from_cases(extra_errors, float(cfg["target_relative"])),
                errors=extra_errors.tolist(), frame0_projection_gap=extra_gap, dt=extra_dt)
        report["ranks"][str(rank)] = row
        floor_worst = row["floor"]["stats"]["evolved_worst"]
        roll_worst = row["galerkin_dt_truth"]["stats"]["evolved_worst"]
        log(f"rank {rank} floor_evolved_worst={floor_worst:.6f} galerkin_evolved_worst={roll_worst:.6f} "
            f"one_interval_worst={row['one_interval']['worst']:.6f}")
        dump_json(out / "summary_partial.json", report)
        del q, linear, recon, predicted, local_fields
        gc.collect()

    log("timing FOM, Galerkin pieces, and one weak query")
    timing_cases = min(int(cfg["timing_cases"]), len(dev))
    repetitions = int(cfg["timing_repetitions"])
    fom_dts = sorted({dt, extra_dt, 0.01} if not args.smoke else {dt})
    fom_rows = {}
    for fom_dt in fom_dts:
        fom_steps = nsteps_for(fom_dt, horizon)
        if fom_steps % 5:
            raise RuntimeError(f"FOM dt {fom_dt} does not hit the five output intervals")
        solver = F.make_solver(fom_dt, fom_steps, fom_steps // 5)
        initial = jnp.asarray(dev[0, 0])
        jax.block_until_ready(solver(initial, float(viscosities[0]), geom))
        samples = []
        for case in range(timing_cases):
            u0 = jnp.asarray(dev[case, 0])
            nu = float(viscosities[case])
            samples.extend(time_calls(lambda u0=u0, nu=nu: solver(u0, nu, geom), repetitions))
        reference = np.asarray(solver(jnp.asarray(dev[0, 0]), float(viscosities[0]), geom))
        accuracy = rel_rows(reference, dev[0], dev[0, 0])
        fom_rows[f"dt{fom_dt}"] = dict(**median_summary(samples), case0_relative=accuracy.tolist(),
                                      cases=timing_cases, repetitions_per_case=repetitions,
                                      compared_with="development case 0 at the truth time step")
    report["timing"] = dict(fom=fom_rows, note="Warmup calls are discarded. Repetitions are paired on this job only.")

    piece_rank = min(256, ranks[-1])
    piece_basis, _ = orthonormalize_prefix(basis, piece_rank)
    pieces = profile_pieces(piece_basis, geom, n, repetitions)
    report["timing"]["galerkin_stage_pieces"] = {
        name: median_summary(values) for name, values in pieces.items()}
    report["timing"]["galerkin_stage_pieces"]["rank"] = piece_rank
    report["timing"]["galerkin_stage_pieces"]["note"] = (
        "Single stage on a fixed coefficient, after warmup. A full step evaluates the "
        "nonlinearity two or three times and solves one or two linear systems. "
        "Complete-trajectory times are in galerkin_rollout.")
    galerkin_times = {}
    for rank in sorted({int(rank) for rank in (64, 256, 1024, ranks[-1]) if int(rank) in ranks}):
        q, _ = orthonormalize_prefix(basis, rank)
        linear = jax.device_put(jnp.asarray(linear_operator(q, n)))
        basis_j = jax.device_put(jnp.asarray(q))
        warmup_state = jnp.asarray(dev[0, 0])
        jax.block_until_ready(runners["truth"](warmup_state, float(viscosities[0]), basis_j, linear, geom))
        samples = []
        for case in range(timing_cases):
            state = jnp.asarray(dev[case, 0])
            viscosity = float(viscosities[case])
            samples.extend(time_calls(
                lambda state=state, viscosity=viscosity, basis_j=basis_j, linear=linear:
                runners["truth"](state, viscosity, basis_j, linear, geom), repetitions))
        galerkin_times[str(rank)] = dict(**median_summary(samples), dt=dt, cases=timing_cases,
                                         repetitions_per_case=repetitions)
    report["timing"]["galerkin_rollout"] = galerkin_times

    weak_rank = min(int(cfg["weak_profile_rank"]), ranks[-1])
    weak_basis, _ = orthonormalize_prefix(basis, weak_rank)
    report["timing"]["weak_pod"] = weak_profile(
        weak_basis, dev[:, 0], viscosities[0], geom, n,
        float(cfg["rom_dt"]), horizon, int(cfg["weak_profile_modes"]),
        int(cfg["weak_profile_budget"]), repetitions)
    report["elapsed_seconds"] = time.time() - started
    dump_json(out / "summary.json", report)
    log(f"wrote {out / 'summary.json'} elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
