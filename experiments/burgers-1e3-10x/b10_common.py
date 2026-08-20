"""Shared f64 machinery for the preregistered Burgers 1e-3 / 10x search."""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import subprocess
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(HERE, "deps", "burgers2d-coord-rom"),
    os.path.abspath(os.path.join(
        HERE, "..", "..", "..", "2026-08-14-burgers2d-coord-rom",
        "experiments", "burgers2d-coord-rom")),
):
    if os.path.isfile(os.path.join(candidate, "burgers2d_film.py")):
        import sys

        sys.path.insert(0, candidate)
        break
else:
    raise ImportError("burgers2d_film.py not found")

# Only the audited FOM/data functions are used.  Keep its model defaults stable.
os.environ.setdefault("HIDDEN", "256")
import burgers2d_film as bf  # noqa: E402

F64 = jnp.float64
DT = float(bf.DT)
NUM_STEPS = int(bf.NUM_STEPS)


def log(*values):
    print(*values, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", HERE, *args], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    source_hashes = {}
    for path in sorted(glob.glob(os.path.join(HERE, "b10_*.py"))):
        source_hashes[os.path.basename(path)] = sha256(path)
    device = jax.devices()[0]
    return {
        "commit": os.environ.get("B10_COMMIT") or git("rev-parse", "HEAD"),
        "commit_source": "env:B10_COMMIT" if os.environ.get("B10_COMMIT") else "git",
        "source_sha256": source_hashes,
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
        "gpu": str(device),
        "gpu_kind": getattr(device, "device_kind", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "host": os.environ.get("HOSTNAME", os.uname().nodename),
    }


def require_gpu_highest():
    if jax.default_backend() != "gpu":
        raise SystemExit(f"GPU required, got {jax.default_backend()}")
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    if not jax.config.jax_enable_x64:
        raise SystemExit("f64/x64 must be enabled")


def save_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=1, allow_nan=False)


def grid_coords(n):
    axis = np.linspace(0.0, 1.0, n, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((x.reshape(-1), y.reshape(-1)), axis=1)


def binary_boundary_mask(n):
    mask = np.ones((n, n), dtype=np.float64)
    mask[[0, -1], :] = 0.0
    mask[:, [0, -1]] = 0.0
    return mask.reshape(-1)


def generate_population(n, seed, draw_count, indices, chunk=32):
    """Regenerate a fixed indexed population with the inherited reference FOM."""
    cx, cy, width, amplitude, nu, normalized = bf.sample_params(
        seed=int(seed), m=int(draw_count)
    )
    selected = np.asarray(indices, dtype=np.int64)
    rollout, residual = bf.make_rollout(int(n))
    fields = np.empty((selected.size, NUM_STEPS + 1, n * n), np.float64)
    worst = 0.0
    for start in range(0, selected.size, chunk):
        take = selected[start:start + chunk]
        u0 = np.stack([
            bf.blob_ic(n, cx[i], cy[i], width[i], amplitude[i]) for i in take
        ])
        snapshots, residuals = rollout(jnp.asarray(u0), jnp.asarray(nu[take]))
        snapshots = np.asarray(snapshots).transpose(1, 0, 2)
        residuals = np.asarray(residuals)
        fields[start:start + take.size] = snapshots
        chunk_worst = float(np.max(residuals))
        if not np.isfinite(chunk_worst) or chunk_worst > worst:
            worst = chunk_worst
    # Independent residual check on every generated step, batched by trajectory.
    residual_norm = jax.jit(jax.vmap(
        lambda u1, u0, viscosity: jnp.linalg.norm(residual(u1, u0, viscosity))
        / jnp.maximum(jnp.linalg.norm(u0), 1e-300)
    ))
    independent = 0.0
    for trajectory, viscosity in zip(fields, nu[selected]):
        values = np.asarray(residual_norm(
            jnp.asarray(trajectory[1:]), jnp.asarray(trajectory[:-1]),
            jnp.full((NUM_STEPS,), viscosity, dtype=F64)))
        value = float(np.max(values))
        if not np.isfinite(value) or value > independent:
            independent = value
    if not np.isfinite(worst) or not np.isfinite(independent):
        raise SystemExit("non-finite reference population")
    parameters = {
        "cx": np.asarray(cx[selected], np.float64),
        "cy": np.asarray(cy[selected], np.float64),
        "width": np.asarray(width[selected], np.float64),
        "amplitude": np.asarray(amplitude[selected], np.float64),
        "nu": np.asarray(nu[selected], np.float64),
        "normalized": np.asarray(normalized[selected], np.float64),
    }
    return fields, parameters, {
        "reported_max_relative_residual": worst,
        "independent_max_relative_residual": independent,
        "seed": int(seed),
        "draw_count": int(draw_count),
        "indices": [int(i) for i in selected],
    }


def hermite_pairs(degree):
    return tuple(
        (px, py) for total in range(int(degree) + 1)
        for px in range(total + 1) for py in (total - px,)
    )


def _probabilists_hermite(x, degree):
    values = [jnp.ones_like(x)]
    if degree >= 1:
        values.append(x)
    for order in range(1, degree):
        values.append(x * values[-1] - order * values[-2])
    return values


def moment_warp(fields, coords):
    """L2-weighted diagonal Gaussian warp for a batch of fields."""
    weights = jnp.square(jnp.maximum(fields, 0.0))
    mass = jnp.sum(weights, axis=1) + 1e-300
    center = jnp.sum(weights[:, :, None] * coords[None], axis=1) / mass[:, None]
    delta = coords[None] - center[:, None]
    # For u=exp(-r^2/(2s^2)), the u^2-weighted variance is s^2/2.
    variance = jnp.sum(weights[:, :, None] * jnp.square(delta), axis=1) / mass[:, None]
    scale = jnp.sqrt(jnp.maximum(2.0 * variance, 1e-8))
    return center, scale


def basis_from_warp(coords, boundary_mask, center, scale, degree):
    xi = (coords[:, 0] - center[0]) / scale[0]
    eta = (coords[:, 1] - center[1]) / scale[1]
    hx = _probabilists_hermite(xi, degree)
    hy = _probabilists_hermite(eta, degree)
    envelope = jnp.exp(-0.5 * (jnp.square(xi) + jnp.square(eta))) * boundary_mask
    return jnp.stack([
        envelope * hx[px] * hy[py] for px, py in hermite_pairs(degree)
    ], axis=1)


def fit_hermite_states(fields, coords, boundary_mask, degree, batch=32):
    """Moment-align fields, then solve their analytic-basis projection oracle."""
    fields = jnp.asarray(fields, F64)
    coords = jnp.asarray(coords, F64)
    boundary_mask = jnp.asarray(boundary_mask, F64)
    n_coeff = len(hermite_pairs(degree))

    def fit_one(field):
        center, scale = moment_warp(field[None], coords)
        center, scale = center[0], scale[0]
        basis = basis_from_warp(coords, boundary_mask, center, scale, degree)
        norms = jnp.linalg.norm(basis, axis=0) + 1e-300
        normalized = basis / norms[None]
        gram = normalized.T @ normalized
        rhs = normalized.T @ field
        ridge = 1e-12 * jnp.maximum(jnp.trace(gram) / n_coeff, 1e-300)
        scaled = jnp.linalg.solve(gram + ridge * jnp.eye(n_coeff, dtype=F64), rhs)
        coefficients = scaled / norms
        prediction = basis @ coefficients
        state = jnp.concatenate((center, jnp.log(scale), coefficients))
        return state, prediction, jnp.linalg.cond(gram)

    vmapped = jax.jit(jax.vmap(fit_one))
    states, predictions, conditions = [], [], []
    for start in range(0, fields.shape[0], batch):
        state, prediction, condition = vmapped(fields[start:start + batch])
        states.append(np.asarray(state))
        predictions.append(np.asarray(prediction))
        conditions.append(np.asarray(condition))
    return np.concatenate(states), np.concatenate(predictions), np.concatenate(conditions)


def decode_hermite_states(states, coords, boundary_mask, degree, batch=64):
    states = jnp.asarray(states, F64)
    coords = jnp.asarray(coords, F64)
    boundary_mask = jnp.asarray(boundary_mask, F64)

    def decode_one(state):
        center = state[:2]
        scale = jnp.exp(state[2:4])
        coefficients = state[4:]
        return basis_from_warp(
            coords, boundary_mask, center, scale, degree
        ) @ coefficients

    vmapped = jax.jit(jax.vmap(decode_one))
    chunks = []
    for start in range(0, states.shape[0], batch):
        chunks.append(np.asarray(vmapped(states[start:start + batch])))
    return np.concatenate(chunks)


def affine_moment_warp(fields, coords):
    """L2 moment center and lower Cholesky factor for full-covariance alignment."""
    weights = jnp.square(jnp.maximum(fields, 0.0))
    mass = jnp.sum(weights, axis=1) + 1e-300
    center = jnp.sum(weights[:, :, None] * coords[None], axis=1) / mass[:, None]
    delta = coords[None] - center[:, None]
    covariance = jnp.einsum("bn,bni,bnj->bij", weights, delta, delta) / mass[:, None, None]
    covariance = 2.0 * covariance + 1e-8 * jnp.eye(2, dtype=F64)[None]
    return center, jnp.linalg.cholesky(covariance)


def hg5s_basis(coords, boundary_mask, center, cholesky):
    """HG5 plus six smooth nearest-wall chart functions (32-state decoder)."""
    delta = coords - center[None]
    xi = delta[:, 0] / cholesky[0, 0]
    eta = (delta[:, 1] - cholesky[1, 0] * xi) / cholesky[1, 1]
    hx = _probabilists_hermite(xi, 5)
    hy = _probabilists_hermite(eta, 5)
    envelope = jnp.exp(-0.5 * (jnp.square(xi) + jnp.square(eta))) * boundary_mask
    global_columns = [
        envelope * hx[px] * hy[py] for px, py in hermite_pairs(5)
    ]
    distances = jnp.asarray((center[0], 1.0 - center[0], center[1], 1.0 - center[1]))
    chart_weights = jax.nn.softmax(-distances / 0.05)
    wall_scale = jnp.maximum(0.75 * (cholesky[0, 0] + cholesky[1, 1]), 0.02)
    wall_profiles = jnp.stack((
        jnp.exp(-coords[:, 0] / wall_scale),
        jnp.exp(-(1.0 - coords[:, 0]) / wall_scale),
        jnp.exp(-coords[:, 1] / wall_scale),
        jnp.exp(-(1.0 - coords[:, 1]) / wall_scale),
    ), axis=1)
    wall_gate = wall_profiles @ chart_weights
    wall_columns = [
        wall_gate * envelope * hx[px] * hy[py] for px, py in hermite_pairs(2)
    ]
    return jnp.stack(global_columns + wall_columns, axis=1)


def fit_hg5s_states(fields, coords, boundary_mask, batch=32):
    fields = jnp.asarray(fields, F64)
    coords = jnp.asarray(coords, F64)
    boundary_mask = jnp.asarray(boundary_mask, F64)
    n_coeff = len(hermite_pairs(5)) + len(hermite_pairs(2))
    assert 5 + n_coeff == 32

    def fit_one(field):
        center, cholesky = affine_moment_warp(field[None], coords)
        center, cholesky = center[0], cholesky[0]
        basis = hg5s_basis(coords, boundary_mask, center, cholesky)
        norms = jnp.linalg.norm(basis, axis=0) + 1e-300
        normalized = basis / norms[None]
        gram = normalized.T @ normalized
        rhs = normalized.T @ field
        ridge = 1e-12 * jnp.maximum(jnp.trace(gram) / n_coeff, 1e-300)
        scaled = jnp.linalg.solve(gram + ridge * jnp.eye(n_coeff, dtype=F64), rhs)
        coefficients = scaled / norms
        prediction = basis @ coefficients
        state = jnp.concatenate((
            center,
            jnp.asarray((jnp.log(cholesky[0, 0]), cholesky[1, 0],
                         jnp.log(cholesky[1, 1]))),
            coefficients,
        ))
        return state, prediction, jnp.linalg.cond(gram)

    vmapped = jax.jit(jax.vmap(fit_one))
    states, predictions, conditions = [], [], []
    for start in range(0, fields.shape[0], batch):
        state, prediction, condition = vmapped(fields[start:start + batch])
        states.append(np.asarray(state))
        predictions.append(np.asarray(prediction))
        conditions.append(np.asarray(condition))
    return np.concatenate(states), np.concatenate(predictions), np.concatenate(conditions)


def hg5s_cholesky(state):
    return jnp.asarray((
        (jnp.exp(state[2]), 0.0),
        (state[3], jnp.exp(state[4])),
    ), dtype=F64)


def decode_hg5s_states(states, coords, boundary_mask, batch=64):
    states = jnp.asarray(states, F64)
    coords = jnp.asarray(coords, F64)
    boundary_mask = jnp.asarray(boundary_mask, F64)

    def decode_one(state):
        return hg5s_basis(
            coords, boundary_mask, state[:2], hg5s_cholesky(state)
        ) @ state[5:]

    vmapped = jax.jit(jax.vmap(decode_one))
    chunks = []
    for start in range(0, states.shape[0], batch):
        chunks.append(np.asarray(vmapped(states[start:start + batch])))
    return np.concatenate(chunks)


def polynomial_design(features, degree=3):
    """Deterministic complete polynomial features through the named degree."""
    features = np.asarray(features, np.float64)
    columns = [np.ones(features.shape[0], np.float64)]
    for order in range(1, int(degree) + 1):
        for indices in itertools.combinations_with_replacement(
            range(features.shape[1]), order
        ):
            columns.append(np.prod(features[:, indices], axis=1))
    return np.stack(columns, axis=1)


def fit_ridge_predictor(features, targets, ridge=1e-8, degree=3):
    design = polynomial_design(features, degree)
    mean = np.mean(targets, axis=0)
    scale = np.std(targets, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (targets - mean) / scale
    gram = design.T @ design
    amount = float(ridge) * max(float(np.trace(gram) / gram.shape[0]), 1e-300)
    weights = np.linalg.solve(
        gram + amount * np.eye(gram.shape[0]), design.T @ normalized
    )
    return {"weights": weights, "mean": mean, "scale": scale,
            "ridge": float(ridge), "degree": int(degree)}


def apply_ridge_predictor(model, features):
    design = polynomial_design(features, model["degree"])
    return (design @ model["weights"]) * model["scale"] + model["mean"]


def trajectory_features(parameters, n):
    """Known/deployably recovered IC parameters, time, and target mesh spacing."""
    count = parameters["cx"].shape[0]
    tau = np.linspace(0.0, 1.0, NUM_STEPS + 1, dtype=np.float64)
    base = np.column_stack((
        (parameters["cx"] - 0.5) / 0.35,
        (parameters["cy"] - 0.5) / 0.35,
        (parameters["width"] - 0.125) / 0.075,
        (parameters["amplitude"] - 1.25) / 0.75,
        np.log(parameters["nu"] / np.sqrt(0.001)) / (0.5 * np.log(10.0)),
    ))
    tiled = np.repeat(base[:, None, :], NUM_STEPS + 1, axis=1)
    times = np.broadcast_to(tau[None, :, None], (count, NUM_STEPS + 1, 1))
    spacing = np.full((count, NUM_STEPS + 1, 1), 1.0 / (n - 1), np.float64)
    return np.concatenate((tiled, times, spacing), axis=2)


def recover_blob_parameters(u0, n):
    """Recover the exact Gaussian IC from a fixed grid by a log-quadratic fit."""
    coords = grid_coords(n)
    values = np.asarray(u0, np.float64).reshape(-1)
    interior = binary_boundary_mask(n) > 0
    threshold = max(float(np.max(values)) * 1e-12, 1e-300)
    keep = interior & (values > threshold)
    x, y = coords[keep, 0], coords[keep, 1]
    design = np.column_stack((np.ones(x.size), x, y, x * x + y * y))
    coefficients, *_ = np.linalg.lstsq(design, np.log(values[keep]), rcond=None)
    constant, bx, by, quadratic = coefficients
    width = np.sqrt(-1.0 / (2.0 * quadratic))
    cx = -bx / (2.0 * quadratic)
    cy = -by / (2.0 * quadratic)
    log_amplitude = constant + (cx * cx + cy * cy) / (2.0 * width * width)
    return np.asarray((cx, cy, width, np.exp(log_amplitude)), np.float64)


def error_metrics(prediction, truth):
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    snapshot = np.linalg.norm(prediction - truth, axis=-1) / np.maximum(
        np.linalg.norm(truth, axis=-1), 1e-300
    )
    trajectory = np.linalg.norm(
        prediction.reshape(prediction.shape[0], -1)
        - truth.reshape(truth.shape[0], -1), axis=1
    ) / np.maximum(np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1), 1e-300)
    return {
        "trajectory_mean": float(np.mean(trajectory)),
        "trajectory_median": float(np.median(trajectory)),
        "trajectory_worst": float(np.max(trajectory)),
        "trajectory_all": trajectory.tolist(),
        "snapshot_mean": float(np.mean(snapshot)),
        "snapshot_median": float(np.median(snapshot)),
        "snapshot_worst": float(np.max(snapshot)),
        "per_time_mean": np.mean(snapshot, axis=0).tolist(),
    }


def gpu_burn(seconds=1.0):
    matrix = jnp.ones((1536, 1536), F64)
    kernel = jax.jit(lambda value: value @ value)
    kernel(matrix).block_until_ready()
    start = time.perf_counter()
    count = 0
    while time.perf_counter() - start < seconds:
        kernel(matrix).block_until_ready()
        count += 1
    return count
