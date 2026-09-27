"""Train/validation-only feasibility gate for a cached correction ROM.

The target is the part linear extrapolation misses:

    e_0 = u_1 - u_0,
    e_t = u_{t+1} - (2 u_t - u_{t-1}), t >= 1.

No final seed-1 test trajectory is touched. The gate first measures the
correction snapshot singular spectrum, then predicts reduced coefficients from
either generator metadata (extra-information control) or moments derived from
the actually available initial field plus known viscosity (deployable arm).
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc

OUT = sys.argv[1]
MODEL_OUT = os.environ.get("MODEL_OUT", os.path.splitext(OUT)[0] + ".npz")
N = int(os.environ.get("N", "64"))
N_TRAIN = int(os.environ.get("CORR_N_TRAIN", "128"))
N_VAL = int(os.environ.get("CORR_N_VAL", "32"))
K_MAX = int(os.environ.get("K_MAX", "32"))
RANKS = [int(value) for value in os.environ.get("RANKS", "4,8,16,32").split(",")]
RFF_DIM = int(os.environ.get("RFF_DIM", "256"))
BANDWIDTHS = [float(value) for value in os.environ.get(
    "BANDWIDTHS", "0.5,1,2,4"
).split(",")]
RIDGES = [float(value) for value in os.environ.get(
    "RIDGES", "1e-8,1e-6,1e-4"
).split(",")]
SEED = int(os.environ.get("CORR_SEED", str(bc.bf.SEED)))
VAL_START = int(os.environ.get("VAL_START", str(bc.bf.N_TRAIN)))
SVD_OVERSAMPLE = int(os.environ.get("SVD_OVERSAMPLE", "16"))
SVD_POWER_ITERS = int(os.environ.get("SVD_POWER_ITERS", "2"))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def batch_trajectories(indices, parameters):
    cx, cy, width, amplitude, nu, z = parameters
    U0 = np.stack([
        bc.bf.blob_ic(N, cx[index], cy[index], width[index], amplitude[index])
        for index in indices
    ])
    rollout, _ = bc.bf.make_rollout(N)
    snaps, rel_res = rollout(jnp.asarray(U0), jnp.asarray(nu[indices]))
    U = np.asarray(snaps).transpose(1, 0, 2)
    worst = float(jnp.max(rel_res))
    if not np.isfinite(worst) or worst > 1e-8:
        raise SystemExit(f"reference trajectory residual {worst:.3e} exceeds gate")
    return U, worst


def corrections(U):
    extrap = np.empty_like(U[:, 1:])
    extrap[:, 0] = U[:, 0]
    extrap[:, 1:] = 2.0 * U[:, 1:-1] - U[:, :-2]
    return U[:, 1:] - extrap


def randomized_basis(matrix, rank, oversample, power_iters, seed):
    rng = np.random.default_rng(seed)
    width = min(rank + oversample, min(matrix.shape))
    omega = rng.standard_normal((matrix.shape[1], width))
    Y = matrix @ omega
    for _ in range(power_iters):
        Q, _ = np.linalg.qr(Y, mode="reduced")
        Y = matrix @ (matrix.T @ Q)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    small = Q.T @ matrix
    _, singular_values, right = np.linalg.svd(small, full_matrices=False)
    basis = right[:rank].T
    orthogonality = float(np.max(np.abs(basis.T @ basis - np.eye(rank))))
    return basis, singular_values[:rank], orthogonality


def field_features(U0, nu):
    """Recover blob summaries from u0; no generator metadata is used."""
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    flat_x, flat_y = X.reshape(-1), Y.reshape(-1)
    features = []
    for field, viscosity in zip(U0, nu):
        positive = np.maximum(field, 0.0)
        mass = float(np.sum(positive)) + 1e-300
        cx = float(np.sum(positive * flat_x) / mass)
        cy = float(np.sum(positive * flat_y) / mass)
        radial_variance = float(np.sum(
            positive * ((flat_x - cx) ** 2 + (flat_y - cy) ** 2)
        ) / mass)
        width = np.sqrt(max(radial_variance / 2.0, 1e-12))
        amplitude = float(np.max(positive))
        lognu = np.log(viscosity)
        features.append([
            (cx - 0.5) / 0.35,
            (cy - 0.5) / 0.35,
            (width - 0.125) / 0.075,
            (amplitude - 1.25) / 0.75,
            (lognu - np.log(np.sqrt(0.001))) / (0.5 * np.log(10.0)),
        ])
    return np.asarray(features, dtype=np.float64)


def space_time_features(parameter_features):
    times = 2.0 * (np.arange(1, bc.T + 1) / bc.T) - 1.0
    repeated = np.repeat(parameter_features, bc.T, axis=0)
    tiled_time = np.tile(times, parameter_features.shape[0])[:, None]
    return np.concatenate([repeated, tiled_time], axis=1)


def rff_features(inputs, bandwidth, base_frequency, phase):
    projection = inputs @ (bandwidth * base_frequency) + phase
    return np.concatenate([
        np.ones((inputs.shape[0], 1)),
        np.sqrt(2.0 / base_frequency.shape[1]) * np.cos(projection),
    ], axis=1)


def fit_ridge(features, targets, ridge):
    gram = features.T @ features
    scale = np.trace(gram) / gram.shape[0]
    regularizer = ridge * max(float(scale), 1e-300)
    return np.linalg.solve(
        gram + regularizer * np.eye(gram.shape[0]), features.T @ targets
    )


def relative_field_error(target, coefficients, basis):
    prediction = coefficients @ basis.T
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def alignment_metrics(target, prediction):
    target_flat = target.reshape(target.shape[0], -1)
    pred_flat = prediction.reshape(prediction.shape[0], -1)
    target_norm2 = np.sum(target_flat * target_flat, axis=1)
    dot = np.sum(target_flat * pred_flat, axis=1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.sqrt(target_norm2)
    cosine = dot / np.maximum(target_norm * pred_norm, 1e-300)
    q = dot / np.maximum(target_norm2, 1e-300)
    remaining = np.linalg.norm(target_flat - pred_flat, axis=1) / np.maximum(
        target_norm, 1e-300
    )
    return {
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.1)),
        "oracle_direction_q_median": float(np.median(q)),
        "remaining_correction_ratio_median": float(np.median(remaining)),
        "remaining_correction_ratio_p90": float(np.quantile(remaining, 0.9)),
        "fraction_cases_q75": float(np.mean(remaining <= 0.25)),
    }


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing correction gate")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    if max(RANKS) > K_MAX:
        raise SystemExit("every requested rank must be <= K_MAX")
    report = {
        "config": {
            "pde": "burgers2d",
            "purpose": "train/validation-only correction-ROM feasibility gate",
            "N": N,
            "n_train_trajectories": N_TRAIN,
            "n_validation_trajectories": N_VAL,
            "train_indices": [0, N_TRAIN - 1],
            "validation_indices": [VAL_START, VAL_START + N_VAL - 1],
            "seed": SEED,
            "canonical_draw_count": bc.bf.N_TRAIN + bc.bf.N_VAL,
            "test_seed_touched": False,
            "ranks": RANKS,
            "rff_dim": RFF_DIM,
            "bandwidths": BANDWIDTHS,
            "ridges": RIDGES,
            "selection_metric": "validation relative correction-field L2",
            "oracle_gate_target": "remaining correction ratio <= 0.25 (q75)",
            "f64": True,
        },
        "provenance": provenance,
        "spectrum": {},
        "model_rows": [],
        "selection": {},
        "complete": False,
    }
    save(report)

    draw_count = bc.bf.N_TRAIN + bc.bf.N_VAL
    parameters = bc.bf.sample_params(seed=SEED, m=draw_count)
    train_indices = np.arange(N_TRAIN)
    val_indices = np.arange(VAL_START, VAL_START + N_VAL)
    start = time.time()
    U_train, train_residual = batch_trajectories(train_indices, parameters)
    U_val, val_residual = batch_trajectories(val_indices, parameters)
    E_train_3d = corrections(U_train)
    E_val_3d = corrections(U_val)
    E_train = E_train_3d.reshape(-1, N * N)
    E_val = E_val_3d.reshape(-1, N * N)
    total_energy = float(np.sum(E_train * E_train))
    basis, singular_values, orthogonality = randomized_basis(
        E_train, K_MAX, SVD_OVERSAMPLE, SVD_POWER_ITERS, SEED + 20260819
    )
    report["spectrum"] = {
        "singular_values_top": singular_values.tolist(),
        "total_snapshot_energy": total_energy,
        "basis_orthogonality_max": orthogonality,
        "train_reference_max_residual": train_residual,
        "validation_reference_max_residual": val_residual,
        "rank_metrics": {},
    }
    for rank in RANKS:
        V = basis[:, :rank]
        train_remaining = relative_field_error(E_train, E_train @ V, V)
        val_remaining = relative_field_error(E_val, E_val @ V, V)
        report["spectrum"]["rank_metrics"][str(rank)] = {
            "captured_train_energy": float(
                np.sum(singular_values[:rank] ** 2) / total_energy
            ),
            "train_projection_remaining_ratio": train_remaining,
            "validation_projection_remaining_ratio": val_remaining,
        }
        bc.log(
            f"rank {rank}: train remaining={train_remaining:.3e} "
            f"val remaining={val_remaining:.3e}"
        )

    cx, cy, width, amplitude, nu, z = parameters
    train_true_parameters = np.asarray(z[train_indices], dtype=np.float64)
    val_true_parameters = np.asarray(z[val_indices], dtype=np.float64)
    train_field_parameters = field_features(U_train[:, 0], nu[train_indices])
    val_field_parameters = field_features(U_val[:, 0], nu[val_indices])
    feature_sets = {
        "parameter_extra_info": (
            space_time_features(train_true_parameters),
            space_time_features(val_true_parameters),
        ),
        "field_moments_deployable": (
            space_time_features(train_field_parameters),
            space_time_features(val_field_parameters),
        ),
    }
    rng = np.random.default_rng(SEED + 314159)
    base_frequency = rng.standard_normal((6, RFF_DIM))
    phase = rng.uniform(0.0, 2.0 * np.pi, RFF_DIM)
    best_models = {}
    for feature_kind, (X_train_raw, X_val_raw) in feature_sets.items():
        best = None
        for rank in RANKS:
            V = basis[:, :rank]
            coefficients_train = E_train @ V
            for bandwidth in BANDWIDTHS:
                X_train = rff_features(
                    X_train_raw, bandwidth, base_frequency, phase
                )
                X_val = rff_features(X_val_raw, bandwidth, base_frequency, phase)
                for ridge in RIDGES:
                    weights = fit_ridge(X_train, coefficients_train, ridge)
                    predicted_coefficients = X_val @ weights
                    val_remaining = relative_field_error(
                        E_val, predicted_coefficients, V
                    )
                    predicted = (predicted_coefficients @ V.T).reshape(E_val_3d.shape)
                    metrics = alignment_metrics(E_val_3d, predicted)
                    row = {
                        "feature_kind": feature_kind,
                        "rank": rank,
                        "bandwidth": bandwidth,
                        "ridge": ridge,
                        "validation_remaining_correction_ratio_global": val_remaining,
                        **metrics,
                    }
                    report["model_rows"].append(row)
                    if best is None or val_remaining < best[0]:
                        best = (val_remaining, row, weights)
        best_models[feature_kind] = best
        report["selection"][feature_kind] = best[1]
        bc.log(f"best {feature_kind}: {best[1]}")

    # One shared spatial basis plus one selected coefficient model per input
    # contract. This is an offline checkpoint; final test is a separate job.
    arrays = {
        "basis_N64": basis,
        "singular_values": singular_values,
        "rff_base_frequency": base_frequency,
        "rff_phase": phase,
        "N": np.asarray(N),
        "num_steps": np.asarray(bc.T),
    }
    for feature_kind, (_, row, weights) in best_models.items():
        prefix = feature_kind
        arrays[f"{prefix}_weights"] = weights
        arrays[f"{prefix}_rank"] = np.asarray(row["rank"])
        arrays[f"{prefix}_bandwidth"] = np.asarray(row["bandwidth"])
        arrays[f"{prefix}_ridge"] = np.asarray(row["ridge"])
    np.savez_compressed(MODEL_OUT, **arrays)
    report["model_file"] = os.path.basename(MODEL_OUT)
    with open(MODEL_OUT, "rb") as model_handle:
        report["model_file_sha256"] = hashlib.sha256(model_handle.read()).hexdigest()
    report["elapsed_s"] = time.time() - start
    report["complete"] = True
    save(report)
    bc.log(f"DONE in {report['elapsed_s']:.1f}s model={MODEL_OUT}")


if __name__ == "__main__":
    main()
