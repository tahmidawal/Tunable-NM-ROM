"""Train/validation gate for a shift-aware correction surrogate.

The unaligned correction POD failed its oracle gate even at rank 32.  This
script tests the specific hypothesis that translation and diffusion-induced
width change caused that failure.  Corrections are pulled into a canonical
frame using moments of the *currently available field*, compressed there, and
then pushed back to physical coordinates before grading.  Generator metadata
is retained only as a separately labelled extra-information control.

This is not called an NM-ROM: coefficient prediction is supervised and no
weak reduced PDE solve is performed here.  A successful gate only licenses a
later weak-form correction-manifold experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import jax
import numpy as np
from scipy.ndimage import affine_transform

import bh_common as bc
import bh_correction_spectrum as cg

OUT = sys.argv[1]
MODEL_OUT = os.environ.get("MODEL_OUT", os.path.splitext(OUT)[0] + ".npz")
N = int(os.environ.get("N", "64"))
N_TRAIN = int(os.environ.get("CORR_N_TRAIN", "128"))
N_VAL = int(os.environ.get("CORR_N_VAL", "32"))
RANKS = [int(v) for v in os.environ.get("RANKS", "4,8,16,32").split(",")]
K_MAX = max(RANKS)
SEED = int(os.environ.get("CORR_SEED", str(bc.bf.SEED)))
VAL_START = int(os.environ.get("VAL_START", str(bc.bf.N_TRAIN)))
ALIGNMENTS = os.environ.get("ALIGNMENTS", "translate,translate_scale").split(",")
RIDGES = [float(v) for v in os.environ.get("RIDGES", "1e-8,1e-6,1e-4,1e-2").split(",")]
REFERENCE_WIDTH = float(os.environ.get("REFERENCE_WIDTH", "0.13"))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def current_fields(U):
    """Fields available when each of the 50 corrections is constructed."""
    return U[:, :-1]


def field_moments(fields):
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    flat_x, flat_y = X.reshape(-1), Y.reshape(-1)
    flat = fields.reshape(-1, N * N)
    positive = np.maximum(flat, 0.0)
    mass = np.sum(positive, axis=1) + 1e-300
    cx = np.sum(positive * flat_x[None], axis=1) / mass
    cy = np.sum(positive * flat_y[None], axis=1) / mass
    variance = np.sum(
        positive
        * ((flat_x[None] - cx[:, None]) ** 2
           + (flat_y[None] - cy[:, None]) ** 2),
        axis=1,
    ) / mass
    width = np.sqrt(np.maximum(variance / 2.0, 1e-12))
    return np.stack([cx, cy, width], axis=1).reshape(fields.shape[0], fields.shape[1], 3)


def transforms(moments, alignment):
    """Return canonical-pull and physical-push affine maps in index units."""
    center = moments[..., :2] * (N - 1)
    reference_center = np.asarray([(N - 1) / 2.0, (N - 1) / 2.0])
    if alignment == "translate":
        scale_pull = np.ones(moments.shape[:-1])
    elif alignment == "translate_scale":
        # canonical output samples physical input at c + s/ref_s*(j-ref_c)
        scale_pull = moments[..., 2] / REFERENCE_WIDTH
    else:
        raise ValueError(f"unknown alignment {alignment}")
    pull_offset = center - scale_pull[..., None] * reference_center
    scale_push = 1.0 / scale_pull
    push_offset = reference_center - scale_push[..., None] * center
    return scale_pull, pull_offset, scale_push, push_offset


def warp_batch(fields, scale, offset):
    # ``corrections`` can return a non-C-contiguous view.  Allocating with
    # ``empty_like(order='K')`` and assigning through ``reshape`` would then
    # assign into a temporary copy and leave the result uninitialised.
    flat_fields = np.asarray(fields).reshape(-1, N, N)
    flat_result = np.empty(flat_fields.shape, dtype=fields.dtype, order="C")
    flat_scale = scale.reshape(-1)
    flat_offset = offset.reshape(-1, 2)
    for index, field in enumerate(flat_fields):
        flat_result[index] = affine_transform(
            field,
            np.eye(2) * flat_scale[index],
            offset=flat_offset[index],
            output_shape=(N, N),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return flat_result.reshape(fields.shape)


def quadratic_features(features):
    """Small deterministic feature map suitable for 128 training cases."""
    columns = [np.ones(features.shape[0])]
    columns.extend(features[:, i] for i in range(features.shape[1]))
    for i in range(features.shape[1]):
        for j in range(i, features.shape[1]):
            columns.append(features[:, i] * features[:, j])
    return np.stack(columns, axis=1)


def fit_time_sliced(train_features, coefficients, ridge):
    """One smooth low-capacity ridge map per named time step."""
    design = quadratic_features(train_features)
    weights = []
    for step in range(bc.T):
        target = coefficients[:, step]
        gram = design.T @ design
        scale = max(float(np.trace(gram) / gram.shape[0]), 1e-300)
        weights.append(np.linalg.solve(
            gram + ridge * scale * np.eye(gram.shape[0]), design.T @ target
        ))
    return np.stack(weights)


def predict_time_sliced(features, weights):
    design = quadratic_features(features)
    return np.einsum("vf,tfr->vtr", design, weights)


def original_frame_metrics(target, predicted):
    return {
        "remaining_ratio_global": float(
            np.linalg.norm(predicted - target) / np.linalg.norm(target)
        ),
        **cg.alignment_metrics(target, predicted),
    }


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing shifted correction gate")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    report = {
        "config": {
            "purpose": "train/validation-only shifted correction surrogate gate",
            "classification": "supervised surrogate; not an NM-ROM",
            "N": N,
            "n_train": N_TRAIN,
            "n_validation": N_VAL,
            "train_indices": [0, N_TRAIN - 1],
            "validation_indices": [VAL_START, VAL_START + N_VAL - 1],
            "seed": SEED,
            "canonical_draw_count": bc.bf.N_TRAIN + bc.bf.N_VAL,
            "test_seed_touched": False,
            "alignments": ALIGNMENTS,
            "reference_width": REFERENCE_WIDTH,
            "ranks": RANKS,
            "ridges": RIDGES,
            "oracle_target": "original-frame remaining correction ratio <= 0.25",
            "interpolation": "scipy.ndimage affine_transform order=1 constant-zero",
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "selection": {},
        "reference_health": {},
        "complete": False,
    }
    save(report)

    draw_count = bc.bf.N_TRAIN + bc.bf.N_VAL
    parameters = bc.bf.sample_params(seed=SEED, m=draw_count)
    train_indices = np.arange(N_TRAIN)
    val_indices = np.arange(VAL_START, VAL_START + N_VAL)
    start = time.time()
    U_train, train_ref = cg.batch_trajectories(train_indices, parameters)
    U_val, val_ref = cg.batch_trajectories(val_indices, parameters)
    E_train = cg.corrections(U_train).reshape(N_TRAIN, bc.T, N, N)
    E_val = cg.corrections(U_val).reshape(N_VAL, bc.T, N, N)
    report["reference_health"] = {
        "train_max_residual": train_ref,
        "validation_max_residual": val_ref,
    }

    train_moments = field_moments(current_fields(U_train))
    val_moments = field_moments(current_fields(U_val))
    cx, cy, width, amplitude, nu, z = parameters
    feature_sets = {
        "parameter_extra_info": (
            np.asarray(z[train_indices], np.float64),
            np.asarray(z[val_indices], np.float64),
        ),
        "field_moments_deployable": (
            cg.field_features(U_train[:, 0], nu[train_indices]),
            cg.field_features(U_val[:, 0], nu[val_indices]),
        ),
    }
    model_arrays = {}
    best_overall = None
    for alignment in ALIGNMENTS:
        tr_pull_s, tr_pull_o, _, _ = transforms(train_moments, alignment)
        va_pull_s, va_pull_o, va_push_s, va_push_o = transforms(val_moments, alignment)
        E_train_aligned = warp_batch(E_train, tr_pull_s, tr_pull_o)
        E_val_aligned = warp_batch(E_val, va_pull_s, va_pull_o)
        matrix = E_train_aligned.reshape(-1, N * N)
        basis, singular_values, orthogonality = cg.randomized_basis(
            matrix, K_MAX, 16, 2, SEED + 20260819
        )
        model_arrays[f"basis_{alignment}"] = basis
        model_arrays[f"singular_values_{alignment}"] = singular_values
        for rank in RANKS:
            V = basis[:, :rank]
            projected_aligned = (E_val_aligned.reshape(-1, N * N) @ V @ V.T)
            projected_aligned = projected_aligned.reshape(E_val.shape)
            projected_physical = warp_batch(projected_aligned, va_push_s, va_push_o)
            projection_metrics = original_frame_metrics(E_val, projected_physical)
            projection_row = {
                "alignment": alignment,
                "model": "projection_oracle",
                "feature_kind": "nondeployable_exact_coefficients",
                "rank": rank,
                "ridge": None,
                "basis_orthogonality_max": orthogonality,
                "aligned_train_captured_energy": float(
                    np.sum(singular_values[:rank] ** 2) / np.sum(matrix * matrix)
                ),
                **projection_metrics,
            }
            report["rows"].append(projection_row)
            bc.log(alignment, "rank", rank, "projection", projection_metrics)

            coeff_train = (E_train_aligned.reshape(-1, N * N) @ V).reshape(
                N_TRAIN, bc.T, rank
            )
            for feature_kind, (features_train, features_val) in feature_sets.items():
                for ridge in RIDGES:
                    weights = fit_time_sliced(features_train, coeff_train, ridge)
                    coeff_val = predict_time_sliced(features_val, weights)
                    predicted_aligned = (coeff_val.reshape(-1, rank) @ V.T).reshape(
                        E_val.shape
                    )
                    predicted_physical = warp_batch(
                        predicted_aligned, va_push_s, va_push_o
                    )
                    metrics = original_frame_metrics(E_val, predicted_physical)
                    row = {
                        "alignment": alignment,
                        "model": "time_sliced_quadratic_ridge",
                        "feature_kind": feature_kind,
                        "rank": rank,
                        "ridge": ridge,
                        **metrics,
                    }
                    report["rows"].append(row)
                    key = f"{alignment}:{feature_kind}"
                    previous = report["selection"].get(key)
                    if previous is None or row["remaining_ratio_global"] < previous[
                        "remaining_ratio_global"
                    ]:
                        report["selection"][key] = row
                        model_arrays[f"weights_{alignment}_{feature_kind}"] = weights
                    if best_overall is None or row["remaining_ratio_global"] < best_overall[
                        "remaining_ratio_global"
                    ]:
                        best_overall = row

    report["selection"]["best_supervised_surrogate"] = best_overall
    report["elapsed_seconds"] = float(time.time() - start)
    report["complete"] = True
    np.savez_compressed(MODEL_OUT, **model_arrays)
    with open(MODEL_OUT, "rb") as handle:
        report["model_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    save(report)
    bc.log("BEST", best_overall)


if __name__ == "__main__":
    main()
