"""D0: decompose the inherited floor and falsify transported HG representation.

Usage: b10_d0.py OUTPUT_JSON OUTPUT_NPZ INHERITED_JSON
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import b10_common as c

OUTPUT_JSON, OUTPUT_NPZ, INHERITED_JSON = sys.argv[1:4]
N = int(os.environ.get("N", "64"))
DATA_SEED = int(os.environ.get("DATA_SEED", "0"))
DRAW_COUNT = int(os.environ.get("DRAW_COUNT", "704"))
TRAIN_COUNT = int(os.environ.get("D0_TRAIN_COUNT", "512"))
SELECTION_START = int(os.environ.get("SELECTION_START", "512"))
SELECTION_COUNT = int(os.environ.get("SELECTION_COUNT", "64"))
DEGREES = tuple(int(value) for value in os.environ.get("HG_DEGREES", "4,5").split(","))


def inherited_metrics(path):
    with open(path) as handle:
        report = json.load(handle)
    return {
        "artifact": os.path.basename(path),
        "sha256": c.sha256(path),
        "ic_fit_mean": report["ic_fit"]["rel_mean"],
        "ic_fit_worst": report["ic_fit"]["rel_max"],
        "oracle_inferred_latent_trajectory_mean": report[
            "oracle_inferred_latent_test"]["traj_rel_mean"],
        "oracle_inferred_latent_per_time_mean": report[
            "oracle_inferred_latent_test"]["per_time_mean"],
        "full_weak_trajectory_mean": report["rom"]["lspg:full:weak64"]["traj_rel_mean"],
        "full_weak_trajectory_worst": report["rom"]["lspg:full:weak64"]["traj_rel_max"],
        "eq256_trajectory_mean": report["rom"]["lspg:eq256:weak64"]["traj_rel_mean"],
        "eq512_trajectory_mean": report["rom"]["lspg:eq512:weak64"]["traj_rel_mean"],
    }


def wall_strata(metrics, parameters):
    distance = np.minimum.reduce((
        parameters["cx"], parameters["cy"],
        1.0 - parameters["cx"], 1.0 - parameters["cy"],
    ))
    errors = np.asarray(metrics["trajectory_all"])
    order = np.argsort(distance)
    groups = np.array_split(order, 4)
    return [{
        "quartile": index,
        "distance_min": float(np.min(distance[group])),
        "distance_max": float(np.max(distance[group])),
        "count": int(group.size),
        "mean_trajectory_error": float(np.mean(errors[group])),
        "squared_error_fraction": float(
            np.sum(np.square(errors[group])) / np.sum(np.square(errors))
        ),
    } for index, group in enumerate(groups) if group.size]


def main():
    c.require_gpu_highest()
    if N != 64:
        raise SystemExit("D0 is preregistered at N=64")
    started = time.time()
    provenance = c.provenance()
    report = {
        "stage": "D0 inherited-floor and transported-HG decomposition",
        "status": "running",
        "config": {
            "N": N,
            "dt": c.DT,
            "num_steps": c.NUM_STEPS,
            "data_seed": DATA_SEED,
            "draw_count": DRAW_COUNT,
            "train_indices": [0, TRAIN_COUNT - 1],
            "selection_indices": [SELECTION_START, SELECTION_START + SELECTION_COUNT - 1],
            "model_validation_touched": False,
            "confirmation_touched": False,
            "degrees": list(DEGREES),
            "predictor": "degree-3 complete polynomial ridge 1e-8",
            "boundary": "exact binary FOM grid mask",
            "warp": "L2-moment diagonal Gaussian",
            "f64": True,
        },
        "provenance": provenance,
        "inherited": inherited_metrics(INHERITED_JSON),
        "reference_health": {},
        "concepts": {},
    }
    c.save_json(OUTPUT_JSON, report)
    c.log("D0", report["config"], provenance)

    train_indices = np.arange(TRAIN_COUNT)
    selection_indices = np.arange(SELECTION_START, SELECTION_START + SELECTION_COUNT)
    train_fields, train_parameters, train_health = c.generate_population(
        N, DATA_SEED, DRAW_COUNT, train_indices
    )
    selection_fields, selection_parameters, selection_health = c.generate_population(
        N, DATA_SEED, DRAW_COUNT, selection_indices
    )
    report["reference_health"] = {"train": train_health, "selection": selection_health}
    reference_gate = max(
        train_health["independent_max_relative_residual"],
        selection_health["independent_max_relative_residual"],
    )
    if reference_gate > 1e-8:
        raise SystemExit(f"reference residual gate failed: {reference_gate:.3e}")

    recovered = np.stack([
        c.recover_blob_parameters(field[0], N) for field in selection_fields
    ])
    expected = np.column_stack((
        selection_parameters["cx"], selection_parameters["cy"],
        selection_parameters["width"], selection_parameters["amplitude"],
    ))
    relative = np.abs(recovered - expected) / np.maximum(np.abs(expected), 1e-300)
    report["deployable_initial_parameter_recovery"] = {
        "sample_resolution": N,
        "method": "log-quadratic least squares on positive interior nodes",
        "mean_relative": float(np.mean(relative)),
        "max_relative": float(np.max(relative)),
        "all_finite": bool(np.all(np.isfinite(recovered))),
    }

    coords = c.grid_coords(N)
    mask = c.binary_boundary_mask(N)
    train_flat = train_fields.reshape(-1, N * N)
    selection_flat = selection_fields.reshape(-1, N * N)
    train_features = c.trajectory_features(train_parameters, N).reshape(-1, 7)
    selection_features = c.trajectory_features(selection_parameters, N).reshape(-1, 7)
    arrays = {
        "train_features": train_features,
        "selection_features": selection_features,
        "selection_parameters": expected,
        "selection_nu": selection_parameters["nu"],
    }

    for degree in DEGREES:
        c.log("fit HG", degree, "train", train_flat.shape, "selection", selection_flat.shape)
        train_states, _, train_condition = c.fit_hermite_states(
            train_flat, coords, mask, degree
        )
        selection_states, selection_oracle, selection_condition = c.fit_hermite_states(
            selection_flat, coords, mask, degree
        )
        oracle_fields = selection_oracle.reshape(selection_fields.shape)
        oracle_metrics = c.error_metrics(oracle_fields, selection_fields)

        predictor = c.fit_ridge_predictor(train_features, train_states, ridge=1e-8, degree=3)
        predicted_states = c.apply_ridge_predictor(predictor, selection_features)
        predicted_flat = c.decode_hermite_states(
            predicted_states, coords, mask, degree
        )
        predicted_fields = predicted_flat.reshape(selection_fields.shape)
        predicted_metrics = c.error_metrics(predicted_fields, selection_fields)
        state_error = np.linalg.norm(predicted_states - selection_states, axis=1) / np.maximum(
            np.linalg.norm(selection_states, axis=1), 1e-300
        )
        wall = wall_strata(predicted_metrics, selection_parameters)
        nearest_wall_fraction = wall[0]["squared_error_fraction"]
        concept = {
            "degree": degree,
            "latent_dimension": int(selection_states.shape[1]),
            "coefficient_count": len(c.hermite_pairs(degree)),
            "representation_oracle": oracle_metrics,
            "predictor_decoder": predicted_metrics,
            "predictor_state_relative_mean": float(np.mean(state_error)),
            "predictor_state_relative_worst": float(np.max(state_error)),
            "oracle_to_predictor_trajectory_mean_ratio": float(
                predicted_metrics["trajectory_mean"]
                / max(oracle_metrics["trajectory_mean"], 1e-300)
            ),
            "basis_gram_condition": {
                "train_median": float(np.median(train_condition)),
                "train_worst": float(np.max(train_condition)),
                "selection_median": float(np.median(selection_condition)),
                "selection_worst": float(np.max(selection_condition)),
            },
            "wall_distance_strata": wall,
            "nearest_wall_squared_error_fraction": nearest_wall_fraction,
            "representation_gate_pass": bool(
                oracle_metrics["trajectory_mean"] <= 2e-4
                and oracle_metrics["trajectory_worst"] <= 7e-4
            ),
            "hg5_licensed_by_hg4": None,
            "wall_chart_licensed": bool(nearest_wall_fraction >= 0.5),
        }
        report["concepts"][f"HG{degree}"] = concept
        arrays[f"train_states_hg{degree}"] = train_states
        arrays[f"selection_states_hg{degree}"] = selection_states
        arrays[f"selection_predicted_states_hg{degree}"] = predicted_states
        c.log("HG", degree, "oracle", oracle_metrics["trajectory_mean"],
              oracle_metrics["trajectory_worst"], "predictor",
              predicted_metrics["trajectory_mean"], predicted_metrics["trajectory_worst"])

    if "HG4" in report["concepts"]:
        hg4 = report["concepts"]["HG4"]
        license_hg5 = (
            hg4["representation_gate_pass"]
            and hg4["predictor_decoder"]["trajectory_mean"] > 3e-4
        )
        hg4["hg5_licensed_by_hg4"] = bool(license_hg5)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **arrays)
    report["npz"] = {"path": os.path.basename(OUTPUT_NPZ), "sha256": c.sha256(OUTPUT_NPZ)}
    report["elapsed_seconds"] = float(time.time() - started)
    report["status"] = "complete"
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps(report, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
