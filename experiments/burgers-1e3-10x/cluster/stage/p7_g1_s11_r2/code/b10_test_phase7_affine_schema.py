#!/usr/bin/env python
"""Read-only regression for immutable P5 physical-affine Phase7 mapping."""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

import b10_common as c
import b10_spline as spline
import b10_phase7_train as trainer


EXPECTED_P5 = "97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96"
EXPECTED_P5_NPZ = "5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54"


def require(value, message):
    if not value:
        raise SystemExit(f"PHASE7 AFFINE REGRESSION FAILED: {message}")


def independent_normalize(physical):
    value = np.asarray(physical, np.float64)
    l11, l22 = np.exp(value[:, 2]), np.exp(value[:, 4])
    ratio = np.log(1.0 / 0.015)
    return np.column_stack((
        (value[:, 0] - 0.5) / 0.75,
        (value[:, 1] - 0.5) / 0.75,
        2.0 * (np.log(l11) - np.log(0.015)) / ratio - 1.0,
        value[:, 3] / (2.0 * np.sqrt(l11 * l22)),
        2.0 * (np.log(l22) - np.log(0.015)) / ratio - 1.0,
    ))


def build(p5_json, p5_npz, target_dir):
    require(c.sha256(p5_json) == EXPECTED_P5, "immutable P5 JSON hash")
    require(c.sha256(p5_npz) == EXPECTED_P5_NPZ, "immutable P5 NPZ hash")
    with open(p5_json, encoding="utf-8") as handle:
        report = json.load(handle)
    sampled_values = c.bf.sample_params(seed=0, m=704)
    sampled = dict(zip(
        ("cx", "cy", "width", "amplitude", "nu", "normalized"),
        (np.asarray(value, np.float64) for value in sampled_values),
    ))
    chunk_records, next_global, snapshot_count = [], 0, 0
    physical_min = np.full(5, np.inf)
    physical_max = np.full(5, -np.inf)
    normalized_min = np.full(5, np.inf)
    normalized_max = np.full(5, -np.inf)
    max_mapping_delta = 0.0
    first_locked_normalized = None
    all_physical, all_normalized = [], []
    for row in report["train_targets"]["chunks"]:
        path = os.path.join(target_dir, row["basename"])
        require(c.sha256(path) == row["sha256"], f"chunk hash {row['basename']}")
        with np.load(path, allow_pickle=False) as values:
            physical = np.asarray(values["affine"], np.float64).reshape(-1, 5)
            locked = np.stack([
                spline.normalized_state_from_affine(item) for item in physical
            ])
            if first_locked_normalized is None:
                first_locked_normalized = locked[:51].copy()
            all_physical.append(physical)
            all_normalized.append(locked)
            independent = independent_normalize(physical)
            require(np.array_equal(locked, independent),
                    f"locked/independent mapping {row['basename']}")
            require(np.all(np.isfinite(locked)) and np.all(np.abs(locked) <= 1.0),
                    f"normalized range {row['basename']}")
            indices = np.asarray(row["indices"], np.int64)
            cases, times = indices.size, 51
            shape = (cases, times)
            require(np.array_equal(
                values["source_draw_index"], np.broadcast_to(indices[:, None], shape)
            ), f"source draw {row['basename']}")
            require(np.array_equal(
                values["global_snapshot_index"],
                np.arange(next_global, next_global + cases * times).reshape(shape),
            ), f"global snapshot {row['basename']}")
            require(np.array_equal(
                values["time_index"], np.broadcast_to(np.arange(times), shape)
            ) and np.all(values["N"] == row["N"]), f"time/N {row['basename']}")
            parameters = {name: np.asarray(values[f"parameter_{name}"])
                          for name in ("cx", "cy", "width", "amplitude", "nu")}
            require(np.array_equal(values["normalized_parameters"],
                                   sampled["normalized"][indices]),
                    f"normalized parameters {row['basename']}")
            for name, observed in parameters.items():
                expected = sampled[name][indices]
                if name == "nu":
                    tolerance = np.abs(np.spacing(np.maximum(
                        np.abs(observed), np.abs(expected)
                    )))
                    matched = np.all(np.abs(observed - expected) <= tolerance)
                else:
                    matched = np.array_equal(observed, expected)
                require(matched, f"parameter {name} {row['basename']}")
            regenerated_features = c.trajectory_features(parameters, int(row["N"]))
            observed_features = np.asarray(values["features"])
            exact = (0, 1, 2, 3, 5, 6)
            feature_scale = np.maximum(
                np.abs(observed_features[..., 4]), np.abs(regenerated_features[..., 4])
            )
            tolerance = np.abs(np.spacing(feature_scale))
            require(np.array_equal(observed_features[..., exact],
                                   regenerated_features[..., exact])
                    and np.all(np.abs(observed_features[..., 4]
                                      - regenerated_features[..., 4]) <= tolerance),
                    f"features {row['basename']}")
            physical_min = np.minimum(physical_min, np.min(physical, axis=0))
            physical_max = np.maximum(physical_max, np.max(physical, axis=0))
            normalized_min = np.minimum(normalized_min, np.min(locked, axis=0))
            normalized_max = np.maximum(normalized_max, np.max(locked, axis=0))
            max_mapping_delta = max(
                max_mapping_delta, float(np.max(np.abs(locked - independent)))
            )
            next_global += cases * times
            snapshot_count += cases * times
        chunk_records.append({
            "basename": row["basename"], "sha256": row["sha256"],
            "N": row["N"], "snapshot_count": row["snapshot_count"],
        })
    require(snapshot_count == next_global == 35_904, "snapshot total")
    all_physical = np.concatenate(all_physical)
    all_normalized = np.concatenate(all_normalized)
    loaded = trainer.load_target_coefficients(
        SimpleNamespace(p5_npz=p5_npz, target_dir=target_dir), report, False
    )
    require(np.array_equal(loaded["physical_affine"], all_physical)
            and np.array_equal(loaded["affine"], all_normalized),
            "actual Phase7 loader physical/normalized mapping")
    fields, _parameters, reference_health = c.generate_population(
        64, 0, 704, np.asarray((0,), np.int64), chunk=1
    )
    coords = c.grid_coords(64)
    regenerated_normalized = np.stack([
        spline.normalized_state_from_affine(
            spline.affine_state_from_field(field, coords)
        ) for field in fields.reshape(51, 64 * 64)
    ])
    regenerated_delta = np.abs(regenerated_normalized - first_locked_normalized)
    # The local JAX 0.10.1 and immutable cluster JAX 0.10.2 reference chains
    # differ by sub-femtoscale reductions near zero; scientific same-version
    # execution retains the driver's bitwise equality requirement.
    regenerated_tolerance = 2e-15
    require(reference_health["reported_max_relative_residual"] <= 1e-8
            and reference_health["independent_max_relative_residual"] <= 1e-8,
            "regenerated N64 draw0 reference health")
    require(np.max(regenerated_delta) <= regenerated_tolerance,
            "regenerated N64 draw0 normalized affine mapping")
    return {
        "status": "pass", "classification": "read_only_real_P5_schema_regression",
        "p5_json_sha256": EXPECTED_P5, "chunk_count": len(chunk_records),
        "p5_npz_sha256": EXPECTED_P5_NPZ,
        "snapshot_count": snapshot_count, "chunks": chunk_records,
        "source_parameter_seed": 0, "source_parameter_draw_count": 704,
        "physical_affine_min": physical_min.tolist(),
        "physical_affine_max": physical_max.tolist(),
        "normalized_affine_min": normalized_min.tolist(),
        "normalized_affine_max": normalized_max.tolist(),
        "locked_vs_independent_max_abs": max_mapping_delta,
        "regenerated_N64_draw0_normalized_affine_max_abs": float(
            np.max(regenerated_delta)
        ),
        "regenerated_N64_draw0_tolerance_max_abs": float(
            regenerated_tolerance
        ),
        "regenerated_N64_draw0_reference_health": reference_health,
        "all_normalized_finite_and_in_unit_box": True,
        "all_regenerated_parameter_feature_metadata_match": True,
        "actual_phase7_loader_mapping_match": True,
        "scientific_training_executed": False,
        "model_validation_touched": False, "confirmation_touched": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p5-json", required=True)
    parser.add_argument("--p5-npz", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.p5_json, args.p5_npz, args.target_dir)
    if args.check:
        with open(args.output, encoding="utf-8") as handle:
            observed = json.load(handle)
        require(observed == result, "checked result drift")
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": "pass", "snapshots": result["snapshot_count"],
                      "mapping_max_abs": result["locked_vs_independent_max_abs"]}))


if __name__ == "__main__":
    main()
