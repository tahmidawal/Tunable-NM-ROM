"""D1 conditional HG5S representation oracle and analytic-kernel cost bound."""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import b10_common as c

OUTPUT_JSON, OUTPUT_NPZ, D0_JSON = sys.argv[1:4]
N = int(os.environ.get("N", "64"))
SEED = int(os.environ.get("DATA_SEED", "0"))
DRAW_COUNT = int(os.environ.get("DRAW_COUNT", "704"))
START = int(os.environ.get("SELECTION_START", "512"))
COUNT = int(os.environ.get("SELECTION_COUNT", "64"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
BURN_SECONDS = float(os.environ.get("BURN_SECONDS", "3"))


def authoritative_wall_license(d0):
    """Recompute the preregistered license from representation-oracle errors."""
    cx, cy, *_ = c.bf.sample_params(seed=SEED, m=DRAW_COUNT)
    selected = np.arange(START, START + COUNT)
    distance = np.minimum.reduce((
        cx[selected], cy[selected], 1.0 - cx[selected], 1.0 - cy[selected]
    ))
    nearest = np.array_split(np.argsort(distance), 4)[0]
    concepts = {}
    for name in ("HG4", "HG5"):
        if name not in d0.get("concepts", {}):
            continue
        errors = np.asarray(
            d0["concepts"][name]["representation_oracle"]["trajectory_all"],
            np.float64,
        )
        if errors.shape != (COUNT,):
            raise SystemExit(f"{name} oracle error shape does not match locked selection")
        fraction = float(
            np.sum(np.square(errors[nearest])) / np.sum(np.square(errors))
        )
        concepts[name] = {
            "nearest_wall_quartile_indices": selected[nearest].tolist(),
            "nearest_wall_squared_oracle_error_fraction": fraction,
            "license_pass": bool(fraction >= 0.5),
        }
    return {"concepts": concepts, "fires": bool(any(
        value["license_pass"] for value in concepts.values()
    ))}


def wall_strata(errors, parameters):
    distance = np.minimum.reduce((
        parameters["cx"], parameters["cy"],
        1.0 - parameters["cx"], 1.0 - parameters["cy"],
    ))
    order = np.argsort(distance)
    return [{
        "quartile": quartile,
        "count": int(group.size),
        "distance_range": [float(np.min(distance[group])), float(np.max(distance[group]))],
        "mean_error": float(np.mean(errors[group])),
        "squared_error_fraction": float(
            np.sum(np.square(errors[group])) / np.sum(np.square(errors))
        ),
    } for quartile, group in enumerate(np.array_split(order, 4)) if group.size]


def timed(kernel, argument):
    started = time.perf_counter()
    kernel(argument).block_until_ready()
    first_use = time.perf_counter() - started
    c.gpu_burn(BURN_SECONDS)
    kernel(argument).block_until_ready()
    samples = []
    for _ in range(TIME_REPS):
        started = time.perf_counter()
        kernel(argument).block_until_ready()
        samples.append(time.perf_counter() - started)
    return {
        "first_use_compile_and_call_s": float(first_use),
        "all_s": [float(value) for value in samples],
        "median_s": float(np.median(samples)),
        "outliers_gt_1p5_median": int(np.sum(
            np.asarray(samples) > 1.5 * np.median(samples)
        )),
    }


def main():
    c.require_gpu_highest()
    with open(D0_JSON) as handle:
        d0 = json.load(handle)
    if d0.get("status") != "complete":
        raise SystemExit("D0 is not complete")
    wall_license = authoritative_wall_license(d0)
    wall_licensed = wall_license["fires"]
    if not wall_licensed:
        raise SystemExit("D1 HG5S lacks the authoritative oracle-error wall license")
    started = time.time()
    fields, parameters, health = c.generate_population(
        N, SEED, DRAW_COUNT, np.arange(START, START + COUNT)
    )
    flat = fields.reshape(-1, N * N)
    states, oracle, condition = c.fit_hg5s_states(
        flat, c.grid_coords(N), c.binary_boundary_mask(N), batch=32
    )
    oracle = oracle.reshape(fields.shape)
    metrics = c.error_metrics(oracle, fields)
    strata = wall_strata(np.asarray(metrics["trajectory_all"]), parameters)
    representative_state = jnp.asarray(states[0])
    representative_cholesky = c.hg5s_cholesky(representative_state)
    representative_coords = jnp.asarray(c.grid_coords(N), jnp.float64)
    interior_weight, wall_weight = c.hg5s_chart_weights(
        representative_coords, representative_state[:2], representative_cholesky
    )
    pou_error = float(jnp.max(jnp.abs(interior_weight + wall_weight - 1.0)))
    if pou_error > 1e-15:
        raise SystemExit(f"partition-of-unity identity failed: {pou_error:.3e}")

    # Lower-bound the analytic decoder and one weak Jacobian before training it.
    representative = jnp.asarray(states.reshape(COUNT, c.NUM_STEPS + 1, 32)[0])
    cost = {}
    for q in (64, 128, 256):
        coords = jnp.asarray(c.grid_coords(q), jnp.float64)
        mask = jnp.asarray(c.binary_boundary_mask(q), jnp.float64)

        @jax.jit
        def decode_trajectory(values, coords_=coords, mask_=mask):
            return jax.vmap(lambda state: c.hg5s_basis(
                coords_, mask_, state[:2], c.hg5s_cholesky(state)
            ) @ state[5:])(values)

        cost[f"decode_51_q{q}"] = timed(decode_trajectory, representative)

    k = 32
    M = 128
    m = 4 * M
    target_coords = jnp.asarray(c.grid_coords(1024), jnp.float64)
    rng = np.random.default_rng(20260819)
    point_indices = np.sort(rng.choice(
        np.arange(1024 * 1024), size=5 * m, replace=False
    ))
    weak_coords = target_coords[jnp.asarray(point_indices)]
    weak_mask = jnp.asarray(c.binary_boundary_mask(1024), jnp.float64)[
        jnp.asarray(point_indices)
    ]

    @jax.jit
    def decoder_jacobian(state):
        return jax.jacfwd(lambda value: c.hg5s_basis(
            weak_coords, weak_mask, value[:2], c.hg5s_cholesky(value)
        ) @ value[5:])(state)

    cost["one_decoder_jacobian_5m"] = timed(decoder_jacobian, representative[1])
    jacobian_median = cost["one_decoder_jacobian_5m"]["median_s"]
    cost["one_update_50_step_jacobian_lower_bound_s"] = 50.0 * jacobian_median

    report = {
        "stage": "D1 conditional HG5S representation and cost oracle",
        "status": "complete",
        "config": {
            "N": N, "seed": SEED, "draw_count": DRAW_COUNT,
            "selection_indices": [START, START + COUNT - 1],
            "model_validation_touched": False, "confirmation_touched": False,
            "latent_dimension": k, "M": M, "m": m,
            "architecture": "full-covariance HG5/HG2 two-way wall partition of unity",
            "parameter_count_grid_independent": True, "f64": True,
        },
        "provenance": c.provenance(),
        "d0": {
            "sha256": c.sha256(D0_JSON),
            "predictor_based_wall_chart_licensed_non_authoritative": True,
            "authoritative_representation_oracle_wall_license": wall_license,
        },
        "partition_of_unity": {
            "definition": "w_interior=1-w_wall",
            "representative_max_abs_sum_minus_one": pou_error,
            "gate": 1e-15,
            "gate_pass": bool(pou_error <= 1e-15),
        },
        "reference_health": health,
        "representation_oracle": metrics,
        "basis_condition": {
            "median": float(np.median(condition)), "worst": float(np.max(condition)),
        },
        "wall_distance_strata": strata,
        "representation_gate_pass": bool(
            metrics["trajectory_mean"] <= 2e-4
            and metrics["trajectory_worst"] <= 7e-4
        ),
        "cost_oracle": cost,
        "elapsed_seconds": float(time.time() - started),
    }
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, states=states)
    report["npz"] = {"path": os.path.basename(OUTPUT_NPZ), "sha256": c.sha256(OUTPUT_NPZ)}
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps(report, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
