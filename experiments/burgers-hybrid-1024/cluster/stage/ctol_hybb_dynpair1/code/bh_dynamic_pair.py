"""Exact AB/BA development confirmation for one dynamic correction policy."""
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
import bh_dynamic_correction as dc

OUT = sys.argv[1]
N = int(os.environ.get("N", "256"))
COARSE_N = int(os.environ.get("COARSE_N", str(N)))
RELAXATION = float(os.environ.get("RELAXATION", "1.0"))
FOM_TAU = float(os.environ.get("FOM_TAU", "1e-6"))
LIN_TOL = float(os.environ.get("LIN_TOL", "1e-2"))
DEV_SEED = int(os.environ.get("DEV_SEED", "20260818"))
DRAW_COUNT = int(os.environ.get("DRAW_COUNT", "16"))
CASE_START = int(os.environ.get("CASE_START", "8"))
N_CASES = int(os.environ.get("N_CASES", "4"))
PAIR_BLOCKS = int(os.environ.get("PAIR_BLOCKS", "6"))
BURN_S = float(os.environ.get("BURN_S", "3"))
SELECTION_JSON = os.environ.get(
    "SELECTION_JSON", "selection/dynamic_selection_choice.json"
)


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def bootstrap_interval(values, seed=20260820, draws=20000):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    medians = np.median(sampled, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tukey_counts(records, trajectories):
    by_case = {}
    for trajectory in trajectories:
        values = np.asarray(
            [
                record["elapsed_s"]
                for record in records
                if record["trajectory_index"] == trajectory["index"]
            ],
            np.float64,
        )
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        count = int(np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))
        by_case[str(trajectory["index"])] = count
    return {"per_trajectory": by_case, "total": int(sum(by_case.values()))}


def grade(outputs, trajectory, repetition, order, elapsed):
    U, newton, linear, breakdowns, flags, rel_res = [np.asarray(value) for value in outputs]
    record = {
        "trajectory_index": trajectory["index"],
        "repetition": repetition,
        "order": order,
        "elapsed_s": elapsed,
        "newton_total": int(np.sum(newton)),
        "linear_total": int(np.sum(linear)),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "max_returned_residual": float(np.max(rel_res)),
        "trajectory_rel_l2": float(
            np.linalg.norm(U - trajectory["U"][1:])
            / np.linalg.norm(trajectory["U"][1:])
        ),
    }
    if (
        record["breakdowns"]
        or record["flags_nonzero"]
        or not np.isfinite(record["max_returned_residual"])
        or record["max_returned_residual"] > FOM_TAU
    ):
        raise SystemExit(f"timed invocation health failure: {record}")
    return record


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("matmul precision must be highest")
    if PAIR_BLOCKS < 2:
        raise SystemExit("at least two AB/BA blocks are required")
    with open(SELECTION_JSON) as handle:
        selection = json.load(handle)
    selected = selection.get("selected")
    if selection.get("status") != "promoted" or selected is None:
        raise SystemExit("dynamic policy was not promoted by the selection gate")
    if (
        int(selected["coarse_n"]) != COARSE_N
        or float(selected["relaxation"]) != RELAXATION
    ):
        raise SystemExit("requested pair policy does not match selected policy")
    trajectories = bc.generate_reference(
        N,
        list(range(CASE_START, CASE_START + N_CASES)),
        DEV_SEED,
        draw_count=DRAW_COUNT,
    )
    predictor = dc.make_predictor(N, COARSE_N, RELAXATION)
    cubic_chain, _ = bc.make_chain(
        N, FOM_TAU, lin_tol=LIN_TOL, preconditioner="helmholtz"
    )
    candidate_chain, _ = bc.make_chain(
        N,
        FOM_TAU,
        predictor=predictor,
        lin_tol=LIN_TOL,
        preconditioner="helmholtz",
    )
    dummy = jnp.zeros((bc.T, N * N), jnp.float64)
    calls = {
        "cubic": [
            lambda tr=tr: cubic_chain(
                jnp.asarray(tr["U"][0]), tr["nu"], dummy, jnp.int32(5)
            )
            for tr in trajectories
        ],
        "dynamic": [
            lambda tr=tr: candidate_chain(
                jnp.asarray(tr["U"][0]), tr["nu"], dummy, jnp.int32(3)
            )
            for tr in trajectories
        ],
    }
    report = {
        "config": {
            "purpose": "disjoint development AB/BA confirmation of selected classical dynamic correction",
            "classification": "classical FOM warm-start control; not learned and not NM-ROM",
            "N": N,
            "coarse_n": COARSE_N,
            "relaxation": RELAXATION,
            "fom_tau": FOM_TAU,
            "lin_tol": LIN_TOL,
            "preconditioner": "exact target-grid Helmholtz",
            "development_seed": DEV_SEED,
            "development_indices": list(range(CASE_START, CASE_START + N_CASES)),
            "draw_count": DRAW_COUNT,
            "selection_indices_reused": False,
            "pair_blocks": PAIR_BLOCKS,
            "repetitions_per_trajectory": 2 * PAIR_BLOCKS,
            "burn_before_every_AB_and_BA_block": True,
            "burn_seconds": BURN_S,
            "accuracy_work_residual_from_every_timed_invocation": True,
            "new_final_seed_touched": False,
            "f64": True,
            "selection_artifact": SELECTION_JSON,
            "selection_sha256": sha256(SELECTION_JSON),
            "selected_policy": selected,
        },
        "provenance": provenance,
        "reference_health": {
            "max_residual": float(
                max(item["max_reference_newton_residual"] for item in trajectories)
            )
        },
        "records": {"cubic": [], "dynamic": []},
        "burn_records": [],
        "summary": {},
        "complete": False,
    }
    save(report)
    for method_calls in calls.values():
        for call in method_calls:
            call()[0].block_until_ready()

    repetition = 0
    for block in range(PAIR_BLOCKS):
        for trajectory_slot, trajectory in enumerate(trajectories):
            for order in (("cubic", "dynamic"), ("dynamic", "cubic")):
                burn_count = bc.gpu_burn(
                    lambda: calls["cubic"][trajectory_slot]()[0].block_until_ready(),
                    BURN_S,
                )
                report["burn_records"].append(
                    {
                        "block": block,
                        "trajectory_index": trajectory["index"],
                        "order": list(order),
                        "iterations": burn_count,
                    }
                )
                for arm in order:
                    started = time.perf_counter()
                    outputs = calls[arm][trajectory_slot]()
                    outputs[0].block_until_ready()
                    elapsed = float(time.perf_counter() - started)
                    report["records"][arm].append(
                        grade(outputs, trajectory, repetition, "/".join(order), elapsed)
                    )
                repetition += 1
        save(report)

    case_medians = {}
    for arm, records in report["records"].items():
        case_medians[arm] = []
        for trajectory in trajectories:
            values = [
                record["elapsed_s"]
                for record in records
                if record["trajectory_index"] == trajectory["index"]
            ]
            case_medians[arm].append(float(np.median(values)))
    cubic = np.asarray(case_medians["cubic"])
    dynamic = np.asarray(case_medians["dynamic"])
    savings = (cubic - dynamic) * 1e3
    report["summary"] = {
        "case_medians_s": case_medians,
        "cubic_median_ms": float(np.median(cubic) * 1e3),
        "dynamic_median_ms": float(np.median(dynamic) * 1e3),
        "speedup_cubic_over_dynamic": float(np.median(cubic) / np.median(dynamic)),
        "saving_vs_cubic_case_medians_ms": savings.tolist(),
        "saving_vs_cubic_median_ms": float(np.median(savings)),
        "saving_vs_cubic_trajectory_cluster_95ci_ms": bootstrap_interval(savings),
        "dynamic_newton_total_median": float(
            np.median([record["newton_total"] for record in report["records"]["dynamic"]])
        ),
        "cubic_newton_total_median": float(
            np.median([record["newton_total"] for record in report["records"]["cubic"]])
        ),
        "dynamic_linear_total_median": float(
            np.median([record["linear_total"] for record in report["records"]["dynamic"]])
        ),
        "cubic_linear_total_median": float(
            np.median([record["linear_total"] for record in report["records"]["cubic"]])
        ),
        "max_returned_residual": float(
            max(
                record["max_returned_residual"]
                for records in report["records"].values()
                for record in records
            )
        ),
        "max_trajectory_rel_l2": float(
            max(
                record["trajectory_rel_l2"]
                for records in report["records"].values()
                for record in records
            )
        ),
        "tukey_1p5iqr_outliers_retained": {
            arm: tukey_counts(records, trajectories)
            for arm, records in report["records"].items()
        },
    }
    report["complete"] = True
    save(report)
    bc.log(json.dumps(report["summary"], indent=1), "DONE")


if __name__ == "__main__":
    main()
