"""N=2048 paired confirmation of the selected classical Burgers warm start."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc
import bh_dynamic_correction as dc

OUT = sys.argv[1]
NS = tuple(int(value) for value in os.environ.get("NS", "2048").split(","))
FOM_TAUS = tuple(float(value) for value in os.environ.get(
    "FOM_TAUS", "1e-6,1e-8,1e-10"
).split(","))
LINEAR_TOLS = tuple(float(value) for value in os.environ.get(
    "LINEAR_TOLS", "1e-2,1e-4,1e-5"
).split(","))
CONDITIONS = tuple(zip(FOM_TAUS, LINEAR_TOLS))
TEST_SEED = int(os.environ.get("TEST_SEED", "20260827"))
DRAW_COUNT = int(os.environ.get("DRAW_COUNT", "16"))
TEST_START = int(os.environ.get("TEST_START", "0"))
N_CASES = int(os.environ.get("N_CASES", "4"))
PAIR_BLOCKS = int(os.environ.get("PAIR_BLOCKS", "6"))
BURN_S = float(os.environ.get("BURN_S", "3"))
REFERENCE_RESIDUAL_GATE = float(os.environ.get("REFERENCE_RESIDUAL_GATE", "1e-11"))
REFERENCE_NEWTON_ITERS = int(os.environ.get("NEWTON_ITERS", "25"))
SELECTION_JSON = os.environ.get(
    "SELECTION_JSON", "selection/dynamic_selection_choice.json"
)
PAIR_AUDIT_JSON = os.environ.get(
    "PAIR_AUDIT_JSON", "selection/dynamic_pair_audit.json"
)
SMOKE = os.environ.get("BH_2048_SMOKE", "0") == "1"

LOCKED_NS = (2048,)
LOCKED_CONDITIONS = ((1e-6, 1e-2), (1e-8, 1e-4), (1e-10, 1e-5))
LOCKED_FINAL = {
    "seed": 20260827,
    "draw_count": 16,
    "start": 0,
    "cases": 4,
    "pair_blocks": 6,
    "burn_s": 3.0,
}
LOCKED_SMOKE = {
    "seed": 20260818,
    "draw_count": 16,
    "start": 12,
    "cases": 1,
    "pair_blocks": 1,
    "burn_s": 0.1,
}


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def device_memory():
    stats = jax.devices()[0].memory_stats() or {}
    converted = {}
    for key, value in stats.items():
        if isinstance(value, (int, np.integer)):
            converted[key] = int(value)
        elif isinstance(value, (float, np.floating)) and np.isfinite(value):
            converted[key] = float(value)
    peak = converted.get("peak_bytes_in_use")
    limit = converted.get("bytes_limit")
    converted["peak_fraction_of_limit"] = (
        None if peak is None or limit in (None, 0) else float(peak / limit)
    )
    return converted


def bootstrap_interval(values, seed, draws=20000):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [float(value) for value in np.quantile(np.median(sampled, axis=1), [0.025, 0.975])]


def tukey_counts(records, trajectory_indices):
    by_case = {}
    for trajectory_index in trajectory_indices:
        values = np.asarray([
            record["elapsed_s"]
            for record in records
            if record["trajectory_index"] == trajectory_index
        ])
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        by_case[str(trajectory_index)] = int(np.sum(
            (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        ))
    return {"per_trajectory": by_case, "total": int(sum(by_case.values()))}


def grade(outputs, trajectory, block, sample_index, order, elapsed, arm, fom_tau):
    U, newton, linear, breakdowns, flags, rel_res = [np.asarray(value) for value in outputs]
    is_dynamic = arm == "dynamic"
    record = {
        "trajectory_index": trajectory["index"],
        "block": block,
        "sample_index": sample_index,
        "order": "/".join(order),
        "elapsed_s": elapsed,
        "finish_newton_total": int(np.sum(newton)),
        "finish_bicgstab_total": int(np.sum(linear)),
        "extra_full_grid_residual_evaluations": int(bc.T if is_dynamic else 0),
        "extra_exact_helmholtz_inverses": int(bc.T if is_dynamic else 0),
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
        or record["max_returned_residual"] > fom_tau
        or not np.isfinite(record["trajectory_rel_l2"])
    ):
        raise SystemExit(f"timed invocation health failure: {record}")
    return record


def summarize(row, trajectory_indices, bootstrap_seed):
    case_medians = {}
    for arm, records in row["records"].items():
        case_medians[arm] = [
            float(np.median([
                record["elapsed_s"]
                for record in records
                if record["trajectory_index"] == trajectory_index
            ]))
            for trajectory_index in trajectory_indices
        ]
    paired_case_savings_ms = []
    for trajectory_index in trajectory_indices:
        cubic = {
            record["sample_index"]: record["elapsed_s"]
            for record in row["records"]["cubic"]
            if record["trajectory_index"] == trajectory_index
        }
        dynamic = {
            record["sample_index"]: record["elapsed_s"]
            for record in row["records"]["dynamic"]
            if record["trajectory_index"] == trajectory_index
        }
        if cubic.keys() != dynamic.keys() or len(cubic) != 2 * PAIR_BLOCKS:
            raise SystemExit(f"incomplete paired timing grid for trajectory {trajectory_index}")
        paired_case_savings_ms.append(float(np.median([
            (cubic[index] - dynamic[index]) * 1e3 for index in sorted(cubic)
        ])))
    cubic_case = np.asarray(case_medians["cubic"])
    dynamic_case = np.asarray(case_medians["dynamic"])
    return {
        "case_medians_s": case_medians,
        "cubic_median_ms": float(np.median(cubic_case) * 1e3),
        "dynamic_median_ms": float(np.median(dynamic_case) * 1e3),
        "speedup_cubic_over_dynamic": float(np.median(cubic_case) / np.median(dynamic_case)),
        "paired_saving_vs_cubic_case_medians_ms": paired_case_savings_ms,
        "paired_saving_vs_cubic_median_ms": float(np.median(paired_case_savings_ms)),
        "paired_saving_trajectory_cluster_95ci_ms": bootstrap_interval(
            paired_case_savings_ms, bootstrap_seed
        ),
        "finish_newton_total_median": {
            arm: float(np.median([
                record["finish_newton_total"] for record in records
            ]))
            for arm, records in row["records"].items()
        },
        "finish_bicgstab_total_median": {
            arm: float(np.median([
                record["finish_bicgstab_total"] for record in records
            ]))
            for arm, records in row["records"].items()
        },
        "dynamic_extra_full_grid_residual_evaluations": int(bc.T),
        "dynamic_extra_exact_helmholtz_inverses": int(bc.T),
        "max_returned_residual": float(max(
            record["max_returned_residual"]
            for records in row["records"].values()
            for record in records
        )),
        "max_trajectory_rel_l2": float(max(
            record["trajectory_rel_l2"]
            for records in row["records"].values()
            for record in records
        )),
        "tukey_1p5iqr_outliers_retained": {
            arm: tukey_counts(records, trajectory_indices)
            for arm, records in row["records"].items()
        },
    }


def run_condition(report, n, trajectories, fom_tau, linear_tol, condition_index):
    predictor = dc.make_predictor(n, n, 1.0)
    cubic_chain, _ = bc.make_chain(
        n, fom_tau, lin_tol=linear_tol, preconditioner="helmholtz"
    )
    dynamic_chain, _ = bc.make_chain(
        n,
        fom_tau,
        predictor=predictor,
        lin_tol=linear_tol,
        preconditioner="helmholtz",
    )
    dummy = jnp.zeros((bc.T, n * n), jnp.float64)
    calls = {
        "cubic": [
            lambda tr=tr: cubic_chain(
                jnp.asarray(tr["U"][0]), tr["nu"], dummy, jnp.int32(5)
            )
            for tr in trajectories
        ],
        "dynamic": [
            lambda tr=tr: dynamic_chain(
                jnp.asarray(tr["U"][0]), tr["nu"], dummy, jnp.int32(3)
            )
            for tr in trajectories
        ],
    }
    row = {
        "N": n,
        "fom_tau": fom_tau,
        "linear_tol": linear_tol,
        "coarse_n": n,
        "relaxation": 1.0,
        "records": {"cubic": [], "dynamic": []},
        "burn_records": [],
        "summary": {},
        "equivalence": {},
        "device_memory_after_compile": {},
        "device_memory_after_condition": {},
        "complete": False,
    }
    report["rows"].append(row)
    save(report)

    for method_calls in calls.values():
        for call in method_calls:
            call()[0].block_until_ready()
    row["device_memory_after_compile"] = device_memory()
    save(report)

    for block in range(PAIR_BLOCKS):
        for trajectory_slot, trajectory in enumerate(trajectories):
            for order_index, order in enumerate(
                (("cubic", "dynamic"), ("dynamic", "cubic"))
            ):
                burn_count = bc.gpu_burn(
                    lambda: calls["cubic"][trajectory_slot]()[0].block_until_ready(),
                    BURN_S,
                )
                sample_index = 2 * block + order_index
                row["burn_records"].append({
                    "block": block,
                    "sample_index": sample_index,
                    "trajectory_index": trajectory["index"],
                    "order": list(order),
                    "iterations": burn_count,
                })
                for arm in order:
                    started = time.perf_counter()
                    outputs = calls[arm][trajectory_slot]()
                    outputs[0].block_until_ready()
                    elapsed = float(time.perf_counter() - started)
                    row["records"][arm].append(grade(
                        outputs,
                        trajectory,
                        block,
                        sample_index,
                        order,
                        elapsed,
                        arm,
                        fom_tau,
                    ))
        save(report)

    trajectory_indices = [trajectory["index"] for trajectory in trajectories]
    row["summary"] = summarize(
        row, trajectory_indices, TEST_SEED + n + 1009 * condition_index
    )
    row["equivalence"] = bc.reference_equivalence(
        n,
        trajectories,
        cubic_chain,
        lin_tol=linear_tol,
        preconditioner="helmholtz",
    )
    row["device_memory_after_condition"] = device_memory()
    row["complete"] = True
    save(report)
    bc.log(
        f"N={n} tau={fom_tau:.0e}: cubic={row['summary']['cubic_median_ms']:.3f}ms "
        f"dynamic={row['summary']['dynamic_median_ms']:.3f}ms "
        f"paired_save={row['summary']['paired_saving_vs_cubic_median_ms']:.3f}ms "
        f"speedup={row['summary']['speedup_cubic_over_dynamic']:.3f}x "
        f"peak_fraction={row['device_memory_after_condition'].get('peak_fraction_of_limit')}"
    )


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu")
    if getattr(jax.devices()[0], "device_kind", "") != "NVIDIA H200":
        raise SystemExit("N=2048 preregistration requires an NVIDIA H200")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest" or not provenance["x64"]:
        raise SystemExit("f64/highest precision contract failed")
    if len(FOM_TAUS) != len(LINEAR_TOLS):
        raise SystemExit("FOM_TAUS and LINEAR_TOLS must have equal length")
    if not SMOKE and (
        REFERENCE_NEWTON_ITERS != 25 or bc.bf.NEWTON_ITERS != 25
    ):
        raise SystemExit("N=2048 reference generation requires fixed 25 Newton iterations")
    lock = LOCKED_SMOKE if SMOKE else LOCKED_FINAL
    expected_conditions = ((1e-6, 1e-2),) if SMOKE else LOCKED_CONDITIONS
    if (
        NS != LOCKED_NS
        or CONDITIONS != expected_conditions
        or TEST_SEED != lock["seed"]
        or DRAW_COUNT != lock["draw_count"]
        or TEST_START != lock["start"]
        or N_CASES != lock["cases"]
        or PAIR_BLOCKS != lock["pair_blocks"]
        or BURN_S != lock["burn_s"]
        or REFERENCE_RESIDUAL_GATE != 1e-11
    ):
        raise SystemExit("N=2048 execution configuration drifted from preregistration")
    with open(SELECTION_JSON) as handle:
        selection = json.load(handle)
    with open(PAIR_AUDIT_JSON) as handle:
        pair_audit = json.load(handle)
    selected = selection.get("selected")
    if (
        selection.get("status") != "promoted"
        or selected is None
        or selected.get("arm") != "coarse256_a1"
        or int(selected["coarse_n"]) != 256
        or float(selected["relaxation"]) != 1.0
        or not selected.get("eligible")
    ):
        raise SystemExit("full-grid development policy was not promoted")
    if (
        pair_audit.get("status") != "promoted"
        or pair_audit.get("selected_policy", {}).get("resolution_policy")
        != "coarse_n=N at every mesh (full-grid fraction 1.0)"
        or pair_audit.get("selection_sha256") != sha256(SELECTION_JSON)
    ):
        raise SystemExit("disjoint AB/BA audit did not promote the full-grid policy")

    report = {
        "config": {
            "purpose": (
                "N=2048 memory/execution smoke" if SMOKE else
                "fresh-seed N=2048 classical warm-start extension"
            ),
            "classification": {
                "cubic": "classical live cubic-history FOM warm start",
                "dynamic": (
                    "classical FOM warm start; live cubic plus one target-grid exact-upwind "
                    "residual and exact Helmholtz inverse; not learned and not NM-ROM"
                ),
            },
            "execution_kind": "smoke_excluded" if SMOKE else "authoritative_fresh_seed",
            "ns": list(NS),
            "conditions": [
                {"fom_tau": tau, "linear_tol": linear_tol}
                for tau, linear_tol in CONDITIONS
            ],
            "resolution_policy": "coarse_n=N at every mesh (full-grid fraction 1.0)",
            "relaxation": 1.0,
            "preconditioner": "exact target-grid Helmholtz for both FOM finishes",
            "dynamic_correction_steps_per_trajectory": int(bc.T),
            "dynamic_extra_full_grid_residual_evaluations": int(bc.T),
            "dynamic_extra_exact_helmholtz_inverses": int(bc.T),
            "test_seed": TEST_SEED,
            "canonical_draw_count": DRAW_COUNT,
            "selected_test_indices": list(range(TEST_START, TEST_START + N_CASES)),
            "pair_blocks": PAIR_BLOCKS,
            "repetitions_per_arm_per_trajectory": 2 * PAIR_BLOCKS,
            "burn_before_every_AB_and_BA_block": True,
            "burn_seconds": BURN_S,
            "timing_estimator": (
                "median paired repetitions within trajectory, then median trajectories"
            ),
            "confidence_interval": (
                "20000-draw trajectory-cluster bootstrap of per-trajectory paired medians"
            ),
            "timing_scope": (
                "warmed compiled online trajectory solve; includes dynamic construction "
                "and FOM finish; excludes compilation, loading, test/reference generation, "
                "and first-query latency"
            ),
            "accuracy_work_residual_from_every_timed_invocation": True,
            "selection_artifact": SELECTION_JSON,
            "selection_sha256": sha256(SELECTION_JSON),
            "pair_audit_artifact": PAIR_AUDIT_JSON,
            "pair_audit_sha256": sha256(PAIR_AUDIT_JSON),
            "reference_residual_gate": REFERENCE_RESIDUAL_GATE,
            "fresh_seed_touched": not SMOKE,
            "smoke": SMOKE,
            "f64": True,
        },
        "provenance": provenance,
        "reference_health": {},
        "rows": [],
        "device_memory_at_start": device_memory(),
        "device_memory_at_end": {},
        "complete": False,
    }
    if not SMOKE:
        report["config"].update({
            "reference_newton_iterations": REFERENCE_NEWTON_ITERS,
            "reference_solver_scope": (
                "offline truth generation and independent equivalence only; "
                "online counting solvers remain tolerance-stopped with MAX_NEWTON=25"
            ),
        })
    save(report)

    for n in NS:
        trajectories = bc.generate_reference(
            n,
            list(range(TEST_START, TEST_START + N_CASES)),
            TEST_SEED,
            draw_count=DRAW_COUNT,
        )
        worst_reference = float(max(
            trajectory["max_reference_newton_residual"] for trajectory in trajectories
        ))
        if not np.isfinite(worst_reference) or worst_reference > REFERENCE_RESIDUAL_GATE:
            raise SystemExit(f"N={n}: reference residual {worst_reference:.3e}")
        report["reference_health"][str(n)] = {
            "max_reference_newton_residual": worst_reference
        }
        save(report)
        for condition_index, (fom_tau, linear_tol) in enumerate(CONDITIONS):
            run_condition(
                report, n, trajectories, fom_tau, linear_tol, condition_index
            )

    report["device_memory_at_end"] = device_memory()
    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
