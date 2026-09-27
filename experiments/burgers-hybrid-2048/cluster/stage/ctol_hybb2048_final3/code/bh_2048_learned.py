"""Preregistered N=2048 sensitivity of the audited weak FiLM NM-ROM.

This is not selected from the classical N=2048 outcome.  The excluded memory
smoke independently licensed it before either fresh seed was opened.  Every
timed FiLM call contains weak construction, fixed-N=64 decode/prolongation,
the exact-residual guard, and the FOM finish in one dependent invocation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# The checkpoint was trained with the N=64 decoder/testbed configuration.
# Target-grid FOM and EQ routines below still receive N=2048 explicitly.
_FILM_ENV = {
    "N": "64",
    "N_TRAIN": "512",
    "N_VAL": "64",
    "SEED": "0",
    "BC_MODE": "poly",
    "K_LAT": "8",
    "AD_HIDDEN": "256",
    "AD_LAYERS": "5",
    "GN_BUDGET": "30",
    "GN_TOL": "1e-9",
    "IC_BUDGET": "100",
    "EQ_SNAPS": "64",
    "EQ_GRID_POOL": "4096",
}
for _key, _value in _FILM_ENV.items():
    if _key in os.environ and os.environ[_key] != _value:
        raise SystemExit(
            f"audited FiLM environment drifted: {_key}={os.environ[_key]!r}"
        )
    os.environ.setdefault(_key, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.abspath(os.path.join(HERE, "..", "burgers-hybrid-1024"))
if os.path.isfile(os.path.join(OLD, "bh_film_control.py")):
    sys.path.insert(0, OLD)

import bh_common as bc  # noqa: E402
import bh_dynamic_correction as dc  # noqa: E402
from bh_film_control import FilmControl, VARIANT  # noqa: E402

OUT = sys.argv[1]
CHECKPOINT = os.environ["FILM_CHECKPOINT"]
SMOKE_AUDIT = os.environ.get("SMOKE_AUDIT", "smoke/smoke_audit.json")

N = 2048
CONDITIONS = ((1e-6, 1e-2), (1e-8, 1e-4), (1e-10, 1e-5))
TEST_SEED = 20260828
DRAW_COUNT = 16
TRAJECTORY_INDICES = (0, 1, 2, 3)
PAIR_BLOCKS = 6
BURN_S = 3.0
DECODE_RESOLUTION = 64
DECODE_CHUNK = 2
LATENT_HISTORY_MODE = "extrapolation"
MAX_STEP_JACOBIANS = 2
REFERENCE_RESIDUAL_GATE = 1e-11
EXPECTED_CHECKPOINT_SHA256 = (
    "aa07cd4a1471c59ad34741b41d620731ebd40d5bf445bd75416ce60358f7ecb9"
)
COMPARISONS = ("cubic", "dynamic")


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
    return [
        float(value)
        for value in np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
    ]


def tukey_count(values):
    values = np.asarray(values, np.float64)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    return int(np.sum(
        (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    ))


def grade_fom(outputs, trajectory, elapsed, arm, comparison, block,
              sample_index, order, fom_tau, construction=None):
    values = [np.asarray(value) for value in outputs]
    U, newton, linear, breakdowns, flags, rel_res = values[:6]
    record = {
        "trajectory_index": trajectory["index"],
        "comparison": comparison,
        "block": block,
        "sample_index": sample_index,
        "order": "/".join(order),
        "arm": arm,
        "elapsed_s": float(elapsed),
        "finish_newton_total": int(np.sum(newton)),
        "finish_bicgstab_total": int(np.sum(linear)),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "max_returned_residual": float(np.max(rel_res)),
        "trajectory_rel_l2": float(
            np.linalg.norm(U - trajectory["U"][1:])
            / np.linalg.norm(trajectory["U"][1:])
        ),
        "extra_full_grid_residual_evaluations": int(
            bc.T if arm in ("dynamic", "film_nmrom") else 0
        ),
        "extra_exact_helmholtz_inverses": int(
            bc.T if arm == "dynamic" else 0
        ),
    }
    if arm == "film_nmrom":
        guard = values[6]
        record.update({
            "film_guard_accepted_count": int(np.sum(guard)),
            "film_guard_accepted_fraction": float(np.mean(guard)),
            "weak_residual_steps": int(bc.T),
            "weak_test_modes": 64,
            "eq_points": 256,
        })
        if construction is None:
            raise SystemExit("FiLM construction telemetry is missing")
        (
            ic_relative,
            ic_jacobians,
            best_start,
            ic_attempts,
            reduced_norm,
            step_jacobians,
            reasons,
            step_attempts,
            field_features,
        ) = [np.asarray(value) for value in construction]
        record.update({
            "ic_relative_eq_misfit": float(ic_relative),
            "ic_jacobians": int(ic_jacobians),
            "ic_best_start": int(best_start),
            "ic_attempts": int(ic_attempts),
            "reduced_final_norm_per_step": reduced_norm.tolist(),
            "reduced_jacobians_total": int(np.sum(step_jacobians)),
            "reduced_jacobians_per_step": step_jacobians.tolist(),
            "reduced_reason_per_step": reasons.tolist(),
            "reduced_attempts_total": int(np.sum(step_attempts)),
            "reduced_attempts_per_step": step_attempts.tolist(),
            "field_features_from_u0_and_nu": field_features.tolist(),
        })
    if (
        record["breakdowns"]
        or record["flags_nonzero"]
        or not math.isfinite(record["elapsed_s"])
        or record["elapsed_s"] <= 0
        or not math.isfinite(record["max_returned_residual"])
        or record["max_returned_residual"] > fom_tau
        or not math.isfinite(record["trajectory_rel_l2"])
    ):
        raise SystemExit(f"unhealthy timed invocation: {record}")
    return record


def summarize_comparison(records, comparison, condition_index):
    selected = [record for record in records if record["comparison"] == comparison]
    by_arm = {
        arm: [record for record in selected if record["arm"] == arm]
        for arm in (comparison, "film_nmrom")
    }
    case_medians = {arm: [] for arm in by_arm}
    paired_case_savings = []
    outliers = {arm: {} for arm in by_arm}
    for trajectory_index in TRAJECTORY_INDICES:
        keyed = {}
        for arm, arm_records in by_arm.items():
            case = [
                record for record in arm_records
                if record["trajectory_index"] == trajectory_index
            ]
            keyed[arm] = {
                record["sample_index"]: record["elapsed_s"] for record in case
            }
            if len(case) != 2 * PAIR_BLOCKS or len(keyed[arm]) != 2 * PAIR_BLOCKS:
                raise SystemExit(
                    f"{comparison} trajectory {trajectory_index}: timing grid incomplete"
                )
            case_medians[arm].append(float(np.median([
                record["elapsed_s"] for record in case
            ])))
            outliers[arm][str(trajectory_index)] = tukey_count([
                record["elapsed_s"] for record in case
            ])
        if keyed[comparison].keys() != keyed["film_nmrom"].keys():
            raise SystemExit(f"{comparison}: paired keys drifted")
        paired_case_savings.append(float(np.median([
            1e3 * (
                keyed[comparison][sample_index]
                - keyed["film_nmrom"][sample_index]
            )
            for sample_index in sorted(keyed[comparison])
        ])))
    control = np.asarray(case_medians[comparison])
    film = np.asarray(case_medians["film_nmrom"])
    ci = bootstrap_interval(
        paired_case_savings,
        TEST_SEED + 1009 * condition_index + (0 if comparison == "cubic" else 1),
    )
    return {
        "comparison": f"{comparison}_vs_film_nmrom",
        "case_medians_s": case_medians,
        "control_median_ms": float(np.median(control) * 1e3),
        "film_nmrom_median_ms": float(np.median(film) * 1e3),
        "speedup_control_over_film_nmrom": float(
            np.median(control) / np.median(film)
        ),
        "paired_saving_control_minus_film_case_medians_ms": paired_case_savings,
        "paired_saving_control_minus_film_median_ms": float(
            np.median(paired_case_savings)
        ),
        "paired_saving_trajectory_cluster_95ci_ms": ci,
        "supported_film_speedup": bool(ci[0] > 0),
        "finish_newton_total_median": {
            arm: float(np.median([
                record["finish_newton_total"] for record in arm_records
            ]))
            for arm, arm_records in by_arm.items()
        },
        "finish_bicgstab_total_median": {
            arm: float(np.median([
                record["finish_bicgstab_total"] for record in arm_records
            ]))
            for arm, arm_records in by_arm.items()
        },
        "max_returned_residual": float(max(
            record["max_returned_residual"] for record in selected
        )),
        "max_trajectory_rel_l2": float(max(
            record["trajectory_rel_l2"] for record in selected
        )),
        "film_guard_accepted_count_median": float(np.median([
            record["film_guard_accepted_count"]
            for record in by_arm["film_nmrom"]
        ])),
        "reduced_jacobians_total_median": float(np.median([
            record["reduced_jacobians_total"]
            for record in by_arm["film_nmrom"]
        ])),
        "outliers_retained": {
            arm: {
                "per_trajectory": values,
                "total": int(sum(values.values())),
            }
            for arm, values in outliers.items()
        },
    }


def run_condition(report, trajectories, built, fom_tau, linear_tol,
                  condition_index):
    predictor = dc.make_predictor(N, N, 1.0)
    cubic_chain, _ = bc.make_chain(
        N, fom_tau, lin_tol=linear_tol, preconditioner="helmholtz"
    )
    dynamic_chain, _ = bc.make_chain(
        N,
        fom_tau,
        predictor=predictor,
        lin_tol=linear_tol,
        preconditioner="helmholtz",
    )
    guarded_film_chain, _ = bc.make_chain(
        N,
        fom_tau,
        lin_tol=linear_tol,
        preconditioner="helmholtz",
        return_guard=True,
    )
    dummy = jnp.zeros((bc.T, N * N), jnp.float64)
    construct = built["construct"]

    @jax.jit
    def film_end_to_end(u0, nu):
        construction = construct(u0, nu)
        fom = guarded_film_chain(u0, nu, construction[0], jnp.int32(8))
        # The full guess is consumed by the finishing chain, not retained as an
        # output.  All scalar/per-step construction work telemetry is retained.
        return fom, construction[1:]

    calls = {}
    for trajectory in trajectories:
        u0 = jnp.asarray(trajectory["U"][0])
        nu = trajectory["nu"]
        index = trajectory["index"]
        calls[("cubic", index)] = (
            lambda u0_value=u0, nu_value=nu: cubic_chain(
                u0_value, nu_value, dummy, jnp.int32(5)
            )
        )
        calls[("dynamic", index)] = (
            lambda u0_value=u0, nu_value=nu: dynamic_chain(
                u0_value, nu_value, dummy, jnp.int32(3)
            )
        )
        calls[("film_nmrom", index)] = (
            lambda u0_value=u0, nu_value=nu: film_end_to_end(
                u0_value, nu_value
            )
        )

    for call in calls.values():
        jax.block_until_ready(call())

    row = {
        "N": N,
        "fom_tau": fom_tau,
        "linear_tol": linear_tol,
        "records": [],
        "burn_records": [],
        "summaries": {},
        "equivalence": {},
        "device_memory_after_compile": device_memory(),
        "device_memory_after_condition": {},
        "complete": False,
    }
    report["rows"].append(row)
    save(report)

    pair_orders = (
        ("cubic", ("cubic", "film_nmrom"), 0),
        ("cubic", ("film_nmrom", "cubic"), 1),
        ("dynamic", ("dynamic", "film_nmrom"), 0),
        ("dynamic", ("film_nmrom", "dynamic"), 1),
    )
    for block in range(PAIR_BLOCKS):
        for trajectory in trajectories:
            index = trajectory["index"]
            for comparison, order, order_index in pair_orders:
                burn_count = bc.gpu_burn(
                    lambda trajectory_index=index: jax.block_until_ready(
                        calls[("cubic", trajectory_index)]()
                    ),
                    BURN_S,
                )
                sample_index = 2 * block + order_index
                row["burn_records"].append({
                    "comparison": comparison,
                    "block": block,
                    "sample_index": sample_index,
                    "trajectory_index": index,
                    "order": list(order),
                    "iterations": burn_count,
                })
                for arm in order:
                    started = time.perf_counter()
                    output = calls[(arm, index)]()
                    jax.block_until_ready(output)
                    elapsed = float(time.perf_counter() - started)
                    if arm == "film_nmrom":
                        fom, construction = output
                    else:
                        fom, construction = output, None
                    row["records"].append(grade_fom(
                        fom,
                        trajectory,
                        elapsed,
                        arm,
                        comparison,
                        block,
                        sample_index,
                        order,
                        fom_tau,
                        construction,
                    ))
        save(report)

    row["summaries"] = {
        comparison: summarize_comparison(
            row["records"], comparison, condition_index
        )
        for comparison in COMPARISONS
    }
    row["equivalence"] = bc.reference_equivalence(
        N,
        trajectories,
        cubic_chain,
        lin_tol=linear_tol,
        preconditioner="helmholtz",
    )
    row["device_memory_after_condition"] = device_memory()
    row["complete"] = True
    save(report)
    for comparison, summary in row["summaries"].items():
        bc.log(
            f"N={N} tau={fom_tau:.0e} {comparison}/FiLM: "
            f"{summary['control_median_ms']:.3f}/"
            f"{summary['film_nmrom_median_ms']:.3f}ms "
            f"speedup={summary['speedup_control_over_film_nmrom']:.3f}x"
        )


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu")
    if getattr(jax.devices()[0], "device_kind", "") != "NVIDIA H200":
        raise SystemExit("N=2048 learned sensitivity requires NVIDIA H200")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest" or not provenance["x64"]:
        raise SystemExit("f64/highest precision contract failed")
    with open(SMOKE_AUDIT) as handle:
        smoke = json.load(handle)
    if (
        smoke.get("status") != "smoke_pass"
        or not smoke.get("primary_final_panel_licensed")
        or smoke.get("learned_sensitivity_decision") != "shall_run"
        or smoke.get("peak_device_fraction_of_limit", 1.0) > 0.75
    ):
        raise SystemExit("excluded smoke did not license learned sensitivity")
    film = FilmControl(
        CHECKPOINT,
        decode_chunk=DECODE_CHUNK,
        decode_resolution=DECODE_RESOLUTION,
    )
    if film.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("audited K=8 checkpoint hash drifted")

    report = {
        "config": {
            "purpose": "preregistered N=2048 genuine weak FiLM sensitivity",
            "classification": {
                "film_nmrom": (
                    "genuine weak FiLM NM-ROM; M=64,m=256, two latent "
                    "Jacobians per step, fixed-N64 decode/prolongation, exact guard"
                ),
                "cubic": "classical live cubic-history FOM warm start",
                "dynamic": (
                    "classical residual-plus-exact-Helmholtz FOM warm start; "
                    "not learned and not NM-ROM"
                ),
            },
            "N": N,
            "conditions": [
                {"fom_tau": tau, "linear_tol": linear_tol}
                for tau, linear_tol in CONDITIONS
            ],
            "test_seed": TEST_SEED,
            "canonical_draw_count": DRAW_COUNT,
            "selected_test_indices": list(TRAJECTORY_INDICES),
            "pair_blocks_per_comparison": PAIR_BLOCKS,
            "repetitions_per_arm_per_trajectory_per_comparison": 2 * PAIR_BLOCKS,
            "pair_schedule": [
                ["cubic", "film_nmrom"],
                ["film_nmrom", "cubic"],
                ["dynamic", "film_nmrom"],
                ["film_nmrom", "dynamic"],
            ],
            "burn_before_every_pair": True,
            "burn_seconds": BURN_S,
            "variant": VARIANT,
            "weak_test_modes": 64,
            "eq_points": 256,
            "eq_fit_snapshots": "decoder-output snapshots",
            "exact_upwind_weak_contract": True,
            "latent_history_mode": LATENT_HISTORY_MODE,
            "max_step_jacobians": MAX_STEP_JACOBIANS,
            "decode_resolution": DECODE_RESOLUTION,
            "decode_chunk": DECODE_CHUNK,
            "charged_exact_fom_residual_guard_vs_live_cubic": True,
            "same_invocation_nmrom_construction_finish_accuracy_work_residual": True,
            "preconditioner": "exact target-grid Helmholtz for every FOM finish",
            "timing_scope": (
                "warmed compiled online trajectory solve; includes complete FiLM "
                "construction/guard/FOM finish or complete classical construction/finish; "
                "excludes compilation, checkpoint loading, EQ fitting, data/reference generation"
            ),
            "timing_estimator": (
                "paired median within trajectory, then median four trajectories"
            ),
            "confidence_interval": (
                "20000-draw trajectory-cluster bootstrap of per-trajectory paired medians"
            ),
            "outliers": "Tukey 1.5-IQR within trajectory; counted and retained",
            "checkpoint_basename": os.path.basename(CHECKPOINT),
            "checkpoint_sha256": film.checkpoint_sha256,
            "checkpoint_config": film.checkpoint_config,
            "smoke_audit": SMOKE_AUDIT,
            "smoke_audit_sha256": sha256(SMOKE_AUDIT),
            "smoke_peak_fraction": smoke["peak_device_fraction_of_limit"],
            "learned_license_fixed_before_primary_outcome": True,
            "reference_residual_gate": REFERENCE_RESIDUAL_GATE,
            "f64": True,
        },
        "provenance": provenance,
        "reference_health": {},
        "offline": {},
        "rows": [],
        "device_memory_at_start": device_memory(),
        "device_memory_at_end": {},
        "complete": False,
    }
    save(report)

    trajectories = bc.generate_reference(
        N,
        list(TRAJECTORY_INDICES),
        TEST_SEED,
        draw_count=DRAW_COUNT,
    )
    worst_reference = float(max(
        trajectory["max_reference_newton_residual"] for trajectory in trajectories
    ))
    if not np.isfinite(worst_reference) or worst_reference > REFERENCE_RESIDUAL_GATE:
        raise SystemExit(f"N={N}: reference residual {worst_reference:.3e}")
    report["reference_health"][str(N)] = {
        "max_reference_newton_residual": worst_reference
    }
    save(report)

    offline_started = time.time()
    built = film.build(
        N,
        latent_extrapolation_scale=1.0,
        max_step_jacobians=MAX_STEP_JACOBIANS,
    )
    report["offline"] = {
        "build_seconds": float(time.time() - offline_started),
        "eq_info": built["eq_info"],
        "eq_indices": built["eq_indices"].tolist(),
        "eq_weights": built["eq_weights"].tolist(),
        "decode_resolution": built["decode_resolution"],
        "decode_chunk": built["decode_chunk"],
        "feature_sample_resolution": built["feature_sample_resolution"],
        "feature_sample_count": built["feature_sample_count"],
        "feature_sample_indices": built["feature_sample_indices"].tolist(),
        "latent_extrapolation_scale": built["latent_extrapolation_scale"],
        "max_step_jacobians": built["max_step_jacobians"],
        "rollout_attempt_budget": built["rollout_attempt_budget"],
    }
    save(report)

    for condition_index, (fom_tau, linear_tol) in enumerate(CONDITIONS):
        run_condition(
            report,
            trajectories,
            built,
            fom_tau,
            linear_tol,
            condition_index,
        )
        gc.collect()
        jax.clear_caches()

    report["device_memory_at_end"] = device_memory()
    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
