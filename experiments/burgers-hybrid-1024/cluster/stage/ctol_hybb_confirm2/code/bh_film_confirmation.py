"""Audited genuine FiLM NM-ROM negative control through N=1024.

The NM-ROM construction and locked FOM finish are composed into one jitted,
timed invocation.  Linear extrapolation and cubic history use the identical
FOM operator, inner tolerance, and preconditioner and are rotated in the same
post-burn timing block.  Thus an NM-ROM loss is not inferred from timings in a
different job or from separately graded fields.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc
from bh_film_control import FilmControl, VARIANT

OUT = sys.argv[1]
CHECKPOINT = os.environ["FILM_CHECKPOINT"]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256,512,1024").split(",")]
FOM_TAUS = tuple(float(value) for value in os.environ.get(
    "FOM_TAUS", os.environ.get("FOM_TAU", "1e-6")
).split(","))
LINEAR_TOLS = tuple(float(value) for value in os.environ.get(
    "LINEAR_TOLS", os.environ.get("LINEAR_TOL", "1e-2")
).split(","))
if len(FOM_TAUS) != len(LINEAR_TOLS) or not FOM_TAUS:
    raise ValueError("FOM_TAUS and LINEAR_TOLS must have the same non-zero length")
CONDITIONS = tuple(zip(FOM_TAUS, LINEAR_TOLS))
PRECONDITIONER = os.environ.get("PRECONDITIONER", "helmholtz")
DECODE_CHUNK = int(os.environ.get("DECODE_CHUNK", "2"))
DECODE_RESOLUTION = int(os.environ.get("DECODE_RESOLUTION", "64"))
LATENT_HISTORY_MODES = tuple(
    value.strip() for value in
    os.environ.get("LATENT_HISTORY_MODES", "previous").split(",") if value.strip()
)
MAX_STEP_JACOBIANS = tuple(int(value) for value in os.environ.get(
    "MAX_STEP_JACOBIANS", "31"
).split(","))
TEST_SEED = int(os.environ.get("TEST_SEED", "1"))
TEST_DRAW_COUNT = int(os.environ.get("TEST_DRAW_COUNT", "16"))
TEST_START = int(os.environ.get("TEST_START", "0"))
N_TEST_TRAJ = int(os.environ.get("N_TEST_TRAJ", "4"))
TIME_REPS = int(os.environ.get("TIME_REPS", "21"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_S = float(os.environ.get("BURN_S", "3"))
REFERENCE_RESIDUAL_GATE = float(os.environ.get("REFERENCE_RESIDUAL_GATE", "1e-11"))
_HISTORY_SCALES = {"previous": 0.0, "extrapolation": 1.0}
if not LATENT_HISTORY_MODES or any(mode not in _HISTORY_SCALES for mode in LATENT_HISTORY_MODES):
    raise ValueError(f"invalid LATENT_HISTORY_MODES={LATENT_HISTORY_MODES}")
if not MAX_STEP_JACOBIANS or any(value < 1 for value in MAX_STEP_JACOBIANS):
    raise ValueError(f"invalid MAX_STEP_JACOBIANS={MAX_STEP_JACOBIANS}")
FILM_VARIANTS = tuple(
    (mode, max_jacobians)
    for mode in LATENT_HISTORY_MODES
    for max_jacobians in MAX_STEP_JACOBIANS
)
FILM_ARMS = (
    ("film_nmrom",) if len(FILM_VARIANTS) == 1 else
    tuple(
        f"film_nmrom_{mode}_j{max_jacobians}"
        for mode, max_jacobians in FILM_VARIANTS
    )
)
ARMS = ("linear", "cubic") + FILM_ARMS


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def outlier_count(values):
    values = np.asarray(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    return int(np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))


def cluster_bootstrap_ci(case_medians, seed):
    """Bootstrap trajectory clusters after within-trajectory medians."""
    values = np.asarray(case_medians, np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, values.size, size=(10000, values.size))]
    return [float(v) for v in np.quantile(np.median(sampled, axis=1), [0.025, 0.975])]


def per_case_medians(records, values):
    grouped = {}
    for record, value in zip(records, values):
        grouped.setdefault(record["trajectory_index"], []).append(value)
    expected = {record["trajectory_index"] for record in records}
    if set(grouped) != expected or any(len(group) != TIME_REPS for group in grouped.values()):
        raise ValueError("timing records are not balanced by trajectory")
    return {
        str(index): float(np.median(grouped[index])) for index in sorted(grouped)
    }


def per_case_outliers(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["trajectory_index"], []).append(record["elapsed_s"])
    counts = {str(index): outlier_count(grouped[index]) for index in sorted(grouped)}
    return counts, int(sum(counts.values()))


def grade_fom(fom_output, trajectory, elapsed):
    values = [np.asarray(value) for value in fom_output]
    U, newton, linear, breakdowns, flags, rel_res = values[:6]
    target = trajectory["U"][1:]
    grade = {
        "elapsed_s": float(elapsed),
        "trajectory_rel_l2": float(np.linalg.norm(U - target) / np.linalg.norm(target)),
        "max_final_rel_residual": float(np.max(rel_res)),
        "newton_total": int(np.sum(newton)),
        "linear_total": int(np.sum(linear)),
        "newton_per_step": newton.tolist(),
        "linear_per_step": linear.tolist(),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "finite": bool(np.all(np.isfinite(U)) and np.all(np.isfinite(rel_res))),
    }
    if len(values) == 7:
        guard = values[6]
        grade.update({
            "film_guard_accepted_count": int(np.sum(guard)),
            "film_guard_accepted_fraction": float(np.mean(guard)),
            "film_guard_accepted_per_step": guard.tolist(),
        })
    return grade


def run_condition(
    report,
    n,
    trajectories,
    worst_reference,
    built_by_arm,
    fom_tau,
    linear_tol,
    condition_index,
):
    """Run one locked tolerance with every arm in one rotated timing block."""
    chain, _ = bc.make_chain(
        n,
        fom_tau,
        lin_tol=linear_tol,
        preconditioner=PRECONDITIONER,
    )
    guarded_film_chain, _ = bc.make_chain(
        n,
        fom_tau,
        lin_tol=linear_tol,
        preconditioner=PRECONDITIONER,
        return_guard=True,
    )
    dummy = jnp.zeros((bc.T, n * n), jnp.float64)

    end_to_end_by_arm = {}
    for arm in FILM_ARMS:
        construct = built_by_arm[arm]["construct"]

        @jax.jit
        def film_end_to_end(u0, nu, construct_fn=construct):
            construction = construct_fn(u0, nu)
            guesses = construction[0]
            fom = guarded_film_chain(u0, nu, guesses, jnp.int32(8))
            return fom, construction

        end_to_end_by_arm[arm] = film_end_to_end

    calls = {}
    for trajectory in trajectories:
        u0 = jnp.asarray(trajectory["U"][0])
        nu = trajectory["nu"]
        calls[("linear", trajectory["index"])] = (
            lambda u0_value=u0, nu_value=nu: chain(
                u0_value, nu_value, dummy, jnp.int32(1)
            )
        )
        calls[("cubic", trajectory["index"])] = (
            lambda u0_value=u0, nu_value=nu: chain(
                u0_value, nu_value, dummy, jnp.int32(5)
            )
        )
        for arm in FILM_ARMS:
            end_to_end = end_to_end_by_arm[arm]
            calls[(arm, trajectory["index"])] = (
                lambda u0_value=u0, nu_value=nu, fn=end_to_end: fn(
                    u0_value, nu_value
                )
            )

    for call in calls.values():
        jax.block_until_ready(call())
    burn_count = bc.gpu_burn(
        lambda: jax.block_until_ready(
            calls[("linear", trajectories[0]["index"])]()
        ),
        BURN_S,
    )
    for warm in range(TIME_WARM):
        order = ARMS[warm % len(ARMS):] + ARMS[:warm % len(ARMS)]
        for trajectory in trajectories:
            for arm in order:
                jax.block_until_ready(calls[(arm, trajectory["index"])]())

    records = {arm: [] for arm in ARMS}
    timing_orders = []
    for repetition in range(TIME_REPS):
        offset = repetition % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        if (repetition // len(ARMS)) % 2:
            order = tuple(reversed(order))
        timing_orders.append(list(order))
        trajectory_order = (
            trajectories[repetition % N_TEST_TRAJ:]
            + trajectories[:repetition % N_TEST_TRAJ]
        )
        for trajectory in trajectory_order:
            for arm in order:
                start = time.perf_counter()
                output = calls[(arm, trajectory["index"])]()
                jax.block_until_ready(output)
                elapsed = time.perf_counter() - start
                if arm in FILM_ARMS:
                    fom_output, construction = output
                else:
                    fom_output, construction = output, None
                record = grade_fom(fom_output, trajectory, elapsed)
                record.update(
                    trajectory_index=trajectory["index"], repetition=repetition
                )
                if construction is not None:
                    (
                        guesses,
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
                    target = trajectory["U"][1:]
                    record.update({
                        "rom_guess_trajectory_rel_l2": float(
                            np.linalg.norm(guesses - target) / np.linalg.norm(target)
                        ),
                        "rom_guess_step_rel_l2": (
                            np.linalg.norm(guesses - target, axis=1)
                            / np.maximum(np.linalg.norm(target, axis=1), 1e-300)
                        ).tolist(),
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
                        "reduced_tol_at_init_steps": int(np.sum(reasons == 4)),
                        "field_features_from_u0_and_nu": field_features.tolist(),
                    })
                if (
                    not record["finite"]
                    or record["breakdowns"]
                    or record["flags_nonzero"]
                    or record["max_final_rel_residual"] > fom_tau
                ):
                    raise SystemExit(
                        f"N={n} tau={fom_tau:.0e} {arm}: "
                        f"unhealthy timed solve {record}"
                    )
                records[arm].append(record)

    linear_pairs = {
        (record["trajectory_index"], record["repetition"]): record["elapsed_s"]
        for record in records["linear"]
    }
    cubic_pairs = {
        (record["trajectory_index"], record["repetition"]): record["elapsed_s"]
        for record in records["cubic"]
    }
    for arm_index, arm in enumerate(ARMS):
        elapsed = [record["elapsed_s"] for record in records[arm]]
        delta_linear = [
            linear_pairs[(record["trajectory_index"], record["repetition"])]
            - record["elapsed_s"]
            for record in records[arm]
        ]
        delta_cubic = [
            cubic_pairs[(record["trajectory_index"], record["repetition"])]
            - record["elapsed_s"]
            for record in records[arm]
        ]
        timing_case_medians = per_case_medians(records[arm], elapsed)
        linear_case_medians = per_case_medians(records[arm], delta_linear)
        cubic_case_medians = per_case_medians(records[arm], delta_cubic)
        newton_case_medians = per_case_medians(
            records[arm], [record["newton_total"] for record in records[arm]]
        )
        linear_work_case_medians = per_case_medians(
            records[arm], [record["linear_total"] for record in records[arm]]
        )
        outliers_by_case, outlier_total = per_case_outliers(records[arm])
        bootstrap_seed = condition_index * 100000
        row = {
            "N": n,
            "n_dof": n * n,
            "fom_tau": fom_tau,
            "linear_tol": linear_tol,
            "arm": arm,
            "reference_max_residual": worst_reference,
            "timed_records": records[arm],
            "timing_repetitions_s": elapsed,
            "timing_shape": [N_TEST_TRAJ, TIME_REPS],
            "timing_case_medians_s": timing_case_medians,
            "timing_median_ms": float(
                np.median(list(timing_case_medians.values())) * 1e3
            ),
            "timing_outlier_count_tukey_by_case": outliers_by_case,
            "timing_outlier_count_tukey": outlier_total,
            "paired_saving_vs_linear_s": delta_linear,
            "paired_saving_vs_linear_case_medians_s": linear_case_medians,
            "paired_saving_vs_linear_median_ms": float(
                np.median(list(linear_case_medians.values())) * 1e3
            ),
            "paired_saving_vs_linear_median_95ci_ms": [
                1e3 * value for value in cluster_bootstrap_ci(
                    list(linear_case_medians.values()),
                    20260819 + n + arm_index + bootstrap_seed,
                )
            ],
            "paired_saving_vs_cubic_s": delta_cubic,
            "paired_saving_vs_cubic_case_medians_s": cubic_case_medians,
            "paired_saving_vs_cubic_median_ms": float(
                np.median(list(cubic_case_medians.values())) * 1e3
            ),
            "paired_saving_vs_cubic_median_95ci_ms": [
                1e3 * value for value in cluster_bootstrap_ci(
                    list(cubic_case_medians.values()),
                    20260829 + n + arm_index + bootstrap_seed,
                )
            ],
            "newton_total_case_medians": newton_case_medians,
            "newton_total_median": float(np.median(list(
                newton_case_medians.values()
            ))),
            "linear_total_case_medians": linear_work_case_medians,
            "linear_total_median": float(np.median(list(
                linear_work_case_medians.values()
            ))),
            "max_timed_outer_residual": float(max(
                record["max_final_rel_residual"] for record in records[arm]
            )),
            "max_timed_trajectory_rel_l2": float(max(
                record["trajectory_rel_l2"] for record in records[arm]
            )),
            "gpu_burn_iterations": burn_count,
            "paired_timing_orders": timing_orders,
        }
        if arm in FILM_ARMS:
            row.update({
                "film_guard_accepted_fraction_median": float(np.median([
                    record["film_guard_accepted_fraction"]
                    for record in records[arm]
                ])),
                "film_guard_accepted_count_median": float(np.median([
                    record["film_guard_accepted_count"]
                    for record in records[arm]
                ])),
                "reduced_jacobians_total_median": float(np.median([
                    record["reduced_jacobians_total"]
                    for record in records[arm]
                ])),
                "reduced_attempts_total_median": float(np.median([
                    record["reduced_attempts_total"]
                    for record in records[arm]
                ])),
                "reduced_tol_at_init_steps_median": float(np.median([
                    record["reduced_tol_at_init_steps"]
                    for record in records[arm]
                ])),
            })
        report["rows"].append(row)
        bc.log(
            f"N={n} tau={fom_tau:.0e} {arm:10s}: "
            f"{row['timing_median_ms']:.2f}ms "
            f"vs cubic={row['paired_saving_vs_cubic_median_ms']:.2f}ms "
            f"Newton={row['newton_total_median']:.0f} "
            f"BiCG={row['linear_total_median']:.0f}"
        )
    report["equivalence"].setdefault(str(n), {})[f"{fom_tau:.0e}"] = (
        bc.reference_equivalence(
            n,
            trajectories,
            chain,
            lin_tol=linear_tol,
            preconditioner=PRECONDITIONER,
        )
    )
    save(report)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing FiLM confirmation")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    film = FilmControl(
        CHECKPOINT,
        decode_chunk=DECODE_CHUNK,
        decode_resolution=DECODE_RESOLUTION,
    )
    report = {
        "config": {
            "purpose": "genuine weak FiLM NM-ROM negative control",
            "classification": {
                "film_nmrom": (
                    "optimized genuine NM-ROM: weak EQ LSPG at every time step; "
                    "fixed-coarse decode and charged fused prolongation"
                ),
                "linear": "classical live-history control",
                "cubic": "classical live-history control",
            },
            "variant": VARIANT,
            "ns": NS,
            "fom_taus": FOM_TAUS,
            "linear_tols": LINEAR_TOLS,
            "locked_conditions": [
                {"fom_tau": tau, "linear_tol": linear_tol}
                for tau, linear_tol in CONDITIONS
            ],
            "preconditioner": PRECONDITIONER,
            "decode_chunk": DECODE_CHUNK,
            "decode_resolution": DECODE_RESOLUTION,
            "latent_history_modes": LATENT_HISTORY_MODES,
            "max_step_jacobians": MAX_STEP_JACOBIANS,
            "film_variants": [
                {"mode": mode, "max_step_jacobians": max_jacobians}
                for mode, max_jacobians in FILM_VARIANTS
            ],
            "exact_upwind_weak_contract": True,
            "charged_exact_fom_residual_guard_vs_live_cubic": True,
            "cold_start_fit_hyper_reduced_on_same_eq_nodes": True,
            "initializer_features_hyper_reduced": (
                "charged fixed endpoint sample of at most 64x64 u0 values"
            ),
            "historical_full_grid_path": (
                "audited 479.569ms N256 construction retained as context only; "
                "not used for optimized timing"
            ),
            "checkpoint_basename": os.path.basename(CHECKPOINT),
            "checkpoint_sha256": film.checkpoint_sha256,
            "checkpoint_config": film.checkpoint_config,
            "test_seed": TEST_SEED,
            "canonical_draw_count": TEST_DRAW_COUNT,
            "selected_test_indices": list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            "time_reps": TIME_REPS,
            "timing_estimator": (
                "median repetitions within each trajectory, then median across "
                "trajectory clusters"
            ),
            "confidence_interval": (
                "10000-draw trajectory-cluster bootstrap of per-trajectory "
                "paired medians; repetitions are not treated as IID"
            ),
            "outlier_method": "Tukey fences within each trajectory only",
            "same_invocation_nmrom_plus_fom_cost_accuracy_work": True,
            "reference_residual_gate": REFERENCE_RESIDUAL_GATE,
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "equivalence": {},
        "offline_per_mesh": {},
        "complete": False,
    }
    save(report)

    for n in NS:
        trajectories = bc.generate_reference(
            n,
            list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            TEST_SEED,
            draw_count=TEST_DRAW_COUNT,
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > REFERENCE_RESIDUAL_GATE:
            raise SystemExit(f"N={n}: reference residual {worst_reference:.3e}")
        offline_start = time.time()
        built_by_arm = {}
        shared_ops = None
        for (mode, max_jacobians), arm in zip(FILM_VARIANTS, FILM_ARMS):
            built_by_arm[arm] = film.build(
                n,
                latent_extrapolation_scale=_HISTORY_SCALES[mode],
                max_step_jacobians=max_jacobians,
                shared_ops=shared_ops,
            )
            shared_ops = built_by_arm[arm]["ops"]
        built = built_by_arm[FILM_ARMS[0]]
        report["offline_per_mesh"][str(n)] = {
            "build_seconds": float(time.time() - offline_start),
            "eq_info": built["eq_info"],
            "eq_indices": built["eq_indices"].tolist(),
            "eq_weights": built["eq_weights"].tolist(),
            "decode_chunk": built["decode_chunk"],
            "decode_resolution": built["decode_resolution"],
            "feature_sample_resolution": built["feature_sample_resolution"],
            "feature_sample_count": built["feature_sample_count"],
            "feature_sample_indices": built["feature_sample_indices"].tolist(),
            "latent_history_variants": {
                arm: {
                    "mode": mode,
                    "scale": built_by_arm[arm]["latent_extrapolation_scale"],
                    "max_step_jacobians": built_by_arm[arm]["max_step_jacobians"],
                    "rollout_attempt_budget": built_by_arm[arm]["rollout_attempt_budget"],
                    "eq_indices_equal_to_first": bool(np.array_equal(
                        built_by_arm[arm]["eq_indices"], built["eq_indices"]
                    )),
                    "eq_weights_equal_to_first": bool(np.array_equal(
                        built_by_arm[arm]["eq_weights"], built["eq_weights"]
                    )),
                }
                for (mode, _), arm in zip(FILM_VARIANTS, FILM_ARMS)
            },
        }
        for condition_index, (fom_tau, linear_tol) in enumerate(CONDITIONS):
            run_condition(
                report,
                n,
                trajectories,
                worst_reference,
                built_by_arm,
                fom_tau,
                linear_tol,
                condition_index,
            )

    report["complete"] = True
    save(report)


if __name__ == "__main__":
    main()
