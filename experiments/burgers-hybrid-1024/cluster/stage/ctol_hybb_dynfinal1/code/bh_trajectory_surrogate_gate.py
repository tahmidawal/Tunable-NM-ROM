"""Development-only quality/cost gate for a trajectory-surrogate control."""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc
import bh_trajectory_surrogate as ts

OUT = sys.argv[1]
NS = [int(value) for value in os.environ.get("NS", "64,256").split(",")]
COARSE_NS = [int(value) for value in os.environ.get("COARSE_NS", "16,24,32,48,64").split(",")]
FOM_TAU = float(os.environ.get("FOM_TAU", "1e-6"))
LIN_TOL = float(os.environ.get("LIN_TOL", "1e-2"))
N_VAL = int(os.environ.get("N_VAL", "4"))
VAL_SEED = int(os.environ.get("VAL_SEED", "20260818"))
VAL_DRAW_COUNT = int(os.environ.get("VAL_DRAW_COUNT", "16"))
VAL_START = int(os.environ.get("VAL_START", "4"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
BURN_S = float(os.environ.get("BURN_S", "3"))
CHECKPOINT = ts.default_checkpoint()


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def cubic_guesses(U):
    return bc.polynomial_guesses(U, 3)


def cluster_bootstrap(values, seed=20260820, draws=20000):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    medians = np.median(samples, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing trajectory gate")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    report = {
        "config": {
            "purpose": "development gate for one-shot time-conditioned trajectory correction",
            "classification": "supervised trajectory surrogate control; not yet a genuine weak NM-ROM",
            "ns": NS,
            "coarse_ns": COARSE_NS,
            "fom_tau": FOM_TAU,
            "lin_tol": LIN_TOL,
            "validation_seed": VAL_SEED,
            "validation_indices": list(range(VAL_START, VAL_START + N_VAL)),
            "validation_draw_count": VAL_DRAW_COUNT,
            "final_seed_touched": False,
            "checkpoint": os.path.basename(CHECKPOINT),
            "checkpoint_sha256": ts.file_sha256(CHECKPOINT),
            "checkpoint_training": "seed 0, trajectories 0:512; selection validation 512:576",
            "online_parameter_inference": "fixed-coarse f64 log-Gaussian fit to u0 plus known nu",
            "all_online_arithmetic": "f64",
            "correction": "temporal differences of decoded states around live cubic FOM history",
            "time_reps": TIME_REPS,
            "burn_seconds": BURN_S,
        },
        "provenance": provenance,
        "rows": [],
        "complete": False,
    }
    save(report)

    for n in NS:
        trajectories = bc.generate_reference(
            n,
            list(range(VAL_START, VAL_START + N_VAL)),
            VAL_SEED,
            draw_count=VAL_DRAW_COUNT,
        )
        if max(item["max_reference_newton_residual"] for item in trajectories) > 1e-8:
            raise SystemExit(f"N={n}: reference health gate failed")
        base_chain, residual = bc.make_chain(
            n, FOM_TAU, lin_tol=LIN_TOL, preconditioner="helmholtz"
        )
        dummy = jnp.zeros((bc.T, n * n), jnp.float64)
        calls = {}
        metadata = {}

        def base_call(trajectory):
            return base_chain(
                jnp.asarray(trajectory["U"][0]), trajectory["nu"], dummy, jnp.int32(5)
            )

        calls["cubic"] = [lambda tr=tr: base_call(tr) for tr in trajectories]
        for coarse_n in COARSE_NS:
            if coarse_n > n:
                continue
            constructor = ts.make_constructor(n, coarse_n, CHECKPOINT)
            chain, _ = bc.make_chain(
                n, FOM_TAU, lin_tol=LIN_TOL, preconditioner="helmholtz"
            )

            def solve(u0, nu, ctor=constructor, chain_fn=chain):
                # Keep construction and finishing as separately compiled
                # kernels.  The Python call is still the single timed solver
                # invocation, but avoids recompiling the large FOM scan for
                # every coarse decoder resolution.
                corrections, latent = ctor(u0, nu)
                outputs = chain_fn(u0, nu, corrections, jnp.int32(9))
                return (*outputs, corrections, latent)

            key = f"trajectory_q{coarse_n}"
            calls[key] = [
                lambda tr=tr, solve_fn=solve: solve_fn(
                    jnp.asarray(tr["U"][0]), tr["nu"]
                )
                for tr in trajectories
            ]
            per_trajectory = []
            for trajectory, call in zip(trajectories, calls[key]):
                outputs = call()
                U, newton, linear, breakdowns, flags, rel_res, corrections, latent = [
                    np.asarray(value) for value in outputs
                ]
                if int(np.sum(breakdowns)) or int(np.sum(flags != 0)):
                    raise SystemExit(f"N={n} {key}: solver health failure")
                if not np.all(np.isfinite(rel_res)) or np.max(rel_res) > FOM_TAU:
                    raise SystemExit(f"N={n} {key}: residual health failure")
                truth = trajectory["U"]
                exact_corrections = truth[1:] - cubic_guesses(truth)
                correction_remaining = np.linalg.norm(
                    np.asarray(corrections) - exact_corrections
                ) / np.linalg.norm(exact_corrections)
                guesses = cubic_guesses(truth) + np.asarray(corrections)
                guess_l2, guess_residual = bc.guess_diagnostics(
                    truth, guesses, residual, trajectory["nu"]
                )
                recovered = np.asarray(latent)
                p = trajectory["parameters"]
                exact_latent = np.asarray(
                    [
                        (p["cx"] - 0.5) / 0.35,
                        (p["cy"] - 0.5) / 0.35,
                        (p["width"] - 0.125) / 0.075,
                        (p["amplitude"] - 1.25) / 0.75,
                        (np.log(p["nu"]) - np.log(np.sqrt(0.001))) / (0.5 * np.log(10.0)),
                    ]
                )
                per_trajectory.append(
                    {
                        "trajectory_index": trajectory["index"],
                        "correction_remaining_ratio": float(correction_remaining),
                        "guess_rel_l2_mean": float(np.mean(guess_l2)),
                        "guess_rel_residual_mean": float(np.mean(guess_residual)),
                        "newton_total": int(np.sum(newton)),
                        "linear_total": int(np.sum(linear)),
                        "max_returned_residual": float(np.max(rel_res)),
                        "trajectory_rel_l2": float(
                            np.linalg.norm(U - truth[1:]) / np.linalg.norm(truth[1:])
                        ),
                        "latent_max_abs_error": float(np.max(np.abs(recovered - exact_latent))),
                    }
                )
            metadata[key] = per_trajectory

        # Compile all candidates, burn the baseline, then rotate methods within
        # each trajectory.  A second burn precedes the reversed half.
        for method_calls in calls.values():
            for call in method_calls:
                jax.tree_util.tree_map(
                    lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
                    call(),
                )
        burn_iterations = bc.gpu_burn(
            lambda: calls["cubic"][0]()[0].block_until_ready(), BURN_S
        )
        samples = {key: [[] for _ in trajectories] for key in calls}
        orders = []
        keys = list(calls)
        for repetition in range(TIME_REPS):
            if repetition == (TIME_REPS + 1) // 2:
                bc.gpu_burn(lambda: calls["cubic"][0]()[0].block_until_ready(), BURN_S)
            for trajectory_index in range(len(trajectories)):
                offset = (repetition + trajectory_index) % len(keys)
                order = keys[offset:] + keys[:offset]
                if repetition >= (TIME_REPS + 1) // 2:
                    order = list(reversed(order))
                orders.append(
                    {"repetition": repetition, "trajectory_slot": trajectory_index, "order": order}
                )
                for key in order:
                    started = time.perf_counter()
                    calls[key][trajectory_index]()[0].block_until_ready()
                    samples[key][trajectory_index].append(float(time.perf_counter() - started))

        cubic_case_medians = np.asarray(
            [np.median(values) for values in samples["cubic"]], np.float64
        )
        cubic_outputs = []
        for trajectory, call in zip(trajectories, calls["cubic"]):
            U, newton, linear, breakdowns, flags, rel_res = [np.asarray(x) for x in call()]
            cubic_outputs.append(
                {
                    "trajectory_index": trajectory["index"],
                    "newton_total": int(np.sum(newton)),
                    "linear_total": int(np.sum(linear)),
                    "max_returned_residual": float(np.max(rel_res)),
                    "trajectory_rel_l2": float(
                        np.linalg.norm(U - trajectory["U"][1:]) / np.linalg.norm(trajectory["U"][1:])
                    ),
                    "flags": int(np.sum(flags != 0)),
                    "breakdowns": int(np.sum(breakdowns)),
                }
            )
        for key in keys:
            case_medians = np.asarray([np.median(values) for values in samples[key]])
            savings = cubic_case_medians - case_medians
            per_trajectory = cubic_outputs if key == "cubic" else metadata[key]
            row = {
                "N": n,
                "arm": key,
                "coarse_n": None if key == "cubic" else int(key.split("q")[1]),
                "case_timing_repetitions_s": samples[key],
                "trajectory_case_medians_s": case_medians.tolist(),
                "timing_median_ms": float(np.median(case_medians) * 1e3),
                "paired_saving_vs_cubic_case_medians_ms": (savings * 1e3).tolist(),
                "paired_saving_vs_cubic_median_ms": float(np.median(savings) * 1e3),
                "paired_saving_vs_cubic_95ci_ms": cluster_bootstrap(savings * 1e3),
                "newton_total_median": float(np.median([x["newton_total"] for x in per_trajectory])),
                "linear_total_median": float(np.median([x["linear_total"] for x in per_trajectory])),
                "per_trajectory": per_trajectory,
                "gpu_burn_iterations": burn_iterations,
                "timing_orders": orders,
            }
            if key != "cubic":
                row.update(
                    correction_remaining_median=float(np.median([
                        x["correction_remaining_ratio"] for x in per_trajectory
                    ])),
                    guess_rel_l2_mean=float(np.mean([
                        x["guess_rel_l2_mean"] for x in per_trajectory
                    ])),
                    guess_rel_residual_mean=float(np.mean([
                        x["guess_rel_residual_mean"] for x in per_trajectory
                    ])),
                    latent_max_abs_error=float(max([
                        x["latent_max_abs_error"] for x in per_trajectory
                    ])),
                )
            report["rows"].append(row)
            bc.log(
                f"N={n} {key}: {row['timing_median_ms']:.3f} ms, "
                f"save/cubic={row['paired_saving_vs_cubic_median_ms']:.3f} ms, "
                f"Newton/BiCG={row['newton_total_median']:.1f}/{row['linear_total_median']:.1f}, "
                f"corr={row.get('correction_remaining_median', float('nan')):.3f}"
            )
        save(report)

    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
