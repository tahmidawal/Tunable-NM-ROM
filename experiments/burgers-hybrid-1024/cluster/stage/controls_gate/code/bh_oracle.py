"""Warm-start oracle and break-even diagnostic for Burgers-2D.

Oracle arms interpolate from linear extrapolation to the already-computed exact
next FOM state. They are deliberately non-deployable and are used only to answer:
how accurate must a cheap predictor be, and how much finishing cost can it save?
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

OUT = sys.argv[1]
NS = [int(value) for value in os.environ.get("NS", "32,64,128,256").split(",")]
FOM_TAUS = [float(value) for value in os.environ.get("FOM_TAUS", "1e-6").split(",")]
QUALITY = [float(value) for value in os.environ.get(
    "QUALITY", "0.0,0.1,0.25,0.5,0.75,0.9,0.99,1.0"
).split(",")]
N_TEST_TRAJ = int(os.environ.get("N_TEST_TRAJ", "4"))
TEST_START = int(os.environ.get("TEST_START", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", str(bc.bf.SEED + 1)))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_S = float(os.environ.get("BURN_S", "3"))
RUN_ROLE = os.environ.get("RUN_ROLE", "panel")
REFERENCE_RESIDUAL_GATE = float(os.environ.get("REFERENCE_RESIDUAL_GATE", "1e-8"))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def collect(chain, u0, nu, guesses, mode):
    U, newton, linear, breakdowns, flags, rel_res = chain(
        jnp.asarray(u0), nu, jnp.asarray(guesses), jnp.int32(mode)
    )
    return {
        "U": np.asarray(U),
        "newton": np.asarray(newton),
        "linear": np.asarray(linear),
        "breakdowns": np.asarray(breakdowns),
        "flags": np.asarray(flags),
        "rel_res": np.asarray(rel_res),
    }


def validate_arm(arm, tau, label):
    if not np.all(np.isfinite(arm["rel_res"])):
        raise SystemExit(f"{label}: non-finite Newton residual")
    if float(np.max(arm["rel_res"])) > tau:
        raise SystemExit(
            f"{label}: max relative Newton residual {np.max(arm['rel_res']):.3e} "
            f"exceeds tau={tau:.1e}"
        )
    if int(np.sum(arm["breakdowns"])) or int(np.sum(arm["flags"] != 0)):
        raise SystemExit(
            f"{label}: {np.sum(arm['breakdowns'])} linear failures and "
            f"{np.sum(arm['flags'] != 0)} nonzero Newton flags"
        )


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing to produce results")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    report = {
        "config": {
            "pde": "burgers2d",
            "dt": bc.bf.DT,
            "num_steps": bc.T,
            "ns": NS,
            "fom_taus": FOM_TAUS,
            "quality": QUALITY,
            "quality_definition": "g_q = extrap + q * (exact_next - extrap)",
            "oracle_non_deployable": True,
            "n_test_trajectories": N_TEST_TRAJ,
            "test_indices": list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            "test_seed": TEST_SEED,
            "time_reps": TIME_REPS,
            "time_warm": TIME_WARM,
            "burn_seconds": BURN_S,
            "run_role": RUN_ROLE,
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "complete": False,
    }
    save(report)
    bc.log(
        f"backend={jax.default_backend()} gpu={provenance['gpu_kind']} NS={NS} "
        f"taus={FOM_TAUS} quality={QUALITY} ntraj={N_TEST_TRAJ}"
    )

    for n in NS:
        started = time.time()
        trajectories = bc.generate_reference(
            n, list(range(TEST_START, TEST_START + N_TEST_TRAJ)), TEST_SEED
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > REFERENCE_RESIDUAL_GATE:
            raise SystemExit(
                f"N={n}: reference FOM residual {worst_reference:.3e} exceeds gate"
            )
        bc.log(f"N={n}: regenerated references, max residual={worst_reference:.2e}")

        for tau in FOM_TAUS:
            chain, residual = bc.make_chain(n, tau)
            dummy = np.zeros((bc.T, n * n), dtype=np.float64)
            # Compile and burn on the weakest baseline before timing any candidate.
            chain(
                jnp.asarray(trajectories[0]["U"][0]),
                trajectories[0]["nu"],
                jnp.asarray(dummy),
                jnp.int32(0),
            )[0].block_until_ready()
            burn_count = bc.gpu_burn(
                lambda: chain(
                    jnp.asarray(trajectories[0]["U"][0]),
                    trajectories[0]["nu"],
                    jnp.asarray(dummy),
                    jnp.int32(0),
                )[0].block_until_ready(),
                BURN_S,
            )
            candidate_specs = [("prev", None), ("extrap", 0.0)] + [
                ("oracle", q) for q in QUALITY if q != 0.0
            ]
            # q=0 is exactly extrap and is represented by the traced extrap arm.
            arm_data = {}
            timed_calls = {}
            for name, q in candidate_specs:
                per_trajectory = []
                for trajectory in trajectories:
                    U = trajectory["U"]
                    extrap = bc.extrapolated_guesses(U)
                    if name == "prev":
                        guesses, mode = dummy, 0
                    elif name == "extrap":
                        guesses, mode = dummy, 1
                    else:
                        guesses = extrap + q * (U[1:] - extrap)
                        mode = 2
                    result = collect(
                        chain, U[0], trajectory["nu"], guesses, mode
                    )
                    validate_arm(result, tau, f"N={n} tau={tau} {name}:{q}")
                    if name == "prev":
                        diagnostic_guesses = U[:-1]
                    elif name == "extrap":
                        diagnostic_guesses = extrap
                    else:
                        diagnostic_guesses = guesses
                    guess_error, guess_residual = bc.guess_diagnostics(
                        U, diagnostic_guesses, residual, trajectory["nu"]
                    )
                    final_error = float(
                        np.linalg.norm(result["U"] - U[1:])
                        / np.linalg.norm(U[1:])
                    )
                    per_trajectory.append(
                        {
                            "trajectory_index": trajectory["index"],
                            "parameters": trajectory["parameters"],
                            "newton_total": int(np.sum(result["newton"])),
                            "linear_total": int(np.sum(result["linear"])),
                            "newton_per_step": result["newton"].tolist(),
                            "linear_per_step": result["linear"].tolist(),
                            "max_final_rel_residual": float(np.max(result["rel_res"])),
                            "final_trajectory_rel_l2": final_error,
                            "guess_rel_l2_mean": float(np.mean(guess_error)),
                            "guess_rel_l2_max": float(np.max(guess_error)),
                            "guess_rel_residual_mean": float(np.mean(guess_residual)),
                            "guess_rel_residual_max": float(np.max(guess_residual)),
                        }
                    )

                first = trajectories[0]
                extrap0 = bc.extrapolated_guesses(first["U"])
                if name == "prev":
                    timed_guesses, timed_mode = dummy, 0
                elif name == "extrap":
                    timed_guesses, timed_mode = dummy, 1
                else:
                    timed_guesses = extrap0 + q * (first["U"][1:] - extrap0)
                    timed_mode = 2
                key = name if q is None else f"{name}:q={q:g}"
                timed_guesses_j = jnp.asarray(timed_guesses)
                timed_mode_j = jnp.int32(timed_mode)
                u0_timed_j = jnp.asarray(first["U"][0])
                timed_calls[key] = lambda u0=u0_timed_j, nu=first["nu"], \
                    guesses=timed_guesses_j, mode=timed_mode_j: chain(
                        u0,
                        nu,
                        guesses,
                        mode,
                    )[0].block_until_ready()
                arm_data[key] = {
                    "name": name,
                    "quality": q,
                    "oracle_non_deployable": name == "oracle",
                    "newton_total_mean": float(
                        np.mean([p["newton_total"] for p in per_trajectory])
                    ),
                    "linear_total_mean": float(
                        np.mean([p["linear_total"] for p in per_trajectory])
                    ),
                    "guess_rel_l2_mean": float(
                        np.mean([p["guess_rel_l2_mean"] for p in per_trajectory])
                    ),
                    "guess_rel_residual_mean": float(
                        np.mean([p["guess_rel_residual_mean"] for p in per_trajectory])
                    ),
                    "per_trajectory": per_trajectory,
                }

            # Pair/interleave every arm inside each repetition. Rotating the order
            # makes each candidate appear at every clock position across the block,
            # preventing the sequential prev->extrap->oracle order from turning
            # residual device-clock drift into an apparent warm-start benefit.
            timing_keys = list(timed_calls)
            for warm_index in range(TIME_WARM):
                offset = warm_index % len(timing_keys)
                for key in timing_keys[offset:] + timing_keys[:offset]:
                    timed_calls[key]()
            timing_samples = {key: [] for key in timing_keys}
            timing_orders = []
            for repetition in range(TIME_REPS):
                offset = repetition % len(timing_keys)
                order = timing_keys[offset:] + timing_keys[:offset]
                if (repetition // len(timing_keys)) % 2:
                    order = list(reversed(order))
                timing_orders.append(order)
                for key in order:
                    start_timing = time.perf_counter()
                    timed_calls[key]()
                    timing_samples[key].append(float(time.perf_counter() - start_timing))
            for key in timing_keys:
                samples = timing_samples[key]
                arm_data[key]["fom_finish_ms"] = float(np.median(samples) * 1e3)
                arm_data[key]["fom_finish_repetitions_s"] = samples
                arm_data[key]["paired_timing_orders"] = timing_orders

            extrap_time = arm_data["extrap:q=0"]["fom_finish_ms"]
            prev_time = arm_data["prev"]["fom_finish_ms"]
            extrap_samples = np.asarray(
                arm_data["extrap:q=0"]["fom_finish_repetitions_s"]
            )
            prev_samples = np.asarray(arm_data["prev"]["fom_finish_repetitions_s"])
            for key, arm in arm_data.items():
                samples = np.asarray(arm["fom_finish_repetitions_s"])
                arm["paired_delta_vs_extrap_s"] = (extrap_samples - samples).tolist()
                arm["paired_delta_vs_prev_s"] = (prev_samples - samples).tolist()
                arm["paired_delta_vs_extrap_median_ms"] = float(
                    np.median(extrap_samples - samples) * 1e3
                )
                arm["max_construction_cost_to_beat_extrap_ms"] = (
                    extrap_time - arm["fom_finish_ms"]
                )
                arm["max_construction_cost_to_beat_prev_ms"] = (
                    prev_time - arm["fom_finish_ms"]
                )
                report["rows"].append(
                    {
                        "N": n,
                        "n_dof": n * n,
                        "fom_tau": tau,
                        "run_role": RUN_ROLE,
                        "reference_max_newton_residual": worst_reference,
                        "gpu_burn_iterations": burn_count,
                        "candidate_key": key,
                        **arm,
                    }
                )
                bc.log(
                    f"  tau={tau:.0e} {key:16s}: "
                    f"Newton={arm['newton_total_mean']:.1f} "
                    f"BiCG={arm['linear_total_mean']:.0f} "
                    f"time={arm['fom_finish_ms']:.2f}ms "
                    f"budget_vs_extrap={arm['max_construction_cost_to_beat_extrap_ms']:.2f}ms"
                )
            save(report)
        bc.log(f"N={n} complete in {time.time() - started:.1f}s")

    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
