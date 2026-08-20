"""Development timing/work gate for fixed-coarse residual corrections."""
from __future__ import annotations

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
NS = [int(value) for value in os.environ.get("NS", "64,256").split(",")]
COARSE_NS = [int(value) for value in os.environ.get("COARSE_NS", "16,32,64").split(",")]
RELAXATIONS = [float(value) for value in os.environ.get("RELAXATIONS", "0.5,0.75,1.0").split(",")]
FOM_TAU = float(os.environ.get("FOM_TAU", "1e-6"))
LIN_TOL = float(os.environ.get("LIN_TOL", "1e-2"))
N_CASES = int(os.environ.get("N_CASES", "4"))
CASE_START = int(os.environ.get("CASE_START", "0"))
DEV_SEED = int(os.environ.get("DEV_SEED", "20260818"))
DRAW_COUNT = int(os.environ.get("DRAW_COUNT", "16"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
BURN_S = float(os.environ.get("BURN_S", "3"))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("matmul precision must be highest")
    report = {
        "config": {
            "purpose": "development gate for fixed-coarse dynamic residual correction",
            "classification": "classical reduced correction control; not a learned NM-ROM",
            "ns": NS,
            "coarse_ns": COARSE_NS,
            "relaxations": RELAXATIONS,
            "fom_tau": FOM_TAU,
            "lin_tol": LIN_TOL,
            "preconditioner": "exact target-grid Helmholtz FOM; one exact coarse Helmholtz correction",
            "history": "live cubic",
            "development_seed": DEV_SEED,
            "development_indices": list(range(CASE_START, CASE_START + N_CASES)),
            "draw_count": DRAW_COUNT,
            "confirmation_touched": False,
            "time_reps": TIME_REPS,
            "burn_seconds": BURN_S,
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "complete": False,
    }
    save(report)

    for n in NS:
        trajectories = bc.generate_reference(
            n,
            list(range(CASE_START, CASE_START + N_CASES)),
            DEV_SEED,
            draw_count=DRAW_COUNT,
        )
        dummy = jnp.zeros((bc.T, n * n), jnp.float64)
        calls = {}
        base_chain, residual = bc.make_chain(
            n, FOM_TAU, lin_tol=LIN_TOL, preconditioner="helmholtz"
        )
        calls["cubic"] = [
            lambda tr=tr: base_chain(
                jnp.asarray(tr["U"][0]), tr["nu"], dummy, jnp.int32(5)
            )
            for tr in trajectories
        ]
        for coarse_n in COARSE_NS:
            if coarse_n > n:
                continue
            for relaxation in RELAXATIONS:
                key = f"coarse{coarse_n}_a{relaxation:g}"
                predictor = dc.make_predictor(n, coarse_n, relaxation)
                chain, _ = bc.make_chain(
                    n,
                    FOM_TAU,
                    predictor=predictor,
                    lin_tol=LIN_TOL,
                    preconditioner="helmholtz",
                )
                calls[key] = [
                    lambda tr=tr, chain_fn=chain: chain_fn(
                        jnp.asarray(tr["U"][0]), tr["nu"], dummy, jnp.int32(3)
                    )
                    for tr in trajectories
                ]

        # Compile, then burn/reburn before each AB/BA repetition.  Candidate
        # count is intentionally small; this is a work/cost gate, not final
        # inference.
        for method_calls in calls.values():
            for call in method_calls:
                call()[0].block_until_ready()
        samples = {key: [[] for _ in trajectories] for key in calls}
        records = {key: [] for key in calls}
        orders = []
        keys = list(calls)
        burn_iterations = 0
        for repetition in range(TIME_REPS):
            burn_iterations += bc.gpu_burn(
                lambda: calls["cubic"][0]()[0].block_until_ready(), BURN_S
            )
            for trajectory_slot, trajectory in enumerate(trajectories):
                offset = (repetition + trajectory_slot) % len(keys)
                order = keys[offset:] + keys[:offset]
                if repetition % 2:
                    order = list(reversed(order))
                orders.append(
                    {"repetition": repetition, "trajectory_slot": trajectory_slot, "order": order}
                )
                for key in order:
                    started = time.perf_counter()
                    outputs = calls[key][trajectory_slot]()
                    outputs[0].block_until_ready()
                    elapsed = float(time.perf_counter() - started)
                    U, newton, linear, breakdowns, flags, rel_res = [
                        np.asarray(value) for value in outputs
                    ]
                    record = {
                        "trajectory_index": trajectory["index"],
                        "repetition": repetition,
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
                        raise SystemExit(f"N={n} {key}: solver health failure {record}")
                    samples[key][trajectory_slot].append(elapsed)
                    records[key].append(record)
        cubic_case = np.asarray([np.median(values) for values in samples["cubic"]])
        for key in keys:
            case = np.asarray([np.median(values) for values in samples[key]])
            savings = cubic_case - case
            row = {
                "N": n,
                "arm": key,
                "coarse_n": None if key == "cubic" else int(key.split("_")[0][6:]),
                "relaxation": None if key == "cubic" else float(key.rsplit("_a", 1)[1]),
                "timed_records": records[key],
                "case_timing_repetitions_s": samples[key],
                "case_medians_s": case.tolist(),
                "timing_median_ms": float(np.median(case) * 1e3),
                "saving_vs_cubic_case_medians_ms": (savings * 1e3).tolist(),
                "saving_vs_cubic_median_ms": float(np.median(savings) * 1e3),
                "newton_total_median": float(np.median([r["newton_total"] for r in records[key]])),
                "linear_total_median": float(np.median([r["linear_total"] for r in records[key]])),
                "max_returned_residual": float(max(r["max_returned_residual"] for r in records[key])),
                "max_trajectory_rel_l2": float(max(r["trajectory_rel_l2"] for r in records[key])),
                "timing_orders": orders,
                "gpu_burn_iterations": burn_iterations,
            }
            report["rows"].append(row)
            bc.log(
                f"N={n} {key}: {row['timing_median_ms']:.3f} ms "
                f"save={row['saving_vs_cubic_median_ms']:.3f} ms "
                f"N/B={row['newton_total_median']:.1f}/{row['linear_total_median']:.1f}"
            )
        save(report)
    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
