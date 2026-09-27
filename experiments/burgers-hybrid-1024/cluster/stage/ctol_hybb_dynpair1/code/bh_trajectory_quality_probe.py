"""Cheap train/validation-only quality probe before any timed FOM gate."""
from __future__ import annotations

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc
import bh_trajectory_surrogate as ts

OUT = sys.argv[1]
NS = [int(value) for value in os.environ.get("NS", "64,256").split(",")]
COARSE_NS = [int(value) for value in os.environ.get("COARSE_NS", "32,48,64").split(",")]
N_VAL = int(os.environ.get("N_VAL", "4"))
VAL_START = int(os.environ.get("VAL_START", "4"))
VAL_SEED = int(os.environ.get("VAL_SEED", "20260818"))
VAL_DRAW_COUNT = int(os.environ.get("VAL_DRAW_COUNT", "16"))
CHECKPOINT = ts.default_checkpoint()


def cubic_guesses(U):
    return bc.polynomial_guesses(U, 3)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("matmul precision must be highest")
    report = {
        "config": {
            "purpose": "development-only trajectory correction quality probe",
            "classification": "supervised surrogate control; not a genuine NM-ROM",
            "ns": NS,
            "coarse_ns": COARSE_NS,
            "validation_seed": VAL_SEED,
            "validation_indices": list(range(VAL_START, VAL_START + N_VAL)),
            "validation_draw_count": VAL_DRAW_COUNT,
            "final_seed_touched": False,
            "checkpoint_sha256": ts.file_sha256(CHECKPOINT),
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "complete": False,
    }
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)

    for n in NS:
        trajectories = bc.generate_reference(
            n,
            list(range(VAL_START, VAL_START + N_VAL)),
            VAL_SEED,
            draw_count=VAL_DRAW_COUNT,
        )
        _, residual = bc.bf.make_rollout(n)
        for coarse_n in COARSE_NS:
            if coarse_n > n:
                continue
            constructor = ts.make_constructor(n, coarse_n, CHECKPOINT)
            per_trajectory = []
            for trajectory in trajectories:
                corrections, recovered = constructor(
                    jnp.asarray(trajectory["U"][0]), trajectory["nu"]
                )
                corrections, recovered = np.asarray(corrections), np.asarray(recovered)
                truth = trajectory["U"]
                exact_corrections = truth[1:] - cubic_guesses(truth)
                guesses = cubic_guesses(truth) + corrections
                guess_l2, guess_residual = bc.guess_diagnostics(
                    truth, guesses, residual, trajectory["nu"]
                )
                p = trajectory["parameters"]
                exact = np.asarray(
                    [
                        (p["cx"] - 0.5) / 0.35,
                        (p["cy"] - 0.5) / 0.35,
                        (p["width"] - 0.125) / 0.075,
                        (p["amplitude"] - 1.25) / 0.75,
                        (np.log(p["nu"]) - np.log(np.sqrt(0.001))) / (0.5 * np.log(10.0)),
                    ]
                )
                snapshot_remaining = np.linalg.norm(
                    corrections - exact_corrections, axis=1
                ) / np.maximum(np.linalg.norm(exact_corrections, axis=1), 1e-300)
                per_trajectory.append(
                    {
                        "trajectory_index": trajectory["index"],
                        "remaining_ratio_global": float(
                            np.linalg.norm(corrections - exact_corrections)
                            / np.linalg.norm(exact_corrections)
                        ),
                        "remaining_ratio_median": float(np.median(snapshot_remaining)),
                        "fraction_snapshots_q75": float(np.mean(snapshot_remaining <= 0.25)),
                        "fraction_snapshots_q90": float(np.mean(snapshot_remaining <= 0.10)),
                        "guess_rel_l2_mean": float(np.mean(guess_l2)),
                        "guess_rel_residual_mean": float(np.mean(guess_residual)),
                        "latent_max_abs_error": float(np.max(np.abs(recovered - exact))),
                    }
                )
            row = {
                "N": n,
                "coarse_n": coarse_n,
                "remaining_ratio_global": float(
                    np.sqrt(
                        np.mean([item["remaining_ratio_global"] ** 2 for item in per_trajectory])
                    )
                ),
                "remaining_ratio_median": float(
                    np.median([item["remaining_ratio_global"] for item in per_trajectory])
                ),
                "guess_rel_l2_mean": float(
                    np.mean([item["guess_rel_l2_mean"] for item in per_trajectory])
                ),
                "guess_rel_residual_mean": float(
                    np.mean([item["guess_rel_residual_mean"] for item in per_trajectory])
                ),
                "latent_max_abs_error": float(
                    max(item["latent_max_abs_error"] for item in per_trajectory)
                ),
                "per_trajectory": per_trajectory,
            }
            report["rows"].append(row)
            bc.log(
                f"N={n} q={coarse_n}: remaining={row['remaining_ratio_global']:.4f} "
                f"guessL2={row['guess_rel_l2_mean']:.4e} "
                f"guessR={row['guess_rel_residual_mean']:.4e} "
                f"latent={row['latent_max_abs_error']:.2e}"
            )
            with open(OUT, "w") as handle:
                json.dump(report, handle, indent=1, allow_nan=False)
    report["complete"] = True
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)
    bc.log("DONE")


if __name__ == "__main__":
    main()
