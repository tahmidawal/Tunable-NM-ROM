"""Immutable, case-split Poisson datasets using the inherited FD/DST solver.

Production commands require a Slurm GPU job. The smoke command is bounded and
marks all its output as smoke evidence. Gaussian descriptors are metadata only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / "experiments/multiresolution-poisson"
SCHEMA = "neural-operator-case-v1"
DATASET_SEED = 20260914
SPLITS = {"calibration": 0, "train": 1, "validation": 2}
OPERATOR = "zero-dirichlet-five-point-negative-laplacian-unit-square-v1"
PROVENANCE_FILES = [
    HERE / "dataset.py",
    NATIVE / "core.py",
    ROOT / "experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py",
]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def case_seed(split, index):
    if split not in SPLITS or index < 0:
        raise ValueError("Only nonnegative calibration/train/validation case indices are allowed")
    return int(np.random.SeedSequence(
        [DATASET_SEED, 11, SPLITS[split], index]
    ).generate_state(1, dtype=np.uint32)[0])


def preflight(smoke=False):
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise RuntimeError("Set JAX_DEFAULT_MATMUL_PRECISION=highest")
    if not smoke and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Production data generation requires a cluster Slurm GPU allocation")
    import jax
    jax.config.update("jax_enable_x64", True)
    backend = jax.default_backend()
    print(f"jax_backend={backend}", flush=True)
    if backend != "gpu":
        raise SystemExit(42)
    import jax.numpy as jnp
    if jnp.asarray(1.0).dtype != jnp.float64:
        raise RuntimeError("f64 preflight failed")
    if str(jax.config.jax_default_matmul_precision) != "highest":
        raise RuntimeError("JAX matmul precision preflight failed")
    sys.path.insert(0, str(NATIVE))
    import core
    info = {
        "jax_backend": backend, "jax_version": jax.__version__,
        "numpy_version": np.__version__, "jax_enable_x64": True,
        "matmul_precision": "highest", "gpu_type": jax.devices()[0].device_kind,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "purpose": "local-smoke-only" if smoke else "cluster-dataset-generation",
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "source_files": {str(p.relative_to(ROOT)): sha_file(p) for p in PROVENANCE_FILES},
        "source_commit": os.environ.get("CAMPAIGN_SOURCE_COMMIT"),
        "provenance_note": "Content hashes are authoritative; no cluster git ancestor lookup.",
    }
    return core, jax, info


def params_for(core, split, index):
    seed = case_seed(split, index)
    return seed, core.source_params(seed, 1)[0]


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def fd_apply(field, intervals):
    return intervals**2 * (4 * field[1:-1, 1:-1] - field[2:, 1:-1]
                          - field[:-2, 1:-1] - field[1:-1, 2:] - field[1:-1, :-2])


def solve(core, jax, intervals, parameters):
    source = np.asarray(core.full_source(intervals, parameters), dtype=np.float64)
    field = np.asarray(jax.device_get(core.dst_solve(
        jax.device_put(source), jax.device_put(core.eigenvalues(intervals)))))
    residual = relative(fd_apply(field, intervals), source[1:-1, 1:-1])
    boundary = float(max(np.max(np.abs(field[0])), np.max(np.abs(field[-1])),
                         np.max(np.abs(field[:, 0])), np.max(np.abs(field[:, -1]))))
    if field.dtype != np.float64 or not np.isfinite(field).all() or residual > 1e-8 or boundary != 0:
        raise RuntimeError(f"Invalid reference N={intervals}: residual={residual}, boundary={boundary}")
    return source, field, {"relative_discrete_residual": residual, "boundary_max_abs": boundary}


def new_directory(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def validate_meshes(meshes, reference_intervals, refinement_intervals):
    if not meshes or any(n < 4 for n in meshes) or len(meshes) != len(set(meshes)):
        raise ValueError("Meshes must be distinct integers >= 4")
    if reference_intervals <= max(meshes) or refinement_intervals != 2 * reference_intervals:
        raise ValueError("Reference must exceed every requested mesh; refinement must double reference")
    if any(reference_intervals % n for n in meshes):
        raise ValueError("Requested meshes must divide the reference mesh")


def calibrate(core, jax, info, output, count, meshes, reference_intervals,
              refinement_intervals, threshold):
    validate_meshes(meshes, reference_intervals, refinement_intervals)
    if count < 1 or not 0 < threshold < 1:
        raise ValueError("Count must be positive and refinement budget between zero and one")
    output = new_directory(output)
    records = []
    for index in range(count):
        seed, params = params_for(core, "calibration", index)
        _, refined, refined_checks = solve(core, jax, refinement_intervals, params)
        _, reference, reference_checks = solve(core, jax, reference_intervals, params)
        refined_restricted = refined[::2, ::2]
        reference_change = relative(reference, refined_restricted)
        mesh_records = []
        for intervals in meshes:
            _, field, checks = solve(core, jax, intervals, params)
            mesh_records.append({"intervals": intervals, **checks,
                "relative_to_restricted_reference": relative(
                    field, reference[::reference_intervals // intervals,
                                     ::reference_intervals // intervals]),
                "relative_to_restricted_refinement": relative(
                    field, refined[::refinement_intervals // intervals,
                                   ::refinement_intervals // intervals])})
        field_path = f"case-{index:05d}-references.npz"
        observe = max(meshes)
        np.savez_compressed(output / field_path,
            reference=reference[::reference_intervals//observe, ::reference_intervals//observe],
            refinement=refined[::refinement_intervals//observe, ::refinement_intervals//observe])
        record = {"field_path": field_path, "field_sha256": sha_file(output / field_path),
                  "field_mesh": observe, "case_id": f"poisson-calibration-{index:05d}", "split": "calibration",
                  "seed": seed, "offline_generation_parameters": params.tolist(),
                  "reference_relative_change": reference_change,
                  "empirical_reference_budget_pass": reference_change <= threshold,
                  "reference_checks": reference_checks, "refinement_checks": refined_checks,
                  "meshes": mesh_records}
        records.append(record)
        print(json.dumps({"case_id": record["case_id"], "reference_relative_change": reference_change}), flush=True)
    result = {"schema": SCHEMA, "pde": "poisson", "operator": OPERATOR,
        "dataset_seed": DATASET_SEED, "runtime": info, "complete": True,
        "reference_intervals": reference_intervals, "refinement_intervals": refinement_intervals,
        "meshes": meshes, "count": count, "records": records,
        "relative_reference_budget": threshold,
        "empirical_reference_budget_pass": all(r["empirical_reference_budget_pass"] for r in records),
        "interpretation": "Empirical refinement diagnostic on calibration cases; not a rigorous continuum error bound or a guarantee for other cases."}
    write_json(output / "calibration.json", result)
    write_json(output / "checksums.json", {str(p.relative_to(output)): sha_file(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return result


def check_calibration(path, intervals, info):
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    required = (result.get("complete") is True and result.get("pde") == "poisson"
                and result.get("schema") == SCHEMA and result.get("operator") == OPERATOR
                and result.get("dataset_seed") == DATASET_SEED
                and result.get("empirical_reference_budget_pass") is True
                and intervals in result.get("meshes", [])
                and result.get("runtime", {}).get("purpose") == info["purpose"]
                and result.get("runtime", {}).get("source_files") == info["source_files"])
    if not required:
        raise RuntimeError("Calibration failed, used a different source/schema/mesh, or is a smoke artifact")
    return result, {"path": str(path), "sha256": sha_file(path)}


def generate(core, jax, info, output, split, count, intervals, calibration_path,
             include_reference=False):
    if split not in ("train", "validation") or count < 1:
        raise ValueError("Only positive train/validation cohorts can be generated")
    calibration, calibration_record = check_calibration(calibration_path, intervals, info)
    output = new_directory(output)
    (output / "cases").mkdir()
    if include_reference:
        (output / "physical-reference").mkdir()
    records = []
    for index in range(count):
        seed, params = params_for(core, split, index)
        source, field, checks = solve(core, jax, intervals, params)
        case_id = f"poisson-{split}-{index:05d}"
        relative_path = f"cases/{case_id}.npz"
        arrays = {"input": source[None], "target": field[None, None],
                  "parameters": np.empty((0,), dtype=np.float64),
                  "times": np.asarray([0.0], dtype=np.float64)}
        reference = {"kind": "exact-discrete-fd-dst", "operator": OPERATOR,
                     "target_intervals": intervals, **checks,
                     "calibration": calibration_record,
                     "physical_accuracy_status": "Discrete target; empirical refinement only."}
        if include_reference:
            ref_intervals = calibration["reference_intervals"]
            _, ref_field, ref_checks = solve(core, jax, ref_intervals, params)
            restriction = ref_intervals // intervals
            ref_target = ref_field[::restriction, ::restriction][None, None]
            fine_intervals = calibration["refinement_intervals"]
            _, fine_field, fine_checks = solve(core, jax, fine_intervals, params)
            fine_target = fine_field[::fine_intervals//intervals, ::fine_intervals//intervals][None, None]
            ref_change = relative(ref_field, fine_field[::2, ::2])
            sidecar = f"physical-reference/{case_id}.npz"
            np.savez_compressed(output / sidecar, target=ref_target, refinement=fine_target)
            reference.update({"path": sidecar, "sha256": sha_file(output / sidecar),
                "refinement_intervals": fine_intervals, "refinement_checks": fine_checks,
                "reference_relative_change": ref_change,
                "empirical_reference_budget_pass": ref_change <= calibration["relative_reference_budget"]})
            reference.update({"reference_intervals": ref_intervals,
                "reference_checks": ref_checks,
                "target_relative_to_reference": relative(field, ref_target[0, 0]),
                "reference_target_use": "Evaluation only; forbidden as training target under this dataset protocol."})
        with (output / relative_path).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        records.append({"case_id": case_id, "split": split, "seed": seed,
                        "path": relative_path, "sha256": sha_file(output / relative_path),
                        "mesh": intervals, "intervals": intervals, "nodes_per_axis": intervals + 1,
                        "offline_generation_parameters": {"cx": float(params[0]), "cy": float(params[1]),
                            "width": float(params[2]), "amplitude": float(params[3])},
                        "reference": reference})
        print(json.dumps({"case_id": case_id, "intervals": intervals, **checks}), flush=True)
    manifest = {"schema": SCHEMA, "pde": "poisson", "operator": OPERATOR,
                "dataset_seed": DATASET_SEED, "split": split, "count": count,
                "intervals": intervals, "runtime": info, "complete": True,
                "records": records, "calibration": calibration_record,
                "model_input_contract": "Only input, known-physics parameters and coordinates; generation descriptors are forbidden.",
                "target_contract": "Target is the native discrete solution on this mesh; physical-reference sidecars are evaluation-only.",
                "final_cohort": "sealed; unsupported by generator"}
    write_json(output / "index.json", manifest)
    write_json(output / "checksums.json", {str(p.relative_to(output)): sha_file(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return manifest


def smoke(core, jax, info, output):
    """Independent manufactured discrete operator and output contract checks."""
    output = new_directory(output)
    n = 16
    x = np.arange(n + 1, dtype=np.float64) / n
    field = np.sin(np.pi * x[:, None]) * np.sin(3 * np.pi * x[None, :])
    field[[0, -1], :] = 0
    field[:, [0, -1]] = 0
    source = np.pad(fd_apply(field, n), 1)
    actual = np.asarray(jax.device_get(core.dst_solve(
        jax.device_put(source), jax.device_put(core.eigenvalues(n)))))
    manufactured_error = relative(actual, field)
    if manufactured_error > 1e-12:
        raise RuntimeError(f"Manufactured independent FD/DST parity failed: {manufactured_error}")
    # This loose smoke-only threshold tests plumbing, not physical accuracy.
    calibration = calibrate(core, jax, info, output / "calibration", 2, [8, 16], 32, 64, 0.5)
    manifest = generate(core, jax, info, output / "train", "train", 2, 16,
                        output / "calibration/calibration.json", include_reference=True)
    for record in manifest["records"]:
        with np.load(output / "train" / record["path"], allow_pickle=False) as sample:
            assert sample["input"].shape == (1, 17, 17)
            assert sample["target"].shape == (1, 1, 17, 17)
            assert sample["parameters"].shape == (0,)
            assert sample["times"].tolist() == [0.0]
            assert all(sample[key].dtype == np.float64 for key in sample.files)
            assert set(sample.files) == {"input", "target", "parameters", "times"}
    try:
        check_calibration(output / "calibration/calibration.json", 16,
                          {**info, "purpose": "cluster-dataset-generation"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("A production run accepted smoke calibration")
    result = {"complete": True, "purpose": "local-smoke-only", "runtime": info,
              "manufactured_relative_error": manufactured_error,
              "schema_validation_pass": True, "production_rejects_smoke": True,
              "calibration_complete": calibration["complete"]}
    write_json(output / "smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--count", type=int, default=8)
    cal.add_argument("--meshes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    cal.add_argument("--reference-intervals", type=int, default=2048)
    cal.add_argument("--refinement-intervals", type=int, default=4096)
    cal.add_argument("--reference-budget", type=float, default=1e-3)
    gen = sub.add_parser("generate")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--split", choices=["train", "validation"], required=True)
    gen.add_argument("--count", type=int, required=True)
    gen.add_argument("--intervals", type=int, default=256)
    gen.add_argument("--calibration", type=Path, required=True)
    gen.add_argument("--include-reference", action="store_true")
    sm = sub.add_parser("smoke")
    sm.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core, jax, info = preflight(smoke=args.command == "smoke")
    if args.command == "calibrate":
        result = calibrate(core, jax, info, args.output, args.count, args.meshes,
                           args.reference_intervals, args.refinement_intervals, args.reference_budget)
        if not result["empirical_reference_budget_pass"]:
            raise SystemExit(2)
    elif args.command == "generate":
        generate(core, jax, info, args.output, args.split, args.count, args.intervals,
                 args.calibration, args.include_reference)
    else:
        smoke(core, jax, info, args.output)


if __name__ == "__main__":
    main()
