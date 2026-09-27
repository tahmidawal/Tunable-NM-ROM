"""Immutable Burgers case files with a calibration gate before bulk generation.

The native converged backward-Euler / sign-upwind FOM is imported unchanged.
Only ``plan`` is CPU-only; every numerical command requires a real f64 GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL_PATH = HERE / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
NATIVE = ROOT / "experiments/mr-burgers2d/engines.py"
DEPENDENCIES = [NATIVE, ROOT / "experiments/separable-decoder/sep_common.py"]
SOURCE_FILES = [Path(__file__).resolve(), PROTOCOL_PATH, *DEPENDENCIES]
TIMES = np.asarray(PROTOCOL["times"], dtype=np.float64)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in SOURCE_FILES}


def case_seed(split, index):
    if split not in PROTOCOL["split_codes"] or index < 0:
        raise ValueError("Only nonnegative calibration/train/validation case IDs are allowed")
    return int(np.random.SeedSequence([
        PROTOCOL["dataset_seed"], PROTOCOL["pde_seed_code"],
        PROTOCOL["split_codes"][split], index,
    ]).generate_state(1, dtype=np.uint32)[0])


def case_record(split, index):
    return dict(case_id=f"burgers-{split}-{index:05d}", split=split,
                case_index=index, seed=case_seed(split, index))


def allowed_count(split, count):
    limit = PROTOCOL["counts"][split]
    if not 1 <= count <= limit:
        raise ValueError(f"count must be in [1, {limit}] for {split}")


def gpu_modules():
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise RuntimeError("JAX_DEFAULT_MATMUL_PRECISION=highest is required")
    import jax
    jax.config.update("jax_enable_x64", True)
    backend = jax.default_backend()
    print(f"jax_backend={backend}", flush=True)
    if backend != "gpu":
        raise SystemExit(42)
    if not jax.config.jax_enable_x64 or str(jax.config.jax_default_matmul_precision) != "highest":
        raise RuntimeError("f64 and highest matmul precision are required")
    sys.path.insert(0, str(NATIVE.parent))
    import engines
    if Path(engines.__file__).resolve() != NATIVE:
        raise RuntimeError("Unexpected native FOM module path")
    return jax, engines


def provenance(jax):
    # A staged directory may sit inside an unrelated ancestor git repository.
    # Only use local git when its top-level equals this exact source bundle root.
    commit = os.environ.get("SOURCE_COMMIT")
    if not commit:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and Path(result.stdout.strip()).resolve() == ROOT:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return dict(source_commit=commit, source_sha256=source_hashes(),
                job_id=os.environ.get("SLURM_JOB_ID"), gpu=jax.devices()[0].device_kind,
                backend=jax.default_backend(), f64=bool(jax.config.jax_enable_x64),
                matmul_precision=os.environ["JAX_DEFAULT_MATMUL_PRECISION"],
                python=sys.version, platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in
                          ("jax", "jaxlib", "numpy", "scipy", "optax")})


def make_output(path):
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def fixed_initial_errors(field, reference):
    if field.shape != reference.shape or field.ndim != 3:
        raise ValueError("Expected matching (time, x, y) arrays")
    denominator = np.linalg.norm(reference[0, 1:-1, 1:-1])
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Invalid reference initial norm")
    values = np.linalg.norm((field - reference)[:, 1:-1, 1:-1].reshape(len(TIMES), -1), axis=1) / denominator
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite error metric")
    return dict(per_time=values.tolist(), maximum=float(values.max()))


def restrict(field, intervals):
    source_intervals = field.shape[-1] - 1
    if source_intervals % intervals:
        raise ValueError("Restriction must use nested grid nodes")
    stride = source_intervals // intervals
    return np.ascontiguousarray(field[:, ::stride, ::stride], dtype=np.float64)


def setting_id(intervals, dt):
    return f"L{intervals}_dt{dt:.9f}"


def anchor_settings():
    L, dt = PROTOCOL["anchor"]["intervals"], PROTOCOL["anchor"]["dt"]
    return [(L // 4, 2 * dt), (L // 2, 2 * dt), (L // 2, dt),
            (L, 4 * dt), (L, 2 * dt), (L, dt)]


def make_solver(engines, intervals, dt, output_intervals):
    dense = max(intervals, output_intervals)
    if dense % min(intervals, output_intervals):
        raise ValueError("Solver and output meshes must be nested")
    query, _ = engines.make_fom(intervals, dt, target=dense,
                               max_newton=PROTOCOL["max_newton"])
    return query, dense


def solve(jax, engines, query, dense, output_intervals, physical):
    import jax.numpy as jnp
    # Short clock warmup before each runtime profile. These compile-inclusive
    # wall times plan resources only and are never a solver speedup benchmark.
    burn = jax.jit(lambda x: x @ x)
    x = jnp.eye(512, dtype=jnp.float64)
    until = time.monotonic() + 2.
    while time.monotonic() < until:
        burn(x).block_until_ready()
    started = time.perf_counter()
    fields, iterations, residuals = jax.tree_util.tree_map(np.asarray, query(
        jnp.asarray(engines.initial(dense, physical), dtype=jnp.float64),
        float(physical[4]), PROTOCOL["newton_tolerance"], PROTOCOL["linear_tolerance"],
    ))
    elapsed = time.perf_counter() - started
    if fields.dtype != np.float64 or fields.shape != (len(TIMES), dense + 1, dense + 1):
        raise RuntimeError("Native FOM returned wrong dtype or shape")
    if not np.isfinite(fields).all() or not np.isfinite(residuals).all():
        raise RuntimeError("Native FOM returned nonfinite fields/residuals")
    if float(np.max(residuals)) > PROTOCOL["maximum_relative_residual"]:
        raise RuntimeError(f"FOM convergence failed: maximum residual {np.max(residuals)}")
    if np.any(iterations > PROTOCOL["max_newton"]):
        raise RuntimeError("Native FOM exceeded Newton iteration budget")
    if np.any(fields[:, [0, -1], :]) or np.any(fields[:, :, [0, -1]]):
        raise RuntimeError("Dirichlet boundary check failed")
    # The solver restricts the supplied field to its mesh. Requested t=0 is
    # still the original supplied field, not its coarse-grid interpolant.
    fields = fields.copy()
    fields[0] = engines.initial(dense, physical)
    return restrict(fields, output_intervals), iterations, residuals, elapsed


def save_case(out, record, fields, physical, intervals, reference):
    name = f"{record['case_id']}.npz"
    # Generation descriptors remain outside the model-facing NPZ. Only viscosity
    # is a physical query coefficient; centre/width/amplitude are never inputs.
    np.savez(out / name, input=fields[0][None], target=fields[:, None],
             parameters=np.asarray([physical[4]], dtype=np.float64), times=TIMES)
    return dict(**record, path=name, sha256=sha(out / name), mesh=intervals,
                generation_descriptors=dict(zip(("cx", "cy", "width", "amplitude", "nu"), physical.tolist())),
                reference=reference)


def evaluate_gate(report, fields):
    L, dt = PROTOCOL["anchor"]["intervals"], PROTOCOL["anchor"]["dt"]
    settings = [(x["intervals"], x["dt"]) for x in PROTOCOL["candidates"]] + [(L, dt)]
    by_output = {}
    full_count = report["count"] == PROTOCOL["counts"]["calibration"]
    for output in PROTOCOL["evaluation_intervals"]:
        rows = []
        for index in range(report["count"]):
            get = lambda mesh, step: restrict(fields[(mesh, step, index)], output)
            fine = get(L, dt)
            space = fixed_initial_errors(get(L // 2, dt), fine)["maximum"]
            temporal = fixed_initial_errors(get(L, 2 * dt), fine)["maximum"]
            sc = fixed_initial_errors(get(L // 4, 2 * dt), get(L // 2, 2 * dt))["maximum"]
            sf = fixed_initial_errors(get(L // 2, 2 * dt), get(L, 2 * dt))["maximum"]
            tc = fixed_initial_errors(get(L, 4 * dt), get(L, 2 * dt))["maximum"]
            rows.append(dict(case_id=case_record("calibration", index)["case_id"],
                             anchor_space_difference=space, anchor_time_difference=temporal,
                             anchor_difference_sum=space + temporal,
                             spatial_coarse_difference=sc, spatial_fine_difference=sf,
                             temporal_coarse_difference=tc, temporal_fine_difference=temporal,
                             refinement_decreases=bool(sf < sc and temporal < tc),
                             candidates=[dict(intervals=mesh, dt=step,
                                              difference_from_anchor=fixed_initial_errors(get(mesh, step), fine)["maximum"])
                                         for mesh, step in settings]))
        candidates = []
        for j, (mesh, step) in enumerate(settings):
            margins = [row["anchor_difference_sum"] + row["candidates"][j]["difference_from_anchor"] for row in rows]
            passed = mesh >= output and full_count and all(row["refinement_decreases"] for row in rows) and max(margins) <= PROTOCOL["empirical_reference_budget"]
            candidates.append(dict(intervals=mesh, dt=step, passing=passed,
                                   worst_empirical_margin=max(margins),
                                   work_proxy=mesh * mesh * round(TIMES[-1] / step)))
        passing = sorted((item for item in candidates if item["passing"]), key=lambda item: item["work_proxy"])
        by_output[str(output)] = dict(passing=bool(passing), selected=passing[0] if passing else None,
                                     cases=rows, candidates=candidates)
    return dict(status="calibrated" if full_count else "profiling_only",
                calibration_count=report["count"], required_count=PROTOCOL["counts"]["calibration"],
                empirical_budget=PROTOCOL["empirical_reference_budget"],
                interpretation=PROTOCOL["reference_interpretation"],
                selection="lowest mesh^2 times time-step-count proxy among passing settings; no cross-job timing selection",
                by_output=by_output)


def calibration(args):
    allowed_count("calibration", args.count)
    jax, engines = gpu_modules()
    out = make_output(args.out)
    report = dict(schema_version=1, kind="burgers-reference-calibration", pde="burgers",
                  count=args.count, complete=False, protocol_sha256=sha(PROTOCOL_PATH),
                  provenance=provenance(jax), records=[], solves=[],
                  role="development calibration; never training or final test")
    path = out / "index.json"
    write_json(path, report)
    all_fields = {}
    physical = [engines.params_draw(case_seed("calibration", i), 1)[0] for i in range(args.count)]
    dense_output = max(PROTOCOL["evaluation_intervals"])
    settings = list(dict.fromkeys(anchor_settings() + [(s["intervals"], s["dt"]) for s in PROTOCOL["candidates"]]))
    started = time.monotonic()
    setting_seconds = {}
    try:
        # Case zero profiles all six refinement settings before other cases.
        # It is then retained as one of eight independent calibration inputs.
        for index, params in enumerate(physical):
            for intervals, dt in settings:
                estimate = setting_seconds.get((intervals, dt), 0.)
                if time.monotonic() - started + max(120., 1.5 * estimate) > args.wall_budget_seconds:
                    report["stop_reason"] = "allocation budget checkpoint; calibration incomplete and bulk locked"
                    write_json(path, report)
                    return
                query, dense = make_solver(engines, intervals, dt, dense_output)
                print(f"CALIBRATION {setting_id(intervals, dt)} case={index}", flush=True)
                fields, iterations, residuals, seconds = solve(jax, engines, query, dense, dense_output, params)
                all_fields[(intervals, dt, index)] = fields
                name = f"{case_record('calibration', index)['case_id']}_{setting_id(intervals, dt)}.npz"
                np.savez(out / name, fields=fields, iterations=iterations, residuals=residuals)
                report["solves"].append(dict(case_id=case_record("calibration", index)["case_id"],
                    seed=case_seed("calibration", index), intervals=intervals, dt=dt,
                    path=name, sha256=sha(out / name), output_intervals=dense_output,
                    wall_seconds_including_first_compile=seconds,
                    max_relative_residual=float(residuals.max()), total_newton_iterations=int(iterations.sum())))
                setting_seconds[(intervals, dt)] = seconds
                write_json(path, report)
                del query
                jax.clear_caches()
            if index == 0:
                profile_report = dict(report, count=1)
                report["profile_gate"] = evaluate_gate(profile_report, all_fields)
                report["profile_seconds"] = time.monotonic() - started
                report["remaining_cases_seconds_estimate"] = sum(setting_seconds.values()) * (args.count - 1)
                report["profile_unlocks_bulk"] = False
                write_json(path, report)
                print("PROFILE COMPLETE " + json.dumps({key: report[key] for key in
                    ("profile_seconds", "remaining_cases_seconds_estimate", "profile_unlocks_bulk")}), flush=True)
        report["gate"] = evaluate_gate(report, all_fields)
        L, dt = PROTOCOL["anchor"]["intervals"], PROTOCOL["anchor"]["dt"]
        for index, params in enumerate(physical):
            record = case_record("calibration", index)
            report["records"].append(save_case(out, record, all_fields[(L, dt, index)], params, dense_output,
                dict(intervals=L, dt=dt, role="reference anchor; see per-output empirical margin in gate")))
        report["complete"] = True
        write_json(path, report)
        print(json.dumps(dict(index=str(path), gate_status=report["gate"]["status"],
                              passing_meshes=[key for key, gate in report["gate"]["by_output"].items() if gate["passing"]])), flush=True)
    except BaseException as error:
        report["failure"] = str(error)
        write_json(path, report)
        raise


def read_calibration(path, intervals):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get("kind") != "burgers-reference-calibration" or not report.get("complete"):
        raise ValueError("An independently completed Burgers calibration is required")
    if report.get("protocol_sha256") != sha(PROTOCOL_PATH):
        raise ValueError("Calibration protocol hash mismatch")
    if report.get("provenance", {}).get("source_sha256") != source_hashes():
        raise ValueError("Calibration implementation hash mismatch; recalibrate changed numerical code")
    if report.get("count") != PROTOCOL["counts"]["calibration"] or report.get("gate", {}).get("status") != "calibrated":
        raise ValueError("Profiling-only calibration cannot unlock bulk generation")
    gate = report["gate"]["by_output"].get(str(intervals), {})
    if not gate.get("passing") or not gate.get("selected"):
        raise ValueError(f"Empirical reference quality gate has not passed for {intervals} intervals")
    expected = [case_record("calibration", i)["case_id"] for i in range(PROTOCOL["counts"]["calibration"])]
    if [item["case_id"] for item in report["records"]] != expected:
        raise ValueError("Calibration case IDs do not match the frozen protocol")
    for record in report["records"] + report["solves"]:
        artifact = (path.parent / record["path"]).resolve()
        if artifact.parent != path.parent or sha(artifact) != record["sha256"]:
            raise ValueError("Calibration artifact path/checksum verification failed")
    return gate["selected"], dict(path=str(path), sha256=sha(path),
                                  interpretation=PROTOCOL["reference_interpretation"],
                                  worst_empirical_margin=gate["selected"]["worst_empirical_margin"])


def generate(args):
    count = args.count if args.count is not None else PROTOCOL["counts"][args.split]
    allowed_count(args.split, count)
    selected, evidence = read_calibration(args.calibration, args.intervals)
    jax, engines = gpu_modules()
    out = make_output(args.out)
    report = dict(schema_version=1, pde="burgers", kind="matched-neural-operator-dataset",
                  split=args.split, count=count, mesh=args.intervals, complete=False,
                  protocol_sha256=sha(PROTOCOL_PATH), provenance=provenance(jax), records=[],
                  calibration=evidence, reference_setting=selected,
                  physical_accuracy_status="empirically calibrated on independent development cases; not a per-case continuum certificate",
                  descriptors_are_model_inputs=False, final_cohort="sealed")
    path = out / "index.json"
    write_json(path, report)
    query, dense = make_solver(engines, selected["intervals"], selected["dt"], args.intervals)
    try:
        for index in range(count):
            record = case_record(args.split, index)
            physical = engines.params_draw(record["seed"], 1)[0]
            fields, iterations, residuals, elapsed = solve(jax, engines, query, dense, args.intervals, physical)
            # The input and target initial field must reproduce the physical
            # Gaussian on requested nodes, not acquire interpolation distortion.
            if not np.array_equal(fields[0], engines.initial(args.intervals, physical)):
                raise RuntimeError("Candidate changed the requested initial field; use a solver at least as fine as output")
            reference = dict(intervals=selected["intervals"], dt=selected["dt"],
                             max_relative_residual=float(residuals.max()),
                             wall_seconds_including_first_compile=elapsed,
                             total_newton_iterations=int(iterations.sum()),
                             calibration_sha256=evidence["sha256"])
            generated = save_case(out, record, fields, physical, args.intervals, reference)
            audit_name = record["case_id"] + ".solver.npz"
            np.savez(out / audit_name, iterations=iterations, residuals=residuals)
            generated["solver_audit"] = dict(path=audit_name, sha256=sha(out / audit_name))
            report["records"].append(generated)
            write_json(path, report)
            print(f"GENERATED {record['case_id']} mesh={args.intervals}", flush=True)
        report["complete"] = True
        write_json(path, report)
    except BaseException as error:
        report["failure"] = str(error)
        write_json(path, report)
        raise


def plan(args):
    out = make_output(args.out)
    records = [case_record(split, index) for split, count in PROTOCOL["counts"].items() for index in range(count)]
    write_json(out / "case-plan.json", dict(schema_version=1, pde="burgers", complete=True,
               kind="case-plan-only-no-solution-data", protocol=PROTOCOL,
               protocol_sha256=sha(PROTOCOL_PATH), source_sha256=source_hashes(), records=records))
    print(out / "case-plan.json")


def smoke(args):
    jax, engines = gpu_modules()
    out = make_output(args.out)
    params = engines.params_draw(case_seed("calibration", 0), 1)[0]
    L, dt = 16, 0.05
    query, dense = make_solver(engines, L, dt, L)
    fields, iterations, residuals, _ = solve(jax, engines, query, dense, L, params)
    u, previous = fields[1, 1:-1, 1:-1], fields[0, 1:-1, 1:-1]
    padded = np.pad(u, 1)
    xm, xp = padded[:-2, 1:-1], padded[2:, 1:-1]
    ym, yp = padded[1:-1, :-2], padded[1:-1, 2:]
    adv = u * L * (np.where(u > 0, u - xm, xp - u) + np.where(u > 0, u - ym, yp - u))
    lap = L * L * (xm + xp + ym + yp - 4 * u)
    independent = np.linalg.norm(u - previous + dt * (adv - params[4] * lap)) / np.linalg.norm(previous)
    if independent > PROTOCOL["maximum_relative_residual"]:
        raise RuntimeError("Independent NumPy one-step residual failed")
    record = save_case(out, case_record("calibration", 0), fields, params, L,
                       dict(role="smoke only; no reference accuracy claim", intervals=L, dt=dt))
    write_json(out / "index.json", dict(schema_version=1, pde="burgers", kind="smoke-only", complete=True,
               provenance=provenance(jax), independent_numpy_relative_residual=float(independent),
               records=[record], bulk_generation_unlocked=False))
    print(f"SMOKE PASS NumPy relative residual={independent:.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("plan", help="Write deterministic case IDs without generating solution data")
    command.add_argument("--out", required=True)
    command.set_defaults(func=plan)
    command = commands.add_parser("calibrate", help="Run native 4096 anchor and cheaper candidates; count1 profiles only")
    command.add_argument("--count", type=int, default=PROTOCOL["counts"]["calibration"])
    command.add_argument("--out", required=True)
    command.add_argument("--wall-budget-seconds", type=float, default=27000.)
    command.set_defaults(func=calibration)
    command = commands.add_parser("generate", help="Generate a non-final split after the independent reference gate passes")
    command.add_argument("--split", choices=["train", "validation"], required=True)
    command.add_argument("--count", type=int)
    command.add_argument("--intervals", type=int, choices=PROTOCOL["evaluation_intervals"], default=256)
    command.add_argument("--calibration", required=True)
    command.add_argument("--out", required=True)
    command.set_defaults(func=generate)
    command = commands.add_parser("smoke", help="Tiny GPU native-FOM and model-facing schema check, not calibration")
    command.add_argument("--out", required=True)
    command.set_defaults(func=smoke)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
