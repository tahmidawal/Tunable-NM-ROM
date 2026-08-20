#!/usr/bin/env python
"""Phase-5 diagnostic: locked train coefficients and nonlinear-generator cost."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase4_d as p4d
import b10_phase5 as p5
import b10_s0_spline as base
import b10_spline as s


TRAIN = {64: (0, 512, 64), 128: (0, 128, 32), 256: (0, 64, 16)}
TARGET_COUNT = 35_904
TARGET_WORKERS = int(os.environ.get("B10_ORACLE_WORKERS", "8"))
TIME_REPS = 20
TIME_WARM = 1
BURN_SECONDS = 3.0
IDENTITY_TOL = 2e-14
TARGET_SEED = 0
TARGET_DRAW_COUNT = 704
OUTPUT_BASENAME = "phase5_d"


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            rows[relative.removeprefix("./")] = digest
    return rows


def validate_p4_chain(json_path, npz_path, audit_path, manifest_path):
    report, audit = load_json(json_path), load_json(audit_path)
    if report.get("status") != "complete" or audit.get("status") != "pass":
        raise SystemExit("P4-D immutable chain is not complete")
    if report.get("decision") != {
        "selected_spatial_arm": "H1",
        "training_seed11_k24_licensed": True,
        "correction_classification": "conditional-zero-or-occasional-attempt",
        "phase4_hard_stop": False,
        "scientific_promotion_allowed": True,
    }:
        raise SystemExit("P4-D immutable decision mismatch")
    if (
        report["npz"]["sha256"] != c.sha256(npz_path)
        or audit["source_json_sha256"] != c.sha256(json_path)
        or audit["source_npz_sha256"] != c.sha256(npz_path)
        or audit["decision"] != report["decision"]
    ):
        raise SystemExit("P4-D checksum/audit binding mismatch")
    return report, audit, manifest_rows(manifest_path)


def validate_rank_chain(rank_json, rank_script, rank_checkpoint, prereg):
    report = load_json(rank_json)
    if report.get("status") != "pass":
        raise SystemExit("Phase-5 rank diagnostic incomplete")
    if report["diagnostic_source_sha256"] != c.sha256(rank_script):
        raise SystemExit("rank diagnostic source binding mismatch")
    if report.get("phase5_bracket_adaptation_allowed") is not False:
        raise SystemExit("Phase-5 bracket must be prospectively fixed")
    return {
        "json": os.path.abspath(rank_json),
        "json_sha256": c.sha256(rank_json),
        "script": os.path.abspath(rank_script),
        "script_sha256": c.sha256(rank_script),
        "checkpoint": os.path.abspath(rank_checkpoint),
        "checkpoint_sha256": c.sha256(rank_checkpoint),
        "preregistration": os.path.abspath(prereg),
        "preregistration_sha256": c.sha256(prereg),
        "dense_hyperdecoder_geometry": report["dense_hyperdecoder_geometry"],
        "randomized_svd": report["randomized_svd"],
        "immutable_inputs": report["immutable_inputs"],
    }


def p4_binding(json_path, npz_path, audit_path, manifest_path, report, audit):
    return {
        "json": os.path.abspath(json_path), "json_sha256": c.sha256(json_path),
        "npz": os.path.abspath(npz_path), "npz_sha256": c.sha256(npz_path),
        "audit": os.path.abspath(audit_path), "audit_sha256": c.sha256(audit_path),
        "manifest": os.path.abspath(manifest_path),
        "manifest_sha256": c.sha256(manifest_path),
        "commit": report["provenance"]["commit"],
        "job_id": report["provenance"]["slurm_job_id"],
        "decision": audit["decision"],
    }


def fit_chunk(fields, coords, mask):
    flat = np.asarray(fields, np.float64).reshape(-1, fields.shape[-1])
    count = flat.shape[0]
    coefficient = np.empty((count, 3328), np.float64)
    affine = np.empty((count, 5), np.float64)
    normal = np.empty(count, np.float64)
    elapsed = np.empty(count, np.float64)
    healthy = np.empty(count, bool)
    boundary = np.empty(count, bool)
    pou = np.empty(count, np.float64)
    support = np.empty(count, np.int64)
    rhs_finite = np.empty(count, bool)
    prediction_finite = np.empty(count, bool)
    coefficient_finite = np.empty(count, bool)
    nnz_design = np.empty(count, np.int64)
    nnz_normal = np.empty(count, np.int64)
    nnz_factors = np.empty(count, np.int64)

    def one(field):
        affine_value, values, prediction, info = p5.p4.fit_hierarchical_oracle(
            field, coords, mask, p5.H1
        )
        # Reconstruct the locked RHS finiteness independently.  The same design
        # is intentionally rebuilt: target integrity is more important than
        # avoiding this offline-only diagnostic work.
        _, design, _ = p5.p4.hierarchical_sparse_design(field, coords, mask, p5.H1)
        rhs_ok = bool(np.all(np.isfinite(np.asarray(design.T @ field))))
        return affine_value, values, prediction, info, rhs_ok

    if TARGET_WORKERS == 1:
        fitted, executor = map(one, flat), None
    else:
        executor = ThreadPoolExecutor(max_workers=TARGET_WORKERS)
        fitted = executor.map(one, flat, chunksize=1)
    for index, (one_affine, values, prediction, info, one_rhs_finite) in enumerate(fitted):
        affine[index], coefficient[index] = one_affine, values
        normal[index], elapsed[index] = info["relative_normal_residual"], info["elapsed_s"]
        boundary[index], pou[index] = info["boundary_exact"], info["pou_max_abs"]
        support[index] = info["max_local_support"]
        rhs_finite[index] = one_rhs_finite
        prediction_finite[index] = np.all(np.isfinite(prediction))
        coefficient_finite[index] = np.all(np.isfinite(values))
        nnz_design[index] = info["nnz_design"]
        nnz_normal[index] = info["nnz_normal"]
        nnz_factors[index] = info["nnz_factors"]
        healthy[index] = bool(
            info["healthy"] and one_rhs_finite and prediction_finite[index]
            and coefficient_finite[index]
        )
    if executor is not None:
        executor.shutdown(wait=True)
    shape = fields.shape[:-1]
    return {
        "affine": affine.reshape(shape + (5,)),
        "coefficients": coefficient.reshape(shape + (3328,)),
        "normal": normal.reshape(shape), "elapsed": elapsed.reshape(shape),
        "healthy": healthy.reshape(shape), "boundary": boundary.reshape(shape),
        "pou": pou.reshape(shape), "support": support.reshape(shape),
        "rhs_finite": rhs_finite.reshape(shape),
        "prediction_finite": prediction_finite.reshape(shape),
        "coefficient_finite": coefficient_finite.reshape(shape),
        "nnz_design": nnz_design.reshape(shape),
        "nnz_normal": nnz_normal.reshape(shape),
        "nnz_factors": nnz_factors.reshape(shape),
    }


def save_target_chunk(path, n, indices, fields, parameters, health, fitted):
    times = int(fields.shape[1])
    features = c.trajectory_features(parameters, n)[:, :times]
    case = np.repeat(np.asarray(indices, np.int64)[:, None], times, axis=1)
    time_index = np.broadcast_to(np.arange(times, dtype=np.int64), case.shape)
    mesh = np.full(case.shape, n, np.int64)
    arrays = dict(fitted)
    arrays.update({
        "features": features, "global_index": case, "time_index": time_index,
        "N": mesh, "normalized_parameters": parameters["normalized"],
    })
    for name in ("cx", "cy", "width", "amplitude", "nu"):
        arrays[f"parameter_{name}"] = parameters[name]
    np.savez(path, **arrays)
    return {
        "basename": os.path.basename(path), "sha256": c.sha256(path),
        "N": n, "indices": [int(value) for value in indices],
        "snapshot_count": int(case.size), "reference_health": health,
        "healthy_count": int(np.sum(fitted["healthy"])),
        "normal_worst": float(np.max(fitted["normal"])),
        "boundary_all": bool(np.all(fitted["boundary"])),
        "pou_worst": float(np.max(fitted["pou"])),
        "support_min": int(np.min(fitted["support"])),
        "support_max": int(np.max(fitted["support"])),
        "rhs_finite_all": bool(np.all(fitted["rhs_finite"])),
        "prediction_finite_all": bool(np.all(fitted["prediction_finite"])),
        "coefficient_finite_all": bool(np.all(fitted["coefficient_finite"])),
        "fit_elapsed_sum_s": float(np.sum(fitted["elapsed"])),
    }


def run_targets(target_dir, smoke=False):
    os.makedirs(target_dir, exist_ok=True)
    chunks = []
    total = 0
    sum_coeff = np.zeros(3328, np.float64)
    if smoke:
        meshes = {24: (0, 2, 2)}
    else:
        meshes = TRAIN
    started = time.perf_counter()
    for n, (start, count, chunk_size) in meshes.items():
        coords, mask = c.grid_coords(n), c.binary_boundary_mask(n)
        for chunk_start in range(start, start + count, chunk_size):
            indices = np.arange(chunk_start, min(chunk_start + chunk_size, start + count))
            if smoke:
                parameters = {
                    "cx": np.asarray((0.42, 0.58)), "cy": np.asarray((0.58, 0.42)),
                    "width": np.asarray((0.12, 0.14)), "amplitude": np.asarray((1.2, 1.4)),
                    "nu": np.asarray((0.001, 0.003)),
                    "normalized": np.zeros((2, 5), np.float64),
                }
                initial = np.stack([
                    c.bf.blob_ic(n, parameters["cx"][i], parameters["cy"][i],
                                 parameters["width"][i], parameters["amplitude"][i])
                    for i in range(2)
                ])
                fields = initial[:, None, :]
                health = {"smoke": True, "reported_max_relative_residual": 0.0,
                          "independent_max_relative_residual": 0.0,
                          "seed": None, "draw_count": 2, "indices": indices.tolist()}
            else:
                fields, parameters, health = c.generate_population(
                    n, TARGET_SEED, TARGET_DRAW_COUNT, indices, chunk=4
                )
                if not (
                    np.isfinite(health["reported_max_relative_residual"])
                    and np.isfinite(health["independent_max_relative_residual"])
                    and health["reported_max_relative_residual"] <= 1e-8
                    and health["independent_max_relative_residual"] <= 1e-8
                ):
                    raise SystemExit(f"N{n} training reference health failed")
            fitted = fit_chunk(fields, coords, mask)
            filename = f"targets_N{n}_{indices[0]:04d}_{indices[-1] + 1:04d}.npz"
            row = save_target_chunk(
                os.path.join(target_dir, filename), n, indices,
                fields, parameters, health, fitted,
            )
            chunks.append(row)
            values = fitted["coefficients"].reshape(-1, 3328)
            sum_coeff += np.sum(values, axis=0)
            total += values.shape[0]
            c.log({"target_chunk": filename, "snapshots": values.shape[0],
                   "healthy": row["healthy_count"], "normal_worst": row["normal_worst"]})
    mean = sum_coeff / total
    centered_sumsq = np.zeros(2, np.float64)
    source_min = np.full(2, np.inf)
    source_max = np.full(2, -np.inf)
    for row in chunks:
        with np.load(os.path.join(target_dir, row["basename"]), allow_pickle=False) as data:
            values = np.asarray(data["coefficients"]).reshape(-1, 3328)
        for head, one in enumerate((values[:, :2304], values[:, 2304:])):
            center = mean[:2304] if head == 0 else mean[2304:]
            centered_sumsq[head] += np.sum((one - center) ** 2)
            source_min[head] = min(source_min[head], float(np.min(one)))
            source_max[head] = max(source_max[head], float(np.max(one)))
    sizes = np.asarray((2304, 1024), np.float64)
    rms = np.sqrt(centered_sumsq / (total * sizes))
    scales = np.maximum(rms, 1e-12)
    integrity = bool(
        total == (2 if smoke else TARGET_COUNT)
        and np.all(np.isfinite(mean)) and np.all(np.isfinite(rms))
        and np.all(rms > 0.0)
        and all(
            row["healthy_count"] == row["snapshot_count"]
            and row["normal_worst"] <= s.ORACLE_NORMAL_TOL
            and row["boundary_all"] and row["pou_worst"] == 0.0
            and row["support_min"] == 32 and row["support_max"] == 32
            and row["rhs_finite_all"] and row["prediction_finite_all"]
            and row["coefficient_finite_all"]
            for row in chunks
        )
    )
    return {
        "mode": "excluded_smoke" if smoke else "scientific_train_mix",
        "chunks": chunks, "chunk_count": len(chunks), "snapshot_count": total,
        "expected_snapshot_count": 2 if smoke else TARGET_COUNT,
        "coefficient_mean": mean, "head_rms": rms, "head_scales": scales,
        "source_min": source_min, "source_max": source_max,
        "normalization_definition": "per-control mean; per-head global RMS over centered train controls",
        "integrity_pass": integrity, "elapsed_s": float(time.perf_counter() - started),
    }


def relative_l2(value, control):
    return float(np.linalg.norm(np.asarray(value) - np.asarray(control)) /
                 max(np.linalg.norm(np.asarray(control)), 1e-300))


def run_cost_smoke(mean, scales):
    """Reduced execution smoke for both exact generators and decoder identities.

    The scientific full-trajectory compile is intentionally not represented as a
    timing result here: compiling both 51-state max-one kernels exceeds the local
    one-minute rule.  The cluster P5-D path below is the sole cost evidence.
    """
    n = 32
    parameters = {"cx": np.asarray((0.42,)), "cy": np.asarray((0.58,)),
                  "width": np.asarray((0.12,)), "amplitude": np.asarray((1.2,)),
                  "nu": np.asarray((0.001,))}
    features = c.trajectory_features(parameters, n)[0, :2]
    predictor = p5.init_cost_predictor()
    states = np.tanh(np.asarray(s.apply_mlp(predictor, jnp.asarray(features))))
    coords = jnp.asarray(c.grid_coords(n))
    mask = jnp.asarray(c.binary_boundary_mask(n))
    coarse = jnp.asarray(p5.p3.span_polynomial_table_np(48))
    fine = jnp.asarray(p5.p3.span_polynomial_table_np(32))
    setup, geometry, variance, identity, canonical, gates = {}, {}, {}, {}, {}, {}
    for candidate in p5.CANDIDATES:
        arm = candidate["arm"]
        started = time.perf_counter()
        generator = p5.init_generator(candidate, p5.COST_SEED, nonzero_bias=True)
        coefficients = np.asarray(p5.apply_generator(
            generator, jnp.asarray(states[:, 5:]), jnp.asarray(mean),
            jnp.asarray(scales), candidate,
        ))
        k3 = np.asarray(jax.jit(jax.vmap(
            lambda state, value: p5.p4.decode_one_polynomial_jax(
                state, value, coords, mask, p5.H1, coarse, fine
            )
        ))(jnp.asarray(states), jnp.asarray(coefficients)))
        cox = np.asarray(jax.jit(jax.vmap(
            lambda state, value: p5.p4.decode_one_cox_jax(
                state, value, coords, mask, p5.H1
            )
        ))(jnp.asarray(states), jnp.asarray(coefficients)))
        relative = relative_l2(k3, cox)
        boundary = bool(np.all(k3[:, np.asarray(mask) == 0.0] == 0.0))
        basis = p4d.basis_identity(p5.H1, True)
        try:
            pallas = p4d.pallas_basis_identity(p5.H1, True)
        except Exception as error:
            pallas = {"available": p5.p3.pallas_is_available(), "executed": False,
                      "pass": False, "error": f"{type(error).__name__}: {error}"}
        geometry[arm] = p5.output_geometry(candidate, generator, mean, scales)
        variance[arm] = [{"case_index": 0, "q_std": float(np.std(states[:, 5:])),
                          "coefficient_std": float(np.std(coefficients)),
                          "finite": bool(np.all(np.isfinite(coefficients)))}]
        identity[arm] = [{"case_index": 0,
                          "relative_l2": {"full_fields": relative},
                          "max_relative_l2": relative, "exact_boundary": boundary,
                          "pass": bool(relative <= IDENTITY_TOL and boundary)}]
        canonical[f"{arm}_mandatory"] = [{
            "method": f"{arm}_mandatory", "case_index": 0,
            "finite": bool(np.all(np.isfinite(k3))), "output_shape": list(k3.shape),
            "reduced_smoke_only": True,
        }]
        canonical[f"{arm}_maximum_one"] = [{
            "method": f"{arm}_maximum_one", "case_index": 0,
            "finite": bool(np.all(np.isfinite(k3))), "output_shape": list(k3.shape),
            "reduced_smoke_only": True,
        }]
        setup[arm] = {"basis_identity": basis, "pallas_basis_identity": pallas,
                      "parameter_count": p5._parameter_count(generator),
                      "planning_mac_per_state": candidate["mac_per_state"],
                      "reduced_execution_seconds": float(time.perf_counter() - started)}
        noncollapse = bool(geometry[arm]["pass"] and variance[arm][0]["finite"]
                           and variance[arm][0]["q_std"] >= 1e-3
                           and variance[arm][0]["coefficient_std"] >= 1e-3)
        gates[arm] = {"identity_pass": bool(basis["pass"] and pallas["pass"]
                                             and identity[arm][0]["pass"]),
                      "noncollapse_pass": noncollapse,
                      "reduced_execution_finite": canonical[f"{arm}_mandatory"][0]["finite"],
                      "scientific_promotion_allowed": False}
    return {"status": "excluded_execution_smoke_pass", "reference_health": {"smoke": True},
            "smoke_scope": "both generator primitives, exact output geometry, K3/Cox full-grid decode; no scientific timing",
            "setup": setup, "geometry": geometry, "variance": variance,
            "identity": identity, "canonical_work": canonical, "gates": gates}


def run_cost(mean, scales, smoke=False):
    if smoke:
        return run_cost_smoke(mean, scales)
    # N32 has 1,024 points, exactly divisible by the locked K3 block 128.
    n, steps = 1024, c.NUM_STEPS
    candidates = p5.CANDIDATES
    truth, parameters, reference, dummy = base.generate_live_reference(n)
    features_by_case, representative = [], []
    for case in range(4):
        recovered, indices = c.recover_blob_parameters_fixed_sample(truth[case, 0], n)
        expected = np.asarray((parameters["cx"][case], parameters["cy"][case],
                               parameters["width"][case], parameters["amplitude"][case]))
        recovery = float(np.linalg.norm(recovered - expected) / np.linalg.norm(expected))
        if indices.size > 4096 or recovery > 1e-9:
            raise SystemExit("cold recovery gate failed")
        one = {"cx": recovered[0:1], "cy": recovered[1:2],
               "width": recovered[2:3], "amplitude": recovered[3:4],
               "nu": parameters["nu"][case:case + 1]}
        features_by_case.append(c.trajectory_features(one, n)[0])
        representative.append({"case_index": case, "source_draw_index": case,
                               "sample_count": int(indices.size),
                               "recovery_relative_error": recovery,
                               "features": features_by_case[-1].tolist()})
    case_count = 4
    architecture, setup, identities, geometry, variance = {}, {}, {}, {}, {}
    for candidate in candidates:
        arm = candidate["arm"]
        basis = p4d.basis_identity(p5.H1, smoke)
        try:
            pallas = p4d.pallas_basis_identity(p5.H1, False)
        except Exception as error:
            pallas = {"available": p5.p3.pallas_is_available(), "executed": False,
                      "pass": False, "error": f"{type(error).__name__}: {error}"}
        compiled, one_setup, weak, arguments = p5.compile_online_kernels(
            n, candidate, features_by_case[0], parameters["nu"][0], mean, scales, steps
        )
        architecture[arm] = {"compiled": compiled, "args": arguments}
        setup[arm] = {"kernels": one_setup, "basis_identity": basis,
                      "pallas_basis_identity": pallas,
                      "representative_inputs": representative,
                      "weak_geometry": {"M": candidate["M"], "m": candidate["m"],
                                        "rule": weak["rule"]},
                      "parameter_count": p5._parameter_count(arguments[1]),
                      "planning_mac_per_state": candidate["mac_per_state"]}
        geometry[arm] = p5.output_geometry(candidate, arguments[1], mean, scales)
        identities[arm], variance[arm] = [], []
        for case in range(case_count):
            args = list(arguments)
            args[4] = jnp.asarray(features_by_case[case])
            args[7] = jnp.asarray(parameters["nu"][case])
            args = tuple(args)
            cox, k3 = compiled["identity_cox"](*args), compiled["identity_k3"](*args)
            jax.block_until_ready((cox, k3))
            names = ("full_fields", "current_stencils", "previous_centers", "weak_residual", "rho")
            relative = {name: relative_l2(value, control)
                        for name, value, control in zip(names, k3, cox)}
            boundary = bool(np.all(np.asarray(k3[0])[:, c.binary_boundary_mask(n) == 0] == 0.0))
            identities[arm].append({"case_index": case, "relative_l2": relative,
                                    "max_relative_l2": max(relative.values()),
                                    "exact_boundary": boundary,
                                    "pass": bool(max(relative.values()) <= IDENTITY_TOL and boundary)})
            q = np.asarray(k3[0])  # overwritten below with actual states/coefficients
            mandatory = compiled["mandatory"](*args)
            jax.block_until_ready(mandatory)
            q = np.asarray(mandatory[1])[:, 5:]
            coefficient = np.asarray(mandatory[2])
            variance[arm].append({"case_index": case, "q_std": float(np.std(q)),
                                  "coefficient_std": float(np.std(coefficient)),
                                  "finite": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(coefficient)))})
    methods = ["fom"]
    methods += [f"{row['arm']}_{suffix}" for row in candidates
                for suffix in ("mandatory", "maximum_one")]
    fom = None if smoke else base.bc.make_chain(
        n, base.FOM_OUTER, lin_tol=base.FOM_INNER, preconditioner="helmholtz"
    )[0]

    def invoke(method, case):
        started = time.perf_counter()
        if method == "fom":
            output = fom(jnp.asarray(truth[case, 0]), parameters["nu"][case], dummy, jnp.int32(5))
        else:
            arm, suffix = method.split("_", 1)
            recovered, sample_indices = c.recover_blob_parameters_fixed_sample(truth[case, 0], n)
            if sample_indices.size > 4096 or not np.all(np.isfinite(recovered)):
                raise SystemExit("charged cold recovery failed")
            one = {"cx": recovered[0:1], "cy": recovered[1:2],
                   "width": recovered[2:3], "amplitude": recovered[3:4],
                   "nu": parameters["nu"][case:case + 1]}
            features = c.trajectory_features(one, n)[0]
            args = list(architecture[arm]["args"])
            args[4], args[7] = jnp.asarray(features), jnp.asarray(parameters["nu"][case])
            output = architecture[arm]["compiled"][suffix](*tuple(args))
        jax.block_until_ready(output)
        return output, float(time.perf_counter() - started)

    first_execution, canonical = {}, {}
    for method in methods:
        _, first_execution[method] = invoke(method, 0)
    for method in methods:
        if method != "fom":
            canonical[method] = [p4d.work_record(method, case, invoke(method, case)[0])
                                 for case in range(case_count)]
    for _ in range(TIME_WARM):
        for case in range(case_count):
            for method in methods:
                invoke(method, case)
    burn_count = c.gpu_burn(BURN_SECONDS)
    records = {method: [] for method in methods}
    orders = []
    for repetition in range(TIME_REPS):
        offset = repetition % len(methods)
        order = methods[offset:] + methods[:offset]
        if (repetition // len(methods)) % 2:
            order = list(reversed(order))
        orders.append(order)
        cases = list(range(case_count))
        cases = cases[repetition % case_count:] + cases[:repetition % case_count]
        for case in cases:
            for method in order:
                output, elapsed = invoke(method, case)
                item = {"case_index": case, "repetition": repetition, "elapsed_s": elapsed}
                if method == "fom":
                    item.update(base.fom_grade(output, truth[case]))
                else:
                    item["finite"] = bool(np.all(np.isfinite(np.asarray(output[0]))))
                records[method].append(item)
    position_counts = {method: [sum(order[position] == method for order in orders)
                                for position in range(len(methods))]
                       for method in methods}
    if not all(value == 4 for counts in position_counts.values() for value in counts):
        raise SystemExit("P5-D timing position balance failed")
    summaries = {method: base.summarize_timing(rows, case_count)
                 for method, rows in records.items()}
    fom_rows = [row for row in records["fom"] if row["repetition"] == 0]
    fom_mean = float(np.mean([row["trajectory_relative_l2"] for row in fom_rows]))
    fom_worst = float(np.max([row["trajectory_relative_l2"] for row in fom_rows]))
    fom_healthy = bool(all(row["finite"] and row["breakdowns"] == 0
                           and row["flags_nonzero"] == 0
                           and row["max_returned_relative_residual"] <= base.FOM_OUTER
                           for row in records["fom"]))
    fom_eligible = bool(fom_healthy and fom_mean <= 1e-3 and fom_worst <= 3e-3)
    gates = {}
    for index, candidate in enumerate(candidates):
        arm = candidate["arm"]
        identity = bool(setup[arm]["basis_identity"]["pass"]
                        and setup[arm]["pallas_basis_identity"]["pass"]
                        and all(item["pass"] for item in identities[arm]))
        noncollapse = bool(geometry[arm]["pass"] and all(
            item["finite"] and item["q_std"] >= 1e-3
            and item["coefficient_std"] >= 1e-3 for item in variance[arm]))
        one = {"fom_eligible": fom_eligible, "identity_pass": identity,
               "noncollapse_pass": noncollapse}
        for suffix in ("mandatory", "maximum_one"):
            method = f"{arm}_{suffix}"
            speed = summaries["fom"]["median_elapsed_s"] / summaries[method]["median_elapsed_s"]
            ci = base.clustered_speedup_ci(
                summaries["fom"]["per_case_median_elapsed_s"],
                summaries[method]["per_case_median_elapsed_s"],
                20265100 + 20 * index + (suffix == "maximum_one"),
            )
            work = bool(len(canonical[method]) == 4 and all(
                item["finite"] and item["zero_failures"]
                and item["weak_objective_evaluations"] == 50
                and item["weak_jacobian_evaluations"] == (0 if suffix == "mandatory" else 50)
                and item["trial_residual_evaluations"] == (0 if suffix == "mandatory" else 200)
                for item in canonical[method]))
            memory = setup[arm]["kernels"][suffix]["memory_analysis"]["eligibility_device_bytes"]
            one[suffix] = {"paired_median_speedup": float(speed),
                           "clustered_speedup_ci": ci,
                           "canonical_work_pass": work,
                           "compiled_device_bytes": int(memory),
                           "memory_pass": bool(memory <= 20_000_000_000),
                           "pass": bool(fom_eligible and identity and noncollapse and work
                                        and memory <= 20_000_000_000 and speed >= 10.0 and ci[0] >= 8.0)}
        one["training_cost_license"] = one["mandatory"]["pass"]
        one["correction_classification"] = (
            "correction-capable" if one["maximum_one"]["pass"] else
            "conditional-zero-or-occasional-attempt" if one["mandatory"]["pass"] else "cost-fail"
        )
        gates[arm] = one
    return {"status": "complete", "reference_health": reference,
            "setup": setup, "geometry": geometry, "variance": variance,
            "identity": identities, "canonical_work": canonical,
            "first_execution_after_compile_s": first_execution,
            "burn_count": burn_count, "timing_orders": orders,
            "position_counts": position_counts, "exact_position_balance": True,
            "records": records, "summaries": summaries,
            "fom_accuracy": {"mean": fom_mean, "worst": fom_worst,
                             "healthy": fom_healthy, "eligible": fom_eligible},
            "gates": gates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--p4-json")
    parser.add_argument("--p4-npz")
    parser.add_argument("--p4-audit")
    parser.add_argument("--p4-manifest")
    parser.add_argument("--rank-json")
    parser.add_argument("--rank-script")
    parser.add_argument("--rank-checkpoint")
    parser.add_argument("--prereg")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    if args.smoke:
        bindings = None
    else:
        required = (args.p4_json, args.p4_npz, args.p4_audit, args.p4_manifest,
                    args.rank_json, args.rank_script, args.rank_checkpoint, args.prereg)
        if any(value is None for value in required):
            raise SystemExit("scientific P5-D requires complete P4/rank/prereg chains")
        p4_report, p4_audit, p4_manifest = validate_p4_chain(
            args.p4_json, args.p4_npz, args.p4_audit, args.p4_manifest
        )
        film_path = os.path.join(os.path.dirname(__file__), "deps", "burgers2d-coord-rom", "burgers2d_film.py")
        bh_path = os.path.join(os.path.dirname(__file__), "bh_common.py")
        if p4_manifest.get("code/deps/burgers2d-coord-rom/burgers2d_film.py") != c.sha256(film_path):
            raise SystemExit("Burgers FOM dependency differs from P4 manifest")
        if p4_manifest.get("code/bh_common.py") != c.sha256(bh_path):
            raise SystemExit("bh_common dependency differs from P4 manifest")
        phase5_binding = validate_rank_chain(
            args.rank_json, args.rank_script, args.rank_checkpoint, args.prereg
        )
        immutable = phase5_binding["immutable_inputs"]
        expected_immutable = {
            "phase4_d.json": c.sha256(args.p4_json),
            "phase4_d.npz": c.sha256(args.p4_npz),
            "AUDIT.json": c.sha256(args.p4_audit),
            "MANIFEST.sha256": c.sha256(args.p4_manifest),
        }
        if immutable != expected_immutable:
            raise SystemExit("rank diagnostic does not bind the supplied P4 chain")
        bindings = {
            "P4": p4_binding(args.p4_json, args.p4_npz, args.p4_audit,
                             args.p4_manifest, p4_report, p4_audit),
            "Phase5": phase5_binding,
            "runtime_dependencies": {
                "source_manifest": "P4", "burgers2d_film_sha256": c.sha256(film_path),
                "bh_common_sha256": c.sha256(bh_path),
            },
        }
    targets = run_targets(args.target_dir, args.smoke)
    cost = run_cost(targets["coefficient_mean"], targets["head_scales"], args.smoke)
    arm_license = {arm: bool(targets["integrity_pass"] and gate.get("training_cost_license", False))
                   for arm, gate in cost["gates"].items()}
    next_arm = next((row["arm"] for row in p5.CANDIDATES if arm_license[row["arm"]]), None)
    decision = {
        "target_integrity_pass": bool(targets["integrity_pass"]),
        "arm_training_licenses": arm_license,
        "next_seed11_arm": next_arm,
        "phase5_hard_stop": bool(not args.smoke and next_arm is None),
        "scientific_promotion_allowed": not args.smoke,
    }
    timing_arrays = {}
    if not args.smoke:
        for method, rows in cost["records"].items():
            timing_arrays[f"timing_{method}"] = np.asarray(
                [[item["case_index"], item["repetition"], item["elapsed_s"]] for item in rows],
                np.float64,
            )
    np.savez(args.output_npz,
             coefficient_mean=targets["coefficient_mean"],
             head_rms=targets["head_rms"], head_scales=targets["head_scales"],
             source_min=targets["source_min"], source_max=targets["source_max"],
             **timing_arrays)
    target_report = {key: value for key, value in targets.items()
                     if key not in ("coefficient_mean", "head_rms", "head_scales",
                                    "source_min", "source_max")}
    target_report["normalization"] = {
        "coefficient_mean": targets["coefficient_mean"].tolist(),
        "head_rms": targets["head_rms"].tolist(),
        "head_scales": targets["head_scales"].tolist(),
        "source_min": targets["source_min"].tolist(),
        "source_max": targets["source_max"].tolist(),
        "definition": targets["normalization_definition"],
    }
    report = {
        "status": "excluded_execution_smoke_pass" if args.smoke else "complete",
        "provenance": c.provenance(),
        "config": {"train": {str(n): {"start": start, "count": count, "times": 51,
                                             "chunk_cases": chunk}
                              for n, (start, count, chunk) in TRAIN.items()},
                   "target_seed": TARGET_SEED, "target_draw_count": TARGET_DRAW_COUNT,
                   "target_workers": TARGET_WORKERS, "target_fit_count": TARGET_COUNT,
                   "candidates": list(p5.CANDIDATES), "H1": p5.H1,
                   "time_repetitions": TIME_REPS, "time_warmups": TIME_WARM,
                   "burn_seconds": BURN_SECONDS, "identity_tolerance": IDENTITY_TOL,
                   "normal_tolerance": s.ORACLE_NORMAL_TOL,
                   "model_validation_touched": False, "confirmation_touched": False,
                   "smoke": args.smoke, "f64": True, "matmul_precision": "highest"},
        "bindings": bindings, "train_targets": target_report, "cost_panel": cost,
        "decision": decision, "elapsed_s": float(time.perf_counter() - started),
        "npz": {"basename": os.path.basename(args.output_npz),
                "sha256": c.sha256(args.output_npz)},
    }
    c.save_json(args.output_json, report)
    c.log({"status": report["status"], "decision": decision,
           "elapsed_s": report["elapsed_s"]})


if __name__ == "__main__":
    main()
