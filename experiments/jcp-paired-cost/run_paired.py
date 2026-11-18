"""Approved-development-only NS paired full-query pilot.

All scientific settings come from the released configuration. No job should be
submitted until that configuration and its compact P1 parity inputs are pinned.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
os.environ.setdefault("JAX_ENABLE_X64", "true")
import numpy as np
import jax
import jax.numpy as jnp
import bankio as BI
import diag_floor as D
import head_rom as H
import ns3d_fom as F
import shift_rom as SR
import ns_query as Q
from paired_measurement import Case, QueryResult, Subject, measure_one, run_suite, write_json

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def log(message):
    print(message, flush=True)


def burn():
    x = jnp.ones((1024, 1024), dtype=jnp.float64)*1e-3
    fun = jax.jit(lambda a: a @ a*1e-3+1e-3)
    until = time.perf_counter()+2
    while time.perf_counter() < until:
        x = fun(x)
        x.block_until_ready()


def load_head(path):
    data = np.load(path)
    return ({key: np.asarray(data[f"p_{key}"]) for key in
             ("W1", "b1", "W2", "b2", "W3", "b3", "Ws")}, np.asarray(data["Z"]))


def rebuild_pod(original, n, horizon):
    """The original frozen POD recipe, with sign and coefficient-probe parity."""
    train, _, _ = D.generate(F.parameters(original["train_seed"], original["train_cases"]),
                             n, original["dt_truth"], horizon)
    center = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f)*n))
    for c in range(len(train)):
        for t in range(train.shape[1]):
            train[c, t] = np.asarray(center(jnp.asarray(train[c, t])))
    raw, eigenvalues, orth = D.pod_basis(train.reshape(-1, 3*n**3), 64, original["gram_block"])
    del train
    gc.collect()
    basis, _ = D.orthonormalize_prefix(raw, 64)
    del raw
    probe = np.load(HERE / "assets/pod_bank_probe.npz")
    fields, _, _ = D.generate(F.parameters(original["train_seed"], int(probe["cases"])),
                              n, original["dt_truth"], horizon)
    coefficients = np.stack([basis.T @ np.asarray(center(jnp.asarray(f))).ravel()
                             for case in fields for f in case])
    signs = np.sign(np.sum(coefficients*probe["coefficients"], axis=0))
    gap = float(np.linalg.norm(coefficients*signs-probe["coefficients"])
                / np.linalg.norm(probe["coefficients"]))
    if gap > 1e-8:
        raise RuntimeError(f"POD regenerated bank differs from frozen probe: {gap}")
    return np.ascontiguousarray(basis*signs), {
        "coefficient_probe_relative": gap, "rank": 64,
        "training_seed": original["train_seed"], "training_cases": original["train_cases"],
        "recipe": "original snapshot Gram, eigh, prefix QR and frozen probe sign correction",
        "new_training": False, "pod_orthogonality_before_qr": orth,
        "spectrum": eigenvalues.tolist()}


def save_assets(path, basis, operators, params, codes):
    path.mkdir(parents=True, exist_ok=False)
    files = {"basis": basis, "A": operators["A"], "T": operators["T"],
             "lam": operators["lam"], "Dd": operators["Dd"], "codes": codes}
    files.update({"p_"+name: value for name, value in params.items()})
    for name, value in files.items():
        np.save(path / f"{name}.npy", np.asarray(value))
    write_json(path / "MANIFEST.json", {p.name: sha(p) for p in sorted(path.glob("*.npy"))})


def activate_assets(path, rank, kind, rows):
    arrays = {p.stem: np.load(p, allow_pickle=False) for p in path.glob("*.npy")}
    params = {name[2:]: jnp.asarray(value) for name, value in arrays.items() if name.startswith("p_")} if kind == "head" else {}
    codes = jnp.asarray(arrays["codes"]) if kind == "head" else jnp.zeros((0, 0))
    basis = jnp.asarray(np.ascontiguousarray(arrays["basis"][:, :rank]))
    A = jnp.asarray(np.ascontiguousarray(arrays["A"][:rows, :rank]))
    T = jnp.asarray(np.ascontiguousarray(arrays["T"][:rows, :rank, :rank]))
    lam = jnp.asarray(arrays["lam"][:rows])
    Dd = jnp.asarray(np.ascontiguousarray(arrays["Dd"][:, :rows, :rank]))
    candidates = H.head_apply(params, codes) if kind == "head" else jnp.zeros((0, 0))
    return jax.block_until_ready((basis, A, T, lam, Dd, params, codes, candidates))


def native_subject(name, family, kind, rank, cfg, asset_path, override=None):
    settings = dict(cfg["rom"])
    settings.update(override or {})
    steps = round(cfg["horizon"] / settings["dt"])
    compiled = Q.make_query(kind, n=cfg["n"], rank=rank, latent=8, dt=settings["dt"],
        steps=steps, output_every=steps//5, budget=settings["budget"],
        fit_starts=settings["fit_starts"], roundoff_repair=settings["roundoff_repair"])
    assets = activate_assets(asset_path, rank, kind, settings["rows"])
    if assets[1].shape != (settings["rows"], rank) or assets[2].shape != (settings["rows"], rank, rank):
        raise RuntimeError("deployed operator row/rank slice differs from registered numerical configuration")

    def use(u0, viscosity, active):
        output, stats, initial = compiled(jnp.asarray(u0), jnp.asarray(viscosity), *active,
            jnp.asarray(settings["gtol"]), jnp.asarray(settings["fit_gtol"]))
        fields, stats, initial = np.asarray(output), np.asarray(stats), np.asarray(initial)
        valid = bool(np.all(stats[:, 0] == 1) and initial[0] == 1
                     and np.all(np.isfinite(stats)) and np.all(np.isfinite(initial)))
        diagnostics = {
            "step_reasons": stats[:, 0].astype(int).tolist(),
            "step_gradients": stats[:, 1].tolist(),
            "step_iterations": stats[:, 3].astype(int).tolist(),
            "step_roundoff_acceptances": stats[:, 5].astype(int).tolist(),
            "initial_fit_reason": int(initial[0]), "initial_fit_gradient": float(initial[1]),
            "initial_fit_iterations": int(initial[3]),
            "initial_fit_raw_winner": int(initial[6]), "initial_fit_selected": int(initial[7]),
            "initial_fit_tie_band": float(initial[8]),
            "initial_fit_raw_selected_coefficient_gap": float(initial[9]),
        }
        return QueryResult(fields, valid, diagnostics)

    def query(u0, viscosity):
        return use(u0, viscosity, assets)

    def cold(u0, viscosity):
        # clear_caches is called by the outer cold block before the timer.
        # Asset disk loading, device activation, fitting, compilation and output
        # are all inside this directly timed callback.
        active = activate_assets(asset_path, rank, kind, settings["rows"])
        return use(u0, viscosity, active)

    return Subject(name, family, cfg["n"], query, {
        "kind": kind, "rank": rank, "latent": 8 if kind == "head" else None,
        "actual_scalar_weak_rows": int(assets[1].shape[0]),
        "actual_online_unknowns_including_frame": (8 if kind == "head" else rank)+3,
        "solver": "compiled adaptive diagonally scaled LM; returned-state stationarity",
        "settings": settings, "assets_manifest_sha256": sha(asset_path / "MANIFEST.json"),
        "initialization": "native from same original physical input field; recomputed each call",
    }, cold)


def classical_subject(candidate, cfg):
    mesh, steps, scheme = candidate["mesh"], candidate["steps"], candidate["scheme"]
    compiled = Q.make_classical(scoring_mesh=cfg["n"], solver_mesh=mesh,
                                horizon=cfg["horizon"], steps=steps, scheme=scheme)
    geometry = F.geometry(mesh)

    def query(u0, viscosity):
        fields = np.asarray(compiled(jnp.asarray(u0), jnp.asarray(viscosity), geometry))
        # Field-finiteness scoring is performed identically outside the stopwatch
        # by the harness for every family. This explicit method has no iterative
        # solve termination flag to inspect inside the query.
        return QueryResult(fields, True,
                            {"scheme": scheme, "steps": steps, "solver_mesh": mesh,
                             "nonlinear_iteration_policy": "explicit spectral nonlinearity"})

    def cold(u0, viscosity):
        geo = F.geometry(mesh)
        fields = np.asarray(compiled(jnp.asarray(u0), jnp.asarray(viscosity), geo))
        return QueryResult(fields, True,
                            {"scheme": scheme, "steps": steps, "solver_mesh": mesh})

    return Subject(f"fom_m{mesh}_{scheme}_s{steps}", "classical", mesh, query,
                   {"scheme": scheme, "steps": steps, "same_output_mesh": cfg["n"],
                    "input_and_output_transforms_timed": True}, cold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not cfg.get("parent_release") or not cfg.get("approved_reference_states"):
        raise RuntimeError("scientific release and admitted parity states are required before running")
    allowed = list(range(16)) if cfg.get("cohort_scope") == "original-dev16" else [0, 1]
    if cfg["eval_seed"] != 202609202 or cfg["case_ids"] != allowed:
        raise RuntimeError("only the registered original development prefix is allowed")
    if cfg["n"] != 64 or cfg["horizon"] != .2:
        raise RuntimeError("initial cost pilot is the original frozen NS64 problem")
    if jax.default_backend() != "gpu" or not jax.config.jax_enable_x64:
        raise RuntimeError("GPU f64 backend required")
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise RuntimeError("highest matmul precision required")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    for rel, entry in manifest["files"].items():
        if sha(HERE / rel) != entry["sha256"]:
            raise RuntimeError("frozen source or asset changed: "+rel)
    provenance = {"config": cfg, "config_sha256": sha(args.config), "source_manifest": manifest,
        "driver_sha256": sha(__file__), "query_sha256": sha(HERE / "ns_query.py"),
        "measurement_sha256": sha(HERE / "paired_measurement.py"),
        "source_commit": os.environ.get("SOURCE_COMMIT"), "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                         "--format=csv,noheader"], text=True).strip(),
        "backend": jax.default_backend(), "x64": jax.config.jax_enable_x64,
        "precision": str(jax.config.jax_default_matmul_precision)}
    write_json(args.out / "provenance.json", provenance)
    setup = []
    start = time.perf_counter()
    original = json.loads((HERE / "configs/original_dev64.json").read_text())
    bank_file = HERE / "assets/bank_selected.npz"
    bank_data = np.load(bank_file)
    G, bank_info = BI.mesh_bank(BI.load_params(bank_file), bank_data["T"], cfg["n"],
                               chunk=32768, abort=.05)
    bank_info.pop("lowdin_factor")
    log("building learned-bank weak operators")
    operators, checks = SR.build_operators_fast(G, cfg["n"], cfg["rom"]["rows"], check_modes=8)
    params, codes = load_head(HERE / "assets/head_k8_n64.npz")
    learned_assets = args.out / "deployed_assets/learned"
    save_assets(learned_assets, G, operators, params, codes)
    setup.append({"phase": "learned mesh bank and exact operators", "wall_seconds": time.perf_counter()-start,
                  "operator_checks": checks, "bank_info": bank_info, "timing_scope": "offline setup including compilation"})
    start = time.perf_counter()
    log("regenerating original POD bank from training seed")
    pod_G, pod_info = rebuild_pod(original, cfg["n"], cfg["horizon"])
    pod_rows = max([cfg["rom"]["rows"]] + [v["rows"] for v in cfg.get("pod_head_settings", [])])
    pod_ops, pod_checks = SR.build_operators_fast(pod_G, cfg["n"], pod_rows, check_modes=8)
    pod_params, pod_codes = load_head(HERE / "assets/pod_head_k8_n64.npz")
    pod_assets = args.out / "deployed_assets/pod"
    save_assets(pod_assets, pod_G, pod_ops, pod_params, pod_codes)
    setup.append({"phase": "frozen POD recipe regeneration and exact operators", "wall_seconds": time.perf_counter()-start,
                  "pod_info": pod_info, "operator_checks": pod_checks, "timing_scope": "offline setup including compilation"})
    write_json(args.out / "offline_setup.json", setup)
    dt_truth = cfg["truth_dt"]
    steps_truth = round(cfg["horizon"] / dt_truth)
    truth_run = F.make_solver(dt_truth, steps_truth, steps_truth//5)
    fine_run = F.make_solver(dt_truth/2, 2*steps_truth, 2*steps_truth//5)
    geometry = F.geometry(cfg["n"])
    parameters = F.parameters(cfg["eval_seed"], max(cfg["case_ids"])+1)
    cases, reference = [], []
    for index in cfg["case_ids"]:
        parameter = parameters[index]
        initial = F.initial(cfg["n"], parameter)
        truth = np.asarray(truth_run(jnp.asarray(initial), float(parameter[-1]), geometry))
        finer = np.asarray(fine_run(jnp.asarray(initial), float(parameter[-1]), geometry))
        delta = float(np.max(np.linalg.norm((truth-finer).reshape(6, -1), axis=1))/np.linalg.norm(initial))
        # The input field supplied to every query is exactly the reference t0 field.
        # Reprojection from analytic u0 must only change roundoff, recorded here.
        initial_projection_gap = float(np.linalg.norm(truth[0]-initial)/np.linalg.norm(initial))
        if initial_projection_gap > 1e-12 or delta > cfg["reference_absolute_gate"]:
            raise RuntimeError("truth time refinement or physical input parity failed")
        cases.append(Case(index, truth[0].copy(), float(parameter[-1]), truth))
        reference.append({"case": index, "parameter": parameter.tolist(), "truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
                          "dt_refinement_max": delta, "initial_projection_gap": initial_projection_gap})
    write_json(args.out / "references.json", reference)
    subjects = [native_subject("learned_head", "target", "head", 64, cfg, learned_assets),
                native_subject("learned_span64", "target", "span", 64, cfg, learned_assets)]
    for setting in cfg.get("pod_head_settings", [{"rows": cfg["rom"]["rows"], "dt": cfg["rom"]["dt"]}]):
        name = f"pod_head_m{setting['rows']}_dt{setting['dt']:g}"
        subjects.append(native_subject(name, "pod", "head", 64, cfg, pod_assets, setting))
    for rank in cfg["pod_span_ranks"]:
        subjects.append(native_subject(f"pod_span{rank}", "pod", "span", rank, cfg, pod_assets))
    subjects.extend(classical_subject(candidate, cfg) for candidate in cfg["classical_candidates"])

    # Every implementation is evaluated before timing. The target field output
    # must match P1's admitted native head/span coefficients and centers.
    parity, warmups = [], []
    for subject in subjects:
        log(f"compile and numerical check: {subject.name}")
        for case in cases:
            started = time.perf_counter()
            value = subject.query(case.initial, case.viscosity)
            warmups.append({"subject": subject.name, "case": case.case_id,
                            "first_invocation_seconds": time.perf_counter()-started,
                            "numerical_valid": value.numerical_valid,
                            "solver_diagnostics": value.diagnostics})
            if subject.name in ("learned_head", "learned_span64"):
                entry = cfg["approved_reference_states"][subject.name][str(case.case_id)]
                state_path = HERE / entry["file"]
                if sha(state_path) != entry["sha256"]:
                    raise RuntimeError("admitted P1 state checksum changed")
                state = np.load(state_path)
                stride = (len(state["coefficients"])-1)//5
                expected = np.stack([np.asarray(SR.shift_field((jnp.asarray(G) @ jnp.asarray(a)).reshape(3, cfg["n"], cfg["n"], cfg["n"]), jnp.asarray(c)*cfg["n"]))
                    for a, c in zip(state["coefficients"][::stride], state["centers"][::stride])])
                error = float(np.max(np.linalg.norm((value.fields-expected).reshape(6, -1), axis=1))/np.linalg.norm(case.initial))
                physical_error = float(np.max(np.linalg.norm((value.fields-case.truth)[1:].reshape(5, -1), axis=1))/np.linalg.norm(case.initial))
                refinement = next(x["dt_refinement_max"] for x in reference if x["case"] == case.case_id)
                reference_bound = min(cfg["reference_absolute_gate"], cfg["reference_fraction_of_target_error"]*physical_error)
                passed = bool(value.numerical_valid and error <= cfg["implementation_parity_gate"]
                              and refinement <= reference_bound)
                parity.append({"subject": subject.name, "case": case.case_id,
                               "max_relative_field_difference": error, "passed": passed,
                               "physical_evolved_max_error": physical_error,
                               "reference_refinement_difference": refinement,
                               "reference_refinement_bound": reference_bound,
                               "pinned_reference": entry, "solver_diagnostics": value.diagnostics})
                if not passed:
                    write_json(args.out / "implementation_parity.json", parity)
                    write_json(args.out / "first_invocations.json", warmups)
                    raise RuntimeError("compiled query failed admitted scientific reference parity")
        write_json(args.out / "first_invocations.json", warmups)
        write_json(args.out / "implementation_parity.json", parity)
    write_json(args.out / "implementation_parity.json", parity)
    write_json(args.out / "first_invocations.json", warmups)
    result = run_suite(subjects, cases, target_names=["learned_head", "learned_span64"],
        scoring_mesh=cfg["n"], out=args.out / "paired", burn=burn,
        selection_repetitions=cfg["selection_repetitions"],
        confirmation_repetitions=cfg["confirmation_repetitions"],
        seed=cfg["timing_order_seed"], drift_gate_ratio=cfg["drift_gate_ratio"], metadata=provenance)
    selected = set(("learned_head", "learned_span64"))
    frozen = json.loads((args.out / "paired/frozen_comparators.json").read_text())["comparators"]
    for groups in frozen.values():
        selected.update(x["selected"] for x in groups.values() if x["selected"] is not None)
    cold = []
    for subject in subjects:
        if subject.name not in selected:
            continue
        jax.clear_caches()
        burn()
        row = measure_one(subject, cases[0], cold=True)
        row["scope"] = "direct model activation, compilation and first instrumented query in an existing JAX/CUDA process; excludes runtime/context startup; OS page cache not flushed"
        cold.append(row)
        write_json(args.out / "cold_deployment.json", cold)
    result["cold_deployment"] = {"records_file": "../cold_deployment.json",
        "scope": "asset activation, compilation and first instrumented query only; existing runtime/context and OS page cache"}
    result["instrumentation_scope"] = "query timer includes returned solver diagnostics conversion; all families return same host physical fields"
    write_json(args.out / "paired/result.json", result)
    log("paired pilot complete; speed claims remain subject to collector log/checksum audit")


if __name__ == "__main__":
    main()
