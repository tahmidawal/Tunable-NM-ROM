"""Frozen head/library transfer with paired full-query coarse DST envelope."""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import heat_core as hc
from run_pilot import assemble, block, burn_in, dump
from runtime_paths import build_paths, parity
from transfer_core import field_hash, restrict_input, interpolation_tables, make_fom, host_outputs, unique_solvers
from verify_heat import verification, restrict
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config-transfer.json")))
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    cfg = json.loads(Path(args.config).with_name(settings["base_config"]).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    (out/"fields").mkdir(); (out/"checkpoints").mkdir(); (out/"assembly").mkdir()
    assert jax.default_backend() == "gpu" and jax.config.jax_enable_x64
    assert os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    print("jax_backend=gpu x64=True precision=highest", flush=True)
    source = json.loads(Path("SOURCE-MANIFEST.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    started_run = time.perf_counter()
    result = dict(schema=settings["schema"], config=cfg, settings=settings, source_manifest=source,
                  metadata=dict(job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"),
                                gpu=jax.devices()[0].device_kind, backend="gpu", x64=True, precision="highest", jax=jax.__version__),
                  complete=False, models=[], cases=[], rows=[], setups=[], reference_fields=[], case_fields=[], parity_gates=[], warmups=[],
                  reference_evidence=dict(kind=settings["reference_evidence_kind"], rigorous_relative_bound=None),
                  query_contract=settings["initial_output_policy"], boundary_policy="Full interior arrays returned; all boundary nodes identically zero for both methods.")
    def save_field(a):
        a = np.ascontiguousarray(a)
        digest = field_hash(a); path = out/"fields"/f"{digest}.npz"
        if not path.exists(): np.savez_compressed(path, field=a)
        return dict(path=str(path.relative_to(out)), sha256_array=digest, shape=list(a.shape), dtype=a.dtype.str)
    models = {}
    for spec in settings["checkpoints"]:
        blob = Path(spec["name"]+".pkl").read_bytes()
        assert hashlib.sha256(blob).hexdigest() == spec["sha256"]
        checkpoint = pickle.loads(blob)
        assert checkpoint["config"] == cfg
        assert checkpoint["codes"].shape == (160*len(cfg["times"]), cfg["k"])
        params, codes = jax.tree.map(jnp.asarray, checkpoint["params"]), jnp.asarray(checkpoint["codes"])
        models[spec["name"]] = params, codes
        path = out/"checkpoints"/(spec["name"]+".pkl"); path.write_bytes(blob)
        result["models"].append(dict(**spec, frozen_weights=True, frozen_codes=True, output_path=str(path.relative_to(out))))
    base, base_codes = next(iter(models.values()))
    for params, _ in models.values():
        for key in ("B", "g", "out_scale"):
            for a, b in zip(jax.tree.leaves(base[key]), jax.tree.leaves(params[key])): np.testing.assert_array_equal(a, b)
    # No optimization or initializer-library mutation in this program.
    train = np.concatenate((hc.sample_family(790710, 32, cfg), hc.sample_family(790713, 128, cfg)))
    for cohort in settings["cohorts"]:
        for i, draw in enumerate(hc.sample_family(cohort["seed"], cohort["count"], cfg)):
            assert not any(np.array_equal(draw, a) for a in train)
            result["cases"].append(dict(case=len(result["cases"]), cohort=cohort["name"], cohort_index=i, seed=cohort["seed"], draw=draw.tolist()))
    assert len({tuple(c["draw"]) for c in result["cases"]}) == len(result["cases"])
    result["validation_in_training_or_lookup"] = False
    result["final_cohort_opened"] = False
    result["verification"] = verification(cfg)
    times = jnp.asarray(cfg["times"]); later_times = times[1:]
    nc, nf = settings["reference_intervals"]
    assert nf % nc == 0 and nc >= max(settings["requested_intervals"])
    # Preserve the complete nested continuum-spectral pair, not only an error scalar.
    for case in result["cases"]:
        references = []
        for n in (nc, nf):
            u0 = hc.initial_field(jnp.asarray(hc.coords(n)), case["draw"]).reshape(n-1, n-1)
            references.append(np.asarray(block(hc.propagate(u0, hc.eigenvalues(n, True), times, cfg["diffusivity"]))))
        metrics = hc.error_metrics(references[0], restrict(references[1], nf, nc), nc)
        assert max(metrics["relative_current"]) < cfg["reference_uncertainty_budget"]
        result["reference_fields"].append(dict(case=case["case"], coarse_intervals=nc, fine_intervals=nf,
                                               coarse=save_field(references[0]), fine=save_field(references[1]), refinement=metrics))
        del references
        dump(out/"results.json", result)
        print("reference", case["case"], "passed", flush=True)
    result["reference_evidence"]["empirical_relative_delta"] = max(max(r["refinement"]["relative_current"]) for r in result["reference_fields"])
    for n in settings["requested_intervals"]:
        arrays, setup = assemble(base, base_codes, n, cfg)
        setup["rank"] = int(np.linalg.matrix_rank(np.asarray(arrays["triangular"])))
        np.savez_compressed(out/"assembly"/f"n{n}.npz", triangular=np.asarray(arrays["triangular"]), matrix=np.asarray(arrays["matrix"]))
        setup["arrays_path"] = f"assembly/n{n}.npz"
        result["setups"].append(setup)
        lam = hc.eigenvalues(n); m = cfg["modes_per_axis"]
        mode_lam = lam[:m, :m].reshape(-1)
        paths = build_paths(dict(cfg, gradient_tolerance=settings["gradient_tolerance"]), settings["dt"])
        libraries = {name: block(hc.sc.head(p, z)@arrays["triangular"].T) for name, (p, z) in models.items()}
        fom = {}
        for solver in unique_solvers(n, settings["coarse_solver_intervals"]):
            indices, weights = map(jnp.asarray, interpolation_tables(n, solver))
            fom[f"fom_dst_{solver}"] = dict(solver=solver, kernel=make_fom(n, solver), lam=hc.eigenvalues(solver), indices=indices, weights=weights,
                aliases=(["same_grid"] if solver == n else [])+(["coarse_envelope"] if solver in settings["coarse_solver_intervals"] else []))
        names = list(fom)+list(models)
        for case in result["cases"]:
            cid = case["case"]
            u0 = np.ascontiguousarray(hc.initial_field(jnp.asarray(hc.coords(n)), case["draw"])).reshape(n-1, n-1)
            discrete = np.asarray(block(hc.propagate(jnp.asarray(u0), lam, times, cfg["diffusivity"])))
            refs = result["reference_fields"][cid]
            physical = np.ascontiguousarray(restrict(np.load(out/refs["fine"]["path"])["field"], nf, n))
            coarse_ref = np.ascontiguousarray(restrict(np.load(out/refs["coarse"]["path"])["field"], nc, n))
            common_ref = restrict(physical, n, settings["observation_intervals"])
            refinement_full = hc.error_metrics(coarse_ref, physical, n)
            refinement_common = hc.error_metrics(restrict(coarse_ref, n, settings["observation_intervals"]), common_ref, settings["observation_intervals"])
            np.testing.assert_array_equal(physical[0], u0)
            result["case_fields"].append(dict(intervals=n, case=cid, initial=save_field(u0), discrete=save_field(discrete), physical=save_field(physical),
                reference_refinement_full=refinement_full, reference_refinement_common=refinement_common,
                spatial_discrete_vs_physical=hc.error_metrics(discrete, physical, n)))
            del coarse_ref
            def invocation(label):
                started = time.perf_counter(); phases = {}
                if label in fom:
                    arm = fom[label]
                    input_host = restrict_input(u0, n, arm["solver"])
                else: input_host = u0
                inp = block(jax.device_put(input_host)); phases["input_restriction_transfer_seconds"] = time.perf_counter()-started
                begin = time.perf_counter()
                if label in fom:
                    device_fields = block(arm["kernel"](inp, arm["lam"], later_times, cfg["diffusivity"], arm["indices"], arm["weights"]))
                    aux = None
                else:
                    p, codes = models[label]
                    device_fields, initial_info, step_info, zs = block(paths["compiled"](p, arrays["projection"], arrays["triangular"],
                        libraries[label], codes, arrays["bank"], arrays["matrix"], mode_lam, inp))
                    aux = initial_info, step_info, zs
                phases["device_solve_readout_seconds"] = time.perf_counter()-begin
                begin = time.perf_counter()
                fields = host_outputs(u0, device_fields) if label in fom else np.ascontiguousarray(device_fields).reshape(len(times), n-1, n-1)
                phases["full_host_output_seconds"] = time.perf_counter()-begin
                phases["query_seconds"] = time.perf_counter()-started
                # Solver diagnostics are not requested outputs; copy after query timing.
                solver_info = {}
                if aux is not None:
                    initial_info, step_info, zs = jax.device_get(aux)
                    solver_info = dict(initial_fits=initial_info.tolist(), steps=step_info.tolist(), latent=zs.tolist())
                return fields, phases, solver_info
            for label in names:
                started = time.perf_counter(); fields, _, solver = invocation(label)
                result["warmups"].append(dict(intervals=n, case=cid, method=label, seconds=time.perf_counter()-started))
                if label in models:
                    p, codes = models[label]
                    z0, initial = paths["initialize"](p, arrays["projection"], arrays["triangular"], libraries[label], codes, jnp.asarray(u0))
                    zs, steps = paths["rollout"](p, arrays["matrix"], hc.cn_factor(mode_lam, settings["dt"], cfg["diffusivity"]), z0)
                    modular = paths["readout"](p, arrays["bank"], zs)
                    gate = parity((np.asarray(modular).reshape(fields.shape), initial, steps, zs),
                        (fields, np.asarray(solver["initial_fits"]), np.asarray(solver["steps"]), np.asarray(solver["latent"])),
                        settings["field_parity_tolerance"], settings["latent_parity_tolerance"])
                    result["parity_gates"].append(dict(intervals=n, case=cid, method=label, **gate))
                    assert gate["passed"], gate
            rows = {label: dict(intervals=n, case=cid, cohort=case["cohort"], method=label, model=label if label in models else "fom",
                solver_intervals=fom[label]["solver"] if label in fom else n,
                aliases=fom[label]["aliases"] if label in fom else [],
                gradient_tolerance=settings["gradient_tolerance"] if label in models else None, repetitions=[]) for label in names}
            for repetition in range(settings["timing_repetitions"]):
                burn_in(); capture = {}
                order = names if repetition%2 == 0 else names[::-1]
                for label in order: capture[label] = invocation(label)
                # CPU metrics/compression cannot cool the GPU between paired timed arms.
                for label in names:
                    fields, phases, solver = capture[label]
                    digest = field_hash(fields)
                    previous = rows[label]["repetitions"]
                    field_record = previous[0]["field"] if previous and previous[0]["field"]["sha256_array"] == digest else save_field(fields)
                    rows[label]["repetitions"].append(dict(repetition=repetition, invocation_order=order, phases=phases, solver=solver,
                        field=field_record, full_field_sha256=digest,
                        vs_same_grid=hc.error_metrics(fields, discrete, n), vs_physical_per_grid=hc.error_metrics(fields, physical, n),
                        vs_physical_common_grid=hc.error_metrics(restrict(fields, n, settings["observation_intervals"]), common_ref, settings["observation_intervals"])))
                del capture
            for label in names:
                result["rows"].append(rows[label])
                print("timed", n, cid, label, np.median([r["phases"]["query_seconds"] for r in rows[label]["repetitions"]]), flush=True)
            dump(out/"results.json", result)
        del arrays, libraries, fom
        # Release shape-specific compiled programs before building the next mesh.
        jax.clear_caches()
    result["timed_invocations"] = sum(len(r["repetitions"]) for r in result["rows"])
    assert result["timed_invocations"] == settings["estimated_timed_calls"]
    result["elapsed_seconds"] = time.perf_counter()-started_run
    result["complete"] = True
    dump(out/"results.json", result)
    print("TRANSFER PILOT COMPLETE", flush=True)


if __name__ == "__main__": main()
