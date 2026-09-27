"""Frozen-checkpoint LM tolerance and whole-query compilation experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import heat_core as hc
from runtime_paths import build_paths, parity
from run_pilot import assemble, block, burn_in, dump
from verify_heat import verification, restrict
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config-runtime.json")))
    parser.add_argument("--checkpoint", default="checkpoint.pkl")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    cfg = json.loads(Path(args.config).with_name(settings["base_config"]).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/"fields").mkdir()
    dev = jax.devices()[0]
    assert jax.default_backend() == "gpu" and jax.config.jax_enable_x64
    assert os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") == "highest"
    print(f"jax_backend=gpu device={dev} x64=True precision=highest", flush=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert checkpoint_hash == settings["checkpoint_sha256"]
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert checkpoint["config"] == cfg
    params = jax.tree.map(jnp.asarray, checkpoint["params"])
    codes = jnp.asarray(checkpoint["codes"])
    source = json.loads(Path("SOURCE-MANIFEST.json").read_text())
    for path, expected in source["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    result = dict(schema=settings["schema"], settings=settings, config=cfg,
                  source_manifest=source, checkpoint_sha256=checkpoint_hash,
                  metadata=dict(backend="gpu", x64=True, precision="highest", gpu=dev.device_kind,
                                job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"), jax=jax.__version__),
                  complete=False, rows=[], setups=[], parity_gates=[], compile_and_warmups=[],
                  reference_evidence=dict(kind=settings["reference_evidence"], rigorous_relative_bound=None),
                  query_contract="Host float64 full interior initial field to host float64 full interior fields at all six fixed times, homogeneous boundary implied. Data generation/setup/compilation separate. Initial projection, both latent fitting starts, evolution, field readout, input/output transfer charged. Counters transferred after query timer for every path.")

    def save_field(field):
        field = np.ascontiguousarray(field)
        digest = hashlib.sha256(str((field.shape, field.dtype.str)).encode()+field.tobytes()).hexdigest()
        path = out/"fields"/f"{digest}.npz"
        if not path.exists():
            np.savez_compressed(path, field=field)
        return dict(path=str(path.relative_to(out)), sha256_array=digest, shape=list(field.shape), dtype=field.dtype.str)

    started = time.perf_counter()
    result["verification"] = verification(cfg)
    result["verification_seconds"] = time.perf_counter()-started
    result["reference_evidence"]["empirical_relative_delta"] = max(result["verification"]["reference_spectral_refinement_current_errors"])
    draws = hc.sample_family(cfg["validation_seed"], cfg["n_validation"], cfg)
    result["validation_draws"] = draws.tolist()
    times = jnp.asarray(cfg["times"])
    fine = max(cfg["reference_intervals"])
    physical = []
    result["reference_fields"] = []
    for case, draw in enumerate(draws):
        u0 = hc.initial_field(jnp.asarray(hc.coords(fine)), draw).reshape(fine-1, fine-1)
        field = np.asarray(hc.propagate(u0, hc.eigenvalues(fine, True), times, cfg["diffusivity"]))
        physical.append(field)
        result["reference_fields"].append(dict(case=case, intervals=fine, field=save_field(field)))
    dump(out/"results.json", result)
    observation = settings["observation_intervals"]

    for n in cfg["evaluation_intervals"]:
        arrays, setup = assemble(params, codes, n, cfg)
        result["setups"].append(setup)
        m = cfg["modes_per_axis"]
        lam = hc.eigenvalues(n)
        mode_lam = lam[:m, :m].reshape(-1)
        paths = {}
        for tol in settings["gradient_tolerances"]:
            local_cfg = dict(cfg, gradient_tolerance=tol)
            paths[tol] = build_paths(local_cfg, settings["dt"])
        names = ["fom_dst_exact_time"]
        subjects = {}
        for tol in settings["gradient_tolerances"]:
            for path in settings["paths"]:
                name = f"rom_{path}_gtol{tol:g}"
                names.append(name)
                subjects[name] = (tol, path)

        for case, draw in enumerate(draws):
            u0 = np.asarray(hc.initial_field(jnp.asarray(hc.coords(n)), draw)).reshape(n-1, n-1)
            discrete = np.asarray(hc.propagate(jnp.asarray(u0), lam, times, cfg["diffusivity"]))
            reference = restrict(physical[case], fine, n)
            common_reference = restrict(physical[case], fine, observation)
            result.setdefault("case_fields", []).append(dict(intervals=n, case=case,
                                                             initial=save_field(u0), discrete=save_field(discrete),
                                                             physical=save_field(reference)))

            def invocation(name):
                phases = {}
                started = time.perf_counter()
                device_input = block(jax.device_put(u0))
                phases["input_seconds"] = time.perf_counter()-started
                if name == "fom_dst_exact_time":
                    begin = time.perf_counter()
                    field_device = block(hc.propagate(device_input, lam, times, cfg["diffusivity"]))
                    phases["evolve_and_readout_seconds"] = time.perf_counter()-begin
                    auxiliary = None
                else:
                    tol, path = subjects[name]
                    pipe = paths[tol]
                    if path == "compiled":
                        begin = time.perf_counter()
                        field_device, initial_info, step_info, zs = block(pipe["compiled"](
                            params, arrays["projection"], arrays["triangular"], arrays["library"], codes,
                            arrays["bank"], arrays["matrix"], mode_lam, device_input))
                        phases["complete_device_seconds"] = time.perf_counter()-begin
                    else:
                        begin = time.perf_counter()
                        z0, initial_info = block(pipe["initialize"](params, arrays["projection"], arrays["triangular"],
                                                                 arrays["library"], codes, device_input))
                        phases["initial_fit_seconds"] = time.perf_counter()-begin
                        begin = time.perf_counter()
                        # Preserve pilot01 control: factor construction is inside evolution timing.
                        zs, step_info = block(pipe["rollout"](params, arrays["matrix"],
                                            hc.cn_factor(mode_lam, settings["dt"], cfg["diffusivity"]), z0))
                        phases["evolution_seconds"] = time.perf_counter()-begin
                        begin = time.perf_counter()
                        field_device = block(pipe["readout"](params, arrays["bank"], zs))
                        phases["readout_seconds"] = time.perf_counter()-begin
                    auxiliary = (initial_info, step_info, zs)
                begin = time.perf_counter()
                fields = np.asarray(field_device).reshape(len(times), n-1, n-1)
                phases["output_transfer_seconds"] = time.perf_counter()-begin
                phases["query_seconds"] = time.perf_counter()-started
                if auxiliary is not None:
                    initial_info, step_info, zs = jax.device_get(auxiliary)
                    info = dict(initial_fits=initial_info.tolist(), steps=step_info.tolist(), latent=zs.tolist())
                    parity_tuple = (fields, initial_info, step_info, zs)
                else:
                    info, parity_tuple = {}, None
                return fields, phases, info, parity_tuple

            warmups = {}
            for name in names:
                started = time.perf_counter()
                fields, phase, info, parity_tuple = invocation(name)
                warmups[name] = parity_tuple
                result["compile_and_warmups"].append(dict(intervals=n, case=case, method=name, seconds=time.perf_counter()-started))
            for tol in settings["gradient_tolerances"]:
                gate = parity(warmups[f"rom_modular_gtol{tol:g}"], warmups[f"rom_compiled_gtol{tol:g}"],
                              settings["field_parity_tolerance"], settings["latent_parity_tolerance"])
                gate.update(intervals=n, case=case, gradient_tolerance=tol)
                result["parity_gates"].append(gate)
                print("parity", gate, flush=True)
            rows = {name: dict(intervals=n, case=case, method=name,
                               gradient_tolerance=subjects[name][0] if name in subjects else None,
                               path=subjects[name][1] if name in subjects else "direct",
                               repetitions=[]) for name in names}
            for rep in range(settings["repetitions"]):
                burn_in()
                captured = {}
                measurements = {}
                for name in names if rep % 2 == 0 else names[::-1]:
                    fields, phases, info, parity_tuple = invocation(name)
                    captured[name] = parity_tuple
                    measurements[name] = (fields, phases, info)
                # Compression and scientific reporting never interrupt a paired timing block.
                for name in names:
                    fields, phases, info = measurements[name]
                    common = restrict(fields, n, observation)
                    record = dict(repetition=rep, phases=phases, solver=info, field=save_field(fields),
                                  vs_same_grid=hc.error_metrics(fields, discrete, n),
                                  vs_physical_per_grid=hc.error_metrics(fields, reference, n),
                                  vs_physical_common_grid=hc.error_metrics(common, common_reference, observation))
                    rows[name]["repetitions"].append(record)
                # Counter/field equivalence is checked for EVERY captured timed repetition.
                for tol in settings["gradient_tolerances"]:
                    gate = parity(captured[f"rom_modular_gtol{tol:g}"], captured[f"rom_compiled_gtol{tol:g}"],
                                  settings["field_parity_tolerance"], settings["latent_parity_tolerance"])
                    gate.update(intervals=n, case=case, gradient_tolerance=tol, repetition=rep)
                    result.setdefault("timed_parity_gates", []).append(gate)
            for name in names:
                row = rows[name]
                row["median_query_seconds"] = float(np.median([rep["phases"]["query_seconds"] for rep in row["repetitions"]]))
                row["max_common_current_error_all_repetitions"] = max(max(rep["vs_physical_common_grid"]["relative_current"]) for rep in row["repetitions"])
                result["rows"].append(row)
                print("row", n, case, name, row["median_query_seconds"], row["max_common_current_error_all_repetitions"], flush=True)
            dump(out/"results.json", result)
    result["complete"] = True
    result["no_training_and_frozen_checkpoint_verified"] = True
    dump(out/"results.json", result)
    print("RUNTIME PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
