"""Frozen linear-bank / NMROM / direct heat comparison, one GPU allocation."""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import heat_core as hc
import linear_paths as lp
from run_pilot import assemble, block, burn_in, dump
from runtime_paths import build_paths
from transfer_core import field_hash, interpolation_tables, make_fom
from verify_heat import restrict
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/mr-heat2d/config-linear.json")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    cfg = json.loads(Path(args.config).with_name("config-pilot.json").read_text())
    source = json.loads(Path("SOURCE-MANIFEST.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert jax.default_backend() == "gpu" and jax.config.jax_enable_x64
    assert os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    print("jax_backend=gpu x64=True precision=highest", flush=True)
    checkpoint = pickle.loads(Path(settings["checkpoint"]+".pkl").read_bytes())
    assert checkpoint["config"] == cfg
    params, codes = jax.tree.map(jnp.asarray, checkpoint["params"]), jnp.asarray(checkpoint["codes"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    (out/"fields").mkdir(); (out/"assembly").mkdir()
    started_run = time.perf_counter()
    result = dict(schema=settings["schema"], config=cfg, settings=settings, source_manifest=source,
        metadata=dict(job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"),
                      gpu=jax.devices()[0].device_kind, backend="gpu", x64=True, precision="highest", jax=jax.__version__),
        complete=False, cases=[], rows=[], setups=[], case_fields=[], warmups=[],
        verification=lp.verify(), final_cohort_opened=False,
        query_contract="Full supplied GPU initial field to all six full GPU outputs, blocked; host input and output transfers additionally measured from the SAME invocation. FOM preserves supplied initial field; ROM returns its actual projection/fit. Source descriptors used only for data generation. Offline mesh assembly and compilation excluded and recorded.",
        reference_evidence="Empirical continuum-spectral nested-grid refinement, not a rigorous continuum bound.")

    def save_field(a):
        a = np.ascontiguousarray(a); digest = field_hash(a)
        path = out/"fields"/(digest+".npz")
        if not path.exists(): np.savez_compressed(path, field=a)
        return dict(path=str(path.relative_to(out)), sha256_array=digest, shape=list(a.shape), dtype=a.dtype.str)

    for cohort in settings["cohorts"]:
        for draw in hc.sample_family(cohort["seed"], cohort["count"], cfg):
            result["cases"].append(dict(case=len(result["cases"]), cohort=cohort["name"], seed=cohort["seed"], draw=draw.tolist()))
    times = jnp.asarray(cfg["times"])
    # Regenerate each reference from seed on the allocation. Preserve its restriction
    # to the finest requested mesh; discarded finer arrays have recorded hashes.
    references = []
    largest = max(settings["requested_intervals"])
    for case in result["cases"]:
        pair = []
        for nf in settings["reference_pair"]:
            u0 = hc.initial_field(jnp.asarray(hc.coords(nf)), case["draw"]).reshape(nf-1, nf-1)
            ref = np.asarray(block(hc.propagate(u0, hc.eigenvalues(nf, True), times, cfg["diffusivity"])))
            pair.append(np.ascontiguousarray(restrict(ref, nf, largest)))
            case.setdefault("reference_full_array_hashes", []).append(dict(intervals=nf, sha256_array=field_hash(ref)))
            del ref
        metric = hc.error_metrics(pair[0], pair[1], largest)
        assert max(metric["relative_current"]) < cfg["reference_uncertainty_budget"]
        case["reference_refinement"] = metric
        case["reference_coarse"] = save_field(pair[0]); case["reference_fine"] = save_field(pair[1])
        references.append(case["reference_fine"])
        print("reference passed", case["case"], flush=True)
    for n in settings["requested_intervals"]:
        arrays, setup = assemble(params, codes, n, cfg)
        m = cfg["modes_per_axis"]; lam = hc.eigenvalues(n); mode_lam = lam[:m, :m].reshape(-1)
        begin = time.perf_counter()
        maps, evidence = lp.reduced_maps(np.asarray(arrays["triangular"]), np.asarray(arrays["matrix"]),
                                        np.asarray(mode_lam), cfg["times"], cfg["diffusivity"], settings["dt"])
        fine_maps, _ = lp.reduced_maps(np.asarray(arrays["triangular"]), np.asarray(arrays["matrix"]),
                                     np.asarray(mode_lam), cfg["times"], cfg["diffusivity"], settings["dt"]/2)
        setup["linear_assembly_seconds"] = time.perf_counter()-begin
        setup["generator_max_real_eigenvalue"] = float(np.max(evidence["generator_eigenvalues"].real))
        setup["linear_map_bytes"] = {k: a.nbytes for k, a in maps.items()}
        setup["linear_dimension"] = cfg["r"]; setup["nonlinear_dimension"] = cfg["k"]
        setup["weak_condition"] = float(np.linalg.cond(evidence["weak_coordinates"]))
        np.savez_compressed(out/"assembly"/f"n{n}.npz", triangular=np.asarray(arrays["triangular"]),
                            matrix=np.asarray(arrays["matrix"]), mode_lam=np.asarray(mode_lam), **maps, **evidence)
        result["setups"].append(setup)
        maps = {name: jnp.asarray(a) for name, a in maps.items()}
        paths = build_paths(dict(cfg, gradient_tolerance=settings["gradient_tolerance"]), settings["dt"])
        coarse_n = settings["coarse_solver_intervals"]
        indices, weights = map(jnp.asarray, interpolation_tables(n, coarse_n))
        coarse_lam = hc.eigenvalues(coarse_n); coarse = make_fom(n, coarse_n); stride = n//coarse_n
        @jax.jit
        def coarse_query(u0, eigenvalues, ts, nu, ix, wt):
            low = u0[stride-1::stride, stride-1::stride]
            later = coarse(low, eigenvalues, ts[1:], nu, ix, wt)
            return jnp.concatenate((u0[None], later))
        names = list(maps)+["nmrom", "fom_same_grid", "fom_coarse16"]
        for case in result["cases"]:
            cid = case["case"]
            u0 = np.ascontiguousarray(hc.initial_field(jnp.asarray(hc.coords(n)), case["draw"])).reshape(n-1, n-1)
            truth = np.ascontiguousarray(restrict(np.load(out/references[cid]["path"])["field"], largest, n))
            discrete = np.asarray(block(hc.propagate(jnp.asarray(u0), lam, times, cfg["diffusivity"])))
            result["case_fields"].append(dict(intervals=n, case=cid, initial=save_field(u0), physical=save_field(truth), discrete=save_field(discrete)))
            def invocation(name):
                begin = time.perf_counter(); inp = block(jax.device_put(u0)); input_end = time.perf_counter()
                aux = None
                if name in maps:
                    fields = block(lp.query(arrays["projection"], maps[name], inp))
                elif name == "nmrom":
                    fields, init, steps, zs = block(paths["compiled"](params, arrays["projection"], arrays["triangular"],
                        arrays["library"], codes, arrays["bank"], arrays["matrix"], mode_lam, inp))
                    aux = init, steps, zs
                elif name == "fom_same_grid":
                    fields = block(hc.propagate(inp, lam, times, cfg["diffusivity"]))
                else:
                    fields = block(coarse_query(inp, coarse_lam, times, cfg["diffusivity"], indices, weights))
                device_end = time.perf_counter()
                fields = np.ascontiguousarray(fields).reshape(len(times), n-1, n-1)
                end = time.perf_counter()
                solver = {}
                if aux is not None:
                    init, steps, zs = jax.device_get(aux)
                    solver = dict(initial_fits=init.tolist(), steps=steps.tolist(), latents=zs.tolist())
                return fields, dict(input_seconds=input_end-begin, device_seconds=device_end-input_end,
                    output_seconds=end-device_end, host_seconds=end-begin), solver
            for name in names:
                begin = time.perf_counter(); invocation(name)
                result["warmups"].append(dict(intervals=n, case=cid, method=name, seconds=time.perf_counter()-begin))
            refined = np.asarray(block(lp.query(arrays["projection"], jnp.asarray(fine_maps["linear_weak_cn"]), jnp.asarray(u0)))).reshape(truth.shape)
            refined_record = save_field(refined)
            rows = {name: dict(intervals=n, case=cid, cohort=case["cohort"], method=name, repetitions=[]) for name in names}
            for repetition in range(settings["timing_repetitions"]):
                burn_in(); captured = {}; order = names if repetition%2 == 0 else names[::-1]
                for name in order: captured[name] = invocation(name)
                for name, (fields, phases, solver) in captured.items():
                    assert fields.dtype == np.float64 and np.isfinite(fields).all()
                    record = dict(repetition=repetition, invocation_order=order, phases=phases, solver=solver,
                        field=save_field(fields), vs_same_grid=hc.error_metrics(fields, discrete, n),
                        vs_physical=hc.error_metrics(fields, truth, n))
                    if name == "linear_weak_cn":
                        record["refined_field"] = refined_record
                        record["vs_half_step"] = hc.error_metrics(fields, refined, n)
                    rows[name]["repetitions"].append(record)
                del captured
            result["rows"].extend(rows.values()); dump(out/"results.json", result)
            print("timed", n, cid, {name: round(np.median([r["phases"]["device_seconds"] for r in row["repetitions"]])*1000, 4) for name, row in rows.items()}, flush=True)
        del arrays, maps, fine_maps
        jax.clear_caches()
    result["elapsed_seconds"] = time.perf_counter()-started_run
    result["timed_invocations"] = sum(len(row["repetitions"]) for row in result["rows"])
    result["complete"] = True; dump(out/"results.json", result)
    print("LINEAR HEAT COMPLETE", flush=True)


if __name__ == "__main__": main()
