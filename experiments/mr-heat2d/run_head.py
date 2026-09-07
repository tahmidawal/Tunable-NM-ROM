"""Fixed-bank training-coverage comparison with gated paired rollout costs."""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import heat_core as hc
import head_refine as hr
from run_pilot import assemble, block, burn_in, dump
from runtime_paths import build_paths, parity
from verify_heat import verification, restrict
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config-head.json")))
    parser.add_argument("--checkpoint", default="checkpoint.pkl")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    cfg = json.loads(Path(args.config).with_name(settings["base_config"]).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    (out/"fields").mkdir(); (out/"checkpoints").mkdir()
    assert jax.default_backend() == "gpu" and jax.config.jax_enable_x64
    assert os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    print("jax_backend=gpu x64=True precision=highest", flush=True)
    checkpoint_blob = Path(args.checkpoint).read_bytes()
    assert hashlib.sha256(checkpoint_blob).hexdigest() == settings["checkpoint_sha256"]
    checkpoint = pickle.loads(checkpoint_blob)
    assert checkpoint["config"] == cfg
    base = jax.tree.map(jnp.asarray, checkpoint["params"])
    original_codes = jnp.asarray(checkpoint["codes"])
    source = json.loads(Path("SOURCE-MANIFEST.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    result = dict(schema=settings["schema"], config=cfg, settings=settings, source_manifest=source,
                  metadata=dict(job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"),
                                gpu=jax.devices()[0].device_kind, backend="gpu", x64=True, precision="highest", jax=jax.__version__),
                  complete=False, models=[], reconstruction=[], rows=[], setups=[], reference_fields=[], case_fields=[],
                  reference_evidence=dict(kind="empirical_spectral_refinement", rigorous_relative_bound=None),
                  query_contract="Host full f64 initial field to all requested host full f64 output fields; fitted training library, initial projection/two starts, CN evolution, dense readout and transfers charged. Data generation/training/assembly/compilation separate.")
    def save_field(field):
        a = np.ascontiguousarray(field)
        digest = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()
        path = out/"fields"/f"{digest}.npz"
        if not path.exists(): np.savez_compressed(path, field=a)
        return dict(path=str(path.relative_to(out)), sha256_array=digest, shape=list(a.shape), dtype=a.dtype.str)
    def save_model(name, params, codes, details):
        for key in ("B", "g", "out_scale"):
            for a, b in zip(jax.tree.leaves(base[key]), jax.tree.leaves(params[key])):
                np.testing.assert_array_equal(a, b)
        path = out/"checkpoints"/f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(dict(params=jax.device_get(params), codes=np.asarray(codes), config=cfg, refinement=details), handle)
        record = dict(name=name, checkpoint_path=str(path.relative_to(out)), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      frozen_spatial_arrays_exact=True, details=details)
        result["models"].append(record)
        return record
    result["verification"] = verification(cfg)
    result["reference_evidence"]["empirical_relative_delta"] = max(result["verification"]["reference_spectral_refinement_current_errors"])
    times = jnp.asarray(cfg["times"])
    ntrain = cfg["train_intervals"]
    training_draws = hc.sample_family(cfg["train_seed"], cfg["n_train"], cfg)
    new_draws = hc.sample_family(settings["additional_training_seed"], settings["additional_training_trajectories"], cfg)
    val_draws = hc.sample_family(cfg["validation_seed"], cfg["n_validation"], cfg)
    all_draws = np.concatenate((training_draws, new_draws))
    assert not any(np.array_equal(a, b) for a in all_draws for b in val_draws)
    result["cohorts"] = dict(original_train=training_draws.tolist(), additional_train=new_draws.tolist(), validation=val_draws.tolist(),
                             validation_in_training_or_lookup=False, final_cohort_opened=False)
    arrays, setup = assemble(base, original_codes, ntrain, cfg)
    result["setups"].append(dict(model="frozen_shared_bank", **setup))
    xy = jnp.asarray(hc.coords(ntrain))
    def trajectories(draws, n, continuum=False):
        points = jnp.asarray(hc.coords(n))
        return block(jnp.stack([hc.propagate(hc.initial_field(points, draw).reshape(n-1, n-1),
                                             hc.eigenvalues(n, continuum), times, cfg["diffusivity"]) for draw in draws]))
    started = time.perf_counter()
    train_fields = trajectories(all_draws, ntrain).reshape(-1, len(xy))
    target, norm2, perpendicular2 = block(hr.compression(train_fields, arrays["projection"]))
    result["training_data_seconds"] = time.perf_counter()-started
    original_count = cfg["n_train"]*len(times)
    nearest = jnp.argmin(jnp.sum((target[original_count:, None]-arrays["library"][None])**2, axis=2), axis=1)
    expanded_codes = jnp.concatenate((original_codes, original_codes[nearest]))
    np.savez_compressed(out/"training_compression.npz", target=np.asarray(target), norm2=np.asarray(norm2),
                        perpendicular2=np.asarray(perpendicular2), new_code_nearest_original_snapshot=np.asarray(nearest),
                        original_snapshot_count=original_count, original_codes=np.asarray(original_codes), expanded_codes=np.asarray(expanded_codes))
    model_data = {"frozen": (base, original_codes)}
    save_model("frozen", base, original_codes, dict(kind="unchanged_control", trajectories=cfg["n_train"]))
    # Both cohorts use exactly matched update/minibatch counts; only training data differ.
    for cohort, count in (("original", original_count), ("expanded", len(target))):
        for seed in settings["minibatch_seeds"]:
            name = f"{cohort}_seed{seed}"
            params, codes, info = hr.train(base, expanded_codes[:count], arrays["triangular"], target[:count], norm2[:count],
                                          perpendicular2[:count], seed, settings)
            model_data[name] = params, codes
            save_model(name, params, codes, dict(kind=cohort, trajectories=count//len(times), **info))
            dump(out/"results.json", result)
    # Validation is first used only after all training endpoints are fixed and saved.
    val_fields = trajectories(val_draws, ntrain).reshape(-1, len(xy))
    fine = max(cfg["reference_intervals"])
    physical = np.asarray(trajectories(val_draws, fine, True))
    for case in range(len(val_draws)):
        result["reference_fields"].append(dict(case=case, intervals=fine, field=save_field(physical[case])))
    unrestricted = (val_fields@arrays["projection"])@arrays["projection"].T
    result["bank_projection"] = dict(intervals=ntrain, truth=save_field(np.asarray(val_fields).reshape(len(val_draws), len(times), ntrain-1, ntrain-1)),
                                      reconstructed=save_field(np.asarray(unrestricted).reshape(len(val_draws), len(times), ntrain-1, ntrain-1)),
                                      singular_values=setup["bank_singular_values"], rank=int(np.linalg.matrix_rank(np.asarray(arrays["triangular"]))))
    qualified = []
    for name, (params, codes) in model_data.items():
        reconstructed, zs, info, best, lookup = block(hr.fit_fields(params, codes, arrays["bank"], arrays["projection"], arrays["triangular"], val_fields, settings))
        reconstructed = np.asarray(reconstructed).reshape(len(val_draws), len(times), ntrain-1, ntrain-1)
        info, best, zs = np.asarray(info), np.asarray(best), np.asarray(zs)
        selected_info = info[np.arange(len(info)), best]
        row = dict(model=name, intervals=ntrain, fits=info.tolist(), selected_starts=best.tolist(), codes=zs.tolist(),
                   nearest_training_snapshot_indices=np.asarray(lookup).tolist(), reconstructed=save_field(reconstructed), cases=[])
        jac = jax.jit(jax.jacfwd(lambda z, p, triangular: triangular@hc.sc.head(p, z), argnums=0))
        row["selected_jacobian_singular_values"] = [np.linalg.svd(np.asarray(jac(jnp.asarray(z), params, arrays["triangular"])), compute_uv=False).tolist() for z in zs]
        initial_ok = True
        for case in range(len(val_draws)):
            truth = np.asarray(val_fields).reshape(len(val_draws), len(times), ntrain-1, ntrain-1)[case]
            metrics = hc.error_metrics(reconstructed[case], truth, ntrain)
            bank_metrics = hc.error_metrics(np.asarray(unrestricted).reshape(len(val_draws), len(times), ntrain-1, ntrain-1)[case], truth, ntrain)
            initial_info = selected_info[case*len(times)]
            initial_ok &= bool(metrics["relative_current"][0] <= settings["rollout_gate"]["initial_worst_max"]
                               and initial_info[4] <= settings["fit_gradient_tolerance"] and initial_info[2] == 1)
            row["cases"].append(dict(case=case, vs_same_grid=metrics, bank_projection=bank_metrics,
                                     vs_physical=hc.error_metrics(reconstructed[case], restrict(physical[case], fine, ntrain), ntrain)))
        row["rollout_gate_passed"] = bool(initial_ok and name != "frozen")
        result["reconstruction"].append(row)
        if row["rollout_gate_passed"]: qualified.append(name)
        print("reconstruction", name, "initial", [c["vs_same_grid"]["relative_current"][0] for c in row["cases"]], "rollout_gate", row["rollout_gate_passed"], flush=True)
    result["qualified_refinements"] = qualified
    result["rollout_performed"] = bool(qualified)
    dump(out/"results.json", result)
    if qualified:
        for n in settings["rollout_intervals"]:
            base_arrays, info = assemble(base, original_codes, n, cfg)
            result["setups"].append(dict(model="frozen_shared_bank", **info))
            lam = hc.eigenvalues(n); m = cfg["modes_per_axis"]
            low_lam = lam[:m, :m].reshape(-1)
            modes = {}
            for name in ["frozen"]+qualified:
                params, codes = model_data[name]
                library = block(hc.sc.head(params, codes)@base_arrays["triangular"].T)
                for tol in settings["rollout_gradient_tolerances"]:
                    label = f"{name}_gtol{tol:g}"
                    modes[label] = (name, tol, params, codes, library, build_paths(dict(cfg, gradient_tolerance=tol), settings["rollout_dt"]))
            names = ["fom_dst_exact_time"]+list(modes)
            for case, draw in enumerate(val_draws):
                u0 = np.asarray(hc.initial_field(jnp.asarray(hc.coords(n)), draw)).reshape(n-1, n-1)
                discrete = np.asarray(hc.propagate(jnp.asarray(u0), lam, times, cfg["diffusivity"]))
                reference = restrict(physical[case], fine, n)
                common_ref = restrict(physical[case], fine, settings["observation_intervals"])
                result["case_fields"].append(dict(intervals=n, case=case, initial=save_field(u0), discrete=save_field(discrete), physical=save_field(reference)))
                def invocation(label):
                    started = time.perf_counter(); phases = {}
                    inp = block(jax.device_put(u0)); phases["input_seconds"] = time.perf_counter()-started
                    begin = time.perf_counter()
                    if label == "fom_dst_exact_time":
                        fields = block(hc.propagate(inp, lam, times, cfg["diffusivity"])); aux = None
                    else:
                        name, tol, params, codes, library, paths = modes[label]
                        fields, initial_info, step_info, zs = block(paths["compiled"](params, base_arrays["projection"], base_arrays["triangular"],
                                                                                    library, codes, base_arrays["bank"], base_arrays["matrix"], low_lam, inp))
                        aux = initial_info, step_info, zs
                    phases["device_query_seconds"] = time.perf_counter()-begin
                    begin = time.perf_counter(); fields = np.asarray(fields).reshape(len(times), n-1, n-1)
                    phases["output_transfer_seconds"] = time.perf_counter()-begin
                    phases["query_seconds"] = time.perf_counter()-started
                    if aux is not None:
                        initial_info, step_info, zs = jax.device_get(aux)
                        aux = dict(initial_fits=initial_info.tolist(), steps=step_info.tolist(), latent=zs.tolist())
                    return fields, phases, aux or {}
                for label in names:
                    started = time.perf_counter(); fields, phases, aux = invocation(label)
                    result.setdefault("warmups", []).append(dict(intervals=n, case=case, method=label, seconds=time.perf_counter()-started))
                    if label in modes:
                        name, tol, params, codes, library, paths = modes[label]
                        z0, initial = paths["initialize"](params, base_arrays["projection"], base_arrays["triangular"], library, codes, jnp.asarray(u0))
                        zs, steps = paths["rollout"](params, base_arrays["matrix"], hc.cn_factor(low_lam, settings["rollout_dt"], cfg["diffusivity"]), z0)
                        modular = paths["readout"](params, base_arrays["bank"], zs)
                        gate = parity((np.asarray(modular).reshape(fields.shape), initial, steps, zs),
                                      (fields, np.asarray(aux["initial_fits"]), np.asarray(aux["steps"]), np.asarray(aux["latent"])),
                                      settings["field_parity_tolerance"], settings["latent_parity_tolerance"])
                        result.setdefault("parity_gates", []).append(dict(intervals=n, case=case, method=label, **gate))
                rows = {label: dict(intervals=n, case=case, method=label, model=modes[label][0] if label in modes else "fom",
                                    gradient_tolerance=modes[label][1] if label in modes else None, repetitions=[]) for label in names}
                for repetition in range(settings["timing_repetitions"]):
                    burn_in(); capture = {}
                    for label in names if repetition%2 == 0 else names[::-1]: capture[label] = invocation(label)
                    for label in names:
                        fields, phases, solver = capture[label]
                        common = restrict(fields, n, settings["observation_intervals"])
                        rows[label]["repetitions"].append(dict(repetition=repetition, phases=phases, solver=solver, field=save_field(fields),
                            vs_same_grid=hc.error_metrics(fields, discrete, n), vs_physical_per_grid=hc.error_metrics(fields, reference, n),
                            vs_physical_common_grid=hc.error_metrics(common, common_ref, settings["observation_intervals"])))
                for label in names:
                    result["rows"].append(rows[label])
                    print("timed", n, case, label, np.median([r["phases"]["query_seconds"] for r in rows[label]["repetitions"]]), flush=True)
                dump(out/"results.json", result)
    result["complete"] = True
    dump(out/"results.json", result)
    print("HEAD PILOT COMPLETE", flush=True)


if __name__ == "__main__": main()
