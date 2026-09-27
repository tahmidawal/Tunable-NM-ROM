"""Bounded frozen-weight heat transfer pilot with complete paired query timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import heat_core as hc
from verify_heat import verification, restrict
import jax
import jax.numpy as jnp
import numpy as np
import optax


def block(value):
    jax.block_until_ready(value)
    return value


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def burn_in(seconds=.6):
    a = jnp.eye(768, dtype=jnp.float64)+.00001
    kernel = jax.jit(lambda x: jnp.tanh(x@x/768))
    block(kernel(a))
    start = time.perf_counter()
    while time.perf_counter()-start < seconds:
        a = block(kernel(a))


def train(cfg, xy, snapshots):
    key, kp, kz = jax.random.split(jax.random.PRNGKey(cfg["model_seed"]), 3)
    scale = float(jnp.sqrt(jnp.mean(snapshots**2)))
    params = hc.sc.init_separable(kp, cfg["k"], cfg["r"], out_scale=scale, **cfg["arch"])
    codes = .1*jax.random.normal(kz, (len(snapshots), cfg["k"]), dtype=jnp.float64)
    schedule = optax.warmup_cosine_decay_schedule(0, cfg["learning_rate"], 200, cfg["steps"], cfg["learning_rate"]*.01)
    optimizer = optax.adam(schedule)
    state = optimizer.init((params, codes))
    def loss(pz, points, truth):
        p, z = pz
        bank = hc.sc.features(p, points)
        predicted = hc.sc.head(p, z)@bank.T
        # Each snapshot gets equal relative-L2 weight; no decay-dependent loss bias.
        mse = jnp.mean(jnp.mean((predicted-truth)**2, axis=1)/jnp.mean(truth**2, axis=1))
        gram = bank.T@bank/(len(points)*p["out_scale"]**2)
        return mse+1e-5*jnp.mean((gram-jnp.eye(cfg["r"]))**2), mse
    @jax.jit
    def step(pz, state, points, truth):
        (total, mse), gradients = jax.value_and_grad(loss, has_aux=True)(pz, points, truth)
        gradients[0]["out_scale"] = jnp.zeros_like(gradients[0]["out_scale"])
        updates, state = optimizer.update(gradients, state)
        return optax.apply_updates(pz, updates), state, mse
    pz = (params, codes)
    start = time.perf_counter()
    history = []
    for i in range(cfg["steps"]):
        pz, state, loss_value = step(pz, state, xy, snapshots)
        if i == 0 or (i+1) % 500 == 0:
            record = dict(step=i+1, relative_mse=float(loss_value), seconds=time.perf_counter()-start)
            history.append(record)
            print("train", record, flush=True)
    block(pz)
    return pz, dict(seconds=time.perf_counter()-start, history=history,
                   parameters=sum(int(v.size) for v in jax.tree.leaves(pz[0])),
                   code_scalars=int(pz[1].size), snapshots=len(snapshots), points=len(xy))


def assemble(params, codes, n, cfg):
    begin = time.perf_counter()
    xy = jnp.asarray(hc.coords(n))
    bank = block(jax.jit(hc.sc.features)(params, xy))
    bank_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    # QR compresses full-field initial least squares EXACTLY (orthogonal constant omitted).
    q, triangular = block(jnp.linalg.qr(bank, mode="reduced"))
    matrix = block(hc.mode_matrix(n, cfg["modes_per_axis"]).T@bank)
    library = block(hc.sc.head(params, codes)@triangular.T)
    setup_seconds = time.perf_counter()-begin
    singular = np.linalg.svd(np.asarray(triangular), compute_uv=False)
    phi = hc.mode_matrix(n, cfg["modes_per_axis"])
    low_lam = hc.eigenvalues(n)[:cfg["modes_per_axis"], :cfg["modes_per_axis"]].reshape(-1)
    zcheck = codes[0]
    exact = lambda z: low_lam*(matrix@hc.sc.head(params, z))
    stencil = lambda z: phi.T@hc.negative_laplacian((bank@hc.sc.head(params, z)).reshape(n-1, n-1), n).reshape(-1)
    relative = lambda a, b: float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(b), 1e-300))
    operator_error = relative(exact(zcheck), stencil(zcheck))
    jacobian_error = relative(jax.jacfwd(exact)(zcheck), jax.jacfwd(stencil)(zcheck))
    assert max(operator_error, jacobian_error) < 1e-11
    return dict(bank=bank, projection=q, triangular=triangular, matrix=matrix,
                library=library, codes=codes), dict(
                    bank_evaluation_seconds=bank_seconds, operator_qr_seconds=setup_seconds,
                    intervals=n, interior_unknowns=(n-1)**2, boundary_nodes=4*n,
                    total_nodes=(n+1)**2, M=cfg["modes_per_axis"]**2,
                    bank_bytes=bank.size*8, initialization_projection_bytes=q.size*8,
                    reduced_operator_bytes=matrix.size*8,
                    exact_weak_operator_relative_error=operator_error,
                    exact_weak_jacobian_relative_error=jacobian_error,
                    bank_singular_values=singular.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config-pilot.json")))
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = jax.devices()[0]
    assert jax.default_backend() == "gpu"
    assert jax.config.jax_enable_x64
    assert os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") == "highest"
    print(f"jax_backend=gpu device={device} x64=True precision=highest", flush=True)
    result = dict(config=cfg, metadata=dict(backend="gpu", x64=True, precision="highest",
                  gpu=device.device_kind, job_id=os.environ.get("SLURM_JOB_ID"),
                  node=os.environ.get("SLURMD_NODENAME"), jax=jax.__version__), complete=False,
                  rows=[], setups=[], diagnostic_rows=[], query_contract="Host float64 interior initial field to host float64 full interior fields at all six fixed times; homogeneous boundary values are implied and identical for both methods. Timed host transfer, exact full-input projection and latent fitting, evolution, readout and output transfer. Field generation outside query for BOTH methods. Compilation/setup separate.")
    manifest = Path("SOURCE-MANIFEST.json")
    if manifest.exists():
        result["source_manifest"] = json.loads(manifest.read_text())
        for path, expected in result["source_manifest"]["sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    started = time.perf_counter()
    result["verification"] = verification(cfg)
    result["verification_seconds"] = time.perf_counter()-started
    dump(out/"results.json", result)
    print("reference verification passed", result["verification"], flush=True)

    started = time.perf_counter()
    times = jnp.asarray(cfg["times"])
    ntrain = cfg["train_intervals"]
    draws_train = hc.sample_family(cfg["train_seed"], cfg["n_train"], cfg)
    draws_val = hc.sample_family(cfg["validation_seed"], cfg["n_validation"], cfg)
    xy = jnp.asarray(hc.coords(ntrain))
    snapshots = []
    for draw in draws_train:
        u0 = hc.initial_field(xy, draw).reshape(ntrain-1, ntrain-1)
        snapshots.append(hc.propagate(u0, hc.eigenvalues(ntrain), times, cfg["diffusivity"]))
    snapshots = block(jnp.concatenate(snapshots).reshape(-1, len(xy)))
    result["training_data_seconds"] = time.perf_counter()-started
    (params, codes), result["training"] = train(cfg, xy, snapshots)
    result["cohort"] = dict(training_draws=draws_train.tolist(), validation_draws=draws_val.tolist(), final_cohort_opened=False)
    with (out/"checkpoint.pkl").open("wb") as handle:
        pickle.dump(dict(params=jax.device_get(params), codes=np.asarray(codes), config=cfg), handle)
    result["checkpoint_sha256"] = hashlib.sha256((out/"checkpoint.pkl").read_bytes()).hexdigest()
    dump(out/"results.json", result)

    fine = max(cfg["reference_intervals"])
    physical_refs = []
    for draw in draws_val:
        u0 = hc.initial_field(jnp.asarray(hc.coords(fine)), draw).reshape(fine-1, fine-1)
        physical_refs.append(np.asarray(hc.propagate(u0, hc.eigenvalues(fine, True), times, cfg["diffusivity"])))
    np.savez_compressed(out/"physical_reference.npz", fields=np.asarray(physical_refs), intervals=fine,
                        times=np.asarray(times), draws=draws_val)
    raw_fields = {}
    for n in cfg["evaluation_intervals"]:
        print("evaluating frozen network at intervals", n, flush=True)
        arrays, setup = assemble(params, codes, n, cfg)
        result["setups"].append(setup)
        mode_lam = hc.eigenvalues(n)[:cfg["modes_per_axis"], :cfg["modes_per_axis"]].reshape(-1)
        lam = hc.eigenvalues(n)
        pipes = {f"rom_cn_dt{dt:g}": (dt, hc.make_query(cfg, dt)) for dt in cfg["time_steps"]}
        for case, draw in enumerate(draws_val):
            u0 = np.asarray(hc.initial_field(jnp.asarray(hc.coords(n)), draw)).reshape(n-1, n-1)
            discrete_ref = np.asarray(hc.propagate(jnp.asarray(u0), lam, times, cfg["diffusivity"]))
            physical = restrict(physical_refs[case], fine, n)
            raw_fields[f"n{n}_case{case}_discrete_reference"] = discrete_ref
            raw_fields[f"n{n}_case{case}_physical_reference"] = physical
            truth_flat = discrete_ref.reshape(len(times), -1)
            targets = truth_flat@np.asarray(arrays["projection"])
            projection = targets@np.asarray(arrays["projection"]).T
            fit = jax.jit(hc.make_lm(hc.sc.head, cfg["fit_budget"]*2, cfg["gradient_tolerance"]))
            oracle_z, oracle_info, oracle_fields = [], [], []
            for target in targets:
                distances = np.linalg.norm(np.asarray(arrays["library"])-target, axis=1)
                starts = [codes[i] for i in np.argsort(distances)[:3]]+[jnp.mean(codes, axis=0)]
                candidates = [fit(params, arrays["triangular"], jnp.asarray(target), z) for z in starts]
                z, info = min(candidates, key=lambda value: float(value[1][3]))
                oracle_z.append(np.asarray(z)); oracle_info.append(np.asarray(info).tolist())
                oracle_fields.append(np.asarray(arrays["bank"]@hc.sc.head(params, z)))
            diag = dict(intervals=n, case=case,
                        unrestricted_bank=hc.error_metrics(projection, truth_flat, n),
                        multistart_reconstruction=hc.error_metrics(oracle_fields, truth_flat, n),
                        oracle_fit_info=oracle_info, discrete_spatial_error=hc.error_metrics(discrete_ref, physical, n),
                        cn_discretization=[])
            for dt in cfg["time_steps"]:
                factor = hc.cn_factor(lam, dt, cfg["diffusivity"])
                coeff = hc.dst2(jnp.asarray(u0))
                steps = jnp.asarray(np.rint(np.asarray(times)/dt), dtype=jnp.int32)
                fields = jax.vmap(lambda step: hc.dst2(coeff*factor**step))(steps)
                diag["cn_discretization"].append(dict(dt=dt, error_vs_semidiscrete=hc.error_metrics(fields, discrete_ref, n)))
            result["diagnostic_rows"].append(diag)

            def invocation(name):
                phases = {}
                start = time.perf_counter()
                device_input = block(jax.device_put(u0))
                phases["input_seconds"] = time.perf_counter()-start
                if name == "fom_dst_exact_time":
                    begin = time.perf_counter()
                    fields = block(hc.propagate(device_input, lam, times, cfg["diffusivity"]))
                    phases["evolve_and_readout_seconds"] = time.perf_counter()-begin
                    begin = time.perf_counter()
                    host_fields = np.asarray(fields)
                    phases["output_transfer_seconds"] = time.perf_counter()-begin
                    info = {}
                else:
                    dt, (initialize, rollout, readout) = pipes[name]
                    begin = time.perf_counter()
                    z0, init_info = block(initialize(params, arrays["projection"], arrays["triangular"], arrays["library"], codes, device_input))
                    phases["initial_fit_seconds"] = time.perf_counter()-begin
                    begin = time.perf_counter()
                    zs, step_info = block(rollout(params, arrays["matrix"], hc.cn_factor(mode_lam, dt, cfg["diffusivity"]), z0))
                    phases["evolution_seconds"] = time.perf_counter()-begin
                    begin = time.perf_counter()
                    fields = block(readout(params, arrays["bank"], zs))
                    phases["readout_seconds"] = time.perf_counter()-begin
                    begin = time.perf_counter()
                    host_fields = np.asarray(fields).reshape(len(times), n-1, n-1)
                    phases["output_transfer_seconds"] = time.perf_counter()-begin
                    # Collect counters AFTER stopping query timer: physical output is contract.
                    info = (init_info, step_info, zs)
                phases["query_seconds"] = time.perf_counter()-start
                if isinstance(info, tuple):
                    init_info, step_info, zs = jax.device_get(info)
                    info = dict(initial_fits=np.asarray(init_info).tolist(), steps=np.asarray(step_info).tolist(), latent=np.asarray(zs).tolist())
                return host_fields, phases, info

            names = ["fom_dst_exact_time"]+list(pipes)
            compilation_start = time.perf_counter()
            for name in names:
                invocation(name)
            result.setdefault("compile_and_warmups", []).append(dict(intervals=n, case=case, seconds=time.perf_counter()-compilation_start))
            rows = {name: dict(intervals=n, case=case, method=name, frozen_checkpoint=result["checkpoint_sha256"], repetitions=[]) for name in names}
            for rep in range(cfg["repetitions"]):
                burn_in()
                for name in names if rep % 2 == 0 else names[::-1]:
                    fields, phases, info = invocation(name)
                    metrics = hc.error_metrics(fields, discrete_ref, n)
                    physical_metrics = hc.error_metrics(fields, physical, n)
                    record = dict(repetition=rep, phases=phases, vs_same_grid=metrics, vs_physical_reference=physical_metrics, solver=info)
                    rows[name]["repetitions"].append(record)
                    raw_fields[f"n{n}_case{case}_{name}_rep{rep}"] = fields
            for name, row in rows.items():
                row["median_query_seconds"] = float(np.median([rep["phases"]["query_seconds"] for rep in row["repetitions"]]))
                row["time_max_current_error"] = max(row["repetitions"][0]["vs_physical_reference"]["relative_current"])
                row["time_max_initial_error"] = max(row["repetitions"][0]["vs_physical_reference"]["relative_initial"])
                result["rows"].append(row)
                print("row", n, case, name, row["median_query_seconds"], row["time_max_current_error"], flush=True)
            dump(out/"results.json", result)
        np.savez_compressed(out/f"fields_N{n}.npz", **{k: v for k, v in raw_fields.items() if k.startswith(f"n{n}_")})
    result["complete"] = True
    result["trained_weights_unchanged_all_meshes"] = True
    result["scope_limit"] = "Restricted single-bump development cohort, not archived multi-bump heat and not a final paper cohort. No per-resolution retraining or resolution envelope was attempted in this bounded first pilot."
    dump(out/"results.json", result)
    print("PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
