"""Historical device-resident Poisson replay; see PROTOCOL.md and SOURCES.json.

The archived decoder, data generator, trust-LM, source projection and classical
operators are preserved. Large mesh arrays are explicit JIT inputs. Instrumentation
retains actual timed outputs, every repetition and stopping reasons. No training.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import sep_common as sc
import jax
import jax.numpy as jnp
import pro_common as pc
from pro_common import mp
import ctol_tol

F64 = jnp.float64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(a, b):
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-300))


def run_mesh(n, outdir, ns, reps, warm, gate_nz):
    started = time.perf_counter()
    ckpt = HERE / "in" / f"n{n}.pkl"
    params, ztrain, cfg = sc.load_pkl(ckpt)
    assert int(cfg["N"]) == n
    k, rank = int(cfg["k"]), int(cfg["r"])
    modes = int(cfg.get("M", 4*k))
    budget = int(cfg.get("gn_iters", 60))
    tr_factor = float(cfg.get("tr_factor", 1.0))
    fresh_seed = int(cfg.get("fresh_seed", 777))
    taus = [0.001, 0.01]
    cg_tols = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 1e-6]
    dev = jax.devices()[0]
    report = dict(config=dict(
        pde="poisson2d", kind="historical_device_resident_replay", N=n,
        k=k, r=rank, M=modes, gn_iters=budget, tr_factor=tr_factor,
        seed=0, data_seed=mp.SEED, n_train=mp.N_TRAIN, fresh_seed=fresh_seed,
        n_src_per_cohort=ns, reps=reps, burn=warm, gate_nz=gate_nz,
        taus=taus, cg_tolerances=cg_tols, cg_truth_tol=mp.CG_TOL,
        cg_maxiter=mp.CG_MAXITER, ckpt_sha256=sha(ckpt), ckpt_cfg=cfg,
        objective="original alpha=1 weak form; exact matrix B; original mean-code initialization",
        timer="GPU input -> source projection -> trust-LM -> full interior GPU output; excludes transfers",
        fields="last captured timed invocation for each source and subject; unmodified float64",
        baseline_primary="original unpreconditioned CG; same grid",
        baseline_secondary="original dense sine spectral exact solve; same grid",
        cg_selection="minimum cohort median time among tolerances with mean error <= QF mean error",
        omitted="EQ control omitted: no quadrature fitting needed for QF/full identity",
        historical_qf_extension=bool(n == 1024),
        x64=bool(jax.config.jax_enable_x64), matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        jax_version=jax.__version__, hostname=os.uname().nodename,
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        source_manifest_sha256=sha(HERE / "SOURCES.json"),
        replay_sha256=sha(__file__), owner_commit=(HERE / "OWNER_COMMIT").read_text().strip()),
        setup={}, gates={}, rows=[], complete=False)
    resultfile = outdir / f"poisson_n{n}.json"

    def save():
        resultfile.write_text(json.dumps(report, indent=1, default=float)+"\n")

    sc.log(f"START N={n} K={k} R={rank} M={modes}, sources={2*ns}, budget={budget}")
    grid = pc.Grid(n)
    coords = np.asarray(grid.coords_int)
    ni = grid.n_i
    cx, cy, widths, amps, _ = mp.sample_params(seed=0)
    cxf, cyf, wf, af, _ = mp.sample_params(seed=fresh_seed, m=ns)
    sources = np.stack(
        [mp.source_interior(n, cx[mp.N_TRAIN+i], cy[mp.N_TRAIN+i],
                            widths[mp.N_TRAIN+i], amps[mp.N_TRAIN+i]) for i in range(ns)]
        + [mp.source_interior(n, cxf[i], cyf[i], wf[i], af[i]) for i in range(ns)])
    cohort_names = ["heldout_seed0", f"fresh_seed{fresh_seed}"]
    truth_solve = jax.jit(lambda f: jax.scipy.sparse.linalg.cg(
        lambda u: mp.neg_lap_interior(u, n), f,
        tol=mp.CG_TOL, maxiter=mp.CG_MAXITER)[0])
    fdev = [jnp.asarray(f) for f in sources]
    truth = np.stack([np.asarray(truth_solve(f)) for f in fdev])
    truth_res = max(rel(np.asarray(mp.neg_lap_interior(jnp.asarray(u), n)), f)
                    for u, f in zip(truth, sources))
    assert truth_res < 1e-10, truth_res
    report["gates"]["truth_residual_max"] = truth_res
    tnorm = np.linalg.norm(truth.reshape(2*ns, -1), axis=1)
    spec = dict(kind="weak", alpha=1.0, M=modes)
    mask = np.asarray(grid.mode_mask(modes)).astype(bool)
    mode_i, mode_j = np.nonzero(mask)
    sine = jnp.asarray(grid.S)
    lam_full = jnp.asarray(grid.lam)
    mode_i, mode_j = jnp.asarray(mode_i), jnp.asarray(mode_j)
    src_weight = lam_full[mode_i, mode_j] ** -1.0
    phi, wl = pc.colloc_mode_table(grid, spec, "grid", coords)
    decoder = sc.SeparableDecoder(params, k, rank)
    head = decoder.head_fn()
    t0 = time.perf_counter()
    bank = decoder.feat_at(coords, chunk=65536)
    bank.block_until_ready()
    report["setup"]["bank_seconds"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    reduced = phi @ bank
    reduced.block_until_ready()
    report["setup"]["B_seconds"] = time.perf_counter()-t0
    z0 = jnp.asarray(ztrain.mean(0))
    radius = float(np.max(np.linalg.norm(ztrain-ztrain.mean(0), axis=1)))
    trust = tr_factor*radius if tr_factor > 0 else np.inf
    report["config"].update(trust_delta=trust, n_modes_actual=int(phi.shape[0]))

    def project(f, s):
        return (s.T @ f @ s)[mode_i, mode_j]*src_weight

    source_gate = max(rel(np.asarray(project(fdev[i], sine)),
                          np.asarray(pc.weak_source_term(grid, spec, "grid", sources[i])))
                      for i in [0, ns, 2*ns-1])
    assert source_gate < 1e-12
    report["gates"]["source_projection_rel_max"] = source_gate

    # These factories execute during tracing. Passing bank/Phi/sine as arguments
    # prevents large closed-over constants while preserving the archived algebra.
    def qf_pipe(f, tau, g, b, s):
        lm, _ = ctol_tol.lm_tau_poisson(
            lambda z, xy: head(z), k, jnp.zeros((rank, 2), F64),
            jnp.ones(rank, F64), b, wl, budget, trust_delta=trust)
        result = lm(z0, project(f, s), tau)
        return (g @ head(result[0]), result[0]) + result[1:]

    def full_pipe(f, tau, g, p, s):
        lm, _ = ctol_tol.lm_tau_poisson(
            lambda z, xy: g @ head(z), k, jnp.zeros((1, 2), F64),
            jnp.ones(g.shape[0], F64), p, wl, budget, trust_delta=trust)
        result = lm(z0, project(f, s), tau)
        return (g @ head(result[0]), result[0]) + result[1:]

    qf_jit, full_jit = jax.jit(qf_pipe), jax.jit(full_pipe)
    qf_rj = jax.jit(lambda z, fm, b: (wl*(b @ head(z))-fm,
                                            wl[:, None]*(b @ jax.jacfwd(head)(z))))
    full_rj = jax.jit(lambda z, fm, g, p: (wl*(p @ (g @ head(z)))-fm,
                            wl[:, None]*(p @ (g @ jax.jacfwd(head)(z)))))
    # Cached feature identity is checked on deterministic distributed mesh nodes;
    # QF/full weak residual and gradient are checked using every mesh node.
    sample_idx = np.linspace(0, len(coords)-1, min(4096, len(coords)), dtype=int)
    meshfree = jax.jit(lambda z, xy: decoder(z, xy))
    rng = np.random.default_rng(1)
    gates = []
    for j in range(gate_nz):
        z = jnp.asarray(ztrain[rng.integers(len(ztrain))]+0.05*rng.standard_normal(k))
        fm = project(fdev[j % len(fdev)], sine)
        rf, jf = map(np.asarray, full_rj(z, fm, bank, phi))
        rq, jq = map(np.asarray, qf_rj(z, fm, reduced))
        gates.append([rel(rq, rf), rel(jq.T @ rq, jf.T @ rf)])
        if j < 5:
            mesh_dev = rel(np.asarray(bank[sample_idx] @ head(z)),
                           np.asarray(meshfree(z, jnp.asarray(coords[sample_idx]))))
            assert mesh_dev < 1e-12
    gate_max = np.max(gates, axis=0)
    assert gate_max[0] < 1e-12 and gate_max[1] < 1e-10, gate_max
    report["gates"]["Q_random"] = dict(n=gate_nz, resid_rel_max=float(gate_max[0]),
                                         grad_rel_max=float(gate_max[1]))
    spectral = jax.jit(lambda f, s, lam: s @ ((s.T @ f @ s)/lam) @ s.T)
    cg_pipes = {tol: jax.jit(lambda f, _tol=tol: jax.scipy.sparse.linalg.cg(
        lambda u: mp.neg_lap_interior(u, n), f, tol=_tol, maxiter=mp.CG_MAXITER)[0])
        for tol in cg_tols}
    descriptors = [(p, tau, None, f"{p}_{tau:.0e}")
                   for p in ["full", "qf"] for tau in taus]
    descriptors += [("cg", None, tol, f"cg_{tol:.0e}") for tol in cg_tols]
    descriptors += [("spectral_dense", None, None, "spectral_dense")]
    capture = {name: [] for _, _, _, name in descriptors}
    raw = {name: [] for _, _, _, name in descriptors}
    aux = {name: [] for _, _, _, name in descriptors}
    qsol = []
    sc.log(f"N={n} gates pass; compiling {len(descriptors)} subjects")

    def subjects_for(fi):
        subjects = []
        for method, tau, tol, name in descriptors:
            if method == "qf":
                fn = lambda _tau=tau: qf_jit(fi, _tau, bank, reduced, sine)
            elif method == "full":
                fn = lambda _tau=tau: full_jit(fi, _tau, bank, phi, sine)
            elif method == "cg":
                fn = lambda _tol=tol: (cg_pipes[_tol](fi),)
            else:
                fn = lambda: (spectral(fi, sine, lam_full),)

            def blocked(_fn=fn):
                result = _fn()
                result[0].block_until_ready()
                return result
            subjects.append((name, blocked))
        return subjects

    t0 = time.perf_counter()
    for _, fn in subjects_for(fdev[0]):
        fn()
    report["setup"]["compile_and_first_call_seconds"] = time.perf_counter()-t0
    save()
    for i, fi in enumerate(fdev):
        # Host compression/output processing is excluded and can idle the GPU;
        # burn immediately before EACH source's balanced timing block.
        ctol_tol.burn_in(1.5)
        times, results = sc.balanced_time(subjects_for(fi), reps=reps, warm=warm)
        fm = project(fi, sine)
        for method, tau, tol, name in descriptors:
            out = results[name]
            capture[name].append(np.asarray(out[0]).reshape(ni, ni))
            raw[name].append([float(t) for t in times[name]])
            if method in ["full", "qf"]:
                aux[name].append(dict(z=np.asarray(out[1]).tolist(), val=float(out[2]),
                    val0=float(out[3]), jac=int(out[4]), accepted=int(out[5]),
                    attempts=int(out[6]), reason=int(out[7])))
                rf, jf = map(np.asarray, full_rj(out[1], fm, bank, phi))
                rq, jq = map(np.asarray, qf_rj(out[1], fm, reduced))
                rr = rel(rq, rf)
                gr = rel(jq.T @ rq, jf.T @ rf)
                ga = float(np.linalg.norm(jq.T @ rq-jf.T @ rf)/
                           (np.linalg.norm(jf)*np.linalg.norm(rf)+1e-300))
                qsol.append([rr, gr, ga])
                assert rr < 1e-12 and (gr < 1e-10 or ga < 1e-12), qsol[-1]
        sc.log(f"N={n} timed source {i+1}/{len(fdev)}")

    report["gates"]["Q_solutions"] = dict(n=len(qsol), n_fail=0,
        resid_rel_max=float(np.max(qsol, axis=0)[0]),
        grad_rel_max=float(np.max(qsol, axis=0)[1]),
        grad_abs_max=float(np.max(qsol, axis=0)[2]))
    for method, tau, tol, name in descriptors:
        for cohort_id, cohort in enumerate(cohort_names):
            ids = list(range(cohort_id*ns, (cohort_id+1)*ns))
            errors = [rel(capture[name][i], truth[i]) for i in ids]
            medians = [float(np.median(raw[name][i])) for i in ids]
            outliers = sum(sum(t > 3*np.median(raw[name][i]) for t in raw[name][i]) for i in ids)
            row = dict(method=method, tau=tau, fom_tol=tol, field_key=name,
                cohort=cohort, N=n, k=k, r=rank, M=modes, n_sources=ns,
                source_ids=ids, time_ms=1000*float(np.median(medians)),
                time_ms_all=[1000*t for t in medians],
                time_raw_s={str(i):raw[name][i] for i in ids},
                err_rel_l2=float(np.mean(errors)), err_rel_l2_median=float(np.median(errors)),
                err_rel_l2_max=float(np.max(errors)), err_rel_l2_all=errors,
                time_outliers_gt_3x_source_median=int(outliers),
                total_repetitions=ns*reps)
            if method in ["full", "qf"]:
                metrics = [aux[name][i] for i in ids]
                reasons = [m["reason"] for m in metrics]
                row.update(solver_per_source=metrics,
                    jac_evals=float(np.mean([m["jac"] for m in metrics])),
                    censored_frac=float(np.mean([r not in ctol_tol.POISSON_TAU_OK for r in reasons])),
                    stop_reasons={str(r):reasons.count(r) for r in set(reasons)})
            report["rows"].append(row)
            sc.log(f"N={n} {cohort} {name}: {row['time_ms']:.4f} ms; "
                   f"mean error {row['err_rel_l2']:.8g}, worst {row['err_rel_l2_max']:.8g}")
    report["comparisons"] = []
    for row in report["rows"]:
        if row["method"] != "qf":
            continue
        candidates = [r for r in report["rows"] if r["method"] == "cg"
            and r["cohort"] == row["cohort"] and r["err_rel_l2"] <= row["err_rel_l2"]]
        chosen = min(candidates, key=lambda r:r["time_ms"])
        direct = next(r for r in report["rows"] if r["method"] == "spectral_dense"
                      and r["cohort"] == row["cohort"])
        report["comparisons"].append(dict(cohort=row["cohort"], tau=row["tau"],
            rom_ms=row["time_ms"], cg_ms=chosen["time_ms"], cg_tol=chosen["fom_tol"],
            cg_over_rom=chosen["time_ms"]/row["time_ms"],
            spectral_ms=direct["time_ms"], spectral_over_rom=direct["time_ms"]/row["time_ms"]))
    save()
    sc.log(f"N={n}: writing captured float64 fields")
    fieldfile = outdir / f"fields_n{n}.npz"
    np.savez_compressed(fieldfile, sources=sources, truth=truth,
                        **{name:np.stack(fields) for name, fields in capture.items()})
    report["fields_file"] = fieldfile.name
    report["fields_sha256"] = sha(fieldfile)
    report["complete"] = True
    report["total_seconds"] = time.perf_counter()-started
    save()
    sc.log(f"DONE N={n} {report['total_seconds']:.1f}s -> {resultfile}")


def main():
    assert jax.default_backend() == "gpu", "GPU required"
    assert jax.config.jax_enable_x64
    assert os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") == "highest"
    sc.log(f"jax_backend=gpu x64={jax.config.jax_enable_x64} device={jax.devices()[0]}")
    outdir = Path(os.environ["OUT"])
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(os.environ.get("N_SRC", "16"))
    reps = int(os.environ.get("REPS", "12"))
    warm = int(os.environ.get("WARM", "3"))
    gate_nz = int(os.environ.get("GATE_NZ", "32"))
    assert warm >= 3
    for n in map(int, os.environ.get("MESHES", "128,256,512,1024").split(",")):
        run_mesh(n, outdir, ns, reps, warm, gate_nz)
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    main()
