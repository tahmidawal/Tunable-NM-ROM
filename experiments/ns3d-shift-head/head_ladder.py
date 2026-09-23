"""One mesh, one allocation: the paper's head + nested corrections inside the co-moving
frame, the lane's linear-bank arm, the CNAB2 grid and an iterative (FD-CG) FOM grid.

Stages, in order (see DESIGN.md):
  1. bank        the lane's centered POD bank, rank R, rebuilt exactly as ladder64/96
                 (128 training cases, six frames), with its oracle-shift floor
  2. head data   centred bank coefficients of training trajectories (training seed
                 only) at a denser frame spacing; one code per state
  3. heads       for each k: auto-decoder training, best-found codes, corrections C
                 (weighted PCA of the misses), floors on development (oracle shift)
  4. rollouts    linear-bank frontier; head frontier k x q x dt x sweeps; controls
  5. FOMs        CNAB2 step ladder; FD-CG (second-order FD, CG viscous + CG pressure)
  6. timing      burn-in; sentinels; fast arms randomised & interleaved; slow arms in
                 their own phase; sentinels again, each right after a long neighbour
With --heldout the job loads a frozen, committed setting and evaluates it once.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ("experiments/ns3d", "experiments/ns2d", "experiments/separable-decoder",
              "experiments/ns3d-grok", "experiments/ns3d-shift"):
    sys.path.insert(0, str(ROOT / extra))
sys.path.insert(0, str(HERE))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import ns3d_fom as F  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402
import head_rom as H  # noqa: E402
import fd_fom as FD  # noqa: E402
from shift_pilot import sha256_file  # noqa: E402

log = D.log
UNSTABLE = 1.0


def rel_case(pred, truth, case):
    return D.rel_rows(pred, truth[case], truth[case, 0])


def run_cases(call, dev, viscosities, keep=False, inputs=None, restrict=None):
    """Errors over the cohort; optionally keep the fields (float64)."""
    errors = np.empty(dev.shape[:2], dtype=np.float64)
    fields = np.empty_like(dev) if keep else None
    for case in range(len(dev)):
        u0 = dev[case, 0] if inputs is None else inputs[case]
        out = call(jnp.asarray(u0), float(viscosities[case]))
        if restrict is not None:
            out = restrict(out)
        out = np.asarray(out).reshape(dev.shape[1:])
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"nonfinite field on case {case}")
        errors[case] = rel_case(out, dev, case)
        check = D.sumsq_rows(out, dev[case], dev[case, 0])
        if float(np.max(np.abs(errors[case] - check))) > 1e-12:
            raise RuntimeError("error reductions disagree")
        if keep:
            fields[case] = out
    return errors, fields


def disjointness(seed, count, others):
    rows = np.round(F.parameters(int(seed), int(count)), 9)
    out = {}
    for other_seed, other_count in others:
        keys = {tuple(x) for x in np.round(F.parameters(int(other_seed), int(other_count)), 9)}
        out[str(other_seed)] = int(sum(1 for row in rows if tuple(row) in keys))
    return out


def burn_gpu(seconds=2.0):
    a = jnp.ones((1024, 1024)) * 1e-3
    f = jax.jit(lambda a: a @ a * 1e-3 + 1e-3)
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = f(a)
    jax.block_until_ready(a)


def time_block(callables, reps, burn, seed, order="random"):
    """Interleaved repetitions in a freshly shuffled order each round, synchronised
    with block_until_ready, after `burn` untimed calls per arm."""
    rng = np.random.default_rng(seed)
    names = list(callables)
    for name in names:
        for _ in range(burn):
            jax.block_until_ready(callables[name]())
    samples = {name: [] for name in names}
    previous = {name: [] for name in names}
    last = {}
    before = None
    for _ in range(reps):
        seq = list(rng.permutation(names)) if order == "random" else names
        for name in seq:
            start = time.perf_counter()
            value = callables[name]()
            jax.block_until_ready(value)
            samples[name].append((time.perf_counter() - start) * 1e3)
            previous[name].append(before)
            last[name] = value
            before = name
    stats = {name: dict(median_ms=float(np.median(v)), min_ms=float(np.min(v)),
                        max_ms=float(np.max(v)), repetitions=v, preceded_by=previous[name])
             for name, v in samples.items()}
    return stats, last


def select_frozen(frozen_path):
    frozen = json.loads(Path(frozen_path).read_text())
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--frozen", type=Path, default=None,
                        help="held-out mode: frozen settings + head checkpoint directory")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get("smoke", {}))
    heldout = bool(cfg.get("heldout", False))
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["train_seed"]), int(cfg["dev_seed"])):
            raise RuntimeError("refusing to read a closed seed")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log(f"disk_free_gb={shutil.disk_usage(out).free / 2**30:.1f} devices={jax.devices()}")
    if jax.default_backend() != "gpu" and not args.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")

    t_job = time.time()
    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    R = int(cfg["rank"])
    modes = int(cfg["modes"])
    target = float(cfg["target_relative"])
    report = dict(schema="ns3d-shift-head-v1", config=cfg, smoke=bool(args.smoke),
                  heldout_opened=heldout, device=str(jax.devices()),
                  source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-shift/shift_rom.py",
                      "experiments/ns3d-shift-head/head_rom.py",
                      "experiments/ns3d-shift-head/fd_fom.py",
                      "experiments/ns3d-shift-head/head_ladder.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"

    def dump():
        report["elapsed_seconds"] = time.time() - t_job
        D.dump_json(out / "summary.json", report)

    frozen = None
    if heldout:
        if args.frozen is None:
            raise RuntimeError("held-out mode needs --frozen")
        frozen = json.loads((args.frozen / "frozen_settings.json").read_text())
        report["frozen_settings_sha256"] = sha256_file(args.frozen / "frozen_settings.json")
        report["frozen_head_sha256"] = sha256_file(args.frozen / frozen["head_file"])
        others = [(cfg["train_seed"], cfg["head_train_cases"])] + \
            [(s, c) for s, c in cfg["disjoint_against"]]
        overlap = disjointness(cfg["dev_seed"], cfg["dev_cases"], others)
        report["heldout_disjointness"] = overlap
        log(f"held-out disjointness (rounded-row overlaps): {overlap}")
        if any(overlap.values()):
            raise RuntimeError(f"held-out cohort overlaps another cohort: {overlap}")

    # ------------------------------------------------------------ 1. the bank --
    train_par = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    dev_par = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    report["dev_parameter_sha256"] = D.sha256_array(dev_par)
    log("generating bank training trajectories")
    train, _, _ = D.generate(train_par, n, float(cfg["dt_truth"]), horizon)
    D.require_finite(train, "training")
    centre_jit = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))
    for case in range(len(train)):
        for instant in range(train.shape[1]):
            train[case, instant] = np.asarray(centre_jit(jnp.asarray(train[case, instant])))
    basis_all, energy, _ = D.pod_basis(train.reshape(len(train) * train.shape[1], -1), R,
                                       int(cfg["gram_block"]))
    del train
    gc.collect()
    G, _ = D.orthonormalize_prefix(basis_all, R)
    G = np.ascontiguousarray(G)
    del basis_all
    gc.collect()
    if heldout:
        # The frozen head was trained against the development job's bank. Rebuilding it
        # here must give the same span; POD column signs are arbitrary, so align them
        # against the stored reference coefficients and gate the agreement.
        ref = np.load(args.frozen / frozen["bank_probe_file"])
        probe_par = F.parameters(int(cfg["train_seed"]), int(ref["cases"]))
        probe, _, _ = D.generate(probe_par, n, float(cfg["dt_truth"]), horizon)
        a_new = np.stack([G.T @ np.asarray(centre_jit(jnp.asarray(probe[c, t]))).ravel()
                          for c in range(len(probe)) for t in range(probe.shape[1])])
        a_ref = ref["coefficients"]
        signs = np.sign(np.sum(a_new * a_ref, axis=0))
        G = G * signs[None, :]
        a_new = a_new * signs[None, :]
        gap = float(np.linalg.norm(a_new - a_ref) / np.linalg.norm(a_ref))
        report["bank_rebuild_gap"] = gap
        log(f"bank rebuild vs frozen reference coefficients: {gap:.3e}")
        if gap > 1e-8:
            raise RuntimeError(f"rebuilt bank differs from the frozen one: {gap}")
        del probe

    log("generating development trajectories")
    dev, _, _ = D.generate(dev_par, n, float(cfg["dt_truth"]), horizon)
    D.require_finite(dev, "development")
    viscosities = dev_par[:, -1]
    np.save(out / "dev_truth.npy", dev)
    errors, _, travel = D.oracle_shift_errors(G, dev)
    report["bank_floor"] = dict(stats=D.stats_from_cases(errors, target), travel=travel,
                                spectrum_head=[float(x) for x in energy[:8]])
    log(f"bank floor r{R}: evolved worst {report['bank_floor']['stats']['evolved_worst']:.6f}")
    ops, rep = SR.build_operators_fast(G, n, modes, check_modes=int(cfg["check_modes"]))
    report["operator_checks"] = rep
    Gj = jnp.asarray(G)
    opsj = (Gj, jnp.asarray(ops["A"]), jnp.asarray(ops["T"]), jnp.asarray(ops["lam"]),
            jnp.asarray(ops["Dd"]))
    dump()

    # --------------------------------------------------------- 2. head data ----
    @jax.jit
    def project_frames(frames, basis):
        def one(f):
            v = SR.shift_field(f, -SR.grid_centroid(f) * n).ravel()
            a = basis.T @ v
            return a, jnp.sum(v * v) - jnp.sum(a * a)
        return jax.vmap(one)(frames)

    # development coefficients (oracle centring) for the head floors
    pieces = [project_frames(jnp.asarray(dev[c]), Gj) for c in range(len(dev))]
    dev_a = np.stack([np.asarray(a) for a, _ in pieces])
    dev_out = np.maximum(np.stack([np.asarray(o) for _, o in pieces]), 0.0)
    del pieces
    dev_u0sq = np.sum(dev[:, 0].reshape(len(dev), -1) ** 2, axis=1)

    heads = {}
    if heldout:
        ck = np.load(args.frozen / frozen["head_file"])
        k = int(frozen["k"])
        p = {key: jnp.asarray(ck[f"p_{key}"]) for key in ("W1", "b1", "W2", "b2", "W3",
                                                          "b3", "Ws")}
        heads[k] = dict(p=p, Z=np.asarray(ck["Z"]), C=np.asarray(ck["C"]))
        rot = np.load(args.frozen / frozen["rotation_file"])
        V, sig_vals = np.asarray(rot["V"]), np.asarray(rot["eigenvalues"])
        report["head_data"] = dict(note="held-out mode: frozen head and rotation loaded")
    else:
        hpar = F.parameters(int(cfg["train_seed"]), int(cfg["head_train_cases"]))
        steps_truth = D.nsteps_for(float(cfg["dt_truth"]), horizon)
        every = int(cfg["head_frame_every"])
        solver = F.make_solver(float(cfg["dt_truth"]), steps_truth, every)
        geom = F.geometry(n)
        coeffs, outside, u0sq, case_of = [], [], [], []
        log(f"generating {len(hpar)} head-training trajectories, "
            f"{steps_truth // every + 1} frames each")
        for c, row in enumerate(hpar):
            frames = solver(jnp.asarray(F.initial(n, row)), float(row[-1]), geom)
            a, o = project_frames(frames, Gj)
            coeffs.append(np.asarray(a))
            outside.append(np.asarray(o))
            u0sq.append(float(jnp.sum(frames[0] ** 2)))
            case_of.append(np.full(frames.shape[0], c))
            if (c + 1) % 64 == 0:
                log(f"  head data {c + 1}/{len(hpar)}")
        coeffs = np.concatenate(coeffs)
        outside = np.maximum(np.concatenate(outside), 0.0)
        case_of = np.concatenate(case_of)
        u0sq = np.asarray(u0sq)
        weights_all = 1.0 / u0sq[case_of]
        weights_all = weights_all / weights_all.mean()
        is_val = (case_of % int(cfg["val_every"])) == int(cfg["val_every"]) - 1
        tr, va = ~is_val, is_val
        report["head_data"] = dict(
            cases=int(len(hpar)), frames_per_case=int(steps_truth // every + 1),
            states_train=int(tr.sum()), states_val=int(va.sum()),
            bank_outside_relative_worst=float(np.sqrt(np.max(outside / u0sq[case_of]))),
            coefficient_sha256=D.sha256_array(coeffs))
        log(f"head data: {report['head_data']}")

        # importance ordering of the fixed bank, training data only: SVD of
        # G Sigma^{1/2}, Sigma the second moment of the training coefficients.
        # G is orthonormal, so the left singular vectors are G V with V the
        # eigenvectors of Sigma, ordered by eigenvalue.
        sigma = coeffs[tr].T @ coeffs[tr] / int(tr.sum())
        sig_vals, V = np.linalg.eigh(sigma)
        order = np.argsort(sig_vals)[::-1]
        sig_vals, V = sig_vals[order], V[:, order]
        V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(R)])[None, :]
        np.savez(out / "rotation.npz", V=V, eigenvalues=sig_vals)

        for k in cfg["ks"]:
            k = int(k)
            t0 = time.time()
            p, codes, info = H.init_head(jax.random.PRNGKey(int(cfg["head_seed"]) + k),
                                         coeffs[tr], weights_all[tr], k, int(cfg["width"]))
            p, codes, curve = H.train_head(p, codes, coeffs[tr], weights_all[tr],
                                           int(cfg["head_steps"]), float(cfg["head_lr"]),
                                           float(cfg["head_floor_lr"]),
                                           float(cfg["code_reg"]), log=log,
                                           log_every=int(cfg["head_log_every"]))
            Z = H.best_codes(p, codes, coeffs[tr])
            C, cvals = H.correction_directions(p, Z, coeffs[tr], weights_all[tr])
            train_seconds = time.time() - t0
            # training / validation fit through the query's own encoder
            Hc = H.head_apply(p, jnp.asarray(Z))
            fit_stats = {}
            for label, mask in (("train", tr), ("val", va)):
                rows = coeffs[mask]
                for qv in sorted({0} | {int(x) for x in cfg["q_ladder"] if x != "max"}):
                    if qv + k > R:
                        continue
                    Cq = jnp.asarray(C[:, :qv])
                    err = H.head_floor_errors(p, jnp.asarray(Z), Hc, rows[:, None, :],
                                              outside[mask][:, None], u0sq[case_of[mask]],
                                              Cq, int(cfg["ic_iters"]))[:, 0]
                    fit_stats[f"{label}_q{qv}"] = dict(median=float(np.median(err)),
                                                       worst=float(np.max(err)))
            heads[k] = dict(p=p, Z=Z, C=C, cvals=cvals, curve=curve, info=info,
                            train_seconds=train_seconds, fit=fit_stats)
            log(f"head k={k}: {train_seconds:.0f}s fit {fit_stats}")
        del coeffs
        gc.collect()

    # ---- q ladders per k (the corrections are optional; q_ladder=[0] by default), floors
    def q_list(k):
        qs = []
        for x in cfg["q_ladder"]:
            qv = R - k if x == "max" else int(x)
            if 0 <= qv <= R - k and qv not in qs:
                qs.append(qv)
        return qs

    if heldout:
        ks = [int(frozen["k"])]
    else:
        ks = [int(k) for k in cfg["ks"]]
    head_args = {}
    floors = {}
    for k in ks:
        hd = heads[k]
        Zj = jnp.asarray(hd["Z"])
        Hc = H.head_apply(hd["p"], Zj)
        Hn = jnp.sum(Hc * Hc, 1)
        head_args[k] = (hd["p"], Zj, Hc, Hn)
        floors[str(k)] = {}
        for qv in q_list(k):
            err = H.head_floor_errors(hd["p"], Zj, Hc, dev_a, dev_out, dev_u0sq,
                                      jnp.asarray(hd["C"][:, :qv]), int(cfg["ic_iters"]))
            floors[str(k)][str(qv)] = D.stats_from_cases(err, target)
        log(f"floors k={k}: " + ", ".join(
            f"q{key}={v['evolved_worst']:.5f}" for key, v in floors[str(k)].items()))
    report["head_floors"] = floors
    if not heldout:
        report["heads"] = {str(k): dict(fit=heads[k]["fit"], info=heads[k]["info"],
                                        train_seconds=heads[k]["train_seconds"],
                                        curve=heads[k]["curve"],
                                        correction_energy=[float(x) for x in
                                                           heads[k]["cvals"]])
                           for k in ks}
        # the checkpoint travels back with the pull so a held-out job can load it
        for k in ks:
            hd = heads[k]
            np.savez(out / f"head_k{k}.npz", Z=hd["Z"], C=hd["C"],
                     **{f"p_{key}": np.asarray(v) for key, v in hd["p"].items()})
        probe_cases = int(cfg["bank_probe_cases"])
        probe_par = F.parameters(int(cfg["train_seed"]), probe_cases)
        probe, _, _ = D.generate(probe_par, n, float(cfg["dt_truth"]), horizon)
        a_probe = np.stack([G.T @ np.asarray(centre_jit(jnp.asarray(probe[c, t]))).ravel()
                            for c in range(len(probe)) for t in range(probe.shape[1])])
        np.savez(out / "bank_probe.npz", coefficients=a_probe, cases=probe_cases)
        del probe
    dump()

    # ---- the importance-ordered bank: rotated operators and span floors -------
    Gr = np.ascontiguousarray(G @ V)
    rot_ops = dict(A=ops["A"] @ V, Dd=np.einsum("dmr,rs->dms", ops["Dd"], V),
                   T=np.einsum("mjk,ja,kb->mab", ops["T"], V, V, optimize=True))
    rebuilt, rrep = SR.build_operators_fast(Gr, n, modes, check_modes=int(cfg["check_modes"]))
    rot_gap = max(float(np.max(np.abs(rot_ops[key] - rebuilt[key]))
                        / np.max(np.abs(rebuilt[key]))) for key in ("A", "Dd", "T"))
    report["rotation"] = dict(eigenvalues=[float(x) for x in sig_vals],
                              rotated_vs_rebuilt_operators=rot_gap,
                              rebuilt_checks=rrep,
                              orthogonality=float(np.max(np.abs(V.T @ V - np.eye(R)))))
    log(f"rotation: rotated vs rebuilt operators {rot_gap:.3e}")
    if rot_gap > 1e-10:
        raise RuntimeError(f"rotated operators disagree with a rebuild: {rot_gap}")
    del rebuilt
    spans = [int(x) for x in (frozen["span_ladder"] if heldout else cfg["span_ladder"])]
    span_args = {}
    a_rot = dev_a @ V
    span_floors = {}
    for Rp in spans:
        span_args[Rp] = (jnp.asarray(np.ascontiguousarray(Gr[:, :Rp])),
                         jnp.asarray(np.ascontiguousarray(rot_ops["A"][:, :Rp])),
                         jnp.asarray(np.ascontiguousarray(rot_ops["T"][:, :Rp, :Rp])),
                         jnp.asarray(ops["lam"]),
                         jnp.asarray(np.ascontiguousarray(rot_ops["Dd"][:, :, :Rp])))
        tail = np.sum(a_rot[..., Rp:] ** 2, axis=-1)
        err = np.sqrt((tail + dev_out) / dev_u0sq[:, None])
        span_floors[str(Rp)] = D.stats_from_cases(err, target)
    report["span_floors"] = span_floors
    log("span floors: " + ", ".join(f"R'{k}={v['evolved_worst']:.5f}"
                                    for k, v in span_floors.items()))
    del Gr
    gc.collect()
    dump()

    # ------------------------------------------------------------ 4. rollouts --
    lin_cache, head_cache = {}, {}

    def lin_runner(dtv, iters, outputs=5, rank=None):
        rank = R if rank is None else int(rank)
        key = (float(dtv), int(iters), outputs, rank)
        if key not in lin_cache:
            steps = D.nsteps_for(float(dtv), horizon)
            lin_cache[key] = SR.make_frozen_run(float(dtv), steps, steps // outputs, n, rank,
                                                iters=int(iters),
                                                damping=float(cfg["damping"]),
                                                extrapolate=True, diagnose=False)
        return lin_cache[key]

    def head_runner(k, qv, dtv, iters, frame="free"):
        key = (k, qv, float(dtv), int(iters), frame)
        if key not in head_cache:
            steps = D.nsteps_for(float(dtv), horizon)
            head_cache[key] = H.make_head_run(float(dtv), steps, steps // 5, n, R, k, qv,
                                              iters=int(iters), damping=float(cfg["damping"]),
                                              ic_iters=int(cfg["ic_iters"]), frame=frame)
        return head_cache[key]

    def head_call(k, qv, dtv, iters, frame="free"):
        runner = head_runner(k, qv, dtv, iters, frame)
        p, Zj, Hc, Hn = head_args[k]
        C = jnp.asarray(heads[k]["C"][:, :qv])
        return lambda u, nu: runner(u, nu, *opsj, p, C, Hc, Hn, Zj)

    save_keys = set()
    saved = {}
    lin = {}
    if heldout:
        lin_grid = [(float(frozen["dt"]), int(frozen["iters"]))]
    else:
        lin_grid = [(float(d), int(i)) for d in cfg["rom_dt_ladder"] for i in cfg["iters_ladder"]]
    ladder_dt, ladder_iters = ((float(frozen["dt"]), int(frozen["iters"])) if heldout else
                               (float(cfg["ladder_dt"]), int(cfg["ladder_iters"])))
    for dtv, iters in lin_grid:
        key = f"lin_dt{dtv}_it{iters}"
        runner = lin_runner(dtv, iters)
        keep = (dtv == ladder_dt and iters == ladder_iters)
        errors, fields = run_cases(lambda u, nu: runner(u, nu, *opsj), dev, viscosities,
                                   keep=keep)
        lin[key] = dict(dt=dtv, iters=iters, steps=D.nsteps_for(dtv, horizon),
                        stats=D.stats_from_cases(errors, target), errors=errors.tolist())
        if keep:
            np.save(out / f"fields_{key}.npy", fields)
            saved[key] = "lin"
        log(f"{key}: evolved worst {lin[key]['stats']['evolved_worst']:.6f}")
    report["linear"] = lin
    dump()

    span = {}
    span_grid = ([(Rp, ladder_dt, ladder_iters) for Rp in spans] if heldout else
                 [(Rp, float(d), int(i)) for Rp in spans for d in cfg["rom_dt_ladder"]
                  for i in cfg["iters_ladder"]])
    for Rp, dtv, iters in span_grid:
        key = f"span{Rp}_dt{dtv}_it{iters}"
        runner = lin_runner(dtv, iters, rank=Rp)
        keep = (dtv == ladder_dt and iters == ladder_iters)
        try:
            errors, fields = run_cases(lambda u, nu, r_=runner, a_=span_args[Rp]: r_(u, nu, *a_),
                                       dev, viscosities, keep=keep)
            stats, finite = D.stats_from_cases(errors, target), True
        except RuntimeError as exc:
            log(f"{key}: {exc}")
            errors, fields, stats, finite = None, None, None, False
        span[key] = dict(rank=Rp, dt=dtv, iters=iters, steps=D.nsteps_for(dtv, horizon),
                         finite=finite, stats=stats,
                         errors=None if errors is None else errors.tolist())
        if keep and finite:
            np.save(out / f"fields_{key}.npy", fields)
            saved[key] = "span"
        del fields
        log(f"{key}: evolved worst {stats['evolved_worst'] if stats else float('nan'):.6f}")
    report["span"] = span
    dump()

    frontier = {}
    if heldout:
        grid = [(int(frozen["k"]), int(qv), ladder_dt, ladder_iters)
                for qv in frozen["q_ladder"]]
    else:
        grid = [(k, qv, float(d), int(i)) for k in ks for qv in q_list(k)
                for d in cfg["rom_dt_ladder"] for i in cfg["iters_ladder"]]
    for k, qv, dtv, iters in grid:
        key = f"k{k}_q{qv}_dt{dtv}_it{iters}"
        keep = (dtv == ladder_dt and iters == ladder_iters)
        try:
            errors, fields = run_cases(head_call(k, qv, dtv, iters), dev, viscosities,
                                       keep=keep)
            stats = D.stats_from_cases(errors, target)
            finite = True
        except RuntimeError as exc:
            log(f"{key}: {exc}")
            errors, fields, stats, finite = None, None, None, False
        frontier[key] = dict(k=k, q=qv, dt=dtv, iters=iters,
                             steps=D.nsteps_for(dtv, horizon), finite=finite, stats=stats,
                             errors=None if errors is None else errors.tolist())
        if keep and finite:
            saved[key] = "head"
            np.save(out / f"fields_{key}.npy", fields)
        del fields
        log(f"{key}: evolved worst "
            f"{stats['evolved_worst'] if stats else float('nan'):.6f}")
    report["frontier"] = frontier
    dump()

    # selection of k (pre-registered): smallest development worst at q=0 on the
    # ladder setting; within 1 % relative, the smaller k.
    if heldout:
        k_sel = int(frozen["k"])
    else:
        cand = []
        for k in ks:
            e = frontier[f"k{k}_q0_dt{ladder_dt}_it{ladder_iters}"]
            if e["finite"]:
                cand.append((e["stats"]["evolved_worst"], k))
        best = min(v for v, _ in cand)
        k_sel = min(k for v, k in cand if v <= best * 1.01)
    report["k_selected"] = k_sel
    log(f"selected k = {k_sel}")

    # controls on the selected head: frozen frame must fail; driver parity vs LM
    zero_err, _ = run_cases(head_call(k_sel, 0, ladder_dt, ladder_iters, frame="zero"),
                            dev, viscosities)
    report["control_frame_zero"] = dict(k=k_sel, q=0, stats=D.stats_from_cases(zero_err, target))
    log(f"control frame-zero: evolved worst "
        f"{report['control_frame_zero']['stats']['evolved_worst']:.4f}")
    qref = 0
    steps = D.nsteps_for(ladder_dt, horizon)
    refrun = H.make_head_reference(ladder_dt, steps, steps // 5, n, R, k_sel, qref,
                                   budget=int(cfg["lm_budget"]), gtol=float(cfg["lm_gtol"]),
                                   ic_iters=int(cfg["ic_iters"]))
    p, Zj, Hc, Hn = head_args[k_sel]
    Cref = jnp.asarray(heads[k_sel]["C"][:, :qref])
    parity = {}
    reasons = []
    ref_fields = []
    for case in range(min(4, len(dev))):
        f_ref, info = refrun(jnp.asarray(dev[case, 0]), float(viscosities[case]), *opsj,
                             p, Cref, Hc, Hn, Zj)
        ref_fields.append(np.asarray(f_ref))
        reasons.append(np.asarray(info[1]).ravel().tolist())
    ref_fields = np.stack(ref_fields)
    for iters in sorted(set([int(i) for i in cfg["iters_ladder"]] + [ladder_iters])):
        got = np.stack([np.asarray(head_call(k_sel, qref, ladder_dt, iters)(
            jnp.asarray(dev[c, 0]), float(viscosities[c]))) for c in range(len(ref_fields))])
        parity[str(iters)] = float(np.linalg.norm(got - ref_fields) / np.linalg.norm(ref_fields))
    report["head_parity_vs_lm"] = dict(k=k_sel, q=qref, dt=ladder_dt, cases=len(ref_fields),
                                       parity=parity,
                                       lm_reasons=np.bincount(np.concatenate(
                                           [np.asarray(r) for r in reasons]),
                                           minlength=5).tolist())
    log(f"head driver parity vs LM: {parity}")
    dump()

    # ---------------------------------------------------------------- 5. FOMs --
    geom = F.geometry(n)
    cnab = {}
    for st in cfg["cnab2_steps"]:
        st = int(st)
        dtv = horizon / st
        solver = F.make_solver(dtv, st, st // 5)
        try:
            errors, fields = run_cases(lambda u, nu: solver(u, nu, geom), dev, viscosities,
                                       keep=True)
            stats = D.stats_from_cases(errors, target)
            unstable = bool(stats["evolved_worst"] > UNSTABLE)
            np.save(out / f"fields_cnab2_s{st}.npy", fields.astype(np.float32))
            saved[f"cnab2_s{st}"] = "cnab2"
            del fields
        except RuntimeError as exc:
            log(f"CNAB2 steps={st}: {exc}")
            errors, stats, unstable = None, None, True
        cnab[str(st)] = dict(steps=st, dt=dtv, stats=stats, unstable=unstable,
                             errors=None if errors is None else errors.tolist(),
                             reference_itself=bool(abs(dtv - float(cfg["dt_truth"])) < 1e-15))
        log(f"CNAB2 steps={st}: {stats['evolved_worst'] if stats else float('nan'):.6f}")
    report["cnab2"] = cnab
    dump()

    fd = {}
    fd_grid = [(n, int(st), float(tol)) for st in cfg["fd_steps"] for tol in cfg["fd_rtols"]]
    fd_grid += [(int(m) * n, int(st), float(tol)) for m, st, tol in cfg.get("fd_fine", [])]
    fine_inputs = {}
    for mesh, st, tol in fd_grid:
        key = f"fd_n{mesh}_s{st}_tol{tol:g}"
        solver = FD.make_fd_solver(mesh, horizon / st, st, st // 5, rtol=tol,
                                   maxiter=int(cfg["fd_maxiter"]), diagnose=True)
        inputs = None
        restrict = None
        if mesh != n:
            if mesh not in fine_inputs:
                fine_inputs[mesh] = [np.asarray(F.initial(mesh, row)) for row in dev_par]
            inputs = fine_inputs[mesh]
            ratio = mesh // n
            restrict = (lambda o, r=ratio: o[0][..., ::r, ::r, ::r])
        else:
            restrict = (lambda o: o[0])
        iters_v, iters_p, ratios = [], [], []

        def call(u, nu, solver=solver):
            res = solver(u, nu)
            iters_v.append(int(jnp.max(res[1][0])))
            iters_p.append(float(jnp.mean(res[1][2])))
            ratios.append(float(jnp.max(jnp.maximum(res[1][1], res[1][3]))))
            return res
        try:
            errors, fields = run_cases(call, dev, viscosities, inputs=inputs,
                                       restrict=restrict, keep=True)
            stats = D.stats_from_cases(errors, target)
            unstable = bool(stats["evolved_worst"] > UNSTABLE)
            np.save(out / f"fields_{key}.npy", fields.astype(np.float32))
            saved[key] = "fd_cg"
            del fields
        except RuntimeError as exc:
            log(f"{key}: {exc}")
            errors, stats, unstable = None, None, True
        fd[key] = dict(mesh=mesh, steps=st, dt=horizon / st, rtol=tol, stats=stats,
                       unstable=unstable, errors=None if errors is None else errors.tolist(),
                       cg_viscous_max_iters=max(iters_v) if iters_v else None,
                       cg_pressure_mean_iters=float(np.mean(iters_p)) if iters_p else None,
                       cg_worst_final_ratio=max(ratios) if ratios else None,
                       cg_hit_maxiter=bool(iters_v and max(iters_v) >= int(cfg["fd_maxiter"])))
        log(f"{key}: {stats['evolved_worst'] if stats else float('nan'):.6f} "
            f"cg_p_mean={fd[key]['cg_pressure_mean_iters']}")
    report["fd_cg"] = fd
    dump()

    # -------------------------------------------------------------- 6. timing --
    case = int(cfg["timing_case"])
    u0, nu = jnp.asarray(dev[case, 0]), float(viscosities[case])
    fast = {}
    for key, e in lin.items():
        r_ = lin_runner(e["dt"], e["iters"])
        fast[f"query_{key}"] = (lambda r_=r_: r_(u0, nu, *opsj))
    for key, e in span.items():
        if not e["finite"]:
            continue
        r_ = lin_runner(e["dt"], e["iters"], rank=e["rank"])
        fast[f"query_{key}"] = (lambda r_=r_, a_=span_args[e["rank"]]: r_(u0, nu, *a_))
    for key, e in frontier.items():
        if not e["finite"]:
            continue
        fn = head_call(e["k"], e["q"], e["dt"], e["iters"])
        fast[f"query_{key}"] = (lambda fn=fn: fn(u0, nu))
    for key, e in cnab.items():
        st = e["steps"]
        s_ = F.make_solver(horizon / st, st, st // 5)
        fast[f"CNAB2_s{st}"] = (lambda s_=s_: s_(u0, nu, geom))

    @jax.jit
    def piece_initial(u0, basis):
        c0 = SR.grid_centroid(u0)
        return basis.T @ SR.shift_field(u0, -c0 * n).ravel(), c0

    a0, c0 = piece_initial(u0, Gj)

    @jax.jit
    def piece_output(a, c, basis):
        return SR.shift_field((basis @ a).reshape(3, n, n, n), c * n)

    fast["piece_initial"] = lambda: piece_initial(u0, Gj)
    fast["piece_output"] = lambda: piece_output(a0, c0, Gj)

    slow = {}
    for key, e in fd.items():
        solver = FD.make_fd_solver(e["mesh"], e["dt"], e["steps"], e["steps"] // 5,
                                   rtol=e["rtol"], maxiter=int(cfg["fd_maxiter"]))
        uin = jnp.asarray(u0 if e["mesh"] == n else fine_inputs[e["mesh"]][case])
        slow[key] = (lambda s_=solver, uin=uin: s_(uin, nu))

    sent_keys = [f"query_span{max(spans)}_dt{ladder_dt}_it{ladder_iters}",
                 f"query_span{min(spans)}_dt{ladder_dt}_it{ladder_iters}",
                 f"query_k{k_sel}_q0_dt{ladder_dt}_it{ladder_iters}"]
    sent_keys = [s for s in dict.fromkeys(sent_keys) if s in fast]
    sentinels = {s: fast[s] for s in sent_keys}
    reps = int(cfg["timing_repetitions"])
    cool = float(cfg.get("cooldown_seconds", 1.0))
    ladder_keys = [k_ for k_ in fast if k_.startswith("query_")
                   and k_.endswith(f"_dt{ladder_dt}_it{ladder_iters}")
                   and ("_span" in k_ or "_k" in k_)]

    def gpu_state(label):
        try:
            return dict(label=label, t=time.time() - t_job, smi=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                 "clocks_throttle_reasons.active", "--format=csv,noheader"], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            return dict(label=label, smi="unavailable")

    states = [gpu_state("start")]
    log(f"timing v2: {len(fast)} fast arms, {len(slow)} slow arms, {reps} reps, "
        f"cooldown {cool}s, {len(ladder_keys)} ladder arms")
    burn_gpu()
    t_sent0, _ = time_block(sentinels, reps, 2, seed=1)
    t_fast, last = time_block(fast, reps, 2, seed=2)
    states.append(gpu_state("after fast block"))
    # slow arms in their own phase; every timed call starts after the same idle
    # cool-down, so a heavy arm is not timed in the wake of another heavy arm
    rng_s = np.random.default_rng(3)
    for name in slow:
        jax.block_until_ready(slow[name]())
    t_slow = {name: [] for name in slow}
    for _ in range(max(5, int(cfg["slow_repetitions"]))):
        for name in rng_s.permutation(list(slow)):
            time.sleep(cool)
            t0 = time.perf_counter()
            jax.block_until_ready(slow[name]())
            t_slow[name].append((time.perf_counter() - t0) * 1e3)
    t_slow = {k_: dict(median_ms=float(np.median(v_)), min_ms=float(np.min(v_)),
                       max_ms=float(np.max(v_)), repetitions=v_) for k_, v_ in t_slow.items()}
    states.append(gpu_state("after slow block"))
    # the heavy neighbour: the longest FD-CG arm on the job's own mesh
    heavy = max((k_ for k_ in slow if fd[k_]["mesh"] == n), key=lambda k_: t_slow[k_]["median_ms"])
    heavy_fn = slow[heavy]
    after_heavy = {k_: [] for k_ in ladder_keys}
    after_cool = {k_: [] for k_ in ladder_keys}
    rng_h = np.random.default_rng(4)
    for _ in range(reps):
        for name in rng_h.permutation(ladder_keys):
            jax.block_until_ready(heavy_fn())
            t0 = time.perf_counter()
            jax.block_until_ready(fast[name]())
            after_heavy[name].append((time.perf_counter() - t0) * 1e3)
            jax.block_until_ready(heavy_fn())
            time.sleep(cool)
            t0 = time.perf_counter()
            jax.block_until_ready(fast[name]())
            after_cool[name].append((time.perf_counter() - t0) * 1e3)
    states.append(gpu_state("after neighbour block"))
    t_sent1, _ = time_block(sentinels, reps, 1, seed=5)
    gate_ratio = float(cfg["neighbour_gate_ratio"])
    gate = {}
    for name in ladder_keys:
        base = t_fast[name]["median_ms"]
        gate[name] = dict(fast_block_ms=base,
                          after_heavy_ms=float(np.median(after_heavy[name])),
                          after_heavy_cooldown_ms=float(np.median(after_cool[name])),
                          after_heavy_ratio=float(np.median(after_heavy[name]) / base),
                          after_cooldown_ratio=float(np.median(after_cool[name]) / base),
                          repetitions_after_heavy=after_heavy[name],
                          repetitions_after_cooldown=after_cool[name])
    for name in sentinels:
        gate.setdefault(name, {}).update(
            solo_before_ms=t_sent0[name]["median_ms"], solo_end_ms=t_sent1[name]["median_ms"],
            fast_block_ratio=float(t_fast[name]["median_ms"] / t_sent0[name]["median_ms"]))
    passed_block = all(g["fast_block_ratio"] <= gate_ratio for g in gate.values()
                       if "fast_block_ratio" in g)
    passed_cool = all(g["after_cooldown_ratio"] <= gate_ratio for g in gate.values()
                      if "after_cooldown_ratio" in g)
    report["timing"] = dict(
        protocol="v2", case=case, fast=t_fast, slow=t_slow, sentinels_before=t_sent0,
        sentinels_end=t_sent1, heavy_neighbour=heavy, cooldown_seconds=cool,
        neighbour_gate=gate, gpu_states=states,
        neighbour_gate_fast_block_passed=bool(passed_block),
        neighbour_gate_after_cooldown_passed=bool(passed_cool),
        neighbour_gate_passed=bool(passed_block and passed_cool),
        note=("CNAB2 comparisons use fast-block medians (all arms interleaved, randomised). "
              "FD-CG arms are timed after an idle cool-down; the ladder arms are also timed "
              "immediately after the heavy FD-CG neighbour (after_heavy, no cool-down), and "
              "FD-CG speedups are reported against both the fast-block and the after-heavy "
              "ROM medians, the latter being the conservative one."))
    log(f"neighbour gate: block {passed_block} cooldown {passed_cool}; after-heavy ratios "
        f"{ {k_.replace('query_', ''): round(g['after_heavy_ratio'], 3) for k_, g in gate.items() if 'after_heavy_ratio' in g} }")
    # errors of the timed outputs must equal the accuracy pass for that case
    agree = 0.0
    for name, value in last.items():
        if not name.startswith("query_"):
            continue
        key = name[len("query_"):]
        src = lin.get(key) or span.get(key) or frontier.get(key)
        e = rel_case(np.asarray(value).reshape(dev.shape[1:]), dev, case)
        agree = max(agree, float(np.max(np.abs(e - np.asarray(src["errors"][case])))))
    for name, value in last.items():
        if name.startswith("CNAB2_s"):
            src = cnab[name[len("CNAB2_s"):]]
            if src["errors"] is not None:
                e = rel_case(np.asarray(value).reshape(dev.shape[1:]), dev, case)
                agree = max(agree, float(np.max(np.abs(e - np.asarray(src["errors"][case])))))
    report["timing"]["timed_output_error_agreement"] = agree
    log(f"timed outputs vs accuracy pass: {agree:.3e}")
    report["saved_fields"] = saved
    dump()
    log("done")


if __name__ == "__main__":
    main()
