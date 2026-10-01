"""Stage 1 (one allocation per bank): train ONE coordinate-network bank and measure its
projection floor at every mesh (bar (a)).

  1. data     training seed only; trajectory c is generated at mesh meshes[c % 3]
              (pooled across meshes), six frames each; every frame is centred on its own
              energy centroid (or left uncentred for the fixed-frame control) and
              resampled spectrally onto the training grid n_t; weight 1/||u0||^2 per
              trajectory (the error metric's normalisation)
  2. compress the weighted snapshot matrix to its leading K singular pairs, Y = U_K S_K;
              the variable-projection objective on Y equals the snapshot objective up
              to the reported tail
  3. train    for each architecture in the config: Adam on 1 - ||P_G Y||^2/||Y||^2 with
              G the curl-coordnet sampled on the n_t grid; keep the architecture with the
              lower final TRAINING loss (no development data is used to choose)
  4. order    the paper's QR+SVD rotation T on the training grid (training states only)
  5. floors   at every mesh n: G_n = lowdin(P_n(sample_n(g)) T); oracle-shift floor of
              the 16 development cases, nested prefixes R', plus references in the same
              job: the parent's rank-64 centred POD (128 cases, six frames, at mesh n) and
              the weighted POD of this job's own training data (the optimum of step 3)
  6. checks   autodiff vs spectral derivatives; autodiff divergence; how much the mesh's
              Leray / 2-3 projection removes; how far the sampled bank is from orthonormal
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
import optax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import ns3d_fom as F  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402
import coordnet as CN  # noqa: E402
import bankio as BI  # noqa: E402

log = D.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get("smoke", {}))
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["train_seed"]), int(cfg["dev_seed"])):
            raise RuntimeError("refusing to read a closed seed")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu" and not args.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    t_job = time.time()
    nt = int(cfg["train_grid"])
    centred = bool(cfg.get("centred", True))
    report = dict(schema="ns3d-coordnet-bank-v1", config=cfg, smoke=bool(args.smoke),
                  device=str(jax.devices()), source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: D.sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-coordnet/coordnet.py",
                      "experiments/ns3d-coordnet/bankio.py",
                      "experiments/ns3d-coordnet/train_bank.py",
                      "experiments/ns3d-shift/shift_rom.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"

    def dump():
        report["elapsed_seconds"] = time.time() - t_job
        D.dump_json(out / "summary.json", report)

    # ------------------------------------------------------------- 1. data ----
    par = F.parameters(int(cfg["train_seed"]), int(cfg["bank_cases"]))
    meshes = [int(m) for m in cfg["data_meshes"]]
    states, norms, u0sq, case_of, dropped = [], [], [], [], []
    solvers = {}
    centre = {}
    for m in set(meshes):
        steps = D.nsteps_for(float(cfg["dt_truth"]), float(cfg["horizon"]))
        solvers[m] = (F.make_solver(float(cfg["dt_truth"]), steps, steps // 5), F.geometry(m))
        centre[m] = jax.jit(lambda f, m=m: SR.shift_field(f, -SR.grid_centroid(f) * m))
    for c, row in enumerate(par):
        m = meshes[c % len(meshes)]
        solver, geom = solvers[m]
        frames = solver(jnp.asarray(F.initial(m, row)), float(row[-1]), geom)
        u0sq.append(float(jnp.mean(frames[0] ** 2)))      # native mesh, before resampling
        for t in range(frames.shape[0]):
            f = centre[m](frames[t]) if centred else frames[t]
            g, lost = BI.resample(np.asarray(f), nt)
            states.append(g.ravel())
            dropped.append(lost)
            case_of.append(c)
        if (c + 1) % 64 == 0:
            log(f"bank data {c + 1}/{len(par)}")
    X = np.stack(states)
    del states
    case_of = np.asarray(case_of)
    u0sq = np.asarray(u0sq)
    w = 1.0 / u0sq[case_of]
    w = w / w.mean()
    norms = np.sqrt(np.sum(X * X, axis=1))
    report["data"] = dict(states=int(len(X)), cases=int(len(par)), meshes=meshes,
                          train_grid=nt, centred=centred,
                          resample_dropped_energy_worst=float(np.max(dropped)),
                          state_sha256=D.sha256_array(X[:: max(1, len(X) // 64)]))
    log(f"data: {report['data']}")

    # --------------------------------------------------------- 2. compress ----
    K = int(cfg["compress_K"])
    Xw = jnp.asarray(X * np.sqrt(w)[:, None])
    gram = np.asarray(Xw @ Xw.T)
    ev, V = np.linalg.eigh(gram)
    order = np.argsort(ev)[::-1]
    ev, V = np.maximum(ev[order], 0.0), V[:, order]
    K = min(K, int(np.sum(ev > ev[0] * 1e-15)))
    Y = np.asarray(Xw.T @ jnp.asarray(V[:, :K]))                    # (dof, K) = U_K S_K
    del Xw, gram
    gc.collect()
    tail = float(ev[K:].sum() / ev.sum())
    Yn = Y / np.linalg.norm(Y)
    report["compression"] = dict(K=K, tail_fraction=tail,
                                 spectrum=[float(x) for x in (ev[:K] / ev[0])[::8]])
    log(f"compression K={K} tail={tail:.3e}")
    # weighted POD of this job's own data: the optimum of the training objective
    pod_same = Y / np.sqrt(ev[:K])[None, :]                         # orthonormal U_K
    dump()

    # ------------------------------------------------------------ 3. train ----
    R = int(cfg["rank"])
    Yj = jnp.asarray(Yn)
    sample_t = CN.make_sampler(nt, chunk=int(cfg.get("sample_chunk", nt ** 3)))
    sample_fn = CN.make_sample_fn(nt, chunk=int(cfg.get("sample_chunk", nt ** 3)))
    pts_t = jnp.asarray(CN.grid_points(nt))
    failures = []
    fits = {}
    best = None
    for arch in cfg["architectures"]:
        name = arch["name"]
        key = jax.random.PRNGKey(int(cfg["seed"]))
        p = CN.init_params(key, R, int(arch["width"]), int(arch["depth"]), int(arch["kmax"]))
        G0 = sample_t(p)
        s = float(jnp.sqrt(R / jnp.sum(G0 * G0)))
        w_, b_ = p["layers"][-1]
        p["layers"][-1] = (w_ * s, b_)
        every = int(cfg["log_every"])

        def make_chunk(opt):
            @jax.jit
            def chunk(p, state, Y, pts):
                def body(carry, _):
                    p, state = carry
                    v, g = jax.value_and_grad(lambda q: CN.projection_loss(sample_fn(q, pts), Y))(p)
                    g = dict(g, B=jnp.zeros_like(g["B"]))           # frequencies are fixed
                    upd, state = opt.update(g, state, p)
                    return (optax.apply_updates(p, upd), state), v
                (p, state), vals = jax.lax.scan(body, (p, state), None, length=every)
                return p, state, vals[-1]
            return chunk

        # The step count is fixed from the measured step time and a wall budget, so the
        # cosine schedule always completes. Calibration runs on a throw-away copy.
        if "steps" in arch:
            steps = int(arch["steps"])
            calib = None
        else:
            # the same clipped Adam as the real run, constant small rate
            copt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
            cchunk = make_chunk(copt)
            cp, cs, _ = cchunk(p, copt.init(p), Yj, pts_t)               # compile
            jax.block_until_ready(cp)
            t_c = time.time()
            cp, cs, _ = cchunk(cp, cs, Yj, pts_t)
            jax.block_until_ready(cp)
            per_step = (time.time() - t_c) / every
            # 3 % reserve for logging / checkpoints / recompilation
            steps = int(0.97 * float(arch["train_seconds"]) / per_step) // every * every
            steps = max(steps, 4 * int(arch.get("warmup", 1000)), every)
            calib = dict(seconds_per_step=per_step, steps=steps)
            del cp, cs, cchunk
            log(f"[{name}] calibration: {per_step * 1e3:.1f} ms/step -> {steps} steps")
        sched = optax.warmup_cosine_decay_schedule(0.0, float(arch["lr"]),
                                                   int(arch.get("warmup", 1000)), steps,
                                                   float(arch["lr"]) * float(arch["final_lr_ratio"]))
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
        state = opt.init(p)
        chunk = make_chunk(opt)

        curve = []
        t0 = time.time()
        done = 0
        while done < steps:
            p, state, v = chunk(p, state, Yj, pts_t)
            done += every
            curve.append((done, float(v), time.time() - t0))
            log(f"  [{name}] step {done}/{steps} loss {float(v):.4e} ({time.time() - t0:.0f}s)")
            if not np.isfinite(float(v)):
                raise RuntimeError("nonfinite bank training")
            if done % (every * 10) == 0:
                BI.save_params(out / f"bank_{name}_partial.npz", p)
        if np.max(np.abs(np.asarray(p["B"]) - CN.frequency_set(int(arch["kmax"])))) != 0.0:
            raise RuntimeError("Fourier frequencies changed during training")
        final = float(CN.projection_loss(sample_t(p), Yj))
        G_end = np.asarray(sample_t(p))
        exact_raw, sv_raw = CN.projection_loss_exact(G_end, Yn)
        exact_proj, _ = CN.projection_loss_exact(CN.leray_mask(G_end, nt)[0], Yn)
        del G_end
        fits[name] = dict(arch=arch, final_train_loss=final, curve=curve, calibration=calib,
                          final_loss_qr_raw=exact_raw, final_loss_qr_projected=exact_proj,
                          bank_singular_value_ratio=float(sv_raw[-1] / sv_raw[0]),
                          steps=steps,
                          seconds=time.time() - t0,
                          eckart_young_optimum=float(ev[R:K].sum() / ev[:K].sum()))
        BI.save_params(out / f"bank_{name}.npz", p)
        (out / f"bank_{name}_partial.npz").unlink(missing_ok=True)
        log(f"[{name}] final training loss {final:.4e} (optimum at R {fits[name]['eckart_young_optimum']:.4e})")
        if best is None or final < fits[best]["final_train_loss"]:
            best = name
        report["fits"] = fits
        dump()
    report["selected_architecture"] = best
    p = BI.load_params(out / f"bank_{best}.npz")
    log(f"selected architecture by training loss: {best}")

    # ------------------------------------------------------------ 4. order ----
    Gt = np.asarray(sample_t(p))
    Gt_p, removed_t = CN.leray_mask(Gt, nt)
    T, svals, _ = CN.order_bank(Gt_p, X, norms)
    np.savez(out / "bank_selected.npz", T=T, singular_values=svals,
             **{k: v for k, v in np.load(out / f"bank_{best}.npz").items()})
    report["ordering"] = dict(grid=nt, singular_values=[float(x) for x in svals],
                              leray_removed_worst=float(np.max(removed_t)))
    del X
    gc.collect()
    dump()

    # ----------------------------------------------------------- 5. floors ----
    dev_par = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    target = float(cfg["target_relative"])
    floors = {}
    ref_bank = None
    for n in sorted(int(m) for m in cfg["floor_meshes"]):
        log(f"floors at {n}^3")
        dev, _, _ = D.generate(dev_par, n, float(cfg["dt_truth"]), float(cfg["horizon"]))
        Gn, info = BI.mesh_bank(p, T, n, chunk=int(cfg.get("sample_chunk_eval", 32768)),
                                abort=float(cfg.get("orthonormality_abort", 0.05)))
        info.pop("lowdin_factor")
        entry = dict(bank=info)
        if info["orthonormality_before_lowdin"] > 1e-3:
            failures.append(f"orthonormality before Lowdin {info['orthonormality_before_lowdin']:.2e} at {n}^3 (> 1e-3, bar c flag)")
        if centred:
            for Rp in sorted({int(x) for x in cfg["prefixes"] if int(x) <= R} | {R}):
                errors, _, travel = D.oracle_shift_errors(Gn[:, :Rp], dev)
                entry[f"coordnet_R{Rp}"] = D.stats_from_cases(errors, target)
                log(f"  coordnet R'={Rp}: evolved worst {entry[f'coordnet_R{Rp}']['evolved_worst']:.6f}")
            entry["travel"] = travel
            # no-shift control on the same bank: development states are not centred
            errors, _ = D.project_errors(Gn, dev)
            entry["coordnet_no_centring"] = D.stats_from_cases(errors, target)
            # the optimum of the training objective, resampled to this mesh
            Up = np.stack([BI.resample(pod_same[:, j].reshape(3, nt, nt, nt), n)[0].ravel()
                           for j in range(R)], axis=1)
            Up, _ = np.linalg.qr(CN.leray_mask(Up, n)[0])
            errors, _, _ = D.oracle_shift_errors(Up, dev)
            entry[f"same_data_pod_R{R}"] = D.stats_from_cases(errors, target)
            if n in [int(m) for m in cfg.get("parent_pod_meshes", [])]:
                tp = F.parameters(int(cfg["train_seed"]), int(cfg["parent_pod_cases"]))
                tr, _, _ = D.generate(tp, n, float(cfg["dt_truth"]), float(cfg["horizon"]))
                cj = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))
                for a in range(len(tr)):
                    for b in range(tr.shape[1]):
                        tr[a, b] = np.asarray(cj(jnp.asarray(tr[a, b])))
                basis, _, _ = D.pod_basis(tr.reshape(len(tr) * tr.shape[1], -1), 64,
                                          int(cfg["gram_block"]))
                del tr
                gc.collect()
                Gpod, _ = D.orthonormalize_prefix(basis, 64)
                errors, _, _ = D.oracle_shift_errors(Gpod, dev)
                entry["parent_pod_R64"] = D.stats_from_cases(errors, target)
                want = cfg.get("parent_pod_floor_reference", {}).get(str(n))
                if want is not None:
                    gapp = abs(entry["parent_pod_R64"]["evolved_worst"] - want) / want
                    entry["parent_pod_R64_reproduction"] = dict(reference=want, relative_gap=gapp,
                                                                passed=bool(gapp <= 1e-6))
                    if gapp > 1e-6:
                        failures.append(f"parent POD-64 floor at {n}^3 not reproduced: {gapp:.2e}")
                del basis, Gpod
                log(f"  parent POD-64: evolved worst {entry['parent_pod_R64']['evolved_worst']:.6f}")
        else:
            # fixed-frame control: the bank was trained on uncentred states; the
            # development states are projected as they are (no shift anywhere)
            errors, _ = D.project_errors(Gn, dev)
            entry[f"coordnet_fixed_frame_R{R}"] = D.stats_from_cases(errors, target)
            log(f"  fixed-frame coordnet R={R}: evolved worst "
                f"{entry[f'coordnet_fixed_frame_R{R}']['evolved_worst']:.6f}")
            Up = np.stack([BI.resample(pod_same[:, j].reshape(3, nt, nt, nt), n)[0].ravel()
                           for j in range(R)], axis=1)
            Up, _ = np.linalg.qr(CN.leray_mask(Up, n)[0])
            errors, _ = D.project_errors(Up, dev)
            entry[f"same_data_pod_fixed_frame_R{R}"] = D.stats_from_cases(errors, target)
        # 6. derivative and divergence checks at this mesh
        if n in [int(m) for m in cfg.get("derivative_meshes", [])]:
            entry["derivatives"] = BI.derivative_check(p, T, n)
            log(f"  derivative check: {entry['derivatives']}")
        # cross-mesh consistency: this mesh's bank, spectrally restricted to the
        # smallest floor mesh, against the bank sampled there directly
        if ref_bank is None:
            ref_bank = (n, Gn)
        else:
            n0, G0 = ref_bank
            Gr_ = np.stack([BI.resample(Gn[:, j].reshape(3, n, n, n), n0)[0].ravel()
                            for j in range(Gn.shape[1])], axis=1) * (n / n0) ** 1.5
            col = np.linalg.norm(Gr_ - G0, axis=0) / np.linalg.norm(G0, axis=0)
            entry["cross_mesh_vs_smallest"] = dict(against=int(n0), worst_column=float(col.max()),
                                                   median_column=float(np.median(col)))
            log(f"  cross-mesh consistency vs {n0}^3: worst column {col.max():.3e}")
        floors[str(n)] = entry
        del dev, Gn
        gc.collect()
        report["floors"] = floors
        dump()
    pts = jnp.asarray(np.random.default_rng(5).uniform(size=(256, 3)))
    div = CN.divergence(p, pts)
    vel = CN.velocity(p, pts)
    report["autodiff_divergence_relative"] = float(jnp.max(jnp.abs(div)) / jnp.max(jnp.abs(vel)))
    report["failures"] = failures
    report["status"] = "final" if not failures else "flagged"
    dump()
    log(f"done; failures: {failures}")


if __name__ == "__main__":
    main()
