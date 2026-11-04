"""Stage 1 of ns3d-pod-match (one allocation per arm): train ONE coordinate-network bank
with a STAGED optimiser (Adam, then L-BFGS) toward convergence, and measure its projection
floor at 32^3 / 64^3 / 96^3 against POD-64.

Derived from experiments/ns3d-coordnet/train_bank.py (same data, compression, ordering,
floors, controls and checks; coordnet.py / bankio.py are imported unchanged from that
lane). What is new:

  objective  "vp_band"     (arm a, the A3 recipe): 1 - ||P_G Y||^2/||Y||^2
                           + lambda ||G - Pi_kcut G||^2/||G||^2
             "pod_regress" (arm b): regression of column r onto the r-th weighted POD mode
                           of the SAME pooled training data (the optimum of the variable-
                           projection objective), with energy weights w_r = s_r^2 / sum s^2
                           (importance ordering kept: column r <-> mode r):
                             sum_r w_r ||G_r - U_r||^2                      (48^3 nodes)
                           + sum_r w_r ||g_r(Q) - f_r(Q)||^2 / sum_r w_r ||f_r(Q)||^2
                           + sum_r w_r ||grad g_r(Q) - grad f_r(Q)||^2 / sum_r w_r ||grad f_r(Q)||^2
                           f_r = the trigonometric interpolant of mode r (exact: the modes are
                           trig polynomials on the 48^3 grid), Q = fixed random off-grid
                           points; grad g by autodiff of the network, grad f spectral.
  optimiser  Adam (warm-up + cosine, clip 1, step count from a measured step time and a wall
             budget) then optax L-BFGS with a zoom line search on the SAME objective, to a
             wall deadline. The Fourier frequencies are never optimised (only the layers are).
  reporting  at the end of each stage: the training objective, the exact (QR) variable-
             projection objective of the raw and P_48-projected bank against its Eckart-Young
             optimum, the band fraction, per-column regression errors (arm b). The final bank
             is the end of the last stage (pre-registered); the Adam-stage bank's R=64 floors
             are measured as a diagnostic only.
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
              "experiments/ns3d-grok", "experiments/ns3d-shift", "experiments/ns3d-coordnet"):
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


# --------------------------------------------------------------- POD targets ---

def trig_interpolant(modes, nt, pts, chunk=128):
    """Exact trigonometric interpolant of grid fields and its gradient at points.

    modes (3 nt^3, R) on the nt^3 grid (field layout (3, nt, nt, nt), Euclidean scaling);
    pts (P, 3) in [0, 1)^3. Returns values (P, 3, R) and gradients (3_e, P, 3, R) in the
    NETWORK's units (the sampled bank is nt^{-3/2} g(x_i), so f = nt^{3/2} * interpolant)."""
    R = modes.shape[1]
    f = np.asarray(modes).T.reshape(R, 3, nt, nt, nt)
    c = jnp.asarray(np.fft.fftn(f, axes=(-3, -2, -1), norm="forward") * nt ** 1.5)
    k = jnp.asarray(np.fft.fftfreq(nt) * nt)                         # integer wave numbers
    P = pts.shape[0]
    assert P % chunk == 0

    @jax.jit
    def run(c, pts):
        def one(x):                                                  # x (chunk, 3)
            e = [jnp.exp(2j * jnp.pi * x[:, a, None] * k[None, :]) for a in range(3)]
            def contract(cc):
                t = jnp.einsum("rcabd,pd->rcabp", cc, e[2])
                t = jnp.einsum("rcabp,pb->rcap", t, e[1])
                return jnp.einsum("rcap,pa->pcr", t, e[0]).real
            val = contract(c)
            grads = [contract(c * (2j * jnp.pi * k.reshape(
                [nt if i == d else 1 for i in range(3)]))[None, None]) for d in range(3)]
            return val, jnp.stack(grads)
        vals, grads = jax.lax.map(one, pts.reshape(P // chunk, chunk, 3))
        vals = vals.reshape(P, 3, R)
        grads = jnp.moveaxis(grads, 1, 0).reshape(3, P, 3, R)
        return vals, grads
    vals, grads = run(c, jnp.asarray(pts))
    return np.asarray(vals), np.asarray(grads)


def spectral_content(modes, nt, cuts):
    """Fraction of each mode's energy with max_i |k_i| > cut, and on the Nyquist planes."""
    R = modes.shape[1]
    s = np.abs(np.fft.fftn(np.asarray(modes).T.reshape(R, 3, nt, nt, nt), axes=(-3, -2, -1))) ** 2
    k = np.abs(np.fft.fftfreq(nt) * nt)
    kk = np.maximum(np.maximum(k[:, None, None], k[None, :, None]), k[None, None, :])
    tot = s.sum(axis=(1, 2, 3, 4))
    out = {f"above_{c}": [float(x) for x in s[:, :, kk > c].sum(axis=(1, 2)) / tot] for c in cuts}
    out["nyquist"] = [float(x) for x in s[:, :, kk >= nt // 2].sum(axis=(1, 2)) / tot]
    return out


# ------------------------------------------------------------------- losses ---

def make_objective(kind, arch, nt, R, sample_fn):
    """Returns loss(layers, B, data) -> scalar and parts(layers, B, data) -> dict."""
    if kind == "vp_band":
        lam = float(arch["band_weight"])
        amask = CN.band_mask(nt, int(arch["band_kcut"]))

        def parts(layers, B, data):
            G = sample_fn(dict(B=B, layers=layers), data["pts"])
            GH = G - CN.lowpass(G, nt, amask)
            fit = CN.projection_loss(G, data["Y"])
            band = jnp.sum(GH * GH) / jnp.sum(G * G)
            return dict(total=fit + lam * band, vp_fit=fit, band=band)
    elif kind == "pod_regress":
        def parts(layers, B, data):
            p = dict(B=B, layers=layers)
            G = sample_fn(p, data["pts"])
            w = data["w"]
            dg = G - data["U"]
            grid = jnp.sum(w * jnp.sum(dg * dg, axis=0))
            vq = CN.velocity(p, data["Q"])                            # (P, 3, R)
            dv = vq - data["Fq"]
            val = jnp.sum(w * jnp.sum(dv * dv, axis=(0, 1))) / data["nF"]
            gq = CN.velocity_gradient(p, data["Q"])                   # (3, P, 3, R)
            dd = gq - data["dFq"]
            grad = jnp.sum(w * jnp.sum(dd * dd, axis=(0, 1, 2))) / data["ndF"]
            return dict(total=grid + val + grad, grid=grid, offgrid_value=val, offgrid_gradient=grad)
    else:
        raise ValueError(kind)

    def loss(layers, B, data):
        return parts(layers, B, data)["total"]
    return loss, parts


def column_errors(G, U, pts_vals, Fq):
    """Per-column relative regression errors (grid, off-grid values)."""
    g = np.linalg.norm(G - U, axis=0) / np.linalg.norm(U, axis=0)
    v = np.linalg.norm((pts_vals - Fq).reshape(-1, Fq.shape[-1]), axis=0) / np.linalg.norm(
        Fq.reshape(-1, Fq.shape[-1]), axis=0)
    return [float(x) for x in g], [float(x) for x in v]


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
    kind = cfg["objective"]
    report = dict(schema="ns3d-pod-match-bank-v1", config=cfg, smoke=bool(args.smoke),
                  device=str(jax.devices()), source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: D.sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-coordnet/coordnet.py",
                      "experiments/ns3d-coordnet/bankio.py",
                      "experiments/ns3d-pod-match/train_bank2.py",
                      "experiments/ns3d-shift/shift_rom.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"
    failures = []

    def dump():
        report["elapsed_seconds"] = time.time() - t_job
        D.dump_json(out / "summary.json", report)

    # ------------------------------------------------------------- 1. data ----
    # (unchanged from ns3d-coordnet/train_bank.py, centred branch)
    par = F.parameters(int(cfg["train_seed"]), int(cfg["bank_cases"]))
    meshes = [int(m) for m in cfg["data_meshes"]]
    states, u0sq, case_of, dropped = [], [], [], []
    solvers, centre = {}, {}
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
            g, lost = BI.resample(np.asarray(centre[m](frames[t])), nt)
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
                          train_grid=nt, centred=True,
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
    del Y
    report["compression"] = dict(K=K, tail_fraction=tail,
                                 spectrum=[float(x) for x in (ev[:K] / ev[0])[::8]])
    log(f"compression K={K} tail={tail:.3e}")
    R = int(cfg["rank"])
    optimum = float(ev[R:K].sum() / ev[:K].sum())
    report["eckart_young_optimum"] = optimum
    # the weighted POD of this job's own data (U_R, the optimum of the VP objective):
    # columns of Y are U_K S_K; normalise, then a QR (signs fixed so diag(R) > 0) removes
    # the round-off non-orthogonality without changing any leading prefix's span
    U0 = Yn[:, :R] / np.linalg.norm(Yn[:, :R], axis=0)[None, :]
    pod_gram_dev = float(np.max(np.abs(U0.T @ U0 - np.eye(R))))
    pod_same, rq = np.linalg.qr(U0)
    pod_same = np.ascontiguousarray(pod_same * np.sign(np.diag(rq))[None, :])
    qr_change = float(np.max(np.linalg.norm(pod_same - U0, axis=0)))
    del U0
    report["same_data_pod"] = dict(orthonormality_before_qr=pod_gram_dev, qr_column_change=qr_change)
    if not qr_change <= 1e-6:
        failures.append(f"HARD: same-data POD columns changed by the QR clean-up: {qr_change:.2e}")
    dump()

    # ---------------------------------------------------- 3. the objective ----
    Yj = jnp.asarray(Yn)
    chunk_t = int(cfg.get("sample_chunk", nt ** 3))
    sample_t = CN.make_sampler(nt, chunk=chunk_t)
    sample_fn = CN.make_sample_fn(nt, chunk=chunk_t)
    pts_t = jnp.asarray(CN.grid_points(nt))
    arch = cfg["architecture"]
    data = dict(pts=pts_t, Y=Yj)
    if kind == "pod_regress":
        wr = ev[:R] / ev[:R].sum()
        Qn = int(cfg["offgrid_points"])
        Q = np.random.default_rng(int(cfg["offgrid_seed"])).uniform(size=(Qn, 3))
        Fq, dFq = trig_interpolant(pod_same, nt, Q)
        # target construction checks: the interpolant reproduces the modes at grid nodes,
        # and its gradient equals the FFT derivative there (hard gates)
        idx = np.random.default_rng(3).choice(nt ** 3, size=256, replace=False)
        node = CN.grid_points(nt)[idx]
        Fn, dFn = trig_interpolant(pod_same, nt, node)
        want = pod_same.reshape(3, nt ** 3, R)[:, idx, :].transpose(1, 0, 2) * nt ** 1.5
        geom_t = F.geometry(nt)
        dU = np.asarray(SR.spectral_derivative(jnp.asarray(pod_same.T.reshape(R, 3, nt, nt, nt)), nt, geom_t))
        dU = dU.reshape(3, R, 3, nt ** 3)[:, :, :, idx].transpose(0, 3, 2, 1) * nt ** 1.5
        chk = dict(value_at_nodes=float(np.max(np.abs(Fn - want)) / np.max(np.abs(want))),
                   gradient_at_nodes=float(np.max(np.abs(dFn - dU)) / np.max(np.abs(dU))))
        report["target_checks"] = chk
        log(f"target checks: {chk}")
        if not (chk["value_at_nodes"] <= 1e-10 and chk["gradient_at_nodes"] <= 1e-9):
            failures.append(f"HARD: POD target interpolant check failed {chk}")
        report["target_spectral_content"] = spectral_content(pod_same, nt, (10, 21))
        if max(report["target_spectral_content"]["nyquist"]) > 1e-20:
            failures.append("HARD: POD targets carry Nyquist-plane energy (interpolant ambiguous)")
        report["target_weights"] = [float(x) for x in wr]
        data.update(U=jnp.asarray(pod_same), w=jnp.asarray(wr), Q=jnp.asarray(Q),
                    Fq=jnp.asarray(Fq), dFq=jnp.asarray(dFq),
                    nF=jnp.asarray(float(np.sum(wr * np.sum(Fq * Fq, axis=(0, 1))))),
                    ndF=jnp.asarray(float(np.sum(wr * np.sum(dFq * dFq, axis=(0, 1, 2))))))
        del Fn, dFn, dU
    else:
        report["target_spectral_content"] = spectral_content(pod_same, nt, (10, 21))
    hard = [f for f in failures if f.startswith("HARD")]
    if hard:
        report["failures"] = failures
        report["status"] = "failed-gates"
        dump()
        sys.exit(3)
    loss_fn, parts_fn = make_objective(kind, arch, nt, R, sample_fn)
    parts_j = jax.jit(parts_fn)
    loss_j = jax.jit(loss_fn)
    dump()

    key = jax.random.PRNGKey(int(cfg["seed"]))
    p = CN.init_params(key, R, int(arch["width"]), int(arch["depth"]), int(arch["kmax"]))
    G0 = sample_t(p)
    s = float(jnp.sqrt(R / jnp.sum(G0 * G0)))
    w_, b_ = p["layers"][-1]
    p["layers"][-1] = (w_ * s, b_)
    Bf = p["B"]
    layers = p["layers"]

    def certify(layers, label):
        """Objective parts, exact (QR) VP objective vs optimum, regression columns."""
        pp = dict(B=Bf, layers=layers)
        pr = {k_: float(v) for k_, v in parts_j(layers, Bf, data).items()}
        G_end = np.asarray(sample_t(pp))
        exact_raw, sv_raw = CN.projection_loss_exact(G_end, Yn)
        exact_proj, _ = CN.projection_loss_exact(CN.leray_mask(G_end, nt)[0], Yn)
        amask = CN.band_mask(nt, 10)
        GH = G_end - np.asarray(CN.lowpass(jnp.asarray(G_end), nt, amask))
        rec = dict(stage=label, objective=pr, vp_exact_raw=exact_raw, vp_exact_projected=exact_proj,
                   eckart_young_optimum=optimum, vp_raw_over_optimum=exact_raw / optimum,
                   band_fraction_above_10=float(np.sum(GH * GH) / np.sum(G_end * G_end)),
                   bank_singular_value_ratio=float(sv_raw[-1] / sv_raw[0]))
        if kind == "pod_regress":
            vq = np.asarray(jax.jit(CN.velocity)(pp, data["Q"]))
            rec["column_rel_error_grid"], rec["column_rel_error_offgrid"] = column_errors(
                G_end, pod_same, vq, Fq)
        log(f"[{label}] {pr}; VP exact raw {exact_raw:.4e} (optimum {optimum:.4e}, ratio {exact_raw / optimum:.2f})")
        return rec

    # --------------------------------------------------------- 4. Adam stage ----
    stage_records = []
    every = int(cfg["log_every"])
    ad = cfg["adam"]
    checkpoint_every = float(cfg.get("checkpoint_seconds", 900))

    def make_adam_chunk(opt):
        @jax.jit
        def chunk(layers, state, B, data):
            def body(carry, _):
                q, st = carry
                v, g = jax.value_and_grad(loss_fn)(q, B, data)
                upd, st = opt.update(g, st, q)
                return (optax.apply_updates(q, upd), st), v
            (q, st), vals = jax.lax.scan(body, (layers, state), None, length=every)
            return q, st, vals[-1]
        return chunk

    copt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
    cchunk = make_adam_chunk(copt)
    cq, cs, _ = cchunk(layers, copt.init(layers), Bf, data)          # compile
    jax.block_until_ready(cq)
    t_c = time.time()
    cq, cs, _ = cchunk(cq, cs, Bf, data)
    jax.block_until_ready(cq)
    per_step = (time.time() - t_c) / every
    steps = int(0.97 * float(ad["train_seconds"]) / per_step) // every * every
    if steps < 4 * int(ad["warmup"]):
        raise RuntimeError(f"budget too small: {steps} steps < 4 x warm-up")
    del cq, cs, cchunk
    log(f"[adam] calibration: {per_step * 1e3:.1f} ms/step -> {steps} steps")
    sched = optax.warmup_cosine_decay_schedule(0.0, float(ad["lr"]), int(ad["warmup"]), steps,
                                               float(ad["lr"]) * float(ad["final_lr_ratio"]))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    state = opt.init(layers)
    chunk = make_adam_chunk(opt)
    curve = []
    t0 = time.time()
    t_ck = t0
    done = 0
    while done < steps:
        layers, state, v = chunk(layers, state, Bf, data)
        done += every
        curve.append((done, float(v), time.time() - t0))
        log(f"  [adam] step {done}/{steps} loss {float(v):.6e} ({time.time() - t0:.0f}s)")
        if not np.isfinite(float(v)):
            raise RuntimeError("nonfinite Adam training")
        if time.time() - t_ck > checkpoint_every:
            BI.save_params(out / "bank_adam_partial.npz", dict(B=Bf, layers=layers))
            t_ck = time.time()
    BI.save_params(out / "bank_adam.npz", dict(B=Bf, layers=layers))
    (out / "bank_adam_partial.npz").unlink(missing_ok=True)
    rec = certify(layers, "adam")
    rec.update(steps=steps, seconds_per_step=per_step, seconds=time.time() - t0, curve=curve)
    stage_records.append(rec)
    report["stages"] = stage_records
    dump()

    # ------------------------------------------------------- 5. L-BFGS stage ----
    lb = cfg["lbfgs"]
    lopt = optax.lbfgs(memory_size=int(lb["memory"]),
                       linesearch=optax.scale_by_zoom_linesearch(
                           max_linesearch_steps=int(lb["max_linesearch_steps"])))
    per_call = int(lb["iterations_per_call"])

    @jax.jit
    def lchunk(layers, state, B, data):
        f = lambda q: loss_fn(q, B, data)
        vg = optax.value_and_grad_from_state(f)

        def body(carry, _):
            q, st = carry
            v, g = vg(q, state=st)
            upd, st = lopt.update(g, st, q, value=v, grad=g, value_fn=f)
            q = optax.apply_updates(q, upd)
            ls = optax.tree_utils.tree_get(st, "info")
            return (q, st), (v, ls.num_linesearch_steps)
        (q, st), (vals, nls) = jax.lax.scan(body, (layers, state), None, length=per_call)
        return q, st, vals, nls

    lstate = lopt.init(layers)
    t0 = time.time()
    t_ck = t0
    it = 0
    evals = 0
    lcurve = []
    stop = "deadline"
    prev = float(loss_j(layers, Bf, data))
    start_value = prev
    stall = 0
    while time.time() - t0 < float(lb["train_seconds"]):
        q_new, s_new, vals, nls = lchunk(layers, lstate, Bf, data)
        cur = float(loss_j(q_new, Bf, data))
        if not np.isfinite(cur) or not np.all(np.isfinite(np.asarray(vals))):
            stop = "nonfinite (reverted to the last finite iterate)"
            break
        if cur > prev:
            stop = f"objective increased over a call ({prev:.6e} -> {cur:.6e}); reverted"
            break
        layers, lstate = q_new, s_new
        it += per_call
        evals += int(np.sum(np.asarray(nls)))
        lcurve.append((it, cur, time.time() - t0, evals))
        log(f"  [lbfgs] iter {it} loss {cur:.6e} line-search evals {evals} ({time.time() - t0:.0f}s)")
        stall = stall + 1 if (prev - cur) <= float(lb["stall_rtol"]) * abs(prev) else 0
        prev = cur
        if stall >= int(lb["stall_calls"]):
            stop = "stalled"
            break
        if time.time() - t_ck > checkpoint_every:
            BI.save_params(out / "bank_lbfgs_partial.npz", dict(B=Bf, layers=layers))
            t_ck = time.time()
    BI.save_params(out / "bank_final.npz", dict(B=Bf, layers=layers))
    (out / "bank_lbfgs_partial.npz").unlink(missing_ok=True)
    rec = certify(layers, "lbfgs")
    rec.update(iterations=it, linesearch_evaluations=evals, stop=stop, start_value=start_value,
               seconds=time.time() - t0, curve=lcurve)
    stage_records.append(rec)
    report["stages"] = stage_records
    report["final_bank"] = "bank_final.npz (end of the L-BFGS stage; pre-registered)"
    if not np.max(np.abs(np.asarray(Bf) - CN.frequency_set(int(arch["kmax"])))) == 0.0:
        raise RuntimeError("Fourier frequencies changed")
    dump()

    # ------------------------------------------------------------ 6. order ----
    banks = {}
    for label, fname in (("final", "bank_final.npz"), ("adam", "bank_adam.npz")):
        pb = BI.load_params(out / fname)
        Gt_p, removed_t = CN.leray_mask(np.asarray(sample_t(pb)), nt)
        T, svals, _ = CN.order_bank(Gt_p, X, norms)
        banks[label] = (pb, T)
        np.savez(out / f"bank_{label}_ordered.npz", T=T, singular_values=svals,
                 **{k_: v for k_, v in np.load(out / fname).items()})
        report.setdefault("ordering", {})[label] = dict(
            grid=nt, singular_values=[float(x) for x in svals],
            leray_removed_worst=float(np.max(removed_t)))
    del X
    gc.collect()
    dump()

    # ----------------------------------------------------------- 7. floors ----
    dev_par = F.parameters(int(cfg["dev_seed"]), int(cfg["dev_cases"]))
    target = float(cfg["target_relative"])
    floors = {}
    ref_bank = None
    for n in sorted(int(m) for m in cfg["floor_meshes"]):
        log(f"floors at {n}^3")
        dev, _, _ = D.generate(dev_par, n, float(cfg["dt_truth"]), float(cfg["horizon"]))
        entry = {}
        for label in ("final", "adam"):
            pb, T = banks[label]
            # the bank job measures, it does not abort: the pre-Loewdin deviation is gate G3
            if label == "adam":    # diagnostic only: R floor of the Adam-stage bank
                try:
                    Gn, info = BI.mesh_bank(pb, T, n, chunk=int(cfg.get("sample_chunk_eval", 32768)),
                                            abort=float("inf"))
                    info.pop("lowdin_factor")
                    errors, _, _ = D.oracle_shift_errors(Gn, dev)
                    entry["adam_stage"] = dict(bank=info, **{f"coordnet_R{R}": D.stats_from_cases(errors, target)})
                except RuntimeError as exc:
                    entry["adam_stage"] = dict(error=str(exc))
                del Gn
                continue
            Gn, info = BI.mesh_bank(pb, T, n, chunk=int(cfg.get("sample_chunk_eval", 32768)),
                                    abort=float("inf"))
            info.pop("lowdin_factor")
            entry["bank"] = info
            for Rp in sorted({int(x) for x in cfg["prefixes"] if int(x) <= R} | {R}):
                errors, _, travel = D.oracle_shift_errors(Gn[:, :Rp], dev)
                entry[f"coordnet_R{Rp}"] = D.stats_from_cases(errors, target)
                log(f"  coordnet R'={Rp}: evolved worst {entry[f'coordnet_R{Rp}']['evolved_worst']:.6f}")
            entry["travel"] = travel
            errors, _ = D.project_errors(Gn, dev)
            entry["coordnet_no_centring"] = D.stats_from_cases(errors, target)
            if not entry["coordnet_no_centring"]["evolved_worst"] > target:
                failures.append(f"HARD: no-centring control did not fire at {n}^3")
            Gfinal = Gn
        Up = np.stack([BI.resample(pod_same[:, j].reshape(3, nt, nt, nt), n)[0].ravel()
                       for j in range(R)], axis=1)
        Up, _ = np.linalg.qr(CN.leray_mask(Up, n)[0])
        errors, _, _ = D.oracle_shift_errors(Up, dev)
        entry[f"same_data_pod_R{R}"] = D.stats_from_cases(errors, target)
        del Up
        tp = F.parameters(int(cfg["train_seed"]), int(cfg["parent_pod_cases"]))
        tr, _, _ = D.generate(tp, n, float(cfg["dt_truth"]), float(cfg["horizon"]))
        cj = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))
        for a in range(len(tr)):
            for b in range(tr.shape[1]):
                tr[a, b] = np.asarray(cj(jnp.asarray(tr[a, b])))
        basis, _, _ = D.pod_basis(tr.reshape(len(tr) * tr.shape[1], -1), 64, int(cfg["gram_block"]))
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
            if not gapp <= 1e-6:
                failures.append(f"HARD: parent POD-64 floor at {n}^3 not reproduced: {gapp:.2e}")
        del basis, Gpod
        log(f"  parent POD-64: evolved worst {entry['parent_pod_R64']['evolved_worst']:.6f}")
        if n in [int(m) for m in cfg.get("derivative_meshes", [])]:
            entry["derivatives"] = BI.derivative_check(banks["final"][0], banks["final"][1], n)
            log(f"  derivative check: {entry['derivatives']}")
        if ref_bank is None:
            ref_bank = (n, Gfinal)
        else:
            n0, G0r = ref_bank
            Gr_ = np.stack([BI.resample(Gfinal[:, j].reshape(3, n, n, n), n0)[0].ravel()
                            for j in range(Gfinal.shape[1])], axis=1) * (n / n0) ** 1.5
            col = np.linalg.norm(Gr_ - G0r, axis=0) / np.linalg.norm(G0r, axis=0)
            entry["cross_mesh_vs_smallest"] = dict(against=int(n0), worst_column=float(col.max()),
                                                   median_column=float(np.median(col)))
            log(f"  cross-mesh consistency vs {n0}^3: worst column {col.max():.3e}")
        floors[str(n)] = entry
        del dev, Gfinal
        gc.collect()
        report["floors"] = floors
        dump()
    pts = jnp.asarray(np.random.default_rng(5).uniform(size=(256, 3)))
    pf = banks["final"][0]
    report["autodiff_divergence_relative"] = float(
        jnp.max(jnp.abs(CN.divergence(pf, pts))) / jnp.max(jnp.abs(CN.velocity(pf, pts))))

    # ------------------------------------------------------- 8. the gates ----
    # (pre-registered in DESIGN.md section 5; computed here and recomputed by the report)
    gates = {}
    ratios = {m: floors[m][f"coordnet_R{R}"]["evolved_worst"] / floors[m]["parent_pod_R64"]["evolved_worst"]
              for m in floors}
    fl = [floors[m][f"coordnet_R{R}"]["evolved_worst"] for m in floors]
    orth = {m: floors[m]["bank"]["orthonormality_before_lowdin"] for m in floors}
    gates["G1_floor_vs_pod"] = dict(ratio=ratios, bound=1.5, passed=bool(all(r <= 1.5 for r in ratios.values())))
    gates["G2_cross_mesh_floor"] = dict(max_over_min=float(max(fl) / min(fl)), bound=1.25,
                                        passed=bool(max(fl) / min(fl) <= 1.25))
    gated = {m: v for m, v in orth.items() if int(m) >= 64}
    gates["G3_orthonormality_64_96"] = dict(value=gated, bound=1e-3, reported_32=orth.get("32"),
                                            passed=bool(gated and all(v <= 1e-3 for v in gated.values())))
    gates["all_passed"] = bool(all(g["passed"] for g in gates.values() if isinstance(g, dict)))
    report["gates"] = gates
    report["failures"] = failures
    hard = [f for f in failures if f.startswith("HARD")]
    report["status"] = "failed-gates" if hard else "final"
    dump()
    log(f"gates: {gates}")
    log(f"done; failures: {failures}")
    if hard:
        sys.exit(3)


if __name__ == "__main__":
    main()
