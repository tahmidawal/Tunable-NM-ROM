"""Confirmation driver (rebuttal push): ns3d-coordnet/rom_mesh.py with the coordinate head
ALWAYS loaded frozen, plus (PROTOCOL.md)
  * the interpolated-POD control (the 64^3 POD bank Fourier-resampled to this mesh, its
    floors, span rollouts and the 64^3 POD head zero-shot),
  * the coarse-mesh solver control (truncate -> CNAB2/RK4 on mesh m < n -> zero-pad),
    scored against the same truth and timed in the same interleaved block as the ROM arms,
  * small field slices for figures.
The parent driver is not modified; sections 1-6 keep its logic and summary layout so the
independent audit (verify_coordnet.py) still applies to the main arms.
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
              "experiments/ns3d-grok", "experiments/ns3d-shift", "experiments/ns3d-shift-head"):
    sys.path.insert(0, str(ROOT / extra))
sys.path.insert(0, str(ROOT / "experiments/ns3d-coordnet"))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import ns3d_fom as F  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402
import head_rom as H  # noqa: E402
import coordnet as CN  # noqa: E402
import bankio as BI  # noqa: E402
from head_ladder import run_cases, disjointness, burn_gpu, time_block, rel_case  # noqa: E402

log = D.log
UNSTABLE = 1.0


def load_head(path):
    ck = np.load(path)
    p = {key: jnp.asarray(ck[f"p_{key}"]) for key in ("W1", "b1", "W2", "b2", "W3", "b3", "Ws")}
    return p, np.asarray(ck["Z"])


def save_head(path, p, Z):
    np.savez(path, Z=Z, C=np.zeros((p["W3"].shape[1], 0)),
             **{f"p_{key}": np.asarray(v) for key, v in p.items()})


def scale_head(p, factor):
    """h -> factor * h (the head is linear in W3, b3, Ws). Bank coefficients of the same
    physical field scale as n^{3/2} with the mesh (unit discrete-l2 columns), so a head
    trained at mesh n0 is used at mesh n with factor (n / n0)^{3/2}."""
    return {key: (val * factor if key in ("W3", "b3", "Ws") else val) for key, val in p.items()}


def spec_resample(u, n_to):
    """Fourier resampling of (..., n, n, n) real fields to (..., n_to, n_to, n_to):
    coefficients with |k_i| <= min(n, n_to)/2 - 1 are copied, the rest are zero."""
    n = u.shape[-1]
    if n == n_to:
        return u
    m = min(n, n_to) // 2
    src_i = np.r_[0:m, n - m + 1:n]
    dst_i = np.r_[0:m, n_to - m + 1:n_to]
    spec = jnp.fft.fftn(u, axes=(-3, -2, -1), norm="forward")
    kept = spec[..., src_i, :, :][..., :, src_i, :][..., :, :, src_i]
    out = jnp.zeros(u.shape[:-3] + (n_to, n_to, n_to), dtype=spec.dtype)
    out = out.at[..., dst_i[:, None, None], dst_i[None, :, None], dst_i[None, None, :]].set(kept)
    return jnp.fft.ifftn(out, axes=(-3, -2, -1), norm="forward").real


def build_pod(cfg, nm, pod_dir, horizon):
    """The paper's rank-64 centred POD at mesh nm (parent recipe), sign-fixed and gated
    against the stored probe of that mesh."""
    train_par = F.parameters(int(cfg["train_seed"]), int(cfg["train_cases"]))
    train, _, _ = D.generate(train_par, nm, float(cfg["dt_truth"]), horizon)
    centre_jit = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * nm))
    for c in range(len(train)):
        for t in range(train.shape[1]):
            train[c, t] = np.asarray(centre_jit(jnp.asarray(train[c, t])))
    basis_all, _, _ = D.pod_basis(train.reshape(len(train) * train.shape[1], -1), 64,
                                  int(cfg["gram_block"]))
    del train
    gc.collect()
    Gpod, _ = D.orthonormalize_prefix(basis_all, 64)
    del basis_all
    ref = np.load(pod_dir / "bank_probe.npz")
    probe, _, _ = D.generate(F.parameters(int(cfg["train_seed"]), int(ref["cases"])), nm,
                             float(cfg["dt_truth"]), horizon)
    a_new = np.stack([Gpod.T @ np.asarray(centre_jit(jnp.asarray(probe[c, t]))).ravel()
                      for c in range(len(probe)) for t in range(probe.shape[1])])
    signs = np.sign(np.sum(a_new * ref["coefficients"], axis=0))
    Gpod = np.ascontiguousarray(Gpod * signs[None, :])
    gap = float(np.linalg.norm(a_new * signs[None, :] - ref["coefficients"])
                / np.linalg.norm(ref["coefficients"]))
    if gap > 1e-8:
        raise RuntimeError(f"rebuilt POD bank at {nm}^3 differs from the paper's: {gap}")
    return Gpod, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--frozen", type=Path, required=True,
                    help="directory with frozen_settings.json and the coordnet heads")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get("smoke", {}))
    test = "disjoint_against" in cfg
    keep = bool(cfg.get("save_fields", True))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu" and not args.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["train_seed"]), int(cfg["eval_seed"])):
            raise RuntimeError("refusing to read a closed seed")
    t_job = time.time()
    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    modes = int(cfg["modes"])
    target = float(cfg["target_relative"])
    k = int(cfg["k"])
    report = dict(schema="ns3d-confirm-v1", config=cfg, smoke=bool(args.smoke), test=test,
                  device=str(jax.devices()), source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  files={name: D.sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-coordnet/coordnet.py",
                      "experiments/ns3d-coordnet/bankio.py",
                      "experiments/ns3d-coordnet/rom_mesh.py",
                      "experiments/ns3d-pod-match/confirm_mesh.py",
                      "experiments/ns3d/ns3d_fom.py",
                      "experiments/ns3d-shift/shift_rom.py",
                      "experiments/ns3d-shift-head/head_rom.py",
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

    # frozen inputs, each checked by sha256 before any work
    bank_file = ROOT / cfg["coordnet_bank_file"]
    has_pod = cfg.get("pod_frozen_dir") is not None
    pod_dir = ROOT / cfg["pod_frozen_dir"] if has_pod else None
    checks = {str(bank_file): cfg["coordnet_bank_sha256"]}
    if has_pod:
        for name, sha in cfg["pod_frozen_sha256"].items():
            checks[str(pod_dir / name)] = sha
    ip = cfg.get("interp_pod") if int(cfg["n"]) != int((cfg.get("interp_pod") or {}).get("source_n", -1)) else None
    if ip:
        ip_dir = ROOT / ip["dir"]
        for name, sha in ip["sha256"].items():
            checks[str(ip_dir / name)] = sha
    frozen = json.loads((args.frozen / "frozen_settings.json").read_text())
    report["frozen_settings_sha256"] = D.sha256_file(args.frozen / "frozen_settings.json")
    checks[str(args.frozen / frozen["heads"][str(n)]["file"])] = frozen["heads"][str(n)]["sha256"]
    for path, sha in checks.items():
        got = D.sha256_file(Path(path))
        if got != sha:
            raise RuntimeError(f"frozen input {path} sha256 {got} != {sha}")
    report["frozen_inputs_checked"] = checks
    failures = []
    for key in ("coordnet_bank_sha256", "bank_prefix", "k", "ladder_dt", "ladder_iters",
                "damping", "ic_iters", "modes"):
        if frozen.get(key) != cfg.get(key):
            raise RuntimeError(f"config {key}={cfg.get(key)} differs from frozen {frozen.get(key)}")
    if test:
        others = [(cfg["train_seed"], cfg["head_train_cases"])] + [tuple(x) for x in cfg["disjoint_against"]]
        overlap = disjointness(cfg["eval_seed"], cfg["eval_cases"], others)
        report["test_disjointness"] = overlap
        log(f"test disjointness (rounded-row overlaps): {overlap}")
        if any(overlap.values()):
            raise RuntimeError(f"test cohort overlaps another cohort: {overlap}")

    # ------------------------------------------------------------ 1. banks ----
    Gpod = None
    if has_pod:
        log("POD reference bank: parent recipe")
        Gpod, gap = build_pod(cfg, n, pod_dir, horizon)
        report["pod_bank_rebuild_gap"] = gap
        log(f"POD bank rebuild vs the paper's probe: {gap:.3e}")
    Gint = None
    if ip:
        ns_ = int(ip["source_n"])
        log(f"interpolated-POD control: POD at {ns_}^3, resampled to {n}^3")
        Gsrc, gap = build_pod(cfg, ns_, ip_dir, horizon)
        cols = np.stack([np.asarray(spec_resample(jnp.asarray(Gsrc[:, j].reshape(3, ns_, ns_, ns_)), n)).ravel()
                         for j in range(Gsrc.shape[1])], axis=1) * (ns_ / n) ** 1.5
        del Gsrc
        proj, removed = CN.leray_mask(cols, n)
        M = proj.T @ proj - np.eye(proj.shape[1])
        Gint, _ = BI.lowdin(proj)
        Gint = np.ascontiguousarray(Gint)
        report["interp_pod_bank"] = dict(
            source_n=ns_, source_rebuild_gap=gap,
            resample_plus_mask_removed_worst=float(np.max(np.sum((cols - proj) ** 2, 0) / np.sum(cols ** 2, 0))),
            column_norm_after_resample_min=float(np.min(np.linalg.norm(cols, axis=0))),
            orthonormality_before_lowdin=float(np.max(np.abs(M))),
            lowdin_relative_change=float(np.linalg.norm(Gint - proj) / np.linalg.norm(proj)),
            orthonormality_after=float(np.max(np.abs(Gint.T @ Gint - np.eye(Gint.shape[1])))))
        log(f"interpolated POD bank: {report['interp_pod_bank']}")
        del cols, proj
        gc.collect()

    bank = np.load(bank_file)
    pc = BI.load_params(bank_file)
    Tc = np.asarray(bank["T"])
    Gc, binfo = BI.mesh_bank(pc, Tc, n, chunk=int(cfg.get("sample_chunk_eval", 32768)),
                              abort=float(cfg["one_bank_orthonormality_gate"]))
    lowdin_S = binfo.pop("lowdin_factor")
    prefix = cfg.get("bank_prefix")
    if prefix is not None:
        if not (isinstance(prefix, int) and 1 <= prefix <= Gc.shape[1]):
            raise RuntimeError(f"invalid bank_prefix {prefix} for a rank-{Gc.shape[1]} bank")
        # the full bank's normalisation, THEN the ordered prefix (as its floor was measured)
        Gc = np.ascontiguousarray(Gc[:, :int(prefix)])
        lowdin_S = lowdin_S[:, :int(prefix)]
    binfo["prefix"] = prefix
    Rc = Gc.shape[1]
    report["coordnet_bank"] = binfo
    log(f"coordnet bank at {n}^3: {binfo}")
    if binfo["orthonormality_before_lowdin"] > float(cfg["one_bank_orthonormality_gate"]):
        raise RuntimeError("the sampled bank is too far from orthonormal at this mesh: "
                           f"{binfo['orthonormality_before_lowdin']}")
    if binfo["orthonormality_before_lowdin"] > 1e-3:
        failures.append(f"flag (bar c): orthonormality before Lowdin {binfo['orthonormality_before_lowdin']:.2e}")

    ev_par = F.parameters(int(cfg["eval_seed"]), int(cfg["eval_cases"]))
    report["eval_parameter_sha256"] = D.sha256_array(ev_par)
    log("generating evaluation trajectories")
    dev, _, _ = D.generate(ev_par, n, float(cfg["dt_truth"]), horizon)
    D.require_finite(dev, "evaluation")
    viscosities = ev_par[:, -1]
    if keep:
        np.save(out / "dev_truth.npy", dev)

    banks = {"coordnet": Gc}
    if has_pod:
        banks["pod"] = Gpod
    if ip:
        banks["interp"] = Gint
    ops_all, opsj = {}, {}
    report["operator_checks"] = {}
    for label, G_ in banks.items():
        ops_, rep_ = SR.build_operators_fast(G_, n, modes, check_modes=int(cfg["check_modes"]))
        for key, val in rep_.items():
            if isinstance(val, float) and not np.isfinite(val):
                raise RuntimeError(f"{label} operator check {key} is not finite")
        report["operator_checks"][label] = rep_
        ops_all[label] = ops_
        opsj[label] = (jnp.asarray(G_), jnp.asarray(ops_["A"]), jnp.asarray(ops_["T"]),
                       jnp.asarray(ops_["lam"]), jnp.asarray(ops_["Dd"]))
    ops_c = ops_all["coordnet"]
    if keep:
        np.save(out / "bank_G.npy", Gc)      # for the independent floor audit; deleted after it
    # D_d of the coordnet bank: spectral (used) vs autodiff of the network (independent)
    ids, _ = SR.test_mode_ids(n, modes)
    subset = np.sort(np.random.default_rng(13).choice(modes, size=8, replace=False))
    Phi = SR.dense_test_modes(n, [ids[i] for i in subset])
    D_auto = np.einsum("emr,rs->ems", BI.autodiff_tested(pc, n, Phi), Tc @ lowdin_S)
    D_used = ops_c["Dd"][:, subset]
    report["coordnet_Dd_autodiff_vs_spectral"] = float(
        np.linalg.norm(D_auto - D_used) / np.linalg.norm(D_used))
    log(f"coordnet D_d autodiff vs spectral: {report['coordnet_Dd_autodiff_vs_spectral']:.3e}")
    if not report["coordnet_Dd_autodiff_vs_spectral"] <= 1e-4:
        failures.append(f"flag: D_d autodiff vs spectral {report['coordnet_Dd_autodiff_vs_spectral']:.2e} > 1e-4")
    opsj_c = opsj["coordnet"]
    dump()

    # ----------------------------------------------------------- 2. floors ----
    floors = {}
    travel = None
    if has_pod:
        errors, _, travel = D.oracle_shift_errors(Gpod, dev)
        floors["pod_R64"] = D.stats_from_cases(errors, target)
    ispans = []
    if ip:
        ispans = [int(x) for x in ip["spans"]]
        for Rp in sorted(set(ispans) | {64}):
            errors, _, _ = D.oracle_shift_errors(Gint[:, :Rp], dev)
            floors[f"interp_R{Rp}"] = D.stats_from_cases(errors, target)
    spans = [int(x) for x in frozen["span_ladder"] if int(x) <= Rc]
    for Rp in sorted(set(spans) | {Rc}):
        errors, _, travel_c = D.oracle_shift_errors(Gc[:, :Rp], dev)
        travel = travel_c if travel is None else travel
        floors[f"coordnet_R{Rp}"] = D.stats_from_cases(errors, target)
    report["floors"] = floors
    report["travel"] = travel
    log("floors: " + ", ".join(f"{a}={b['evolved_worst']:.6f}" for a, b in floors.items()))
    dump()

    @jax.jit
    def project_frames(frames, basis):
        def one(f):
            v = SR.shift_field(f, -SR.grid_centroid(f) * n).ravel()
            a = basis.T @ v
            return a, jnp.sum(v * v) - jnp.sum(a * a)
        return jax.vmap(one)(frames)

    def dev_coeffs(Gj):
        pieces = [project_frames(jnp.asarray(dev[c]), Gj) for c in range(len(dev))]
        a = np.stack([np.asarray(x) for x, _ in pieces])
        o = np.maximum(np.stack([np.asarray(y) for _, y in pieces]), 0.0)
        return a, o
    dev_u0sq = np.sum(dev[:, 0].reshape(len(dev), -1) ** 2, axis=1)

    # ------------------------------------------------------------- 3. heads ----
    p_c, Z_c = load_head(args.frozen / frozen["heads"][str(n)]["file"])
    trained_n = int(frozen["heads"][str(n)].get("trained_n", n))
    if trained_n != n:
        p_c = scale_head(p_c, (n / trained_n) ** 1.5)
        log(f"coordnet head trained at {trained_n}^3 used zero-shot, output scale {(n / trained_n) ** 1.5:.6f}")
    report["coordnet_head"] = dict(note="frozen coordnet head loaded",
                                   file=frozen["heads"][str(n)]["file"],
                                   source=frozen["heads"][str(n)].get("source_run"))
    head_specs = [("coordnet", p_c, Z_c, Rc)]
    if has_pod:
        p_pod, Z_pod = load_head(pod_dir / "head_k8.npz")
        head_specs.append(("pod", p_pod, Z_pod, 64))
    if ip:
        p_i, Z_i = load_head(ip_dir / "head_k8.npz")
        p_i = scale_head(p_i, (n / int(ip["source_n"])) ** 1.5)
        head_specs.append(("interp", p_i, Z_i, 64))

    heads = {}
    for label, p_, Z_, R_ in head_specs:
        Gj = opsj[label][0]
        Zj = jnp.asarray(Z_)
        Hc = H.head_apply(p_, Zj)
        heads[label] = (p_, Zj, Hc, jnp.sum(Hc * Hc, 1), R_, int(Z_.shape[1]))
        a_, o_ = dev_coeffs(Gj)
        err = H.head_floor_errors(p_, Zj, Hc, a_, o_, dev_u0sq, jnp.zeros((R_, 0)),
                                  int(cfg["ic_iters"]))
        floors[f"{label}_head_k{heads[label][5]}"] = D.stats_from_cases(err, target)
    report["floors"] = floors
    log("head floors: " + ", ".join(f"{a}={b['evolved_worst']:.6f}" for a, b in floors.items() if "head" in a))
    dump()

    # ---------------------------------------------------------- 4. rollouts ----
    dtv, iters = float(cfg["ladder_dt"]), int(cfg["ladder_iters"])
    steps = D.nsteps_for(dtv, horizon)
    damping = float(cfg["damping"])
    runners = {}

    def head_fn(label, frame="free"):
        p_, Zj, Hc, Hn, R_, k_ = heads[label]
        key = ("head", R_, k_, frame)
        if key not in runners:
            runners[key] = H.make_head_run(dtv, steps, steps // 5, n, R_, k_, 0, iters=iters,
                                           damping=damping, ic_iters=int(cfg["ic_iters"]),
                                           frame=frame)
        run = runners[key]
        ops = opsj[label]
        C0 = jnp.zeros((R_, 0))
        return lambda u, nu: run(u, nu, *ops, p_, C0, Hc, Hn, Zj)

    span_ops = {}
    for Rp in spans:
        span_ops[Rp] = (jnp.asarray(np.ascontiguousarray(Gc[:, :Rp])),
                        jnp.asarray(np.ascontiguousarray(ops_c["A"][:, :Rp])),
                        jnp.asarray(np.ascontiguousarray(ops_c["T"][:, :Rp, :Rp])),
                        jnp.asarray(ops_c["lam"]),
                        jnp.asarray(np.ascontiguousarray(ops_c["Dd"][:, :, :Rp])))

    def lin_fn(rank, ops):
        key = ("lin", rank)
        if key not in runners:
            runners[key] = SR.make_frozen_run(dtv, steps, steps // 5, n, rank, iters=iters,
                                              damping=damping, extrapolate=True, diagnose=False)
        run = runners[key]
        return lambda u, nu: run(u, nu, *ops)

    arms = {f"coordnet_head_k{k}": head_fn("coordnet")}
    if has_pod:
        arms[f"pod_head_k{heads['pod'][5]}"] = head_fn("pod")
        arms["pod_linear_R64"] = lin_fn(64, opsj["pod"])
    for Rp in spans:
        arms[f"coordnet_span{Rp}"] = lin_fn(Rp, span_ops[Rp])
    if ip:
        arms[f"interp_head_k{heads['interp'][5]}"] = head_fn("interp")
        oi = ops_all["interp"]
        for Rp in ispans:
            arms[f"interp_span{Rp}"] = lin_fn(Rp, (
                jnp.asarray(np.ascontiguousarray(Gint[:, :Rp])),
                jnp.asarray(np.ascontiguousarray(oi["A"][:, :Rp])),
                jnp.asarray(np.ascontiguousarray(oi["T"][:, :Rp, :Rp])),
                jnp.asarray(oi["lam"]),
                jnp.asarray(np.ascontiguousarray(oi["Dd"][:, :, :Rp]))))
    rows = {}
    for name, fn in arms.items():
        try:
            errors, fields = run_cases(fn, dev, viscosities, keep=keep)
            stats, finite = D.stats_from_cases(errors, target), True
            if keep:
                np.save(out / f"fields_{name}.npy", fields)
            del fields
        except RuntimeError as exc:
            log(f"{name}: {exc}")
            errors, stats, finite = None, None, False
        rows[name] = dict(finite=finite, stats=stats,
                          errors=None if errors is None else errors.tolist())
        log(f"{name}: evolved worst {stats['evolved_worst'] if stats else float('nan'):.6f}")
        report["rollouts"] = rows
        dump()

    # reproduction gate: the POD reference arms against the paper's job (development only)
    if cfg.get("reference_errors"):
        repro = {}
        for name, ref_errors in cfg["reference_errors"].items():
            if name not in rows:
                continue
            got = np.asarray(rows[name]["errors"])
            repro[name] = float(np.max(np.abs(got - np.asarray(ref_errors))))
        report["pod_reproduction"] = dict(max_abs_error_difference=repro,
                                          source=cfg["reference_source"],
                                          passed=bool(max(repro.values()) <= 1e-8))
        if not report["pod_reproduction"]["passed"]:
            failures.append(f"POD reproduction failed: {repro}")
        log(f"POD reproduction vs the paper's job: {repro}")

    if cfg.get("controls", True):
        # controls: the coordnet head with the frame frozen must fail; driver parity vs LM
        zero_err, zero_fields = run_cases(head_fn("coordnet", frame="zero"), dev, viscosities, keep=True)
        np.save(out / "fields_control_frame_zero.npy", zero_fields)
        del zero_fields
        report["control_frame_zero"] = D.stats_from_cases(zero_err, target)
        report["control_frame_zero_errors"] = zero_err.tolist()
        if not report["control_frame_zero"]["evolved_worst"] > target:
            failures.append("control did not fire: frame-frozen coordnet head is within 5 %")
        log(f"control frame-zero (coordnet head): {report['control_frame_zero']['evolved_worst']:.4f}")
        p_, Zj, Hc, Hn, R_, _ = heads["coordnet"]
        refrun = H.make_head_reference(dtv, steps, steps // 5, n, R_, k, 0,
                                       budget=int(cfg["lm_budget"]), gtol=float(cfg["lm_gtol"]),
                                       ic_iters=int(cfg["ic_iters"]))
        fn = head_fn("coordnet")
        gaps = []
        for case in range(min(4, len(dev))):
            f_ref, _ = refrun(jnp.asarray(dev[case, 0]), float(viscosities[case]), *opsj_c, p_,
                              jnp.zeros((R_, 0)), Hc, Hn, Zj)
            got = fn(jnp.asarray(dev[case, 0]), float(viscosities[case]))
            gaps.append(float(jnp.linalg.norm(got - f_ref) / jnp.linalg.norm(f_ref)))
        report["coordnet_head_parity_vs_lm"] = dict(cases=len(gaps), worst=max(gaps), gaps=gaps)
        if not (np.all(np.isfinite(gaps)) and max(gaps) <= 1e-6):
            failures.append(f"coordnet head driver parity vs LM {max(gaps):.2e} > 1e-6")
        log(f"coordnet head parity vs LM: {max(gaps):.3e}")
        dump()

    # ------------------------------------------------------------- 5. CNAB2 ----
    geom = F.geometry(n)
    cnab = {}
    for st in cfg["cnab2_steps"]:
        st = int(st)
        solver = F.make_solver(horizon / st, st, st // 5)
        try:
            errors, fields = run_cases(lambda u, nu: solver(u, nu, geom), dev, viscosities, keep=keep)
            stats = D.stats_from_cases(errors, target)
            unstable = bool(stats["evolved_worst"] > UNSTABLE)
            if keep:
                np.save(out / f"fields_cnab2_s{st}.npy", fields.astype(np.float32))
            del fields
        except RuntimeError as exc:
            log(f"CNAB2 steps={st}: {exc}")
            errors, stats, unstable = None, None, True
        cnab[str(st)] = dict(steps=st, dt=horizon / st, stats=stats, unstable=unstable,
                             errors=None if errors is None else errors.tolist())
        log(f"CNAB2 steps={st}: {stats['evolved_worst'] if stats else float('nan'):.6f}")
    report["cnab2"] = cnab
    dump()

    # ------------------------------------------ 5b. coarse-mesh solver control ----
    coarse, coarse_fn, coarse_solve = {}, {}, {}
    cc = cfg.get("coarse") or {}
    for m in [int(x) for x in cc.get("meshes", []) if int(x) < n]:
        geom_m = F.geometry(m)
        for scheme, ladder in (("cnab2", cc["cnab2_steps"]), ("rk4", cc["rk4_steps"])):
            for st in [int(x) for x in ladder]:
                name = f"m{m}_{scheme}_s{st}"
                solver_m = F.make_solver(horizon / st, st, st // 5, scheme=scheme)
                query = jax.jit(lambda u, nu, s_=solver_m, g_=geom_m, m_=m:
                                spec_resample(s_(spec_resample(u, m_), nu, g_), n))
                try:
                    errors, _ = run_cases(query, dev, viscosities, keep=False)
                    stats = D.stats_from_cases(errors, target)
                    unstable = bool(stats["evolved_worst"] > UNSTABLE)
                except RuntimeError as exc:
                    log(f"coarse {name}: {exc}")
                    errors, stats, unstable = None, None, True
                coarse[name] = dict(mesh=m, scheme=scheme, steps=st, dt=horizon / st, stats=stats,
                                    unstable=unstable,
                                    errors=None if errors is None else errors.tolist())
                if stats is not None and not unstable and stats["evolved_worst"] <= float(cc["time_below"]):
                    coarse_fn[name] = query
                    coarse_solve[name] = (solver_m, geom_m, m)
            log(f"coarse m={m} {scheme}: " + ", ".join(
                f"s{e['steps']}={e['stats']['evolved_worst'] if e['stats'] else float('nan'):.5f}"
                for e in coarse.values() if e["mesh"] == m and e["scheme"] == scheme))
        report["coarse_fom"] = coarse
        dump()
    report["coarse_fom"] = coarse

    # ------------------------------------------------------------- 6. timing ----
    case = int(cfg["timing_case"])
    u0, nu = jnp.asarray(dev[case, 0]), float(viscosities[case])
    fast = {}
    for name, query in coarse_fn.items():
        fast[f"COARSE_{name}"] = (lambda q=query: q(u0, nu))
        s_m, g_m, m_ = coarse_solve[name]
        u0m = jax.block_until_ready(spec_resample(u0, m_))
        fast[f"COARSESOLVE_{name}"] = (lambda s_m=s_m, g_m=g_m, u0m=u0m: s_m(u0m, nu, g_m))
    for name, fn in arms.items():
        if rows[name]["finite"]:
            fast[f"query_{name}"] = (lambda fn=fn: fn(u0, nu))
    for key, e in cnab.items():
        st = e["steps"]
        s_ = F.make_solver(horizon / st, st, st // 5)
        fast[f"CNAB2_s{st}"] = (lambda s_=s_: s_(u0, nu, geom))
    sent_keys = [s for s in (f"query_coordnet_head_k{k}", "query_pod_head_k8",
                             f"query_coordnet_span{max(spans)}") if s in fast]
    sentinels = {s: fast[s] for s in sent_keys}
    reps = int(cfg["timing_repetitions"])
    burn_gpu()
    t_sent0, _ = time_block(sentinels, reps, 2, seed=1)
    t_fast, last = time_block(fast, reps, 2, seed=2)
    t_sent1, _ = time_block(sentinels, reps, 1, seed=5)
    gate_ratio = float(cfg["neighbour_gate_ratio"])
    gate = {s: dict(solo_before_ms=t_sent0[s]["median_ms"], solo_end_ms=t_sent1[s]["median_ms"],
                    in_block_ms=t_fast[s]["median_ms"],
                    ratio=float(t_fast[s]["median_ms"] / t_sent0[s]["median_ms"]),
                    end_ratio=float(t_sent1[s]["median_ms"] / t_sent0[s]["median_ms"]))
            for s in sentinels}
    passed = all(1 / gate_ratio <= g["ratio"] <= gate_ratio
                 and 1 / gate_ratio <= g["end_ratio"] <= gate_ratio for g in gate.values())
    agree = 0.0
    for name, value in last.items():
        if name.startswith("COARSESOLVE_"):
            continue
        if name.startswith("query_"):
            src = rows[name[len("query_"):]]
        elif name.startswith("COARSE_"):
            src = coarse[name[len("COARSE_"):]]
        else:
            src = cnab[name[len("CNAB2_s"):]]
        if src.get("errors") is None:
            continue
        e = rel_case(np.asarray(value).reshape(dev.shape[1:]), dev, case)
        d = float(np.max(np.abs(e - np.asarray(src["errors"][case]))))
        agree = d if not np.isfinite(d) else max(agree, d)
        if not np.isfinite(agree):
            break
    report["timing"] = dict(case=case, fast=t_fast, sentinels_before=t_sent0,
                            sentinels_end=t_sent1, gate=gate, gate_passed=bool(passed),
                            timed_output_error_agreement=agree)
    log(f"timing gate {passed}; timed outputs vs accuracy pass {agree:.3e}")
    if not passed:
        failures.append("timing gate failed")
    if not (np.isfinite(agree) and agree <= 1e-9):
        failures.append(f"timed outputs disagree with the accuracy pass: {agree:.2e}")
    # comparator: fastest stable CNAB2 setting (finite, evolved worst <= 100 %) whose
    # evolved worst is no larger than the arm's; speedup = its median / the arm's median
    stable = {st: e for st, e in cnab.items() if e["stats"] is not None and not e["unstable"]}
    comp = {}
    for name, row in rows.items():
        if not row["finite"]:
            continue
        worst = row["stats"]["evolved_worst"]
        arm_ms = t_fast[f"query_{name}"]["median_ms"]
        eligible = {st: t_fast[f"CNAB2_s{st}"]["median_ms"] for st, e in stable.items()
                    if e["stats"]["evolved_worst"] <= worst}
        if eligible:
            st = min(eligible, key=eligible.get)
            comp[name] = dict(arm_ms=arm_ms, arm_worst=worst, comparator=f"CNAB2_s{st}",
                              comparator_worst=stable[st]["stats"]["evolved_worst"],
                              comparator_ms=eligible[st], speedup=eligible[st] / arm_ms,
                              eligible={f"CNAB2_s{a}": b for a, b in eligible.items()})
        else:
            comp[name] = dict(arm_ms=arm_ms, arm_worst=worst, comparator=None, speedup=None,
                              note="no stable CNAB2 setting is as accurate")
    report["comparators"] = comp
    # accuracy-matched mesh-free comparator (PROTOCOL task 1): fastest stable coarse
    # configuration (end-to-end median) whose worst error is no larger than the arm's
    ccomp = {}
    for name, row in rows.items():
        if not row["finite"]:
            continue
        worst = row["stats"]["evolved_worst"]
        arm_ms = t_fast[f"query_{name}"]["median_ms"]
        elig = {c: t_fast[f"COARSE_{c}"]["median_ms"] for c in coarse_fn
                if coarse[c]["stats"]["evolved_worst"] <= worst}
        if elig:
            c = min(elig, key=elig.get)
            cs = min(elig, key=lambda x: t_fast[f"COARSESOLVE_{x}"]["median_ms"])
            ccomp[name] = dict(arm_ms=arm_ms, arm_worst=worst, comparator=c,
                               comparator_worst=coarse[c]["stats"]["evolved_worst"],
                               comparator_ms=elig[c], ratio=elig[c] / arm_ms,
                               comparator_solve_only=cs,
                               comparator_solve_only_ms=t_fast[f"COARSESOLVE_{cs}"]["median_ms"],
                               ratio_solve_only=t_fast[f"COARSESOLVE_{cs}"]["median_ms"] / arm_ms,
                               eligible=len(elig))
        else:
            ccomp[name] = dict(arm_ms=arm_ms, arm_worst=worst, comparator=None, ratio=None,
                               note="no timed coarse configuration is as accurate")
    report["coarse_comparators"] = ccomp
    log("coarse ratios (comparator ms / arm ms): " + ", ".join(
        f"{a}={b['ratio']:.2f}[{b['comparator']}]" for a, b in ccomp.items() if b["ratio"]))
    # small slices for figures: final frame, z = 0 plane, timing case and the worst case of
    # the coordinate head
    try:
        hw = int(np.argmax(np.max(np.asarray(rows[f"coordnet_head_k{k}"]["errors"])[:, 1:], axis=1)))
        sl = {}
        for cidx in sorted({case, hw}):
            uc, nuc = jnp.asarray(dev[cidx, 0]), float(viscosities[cidx])
            sl[f"truth_c{cidx}"] = dev[cidx, -1, :, :, :, 0]
            picks = {a: arms[a] for a in arms if rows[a]["finite"] and
                     (a.endswith("head_k8") or a in ("coordnet_span8", "coordnet_span32", "interp_span32"))}
            for a, fn_ in picks.items():
                sl[f"{a}_c{cidx}"] = np.asarray(fn_(uc, nuc)).reshape(dev.shape[1:])[-1, :, :, :, 0]
            for a in {v["comparator"] for v in ccomp.values() if v.get("comparator")}:
                sl[f"COARSE_{a}_c{cidx}"] = np.asarray(coarse_fn[a](uc, nuc))[-1, :, :, :, 0]
        np.savez_compressed(out / "slices.npz", **{a: b.astype(np.float32) for a, b in sl.items()})
        report["slices"] = dict(cases=sorted({case, hw}), keys=sorted(sl))
    except Exception as exc:  # figures are not a gate
        log(f"slices not saved: {exc}")
    log("speedups: " + ", ".join(f"{a}={b['speedup']:.2f}" for a, b in comp.items() if b["speedup"]))
    report["failures"] = failures
    hard = [f for f in failures if not f.startswith("flag")]
    report["status"] = "final" if not hard else "failed-gates"
    dump()
    log(f"done; failures: {failures}")
    if hard:
        sys.exit(3)


if __name__ == "__main__":
    main()
