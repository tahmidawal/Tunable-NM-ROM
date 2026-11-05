"""ns3d-offmesh-quadrature, one mesh per allocation (DESIGN.md sections 2-6).

  0. inputs     sha256 of bank, head, reference summary, rules (G0); closed seeds refused
  1. bank/ops   the frozen coordnet bank at this mesh (bankio.mesh_bank, unchanged), the
                exact operators A, lam, Dd, T (SR.build_operators_fast, unchanged)
  2. tensor     head and span64 rollouts with this lane's driver (kind=tensor) and with the
                parent drivers (G1 parity), reproduction vs the ns3d-pod-match summary (G1)
  3. checks     off-grid derivatives of the network (G4); NUDFT vs FFT evaluator; the
                positive control trapN under `interp` (G2, hard stop)
  4. ladder     every rule x integrand: rho vs tensor (and vs the `net` continuum target),
                head + span64 rollouts, errors vs reference and vs the tensor ROM
  5. CNAB2      the parent's step ladder
  6. M*         plateau sizes per DESIGN section 4 (also recomputed by the report)
  7. timing     burn-in, sentinels, interleaved block, sentinels; kernel microbenchmark
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
              "experiments/ns3d-grok", "experiments/ns3d-shift", "experiments/ns3d-shift-head",
              "experiments/ns3d-coordnet"):
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
import coordnet as CN  # noqa: E402
import bankio as BI  # noqa: E402
from head_ladder import burn_gpu, time_block, rel_case  # noqa: E402
import qrom as Q  # noqa: E402
from make_rules import trapezoid  # noqa: E402

log = D.log
UNSTABLE = 1.0


def next_pow2(m):
    return 1 << int(np.ceil(np.log2(max(int(m), 1))))


def mstar(sizes, worst, median, w_t, d_t, finite, tol=1.05):
    """DESIGN section 4: smallest ladder size that, with every larger size, reaches tensor accuracy."""
    order = np.argsort(sizes)
    ok = [bool(finite[i] and worst[i] <= tol * w_t and median[i] <= tol * d_t) for i in order]
    best = None
    for j in range(len(order) - 1, -1, -1):
        if not ok[j]:
            break
        best = int(np.asarray(sizes)[order[j]])
    return best


def mstar_delta(sizes, delta, finite, tol=1e-4):
    order = np.argsort(sizes)
    best = None
    for j in range(len(order) - 1, -1, -1):
        i = order[j]
        if not (finite[i] and delta[i] <= tol):
            break
        best = int(np.asarray(sizes)[i])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get("smoke", {}))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    for closed in cfg["closed_seeds"]:
        if int(closed) in (int(cfg["eval_seed"]),):
            raise RuntimeError("refusing to read a closed seed")
    if int(cfg["eval_seed"]) != 202609202:
        raise RuntimeError("this lane is development-cohort only (seed 202609202)")
    t_job = time.time()
    n = int(cfg["n"])
    horizon = float(cfg["horizon"])
    modes = int(cfg["modes"])
    target = float(cfg["target_relative"])
    k = int(cfg["k"])
    report = dict(schema="ns3d-offmesh-quadrature-v1", config=cfg, smoke=bool(args.smoke),
                  device=str(jax.devices()), source_commit=os.environ.get("SOURCE_COMMIT"),
                  job_id=os.environ.get("SLURM_JOB_ID"),
                  precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                  files={name: D.sha256_file(ROOT / name) for name in (
                      "experiments/ns3d-offmesh-quadrature/qrom.py",
                      "experiments/ns3d-offmesh-quadrature/run_mesh.py",
                      "experiments/ns3d-offmesh-quadrature/make_rules.py",
                      "experiments/ns3d-offmesh-quadrature/vendor/quadrature-burgers3d/rules.py",
                      "experiments/ns3d-coordnet/coordnet.py",
                      "experiments/ns3d-coordnet/bankio.py",
                      "experiments/ns3d-shift/shift_rom.py",
                      "experiments/ns3d-shift-head/head_rom.py")})
    try:
        report["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["gpu"] = "unavailable"
    failures = []

    def dump():
        report["elapsed_seconds"] = time.time() - t_job
        report["failures"] = failures
        D.dump_json(out / "summary.json", report)

    # ------------------------------------------------------------ 0. inputs ----
    checks = {cfg["bank_file"]: cfg["bank_sha256"], cfg["head_file"]: cfg["head_sha256"],
              cfg["reference_summary"]: cfg["reference_summary_sha256"],
              cfg["rules_file"]: cfg["rules_sha256"]}
    for path, sha in checks.items():
        got = D.sha256_file(ROOT / path)
        if got != sha:
            raise RuntimeError(f"frozen input {path} sha256 {got} != {sha}")
    report["frozen_inputs_checked"] = checks
    refsum = json.loads((ROOT / cfg["reference_summary"]).read_text())

    # ---------------------------------------------------------- 1. bank, ops ----
    t0 = time.time()
    pc = BI.load_params(ROOT / cfg["bank_file"])
    Tc = np.asarray(np.load(ROOT / cfg["bank_file"])["T"])
    Gc, binfo = BI.mesh_bank(pc, Tc, n, chunk=int(cfg.get("sample_chunk_eval", 32768)),
                              abort=float(cfg["orthonormality_abort"]))
    S = binfo.pop("lowdin_factor")
    TS = Tc @ S
    R = Gc.shape[1]
    report["bank"] = binfo
    report["bank_seconds"] = time.time() - t0
    t0 = time.time()
    ops_c, rep_c = SR.build_operators_fast(Gc, n, modes, check_modes=int(cfg["check_modes"]))
    report["tensor_build_seconds"] = time.time() - t0
    report["operator_checks"] = rep_c
    ids, _ = SR.test_mode_ids(n, modes)
    log(f"bank {binfo}; operators {rep_c}")
    Gj = jnp.asarray(Gc)
    Aj, Tj = jnp.asarray(ops_c["A"]), jnp.asarray(ops_c["T"])
    lamj, Ddj = jnp.asarray(ops_c["lam"]), jnp.asarray(ops_c["Dd"])

    ev_par = F.parameters(int(cfg["eval_seed"]), int(cfg["eval_cases"]))
    report["eval_parameter_sha256"] = D.sha256_array(ev_par)
    dev, _, _ = D.generate(ev_par, n, float(cfg["dt_truth"]), horizon)
    D.require_finite(dev, "evaluation")
    visc = ev_par[:, -1]
    ncase = len(dev)
    devj = jnp.asarray(dev)
    u0n = np.linalg.norm(dev[:, 0].reshape(ncase, -1), axis=1)

    ck = np.load(ROOT / cfg["head_file"])
    p_c = {key: jnp.asarray(ck[f"p_{key}"]) for key in ("W1", "b1", "W2", "b2", "W3", "b3", "Ws")}
    Zj = jnp.asarray(ck["Z"])
    Hc = H.head_apply(p_c, Zj)
    Hn = jnp.sum(Hc * Hc, 1)
    C0 = jnp.zeros((R, 0))
    dump()

    # ------------------------------------------------------------ 2. tensor ----
    dtv, iters, damping = float(cfg["ladder_dt"]), int(cfg["ladder_iters"]), float(cfg["damping"])
    ic_iters = int(cfg["ic_iters"])
    steps = D.nsteps_for(dtv, horizon)
    oe = steps // 5
    runs = {}
    for kind in ("tensor", "quad"):
        for hist in (True, False):
            runs[("head", kind, hist)] = Q.make_head_run(dtv, steps, oe, n, R, k, iters, damping,
                                                         ic_iters, kind, history=hist)
            runs[("span64", kind, hist)] = Q.make_span_run(dtv, steps, oe, n, R, iters, damping,
                                                           kind, history=hist)

    def call(arm, kind, Qd, hist, case):
        u0, nu = devj[case, 0], float(visc[case])
        if arm == "head":
            return runs[(arm, kind, hist)](u0, nu, Gj, Aj, Qd, lamj, Ddj, p_c, C0, Hc, Hn, Zj)
        return runs[(arm, kind, hist)](u0, nu, Gj, Aj, Qd, lamj, Ddj)

    @jax.jit
    def errs(fields, truth, ref, u0norm):
        e1 = jnp.linalg.norm((fields - truth).reshape(fields.shape[0], -1), axis=1) / u0norm
        e2 = jnp.linalg.norm((fields - ref).reshape(fields.shape[0], -1), axis=1) / u0norm
        return e1, e2, jnp.all(jnp.isfinite(fields))

    tensor_fields, states, trows = {}, {}, {}
    for arm in ("head", "span64"):
        fl, hs, er = [], [], []
        for case in range(ncase):
            f, h = call(arm, "tensor", Tj, True, case)
            fl.append(f)
            hs.append(np.asarray(0.5 * (h[1:] + h[:-1])))
            er.append(rel_case(np.asarray(f), dev, case))
        tensor_fields[arm] = fl
        states[arm] = np.concatenate(hs)
        errors = np.stack(er)
        trows[arm] = dict(errors=errors.tolist(), stats=D.stats_from_cases(errors, target))
        log(f"tensor {arm}: worst {trows[arm]['stats']['evolved_worst']:.6e} "
            f"median {trows[arm]['stats']['evolved_median']:.6e}")
    report["tensor"] = trows
    # G1a: parity of this driver with the parent drivers (same inputs, tensor)
    parent_head = H.make_head_run(dtv, steps, oe, n, R, k, 0, iters=iters, damping=damping,
                                  ic_iters=ic_iters, frame="free")
    parent_span = SR.make_frozen_run(dtv, steps, oe, n, R, iters=iters, damping=damping,
                                     extrapolate=True, diagnose=False)
    par = {}
    for arm, fn in (("head", lambda u, nu: parent_head(u, nu, Gj, Aj, Tj, lamj, Ddj, p_c, C0, Hc, Hn, Zj)),
                    ("span64", lambda u, nu: parent_span(u, nu, Gj, Aj, Tj, lamj, Ddj))):
        gap = 0.0
        for case in range(min(4, ncase)):
            ref = fn(devj[case, 0], float(visc[case]))
            gap = max(gap, float(jnp.linalg.norm(tensor_fields[arm][case] - ref) / jnp.linalg.norm(ref)))
        par[arm] = gap
    report["G1_driver_parity"] = dict(gaps=par, passed=bool(max(par.values()) <= 1e-12))
    if not report["G1_driver_parity"]["passed"]:
        failures.append(f"G1 driver parity {par}")
    # G1b: reproduction of the ns3d-pod-match development errors
    rep = {}
    for arm, key in (("head", f"coordnet_head_k{k}"), ("span64", "coordnet_span64")):
        want = np.asarray(refsum["rollouts"][key]["errors"])[:ncase]
        rep[arm] = float(np.max(np.abs(np.asarray(trows[arm]["errors"]) - want)))
    report["G1_reproduction"] = dict(max_abs_difference=rep, passed=bool(max(rep.values()) <= 1e-8),
                                     source=cfg["reference_summary"])
    if not report["G1_reproduction"]["passed"]:
        failures.append(f"G1 reproduction {rep}")
    log(f"G1 parity {par}; reproduction {rep}")
    allstates = np.concatenate((states["head"], states["span64"]))
    nh = len(states["head"])
    Sj = jnp.asarray(allstates)
    aT = np.asarray(Q.tensor_adv_batch(Tj, Sj))
    JT = np.asarray(Q.tensor_jac_batch(Tj, Sj))
    report["states"] = dict(head=int(nh), span64=int(len(states["span64"])))
    dump()

    def rho(U, W, P, ref_a=None, jac=True):
        a = np.concatenate([np.asarray(Q.adv_batch(U, W, P, Sj[s:s + 32]))
                            for s in range(0, len(allstates), 32)])
        out_ = {}
        for label, ref in (("tensor", aT),) + ((("continuum", ref_a),) if ref_a is not None else ()):
            r = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
            out_[f"adv_{label}"] = dict(worst=float(r.max()), median=float(np.median(r)),
                                        head_worst=float(r[:nh].max()), span_worst=float(r[nh:].max()))
        if jac:
            bs = max(1, int(2 ** 20 // max(U.shape[0], 1)))
            J = np.concatenate([np.asarray(Q.jac_batch(U, W, P, Sj[s:s + bs]))
                                for s in range(0, len(allstates), bs)])
            r = (np.linalg.norm((J - JT).reshape(len(J), -1), axis=1)
                 / np.linalg.norm(JT.reshape(len(J), -1), axis=1))
            out_["jac_tensor"] = dict(worst=float(r.max()), median=float(np.median(r)))
        return out_, a

    def rollouts(Qd, keep0=None):
        res = {}
        for arm in ("head", "span64"):
            e_ref = np.full((ncase, 6), np.nan)
            e_t = np.full((ncase, 6), np.nan)
            finite = True
            for case in range(ncase):
                f = call(arm, "quad", Qd, False, case)
                a1, a2, fin = errs(f, devj[case], tensor_fields[arm][case], u0n[case])
                if not bool(fin):
                    finite = False
                    break
                e_ref[case], e_t[case] = np.asarray(a1), np.asarray(a2)
                if keep0 is not None and case == int(cfg["timing_case"]):
                    keep0[arm] = np.asarray(f)
            res[arm] = dict(finite=finite, errors=e_ref.tolist(),
                            stats=D.stats_from_cases(e_ref, target) if finite else None,
                            delta_tensor=float(np.max(e_t)) if finite else None)
        return res

    # ------------------------------------------------------------ 3. checks ----
    rng = np.random.default_rng(5)
    Xr = rng.uniform(size=(int(cfg["derivative_points"]), 3))
    Un, Wn = Q.net_tables(pc, TS, Xr)
    hfd = 1e-5
    grads = []
    for d in range(3):
        e = np.zeros(3)
        e[d] = hfd
        up = jnp.einsum("pcr,rs->pcs", CN.velocity(pc, jnp.asarray(Xr + e)), jnp.asarray(TS))
        um = jnp.einsum("pcr,rs->pcs", CN.velocity(pc, jnp.asarray(Xr - e)), jnp.asarray(TS))
        grads.append((up - um) / (2 * hfd))
    curl_fd = jnp.stack((grads[1][:, 2] - grads[2][:, 1], grads[2][:, 0] - grads[0][:, 2],
                         grads[0][:, 1] - grads[1][:, 0]), axis=1)
    gfull = jnp.einsum("epcr,rs->epcs", CN.velocity_gradient(pc, jnp.asarray(Xr)), jnp.asarray(TS))
    div = gfull[0, :, 0] + gfull[1, :, 1] + gfull[2, :, 2]
    waves, Cspec, sinfo = Q.interp_spectrum(Gc, n)
    Ui, Wi = Q.interp_tables(waves, Cspec, Xr, R)
    g4 = dict(curl_vs_fd=float(jnp.linalg.norm(Wn - curl_fd) / jnp.linalg.norm(Wn)),
              divergence_over_gradient=float(jnp.linalg.norm(div) / jnp.linalg.norm(gfull)),
              net_vs_interp_u=float(jnp.linalg.norm(Un - Ui) / jnp.linalg.norm(Ui)),
              net_vs_interp_curl=float(jnp.linalg.norm(Wn - Wi) / jnp.linalg.norm(Wi)),
              spectrum=sinfo, points=int(len(Xr)))
    g4["passed"] = bool(g4["curl_vs_fd"] <= 1e-6 and g4["divergence_over_gradient"] <= 1e-10)
    report["G4_offgrid_derivatives"] = g4
    if not g4["passed"]:
        failures.append(f"G4 off-grid derivatives {g4}")
    log(f"G4 {g4}")
    del Un, Wn, Ui, Wi, gfull, grads, curl_fd
    # positive control: interp on the shifted n^3 grid (FFT phase shift), G2
    shift = np.random.default_rng(int(cfg["control_shift_seed"])).uniform(size=3) / n
    Xg = Q.shifted_grid_points(n, shift)
    wg = np.full(len(Xg), 1.0 / len(Xg))
    Ug, Wg = Q.shifted_grid_tables(Gc, n, shift)
    sub = np.sort(np.random.default_rng(6).choice(len(Xg), size=512, replace=False))
    Us, Ws = Q.interp_tables(waves, Cspec, Xg[sub], R)
    nudft_gap = float(max(jnp.linalg.norm(Us - Ug[sub]) / jnp.linalg.norm(Ug[sub]),
                          jnp.linalg.norm(Ws - Wg[sub]) / jnp.linalg.norm(Wg[sub])))
    Pg = Q.test_values(Xg, ids, wg, n)
    rg, _ = rho(Ug, Wg, Pg)
    ctrl = rollouts((Ug, Wg, Pg))
    g2 = dict(rule=f"trap{n}_shifted", shift=shift.tolist(), nudft_vs_fft=nudft_gap, rho=rg,
              head_delta_tensor=ctrl["head"]["delta_tensor"],
              span_delta_tensor=ctrl["span64"]["delta_tensor"])
    g2["passed"] = bool(nudft_gap <= 1e-10 and rg["adv_tensor"]["worst"] <= 1e-12
                        and rg["jac_tensor"]["worst"] <= 1e-12 and ctrl["head"]["finite"]
                        and ctrl["head"]["delta_tensor"] <= 1e-10)
    report["G2_positive_control"] = g2
    log(f"G2 {g2}")
    del Ug, Wg, Pg
    gc.collect()
    dump()
    if not g2["passed"]:
        failures.append("G2 positive control failed (hard stop)")
        report["status"] = "failed-gates"
        dump()
        sys.exit(3)

    # continuum target for `net` (streamed, rho only)
    def net_continuum(p_rule, seed, chunk=32768):
        X, w, _ = trapezoid(p_rule, seed)
        acc = np.zeros_like(aT)
        for s in range(0, len(X), chunk):
            U, W = Q.net_tables(pc, TS, X[s:s + chunk])
            P = Q.test_values(X[s:s + chunk], ids, w[s:s + chunk], n)
            for t in range(0, len(allstates), 64):
                acc[t:t + 64] += np.asarray(Q.adv_batch(U, W, P, Sj[t:t + 64]))
        return acc
    t0 = time.time()
    pa, pb = (int(x) for x in cfg["continuum_p"])
    sa, sb = (int(x) for x in cfg["continuum_seeds"])
    a_cont = net_continuum(pa, sa)
    a_chk = net_continuum(pb, sb)
    conv = float(np.max(np.linalg.norm(a_cont - a_chk, axis=1) / np.linalg.norm(a_cont, axis=1)))
    gap = np.linalg.norm(a_cont - aT, axis=1) / np.linalg.norm(aT, axis=1)
    report["net_continuum"] = dict(p=pa, check_p=pb, convergence=conv,
                                   converged=bool(conv <= 1e-10), seconds=time.time() - t0,
                                   gap_vs_tensor=dict(worst=float(gap.max()), median=float(np.median(gap)),
                                                      head_worst=float(gap[:nh].max()),
                                                      span_worst=float(gap[nh:].max())))
    if conv > 1e-10:
        failures.append(f"flag: net continuum target not converged ({conv:.2e})")
    log(f"net continuum {report['net_continuum']}")
    dump()

    # ------------------------------------------------------------ 4. ladder ----
    rz = np.load(ROOT / cfg["rules_file"])
    rmeta = json.loads((ROOT / cfg["rules_file"]).with_name("rules.json").read_text())["rules"]
    ladder = {}
    case0_fields = {}
    names = [cfg["must_fail"]] + [nm for fam in cfg["families"] for nm in cfg["ladder"][fam]]
    for name in names:
        X, w = rz[f"{name}_X"], rz[f"{name}_w"]
        Mq = len(w)
        P = Q.test_values(X, ids, w, n)
        entry = dict(family=rmeta[name]["family"], m=int(Mq), bucket=next_pow2(Mq))
        for integ in cfg["integrands"]:
            t0 = time.time()
            if integ == "net":
                U, W = Q.net_tables(pc, TS, X)
            else:
                U, W = Q.interp_tables(waves, Cspec, X, R)
            jax.block_until_ready(W)
            build = time.time() - t0
            r_, _ = rho(U, W, P, ref_a=a_cont if integ == "net" else None)
            k0 = {} if integ == "net" else None
            roll = rollouts(Q.pad_tables(U, W, P, next_pow2(Mq)), keep0=k0)
            if k0 is not None:
                case0_fields[name] = k0
            entry[integ] = dict(build_seconds=build, rho=r_, rollouts=roll)
            log(f"{name} {integ}: rho_a {r_['adv_tensor']['worst']:.3e} head worst "
                f"{roll['head']['stats']['evolved_worst'] if roll['head']['finite'] else float('nan'):.6e}")
            del U, W
        ladder[name] = entry
        report["ladder"] = ladder
        dump()
        del P
        gc.collect()
    mf = ladder[cfg["must_fail"]]
    g3 = {integ: mf[integ]["rho"]["adv_tensor"]["worst"] for integ in cfg["integrands"]}
    report["G3_must_fail"] = dict(rho=g3, passed=bool(min(g3.values()) >= 1e-1))
    if not report["G3_must_fail"]["passed"]:
        failures.append(f"flag: G3 must-fail control did not fail {g3}")

    # ------------------------------------------------------------- 5. CNAB2 ----
    geom = F.geometry(n)
    cnab = {}
    for st in cfg["cnab2_steps"]:
        st = int(st)
        solver = F.make_solver(horizon / st, st, st // 5)
        er = np.full((ncase, 6), np.nan)
        ok = True
        for case in range(ncase):
            f = solver(devj[case, 0], float(visc[case]), geom)
            a1, _, fin = errs(f, devj[case], devj[case], u0n[case])
            if not bool(fin):
                ok = False
                break
            er[case] = np.asarray(a1)
        stats = D.stats_from_cases(er, target) if ok else None
        cnab[str(st)] = dict(steps=st, dt=horizon / st, stats=stats, errors=er.tolist(),
                             unstable=bool((not ok) or stats["evolved_worst"] > UNSTABLE))
        log(f"CNAB2 {st}: {stats['evolved_worst'] if stats else float('nan')}")
    report["cnab2"] = cnab
    dump()

    # ---------------------------------------------------------------- 6. M* ----
    ms = {}
    for integ in cfg["integrands"]:
        for fam in cfg["families"]:
            for arm in ("head", "span64"):
                nm = cfg["ladder"][fam]
                rows = [ladder[x][integ]["rollouts"][arm] for x in nm]
                sizes = [ladder[x]["m"] for x in nm]
                fin = [r["finite"] for r in rows]
                wv = [r["stats"]["evolved_worst"] if r["finite"] else np.inf for r in rows]
                dv = [r["stats"]["evolved_median"] if r["finite"] else np.inf for r in rows]
                de = [r["delta_tensor"] if r["finite"] else np.inf for r in rows]
                st_ = trows[arm]["stats"]
                m1 = mstar(sizes, wv, dv, st_["evolved_worst"], st_["evolved_median"], fin)
                m2 = mstar_delta(sizes, de, fin)
                ms[f"{integ}/{fam}/{arm}"] = dict(
                    mstar=m1, mstar_rule=(nm[sizes.index(m1)] if m1 is not None else None),
                    mstar_delta=m2, mstar_delta_rule=(nm[sizes.index(m2)] if m2 is not None else None),
                    endpoint_only=bool(m1 is not None and m1 == max(sizes)))
    report["mstar"] = ms
    log(f"M*: {ms}")
    dump()

    # ------------------------------------------------------------ 7. timing ----
    case = int(cfg["timing_case"])
    u0, nu = devj[case, 0], float(visc[case])
    fast, qtables = {}, {}

    def qtab(name):
        if name not in qtables:
            X, w = rz[f"{name}_X"], rz[f"{name}_w"]
            U, W = Q.net_tables(pc, TS, X)
            qtables[name] = (U, W, Q.test_values(X, ids, w, n))
        return qtables[name]
    fast["tensor_head"] = lambda: runs[("head", "tensor", False)](u0, nu, Gj, Aj, Tj, lamj, Ddj, p_c, C0, Hc, Hn, Zj)
    fast["tensor_span64"] = lambda: runs[("span64", "tensor", False)](u0, nu, Gj, Aj, Tj, lamj, Ddj)
    timed_rules = list(cfg["ladder"]["lat"])
    for fam in cfg["families"]:
        r_ = ms.get(f"net/{fam}/head", {}).get("mstar_rule")
        if r_ is not None and r_ not in timed_rules:
            timed_rules.append(r_)
    for name in timed_rules:
        Qd = qtab(name)
        fast[f"net_head_{name}"] = (lambda Qd=Qd: runs[("head", "quad", False)](
            u0, nu, Gj, Aj, Qd, lamj, Ddj, p_c, C0, Hc, Hn, Zj))
        if name in cfg["ladder"]["lat"]:
            fast[f"net_span64_{name}"] = (lambda Qd=Qd: runs[("span64", "quad", False)](
                u0, nu, Gj, Aj, Qd, lamj, Ddj))
    for key, e in cnab.items():
        st = e["steps"]
        s_ = F.make_solver(horizon / st, st, st // 5)
        fast[f"CNAB2_s{st}"] = (lambda s_=s_: s_(u0, nu, geom))
    sent_keys = [s for s in ("tensor_head", f"net_head_{cfg['sentinel_rule']}") if s in fast]
    sentinels = {s: fast[s] for s in sent_keys}
    reps = int(cfg["timing_repetitions"])
    burn_gpu()
    t_s0, _ = time_block(sentinels, reps, 2, seed=1)
    t_fast, last = time_block(fast, reps, 2, seed=2)
    t_s1, _ = time_block(sentinels, reps, 1, seed=5)
    gr = float(cfg["neighbour_gate_ratio"])
    gate = {s: dict(solo_before_ms=t_s0[s]["median_ms"], solo_end_ms=t_s1[s]["median_ms"],
                    in_block_ms=t_fast[s]["median_ms"],
                    ratio=float(t_fast[s]["median_ms"] / t_s0[s]["median_ms"]),
                    end_ratio=float(t_s1[s]["median_ms"] / t_s0[s]["median_ms"])) for s in sentinels}
    passed = all(1 / gr <= g["ratio"] <= gr and 1 / gr <= g["end_ratio"] <= gr for g in gate.values())
    # G5 (amendment A2): every timed output must be finite and reproduce the accuracy-pass FIELDS
    agree = 0.0
    for name, value in last.items():
        got = np.asarray(value).reshape(dev.shape[1:])
        if not np.all(np.isfinite(got)):
            agree = np.inf
            break
        if name.startswith("tensor_"):
            ref = np.asarray(tensor_fields[name[len("tensor_"):]][case])
        elif name.startswith("net_head_"):
            ref = case0_fields[name[len("net_head_"):]]["head"]
        elif name.startswith("net_span64_"):
            ref = case0_fields[name[len("net_span64_"):]]["span64"]
        else:
            ref = None
        if ref is None:   # CNAB2: compare errors against the accuracy pass
            src = np.asarray(cnab[name[len("CNAB2_s"):]]["errors"][case])
            if not np.all(np.isfinite(src)):
                continue
            d = float(np.max(np.abs(rel_case(got, dev, case) - src)))
        else:
            d = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300))
        if not np.isfinite(d):
            agree = np.inf
            break
        agree = max(agree, d)
    # kernel microbenchmark: 30 Jacobian contractions (one query's sweeps)
    pre0 = 1.0 + 0.5 * dtv * nu * lamj

    def make_kernel(kind):
        @jax.jit
        def kern(adv, mid0):
            def body(m, _):
                J = Q.jac_adv(kind, adv, m)
                return m + 1e-12 * jnp.sum(J, axis=0), None   # every entry feeds the dependency (A2)
            m, _ = jax.lax.scan(body, mid0, None, length=steps * iters)
            return m
        return kern
    kt, kq = make_kernel("tensor"), make_kernel("quad")
    sym = Q.fold_adv("tensor", Tj, pre0)
    mid0 = Sj[0]
    kern = {"kernel_tensor": (lambda: kt(sym, mid0))}
    for name in cfg["ladder"]["lat"]:
        adv = Q.fold_adv("quad", qtab(name), pre0)
        kern[f"kernel_{name}"] = (lambda adv=adv: kq(adv, mid0))
    burn_gpu()
    t_kern, _ = time_block(kern, reps, 2, seed=9)
    report["timing"] = dict(case=case, fast=t_fast, sentinels_before=t_s0, sentinels_end=t_s1,
                            gate=gate, gate_passed=bool(passed), timed_output_error_agreement=agree,
                            kernel=t_kern, timed_rules=timed_rules,
                            flops=dict(tensor_per_sweep=2 * modes * R * R,
                                       quad_per_sweep_per_point=2 * 3 * modes * R + 4 * 3 * R + 12 * R))
    log(f"timing gate {passed}; agreement {agree:.3e}")
    if not passed:
        failures.append("G5 timing gate failed")
    if not (np.isfinite(agree) and agree <= 1e-9):
        failures.append(f"G5 timed outputs disagree with the accuracy pass: {agree:.2e}")
    stable = {st: e for st, e in cnab.items() if e["stats"] is not None and not e["unstable"]}
    comp = {}
    for key in fast:
        if key.startswith("CNAB2_"):
            continue
        if key.startswith("tensor_"):
            worst = trows[key[len("tensor_"):]]["stats"]["evolved_worst"]
        else:
            arm = "head" if key.startswith("net_head_") else "span64"
            nm = key.split("_", 2)[2] if arm == "head" else key[len("net_span64_"):]
            row = ladder[nm]["net"]["rollouts"][arm]
            if not row["finite"]:
                continue
            worst = row["stats"]["evolved_worst"]
        ms_ = t_fast[key]["median_ms"]
        elig = {st: t_fast[f"CNAB2_s{st}"]["median_ms"] for st, e in stable.items()
                if e["stats"]["evolved_worst"] <= worst}
        if elig:
            st = min(elig, key=elig.get)
            comp[key] = dict(arm_ms=ms_, arm_worst=worst, comparator=f"CNAB2_s{st}",
                             comparator_worst=stable[st]["stats"]["evolved_worst"],
                             comparator_ms=elig[st], speedup=elig[st] / ms_)
        else:
            comp[key] = dict(arm_ms=ms_, arm_worst=worst, comparator=None, speedup=None)
    report["comparators"] = comp
    hard = [f for f in failures if not f.startswith("flag")]
    report["status"] = "final" if not hard else "failed-gates"
    dump()
    log(f"done; failures: {failures}")
    if hard:
        sys.exit(3)


if __name__ == "__main__":
    main()
