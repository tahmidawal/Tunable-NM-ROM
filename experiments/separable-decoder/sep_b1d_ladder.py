"""JOB A — "ladder on one GPU" (2026-08-29): is the tensor arm's ONLINE cost
constant in N?  One process loads ALL committed checkpoints (NS env, default
128..4096), builds every arm at every N, then times arms x N x trajectories
with the reps OUTERMOST and the N order alternating between reps, so that
GPU/clock drift within the job cannot masquerade as an N-trend.

Per (N, arm, trajectory, rep): IC-fit ms, latent-solve (rollout) ms, decode
ms; per (N, arm, trajectory) from the last timed rep: rollout error, total
LM attempts (every attempt evaluates r+J in the nocond optimized path), total
accepted steps (nJ), stop reasons.  ms per LM iteration = roll ms / attempts.
Only the OPTIMIZED rollout path is timed (the production configuration, the
same OPT flags as sep_b1d_fast / sep_b1d_tensor).

Artifacts: ART_DIR/sep_b1d_scale_n{N}.pkl, _nodes.npz, .json (parity ref).
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import b1d_common as b1
import b1d_fast_common as fc
import b1d_tensor_common as tc

F64 = jnp.float64
NS = [int(v) for v in os.environ.get("NS", "128,256,512,1024,2048,4096").split(",")]
ARMS = os.environ.get("ARMS", "oracle,base_tight,tensor").split(",")
ART_DIR = os.environ["ART_DIR"]
OUT = os.environ.get("OUT", "/tmp/sep_b1d_ladder.json")
TIME_REPS = int(os.environ.get("TIME_REPS", "5"))
BURN = int(os.environ.get("BURN", "2"))
OPT = dict(solver=os.environ.get("OPT_SOLVER", "gj"),
           onepass=os.environ.get("OPT_ONEPASS", "1") == "1",
           hoist=os.environ.get("OPT_HOIST", "1") == "1",
           nocond=os.environ.get("OPT_NOCOND", "1") == "1",
           lean=os.environ.get("OPT_LEAN", "1") == "1",
           nodot=os.environ.get("OPT_NODOT", "1") == "1",
           unroll=int(os.environ.get("OPT_UNROLL", "1")),
           scan_unroll=int(os.environ.get("OPT_SCAN_UNROLL", "5")),
           ic_unroll=int(os.environ.get("OPT_IC_UNROLL", "1")),
           with_att=True)
log = b1.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def blk(x):
    return jax.block_until_ready(x)


def hist_of(arr):
    a = np.asarray(arr)
    return {str(int(r_)): int(np.sum(a == r_)) for r_ in np.unique(a)}


def main():
    dev = jax.devices()[0]
    gpu = getattr(dev, "device_kind", str(dev))
    log(f"jax_backend={dev.platform} device={dev} gpu={gpu} B1D-LADDER NS={NS} "
        f"arms={ARMS} reps={TIME_REPS} burn={BURN} opt={OPT}")
    t_all = time.time()
    T = b1.NUM_STEPS + 1
    report = dict(config=dict(
        kind="b1d_ladder", NS=NS, arms=ARMS, K=fc.K, R=fc.R, M=fc.M,
        seed=fc.SEED0, test_seed=fc.TEST_SEED, n_test=fc.N_TEST,
        time_reps=TIME_REPS, burn=BURN, opt=OPT, art_dir=ART_DIR,
        order="reps outermost; N ascending on even reps, descending on odd; "
              "arms inner; trajectories innermost",
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=gpu, jax_version=jax.__version__,
        commit=os.environ.get("COMMIT"),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        cells={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- build every cell ------------------------------------
    cells = {}
    for N in NS:
        t0 = time.time()
        su = fc.Setup(f"{ART_DIR}/sep_b1d_scale_n{N}.pkl", N)
        U_test, nu_test = fc.gen_test(N)
        arms_xw = fc.load_arms(f"{ART_DIR}/sep_b1d_scale_n{N}_nodes.npz")
        base = json.load(open(f"{ART_DIR}/sep_b1d_scale_n{N}.json"))
        G_np, Phi_np = np.asarray(su.G_int), np.asarray(su.Phi_j)
        Q = tc.symmetrize(tc.build_T(Phi_np, G_np, su.dx, chunk=256))
        ops = {}
        for arm in ARMS:
            if arm == "oracle":
                ops[arm] = fc.make_device_fast(su, None, None, OPT)
            elif arm == "tensor":
                ops[arm] = fc.make_device_fast(su, None, None, OPT, Q=Q)
            else:
                X_v, w_v = arms_xw[arm]
                ops[arm] = fc.make_device_fast(su, X_v, w_v, OPT)
        ic_fast = fc.make_ic_fast(su, OPT)
        interior = su.interior
        trajs = []
        for ti in range(fc.N_TEST):
            u0 = U_test[ti, 0]
            u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
            trajs.append(dict(
                u0i=jnp.asarray(u0[interior]), nu=float(nu_test[ti]),
                tol=fc.STEP_TOL * u_scale * float(np.sqrt(su.n_i))))
        cells[N] = dict(su=su, U_test=U_test, ops=ops, ic=ic_fast,
                        trajs=trajs, base=base)
        report["cells"][str(N)] = {arm: dict(rollout=[dict(
            traj=ti, nu=trajs[ti]["nu"],
            times=dict(ic=[], roll=[], dec=[])) for ti in range(fc.N_TEST)])
            for arm in ARMS}
        log(f"  cell N={N} built (n_i={su.n_i}) [{time.time()-t0:.1f}s]")

    # ---------------- interleaved timing ----------------------------------
    last = {}
    for rep_ in range(BURN + TIME_REPS):
        order = NS if rep_ % 2 == 0 else list(reversed(NS))
        t_rep = time.time()
        for N in order:
            c = cells[N]
            su = c["su"]
            for arm in ARMS:
                roll = c["ops"][arm]["rollout"]
                for ti, tr in enumerate(c["trajs"]):
                    t0 = time.perf_counter()
                    z0, v0 = blk(c["ic"](tr["u0i"]))
                    t1 = time.perf_counter()
                    Z, rns, nJs, reasons, atts = blk(
                        roll(z0, tr["nu"], tr["tol"], fc.GN_BUDGET))
                    t2 = time.perf_counter()
                    F = blk(su.decode_all(jnp.concatenate([z0[None], Z],
                                                          axis=0)))
                    t3 = time.perf_counter()
                    if rep_ >= BURN:
                        rec = report["cells"][str(N)][arm]["rollout"][ti]
                        rec["times"]["ic"].append(t1 - t0)
                        rec["times"]["roll"].append(t2 - t1)
                        rec["times"]["dec"].append(t3 - t2)
                        last[(N, arm, ti)] = (z0, v0, Z, nJs, reasons, atts, F)
        log(f"  rep {rep_} ({'burn' if rep_ < BURN else 'timed'}, order "
            f"{'asc' if rep_ % 2 == 0 else 'desc'}) done [{time.time()-t_rep:.1f}s]")

    # ---------------- accuracy + summaries from the last timed rep ---------
    for N in NS:
        c = cells[N]
        interior = c["su"].interior
        for arm in ARMS:
            out = report["cells"][str(N)][arm]
            errs, atts_all, nJ_all, reasons_all = [], [], [], {}
            for ti in range(fc.N_TEST):
                z0, v0, Z, nJs, reasons, atts, F = last[(N, arm, ti)]
                Fn = np.asarray(F)
                e_t = [rel(Fn[t], c["U_test"][ti, t][interior])
                       for t in range(T)]
                rec = out["rollout"][ti]
                rec["err_mean"] = float(np.mean(e_t))
                rec["ic_resid"] = float(v0)
                rec["lm_attempts_total"] = int(np.sum(np.asarray(atts)))
                rec["njac_total"] = int(np.sum(np.asarray(nJs)))
                rec["stop_reasons"] = hist_of(reasons)
                rm = float(np.median(rec["times"]["roll"]) * 1e3)
                rec["roll_ms_median"] = rm
                rec["ms_per_lm_attempt"] = rm / rec["lm_attempts_total"]
                errs.append(rec["err_mean"])
                atts_all.append(rec["lm_attempts_total"])
                nJ_all.append(rec["njac_total"])
                for k_, v_ in rec["stop_reasons"].items():
                    reasons_all[k_] = reasons_all.get(k_, 0) + v_
            agg = lambda key: float(np.median(
                [x for r_ in out["rollout"] for x in r_["times"][key]]) * 1e3)
            out["ic_ms"] = agg("ic")
            out["roll_ms"] = agg("roll")
            out["dec_ms"] = agg("dec")
            out["e2e_ms"] = float(np.median(
                [a + b_ + d_ for r_ in out["rollout"]
                 for a, b_, d_ in zip(r_["times"]["ic"], r_["times"]["roll"],
                                      r_["times"]["dec"])]) * 1e3)
            out["err_mean"] = float(np.mean(errs))
            out["lm_attempts_total_mean"] = float(np.mean(atts_all))
            out["njac_total_mean"] = float(np.mean(nJ_all))
            out["ms_per_lm_attempt_median"] = float(np.median(
                [r_["ms_per_lm_attempt"] for r_ in out["rollout"]]))
            out["ms_per_lm_attempt_pooled"] = float(
                np.sum([np.median(r_["times"]["roll"]) for r_ in out["rollout"]])
                * 1e3 / np.sum(atts_all))
            out["stop_reasons"] = reasons_all
            bv = c["base"]["variants"].get(arm)
            if bv is not None:
                out["parity_err_rel_diff_vs_committed"] = \
                    abs(out["err_mean"] - bv["rollout_err_mean"]) \
                    / bv["rollout_err_mean"]
                out["committed_e2e_ms"] = bv["e2e_ms_median"]
                out["committed_roll_ms"] = bv["roll_ms_median"]
            log(f"  N={N:5d} {arm:10s} err {out['err_mean']:.6e} "
                f"(vs committed {out.get('parity_err_rel_diff_vs_committed', float('nan')):.1e})"
                f" | ic {out['ic_ms']:.2f} roll {out['roll_ms']:.2f} dec "
                f"{out['dec_ms']:.2f} e2e {out['e2e_ms']:.2f} ms | attempts "
                f"{out['lm_attempts_total_mean']:.1f} -> "
                f"{out['ms_per_lm_attempt_median']*1e3:.1f} us/attempt")
        vo = report["cells"][str(N)].get("oracle")
        vt = report["cells"][str(N)].get("tensor")
        if vo and vt:
            report["cells"][str(N)]["tensor_vs_oracle"] = dict(
                err_abs_diff_max=float(max(
                    abs(a["err_mean"] - b_["err_mean"])
                    for a, b_ in zip(vt["rollout"], vo["rollout"]))),
                stop_hist_identical=bool(vt["stop_reasons"] == vo["stop_reasons"]),
                attempts_identical=bool(
                    [r_["lm_attempts_total"] for r_ in vt["rollout"]]
                    == [r_["lm_attempts_total"] for r_ in vo["rollout"]]))
    save()

    # ---------------- slopes: log(ms) vs log(N) ---------------------------
    fits = {}
    for arm in ARMS:
        for key in ("roll_ms", "e2e_ms", "ms_per_lm_attempt_median", "ic_ms"):
            ys = [report["cells"][str(N)][arm][key] for N in NS]
            slope, icpt = np.polyfit(np.log(NS), np.log(ys), 1)
            fits[f"{arm}:{key}"] = dict(exponent=float(slope),
                                        N_range=[min(NS), max(NS)],
                                        values=ys)
            log(f"  slope log({key}) vs log(N) [{arm}, N={min(NS)}..{max(NS)}]: "
                f"{slope:+.4f}")
    report["fits"] = fits
    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
