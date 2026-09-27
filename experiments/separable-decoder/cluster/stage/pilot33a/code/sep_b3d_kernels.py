"""Burgers 3D — gate C1: the three checkpoints' latent KERNELS on ONE GPU
(design r3 [A27, A43]).  Kernel-only: A, lambda, Q and the head of each
per-N job (its *_kernel.npz), run from that job's own IC latents of the eight
test trajectories, interleaved (reps outermost, kernels forward on even reps
and reversed on odd), TIME_REPS timed repetitions after BURN.  No bank, no
grid, no IC fit, no decode: this is the reduced arithmetic whose cost the
design claims is N-independent, and it is labelled as such.

    KERNELS=path_n33.npz,path_n65.npz,path_n129.npz OUT=... python sep_b3d_kernels.py
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

import b3d_common as b3
from sep_b3d_tensor import make_step_aux, make_roll_aux

F64 = jnp.float64
KERNELS = os.environ["KERNELS"].split(",")
OUT = os.environ.get("OUT", "/tmp/sep_b3d_kernels.json")
TIME_REPS = int(os.environ.get("TIME_REPS", "5"))
BURN = int(os.environ.get("BURN", "2"))
FLAT_MAX_RATIO = float(os.environ.get("FLAT_MAX_RATIO", "1.25"))   # design r4 [A58]
log = b3.log


def load_kernel(path):
    d = np.load(path)
    hw = [d[f"h{i}_w"] for i in range(10) if f"h{i}_w" in d.files]
    hb = [d[f"h{i}_b"] for i in range(10) if f"h{i}_b" in d.files]
    params = dict(h=[(jnp.asarray(w), jnp.asarray(b)) for w, b in zip(hw, hb)],
                  h_lin=jnp.asarray(d["h_lin"] if "h_lin" in d.files else d["params_h_lin"]))
    return d, params


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B3D-KERNELS {KERNELS}")
    report = dict(config=dict(kernels=KERNELS, time_reps=TIME_REPS, burn=BURN, flat_max_ratio=FLAT_MAX_RATIO,
                              backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
                              jax_version=jax.__version__, x64=True,
                              matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                              slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME", "local"),
                              label="KERNEL-ONLY: latent LM rollout with A, lambda, Q, head; no bank, IC fit or decode"),
                  kernels={}, complete=False)
    ks = []
    for path in KERNELS:
        d, params = load_kernel(path)
        N, K, R, M = int(d["N"]), int(d["K"]), int(d["R"]), int(d["M"])
        A = jnp.asarray(d["A"]); lam = jnp.asarray(d["lam"]); Q = jnp.asarray(d["Q"])
        Qm = Q.reshape(M * R, R)
        h_fn = lambda z, p=params: b3.head(p, z)

        def r_w(z, prev_m, nu, aux, A=A, lam=lam, Qm=Qm, h_fn=h_fn, M=M, R=R):
            w_ = (1.0 + b3.DT * nu * lam) ** (-b3.WEAK_ALPHA)
            h = h_fn(z)
            Ah = A @ h
            q = 0.5 * ((Qm @ h).reshape(M, R) @ h)
            return w_ * ((Ah - prev_m) + b3.DT * (q + nu * lam * Ah))
        step, rn_j, rJ_j = make_step_aux(r_w, K, float(d["stall"]), float(d["TRD"]))
        roll = make_roll_aux(step, rn_j, lambda z, A=A, h_fn=h_fn: A @ h_fn(z), b3.NUM_STEPS, float(d["extrap"]))
        ks.append(dict(N=N, K=K, R=R, M=M, roll=roll, z0=jnp.asarray(d["z0"]), nu=np.asarray(d["nu"]),
                       tol=np.asarray(d["tol_abs"]), budget=int(d["gn_budget"]), path=path))
        log(f"  kernel N={N}: K={K} R={R} M={M}, {len(d['z0'])} trajectories")
    assert len(ks) == 3, "C1 needs exactly the three per-N kernels"
    assert len({(k["K"], k["R"], k["M"]) for k in ks}) == 1, "kernels must share (K, R, M)"
    assert all(len(k["nu"]) == len(ks[0]["nu"]) == 8 for k in ks), "eight test trajectories per kernel"
    n_test = len(ks[0]["nu"])
    times = {k["N"]: [[] for _ in range(n_test)] for k in ks}
    b3.burn_in(1.5)
    for rep_ in range(BURN + TIME_REPS):
        order = ks if rep_ % 2 == 0 else list(reversed(ks))
        for i in range(n_test):
            for k in order:
                us = jnp.full((b3.NUM_STEPS,), float(k["tol"][i]), dtype=F64)
                t0 = time.perf_counter()
                out = jax.block_until_ready(k["roll"](k["z0"][i], float(k["nu"][i]), us, k["budget"], ()))
                dt = time.perf_counter() - t0
                if rep_ >= BURN:
                    times[k["N"]][i].append(dt)
                    if rep_ == BURN:
                        report["kernels"].setdefault(str(k["N"]), {})[f"traj{i}_attempts"] = int(jnp.sum(out[3]))
    meds = {}
    for k in ks:
        allv = [x * 1e3 for i in range(n_test) for x in times[k["N"]][i]]
        meds[k["N"]] = float(np.median(allv))
        report["kernels"][str(k["N"])].update(dict(K=k["K"], R=k["R"], M=k["M"], kernel_ms_median=meds[k["N"]],
                                                   per_traj_ms=[float(np.median(times[k["N"]][i]) * 1e3) for i in range(n_test)],
                                                   raw_s={str(i): [float(x) for x in times[k["N"]][i]] for i in range(n_test)}))
        log(f"  N={k['N']}: kernel {meds[k['N']]:.2f} ms/trajectory (median over {n_test} traj x {TIME_REPS} reps)")
    ratio = max(meds.values()) / min(meds.values())
    report["C1"] = dict(max_over_min=float(ratio), passed=bool(ratio <= FLAT_MAX_RATIO), rule=f"max/min median kernel ms <= {FLAT_MAX_RATIO}")
    log(f"  C1 kernel flatness: max/min {ratio:.3f} -> {'PASS' if ratio <= FLAT_MAX_RATIO else 'FAIL'}")
    report["complete"] = bool(ratio <= FLAT_MAX_RATIO)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1, default=float)
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
