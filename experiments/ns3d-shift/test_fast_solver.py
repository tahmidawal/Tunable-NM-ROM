"""Sub-minute local checks for the mesh-scalable operators and the fixed-iteration
solver. Everything here must hold before either is used on the cluster.

Run: PYTHONPATH=... jaxrun python test_fast_solver.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ("experiments/ns3d", "experiments/ns2d", "experiments/separable-decoder",
              "experiments/ns3d-grok"):
    sys.path.insert(0, str(ROOT / extra))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import ns3d_fom as F  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402


def centered_basis(n, cases, rank, seed=202609201):
    par = F.parameters(seed, cases)
    frames, _, _ = D.generate(par, n, 0.005, 0.05)
    for case in range(len(frames)):
        for instant in range(frames.shape[1]):
            frames[case, instant], _ = D.center_field(frames[case, instant])
    basis, _, _ = D.pod_basis(frames.reshape(len(frames) * frames.shape[1], -1), rank, 16)
    basis, _ = D.orthonormalize_prefix(basis, rank)
    return np.ascontiguousarray(basis)


def main():
    failures = []

    def check(name, value, bound):
        ok = bool(value <= bound)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {value:.3e} (bound {bound:.0e})",
              flush=True)
        if not ok:
            failures.append(name)

    # 1. the cheap test-mode enumeration is the production one
    for n, m in ((8, 32), (16, 64)):
        check(f"test_mode_ids n={n} m={m} eigenvalues", SR.check_test_mode_ids(n, m), 0.0)

    # 2. the Phi-free operator build equals the dense one
    n, rank, modes = 8, 8, 32
    basis = centered_basis(n, 8, rank)
    dense, dense_report = SR.build_operators(basis, n, modes)
    fast, fast_report = SR.build_operators_fast(basis, n, modes, check_modes=8)
    for key in ("A", "Dd", "T", "lam"):
        gap = float(np.max(np.abs(np.asarray(dense[key]) - np.asarray(fast[key])))
                    / max(float(np.max(np.abs(np.asarray(dense[key])))), 1e-300))
        check(f"operator {key} dense vs Phi-free", gap, 1e-12)
    print("  fast build report:", {k: v for k, v in fast_report.items()
                                   if isinstance(v, float)}, flush=True)

    # 3. the fixed-iteration solver reproduces the reference LM trajectory
    par = F.parameters(202609202, 2)
    dev, _, _ = D.generate(par, n, 0.005, 0.05)
    dt, horizon = 0.01, 0.05
    steps = D.nsteps_for(dt, horizon)
    reference = SR.make_shift_run(dt, steps, steps // 5, n, rank, mode="free",
                                  gauge=0.0, budget=60, gtol=1e-7)
    args = (jnp.asarray(basis), jnp.asarray(fast["A"]), jnp.asarray(fast["T"]),
            jnp.asarray(fast["lam"]), jnp.asarray(fast["Dd"]),
            jnp.zeros((rank, rank, 3)).transpose(2, 0, 1), jnp.zeros((steps, 3)))
    for case in range(len(dev)):
        u0 = jnp.asarray(dev[case, 0])
        nu = float(par[case, -1])
        want = np.asarray(reference(u0, nu, *args)[0])
        for iters in (1, 2, 3, 4, 6):
            fast_run = SR.make_frozen_run(dt, steps, steps // 5, n, rank, iters=iters,
                                          damping=1e-6, extrapolate=True, diagnose=True)
            got = np.asarray(fast_run(u0, nu, jnp.asarray(basis), jnp.asarray(fast["A"]),
                                      jnp.asarray(fast["T"]), jnp.asarray(fast["lam"]),
                                      jnp.asarray(fast["Dd"]))[0])
            gap = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300))
            label = f"parity case {case} iters={iters}"
            print(f"     {label}: {gap:.3e}", flush=True)
            if iters == 6:
                check(label, gap, 1e-8)

    # 4. diagnose=False must not change the fields
    quiet = SR.make_frozen_run(dt, steps, steps // 5, n, rank, iters=4, diagnose=False)
    loud = SR.make_frozen_run(dt, steps, steps // 5, n, rank, iters=4, diagnose=True)
    ops = (jnp.asarray(basis), jnp.asarray(fast["A"]), jnp.asarray(fast["T"]),
           jnp.asarray(fast["lam"]), jnp.asarray(fast["Dd"]))
    u0, nu = jnp.asarray(dev[0, 0]), float(par[0, -1])
    check("diagnose flag changes fields",
          float(np.max(np.abs(np.asarray(quiet(u0, nu, *ops))
                              - np.asarray(loud(u0, nu, *ops)[0])))), 0.0)

    # 5. the fixed-iteration query stays translation equivariant
    offset = np.array([0.37, -0.61, 0.94])
    plain = np.asarray(quiet(u0, nu, *ops))
    shifted = np.asarray(quiet(jnp.asarray(D.fourier_shift(np.asarray(u0), offset * n)),
                               nu, *ops))
    want = np.stack([D.fourier_shift(f, offset * n) for f in plain])
    check("equivariance of the fixed-iteration query",
          float(np.linalg.norm(shifted - want) / max(np.linalg.norm(want), 1e-300)), 1e-12)

    print(("FAILURES: " + ", ".join(failures)) if failures else "all checks passed",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
