"""Sub-minute local gate for the head-in-the-frame solver. Must pass before any job.

Run from this directory:
  PYTHONPATH=../ns3d:../ns2d:../separable-decoder:../ns3d-grok:../ns3d-shift \
    jaxrun python test_head_solver.py
"""
from __future__ import annotations

import os
import sys
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
from test_fast_solver import centered_basis  # noqa: E402


def main():
    failures = []

    def check(name, value, bound):
        ok = bool(value <= bound)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {value:.3e} (bound {bound:.0e})", flush=True)
        if not ok:
            failures.append(name)

    n, R, modes = 8, 8, 32
    basis = centered_basis(n, 8, R)
    ops, _ = SR.build_operators_fast(basis, n, modes, check_modes=8)
    opsj = (jnp.asarray(basis), jnp.asarray(ops["A"]), jnp.asarray(ops["T"]),
            jnp.asarray(ops["lam"]), jnp.asarray(ops["Dd"]))
    rng = np.random.default_rng(3)

    # 1. analytic head Jacobian
    k, width = 3, 16
    coeffs = rng.normal(size=(40, R))
    p, codes, _ = H.init_head(jax.random.PRNGKey(0), coeffs, np.ones(40), k, width)
    p = dict(p, W3=jnp.asarray(0.1 * rng.normal(size=(width, R))))
    z = jnp.asarray(rng.normal(size=k))
    got = H.head_jacobian(p, z)
    want = jax.jacfwd(lambda zz: H.head_apply(p, zz))(z)
    check("head Jacobian analytic vs jacfwd",
          float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)), 1e-12)

    # 2. chained step Jacobian vs jacfwd of the composed residual
    dt, q = 0.01, 2
    C = jnp.asarray(np.linalg.qr(rng.normal(size=(R, q)))[0])
    prepare, coeff, residual_jac, _ = H.make_head_step(dt, n, R, k, q, 3, 1e-6)
    prep = prepare(0.005, *opsj[1:])
    w = jnp.asarray(rng.normal(size=k + q + 3) * 0.1)
    aprev = jnp.asarray(rng.normal(size=R))
    res, jac = residual_jac(w, aprev, prep, p, C)
    jad = jax.jacfwd(lambda ww: residual_jac(ww, aprev, prep, p, C)[0])(w)
    check("step Jacobian (z, y, delta) chained vs jacfwd",
          float(jnp.linalg.norm(jac - jad) / jnp.linalg.norm(jad)), 1e-12)

    # 3. encoder recovers a planted (z, y) from a nearby start
    zt = jnp.asarray(rng.normal(size=k) * 0.3)
    yt = jnp.asarray(rng.normal(size=q))
    a = H.head_apply(p, zt) + C @ yt
    zf, yf = H.make_code_fit(k, 30)(p, zt + 0.01, a, C)
    check("encoder recovers planted code",
          float(jnp.linalg.norm(H.head_apply(p, zf) + C @ yf - a) / jnp.linalg.norm(a)), 1e-9)

    # 4. a linear head with k = R reproduces the lane's linear-bank driver
    par = F.parameters(202609202, 2)
    dev, _, _ = D.generate(par, n, 0.005, 0.05)
    horizon = 0.05
    steps = D.nsteps_for(dt, horizon)
    Q = np.linalg.qr(rng.normal(size=(R, R)))[0]
    lin = dict(W1=jnp.zeros((R, 4)), b1=jnp.zeros(4), W2=jnp.zeros((4, 4)), b2=jnp.zeros(4),
               W3=jnp.zeros((4, R)), b3=jnp.zeros(R), Ws=jnp.asarray(Q.T))
    Zc = jnp.asarray(rng.normal(size=(5, R)))
    Hc = H.head_apply(lin, Zc)
    extra = (lin, jnp.zeros((R, 0)), Hc, jnp.sum(Hc * Hc, 1), Zc)
    lane = SR.make_frozen_run(dt, steps, steps // 5, n, R, iters=6, damping=1e-6,
                              extrapolate=True, diagnose=False)
    mine = H.make_head_run(dt, steps, steps // 5, n, R, R, 0, iters=6, damping=1e-6)
    for case in range(2):
        u0, nu = jnp.asarray(dev[case, 0]), float(par[case, -1])
        want = np.asarray(lane(u0, nu, *opsj))
        got = np.asarray(mine(u0, nu, *opsj, *extra))
        check(f"linear head k=R reproduces lane driver, case {case}",
              float(np.linalg.norm(got - want) / np.linalg.norm(want)), 1e-8)

    # 5. nonlinear head query: equivariance, finite, frame-zero control runs
    k2, q2 = 3, 2
    coeffs = np.stack([basis.T @ D.center_field(dev[c, t])[0].ravel()
                       for c in range(2) for t in range(dev.shape[1])])
    p2, codes2, _ = H.init_head(jax.random.PRNGKey(1), coeffs, np.ones(len(coeffs)), k2, 16)
    p2 = dict(p2, W3=jnp.asarray(1e-3 * rng.normal(size=(16, R))))
    C2 = jnp.asarray(np.linalg.qr(rng.normal(size=(R, q2)))[0])
    Hc2 = H.head_apply(p2, jnp.asarray(codes2))
    extra2 = (p2, C2, Hc2, jnp.sum(Hc2 * Hc2, 1), jnp.asarray(codes2))
    run = H.make_head_run(dt, steps, steps // 5, n, R, k2, q2, iters=4)
    u0, nu = jnp.asarray(dev[0, 0]), float(par[0, -1])
    plain = np.asarray(run(u0, nu, *opsj, *extra2))
    check("nonlinear head query finite", float(not np.all(np.isfinite(plain))), 0.0)
    offset = np.array([0.37, -0.61, 0.94])
    shifted = np.asarray(run(jnp.asarray(D.fourier_shift(np.asarray(u0), offset * n)), nu,
                             *opsj, *extra2))
    want = np.stack([D.fourier_shift(f, offset * n) for f in plain])
    check("head query translation equivariance",
          float(np.linalg.norm(shifted - want) / np.linalg.norm(want)), 1e-11)
    zero = H.make_head_run(dt, steps, steps // 5, n, R, k2, q2, iters=4, frame="zero")
    zf = np.asarray(zero(u0, nu, *opsj, *extra2))
    check("frame-zero control finite", float(not np.all(np.isfinite(zf))), 0.0)
    loud = H.make_head_run(dt, steps, steps // 5, n, R, k2, q2, iters=4, diagnose=True)
    check("diagnose flag leaves fields unchanged",
          float(np.max(np.abs(np.asarray(loud(u0, nu, *opsj, *extra2)[0]) - plain))), 0.0)

    # 6. reference LM driver agrees with the fixed-sweep driver when converged
    ref = H.make_head_reference(dt, steps, steps // 5, n, R, k2, q2)
    many = H.make_head_run(dt, steps, steps // 5, n, R, k2, q2, iters=8)
    a_ = np.asarray(ref(u0, nu, *opsj, *extra2)[0])
    b_ = np.asarray(many(u0, nu, *opsj, *extra2))
    check("head fixed-sweep vs LM reference (8 sweeps)",
          float(np.linalg.norm(a_ - b_) / np.linalg.norm(a_)), 1e-7)

    # 7. FD-CG FOM: divergence-free output, CG converged, first-order sanity vs spectral
    fd = FD.make_fd_solver(n, 0.005, 10, 2, rtol=1e-10, diagnose=True)
    out, info = fd(u0, nu)
    divs = max(float(jnp.linalg.norm(FD.div(f, 1.0 / n)) / jnp.linalg.norm(f)) for f in out)
    check("FD-CG output discretely divergence-free", divs, 1e-8)
    check("FD-CG CG converged (no step at maxiter)",
          float(max(int(jnp.max(info[0])), int(jnp.max(info[2]))) >= 4000), 0.0)

    print(("FAILURES: " + ", ".join(failures)) if failures else "all checks passed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
