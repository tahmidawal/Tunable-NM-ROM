"""Audited FOM and measurement machinery for the Burgers hybrid redesign.

The counting BiCGStab/Newton implementation is copied from the independently
reviewed ``rom-warmstart-fom/wsf_burgers.py`` harness. The PDE residual itself is
never reimplemented: it is imported from ``burgers2d_film.make_rollout(n)``.
"""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_BF_CANDIDATES = (
    os.path.join(
        _WT_ROOT,
        "2026-08-14-burgers2d-coord-rom",
        "experiments",
        "burgers2d-coord-rom",
    ),
    os.path.join(HERE, "deps", "burgers2d-coord-rom"),
)
for _candidate in _BF_CANDIDATES:
    if os.path.isfile(os.path.join(_candidate, "burgers2d_film.py")):
        import sys

        sys.path.insert(0, _candidate)
        break
else:
    raise ImportError("cannot locate burgers2d_film.py")

# Keep the reference architecture defaults stable at import time. Only its FOM
# routines are used by this cell.
os.environ.setdefault("HIDDEN", "256")
import burgers2d_film as bf  # noqa: E402

F64 = jnp.float64
T = bf.NUM_STEPS
MAX_NEWTON = int(os.environ.get("MAX_NEWTON", "25"))
LIN_TOL = float(os.environ.get("LIN_TOL", str(bf.LIN_TOL)))
LIN_MAXITER = int(os.environ.get("LIN_MAXITER", str(bf.LIN_MAXITER)))


def log(*args):
    print(*args, flush=True)


def time_fn(fn, reps=7, warm=2):
    """Median wall time and the complete repetition array after warm-up."""
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append(float(time.perf_counter() - start))
    return float(np.median(samples)), samples


def gpu_burn(fn, seconds=3.0):
    """Bring the device out of its idle clock state before a timed block."""
    start = time.perf_counter()
    count = 0
    while time.perf_counter() - start < seconds:
        fn()
        count += 1
    return count


def provenance():
    """Content-hashed provenance; cluster commit comes from the batch wrapper."""
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", HERE, *args], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    env_commit = os.environ.get("BH_COMMIT", "")
    hashes = {}
    for path in sorted(glob.glob(os.path.join(HERE, "bh_*.py"))):
        with open(path, "rb") as handle:
            hashes[os.path.basename(path)] = hashlib.sha256(handle.read()).hexdigest()
    dev = jax.devices()[0]
    return {
        "commit": env_commit or git("rev-parse", "HEAD"),
        "commit_source": "env:BH_COMMIT" if env_commit else "git",
        "source_sha256": hashes,
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
        "gpu": str(dev),
        "gpu_kind": getattr(dev, "device_kind", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "host": os.environ.get("HOSTNAME", os.uname().nodename),
    }


def make_bicgstab(tol=LIN_TOL, maxiter=LIN_MAXITER):
    """Counting BiCGStab for a Newton correction, including alpha half-step."""
    def bicgstab(A, b):
        bn = jnp.linalg.norm(b)
        threshold = tol * bn
        x = jnp.zeros_like(b)
        r = b
        rhat = b
        rho = jnp.asarray(1.0, F64)
        alpha = jnp.asarray(1.0, F64)
        omega = jnp.asarray(1.0, F64)
        v = jnp.zeros_like(b)
        p = jnp.zeros_like(b)
        tiny = 1e-300
        bad0 = ~(jnp.all(jnp.isfinite(b)) & jnp.isfinite(bn))

        def cond(state):
            _, rr, _, _, _, _, _, k, _, bad = state
            return (jnp.linalg.norm(rr) > threshold) & (k < maxiter) & (~bad)

        def body(state):
            x, r, p, v, rho, alpha, omega, k, matvecs, bad = state
            rho_new = jnp.sum(rhat * r)
            breakdown = (jnp.abs(rho_new) < tiny) | (jnp.abs(omega) < tiny)
            beta = (rho_new / jnp.where(breakdown, 1.0, rho)) * (
                alpha / jnp.where(breakdown, 1.0, omega)
            )
            p_new = r + beta * (p - omega * v)
            v_new = A(p_new)
            rv = jnp.sum(rhat * v_new)
            breakdown = breakdown | (jnp.abs(rv) < tiny)
            alpha_new = rho_new / jnp.where(breakdown, 1.0, rv)
            s = r - alpha_new * v_new
            s_converged = jnp.linalg.norm(s) <= threshold
            t = A(s)
            tt = jnp.sum(t * t)
            breakdown = breakdown | ((tt < tiny) & (~s_converged))
            omega_new = jnp.where(
                s_converged,
                0.0,
                jnp.sum(t * s) / jnp.where(breakdown | s_converged, 1.0, tt),
            )
            x_new = x + alpha_new * p_new + omega_new * s
            r_new = s - omega_new * t
            breakdown = breakdown | ~(
                jnp.all(jnp.isfinite(x_new)) & jnp.all(jnp.isfinite(r_new))
            )
            keep = lambda new, old: jnp.where(breakdown, old, new)
            return (
                keep(x_new, x),
                keep(r_new, r),
                keep(p_new, p),
                keep(v_new, v),
                keep(rho_new, rho),
                keep(alpha_new, alpha),
                keep(omega_new, omega),
                k + 1,
                matvecs + 2,
                breakdown,
            )

        x, r, p, v, rho, alpha, omega, k, matvecs, bad = jax.lax.while_loop(
            cond,
            body,
            (
                x,
                r,
                p,
                v,
                rho,
                alpha,
                omega,
                jnp.int32(0),
                jnp.int32(0),
                bad0,
            ),
        )
        flag = jnp.where(
            bad,
            jnp.int32(2),
            jnp.where(jnp.linalg.norm(r) > threshold, jnp.int32(1), jnp.int32(0)),
        )
        return x, k, matvecs, flag

    return bicgstab


def make_chain(n, tol_rel):
    """One FOM chain for previous, extrapolated, or supplied guesses.

    ``mode`` is a traced runtime integer: 0 previous state, 1 linear
    extrapolation, 2 supplied guess array. All arms therefore share the same
    operator, Newton stopping test, linear solver, and compiled executable.
    """
    _, residual = bf.make_rollout(n)
    bicg = make_bicgstab()

    def step(u_prev, u_prev2, guess, mode, nu):
        u_start = jnp.where(
            mode == 0,
            u_prev,
            jnp.where(mode == 1, 2.0 * u_prev - u_prev2, guess),
        )
        tol_abs = tol_rel * jnp.linalg.norm(u_prev)
        r0 = residual(u_start, u_prev, nu)
        rn0 = jnp.linalg.norm(r0)
        init_flag = jnp.where(jnp.isfinite(rn0), jnp.int32(0), jnp.int32(3))
        init = (
            u_start,
            r0,
            rn0,
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            init_flag,
        )

        def cond(state):
            _, _, rn, k, _, _, _, flag = state
            return (rn > tol_abs) & (k < MAX_NEWTON) & (flag == 0)

        def body(state):
            u, r, rn, k, nlin, nbreak, nlinmax, flag = state
            Jv = lambda vec: jax.jvp(
                lambda uu: residual(uu, u_prev, nu), (u,), (vec,)
            )[1]
            du, linear_iters, _, linear_flag = bicg(Jv, -r)
            finite_step = jnp.all(jnp.isfinite(du))
            u2 = jnp.where(finite_step, u + du, u)
            r2 = residual(u2, u_prev, nu)
            rn2 = jnp.linalg.norm(r2)
            finite_residual = jnp.isfinite(rn2)
            new_flag = jnp.where(
                ~finite_step,
                jnp.int32(2),
                jnp.where(
                    ~finite_residual,
                    jnp.int32(3),
                    jnp.where(linear_flag != 0, jnp.int32(4), jnp.int32(0)),
                ),
            )
            continuation_flag = jnp.where(new_flag == 4, jnp.int32(0), new_flag)
            return (
                jnp.where(finite_residual, u2, u),
                jnp.where(finite_residual, r2, r),
                jnp.where(finite_residual, rn2, rn),
                k + 1,
                nlin + linear_iters,
                nbreak + (linear_flag == 2).astype(jnp.int32),
                nlinmax + (linear_flag == 1).astype(jnp.int32),
                continuation_flag,
            )

        u, r, rn, k, nlin, nbreak, nlinmax, flag = jax.lax.while_loop(
            cond, body, init
        )
        flag = jnp.where(
            (flag == 0) & (rn > tol_abs),
            jnp.int32(1),
            jnp.where(
                (flag == 0) & ((nbreak > 0) | (nlinmax > 0)), jnp.int32(4), flag
            ),
        )
        rel_res = rn / jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)
        return u, k, nlin, nbreak + nlinmax, flag, rel_res

    def chain(u0, nu, guesses, mode):
        def body(carry, guess):
            u_prev, u_prev2 = carry
            out = step(u_prev, u_prev2, guess, mode, nu)
            u = out[0]
            return (u, u_prev), out

        _, (U, newton, linear, breakdowns, flags, rel_res) = jax.lax.scan(
            body, (u0, u0), guesses
        )
        return U, newton, linear, breakdowns, flags, rel_res

    return jax.jit(chain), residual


def generate_reference(n, trajectory_indices, test_seed):
    """Regenerate fresh test trajectories from seed with the reference FOM."""
    count = max(trajectory_indices) + 1
    cx, cy, width, amp, nu, _ = bf.sample_params(seed=test_seed, m=count)
    rollout, _ = bf.make_rollout(n)
    trajectories = []
    for idx in trajectory_indices:
        u0 = bf.blob_ic(n, cx[idx], cy[idx], width[idx], amp[idx])
        snaps, residuals = rollout(jnp.asarray(u0)[None], jnp.asarray([nu[idx]]))
        U = np.asarray(snaps)[:, 0]
        max_residual = float(jnp.max(residuals))
        trajectories.append(
            {
                "index": int(idx),
                "U": U,
                "nu": float(nu[idx]),
                "max_reference_newton_residual": max_residual,
                "parameters": {
                    "cx": float(cx[idx]),
                    "cy": float(cy[idx]),
                    "width": float(width[idx]),
                    "amplitude": float(amp[idx]),
                    "nu": float(nu[idx]),
                },
            }
        )
    return trajectories


def extrapolated_guesses(U):
    guesses = np.empty_like(U[1:])
    guesses[0] = U[0]
    guesses[1:] = 2.0 * U[1:-1] - U[:-2]
    return guesses


def guess_diagnostics(U, guesses, residual, nu):
    """Per-step L2 error and FOM residual of a supplied guess stream."""
    Uj = jnp.asarray(U)
    Gj = jnp.asarray(guesses)
    err = jax.vmap(
        lambda g, truth: jnp.linalg.norm(g - truth)
        / jnp.maximum(jnp.linalg.norm(truth), 1e-300)
    )(Gj, Uj[1:])
    rr = jax.vmap(
        lambda g, up: jnp.linalg.norm(residual(g, up, nu))
        / jnp.maximum(jnp.linalg.norm(up), 1e-300)
    )(Gj, Uj[:-1])
    return np.asarray(err), np.asarray(rr)

