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


def make_chain(n, tol_rel, predictor=None, lin_tol=None, preconditioner="none",
               oracle_quality=0.0):
    """One FOM chain for previous, extrapolated, or supplied guesses.

    ``mode`` is a traced runtime integer: 0 previous state, 1 linear
    extrapolation, 2 supplied guess array, 3 a dynamically constructed guess
    from ``predictor(u_prev,u_prev2,nu)``, 4 quadratic history, and 5 cubic
    history. Mode 6 is linear extrapolation plus a supplied correction field.
    Polynomial arms use lower-order startup fallbacks.
    All arms therefore share the same operator, Newton stopping test, linear
    solver, and compiled executable. Mode 7 is the diagnostic dynamic oracle
    ``base + q*(exact_next-base)`` with ``q=oracle_quality`` and the exact next
    reference supplied in ``guess``.
    """
    _, residual = bf.make_rollout(n)
    effective_lin_tol = LIN_TOL if lin_tol is None else float(lin_tol)
    bicg = make_bicgstab(tol=effective_lin_tol)
    if preconditioner not in ("none", "jacobi", "helmholtz"):
        raise ValueError(f"unknown preconditioner {preconditioner}")
    dx = 1.0 / (n - 1)
    interior_size = n - 2
    frequencies = jnp.arange(1, interior_size + 1, dtype=F64)
    helmholtz_eigenvalues = (4.0 / dx**2) * (
        jnp.sin(jnp.pi * frequencies[:, None] / (2.0 * (n - 1))) ** 2
        + jnp.sin(jnp.pi * frequencies[None, :] / (2.0 * (n - 1))) ** 2
    )

    def dst1(values, axis):
        """Unnormalised DST-I implemented by an odd FFT extension."""
        zeros_shape = list(values.shape)
        zeros_shape[axis] = 1
        zeros = jnp.zeros(zeros_shape, values.dtype)
        extension = jnp.concatenate(
            (zeros, values, zeros, -jnp.flip(values, axis=axis)), axis=axis
        )
        transformed = -jnp.fft.fft(extension, axis=axis).imag
        index = [slice(None)] * values.ndim
        index[axis] = slice(1, values.shape[axis] + 1)
        return transformed[tuple(index)]

    def dst2(values):
        return dst1(dst1(values, 0), 1)

    def helmholtz_solve(vector, nu):
        """Apply (I-dt*nu*Laplacian_D)^-1 with exact Dirichlet rows."""
        rhs = vector.reshape(n, n)
        coupling = bf.DT * nu / dx**2
        adjusted = rhs[1:-1, 1:-1]
        adjusted = adjusted.at[0, :].add(coupling * rhs[0, 1:-1])
        adjusted = adjusted.at[-1, :].add(coupling * rhs[-1, 1:-1])
        adjusted = adjusted.at[:, 0].add(coupling * rhs[1:-1, 0])
        adjusted = adjusted.at[:, -1].add(coupling * rhs[1:-1, -1])
        coefficients = dst2(adjusted)
        solution_interior = dst2(
            coefficients / (1.0 + bf.DT * nu * helmholtz_eigenvalues)
        ) / (4.0 * (interior_size + 1) ** 2)
        solution = rhs.at[1:-1, 1:-1].set(solution_interior)
        return solution.reshape(-1)

    def jacobi_diagonal(u, nu):
        field = u.reshape(n, n)
        center = field[1:-1, 1:-1]
        xm = field[:-2, 1:-1]
        xp = field[2:, 1:-1]
        ym = field[1:-1, :-2]
        yp = field[1:-1, 2:]
        adv_diagonal = jnp.where(
            center > 0.0,
            (4.0 * center - xm - ym) / dx,
            (xp + yp - 4.0 * center) / dx,
        )
        diagonal = jnp.ones((n, n), F64)
        diagonal = diagonal.at[1:-1, 1:-1].set(
            1.0 + bf.DT * (adv_diagonal + 4.0 * nu / dx**2)
        )
        # Avoid turning an unusual advection-dominated negative diagonal into a
        # division breakdown. This safeguard is reported through solver health.
        diagonal = jnp.where(jnp.abs(diagonal) > 1e-12, diagonal, 1.0)
        return diagonal.reshape(-1)

    dynamic_predictor = predictor or (
        lambda u_prev, u_prev2, u_prev3, u_prev4, nu, step_index: u_prev
    )

    def step(u_prev, u_prev2, u_prev3, u_prev4, guess, step_index, mode, nu):
        def quadratic(args):
            up, up2, up3, _, _, index = args
            return jax.lax.cond(
                index == 0,
                lambda: up,
                lambda: jax.lax.cond(
                    index == 1,
                    lambda: 2.0 * up - up2,
                    lambda: 3.0 * up - 3.0 * up2 + up3,
                ),
            )

        def cubic(args):
            up, up2, up3, up4, _, index = args
            return jax.lax.cond(
                index < 2,
                lambda: jax.lax.cond(index == 0, lambda: up,
                                     lambda: 2.0 * up - up2),
                lambda: jax.lax.cond(
                    index == 2,
                    lambda: 3.0 * up - 3.0 * up2 + up3,
                    lambda: 4.0 * up - 6.0 * up2 + 4.0 * up3 - up4,
                ),
            )

        args = (u_prev, u_prev2, u_prev3, u_prev4, guess, step_index)
        u_start = jax.lax.switch(
            mode,
            (
                lambda values: values[0],
                lambda values: 2.0 * values[0] - values[1],
                lambda values: values[4],
                lambda values: dynamic_predictor(
                    values[0], values[1], values[2], values[3], nu, values[5]
                ),
                quadratic,
                cubic,
                lambda values: jax.lax.cond(
                    values[5] == 0,
                    lambda: values[0] + values[4],
                    lambda: 2.0 * values[0] - values[1] + values[4],
                ),
                lambda values: (
                    jax.lax.cond(
                        values[5] == 0,
                        lambda: values[0],
                        lambda: 2.0 * values[0] - values[1],
                    )
                    + oracle_quality
                    * (
                        values[4]
                        - jax.lax.cond(
                            values[5] == 0,
                            lambda: values[0],
                            lambda: 2.0 * values[0] - values[1],
                        )
                    )
                ),
            ),
            args,
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
            Jv_raw = lambda vec: jax.jvp(
                lambda uu: residual(uu, u_prev, nu), (u,), (vec,)
            )[1]
            if preconditioner == "jacobi":
                diagonal = jacobi_diagonal(u, nu)
                Jv = lambda vec: Jv_raw(vec) / diagonal
                rhs = -r / diagonal
            elif preconditioner == "helmholtz":
                Jv = lambda vec: helmholtz_solve(Jv_raw(vec), nu)
                rhs = helmholtz_solve(-r, nu)
            else:
                Jv = Jv_raw
                rhs = -r
            du, linear_iters, _, linear_flag = bicg(Jv, rhs)
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
        def body(carry, inputs):
            u_prev, u_prev2, u_prev3, u_prev4 = carry
            guess, step_index = inputs
            out = step(
                u_prev, u_prev2, u_prev3, u_prev4, guess, step_index, mode, nu
            )
            u = out[0]
            return (u, u_prev, u_prev2, u_prev3), out

        _, (U, newton, linear, breakdowns, flags, rel_res) = jax.lax.scan(
            body,
            (u0, u0, u0, u0),
            (guesses, jnp.arange(T, dtype=jnp.int32)),
        )
        return U, newton, linear, breakdowns, flags, rel_res

    return jax.jit(chain), residual


def reference_equivalence(n, trajectories, chain, lin_tol=LIN_TOL,
                          preconditioner="none"):
    """Counting-chain equivalence to the testbed and JAX BiCGStab.

    The previous-state arm is compared against the testbed fixed-8-Newton
    rollout on every supplied trajectory. One representative Newton correction
    is separately checked against ``jax.scipy.sparse.linalg.bicgstab`` under the
    same optional left-Jacobi transformation.
    """
    rollout, residual = bf.make_rollout(n)
    dummy = jnp.zeros((T, n * n), F64)
    per_trajectory = []
    for trajectory in trajectories:
        U_ref_j, reference_residuals = rollout(
            jnp.asarray(trajectory["U"][0])[None],
            jnp.asarray([trajectory["nu"]]),
        )
        U_ref = np.asarray(U_ref_j)[1:, 0]
        U_count, newton, linear, breakdowns, flags, rel_res = chain(
            jnp.asarray(trajectory["U"][0]), trajectory["nu"], dummy, jnp.int32(0)
        )
        U_count = np.asarray(U_count)
        step_error = np.linalg.norm(U_count - U_ref, axis=1) / np.maximum(
            np.linalg.norm(U_ref, axis=1), 1e-300
        )
        per_trajectory.append({
            "trajectory_index": trajectory["index"],
            "trajectory_rel_difference": float(
                np.linalg.norm(U_count - U_ref) / np.linalg.norm(U_ref)
            ),
            "max_step_rel_difference": float(np.max(step_error)),
            "testbed_max_rel_newton_residual": float(jnp.max(reference_residuals)),
            "counting_max_rel_newton_residual": float(jnp.max(rel_res)),
            "newton_total": int(jnp.sum(newton)),
            "linear_total": int(jnp.sum(linear)),
            "breakdowns": int(jnp.sum(breakdowns)),
            "flags_nonzero": int(jnp.sum(flags != 0)),
        })

    trajectory = trajectories[0]
    u_prev = jnp.asarray(trajectory["U"][0])
    nu = trajectory["nu"]
    r = residual(u_prev, u_prev, nu)
    Jv_raw = lambda vec: jax.jvp(
        lambda uu: residual(uu, u_prev, nu), (u_prev,), (vec,)
    )[1]
    if preconditioner == "jacobi":
        dx = 1.0 / (n - 1)
        field = u_prev.reshape(n, n)
        center = field[1:-1, 1:-1]
        xm, xp = field[:-2, 1:-1], field[2:, 1:-1]
        ym, yp = field[1:-1, :-2], field[1:-1, 2:]
        adv = jnp.where(center > 0.0, (4.0 * center - xm - ym) / dx,
                        (xp + yp - 4.0 * center) / dx)
        diag = jnp.ones((n, n), F64).at[1:-1, 1:-1].set(
            1.0 + bf.DT * (adv + 4.0 * nu / dx**2)
        ).reshape(-1)
        diag = jnp.where(jnp.abs(diag) > 1e-12, diag, 1.0)
        op = lambda vec: Jv_raw(vec) / diag
        rhs = -r / diag
    elif preconditioner == "helmholtz":
        # Reconstruct the exact Dirichlet Helmholtz inverse used by the chain
        # so this independent JAX-BiCGStab check has the identical operator.
        dx = 1.0 / (n - 1)
        m = n - 2
        freq = jnp.arange(1, m + 1, dtype=F64)
        eigenvalues = (4.0 / dx**2) * (
            jnp.sin(jnp.pi * freq[:, None] / (2.0 * (n - 1))) ** 2
            + jnp.sin(jnp.pi * freq[None, :] / (2.0 * (n - 1))) ** 2
        )

        def dst_axis(values, axis):
            zeros_shape = list(values.shape)
            zeros_shape[axis] = 1
            zeros = jnp.zeros(zeros_shape, values.dtype)
            extension = jnp.concatenate(
                (zeros, values, zeros, -jnp.flip(values, axis=axis)), axis=axis
            )
            transformed = -jnp.fft.fft(extension, axis=axis).imag
            index = [slice(None)] * values.ndim
            index[axis] = slice(1, values.shape[axis] + 1)
            return transformed[tuple(index)]

        def solve_helmholtz(vector):
            field = vector.reshape(n, n)
            coupling = bf.DT * nu / dx**2
            adjusted = field[1:-1, 1:-1]
            adjusted = adjusted.at[0, :].add(coupling * field[0, 1:-1])
            adjusted = adjusted.at[-1, :].add(coupling * field[-1, 1:-1])
            adjusted = adjusted.at[:, 0].add(coupling * field[1:-1, 0])
            adjusted = adjusted.at[:, -1].add(coupling * field[1:-1, -1])
            coefficients = dst_axis(dst_axis(adjusted, 0), 1)
            interior = dst_axis(
                dst_axis(coefficients / (1.0 + bf.DT * nu * eigenvalues), 0), 1
            ) / (4.0 * (m + 1) ** 2)
            return field.at[1:-1, 1:-1].set(interior).reshape(-1)

        op = lambda vec: solve_helmholtz(Jv_raw(vec))
        rhs = solve_helmholtz(-r)
    else:
        op, rhs = Jv_raw, -r
    ours, ours_iterations, ours_matvecs, ours_flag = make_bicgstab(
        lin_tol, LIN_MAXITER
    )(op, rhs)
    reference, _ = jax.scipy.sparse.linalg.bicgstab(
        op, rhs, tol=lin_tol, maxiter=LIN_MAXITER
    )
    ours_residual = float(
        jnp.linalg.norm(op(ours) - rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1e-300)
    )
    reference_residual = float(
        jnp.linalg.norm(op(reference) - rhs)
        / jnp.maximum(jnp.linalg.norm(rhs), 1e-300)
    )
    return {
        "per_trajectory": per_trajectory,
        "linear_solver": {
            "preconditioner": preconditioner,
            "lin_tol": lin_tol,
            "relative_solution_difference_vs_jax": float(
                jnp.linalg.norm(ours - reference)
                / jnp.maximum(jnp.linalg.norm(reference), 1e-300)
            ),
            "ours_relative_residual": ours_residual,
            "jax_relative_residual": reference_residual,
            "ours_iterations": int(ours_iterations),
            "ours_matvecs": int(ours_matvecs),
            "ours_flag": int(ours_flag),
        },
        "max_step_rel_difference": max(
            item["max_step_rel_difference"] for item in per_trajectory
        ),
        "max_trajectory_rel_difference": max(
            item["trajectory_rel_difference"] for item in per_trajectory
        ),
    }


def test_modes(n, count):
    """Lowest unit-norm discrete sine modes and Laplacian eigenvalues."""
    dx = 1.0 / (n - 1)
    frequencies = np.arange(1, n - 1)
    kx_all, ky_all = np.meshgrid(frequencies, frequencies, indexing="ij")
    eigenvalues_all = (4.0 / dx**2) * (
        np.sin(np.pi * kx_all / (2 * (n - 1))) ** 2
        + np.sin(np.pi * ky_all / (2 * (n - 1))) ** 2
    )
    order = np.argsort(eigenvalues_all.reshape(-1), kind="stable")[:count]
    kx = kx_all.reshape(-1)[order]
    ky = ky_all.reshape(-1)[order]
    xi = frequencies / (n - 1)
    sx = np.sin(np.pi * np.outer(xi, kx))
    sy = np.sin(np.pi * np.outer(xi, ky))
    phi = (sx[:, None, :] * sy[None, :, :]).reshape(-1, count)
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)
    return phi, eigenvalues_all.reshape(-1)[order]


def make_full_weak_history_predictor(n, mode_count=16, reduced_iters=2,
                                     trust_radius=0.5, alpha_min=0.0,
                                     alpha_max=1.5):
    """One-dimensional weak-LSPG predictor on the history-line manifold.

    The per-step reduced manifold is

        u(alpha) = u_n + alpha * (u_n - u_{n-1}).

    Linear extrapolation is alpha=1. Starting there, a few scalar
    Gauss--Newton iterations minimise the exact-FOM residual projected onto
    smooth sine test modes. This is the full-grid diagnostic version; it obeys
    the weak-form rule and keeps the FOM's exact upwind advection. A later EQ
    version can remove its grid projection cost without changing the manifold.
    """
    _, residual = bf.make_rollout(n)
    phi_np, eigenvalues_np = test_modes(n, mode_count)
    phi = jnp.asarray(phi_np, F64)
    eigenvalues = jnp.asarray(eigenvalues_np, F64)
    interior = jnp.asarray(
        (np.arange(1, n - 1)[:, None] * n + np.arange(1, n - 1)[None, :]).reshape(-1)
    )

    def solve_alpha(u_prev, u_prev2, nu):
        velocity = u_prev - u_prev2
        weights = (1.0 + bf.DT * nu * eigenvalues) ** -1.0

        def weak_residual(alpha):
            candidate = u_prev + alpha * velocity
            point_residual = residual(candidate, u_prev, nu)[interior]
            return weights * (phi.T @ point_residual)

        def body(alpha, _):
            reduced_residual = weak_residual(alpha)
            jacobian = jax.jacfwd(weak_residual)(alpha)
            step = -jnp.vdot(jacobian, reduced_residual) / (
                jnp.vdot(jacobian, jacobian) + 1e-30
            )
            step = jnp.clip(step, -trust_radius, trust_radius)
            trial = jnp.clip(alpha + step, alpha_min, alpha_max)
            trial_residual = weak_residual(trial)
            accept = (
                jnp.all(jnp.isfinite(trial_residual))
                & (jnp.linalg.norm(trial_residual) < jnp.linalg.norm(reduced_residual))
            )
            return jnp.where(accept, trial, alpha), None

        initial_norm = jnp.linalg.norm(weak_residual(jnp.asarray(1.0, F64)))
        alpha, _ = jax.lax.scan(
            body, jnp.asarray(1.0, F64), None, length=reduced_iters
        )
        final_norm = jnp.linalg.norm(weak_residual(alpha))
        return alpha, initial_norm, final_norm

    def predictor(u_prev, u_prev2, u_prev3, u_prev4, nu, step_index):
        del u_prev3, u_prev4, step_index
        velocity = u_prev - u_prev2
        alpha, _, _ = solve_alpha(u_prev, u_prev2, nu)
        return u_prev + alpha * velocity

    predictor.diagnostics = jax.jit(solve_alpha)
    return predictor


def make_physics_predictor(n, scheme="imex_euler"):
    """Cheap classical physics predictors used as charged strong controls.

    ``explicit_euler`` evaluates the exact discrete upwind operator once.
    ``imex_euler`` treats the exact Dirichlet diffusion operator implicitly and
    upwind advection explicitly.  ``imex_ab2`` uses the same diffusion solve
    with second-order advection history after an Euler startup.  These are not
    learned and are never labelled NM-ROM arms.
    """
    if scheme not in ("explicit_euler", "imex_euler", "imex_ab2"):
        raise ValueError(f"unknown physics predictor {scheme}")
    _, residual = bf.make_rollout(n)
    dx = 1.0 / (n - 1)
    m = n - 2
    frequencies = jnp.arange(1, m + 1, dtype=F64)
    laplacian_eigenvalues = (4.0 / dx**2) * (
        jnp.sin(jnp.pi * frequencies[:, None] / (2.0 * (n - 1))) ** 2
        + jnp.sin(jnp.pi * frequencies[None, :] / (2.0 * (n - 1))) ** 2
    )

    def dst_axis(values, axis):
        zeros_shape = list(values.shape)
        zeros_shape[axis] = 1
        zeros = jnp.zeros(zeros_shape, values.dtype)
        extension = jnp.concatenate(
            (zeros, values, zeros, -jnp.flip(values, axis=axis)), axis=axis
        )
        transformed = -jnp.fft.fft(extension, axis=axis).imag
        index = [slice(None)] * values.ndim
        index[axis] = slice(1, values.shape[axis] + 1)
        return transformed[tuple(index)]

    def solve_diffusion(rhs, nu):
        interior = rhs.reshape(n, n)[1:-1, 1:-1]
        coefficients = dst_axis(dst_axis(interior, 0), 1)
        solved = dst_axis(
            dst_axis(
                coefficients / (1.0 + bf.DT * nu * laplacian_eigenvalues), 0
            ),
            1,
        ) / (4.0 * (m + 1) ** 2)
        return jnp.zeros((n, n), F64).at[1:-1, 1:-1].set(solved).reshape(-1)

    def advection(field_flat):
        field = field_flat.reshape(n, n)
        center = field[1:-1, 1:-1]
        dxm = (center - field[:-2, 1:-1]) / dx
        dxp = (field[2:, 1:-1] - center) / dx
        dym = (center - field[1:-1, :-2]) / dx
        dyp = (field[1:-1, 2:] - center) / dx
        ux = jnp.where(center > 0.0, dxm, dxp)
        uy = jnp.where(center > 0.0, dym, dyp)
        return center * (ux + uy)

    def predictor(u_prev, u_prev2, u_prev3, u_prev4, nu, step_index):
        del u_prev3, u_prev4
        if scheme == "explicit_euler":
            return u_prev - residual(u_prev, u_prev, nu)
        adv_current = advection(u_prev)
        if scheme == "imex_ab2":
            adv_previous = advection(u_prev2)
            adv_used = jax.lax.cond(
                step_index == 0,
                lambda: adv_current,
                lambda: 1.5 * adv_current - 0.5 * adv_previous,
            )
        else:
            adv_used = adv_current
        rhs = u_prev.reshape(n, n).at[1:-1, 1:-1].add(-bf.DT * adv_used)
        rhs = rhs.at[0, :].set(0.0).at[-1, :].set(0.0)
        rhs = rhs.at[:, 0].set(0.0).at[:, -1].set(0.0)
        return solve_diffusion(rhs.reshape(-1), nu)

    return predictor


def generate_reference(n, trajectory_indices, test_seed, draw_count=None):
    """Regenerate fresh test trajectories from seed with the reference FOM."""
    count = int(draw_count or (max(trajectory_indices) + 1))
    if count <= max(trajectory_indices):
        raise ValueError("draw_count must exceed every trajectory index")
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


def polynomial_guesses(U, order):
    """Reference-history diagnostic stream with deployable startup fallbacks."""
    if order not in (2, 3):
        raise ValueError("polynomial order must be 2 or 3")
    guesses = np.empty_like(U[1:])
    for step in range(T):
        if step == 0:
            guesses[step] = U[step]
        elif step == 1:
            guesses[step] = 2.0 * U[step] - U[step - 1]
        elif order == 2 or step == 2:
            guesses[step] = (
                3.0 * U[step] - 3.0 * U[step - 1] + U[step - 2]
            )
        else:
            guesses[step] = (
                4.0 * U[step]
                - 6.0 * U[step - 1]
                + 4.0 * U[step - 2]
                - U[step - 3]
            )
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
