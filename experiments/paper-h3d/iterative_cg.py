"""Counted matrix-free CG for full-grid Crank--Nicolson heat evolution.

Identity preconditioning (the uniform-grid Jacobi diagonal is a scalar and
cannot improve the condition number). No transform is used in this solver.
"""
import numpy as np
import jax
import jax.numpy as jnp
import common as C

STAT_COLUMNS = ['iterations', 'true_relative_residual', 'converged', 'breakdown',
                'iteration_cap_reached', 'residual_restarts']


def engine(n, cfg, dt, tolerance, max_iterations):
    times = np.asarray(cfg['times'])
    ticks = np.rint(times / dt).astype(int)
    assert times[0] == 0 and np.all(np.diff(times) > 0)
    assert np.max(np.abs(ticks * dt - times)) < 1e-12
    assert tolerance > 0 and max_iterations > 0
    alpha = cfg['diffusivity'] * dt / 2
    operator = lambda u: u + alpha * C.negative_laplacian(u, n)
    dot = lambda u, v: jnp.vdot(u, v).real

    def solve(rhs, guess):
        residual = rhs - operator(guess)
        norm_rhs = jnp.linalg.norm(rhs)
        threshold2 = (tolerance * norm_rhs)**2
        rr = dot(residual, residual)
        # Recurrence stops at the requested threshold, then verifies the true
        # residual. A failed verification restarts CG with the remaining budget.
        initial = (jnp.int32(0), guess, residual, residual, rr,
                   jnp.bool_(False), jnp.int32(0))
        def condition(state):
            iteration, _, _, _, rr, broken, _ = state
            return (iteration < max_iterations) & (rr > threshold2) & ~broken
        def body(state):
            iteration, x, r, p, rr, broken, restarts = state
            ap = operator(p); curvature = dot(p, ap)
            bad = ~jnp.isfinite(curvature) | (curvature <= 0)
            step = jnp.where(bad, 0., rr / curvature)
            xn = x + step * p; rn = r - step * ap; rrn = dot(rn, rn)
            def verify(_):
                true_r = rhs - operator(xn)
                true_rr = dot(true_r, true_r)
                return true_r, true_rr, true_rr > threshold2
            rn, rrn, restart = jax.lax.cond(rrn <= threshold2, verify,
                lambda _: (rn, rrn, jnp.bool_(False)), operand=None)
            pn = jnp.where(restart, rn, rn + (rrn / jnp.maximum(rr, 1e-300)) * p)
            return iteration + 1, xn, rn, pn, rrn, bad | ~jnp.isfinite(rrn), restarts + restart.astype(jnp.int32)
        iteration, x, _, _, _, broken, restarts = jax.lax.while_loop(condition, body, initial)
        true_norm = jnp.linalg.norm(rhs - operator(x))
        converged = jnp.isfinite(true_norm) & (true_norm <= tolerance * norm_rhs) & ~broken
        stats = jnp.array([iteration, true_norm / jnp.maximum(norm_rhs, 1e-300),
            converged, broken, (iteration >= max_iterations) & ~converged, restarts], dtype=jnp.float64)
        return x, stats

    @jax.jit
    def trajectory(u0):
        def step(previous, _):
            rhs = previous - alpha * C.negative_laplacian(previous, n)
            current, stats = solve(rhs, previous)
            return current, (current, stats)
        _, (states, stats) = jax.lax.scan(step, u0, None, length=int(ticks[-1]))
        states = jnp.concatenate((u0[None], states))
        return dict(prediction=states[jnp.asarray(ticks)], cg_stats=stats, cg_states=states)
    return trajectory


def methods(n, cfg):
    result = {}; metadata = {}
    for setting in cfg.get('iterative_cg_controls', []):
        dt = float(setting['dt']); tol = float(setting['relative_tolerance'])
        cap = int(setting.get('max_iterations', 500))
        name = f'fom_cn_cg_dt{dt:g}_rtol{tol:.0e}'
        assert name not in result
        result[name] = engine(n, cfg, dt, tol, cap)
        metadata[name] = dict(kind='full_order', algorithm='Crank-Nicolson; counted matrix-free CG',
            spatial_operator='seven-point finite-difference negative Laplacian; homogeneous Dirichlet',
            dt=dt, relative_tolerance=tol, absolute_tolerance=0., max_iterations=cap,
            preconditioner='identity; Jacobi is a scalar on this uniform constant-coefficient grid',
            warm_start='previous full-grid time-step solution', stopping='true residual norm divided by RHS norm',
            stats_columns=STAT_COLUMNS, retains_all_step_fields=True,
            setup='no factorization, transform, or fitted preconditioner')
    return result, metadata
