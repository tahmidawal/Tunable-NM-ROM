"""Heat acceleration with the original frozen nonlinear head and latent dimension."""
import heat_core as hc
import jax
import jax.numpy as jnp
import numpy as np


def make_adaptive_initialize(cfg, initial_gate):
    fit = hc.make_lm(hc.sc.head, cfg["fit_budget"], cfg["gradient_tolerance"])
    @jax.jit
    def initialize(params, projection, triangular, library, codes, u0):
        flat = u0.reshape(-1)
        target = projection.T@flat
        nearest = jnp.argmin(jnp.sum((library-target)**2, axis=1))
        zfirst, first = fit(params, triangular, target, codes[nearest])
        norm2 = jnp.dot(flat, flat)
        outside2 = jnp.maximum(norm2-jnp.dot(target, target), 0.)
        mismatch = triangular@hc.sc.head(params, zfirst)-target
        error2 = (outside2+jnp.dot(mismatch, mismatch))/jnp.maximum(norm2, 1e-28)
        need_second = (first[2] != 1) | (~jnp.isfinite(error2)) | (error2 > initial_gate**2)
        def second_start():
            zsecond, second = fit(params, triangular, target, jnp.mean(codes, axis=0))
            z = jnp.where(first[3] <= second[3], zfirst, zsecond)
            return z, jnp.stack((first, second))
        def first_only():
            # Reason -1 means not attempted, rather than a failed/converged fit.
            skipped = jnp.array([0., 0., -1., 0., 0.], dtype=jnp.float64)
            return zfirst, jnp.stack((first, skipped))
        return jax.lax.cond(need_second, second_start, first_only)
    return initialize


def make_projected_rollout(head_fn, budget, tolerance, outputs):
    fit = hc.make_lm(head_fn, budget, tolerance)
    @jax.jit
    def rollout(params, matrix, coefficient_step, z0):
        def step(z, _):
            target = matrix@(coefficient_step@head_fn(params, z))
            next_z, info = fit(params, matrix, target, z)
            return next_z, (next_z, info)
        _, (zs, infos) = jax.lax.scan(step, z0, None, length=outputs-1)
        return jnp.concatenate((z0[None], zs)), infos
    return rollout


def build(cfg, settings):
    initialize, original_rollout, readout = hc.make_query(cfg, settings["dt"])
    adaptive = make_adaptive_initialize(cfg, settings["initial_gate"])
    projected_full = make_projected_rollout(hc.sc.head, cfg["step_budget"], cfg["gradient_tolerance"], len(cfg["times"]))
    projected_two = make_projected_rollout(hc.sc.head, settings["projection_budget"], cfg["gradient_tolerance"], len(cfg["times"]))
    specs = {
        "nmrom_adaptive": (adaptive, original_rollout, False),
        "exp_project_full": (initialize, projected_full, True),
        "exp_project2": (initialize, projected_two, True),
        "exp_project2_adaptive": (adaptive, projected_two, True),
    }
    def make_query(init, advance, exponential):
        @jax.jit
        def query(params, projection, triangular, library, codes, bank, matrix, mode_lam, coefficient_step, u0):
            z0, init_info = init(params, projection, triangular, library, codes, u0)
            operator = coefficient_step if exponential else hc.cn_factor(mode_lam, settings["dt"], cfg["diffusivity"])
            zs, step_info = advance(params, matrix, operator, z0)
            return readout(params, bank, zs), init_info, step_info, zs
        return query
    return {name: make_query(*spec) for name, spec in specs.items()}


def verify():
    """Exact linear-head semigroup checks the new recursion, not its own code path."""
    matrix = jnp.vstack((jnp.eye(3), jnp.eye(3)))
    rates = jnp.array([.2, .4, .9]); dt = .1
    operator = jnp.diag(jnp.exp(-dt*rates)); z0 = jnp.array([.5, -.2, 1.])
    rollout = make_projected_rollout(lambda p, z: z, 30, 1e-12, 6)
    zs, info = rollout({}, matrix, operator, z0)
    expected = np.exp(-np.arange(6)[:, None]*dt*np.asarray(rates))*np.asarray(z0)
    error = float(np.max(np.abs(np.asarray(zs)-expected)))
    assert error < 1e-10 and np.all(np.asarray(info)[:, 2] == 1)
    return dict(linear_head_exact_evolution_max_error=error)
