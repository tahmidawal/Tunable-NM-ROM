"""CP-inspired exact weak normal-equation contraction and Cholesky ablation."""
import heat_core as hc
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np


def make_gram_lm(head_fn, budget, tolerance, linear_solver):
    """Same normalized weak objective and LM policy; no weak-Jacobian assembly.

    Retain the small explicit weak residual for stable loss evaluation. Only
    normal equations are contracted; no subtraction of large squared norms.
    The fixed matrix Gram is assembled offline and supplied as a runtime input.
    """
    assert linear_solver in ('lu', 'cholesky')
    def solve(params, matrix, gram, target, z0):
        scale = jnp.maximum(jnp.linalg.norm(target), 1e-14)
        scaled_gram = gram/(scale*scale)
        # Anchor the gradient contraction at the actual initial residual to
        # reduce cancellation in S*h - B.T*target near a good warm start.
        anchor = head_fn(params, z0)
        anchor_residual = (matrix@anchor-target)/scale
        anchor_gradient = matrix.T@anchor_residual/scale
        def parts(z):
            h = head_fn(params, z)
            derivative = jax.jacfwd(head_fn, argnums=1)(params, z)
            residual = (matrix@h-target)/scale
            normal = derivative.T@(scaled_gram@derivative)
            gradient = derivative.T@(scaled_gram@(h-anchor)+anchor_gradient)
            jac_norm = jnp.sqrt(jnp.maximum(jnp.trace(normal), 0.))
            return jnp.dot(residual, residual), normal, gradient, jac_norm
        def value_at(z):
            residual = (matrix@head_fn(params, z)-target)/scale
            return jnp.dot(residual, residual)
        val, normal, grad, jac_norm = parts(z0)
        # z, value, normal matrix, gradient, Jacobian norm, damping,
        # attempts, accepted, termination reason.
        initial = (z0, val, normal, grad, jac_norm, jnp.float64(1e-4),
                   jnp.int32(0), jnp.int32(0), jnp.int32(0))
        def condition(s):
            gradient = jnp.linalg.norm(s[3])/jnp.maximum(s[4], 1e-30)
            return (s[6] < budget) & (s[8] == 0) & (gradient > tolerance)
        def body(s):
            z, val, normal, grad, jnorm, damping, attempts, accepted, _ = s
            d = jnp.maximum(jnp.diag(normal), 1e-12)
            system = normal+damping*jnp.diag(d)
            if linear_solver == 'cholesky':
                lower = jnp.linalg.cholesky(system)
                step = jsl.solve_triangular(lower.T, jsl.solve_triangular(lower, -grad, lower=True), lower=False)
            else:
                step = jnp.linalg.solve(system, -grad)
            candidate = z+step
            vnew = value_at(candidate)
            accept = jnp.isfinite(vnew) & (vnew < val)
            znew = jnp.where(accept, candidate, z)
            valnew, nnew, gnew, jnew = jax.lax.cond(accept, lambda: parts(znew), lambda: (val, normal, grad, jnorm))
            damping = jnp.where(accept, jnp.maximum(damping/3, 1e-12), damping*10)
            tiny = jnp.linalg.norm(step) < 1e-12*(1+jnp.linalg.norm(z))
            reason = jnp.where(tiny, 2, jnp.where(damping > 1e12, 3, 0)).astype(jnp.int32)
            return znew, valnew, nnew, gnew, jnew, damping, attempts+1, accepted+accept.astype(jnp.int32), reason
        z, val, _, grad, jnorm, _, attempts, accepted, reason = jax.lax.while_loop(condition, body, initial)
        gradient = jnp.linalg.norm(grad)/jnp.maximum(jnorm, 1e-30)
        reason = jnp.where(gradient <= tolerance, 1, reason)
        reason = jnp.where(jnp.isfinite(val), reason, 4).astype(jnp.int32)
        return z, jnp.array([attempts, accepted, reason, jnp.sqrt(val), gradient])
    return solve


def build(cfg, settings):
    nsteps = int(round(cfg['times'][-1]/settings['dt']))
    stride = int(round((cfg['times'][1]-cfg['times'][0])/settings['dt']))
    assert np.allclose(np.arange(len(cfg['times']))*stride*settings['dt'], cfg['times'])
    def make_query(contract, linear_solver):
        def solver(budget):
            if contract:
                return make_gram_lm(hc.sc.head, budget, settings['gradient_tolerance'], linear_solver)
            ordinary = hc.make_lm(hc.sc.head, budget, settings['gradient_tolerance'], linear_solver)
            return lambda p, matrix, gram, target, z: ordinary(p, matrix, target, z)
        fit, advance = solver(cfg['fit_budget']), solver(cfg['step_budget'])
        @jax.jit
        def query(params, projection, triangular, library, codes, bank, matrix, mode_lam, initial_gram, weak_gram, u0):
            target = projection.T@u0.reshape(-1)
            nearest = jnp.argmin(jnp.sum((library-target)**2, axis=1))
            starts = jnp.stack((codes[nearest], jnp.mean(codes, axis=0)))
            initial_zs, initial_infos = jax.vmap(lambda z: fit(params, triangular, initial_gram, target, z))(starts)
            z0 = initial_zs[jnp.argmin(initial_infos[:, 3])]
            factor = hc.cn_factor(mode_lam, settings['dt'], cfg['diffusivity'])
            def step(zprev, _):
                target = factor*(matrix@hc.sc.head(params, zprev))
                znew, info = advance(params, matrix, weak_gram, target, zprev)
                return znew, (znew, info)
            _, (all_zs, steps) = jax.lax.scan(step, z0, None, length=nsteps)
            zs = jnp.concatenate((z0[None], all_zs[stride-1::stride]))
            internal = jnp.concatenate((z0[None], all_zs))
            return hc.sc.head(params, zs)@bank.T, initial_infos, steps, zs, internal, initial_zs
        return query
    return {'nmrom': make_query(False, 'lu'), 'nmrom_cholesky': make_query(False, 'cholesky'),
            'nmrom_gram_lu': make_query(True, 'lu'), 'nmrom_gram_cholesky': make_query(True, 'cholesky')}


def verify():
    """Independent exact-solution fixture with a genuinely nonlinear head."""
    rng = np.random.default_rng(790717)
    matrix = jnp.asarray(rng.normal(size=(19, 5)))
    target = jnp.asarray(rng.normal(size=19))
    z0 = jnp.asarray(rng.normal(size=5)*.1)
    def head(p, z): return z+.1*z**3
    gram = matrix.T@matrix
    values = {}
    for name, method in [('explicit_lu', hc.make_lm(head, 80, 1e-10)), ('explicit_chol', hc.make_lm(head, 80, 1e-10, 'cholesky'))]:
        values[name] = jax.jit(method)({}, matrix, target, z0)
    for linear in ['lu', 'cholesky']:
        values['gram_'+linear] = jax.jit(make_gram_lm(head, 80, 1e-10, linear))({}, matrix, gram, target, z0)
    expected = np.linalg.lstsq(np.asarray(matrix), np.asarray(target), rcond=None)[0]
    errors = {}
    for name, (z, info) in values.items():
        error = float(np.linalg.norm(np.asarray(head({}, z))-expected))
        assert error < 1e-8 and int(info[2]) == 1, (name, error, info)
        errors[name] = error
    return dict(nonlinear_head_exact_least_squares_errors=errors)
