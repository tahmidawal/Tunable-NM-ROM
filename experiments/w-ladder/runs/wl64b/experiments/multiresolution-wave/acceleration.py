"""Parity-controlled geometry optimization for the frozen reflective MLP32.

The frozen fresh-wave implementation remains the control.  These functions
change algebra, not the manifold equation, curvature, RK4 method or rank gate.
"""
from functools import partial
import numpy as np
import pilot as base
from pilot import jax, jnp
from fresh_models import head_geometry
from fresh_rom import weak_acceleration


def shared_geometry(p, frozen, z, w):
    """One analytic forward pass for value, tangent, Jacobian and H[w,w]."""
    x, tangent, curve = z, w, jnp.zeros_like(z)
    jac = jnp.eye(z.size, dtype=z.dtype)
    for name in ('l1', 'l2'):
        weights, bias = p[name]['w'], p[name]['b']
        y = x@weights+bias
        yt, yc, yj = tangent@weights, curve@weights, weights.T@jac
        sigmoid = jax.nn.sigmoid(y)
        first = sigmoid+y*sigmoid*(1-sigmoid)
        second = sigmoid*(1-sigmoid)*(2+y*(1-2*sigmoid))
        x = y*sigmoid
        tangent, curve, jac = first*yt, first*yc+second*yt*yt, first[:, None]*yj
    weights, bias, scale = p['out']['w'], p['out']['b'], frozen['output_scale']
    return (p['bias']+p['linear']@z+scale*(x@weights+bias),
            p['linear']@w+scale*(tangent@weights),
            p['linear']+scale*(weights.T@jac), scale*(curve@weights))


def exact_rank(jac):
    singular = jnp.linalg.svd(jac, compute_uv=False)
    return singular[-1]/jnp.maximum(singular[0], 1e-300)


def qr_solve(jac, force, guard):
    q, r = jnp.linalg.qr(jac, mode='reduced')
    acceleration = jax.scipy.linalg.solve_triangular(r, -q.T@force, lower=False)
    if not guard:
        return acceleration, exact_rank(r), jnp.int32(0)
    inverse = jax.scipy.linalg.solve_triangular(r, jnp.eye(r.shape[0]), lower=False)
    bound = 1/(jnp.linalg.norm(r)*jnp.linalg.norm(inverse))
    # Frobenius norms bound spectral norms, hence this is a sufficient lower
    # bound on sigma_min(J)/sigma_max(J) in exact arithmetic.  A 100x safety
    # margin around the frozen 1e-8 threshold triggers the original exact test.
    fast = jnp.isfinite(bound) & (bound > 1e-6)
    ratio = jax.lax.cond(fast, lambda _: bound,
                         lambda _: exact_rank(jac), operand=None)
    return acceleration, ratio, (~fast).astype(jnp.int32)


def cholesky_solve(jac, force):
    gram = jac.T@jac
    lower = jnp.linalg.cholesky(gram)
    inverse = jax.scipy.linalg.solve_triangular(lower, jnp.eye(lower.shape[0]), lower=True)
    bound = 1/(jnp.linalg.norm(jac)*jnp.linalg.norm(inverse))
    # A much stronger threshold limits normal-equation roundoff.  Any dubious
    # factorization/conditioning returns to original QR + exact J singulars.
    fast = jnp.all(jnp.isfinite(lower)) & jnp.isfinite(bound) & (bound > 1e-3)
    def accepted(_):
        rhs = -jac.T@force
        acceleration = jax.scipy.linalg.cho_solve((lower, True), rhs)
        return acceleration, bound, jnp.int32(0)
    def fallback(_):
        accel, _, _ = qr_solve(jac, force, False)
        return accel, exact_rank(jac), jnp.int32(1)
    return jax.lax.cond(fast, accepted, fallback, operand=None)


def accelerated_rhs(p, frozen, z, w, stiffness, damping, variant):
    a, b, jac, curvature = shared_geometry(p, frozen, z, w)
    force = curvature+damping@b+stiffness@a
    if variant == 'shared_svd_r':
        accel, ratio, fallback = qr_solve(jac, force, False)
    elif variant == 'qr_guard':
        accel, ratio, fallback = qr_solve(jac, force, True)
    elif variant == 'chol_guard':
        accel, ratio, fallback = cholesky_solve(jac, force)
    else:
        raise ValueError(variant)
    power = b@damping@b
    valid = ((ratio > 1e-8) & jnp.all(jnp.isfinite(accel)) &
             jnp.isfinite(power) & jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)))
    gram = jac.T@jac
    rhs = -jac.T@force
    backward = jnp.linalg.norm(gram@accel-rhs)/jnp.maximum(jnp.linalg.norm(gram)*jnp.linalg.norm(accel)+jnp.linalg.norm(rhs), 1e-300)
    return accel, power, ratio, valid, fallback, backward


@partial(jax.jit, static_argnames=('variant', 'steps', 'stride'))
def rollout(p, frozen, z0, w0, stiffness, damping, dt, *, variant, steps, stride):
    """Identical RK4 state updates and safety holds, with recorded guard counts."""
    assert steps % stride == 0
    def f(state):
        z, w, _ = state
        acc, power, ratio, valid, fallback, backward = accelerated_rhs(p, frozen, z, w, stiffness, damping, variant)
        return (w, acc, power), ratio, valid, fallback, backward
    def plus(state, value, scale):
        return tuple(s+scale*v for s, v in zip(state, value))
    def step(_, carry):
        state, ratio, completed, fallback, backward = carry
        def advance(_):
            a, ra, va, fa, ba = f(state)
            b, rb, vb, fb, bb = f(plus(state, a, dt/2))
            c, rc, vc, fc, bc = f(plus(state, b, dt/2))
            d, rd, vd, fd, bd = f(plus(state, c, dt))
            nxt = tuple(s+dt*(aa+2*bb+2*cc+dd)/6
                        for s, aa, bb, cc, dd in zip(state, a, b, c, d))
            valid = va & vb & vc & vd & jnp.all(jnp.isfinite(nxt[0])) & jnp.all(jnp.isfinite(nxt[1])) & jnp.isfinite(nxt[2])
            safe = tuple(jnp.where(valid, n, s) for n, s in zip(nxt, state))
            return safe, jnp.minimum(ratio, jnp.min(jnp.array([ra, rb, rc, rd]))), valid, fallback+fa+fb+fc+fd, jnp.maximum(backward,jnp.max(jnp.array([ba,bb,bc,bd])))
        return jax.lax.cond(completed, advance, lambda _: carry, operand=None)
    def block(carry, _):
        carry = jax.lax.fori_loop(0, stride, step, carry)
        state, ratio, completed, fallback, backward = carry
        return carry, (*state, ratio, completed, fallback, backward)
    _, _, initial_ratio, initial_valid, initial_fallback, initial_backward = accelerated_rhs(p, frozen, z0, w0, stiffness, damping, variant)
    initial = ((z0, w0, jnp.array(0.)), initial_ratio, initial_valid, initial_fallback, initial_backward)
    _, (z, w, flux, ratio, completed, fallbacks, backwards) = jax.lax.scan(block, initial, None, length=steps//stride)
    return dict(z=jnp.concatenate((z0[None], z)), w=jnp.concatenate((w0[None], w)),
                outflux=jnp.concatenate((jnp.zeros(1), flux)),
                rank_ratio=jnp.concatenate((initial_ratio[None], ratio)),
                completed=jnp.concatenate((initial_valid[None], completed)),
                fallback_count=jnp.concatenate((initial_fallback[None], fallbacks)),
                normal_backward_error=jnp.concatenate((initial_backward[None], backwards)))


def geometry_checks(bank, z, w):
    original = jax.jit(lambda p, f, z, w: head_geometry(p, f, z, w, 'mlp'))
    shared = jax.jit(shared_geometry)
    p, frozen = bank['p'], bank['frozen']
    aa, bb = original(p, frozen, z, w), shared(p, frozen, z, w)
    errors = [float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a, b in zip(aa, bb)]
    relative = [float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(np.asarray(a)), 1e-12)) for a, b in zip(aa, bb)]
    assert max(relative)<1e-10
    jac = np.asarray(aa[2])
    singular = np.linalg.svd(jac, compute_uv=False)
    return dict(absolute_errors=errors, relative_errors=relative,
                exact_rank_ratio=float(singular[-1]/singular[0]))
