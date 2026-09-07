"""Curvature-inclusive weak manifold dynamics and physical velocity.

All bank tables and parameters are explicit JIT arguments. No full grid is used
by the latent rollout. Accuracy comparison to the FOM remains an external test.
"""
from functools import partial
import jax
import jax.numpy as jnp
from fresh_models import head_apply, head_geometry


def weak_acceleration(p, frozen, z, w, stiffness, damping, kind):
    a, b, jac, curvature = head_geometry(p, frozen, z, w, kind)
    force = curvature+damping@b+stiffness@a
    q, rr = jnp.linalg.qr(jac, mode="reduced")
    accel = jax.scipy.linalg.solve_triangular(rr, -q.T@force, lower=False)
    # Singular values are also checked at every RK stage, without hidden ridge.
    singular = jnp.linalg.svd(jac, compute_uv=False)
    ratio = singular[-1]/jnp.maximum(singular[0], 1e-300)
    power = b@damping@b
    valid = (ratio > 1e-8) & jnp.all(jnp.isfinite(accel)) & jnp.isfinite(power) & jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b))
    return accel, power, ratio, valid


def physical_energy(p, frozen, z, w, stiffness, kind):
    a, b, _, _ = head_geometry(p, frozen, z, w, kind)
    return .5*(b@b+a@stiffness@a)


def rk4_rom_step(p, frozen, state, dt, stiffness, damping, kind):
    def f(st):
        z, w, outflux = st
        a, power, ratio, valid = weak_acceleration(p, frozen, z, w, stiffness, damping, kind)
        return (w, a, power), ratio, valid
    
    def add(y, a, scale):
        return tuple(v+scale*d for v, d in zip(y, a))
    a, ra, va = f(state)
    b, rb, vb = f(add(state, a, dt/2))
    c, rc, vc = f(add(state, b, dt/2))
    d, rd, vd = f(add(state, c, dt))
    nxt = tuple(v+dt*(aa+2*bb+2*cc+dd)/6 for v, aa, bb, cc, dd in zip(state, a, b, c, d))
    valid = va & vb & vc & vd & jnp.all(jnp.isfinite(nxt[0])) & jnp.all(jnp.isfinite(nxt[1])) & jnp.isfinite(nxt[2])
    return nxt, jnp.min(jnp.array([ra, rb, rc, rd])), valid


@partial(jax.jit, static_argnames=("kind", "steps", "stride"))
def rollout(p, frozen, z0, w0, stiffness, damping, dt, *, kind, steps, stride):
    if steps % stride:
        raise ValueError("steps must be divisible by stride")
    def block(carry, _):
        state, ratio, completed = carry
        def step(_, inner):
            st, rr, ok = inner
            def advance(_):
                nxt, rn, good = rk4_rom_step(p, frozen, st, dt, stiffness, damping, kind)
                safe = tuple(jnp.where(good, nn, ss) for nn, ss in zip(nxt, st))
                return safe, jnp.minimum(rr, rn), good
            return jax.lax.cond(ok, advance, lambda _: inner, operand=None)
        state, ratio, completed = jax.lax.fori_loop(0, stride, step, (state, ratio, completed))
        return (state, ratio, completed), (*state, ratio, completed)
    _, _, initial_ratio, initial_valid = weak_acceleration(p, frozen, z0, w0, stiffness, damping, kind)
    initial = ((z0, w0, jnp.array(0.)), initial_ratio, initial_valid)
    _, (z, w, q, ratio, completed) = jax.lax.scan(block, initial, None, length=steps//stride)
    return {"z": jnp.concatenate((z0[None], z)), "w": jnp.concatenate((w0[None], w)), "outflux": jnp.concatenate((jnp.zeros(1), q)),
            "rank_ratio": jnp.concatenate((initial_ratio[None], ratio)), "completed": jnp.concatenate((initial_valid[None], completed))}
