"""An iterative-solver FOM for the same periodic NS3D problem.

The lane's comparator (`ns3d_fom.make_solver`, CNAB2) is a Fourier pseudo-spectral
Galerkin method: its implicit viscous step is a pointwise division in Fourier
space and its pressure is removed by the Leray projector, also pointwise in
Fourier space. It performs no linear solve at all; every "solve" is an FFT pair.

This module is a conventional grid FOM whose two linear systems ARE solved
iteratively, so the comparison can be made against a named iterative solver:

  * second-order central finite differences on the same collocated periodic grid;
  * rotational advection u x curl(u), explicit Adams-Bashforth 2 (forward Euler on
    the first step), exactly like CNAB2's explicit part;
  * Crank-Nicolson viscous step (I - dt nu L/2) u* = ..., L the 7-point Laplacian,
    solved by conjugate gradients;
  * exact discrete projection u = u* - grad phi with div(grad phi) = div u*, the
    central-difference ("wide") Laplacian, solved by conjugate gradients warm
    started from the previous step's phi. The projected field is discretely
    divergence-free in the central-difference sense.

Both CG solves stop on a relative residual `rtol` or `maxiter`; the iteration
counts are returned so an unconverged run cannot pass silently.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

AX = (-3, -2, -1)


def d1(f, axis, h):
    return (jnp.roll(f, -1, axis) - jnp.roll(f, 1, axis)) / (2 * h)


def grad(p, h):
    return jnp.stack([d1(p, a, h) for a in AX])


def div(u, h):
    return sum(d1(u[i], AX[i], h) for i in range(3))


def lap7(f, h):
    return sum(jnp.roll(f, -1, a) + jnp.roll(f, 1, a) - 2 * f for a in AX) / (h * h)


def curl(u, h):
    return jnp.stack((d1(u[2], -2, h) - d1(u[1], -1, h),
                      d1(u[0], -1, h) - d1(u[2], -3, h),
                      d1(u[1], -3, h) - d1(u[0], -2, h)))


def cross(a, b):
    return jnp.stack((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                      a[0] * b[1] - a[1] * b[0]))


def cg(apply, b, x0, rtol, maxiter):
    """Plain conjugate gradients on an SPD (or consistent PSD) operator.

    Stops when ||b - A x|| <= rtol ||b||. Returns (x, iterations, final ratio)."""
    bnorm = jnp.sqrt(jnp.vdot(b, b).real)
    r0 = b - apply(x0)
    rr0 = jnp.vdot(r0, r0).real
    target = (rtol * bnorm) ** 2

    def cond(s):
        _, _, _, rr, it = s
        return (rr > target) & (it < maxiter)

    def body(s):
        x, r, p, rr, it = s
        ap = apply(p)
        alpha = rr / jnp.vdot(p, ap).real
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = jnp.vdot(r, r).real
        p = r + (rr_new / rr) * p
        return x, r, p, rr_new, it + 1

    x, _, _, rr, it = jax.lax.while_loop(cond, body, (x0, r0, r0, rr0, jnp.int32(0)))
    return x, it, jnp.sqrt(rr) / jnp.maximum(bnorm, 1e-300)


def make_fd_solver(n, dt, nsteps, out_every, rtol=1e-8, maxiter=4000, diagnose=False):
    """Complete trajectory u0 -> (nsteps/out_every + 1) fields. Returns fields,
    and with diagnose=True also per-step CG iteration counts and residual ratios."""
    assert nsteps % out_every == 0
    h = 1.0 / n

    @jax.jit
    def run(u0, nu):
        def neg_wide(p):
            return -div(grad(p, h), h)

        def project(u, phi0):
            rhs = -div(u, h)
            rhs = rhs - jnp.mean(rhs)
            phi, it, ratio = cg(neg_wide, rhs, phi0, rtol, maxiter)
            return u - grad(phi, h), phi, it, ratio

        def viscous(v):
            return v - 0.5 * dt * nu * lap7(v, h)

        u, phi, _, _ = project(u0, jnp.zeros_like(u0[0]))

        def advance(u, phi, e_now, e_old, first):
            explicit = jnp.where(first, e_now, 1.5 * e_now - 0.5 * e_old)
            rhs = u + 0.5 * dt * nu * lap7(u, h) + dt * explicit
            ustar, itv, rv = cg(viscous, rhs, u, rtol, maxiter)
            unew, phi, itp, rp = project(ustar, phi)
            return unew, phi, (itv, rv, itp, rp)

        def step(carry, index):
            u, phi, e_old = carry
            e_now = cross(u, curl(u, h))
            unew, phi, info = advance(u, phi, e_now, e_old, index == 0)
            return (unew, phi, e_now), info

        def block(carry, b):
            carry, info = jax.lax.scan(step, carry, b * out_every + jnp.arange(out_every))
            return carry, (carry[0], info)

        _, (fields, info) = jax.lax.scan(block, (u, phi, jnp.zeros_like(u)),
                                         jnp.arange(nsteps // out_every))
        fields = jnp.concatenate((u[None], fields))
        if not diagnose:
            return fields
        return fields, tuple(x.reshape(-1) for x in info)
    return run
