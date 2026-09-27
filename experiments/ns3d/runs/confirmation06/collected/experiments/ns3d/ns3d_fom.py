"""Periodic vector NS3D, strict 2/3 spectral Galerkin space, f64/complex128.

Velocity arrays are (3,N,N,N), bank arrays (3*N**3,R), snapshots (S,3,N,N,N).
The projector/cutoff is part of the discrete space, applied to both nonlinear
inputs and outputs. All grid-sized arrays are arguments to compiled functions.
"""
from __future__ import annotations

import functools
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

AXES = (-3, -2, -1)
INITIAL_AMPLITUDE = 0.2  # prospective DESIGN amendment A1; original screen retained


def geometry(n):
    kk = np.fft.fftfreq(n, d=1.0 / n)
    modes = np.stack(np.meshgrid(kk, kk, kk, indexing='ij'))
    k = 2 * np.pi * modes
    mask = np.all(np.abs(modes) < n / 3, axis=0)
    mask[0, 0, 0] = False
    return tuple(jnp.asarray(a) for a in (k, np.sum(k * k, axis=0), mask))


def fft(u):
    return jnp.fft.fftn(u, axes=AXES, norm='forward')


def ifft(uh):
    return jnp.fft.ifftn(uh, axes=AXES, norm='forward').real


def project(uh, geom):
    k, k2, mask = geom
    dot = jnp.sum(k * uh, axis=0)
    return (uh - k * (dot / jnp.where(k2 > 0, k2, 1.0))) * mask


def cross(a, b):
    return jnp.stack((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                      a[0]*b[1]-a[1]*b[0]))


def nonlinear(uh, geom):
    uh = project(uh, geom)
    u = ifft(uh)
    omega = ifft(1j * cross(geom[0], uh))
    return project(fft(cross(u, omega)), geom)


def rhs(uh, nu, geom):
    return nonlinear(uh, geom) - nu * geom[1] * uh


def project_field(u, geom):
    return ifft(project(fft(u), geom))


def make_solver(dt, nsteps, out_every, scheme='cnab2', forcing=None, adv_sign=1.):
    """Compiled complete trajectory. forcing(t, nu, geom) is already Fourier-space.

    Return run(u0, nu, geom) -> states, plus no hidden grid-sized closure arrays.
    CNAB2 starts with an implicit-diffusion predictor and CN/Heun corrector.
    RK4 is available for a distinct time-integration convergence check.
    """
    assert nsteps % out_every == 0 and nsteps >= out_every
    assert scheme in ('cnab2', 'rk4')

    def explicit(uh, t, nu, geom, forcing_data):
        val = adv_sign * nonlinear(uh, geom)
        return val if forcing is None else val + forcing(t, nu, geom, forcing_data)

    @jax.jit
    def run(u0, nu, geom, forcing_data=()):
        uh0 = project(fft(u0), geom)
        k2 = geom[1]

        def step(carry, index):
            uh, old_e = carry
            t = index * dt
            if scheme == 'rk4':
                def fun(u, tt):
                    return explicit(u, tt, nu, geom, forcing_data) - nu*k2*u
                a = fun(uh, t)
                b = fun(uh + dt*a/2, t+dt/2)
                c = fun(uh + dt*b/2, t+dt/2)
                d = fun(uh + dt*c, t+dt)
                unew = uh + dt*(a+2*b+2*c+d)/6
                enew = old_e
            else:
                e = explicit(uh, t, nu, geom, forcing_data)
                den = 1 + 0.5*dt*nu*k2

                def startup():
                    pred = project((uh+dt*e)/(1+dt*nu*k2), geom)
                    ep = explicit(pred, t+dt, nu, geom, forcing_data)
                    return ((1-0.5*dt*nu*k2)*uh+0.5*dt*(e+ep))/den

                def normal():
                    return ((1-0.5*dt*nu*k2)*uh+dt*(1.5*e-0.5*old_e))/den
                unew = jax.lax.cond(index == 0, startup, normal)
                enew = e
            return (project(unew, geom), enew), None

        def block(carry, b):
            state, _ = jax.lax.scan(step, carry, b*out_every+jnp.arange(out_every))
            return state, ifft(state[0])

        _, out = jax.lax.scan(block, (uh0, jnp.zeros_like(uh0)),
                              jnp.arange(nsteps//out_every))
        return jnp.concatenate((ifft(uh0)[None], out))
    return run


def parameters(seed, count):
    """Row-wise draws preserve prefixes when a cohort is expanded."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        center = rng.uniform(0, 1, 3)
        rows.append([*center, rng.uniform(.12, .24), rng.uniform(.6, 1.4),
                     np.exp(rng.uniform(np.log(.002), np.log(.01)))])
    return np.asarray(rows, dtype=np.float64)


def initial_raw(n, parameter):
    """Analytic curl of two obliquely separated, nonparallel vector potentials."""
    x = np.arange(n, dtype=np.float64)/n
    xyz = np.stack(np.meshgrid(x, x, x, indexing='ij'))
    c, sep, strength = parameter[:3], parameter[3], parameter[4]
    offset = np.array([1., 1., 1.])/np.sqrt(3)
    directions = (np.array([0., 0., 1.]), np.array([1., 1., 0.])/np.sqrt(2))
    u = np.zeros((3, n, n, n), dtype=np.float64)
    for side, weight, direction in ((-1, 1., directions[0]), (1, strength, directions[1])):
        center = c + side*.5*sep*offset
        angle = 2*np.pi*(xyz-center[:, None, None, None])
        scalar = np.exp(2*np.sum(np.cos(angle)-1, axis=0))
        grad = -4*np.pi*np.sin(angle)*scalar
        u += weight*np.moveaxis(np.cross(np.moveaxis(grad, 0, -1), direction), -1, 0)
    return INITIAL_AMPLITUDE*u


def initial(n, parameter):
    return np.asarray(project_field(jnp.asarray(initial_raw(n, parameter)), geometry(n)))


def diagnostics(states, geom):
    states = jnp.asarray(states)
    def one(u):
        uh = fft(u)
        k, k2, _ = geom
        denom = jnp.linalg.norm(uh)
        div = jnp.linalg.norm(jnp.sum(1j*k*uh, axis=0)) / jnp.maximum(
            jnp.sqrt(jnp.sum(k2*jnp.sum(jnp.abs(uh)**2, axis=0))), 1e-300)
        energy = .5*jnp.mean(jnp.sum(u*u, axis=0))
        enstrophy = .5*jnp.mean(jnp.sum(ifft(1j*cross(k, uh))**2, axis=0))
        component = jnp.sqrt(jnp.mean(u*u, axis=(1,2,3)))
        gradients = jnp.asarray([jnp.linalg.norm(k[i]*uh) for i in range(3)])
        nl = nonlinear(uh, geom)
        nonlin_norm = jnp.linalg.norm(nl)/jnp.maximum(denom, 1e-300)
        transfer = jnp.real(jnp.vdot(uh, nl)) / jnp.maximum(
            jnp.linalg.norm(uh)*jnp.linalg.norm(nl), 1e-300)
        return energy, enstrophy, div, component, gradients, nonlin_norm, transfer
    return jax.vmap(one)(states)


def relative_errors(pred, ref, initial_field):
    pred, ref = np.asarray(pred), np.asarray(ref)
    return np.linalg.norm((pred-ref).reshape(len(pred), -1), axis=1) / np.linalg.norm(initial_field)


def restrict_fields(states, target_n):
    """Spectral restriction by evaluating retained Fourier coefficients.

    Handles arbitrary even meshes; norm='forward' removes amplitude scaling.
    This is restriction, not projection into the coarser FOM cutoff space.
    """
    from scipy.signal import resample
    out = np.asarray(states)
    for axis in (-3, -2, -1):
        out = resample(out, target_n, axis=axis)
    return np.asarray(out.real, dtype=np.float64)
