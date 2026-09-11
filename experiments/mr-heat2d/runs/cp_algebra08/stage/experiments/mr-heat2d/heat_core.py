"""Verified heat operators and weak continuous-bank NM-ROM; no old heat imports."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "separable-decoder"))
import sep_common as sc


def coords(intervals):
    axis = np.arange(1, intervals, dtype=np.float64) / intervals
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)


def sample_family(seed, count, cfg):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(*cfg["center_range"], (count, 2)),
                            rng.uniform(*cfg["width_range"], count),
                            rng.uniform(*cfg["amplitude_range"], count)))


def initial_field(xy, draw):
    """Draw descriptors are used ONLY by data generation, never head/initial fit."""
    cx, cy, width, amplitude = draw
    radius2 = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2
    return amplitude * sc.bc_poly(xy) * jnp.exp(-radius2 / (2 * width**2))


def dst_axis(u, axis=-1):
    """Orthonormal DST-I using a length-2N odd FFT extension, self inverse."""
    u = jnp.moveaxis(u, axis, -1)
    n = u.shape[-1] + 1
    zero = jnp.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
    odd = jnp.concatenate((zero, u, zero, -u[..., ::-1]), axis=-1)
    out = -jnp.fft.rfft(odd, axis=-1).imag[..., 1:n] / jnp.sqrt(2.0*n)
    return jnp.moveaxis(out, -1, axis)


def dst2(u):
    return dst_axis(dst_axis(u, -1), -2)


def eigenvalues(intervals, continuum=False):
    m = jnp.arange(1, intervals, dtype=jnp.float64)
    lam = (jnp.pi*m)**2 if continuum else 4*intervals**2*jnp.sin(jnp.pi*m/(2*intervals))**2
    return lam[:, None] + lam[None, :]


def negative_laplacian(u, intervals):
    v = jnp.pad(u, ((1, 1), (1, 1)))
    return intervals**2*(4*u-v[2:, 1:-1]-v[:-2, 1:-1]-v[1:-1, 2:]-v[1:-1, :-2])


@jax.jit
def propagate(u0, lam, times, diffusivity):
    coeff = dst2(u0)
    out = jax.vmap(lambda t: dst2(coeff*jnp.exp(-diffusivity*t*lam)))(times[1:])
    # Exact supplied initial field is part of the output contract.
    return jnp.concatenate((u0[None], out))


def cn_factor(lam, dt, diffusivity):
    return (1-dt*diffusivity*lam/2)/(1+dt*diffusivity*lam/2)


def mode_matrix(intervals, modes_per_axis):
    # Continuum-normalized tests and area weights: comparable scale across N.
    xy = jnp.asarray(coords(intervals))
    modes = jnp.arange(1, modes_per_axis+1, dtype=jnp.float64)
    sx = jnp.sqrt(2.0)*jnp.sin(jnp.pi*xy[:, :1]*modes)
    sy = jnp.sqrt(2.0)*jnp.sin(jnp.pi*xy[:, 1:]*modes)
    return (sx[:, :, None]*sy[:, None, :]).reshape(len(xy), -1)/intervals**2


def make_lm(head_fn, budget, tolerance, linear_solver="lu"):
    """Damped monotone LM with explicit arrays, relative stationarity stop.

    Solves ||matrix @ h(params,z)-target||. Normalized gradient is divided by
    ||J||*||target||; stationarity does not certify global optimality.
    Reasons: 0 budget, 1 stationarity, 2 tiny step, 3 damping limit, 4 nonfinite.
    """
    assert linear_solver in ("lu", "cholesky")
    def solve(params, matrix, target, z0):
        scale = jnp.maximum(jnp.linalg.norm(target), 1e-14)
        residual = lambda z: (matrix @ head_fn(params, z)-target)/scale
        def rj(z):
            return residual(z), jax.jacfwd(residual)(z)
        r, jac = rj(z0)
        value = jnp.dot(r, r)
        # z, residual, Jacobian, value, damping, attempts, accepted, reason
        state = (z0, r, jac, value, jnp.float64(1e-4), jnp.int32(0),
                 jnp.int32(0), jnp.int32(0))
        def condition(s):
            grad = jnp.linalg.norm(s[2].T@s[1])/jnp.maximum(jnp.linalg.norm(s[2]), 1e-30)
            return (s[5] < budget) & (s[7] == 0) & (grad > tolerance)
        def body(s):
            z, r, jac, val, damping, attempts, accepted, _ = s
            gram = jac.T@jac
            d = jnp.maximum(jnp.diag(gram), 1e-12)
            system = gram+damping*jnp.diag(d)
            rhs = -jac.T@r
            if linear_solver == "cholesky":
                lower = jnp.linalg.cholesky(system)
                step = jsl.solve_triangular(lower.T, jsl.solve_triangular(lower, rhs, lower=True), lower=False)
            else:
                step = jnp.linalg.solve(system, rhs)
            candidate = z+step
            rnew = residual(candidate)
            vnew = jnp.dot(rnew, rnew)
            accept = jnp.isfinite(vnew) & (vnew < val)
            znew = jnp.where(accept, candidate, z)
            rnew, jnew = jax.lax.cond(accept, lambda: rj(znew), lambda: (r, jac))
            damping = jnp.where(accept, jnp.maximum(damping/3, 1e-12), damping*10)
            tiny = jnp.linalg.norm(step) < 1e-12*(1+jnp.linalg.norm(z))
            reason = jnp.where(tiny, 2, jnp.where(damping > 1e12, 3, 0)).astype(jnp.int32)
            return znew, rnew, jnew, jnp.where(accept, vnew, val), damping, attempts+1, accepted+accept.astype(jnp.int32), reason
        z, r, jac, val, damping, attempts, accepted, reason = jax.lax.while_loop(condition, body, state)
        gradient = jnp.linalg.norm(jac.T@r)/jnp.maximum(jnp.linalg.norm(jac), 1e-30)
        reason = jnp.where(gradient <= tolerance, 1, reason)
        reason = jnp.where(jnp.isfinite(val), reason, 4).astype(jnp.int32)
        return z, jnp.array([attempts, accepted, reason, jnp.sqrt(val), gradient])
    return solve


def make_rollout(head_fn, step_budget, tolerance, nsteps, output_stride):
    solve = make_lm(head_fn, step_budget, tolerance)
    def rollout(params, matrix, factor, z0):
        def body(zprev, _):
            # Carry the DECODED CURRENT STATE, never the already-satisfied left side.
            target = factor*(matrix@head_fn(params, zprev))
            znext, info = solve(params, matrix, target, zprev)
            return znext, (znext, info)
        _, (zs, infos) = jax.lax.scan(body, z0, None, length=nsteps)
        outputs = jnp.concatenate((z0[None], zs[output_stride-1::output_stride]))
        return outputs, infos
    return jax.jit(rollout)


def make_query(cfg, dt):
    times = np.asarray(cfg["times"])
    nsteps = int(round(times[-1]/dt))
    stride = int(round((times[1]-times[0])/dt))
    assert np.allclose(np.arange(len(times))*stride*dt, times)
    fit = make_lm(sc.head, cfg["fit_budget"], cfg["gradient_tolerance"])
    rollout = make_rollout(sc.head, cfg["step_budget"], cfg["gradient_tolerance"], nsteps, stride)
    @jax.jit
    def initialize(params, projection, triangular, library, codes, u0):
        target = projection.T@u0.reshape(-1)
        nearest = jnp.argmin(jnp.sum((library-target)**2, axis=1))
        starts = jnp.stack((codes[nearest], jnp.mean(codes, axis=0)))
        zs, infos = jax.vmap(lambda z: fit(params, triangular, target, z))(starts)
        best = jnp.argmin(infos[:, 3])
        return zs[best], infos
    @jax.jit
    def output(params, bank, zs):
        return sc.head(params, zs)@bank.T
    return initialize, rollout, output


def error_metrics(fields, truth, intervals):
    a, b = np.asarray(fields), np.asarray(truth)
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    diff = np.linalg.norm(a-b, axis=1)
    norm = np.linalg.norm(b, axis=1)
    initial = norm[0]
    decay = norm/max(initial, 1e-300)
    return {"relative_current": (diff/np.maximum(norm, 1e-300)).tolist(),
            "relative_initial": (diff/max(initial, 1e-300)).tolist(),
            "absolute_l2": (diff/intervals).tolist(),
            "truth_norm_over_initial": decay.tolist(),
            "vanished_below_1e-3_initial": (decay < 1e-3).tolist(),
            "energy": (np.sum(a*a, axis=1)/intervals**2/2).tolist(),
            "state_change_from_initial": (np.linalg.norm(a-a[0], axis=1)/max(np.linalg.norm(a[0]), 1e-300)).tolist()}
