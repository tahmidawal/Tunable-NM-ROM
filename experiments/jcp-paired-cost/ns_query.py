"""Compiled NS full queries; arrays are explicit arguments, no model constants.

Adaptive LM follows P1's diagnostic reference algorithm. It is a candidate
implementation only until paired field/termination parity is measured on the
approved development cases. This file does not authorize any experiment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import head_rom as H
import ns3d_fom as F
import shift_rom as SR


def stationarity(residual, jacobian):
    denominator = jnp.maximum(jnp.linalg.norm(jacobian, axis=0), 1e-30)
    denominator *= jnp.maximum(jnp.linalg.norm(residual), 1e-30)
    return jnp.max(jnp.abs(jacobian.T @ residual) / denominator)


def choose_stationary_fit(fit_info):
    raw = jnp.argmin(fit_info[:, 2])
    band = 1024*jnp.finfo(jnp.float64).eps*jnp.maximum(fit_info[raw, 2], 1e-300)
    eligible = (fit_info[:, 0] == 1) & (fit_info[:, 2] <= fit_info[raw, 2]+band)
    stationary_choice = jnp.argmin(jnp.where(eligible, fit_info[:, 2], jnp.inf))
    choice = jnp.where((fit_info[raw, 0] != 1) & jnp.any(eligible), stationary_choice, raw)
    return choice, raw, band


def make_lm(evaluate, budget, roundoff_repair):
    """Return solution and [reason,gradient,cost,iterations,accepted,roundoff].

    Reasons: 1 stationarity, 2 stagnation, 3 nonfinite, 4 iteration budget.
    A tiny step or cost-only improvement never certifies convergence.
    """
    def solve(x0, arguments, gtol):
        res, jac = evaluate(x0, *arguments)
        cost = .5*jnp.dot(res, res)
        gradient = stationarity(res, jac)
        finite = jnp.isfinite(cost+gradient) & jnp.all(jnp.isfinite(jac))
        reason = jnp.where(~finite, 3, jnp.where(gradient <= gtol, 1, 0))

        def body(carry):
            x, res, jac, cost, gradient, damping, count, accepted, repaired, reason = carry
            scales = jnp.maximum(jnp.linalg.norm(jac, axis=0), 1e-30)
            normalized = jac / scales
            matrix = normalized.T @ normalized + damping*jnp.eye(jac.shape[1])
            delta = jnp.linalg.solve(matrix, -normalized.T @ res) / scales
            candidate = x + delta
            tr, tj = evaluate(candidate, *arguments)
            trial_cost = .5*jnp.dot(tr, tr)
            trial_gradient = stationarity(tr, tj)
            trial_finite = (jnp.isfinite(trial_cost+trial_gradient)
                            & jnp.all(jnp.isfinite(tj)) & jnp.all(jnp.isfinite(delta)))
            band = 1024*jnp.finfo(jnp.float64).eps*jnp.maximum(cost, trial_cost)
            tiny_repair = (roundoff_repair & (gradient <= 1e-5)
                          & (trial_cost <= cost+band) & (trial_gradient <= .5*gradient)
                          & (jnp.linalg.norm(delta) <= 1e-6*(1+jnp.linalg.norm(x))))
            accept = trial_finite & ((trial_cost < cost) | tiny_repair)
            repair_used = accept & (trial_cost >= cost)
            x = jnp.where(accept, candidate, x)
            res = jnp.where(accept, tr, res)
            jac = jnp.where(accept, tj, jac)
            cost = jnp.where(accept, trial_cost, cost)
            gradient = jnp.where(accept, trial_gradient, gradient)
            damping = jnp.where(accept, jnp.maximum(damping/3, 1e-14),
                                jnp.minimum(damping*10, 1e18))
            tiny = jnp.linalg.norm(delta) <= 1e-15*(1+jnp.linalg.norm(x))
            reason = jnp.where(gradient <= gtol, 1,
                               jnp.where(tiny | (damping >= 1e18), 2, 0))
            reason = jnp.where(jnp.isfinite(cost+gradient), reason, 3)
            return (x, res, jac, cost, gradient, damping, count+1,
                    accepted+accept, repaired+repair_used, reason)

        state = jax.lax.while_loop(lambda s: (s[6] < budget) & (s[9] == 0), body,
            (x0, res, jac, cost, gradient, jnp.float64(1e-4), jnp.int32(0),
             jnp.int32(0), jnp.int32(0), reason))
        x, _, _, cost, gradient, _, count, accepted, repaired, reason = state
        reason = jnp.where(reason == 0, 4, reason)
        return x, jnp.array([reason, gradient, cost, count, accepted, repaired])
    return solve


def make_query(kind, *, n, rank, latent, dt, steps, output_every, budget,
               fit_starts, roundoff_repair):
    if kind not in ("head", "span") or steps % output_every:
        raise ValueError("invalid query kind or output schedule")
    _, prepare, residual_jacobian = SR.make_frozen_step(dt, n, rank, 1, 1e-6)

    def coeff(w, params):
        return H.head_apply(params, w[:latent]) if kind == "head" else w[:rank]

    def evaluate(w, previous, operators, params):
        if kind == "span":
            return residual_jacobian(w, previous, operators)
        a = coeff(w, params)
        residual, jac = residual_jacobian(jnp.r_[a, w[latent:]], previous, operators)
        return residual, jnp.concatenate((jac[:, :rank] @ H.head_jacobian(params, w[:latent]),
                                           jac[:, rank:]), axis=1)

    def fit_evaluate(code, target, params):
        return H.head_apply(params, code)-target, H.head_jacobian(params, code)

    lm = make_lm(evaluate, budget, roundoff_repair)
    fit_lm = make_lm(fit_evaluate, budget, roundoff_repair)

    @jax.jit
    def query(u0, viscosity, basis, A, T, lam, Dd, params, codes, head_candidates,
              gtol, fit_gtol):
        operators = prepare(viscosity, A, T, lam, Dd)
        center = SR.grid_centroid(u0)
        projected = basis.T @ SR.shift_field(u0, -center*n).ravel()
        if kind == "head":
            distances = jnp.sum((head_candidates-projected[None])**2, axis=1)
            starts = codes[jnp.argsort(distances, stable=True)[:fit_starts]]
            solutions, fit_info = jax.vmap(lambda z: fit_lm(z, (projected, params), fit_gtol))(starts)
            choice, raw, band = choose_stationary_fit(fit_info)
            initial = jnp.r_[solutions[choice], jnp.zeros(3)]
            coefficient_gap = jnp.linalg.norm(H.head_apply(params, solutions[choice])-H.head_apply(params, solutions[raw]))
            ic_info = jnp.r_[fit_info[choice], raw, choice, band, coefficient_gap]
        else:
            initial = jnp.r_[projected, jnp.zeros(3)]
            ic_info = jnp.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        def decode(w, center):
            field = (basis @ coeff(w, params)).reshape(3, n, n, n)
            return SR.shift_field(field, center*n)

        def step(carry, _):
            w, old, center = carry
            new, info = lm(w+(w-old), (coeff(w, params), operators, params), gtol)
            center = (center+new[-3:]) % 1.
            return (new, w, center), info

        def output_block(carry, _):
            carry, infos = jax.lax.scan(step, carry, None, length=output_every)
            w, _, center = carry
            return carry, (decode(w, center), infos)

        _, (fields, infos) = jax.lax.scan(output_block, (initial, initial, center), None,
                                         length=steps//output_every)
        return jnp.concatenate((decode(initial, center)[None], fields)), infos.reshape(steps, 6), ic_info
    return query


def spec_resample(u, n_to):
    """Pinned confirm_mesh spectral restriction/interpolation, including its Nyquist rule."""
    n = u.shape[-1]
    if n == n_to:
        return u
    m = min(n, n_to)//2
    spectrum = jnp.fft.fftn(u, axes=(-3, -2, -1), norm="forward")
    for axis in (-3, -2, -1):
        low = jax.lax.slice_in_dim(spectrum, 0, m, axis=axis)
        high = jax.lax.slice_in_dim(spectrum, n-m+1, n, axis=axis)
        shape = list(spectrum.shape)
        shape[axis] = n_to-(2*m-1)
        spectrum = jnp.concatenate((low, jnp.zeros(shape, dtype=spectrum.dtype), high), axis=axis)
    return jnp.fft.ifftn(spectrum, axes=(-3, -2, -1), norm="forward").real


def make_classical(*, scoring_mesh, solver_mesh, horizon, steps, scheme):
    if steps % 5:
        raise ValueError("six equally spaced requested outputs required")
    solver = F.make_solver(horizon/steps, steps, steps//5, scheme=scheme)

    @jax.jit
    def query(u0, viscosity, geometry):
        initial = spec_resample(u0, solver_mesh)
        return spec_resample(solver(initial, viscosity, geometry), scoring_mesh)
    return query
