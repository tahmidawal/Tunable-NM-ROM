"""Phase-6 G1 Cox-weak/K3-full numerical-identity repair primitives."""
from __future__ import annotations

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase3 as p3
import b10_phase4 as p4
import b10_phase5 as p5
import b10_s0_spline as base


F64 = jnp.float64
G1 = dict(p5.CANDIDATES[0])
if G1["arm"] != "G1":  # prospective ordering is part of the method
    raise RuntimeError("Phase-5 G1 candidate ordering changed")


def make_online_kernels(n: int, num_steps: int):
    """Build the locked R0 control and actual R1 Cox-weak/K3-full route."""
    candidate = G1
    geometry = base.weak_geometry(n, p5.H1)
    stencil_coords = jnp.asarray(geometry["stencil_coords"], F64)
    stencil_mask = jnp.asarray(geometry["stencil_mask"], F64)
    phi_weighted = jnp.asarray(geometry["phi_weighted"], F64)
    eigenvalues = jnp.asarray(geometry["eigenvalues"], F64)
    coarse_table = jnp.asarray(p3.span_polynomial_table_np(48), F64)
    fine_table = jnp.asarray(p3.span_polynomial_table_np(32), F64)
    m = int(p5.H1["m"])
    full_pallas = p4.make_pallas_hierarchical_decoder(
        num_steps + 1, int(n) * int(n), p5.H1
    )

    def coefficient_values(generator, states, mean, scales):
        return p5.apply_generator(
            generator, states[:, 5:], mean, scales, candidate
        )

    def decode_one(state, coefficients, coords, mask, cox):
        if cox:
            return p4.decode_one_cox_jax(
                state, coefficients, coords, mask, p5.H1
            )
        return p4.decode_one_polynomial_jax(
            state, coefficients, coords, mask, p5.H1,
            coarse_table, fine_table,
        )

    def weak_one(
        state, coefficients, previous_state, previous_coefficients,
        viscosity, cox,
    ):
        current_flat = decode_one(
            state, coefficients, stencil_coords, stencil_mask, cox
        )
        previous = decode_one(
            previous_state, previous_coefficients,
            stencil_coords[::5], stencil_mask[::5], cox,
        )
        current = current_flat.reshape(m, 5)
        center, xp, xm, yp, ym = [current[:, index] for index in range(5)]
        dx = 1.0 / (int(n) - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected_u = phi_weighted.T @ center
        preconditioner = (1.0 + c.DT * viscosity * eigenvalues) ** -1.0
        residual = preconditioner * (
            phi_weighted.T @ (center - previous)
            + c.DT * (
                phi_weighted.T @ advection
                + viscosity * eigenvalues * projected_u
            )
        )
        denominator = jnp.maximum(
            jnp.linalg.norm(phi_weighted.T @ previous), 1e-12
        )
        rho = jnp.linalg.norm(residual) / denominator
        return current_flat, previous, residual, rho

    def weak_trajectory(states, values, viscosity, cox):
        return jax.vmap(
            lambda state, value, previous, previous_value: weak_one(
                state, value, previous, previous_value, viscosity, cox
            )
        )(states[1:], values[1:], states[:-1], values[:-1])

    def predict_states(predictor, features):
        return jnp.tanh(p5.s.apply_mlp(predictor, features[:num_steps + 1]))

    def full_decode(states, values, coords, mask, cox):
        if cox:
            return p4.decode_states_cox_sequential(
                states, values, coords, mask, p5.H1
            )
        return full_pallas(
            states, values, coords, mask, coarse_table, fine_table
        )

    def mandatory(
        predictor, generator, mean, scales, features, coords, mask, viscosity,
        cox_weak,
    ):
        states = predict_states(predictor, features)
        values = coefficient_values(generator, states, mean, scales)
        _, _, residual, rho = weak_trajectory(
            states, values, viscosity, cox_weak
        )
        fields = full_decode(states, values, coords, mask, False)
        return fields, states, values, residual, rho

    def identity_all(
        predictor, generator, mean, scales, features, coords, mask, viscosity,
    ):
        states = predict_states(predictor, features)
        values = coefficient_values(generator, states, mean, scales)
        cox_weak = weak_trajectory(states, values, viscosity, True)
        polynomial_weak = weak_trajectory(states, values, viscosity, False)
        cox_fields = full_decode(states, values, coords, mask, True)
        k3_fields = full_decode(states, values, coords, mask, False)
        control = (cox_fields,) + cox_weak
        r0 = (k3_fields,) + polynomial_weak
        # R1's actual weak path is the common Cox tuple. Sharing these values
        # makes the identity record test the real route without a second,
        # potentially differently fused recomputation.
        r1 = (k3_fields,) + cox_weak
        return control, r0, r1

    return {
        "R0_polynomial_weak": jax.jit(
            lambda *args: mandatory(*args, False)
        ),
        "R1_cox_weak_k3_full": jax.jit(
            lambda *args: mandatory(*args, True)
        ),
        "identity_all": jax.jit(identity_all),
        "geometry": geometry,
    }


def compile_online_kernels(
    n, features, viscosity, coefficient_mean, head_scales, num_steps,
):
    predictor = p5.init_cost_predictor()
    generator = p5.init_generator(G1, p5.COST_SEED, nonzero_bias=True)
    kernels = make_online_kernels(int(n), int(num_steps))
    arguments = (
        predictor, generator,
        jnp.asarray(coefficient_mean, F64), jnp.asarray(head_scales, F64),
        jnp.asarray(features, F64), jnp.asarray(c.grid_coords(int(n)), F64),
        jnp.asarray(c.binary_boundary_mask(int(n)), F64),
        jnp.asarray(viscosity, F64),
    )
    compiled, setup = {}, {}
    for name in (
        "R0_polynomial_weak", "R1_cox_weak_k3_full", "identity_all",
    ):
        lower_started = time.perf_counter()
        lowered = kernels[name].lower(*arguments)
        lower_s = time.perf_counter() - lower_started
        compile_started = time.perf_counter()
        executable = lowered.compile()
        compile_s = time.perf_counter() - compile_started
        compiled[name] = executable
        setup[name] = {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "lower_plus_compile_s": float(lower_s + compile_s),
            "memory_analysis": base.memory_analysis(executable),
        }
    return compiled, setup, kernels["geometry"], arguments


def parameter_count() -> int:
    generator = p5.init_generator(G1, p5.COST_SEED, nonzero_bias=True)
    return p5._parameter_count(generator)


def relative_l2(value, control) -> float:
    value, control = np.asarray(value), np.asarray(control)
    return float(
        np.linalg.norm(value - control)
        / max(np.linalg.norm(control), 1e-300)
    )
