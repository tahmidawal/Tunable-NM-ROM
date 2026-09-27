"""Exact Phase-5 nonlinear H1 coefficient generators and online kernels."""
from __future__ import annotations

import math
import time

import jax

jax.config.update("jax_enable_x64", True)
from jax import lax
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase3 as p3
import b10_phase4 as p4
import b10_s0_spline as base
import b10_spline as s


F64 = jnp.float64
H1 = p4.CANDIDATES[0]
COST_SEED = 20_260_826
GEOMETRY_SEED = 20_260_827
RANK_TOL = 1e-10
CURVATURE_TOL = 1e-8

CANDIDATES = (
    {
        "arm": "G1", "name": "DualUpConv16", "channels": 16,
        "residual": False, "parameter_count": 30_594,
        "encoder_parameter_count": 29_811, "k": 24, "q": 19,
        "R": 48, "P": 32, "M": 96, "m": 384,
        "mac_per_state": 10_132_928,
        "mac_per_51_state_trajectory": 516_779_328,
    },
    {
        "arm": "G2", "name": "DualResUpConv32", "channels": 32,
        "residual": True, "parameter_count": 144_322,
        "encoder_parameter_count": 142_739, "k": 24, "q": 19,
        "R": 48, "P": 32, "M": 96, "m": 384,
        "mac_per_state": 80_649_088,
        "mac_per_51_state_trajectory": 4_113_103_488,
    },
)


def _parameter_count(tree):
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(tree)))


def _initializer(seed, arm_code, nonzero_bias):
    root = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(arm_code))
    counter = 0

    def normal(shape, fan_in, fan_out, bias=False):
        nonlocal counter
        key = jax.random.fold_in(root, counter)
        counter += 1
        if bias:
            if not nonzero_bias:
                return jnp.zeros(shape, F64)
            scale = 0.01 / math.sqrt(max(int(fan_in), 1))
        else:
            scale = math.sqrt(2.0 / (int(fan_in) + int(fan_out)))
        return scale * jax.random.normal(key, shape, dtype=F64)

    return normal


def init_generator(candidate, seed, *, nonzero_bias):
    """Initialize the exact G1/G2 generator; all arrays are f64."""
    channels = int(candidate["channels"])
    arm_code = 1 if candidate["arm"] == "G1" else 2
    draw = _initializer(seed, arm_code, nonzero_bias)

    def dense_seed(height, width):
        outputs = height * width * channels
        return {
            "W": draw((19, outputs), 19, outputs),
            "b": draw((outputs,), 19, outputs, bias=True),
            "shape": (height, width, channels),
        }

    def conv():
        return {
            "W": draw((3, 3, channels, channels), 9 * channels, 9 * channels),
            "b": draw((channels,), 9 * channels, 9 * channels, bias=True),
        }

    block_depth = 2 if candidate["residual"] else 1
    parameters = {
        "coarse_seed": dense_seed(6, 6),
        "fine_seed": dense_seed(4, 4),
        "coarse_blocks": tuple(
            tuple(conv() for _ in range(block_depth)) for _ in range(3)
        ),
        "fine_blocks": tuple(
            tuple(conv() for _ in range(block_depth)) for _ in range(3)
        ),
        "coarse_out": {
            "W": draw((1, 1, channels, 1), channels, 1),
            "b": draw((1,), channels, 1, bias=True),
        },
        "fine_out": {
            "W": draw((1, 1, channels, 1), channels, 1),
            "b": draw((1,), channels, 1, bias=True),
        },
    }
    # Static reshape metadata are not parameters and must not enter a JAX pytree.
    parameters["coarse_seed"].pop("shape")
    parameters["fine_seed"].pop("shape")
    observed = _parameter_count(parameters)
    if observed != int(candidate["parameter_count"]):
        raise AssertionError((candidate["arm"], observed, candidate["parameter_count"]))
    return parameters


def init_cost_predictor(seed=COST_SEED):
    """Deterministic nonzero-bias f64 predictor used only by P5-D cost."""
    draw = _initializer(seed, 0, True)
    dimensions = (7, 32, 32, 24)
    layers = []
    for n_in, n_out in zip(dimensions[:-1], dimensions[1:]):
        layers.append({
            "W": draw((n_in, n_out), n_in, n_out),
            "b": draw((n_out,), n_in, n_out, bias=True),
        })
    if _parameter_count(tuple(layers)) != 2_104:
        raise AssertionError("predictor parameter count")
    return tuple(layers)


def _conv_valid(value, layer):
    return lax.conv_general_dilated(
        value, layer["W"], (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + layer["b"]


def _conv_reflect(value, layer):
    padded = jnp.pad(value, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    return _conv_valid(padded, layer)


def _head(q_flat, seed, blocks, output, height, width, residual):
    channels = seed["b"].size // (height * width)
    value = (q_flat @ seed["W"] + seed["b"]).reshape(
        q_flat.shape[0], height, width, channels
    )
    for block in blocks:
        value = jnp.repeat(jnp.repeat(value, 2, axis=1), 2, axis=2)
        if residual:
            hidden = jax.nn.swish(_conv_reflect(value, block[0]))
            value = jax.nn.swish(value + _conv_reflect(hidden, block[1]))
        else:
            value = jax.nn.swish(_conv_reflect(value, block[0]))
    return _conv_valid(value, output)[..., 0]


def apply_generator(parameters, q, coefficient_mean, head_scales, candidate):
    """Map bounded q to denormalized row-major R48/P32 coefficient grids."""
    q = jnp.asarray(q, F64)
    prefix = q.shape[:-1]
    flat = q.reshape((-1, 19))
    coarse = _head(
        flat, parameters["coarse_seed"], parameters["coarse_blocks"],
        parameters["coarse_out"], 6, 6, bool(candidate["residual"]),
    ).reshape((flat.shape[0], 48 * 48))
    fine = _head(
        flat, parameters["fine_seed"], parameters["fine_blocks"],
        parameters["fine_out"], 4, 4, bool(candidate["residual"]),
    ).reshape((flat.shape[0], 32 * 32))
    normalized = jnp.concatenate((
        coarse * head_scales[0], fine * head_scales[1]
    ), axis=1)
    coefficients = normalized + coefficient_mean[None, :]
    return coefficients.reshape(prefix + (48 * 48 + 32 * 32,))


def output_geometry(candidate, parameters, coefficient_mean, head_scales):
    """Exact preregistered generated-output rank/curvature non-collapse probe."""
    rng = np.random.default_rng(GEOMETRY_SEED)
    q = np.tanh(rng.normal(0.0, 0.5, size=(128, 19))).astype(np.float64)
    midpoint = 0.5 * (q[0::2] + q[1::2])
    evaluate = jax.jit(
        lambda params, values, mean, scales: apply_generator(
            params, values, mean, scales, candidate
        )
    )
    coefficients = np.asarray(evaluate(
        parameters, jnp.asarray(q), jnp.asarray(coefficient_mean),
        jnp.asarray(head_scales),
    ))
    middle = np.asarray(evaluate(
        parameters, jnp.asarray(midpoint), jnp.asarray(coefficient_mean),
        jnp.asarray(head_scales),
    ))
    centered = coefficients - np.mean(coefficients, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    ratios = singular / max(float(singular[0]), 1e-300)
    curvature_numerator = np.linalg.norm(
        middle - 0.5 * (coefficients[0::2] + coefficients[1::2]), axis=1
    )
    curvature_denominator = np.maximum(
        np.linalg.norm(coefficients[0::2] - coefficients[1::2], axis=1), 1e-12
    )
    curvature = curvature_numerator / curvature_denominator
    rank = int(np.sum(ratios > RANK_TOL))
    median = float(np.median(curvature))
    finite = bool(
        np.all(np.isfinite(coefficients)) and np.all(np.isfinite(singular))
        and np.all(np.isfinite(curvature))
    )
    return {
        "q_seed": GEOMETRY_SEED,
        "q_definition": "tanh(PCG64 Normal(0,0.5^2)), shape 128x19",
        "coefficient_shape": list(coefficients.shape),
        "singular_value_ratios": ratios.tolist(),
        "rank_relative_tolerance": RANK_TOL,
        "rank_at_tolerance": rank,
        "curvature_definition": (
            "||G(midpoint)-0.5*(G(a)+G(b))||/max(||G(a)-G(b)||,1e-12)"
        ),
        "curvature_all": curvature.tolist(),
        "curvature_median": median,
        "curvature_tolerance": CURVATURE_TOL,
        "finite": finite,
        "pass": bool(finite and rank >= 33 and median >= CURVATURE_TOL),
    }


def make_online_kernels(n, candidate, num_steps):
    """Build P5 mandatory/max-one K3 and independent Cox paths."""
    geometry = base.weak_geometry(n, H1)
    stencil_coords = jnp.asarray(geometry["stencil_coords"], F64)
    stencil_mask = jnp.asarray(geometry["stencil_mask"], F64)
    phi_weighted = jnp.asarray(geometry["phi_weighted"], F64)
    eigenvalues = jnp.asarray(geometry["eigenvalues"], F64)
    coarse_table = jnp.asarray(p3.span_polynomial_table_np(48), F64)
    fine_table = jnp.asarray(p3.span_polynomial_table_np(32), F64)
    m = int(H1["m"])
    full_pallas = p4.make_pallas_hierarchical_decoder(num_steps + 1, n * n, H1)

    def coefficients(generator, states, mean, scales):
        return apply_generator(generator, states[:, 5:], mean, scales, candidate)

    def decode_one(state, coefficient, query_coords, query_mask, cox=False):
        if cox:
            return p4.decode_one_cox_jax(state, coefficient, query_coords, query_mask, H1)
        return p4.decode_one_polynomial_jax(
            state, coefficient, query_coords, query_mask, H1,
            coarse_table, fine_table,
        )

    def weak_residual(
        state, state_coefficients, previous_state, previous_coefficients,
        viscosity, cox=False,
    ):
        current = decode_one(
            state, state_coefficients, stencil_coords, stencil_mask, cox
        ).reshape(m, 5)
        previous = decode_one(
            previous_state, previous_coefficients,
            stencil_coords[::5], stencil_mask[::5], cox,
        )
        center, xp, xm, yp, ym = [current[:, index] for index in range(5)]
        dx = 1.0 / (n - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected_u = phi_weighted.T @ center
        preconditioner = (1.0 + c.DT * viscosity * eigenvalues) ** -1.0
        residual = preconditioner * (
            phi_weighted.T @ (center - previous)
            + c.DT * (phi_weighted.T @ advection + viscosity * eigenvalues * projected_u)
        )
        denominator = jnp.maximum(jnp.linalg.norm(phi_weighted.T @ previous), 1e-12)
        return residual, jnp.linalg.norm(residual) / denominator

    def predict_states(predictor, features):
        return jnp.tanh(s.apply_mlp(predictor, features[:num_steps + 1]))

    def full_decode(states, values, coords, mask, cox=False):
        if cox:
            return p4.decode_states_cox_sequential(states, values, coords, mask, H1)
        return full_pallas(states, values, coords, mask, coarse_table, fine_table)

    def mandatory(predictor, generator, mean, scales, features, coords, mask, viscosity):
        states = predict_states(predictor, features)
        values = coefficients(generator, states, mean, scales)
        residual, rho = jax.vmap(
            lambda state, coeff, previous, previous_coeff: weak_residual(
                state, coeff, previous, previous_coeff, viscosity, False
            )
        )(states[1:], values[1:], states[:-1], values[:-1])
        fields = full_decode(states, values, coords, mask, False)
        return fields, states, values, residual, rho

    def maximum_one(predictor, generator, mean, scales, features, coords, mask, viscosity):
        predicted = predict_states(predictor, features)

        def one_coeff(state):
            return apply_generator(generator, state[5:], mean, scales, candidate)

        def step(previous, target):
            previous_coefficients = one_coeff(previous)
            target_coefficients = one_coeff(target)
            residual, rho_before = weak_residual(
                target, target_coefficients, previous, previous_coefficients,
                viscosity, False,
            )
            jacobian = jax.jacfwd(
                lambda state: weak_residual(
                    state, one_coeff(state), previous, previous_coefficients,
                    viscosity, False,
                )[0]
            )(target)
            hessian = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            diagonal = jnp.diag(jnp.diag(hessian)) + 1e-12 * jnp.eye(target.size, dtype=F64)
            raw_step = jnp.linalg.solve(hessian + 1e-6 * diagonal, -gradient)
            raw_step = jnp.where(
                jnp.all(jnp.isfinite(raw_step)), raw_step, jnp.zeros_like(raw_step)
            )
            scale = jnp.minimum(1.0, 0.25 / jnp.maximum(jnp.linalg.norm(raw_step), 1e-300))
            bounded = raw_step * scale
            factors = jnp.asarray((1.0, 0.5, 0.25, 0.0), F64)
            trials = jnp.clip(
                target[None] + factors[:, None] * bounded[None],
                -1.0 + 1e-8, 1.0 - 1e-8,
            )
            trial_rho = jax.vmap(
                lambda state: weak_residual(
                    state, one_coeff(state), previous, previous_coefficients,
                    viscosity, False,
                )[1]
            )(trials)
            trial_rho = jnp.where(jnp.isfinite(trial_rho), trial_rho, jnp.inf)
            best = jnp.argmin(trial_rho)
            accepted = trial_rho[best] < rho_before
            corrected = jnp.where(accepted, trials[best], target)
            return corrected, (
                corrected, residual, rho_before, trial_rho[best], jacobian,
                factors[best], jnp.linalg.norm(bounded), accepted,
            )

        _, outputs = jax.lax.scan(step, predicted[0], predicted[1:])
        corrected = jnp.concatenate((predicted[:1], outputs[0]), axis=0)
        values = coefficients(generator, corrected, mean, scales)
        fields = full_decode(corrected, values, coords, mask, False)
        return (fields, corrected, values) + outputs[1:]

    def identity(
        predictor, generator, mean, scales, features, coords, mask, viscosity, cox,
    ):
        states = predict_states(predictor, features)
        values = coefficients(generator, states, mean, scales)
        residual, rho = jax.vmap(
            lambda state, coeff, previous, previous_coeff: weak_residual(
                state, coeff, previous, previous_coeff, viscosity, cox
            )
        )(states[1:], values[1:], states[:-1], values[:-1])
        current = jax.lax.map(
            lambda pair: decode_one(pair[0], pair[1], stencil_coords, stencil_mask, cox),
            (states[1:], values[1:]),
        )
        previous = jax.lax.map(
            lambda pair: decode_one(
                pair[0], pair[1], stencil_coords[::5], stencil_mask[::5], cox
            ),
            (states[:-1], values[:-1]),
        )
        fields = full_decode(states, values, coords, mask, cox)
        return fields, current, previous, residual, rho

    return {
        "mandatory": jax.jit(mandatory),
        "maximum_one": jax.jit(maximum_one),
        "identity_k3": jax.jit(lambda *args: identity(*args, False)),
        "identity_cox": jax.jit(lambda *args: identity(*args, True)),
        "geometry": geometry,
    }


def compile_online_kernels(
    n, candidate, features, viscosity, coefficient_mean, head_scales, num_steps,
):
    predictor = init_cost_predictor()
    generator = init_generator(candidate, COST_SEED, nonzero_bias=True)
    kernels = make_online_kernels(n, candidate, num_steps)
    arguments = (
        predictor, generator,
        jnp.asarray(coefficient_mean, F64), jnp.asarray(head_scales, F64),
        jnp.asarray(features, F64), jnp.asarray(c.grid_coords(n), F64),
        jnp.asarray(c.binary_boundary_mask(n), F64), jnp.asarray(viscosity, F64),
    )
    compiled, setup = {}, {}
    for name in ("mandatory", "maximum_one", "identity_k3", "identity_cox"):
        lower_started = time.perf_counter()
        lowered = kernels[name].lower(*arguments)
        lower_s = time.perf_counter() - lower_started
        compile_started = time.perf_counter()
        executable = lowered.compile()
        compile_s = time.perf_counter() - compile_started
        compiled[name] = executable
        setup[name] = {
            "lower_s": float(lower_s), "compile_s": float(compile_s),
            "lower_plus_compile_s": float(lower_s + compile_s),
            "memory_analysis": base.memory_analysis(executable),
        }
    return compiled, setup, kernels["geometry"], arguments
