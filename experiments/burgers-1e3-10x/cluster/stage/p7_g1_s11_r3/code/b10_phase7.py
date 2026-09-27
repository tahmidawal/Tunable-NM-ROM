"""Exact Phase-7 G1 training primitives; no data access or job side effects."""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)
from jax import lax
import jax.numpy as jnp
import numpy as np

import b10_phase3 as p3
import b10_phase4 as p4
import b10_phase5 as p5


F64 = jnp.float64
G1 = dict(p5.CANDIDATES[0])
H1 = dict(p5.H1)
SEED = 11
GENERATOR_DRAW_COUNT = 20
ENCODER_PARAMETER_COUNT = 29_811
PREDICTOR_PARAMETER_COUNT = 2_104

if G1["arm"] != "G1" or G1["parameter_count"] != 30_594:
    raise RuntimeError("Phase-7 requires immutable G1 ordering/shape")


def parameter_count(tree) -> int:
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(tree)))


def init_generator(seed=SEED):
    """The locked f64 Xavier/zero-bias G1 initialization."""
    return p5.init_generator(G1, int(seed), nonzero_bias=False)


def _draw(root, index, shape, fan_in, fan_out, bias=False):
    if bias:
        return jnp.zeros(shape, F64)
    scale = math.sqrt(2.0 / (int(fan_in) + int(fan_out)))
    key = jax.random.fold_in(root, int(index))
    return scale * jax.random.normal(key, shape, dtype=F64)


def init_encoder(seed=SEED):
    """Locked mirrored G1 encoder with disjoint parameter-index folds."""
    root = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 1)
    index = GENERATOR_DRAW_COUNT

    def conv(in_channels, out_channels, kernel):
        nonlocal index
        fan_in = kernel * kernel * in_channels
        fan_out = kernel * kernel * out_channels
        layer = {
            "W": _draw(root, index, (kernel, kernel, in_channels, out_channels),
                       fan_in, fan_out),
            "b": _draw(root, index + 1, (out_channels,), fan_in, fan_out, True),
        }
        index += 2
        return layer

    def head():
        return {
            "input": conv(1, 16, 1),
            "blocks": tuple(conv(16, 16, 3) for _ in range(3)),
        }

    parameters = {"coarse": head(), "fine": head()}
    parameters["output"] = {
        "W": _draw(root, index, (832, 19), 832, 19),
        "b": _draw(root, index + 1, (19,), 832, 19, True),
    }
    if parameter_count(parameters) != ENCODER_PARAMETER_COUNT:
        raise AssertionError("Phase-7 encoder parameter count")
    return parameters


def _conv_valid(value, layer):
    return lax.conv_general_dilated(
        value, layer["W"], (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + layer["b"]


def _conv_reflect(value, layer):
    value = jnp.pad(value, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    return _conv_valid(value, layer)


def _encode_head(value, parameters):
    value = _conv_valid(value, parameters["input"])
    for layer in parameters["blocks"]:
        value = jax.nn.swish(_conv_reflect(value, layer))
        value = lax.reduce_window(
            value, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "VALID"
        ) / 4.0
    return value.reshape((value.shape[0], -1))


def apply_encoder(parameters, normalized_coefficients):
    values = jnp.asarray(normalized_coefficients, F64)
    coarse = values[:, :48 * 48].reshape((-1, 48, 48, 1))
    fine = values[:, 48 * 48:].reshape((-1, 32, 32, 1))
    hidden = jnp.concatenate((
        _encode_head(coarse, parameters["coarse"]),
        _encode_head(fine, parameters["fine"]),
    ), axis=1)
    raw_affine = hidden @ parameters["output"]["W"] + parameters["output"]["b"]
    q_raw = 3.0 * jnp.tanh(raw_affine / 3.0)
    return q_raw, jnp.tanh(q_raw)


def apply_generator(generator, q, coefficient_mean, head_scales):
    return p5.apply_generator(generator, q, coefficient_mean, head_scales, G1)


def normalized_generated_coefficients(generator, q, coefficient_mean, head_scales):
    coefficients = apply_generator(generator, q, coefficient_mean, head_scales)
    coarse = (
        coefficients[..., :48 * 48] - coefficient_mean[:48 * 48]
    ) / head_scales[0]
    fine = (
        coefficients[..., 48 * 48:] - coefficient_mean[48 * 48:]
    ) / head_scales[1]
    return jnp.concatenate((coarse, fine), axis=-1)


def decode_sample_batch(generator, states, coords, masks, coefficient_mean, head_scales):
    coefficients = apply_generator(generator, states[:, 5:], coefficient_mean, head_scales)
    return jax.vmap(
        lambda state, coefficient, xy, mask: p4.decode_one_cox_jax(
            state, coefficient, xy, mask, H1
        )
    )(states, coefficients, coords, masks)


def full_decode_cox(generator, states, coords, mask, coefficient_mean, head_scales):
    coefficients = apply_generator(generator, states[:, 5:], coefficient_mean, head_scales)
    return jax.vmap(
        lambda state, coefficient: p4.decode_one_cox_jax(
            state, coefficient, coords, mask, H1
        )
    )(states, coefficients)


def full_decode_k3_factory(batch_size, point_count):
    decoder = p4.make_pallas_hierarchical_decoder(batch_size, point_count, H1)
    coarse_table = jnp.asarray(p3.span_polynomial_table_np(48), F64)
    fine_table = jnp.asarray(p3.span_polynomial_table_np(32), F64)

    @jax.jit
    def evaluate(generator, states, coords, mask, coefficient_mean, head_scales):
        coefficients = apply_generator(
            generator, states[:, 5:], coefficient_mean, head_scales
        )
        return decoder(
            states, coefficients, coords, mask, coarse_table, fine_table
        )

    return evaluate


def predictor_parameter_count(parameters) -> int:
    observed = parameter_count(parameters)
    if observed != PREDICTOR_PARAMETER_COUNT:
        raise AssertionError("Phase-7 predictor parameter count")
    return observed
