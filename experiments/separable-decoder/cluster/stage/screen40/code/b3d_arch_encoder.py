"""Shared offline coefficient encoder with the unchanged width-128 decoder.

The encoder sees exact bank projections of training solution snapshots. It is
not a physical-parameter encoder, an online initializer, or an auxiliary loss.
The common evaluator reports direct encoding separately from multistart fits.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import b3d_arch_baseline as baseline

NAME = 'shared_encoder'
MODE = 'encoder'
CONFIG = dict(decoder_width=128, decoder_layers=2, encoder_width=128,
              encoder_layers=2, encoder_input='solution_q_coefficients',
              encoder_input_scale='centered_coefficient_rms',
              encoder_output_scale='initial_code_component_rms',
              encoder_online=False)


def init(key, shared):
    # Reuse the same initialization key, including the random hidden weights.
    # A stray control-arm HEAD_WIDTH must not silently change this experiment.
    if baseline.CONFIG['width'] != CONFIG['decoder_width']:
        raise ValueError('The shared-encoder arm requires HEAD_WIDTH=128')
    decoder, frozen = baseline.init(key, shared)
    r, k = shared['anchor'].shape
    encoder_key = jax.random.fold_in(key, 1)
    encoder = baseline.init_mlp(encoder_key, [r, 128, 128, k], zero_last=True)
    return dict(decoder=decoder, encoder_residual=encoder), frozen


def encode(p, frozen, a):
    centered = jnp.asarray(a) - frozen['center']
    correction = baseline.mlp(p['encoder_residual'], centered / frozen['output_scale'])
    return centered @ frozen['anchor'] + correction * frozen['latent_scale']


def encode_np(p, frozen, a):
    centered = np.asarray(a, dtype=np.float64) - np.asarray(frozen['center'])
    correction = baseline.mlp_np(p['encoder_residual'], centered / float(frozen['output_scale']))
    return centered @ np.asarray(frozen['anchor']) + correction * np.asarray(frozen['latent_scale'])


def apply(p, frozen, z):
    return baseline.apply(p['decoder'], frozen, z)


def apply_np(p, frozen, z):
    return baseline.apply_np(p['decoder'], frozen, z)


def diagnostics(p, frozen, z):
    del frozen, z
    count = lambda tree: sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree))
    return dict(decoder_parameter_count=count(p['decoder']),
                encoder_parameter_count=count(p['encoder_residual']),
                encoder_online_required=False)
