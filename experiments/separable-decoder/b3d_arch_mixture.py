"""Smooth two-expert residual head in exact orthonormal-bank coordinates.

Routing depends only on normalized latent variables and is differentiated with
the experts. There is no balancing loss, top-k dispatch, temperature annealing,
or physical-parameter input. Both output layers start at zero, giving the same
affine initial map as the matched MLP control despite independent hidden layers.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from b3d_arch_baseline import init_mlp, mlp, mlp_np


NAME = 'smooth_mixture'
MODE = 'codes'
CONFIG = dict(experts=2, width=128, layers=2, gate='affine_softmax',
              gate_initial_rms=.1, temperature=1., balancing_loss=False,
              routing_collapse_min_mean_usage=.05,
              identical_expert_relative_rms=1e-6)


def init(key, shared):
    k, r = shared['anchor'].shape[1], shared['anchor'].shape[0]
    keys = jax.random.split(key, CONFIG['experts'] + 1)
    p = dict(
        linear=jnp.asarray(shared['anchor']), bias=jnp.asarray(shared['center']),
        experts=[init_mlp(keys[i], [k, CONFIG['width'], CONFIG['width'], r])
                 for i in range(CONFIG['experts'])],
        gate=(jax.random.normal(keys[-1], (k, CONFIG['experts']), dtype=jnp.float64)
              * CONFIG['gate_initial_rms'] / np.sqrt(k),
              jnp.zeros(CONFIG['experts'], dtype=jnp.float64)))
    return p, {name: jnp.asarray(value) for name, value in shared.items()}


def routing(p, f, z):
    w, b = p['gate']
    return jax.nn.softmax((z / f['latent_scale']) @ w + b, axis=-1)


def expert_outputs(p, f, z):
    """Physical QR-coordinate residuals; expert axis is the penultimate axis."""
    x = z / f['latent_scale']
    return f['output_scale'] * jnp.stack([mlp(expert, x) for expert in p['experts']], axis=-2)


def apply(p, f, z):
    residual = jnp.sum(routing(p, f, z)[..., :, None] * expert_outputs(p, f, z), axis=-2)
    return p['bias'] + z @ p['linear'].T + residual


def parts_np(p, f, z):
    """Independent stable NumPy routing and expert evaluation."""
    x = np.asarray(z, dtype=np.float64) / np.asarray(f['latent_scale'])
    w, b = p['gate']
    logits = x @ np.asarray(w) + np.asarray(b)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    weights = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    experts = float(f['output_scale']) * np.stack(
        [mlp_np(expert, x) for expert in p['experts']], axis=-2)
    return weights, experts


def apply_np(p, f, z):
    weights, experts = parts_np(p, f, z)
    return (np.asarray(p['bias']) + np.asarray(z) @ np.asarray(p['linear']).T
            + np.sum(weights[..., :, None] * experts, axis=-2))


def diagnostics(p, f, z):
    """Validation-only descriptive flags; neither flag is an acceptance gate.

    Routing collapse means an expert receives less than five percent of average
    soft probability mass on this cohort. Identical-expert collapse means their
    RMS output disagreement is negligible relative to the fixed training output
    scale. The two can occur independently and do not establish global collapse.
    """
    zz = np.asarray(z, dtype=np.float64)
    assert zz.ndim == 2 and len(zz) > 0
    weights, experts = parts_np(p, f, zz)
    assert np.all(np.isfinite(weights)) and np.all(np.isfinite(experts))
    mean_usage = np.mean(weights, axis=0)
    maximum = np.max(weights, axis=1)
    entropy = -np.sum(weights * np.log(np.maximum(weights, np.finfo(float).tiny)), axis=1)
    disagreement = np.linalg.norm(experts[:, 0] - experts[:, 1], axis=1)
    reference_scale = float(f['output_scale']) * np.sqrt(experts.shape[-1])
    assert reference_scale > 0
    relative_disagreement = disagreement / reference_scale
    rms_disagreement = float(np.sqrt(np.mean(relative_disagreement**2)))
    return dict(
        routing_weights=weights, mean_usage=mean_usage,
        hard_assignment_counts=np.bincount(np.argmax(weights, axis=1), minlength=2),
        routing_entropy=entropy, mean_entropy=float(np.mean(entropy)),
        entropy_units='nats', maximum_routing_weight=maximum,
        maximum_routing_weight_quantiles=dict(zip(
            ['median', 'p95', 'maximum'], [float(x) for x in np.quantile(maximum, [.5, .95, 1.])])),
        saturation_counts={str(threshold): int(np.sum(maximum > threshold)) for threshold in [.95, .99]},
        expert_disagreement_absolute=disagreement,
        expert_disagreement_relative_to_training_scale=relative_disagreement,
        expert_disagreement_relative_rms=rms_disagreement,
        expert_disagreement_reference_scale=reference_scale,
        routing_collapse=bool(np.min(mean_usage) < CONFIG['routing_collapse_min_mean_usage']),
        identical_expert_collapse=bool(rms_disagreement <= CONFIG['identical_expert_relative_rms']),
        thresholds=dict(routing_min_mean_usage=CONFIG['routing_collapse_min_mean_usage'],
                        identical_expert_relative_rms=CONFIG['identical_expert_relative_rms']),
        interpretation='Descriptive flags on these fitted validation codes, not global claims or acceptance gates.')
