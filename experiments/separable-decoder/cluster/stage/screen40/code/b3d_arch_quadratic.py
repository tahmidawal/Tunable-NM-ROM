"""Explicit quadratic head in the fixed bank's orthonormal coordinates.

There is one undoubled x[i]*x[j] feature per i <= j, in lexicographic order.
Affine coefficients and quadratic weights are trainable; normalization is frozen.
Initialization is deterministic, so campaign seeds are optimizer repeats only.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

NAME = 'quadratic_head'
MODE = 'codes'
CONFIG = dict(degree=2, pairs='lexicographic_i_le_j', mixed_feature_factor=1,
              initialization='common_affine_zero_quadratic',
              repeat_kind='optimizer_minibatch_repeats')


def init(key, shared):
    del key  # The common affine map and zero curvature have no random weights.
    anchor = np.asarray(shared['anchor'], dtype=np.float64)
    center = np.asarray(shared['center'], dtype=np.float64)
    scale = np.asarray(shared['latent_scale'], dtype=np.float64)
    output_scale = np.asarray(shared['output_scale'], dtype=np.float64)
    if anchor.ndim != 2:
        raise ValueError('anchor must have shape (R, K)')
    r, k = anchor.shape
    if center.shape != (r,) or scale.shape != (k,) or output_scale.shape != ():
        raise ValueError('center, latent_scale, and output_scale have inconsistent shapes')
    if not all(np.all(np.isfinite(a)) for a in (anchor, center, scale, output_scale)):
        raise ValueError('initialization and normalization must be finite')
    if np.any(scale <= 0) or output_scale <= 0:
        raise ValueError('normalization scales must be positive')
    p = dict(linear=jnp.asarray(anchor), bias=jnp.asarray(center),
             quadratic=jnp.zeros((k * (k + 1) // 2, r), dtype=jnp.float64))
    frozen = dict(latent_scale=jnp.asarray(scale), output_scale=jnp.asarray(output_scale))
    return p, frozen


def pair_products(x):
    """Upper-triangular products without doubling mixed pairs; preserves batches."""
    i, j = jnp.triu_indices(x.shape[-1])
    return x[..., i] * x[..., j]


def apply(p, frozen, z):
    z = jnp.asarray(z, dtype=jnp.float64)
    products = pair_products(z / frozen['latent_scale'])
    return (p['bias'] + z @ p['linear'].T
            + frozen['output_scale'] * (products @ p['quadratic']))


def apply_np(p, frozen, z):
    """Independent NumPy reference: accumulate polynomial terms directly."""
    z = np.asarray(z, dtype=np.float64)
    x = z / np.asarray(frozen['latent_scale'])
    quadratic = np.asarray(p['quadratic'])
    nonlinear = np.zeros(z.shape[:-1] + (quadratic.shape[1],), dtype=np.float64)
    pair = 0
    for i in range(z.shape[-1]):
        for j in range(i, z.shape[-1]):
            nonlinear += (x[..., i] * x[..., j])[..., None] * quadratic[pair]
            pair += 1
    return (np.asarray(p['bias']) + z @ np.asarray(p['linear']).T
            + float(frozen['output_scale']) * nonlinear)
