"""Matched MLP control for the architecture campaign, with independent NumPy path.

Model modules expose init(key, shared)->(trainable,frozen), apply(p,f,z), and
apply_np(p,f,z). MODE is 'codes' or 'encoder'; the latter also defines encode.
Only trainable receives optimizer updates. shared contains training-only arrays.
All outputs are exact orthonormal-bank coordinates, not raw bank coefficients.
"""
import numpy as np
import os
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

NAME = 'mlp_control'
MODE = 'codes'
CONFIG = dict(width=int(os.environ.get('HEAD_WIDTH', '128')), layers=2)


def init_mlp(key, sizes, zero_last=True):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (ni, no) in enumerate(zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(keys[i], (ni, no), dtype=jnp.float64) * np.sqrt(2. / ni)
        if zero_last and i == len(keys) - 1:
            w = jnp.zeros_like(w)
        layers.append((w, jnp.zeros(no, dtype=jnp.float64)))
    return layers


def mlp(layers, x):
    for w, b in layers[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def mlp_np(layers, x):
    x = np.asarray(x, dtype=np.float64)
    for w, b in layers[:-1]:
        a = x @ np.asarray(w) + np.asarray(b)
        # sigmoid via tanh avoids NumPy overflow in diagnostic extreme codes.
        x = a * (.5 + .5 * np.tanh(.5 * a))
    w, b = layers[-1]
    return x @ np.asarray(w) + np.asarray(b)


def init(key, shared):
    k, r = shared['anchor'].shape[1], shared['anchor'].shape[0]
    p = dict(linear=jnp.asarray(shared['anchor']), bias=jnp.asarray(shared['center']),
             residual=init_mlp(key, [k, CONFIG['width'], CONFIG['width'], r]))
    return p, {key: jnp.asarray(value) for key, value in shared.items()}


def apply(p, f, z):
    return p['bias'] + z @ p['linear'].T + f['output_scale'] * mlp(p['residual'], z / f['latent_scale'])


def apply_np(p, f, z):
    return np.asarray(p['bias']) + np.asarray(z) @ np.asarray(p['linear']).T + float(f['output_scale']) * mlp_np(p['residual'], np.asarray(z) / np.asarray(f['latent_scale']))
