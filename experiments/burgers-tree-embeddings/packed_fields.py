"""Equivalent separable field contractions with geometry supplied at runtime.

Arrays are padded over the integration axis, not over a development-only tree
catalogue. Every valid geometry with the declared node count is supported. This
trades compact patch arithmetic for fixed compiled shapes; its cost is measured,
not assumed better. The original compact implementation remains the reference.
"""
from common import *
import discrete_weak as W


def prepare(basis, cells, transforms, rotation, x, y):
    """x/y have shape (number of stencils, axis points), including boundaries."""
    x, y, basis = np.asarray(x), np.asarray(y), np.asarray(basis)
    assert x.ndim == y.ndim == 2 and len(x) == len(y)
    assert len(cells) == len(transforms)
    N = basis.shape[-1]
    k = np.arange(1, N + 1)
    sine_x = np.sin(np.pi * x[..., None] * k) * ((x > 0) & (x < 1))[..., None]
    sine_y = np.sin(np.pi * y[..., None] * k) * ((y > 0) & (y < 1))[..., None]
    X, Y, T = [], [], []
    for cell, transform in zip(cells, transforms):
        depth, ix, iy = cell
        h = 2. ** (-depth)
        cx, cy = (np.array([ix, iy]) + .5) * h
        X.append(W.raw_axis(x, cx, h))
        Y.append(W.raw_axis(y, cy, h))
        T.append(np.asarray(transform) @ np.asarray(rotation))
    rank = np.asarray(rotation).shape[-1]
    return dict(basis=jnp.asarray(basis), sine_x=jnp.asarray(sine_x), sine_y=jnp.asarray(sine_y),
                X=jnp.asarray(np.stack(X, axis=2) if X else np.empty((*x.shape, 0, 6))),
                Y=jnp.asarray(np.stack(Y, axis=2) if Y else np.empty((*y.shape, 0, 6))),
                transform=jnp.asarray(np.stack(T) if T else np.empty((0, 36, rank))))


def prepare_stencil(basis, cells, transforms, rotation, L, Q):
    assert Q <= L and L % Q == 0
    axis = np.arange(1, Q) / Q
    x = np.stack([axis, axis + 1/L, axis - 1/L, axis, axis])
    y = np.stack([axis, axis, axis, axis + 1/L, axis - 1/L])
    return prepare(basis, cells, transforms, rotation, x, y)


def fields(c, cache):
    q = cache['packed']
    root = len(q['basis'])
    B = jnp.einsum('k,kij->ij', c[:root], q['basis'])
    u = q['sine_x'] @ B @ jnp.swapaxes(q['sine_y'], -1, -2)
    nodes, _, rank = q['transform'].shape
    if nodes:
        raw = jnp.einsum('pir,pr->pi', q['transform'], c[root:].reshape(nodes, rank)).reshape(nodes, 6, 6)
        weighted_x = jnp.einsum('fnpa,pab->fnpb', q['X'], raw)
        left = weighted_x.reshape(*weighted_x.shape[:2], nodes * 6)
        right = q['Y'].reshape(*q['Y'].shape[:2], nodes * 6)
        u = u + left @ jnp.swapaxes(right, -1, -2)
    return u


def center(c, cache):
    return fields(c, cache)[0]


def tested_advection(L, Q):
    def evaluate(c, cache):
        u, xp, xm, yp, ym = fields(c, cache)
        adv = u * L * (jnp.where(u > 0, u-xm, xp-u) + jnp.where(u > 0, u-ym, yp-u))
        transformed = W.dst2(adv) / Q
        return transformed[cache['indices'][0], cache['indices'][1]]
    return evaluate
