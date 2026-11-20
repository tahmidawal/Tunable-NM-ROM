"""Frozen nonlinear parent and shared, cacheable nonlinear child manifolds.

Spatial features are learned anew in B8. The feature span is a capacity control,
not the deployed manifold: each child's independent code has fewer coordinates
than its feature coefficients. Network weights are fixed during online solves.
"""
from common import *
from neural import fields as F, stable as S


def initialize(key, root, child_k=16, features=64, width=64, scale=.1):
    assert 0 < child_k < features
    kf, kh = jax.random.split(key)
    return dict(root=root,
                feature=F.init_mlp(kf, [14, width, width, features]),
                head=F.init_mlp(kh, [child_k, width, width, features-child_k]),
                scale=jnp.full(features, scale, dtype=jnp.float64))


def window(cell, xy):
    h = 2.**(-cell[0]); center = (cell[1:]+.5)*h
    q = (xy-center)/h
    t = jnp.clip((jnp.abs(q)-.5)/.25, 0., 1.)
    w = jnp.prod(1-(10*t**3-15*t**4+6*t**5), axis=-1)
    return q, w


def encoding(q):
    q = jnp.clip(q/.75, -1., 1.)
    phase = jnp.pi*q[..., None]*jnp.asarray([1., 2., 4.])
    return jnp.concatenate([q, jnp.sin(phase).reshape(q.shape[:-1]+(6,)),
                            jnp.cos(phase).reshape(q.shape[:-1]+(6,))], axis=-1)


def feature_matrix(feature, cells, xy):
    """Shape (points, nodes, features); cache only while weights/grid are fixed."""
    def one(cell):
        q, w = window(cell, xy)
        return F.mlp(feature, encoding(q))*(F.boundary(xy)*w)[..., None]
    return jax.vmap(one)(cells).transpose(1, 0, 2)


def coefficients(head, z, scale):
    tail = F.mlp(head, z)-F.mlp(head, jnp.zeros_like(z))
    return jnp.concatenate([z, tail], axis=-1)*scale


def dimensions(params):
    return params['root']['modulation']['w'].shape[0], params['head'][0]['w'].shape[0]


def cached(params, z, xy):
    root_k, child_k = dimensions(params)
    local = z[root_k:].reshape(-1, child_k)
    c = coefficients(params['head'], local, params['scale'])
    return S.apply(params['root'], z[:root_k], xy)+jnp.einsum('pnr,nr->p', params['B'], c)


def cached_jacobian(params, z, xy):
    root_k, child_k = dimensions(params)
    local = z[root_k:].reshape(-1, child_k)
    parent = jax.jacfwd(S.apply, argnums=1)(params['root'], z[:root_k], xy)
    head_jac = jax.vmap(jax.jacfwd(coefficients, argnums=1), in_axes=(None, 0, None))(
        params['head'], local, params['scale'])
    children = jnp.einsum('pnr,nrk->pnk', params['B'], head_jac).reshape(len(xy), -1)
    return jnp.concatenate([parent, children], axis=-1)


def tree(params, z, xy, cells):
    return cached(dict(params, B=feature_matrix(params['feature'], cells, xy)), z, xy)


def tree_jacobian(params, z, xy, cells):
    return cached_jacobian(dict(params, B=feature_matrix(params['feature'], cells, xy)), z, xy)


def cached_jacobian_reverse(params, z, xy):
    """Same block Jacobian, scalar reverse mode for the nonlinear root only."""
    root_k, child_k = dimensions(params)
    local = z[root_k:].reshape(-1, child_k)
    parent = S.code_jacobian(params['root'], z[:root_k], xy)
    head_jac = jax.vmap(jax.jacfwd(coefficients, argnums=1), in_axes=(None, 0, None))(
        params['head'], local, params['scale'])
    children = jnp.einsum('pnr,nrk->pnk', params['B'], head_jac).reshape(len(xy), -1)
    return jnp.concatenate([parent, children], axis=-1)
