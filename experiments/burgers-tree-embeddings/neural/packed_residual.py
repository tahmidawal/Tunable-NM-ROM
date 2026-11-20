"""Support-restricted child feature assembly and cached field evaluation.

prepare is host-directed assembly, not a jitted online kernel. Its coordinate
inspection, transfers and spatial network evaluations must count in complete
query cost. A prepared cache is valid only for its feature weights, coordinates
and cells. Per-node arrays avoid padding shallow and deep nodes to equal size.
The parent still evaluates all points; requested full-resolution output is not
free. No physical stencil or accuracy criterion changes here.
"""
from common import *
from neural import residual_tree as R, stable as S


def prepare(feature, cells, xy):
    """Evaluate the child spatial network only on its nonzero window support.

    Returns tuples of point indices and feature blocks. Their row counts are
    the exact number of spatial-network rows evaluated for each node. Empty
    intersections contain no rows and do not call the spatial network.
    """
    points = np.asarray(xy, dtype=np.float64)
    cells = np.asarray(cells)
    assert points.ndim == 2 and points.shape[1] == 2
    assert cells.size == 0 or (cells.ndim == 2 and cells.shape[1] == 3)
    features = feature[-1]['b'].shape[0]
    indices, blocks = [], []
    for cell in cells:
        h = 2.**(-cell[0]); center = (cell[1:]+.5)*h
        q = (points-center)/h
        ids = np.flatnonzero(np.all(np.abs(q)<.75, axis=-1))
        indices.append(jnp.asarray(ids, dtype=jnp.int32))
        if len(ids):
            block = R.feature_matrix(feature, jnp.asarray(cell)[None],
                                     jnp.asarray(points[ids], dtype=jnp.float64))[:, 0, :]
        else:
            block = jnp.zeros((0, features), dtype=jnp.float64)
        blocks.append(block)
    return dict(indices=tuple(indices), blocks=tuple(blocks))


def feature_rows(packed):
    """Spatial-network evaluation count, excluding the separately charged parent."""
    return sum(block.shape[0] for block in packed['blocks'])


def apply(params, z, xy, packed):
    root_k, child_k = R.dimensions(params)
    assert len(z) == root_k+child_k*len(packed['blocks'])
    u = S.apply(params['root'], z[:root_k], xy)
    if not packed['blocks']:
        return u
    coeff = R.coefficients(params['head'], z[root_k:].reshape(-1, child_k), params['scale'])
    for node, (ids, block) in enumerate(zip(packed['indices'], packed['blocks'])):
        u = u.at[ids].add(block@coeff[node])
    return u


def jacobian(params, z, xy, packed):
    root_k, child_k = R.dimensions(params)
    assert len(z) == root_k+child_k*len(packed['blocks'])
    root = jax.jacfwd(S.apply, argnums=1)(params['root'], z[:root_k], xy)
    if not packed['blocks']:
        return root
    local = z[root_k:].reshape(-1, child_k)
    head_jac = jax.vmap(jax.jacfwd(R.coefficients, argnums=1), in_axes=(None, 0, None))(
        params['head'], local, params['scale'])
    blocks = [root]
    for node, (ids, block) in enumerate(zip(packed['indices'], packed['blocks'])):
        jac = jnp.zeros((len(xy), child_k), dtype=root.dtype)
        blocks.append(jac.at[ids].add(block@head_jac[node]))
    return jnp.concatenate(blocks, axis=-1)


def reconstruct(params, z, xy, cells):
    """Uncached complete field evaluation, including support/feature assembly."""
    packed = prepare(params['feature'], cells, xy)
    return apply(params, z, xy, packed)
