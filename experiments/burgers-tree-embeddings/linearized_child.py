"""Frozen zero-code tangent as a separately labelled direct-coefficient decoder.

This removes nonlinear code curvature; it does not repair or resume the stopped
nonlinear optimizer. Its representation must be screened before any PDE use.
"""
from common import *
from functools import partial
import shared_child_head as H


def zero_map(params, scale, cells):
    geometry, widths = H.geometry(cells)
    latent = params['linear'].shape[1]
    return jax.vmap(jax.jacfwd(lambda z, g, h:H.head(params, z, g, h, scale)))(
        jnp.zeros((len(cells), latent)), geometry, widths)


@partial(jax.jit, static_argnames=('root',))
def fit(gram, rhs, norm2, child_map, root=128):
    """Joint physical least squares with column scaling and no hidden ridge.

The full Gram matrix retains every parent/child/overlap cross term. A singular
span produces failed finite/conditioning checks rather than a silently altered
decoder or an optimistic projection claim.
"""
    S, b, _, inverse_cross, inverse_rhs = H.schur(gram, rhs, norm2, root=root)
    nodes, channels, latent = child_map.shape
    reduced = jnp.einsum('prk,prqs,qsl->pkql', child_map,
                         S.reshape(nodes, channels, nodes, channels), child_map).reshape(nodes*latent, nodes*latent)
    reduced = (reduced+reduced.T)*.5
    target = jnp.einsum('tpr,prk->tpk', b.reshape(len(rhs), nodes, channels), child_map).reshape(len(rhs), -1)
    diagonal = jnp.sqrt(jnp.maximum(jnp.diag(reduced), 1e-300))
    scaled = reduced/diagonal[:, None]/diagonal[None, :]
    factor = jnp.linalg.cholesky((scaled+scaled.T)*.5)
    z = jax.scipy.linalg.cho_solve((factor, True), (target/diagonal).T).T/diagonal
    local = jnp.einsum('tpk,prk->tpr', z.reshape(len(rhs), nodes, latent), child_map).reshape(len(rhs), -1)
    parent = inverse_rhs-local@inverse_cross.T
    full = jnp.concatenate([parent, local], axis=1)
    residual2 = jnp.einsum('ti,ij,tj->t', full, gram, full)-2*jnp.sum(full*rhs, axis=1)+norm2
    child_gradient = (z@reduced-target)/(jnp.sqrt(norm2)[:, None]*diagonal)
    parent_gradient = (full@gram-rhs)[:, :root]/jnp.sqrt(norm2[:, None]*jnp.diag(gram)[:root])
    station = jnp.maximum(jnp.max(jnp.abs(child_gradient), axis=1), jnp.max(jnp.abs(parent_gradient), axis=1))
    eigen = jnp.linalg.eigvalsh((scaled+scaled.T)*.5)
    return dict(coefficients=full, codes=z.reshape(len(rhs), nodes, latent),
                error=jnp.sqrt(jnp.maximum(residual2, 0)/norm2), stationarity=station,
                normalized_gram_eigenvalues=eigen, finite=jnp.all(jnp.isfinite(full)))
