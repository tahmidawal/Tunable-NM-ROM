"""Small CPU mathematical fixtures; no training or scientific accuracy claim."""
from common import *
from neural import residual_tree as R, stable as S
import data

xy, w = data.grid(19, True); xy = jnp.asarray(xy)
root = S.initialize(jax.random.PRNGKey(81), latent=32, width=40, depth=2)
p = R.initialize(jax.random.PRNGKey(82), root, child_k=16, features=64, width=24)
cells = jnp.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
z = .1*jax.random.normal(jax.random.PRNGKey(83), (96,), dtype=jnp.float64)
zero = z.at[32:].set(0.)
pp = dict(p, B=R.feature_matrix(p['feature'], cells, xy))
assert float(jnp.max(abs(R.cached(pp, zero, xy)-S.apply(root, zero[:32], xy)))) == 0.
J = R.cached_jacobian(pp, zero, xy)
weighted = np.sqrt(w)[:, None]*np.asarray(J)
q, _ = np.linalg.qr(weighted[:, :32])
child = weighted[:, 32:]-q@(q.T@weighted[:, 32:])
norm = np.linalg.norm(child, axis=0); assert norm.min() > 1e-8
s = np.linalg.svd(child/norm, compute_uv=False)
assert s[-1]/s[0] > 1e-6, s[-1]/s[0]
block = R.cached_jacobian(pp, z, xy)
full = jax.jacfwd(lambda zz: R.cached(pp, zz, xy))(z)
assert float(jnp.max(abs(block-full))) < 1e-12
direction = jax.random.normal(jax.random.PRNGKey(84), z.shape, dtype=jnp.float64)
direction /= jnp.linalg.norm(direction)
eps = 1e-5
fd = (R.cached(pp, z+eps*direction, xy)-R.cached(pp, z-eps*direction, xy))/(2*eps)
fd_error = float(jnp.linalg.norm(fd-block@direction)/jnp.linalg.norm(block@direction))
assert fd_error < 1e-7
child_direction = direction.at[:32].set(0.)
curvature = float(jnp.linalg.norm(R.cached(pp, zero+child_direction, xy)
                 -R.cached(pp, zero, xy)-J@child_direction))
assert curvature > 1e-7
extra = jnp.array([[2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1]])
extended = jnp.concatenate([cells, extra]); zz = jnp.pad(z, (0, 64))
assert float(jnp.max(abs(R.tree(p, zz, xy, extended)-R.tree(p, z, xy, cells)))) < 1e-14
walls = jnp.array([[0., .2], [1., .3], [.6, 0.], [.9, 1.]])
assert float(jnp.max(abs(R.tree(p, z, walls, cells)))) == 0.
print(json.dumps(dict(zero_child_preservation=True, zero_enrichment_preservation=True,
    exact_boundary=True, child_directions_after_parent_projection=64,
    minimum_normalized_singular_ratio=float(s[-1]/s[0]),
    finite_difference_relative_error=fd_error, nonlinear_child_curvature=curvature,
    block_jacobian_error=float(jnp.max(abs(block-full))))))
