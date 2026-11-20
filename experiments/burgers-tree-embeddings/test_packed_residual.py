"""Small CPU support-restriction equivalence fixture, not a speed benchmark."""
from common import *
from neural import residual_tree as R, packed_residual as P, stable as S

rng=np.random.default_rng(2026111988)
xy=np.concatenate([rng.uniform(0,1,(257,2)),np.array([[0.,.3],[1.,.6],[.2,0.],[.6,1.],
    [.25,.25],[.625,.25],[.0625,.0625],[.15625,.0625]])])
xy=jnp.asarray(xy)
cells=jnp.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1],
    [2,0,0],[2,0,1],[2,1,0],[2,1,1],[3,0,0],[3,0,1],[3,1,0],[3,1,1]])
root=S.initialize(jax.random.PRNGKey(921),latent=4,width=8,depth=2)
params=R.initialize(jax.random.PRNGKey(922),root,child_k=2,features=8,width=8)
z=.2*jax.random.normal(jax.random.PRNGKey(923),(4+len(cells)*2,),dtype=jnp.float64)
packed=P.prepare(params['feature'],cells,xy)
u=P.apply(params,z,xy,packed);dense=R.tree(params,z,xy,cells)
J=P.jacobian(params,z,xy,packed);denseJ=R.tree_jacobian(params,z,xy,cells)
field_error=float(jnp.max(abs(u-dense)));jacobian_error=float(jnp.max(abs(J-denseJ)))
assert field_error<1e-13 and jacobian_error<1e-13
autodiff_error=float(jnp.max(abs(J-jax.jacfwd(P.apply,argnums=1)(params,z,xy,packed))))
assert autodiff_error<1e-13
for cell,ids,block in zip(np.asarray(cells),packed['indices'],packed['blocks']):
    h=2.**-cell[0];center=(cell[1:]+.5)*h
    expected=np.flatnonzero(np.all(abs((np.asarray(xy)-center)/h)<.75,axis=1))
    assert np.array_equal(np.asarray(ids),expected)
    assert block.shape==(len(expected),8)
assert P.feature_rows(packed)<len(xy)*len(cells)
zero=z.at[4:].set(0.)
assert float(jnp.max(abs(P.apply(params,zero,xy,packed)-S.apply(root,zero[:4],xy))))==0.
empty=P.prepare(params['feature'],np.empty((0,3),int),xy)
assert float(jnp.max(abs(P.apply(params,z[:4],xy,empty)-S.apply(root,z[:4],xy))))==0.
assert float(jnp.max(abs(P.jacobian(params,z[:4],xy,empty)
    -jax.jacfwd(S.apply,argnums=1)(root,z[:4],xy))))==0.
# A node with no point intersection has no feature-network evaluation or output.
far_xy=jnp.array([[.9,.9],[1.,1.]])
far=P.prepare(params['feature'],jnp.array([[3,0,0]]),far_xy)
assert P.feature_rows(far)==0 and far['blocks'][0].shape==(0,8)
assert float(jnp.max(abs(P.apply(params,z[:6],far_xy,far)-S.apply(root,z[:4],far_xy))))==0.
assert float(jnp.max(abs(P.jacobian(params,z[:6],far_xy,far)[:,4:])))==0.
assert float(jnp.max(abs(P.reconstruct(params,z,xy,cells)-u)))==0.
# Inspect the implementation invariant with a spy: the network sees gathered rows only.
original=R.feature_matrix;seen=[]
def spy(feature,cells,points):
    seen.append(len(points));return original(feature,cells,points)
R.feature_matrix=spy
try:
    check=P.prepare(params['feature'],cells,xy)
    P.prepare(params['feature'],jnp.array([[3,0,0]]),far_xy)
finally:
    R.feature_matrix=original
assert sum(seen)==P.feature_rows(check)
assert len(seen)==sum(block.shape[0]>0 for block in check['blocks'])
print(json.dumps(dict(field_error=field_error,jacobian_error=jacobian_error,
    autodiff_jacobian_error=autodiff_error,points=len(xy),nodes=len(cells),
    dense_child_network_rows=len(xy)*len(cells),packed_child_network_rows=P.feature_rows(packed),
    observed_spatial_network_rows=sum(seen),zero_child_preservation=True,
    empty_tree_and_empty_intersection=True,
    note='Row count reduction only; assembly, parent, latent solve and complete runtime not benchmarked')))
