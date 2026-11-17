from campaign import *
import local_bank as B
xy=jnp.asarray(E.coords(8,True));p=B.init(jax.random.key(9),width=8,k=4);target=jnp.sin(3*xy[:,0:1])*jnp.sin(2*xy[:,1:2])
def objective(p):
 A=B.features(xy,B.uniform(1),4,'neural',p);c,_,_=B.fit(A,target);return jnp.sum((A@jax.lax.stop_gradient(c)-target)**2)
g=jax.grad(objective)(p);direction=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.001,p)
eps=1e-4;plus=jax.tree_util.tree_map(lambda x,d:x+eps*d,p,direction);minus=jax.tree_util.tree_map(lambda x,d:x-eps*d,p,direction)
fd=float((objective(plus)-objective(minus))/(2*eps));ad=float(sum(jnp.sum(x*y) for x,y in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(direction))))
assert abs(fd-ad)<1e-6*max(1,abs(fd)),(fd,ad)
save('smoke/variable_projection_gradient.json',dict(finite_difference=fd,autodiff=ad,absolute_difference=abs(fd-ad)))
