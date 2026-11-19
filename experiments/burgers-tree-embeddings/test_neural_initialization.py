from common import *
from neural import fields as F
from neural import allocations as A
from neural.initialize import missing_field

axis=np.linspace(.01,.99,16);xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
initial=np.exp(-np.sum((xy-.3)**2,axis=-1)/.02)
layouts={kind:A.allocate(kind,xy,initial,37,target=initial[None]) for kind in ['uniform','prescribed','random','oracle']}
assert all(A.validate(v) for v in layouts.values())
assert max(layouts['uniform'][:,0])==2
keys=jax.random.split(jax.random.PRNGKey(7),2)
p=dict(root=F.init_mlp(keys[0],[10,12,12,1],True),child=F.init_mlp(keys[1],[13,12,12,1],True))
cells=jnp.asarray(layouts['prescribed'][:4]);xy=jnp.asarray(xy);weight=jnp.ones(len(xy))/len(xy)
root=jnp.zeros(8);wanted=jnp.concatenate([root,.02*jax.random.normal(keys[1],(32,),dtype=jnp.float64)])
target=F.tree(p,wanted,xy,cells);norm=jnp.linalg.norm(target)/jnp.sqrt(len(target))
block=F.tree_jacobian(p,wanted,xy,cells)
full=jax.jacfwd(lambda zz:F.tree(p,zz,xy,cells))(wanted)
assert jnp.max(abs(block-full))<1e-12
base=jnp.sum(weight*(F.root(p['root'],root,xy)-target)**2)/norm**2
z,loss=missing_field(p,root,cells,xy,target,weight,norm)
assert jnp.all(jnp.diff(jnp.concatenate([base[None],loss]))<=1e-12)
assert loss[-1]<base
assert jnp.max(abs(F.tree(p,jnp.concatenate([root,jnp.zeros(32)]),xy,cells)-F.root(p['root'],root,xy)))==0
print(json.dumps(dict(layouts_valid=True,initial_relative_error=float(jnp.sqrt(base)),
                     initialized_relative_error=float(jnp.sqrt(loss[-1])),monotone_missing_field_initialization=True)))
