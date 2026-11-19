from common import *
from neural import fields as F,factorized as S
from pretrain_local_features import quadrature,patches,project_readout
import data

key=jax.random.PRNGKey(83);root=F.init_mlp(key,[10,12,12,1],True)
codes=jnp.zeros((3,8));b=jax.random.normal(key,(3,7,7),dtype=jnp.float64)*.01
cells=jnp.array([[2,1,1],[2,2,2]]);q=quadrature(7);norm=jnp.array([1.,2.,3.])
actual,valid=patches(root,codes,b,cells,norm,q);expected=[]
for cell in cells:
    h=2.**(-cell[0]);center=(cell[1:]+.5)*h;xy=center+h*q['xy']
    truth=data.synth(b,xy)
    prediction=jax.vmap(lambda z:F.root(root,z,xy))(codes)
    window_sum=sum(S.window(other,xy)[1] for other in cells)
    expected.append((truth-prediction)/F.boundary(xy)[None]/norm[:,None]*q['window'][None]/window_sum[None])
error=float(jnp.max(abs(actual-jnp.concatenate(expected))))
assert jnp.all(valid) and error<1e-12
hidden=F.init_mlp(key,[2,8,8,2],True)[:-1]
target=jnp.stack([jnp.sin(q['xy'][:,0]),jnp.cos(q['xy'][:,1])],1)*(jnp.sqrt(q['w'])*q['window'])[:,None]
def objective(t,stop):
    p=[dict(v) for v in hidden];p[0]['w']=p[0]['w'].at[0,0].add(t)
    last,_=project_readout(p,q['xy'],q['w'],q['window'],target)
    if stop:last=jax.tree.map(jax.lax.stop_gradient,last)
    prediction=F.mlp(p+[last],q['xy']/.75,True)*(jnp.sqrt(q['w'])*q['window'])[:,None]
    return jnp.sum((prediction-target)**2)+1e-10*sum(jnp.sum(v*v) for v in last.values())
exact=jax.grad(lambda t:objective(t,False))(0.)
envelope=jax.grad(lambda t:objective(t,True))(0.)
assert abs(exact-envelope)<1e-9
print(json.dumps(dict(passed=True,separable_vs_direct_reference_error=error,projected_readout_gradient_difference=float(abs(exact-envelope)))))
