from common import *
from neural import fields as F,factorized as S
from pretrain_local_features import quadrature,patches
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
print(json.dumps(dict(passed=True,separable_vs_direct_reference_error=error)))
