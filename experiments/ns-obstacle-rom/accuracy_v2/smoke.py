from pathlib import Path
import tempfile,json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
import jax.numpy as jnp
from accuracy_v2.spatial_bank import initialize,SpatialBank,evaluate_layers,envelope
x=np.array([[.3,.1],[.6,.31],[1.8,.2],[0.,.1],[.1,0.],[.1,.41],[.25,.2],[0.,0.],[0.,.41],[.2,.15],[.2,.25]])
b=initialize(width=12,depth=2);v,g=b.evaluate(x)
assert np.allclose(b.values_blocked(x,3),v,rtol=1e-14,atol=1e-14)
def f(q):return evaluate_layers([(jnp.asarray(w),jnp.asarray(z)) for w,z in b.layers],q[None,:],jnp)[0][0]
jg=np.asarray(jax.vmap(jax.jacfwd(f))(jnp.asarray(x))).transpose(0,1,3,2)
assert np.max(np.abs(jg-g))<1e-11
fd=[]
for d in range(2):
 h=np.zeros_like(x);h[:,d]=1e-6;fd.append((b.evaluate(x+h)[0]-b.evaluate(x-h)[0])/2e-6)
fd=np.stack(fd,axis=2);assert np.max(np.abs(fd-g))<2e-7
assert np.max(np.abs(v[3:]))<1e-14
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder)/'bank.npz';b.save(p);restored=SpatialBank.load(p)
 for a,c in zip(b.evaluate(x),restored.evaluate(x)):assert np.array_equal(a,c)
# Analytic first derivatives remain differentiable with respect to network weights.
layers=[(jnp.asarray(w),jnp.asarray(z)) for w,z in b.layers]
def loss(t):
 ls=[(layers[0][0].at[0,0].set(t),layers[0][1])]+layers[1:];vv,gg=evaluate_layers(ls,jnp.asarray(x),jnp)
 return jnp.sum(vv*vv)+.01*jnp.sum(gg*gg)
t=float(layers[0][0][0,0]);ad=float(jax.grad(loss)(t));ff=float((loss(t+1e-5)-loss(t-1e-5))/2e-5)
assert abs(ad-ff)<1e-6
print(json.dumps({'passed':True,'jax_backend':jax.default_backend(),'gradient_vs_jax_max':float(np.max(np.abs(jg-g))),'gradient_vs_FD_max':float(np.max(np.abs(fd-g))),'boundary_max':float(np.max(np.abs(v[3:]))),'parameter_gradient_gap':abs(ad-ff),'genuine_data_opened':False}))
