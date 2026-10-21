"""Bounded independent FFT, dtype, gradient and shape checks; no research result."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
from operators import models3d as M

assert jax.default_backend()=='gpu'
rng=np.random.default_rng(920320)
x=rng.normal(size=(2,8,9,10,3));w=rng.normal(size=(4,4,3,3,4,2))
y=np.asarray(jax.jit(lambda w,x:M.spectral_convolve(w,x,(2,2,3)))(jnp.asarray(w),jnp.asarray(x)))
f=np.fft.rfftn(x,axes=(1,2,3));ix=np.r_[0:2,6:8];iy=np.r_[0:2,7:9]
fout=np.zeros((2,8,9,6,4),dtype=np.complex128)
for a,i in enumerate(ix):
 for b,j in enumerate(iy):
  for k in range(3):fout[:,i,j,k]=f[:,i,j,k]@(w[a,b,k,...,0]+1j*w[a,b,k,...,1])
ref=np.fft.irfftn(fout,s=(8,9,10),axes=(1,2,3));error=np.linalg.norm(y-ref)/np.linalg.norm(ref)
assert error<1e-12,error
checks=[]
for spec in [dict(kind='fno3d',width=4,modes=[2,2,2],depth=2,padding=0),dict(kind='unet3d',width=4,levels=2,periodic=False),dict(kind='unet3d',width=4,levels=2,periodic=True)]:
 p=M.init_model(jax.random.PRNGKey(1),spec,3,5)
 def loss(p,x):return jnp.mean((M.apply_model(p,x,spec)-.1)**2)
 value,gradient=jax.jit(jax.value_and_grad(loss))(p,jnp.asarray(x))
 assert all(a.dtype==jnp.float64 and np.isfinite(a).all() for a in jax.tree_util.tree_leaves(gradient))
 assert float(value)>0
 y=jax.jit(lambda p,x:M.apply_model(p,x,spec))(p,jnp.asarray(x))
 assert y.shape==(2,8,9,10,5) and y.dtype==jnp.float64
 checks.append((spec,float(value),M.parameter_count(p)))
print({'gpu':jax.devices()[0].device_kind,'fft_numpy_relative':error,'checks':checks},flush=True)
