"""Bounded charged-interpolation and actual-bank QR cold-fit smoke."""
from pathlib import Path
import signal,sys,time
signal.alarm(55)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cold_fit as c
import engines as e
import numpy as np
import jax
import jax.numpy as jnp
L=32;x=np.arange(L+1)/L;X,Y=np.meshgrid(x,x,indexing='ij');f=X+2*Y+3*X*Y
q=np.random.default_rng(12).uniform(0,1,(100,2));truth=q[:,0]+2*q[:,1]+3*q[:,0]*q[:,1]
error=float(np.max(np.abs(np.asarray(c.sample_field(jnp.asarray(f),jnp.asarray(q),L))-truth)));assert error<2e-15
root=Path(__file__).resolve().parents[3]
params,Z,_=e.sc.load_pkl(root/'experiments/separable-decoder/runs/b2dtensor/n512/out/sep_b2d_tensor_n512_ckpt.pkl')
G=e.sc.SeparableDecoder(params,16,64).feat_at(e.coords(L));Q,R=jnp.linalg.qr(G/(L-1),mode='reduced')
Zc=jnp.asarray(Z[::8]);H=e.sc.head(params,Zc)@R.T
fun=c.make_fit(params,L,60,full=True,starts=4);physical=e.params_draw(7090702,4)[0]
out=fun(jnp.asarray(e.initial(L,physical)),G,jnp.zeros((1,2)),jnp.asarray(1/(L-1)),Q,R,jnp.zeros((1,1)),Zc,H,jnp.sum(H*H,1))
f,z,rn,it,reason,grad=jax.tree_util.tree_map(np.asarray,out)
assert np.isfinite(f).all() and np.isfinite(grad) and f.shape==(L+1,L+1)
print(dict(interpolation_error=error,full_qr_finite=True,fit_iterations=int(it),fit_reason=int(reason),relative_gradient=float(grad)),flush=True)
