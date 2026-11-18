from campaign import *
import continuous as C
N=8;V=jnp.eye(49)[:,:16];W,eigs,_=C.diffusion_basis(V,N);G,H=C.flux_tables(W,N,32);a=jnp.arange(16,dtype=jnp.float64)*.001
f=H.T@(G@a)**2;b=(2*(W@a)).reshape(7,7)
# Use fine sine synthesis for independently evaluated strong product.
hi=jnp.pad(b,((0,57),(0,57)));F,_=P.nonlinear(hi,65,2);g=W.T@F[:7,:7].ravel()/2
rel=float(jnp.linalg.norm(f-g)/jnp.linalg.norm(f));assert rel<1e-4,rel
v=C.flux_trajectory(.0025)(a,.05,G,H,eigs);v2=C.reduced_trajectory(8,.0025)(a,.05,W,eigs)
save('smoke/flux.json',dict(strong_weak_relative=rel,trajectory_difference=float(jnp.linalg.norm(v-v2))))
