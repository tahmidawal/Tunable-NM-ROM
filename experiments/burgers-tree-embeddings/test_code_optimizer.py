"""Asynchronous code visits must match separate optimizers without absent-row drift."""
import optax
from common import *
from neural import code_optimizer as C
codes=jnp.asarray([[.2,-.3],[.5,.4],[-.1,.7]])
s=C.initialize(codes);ref=[codes[i] for i in range(3)];opt=optax.adam(.003);states=[opt.init(v) for v in ref]
step=jax.jit(C.update)
for k,ids in enumerate([[0,2],[1],[0],[2,1],[1],[0,2]]):
    previous=np.array(codes);idx=jnp.asarray(ids);g=jnp.asarray([[.1*(k+1),-.3*(i+1)] for i in ids])
    codes,s=step(codes,s,idx,g,.003)
    for pos,i in enumerate(ids):
        u,states[i]=opt.update(g[pos],states[i],ref[i]);ref[i]=optax.apply_updates(ref[i],u)
    assert np.array_equal(np.asarray(codes)[[i for i in range(3) if i not in ids]],previous[[i for i in range(3) if i not in ids]])
assert np.max(abs(np.asarray(codes)-np.stack(ref)))<1e-14
assert np.array_equal(np.asarray(s['count']),[3,3,3])
print('Independent per-snapshot Adam parity and absent-row preservation passed')
