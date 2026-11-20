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

codes=jnp.asarray([[.2,-.3],[.5,.4],[-.1,.7]])
s=C.initialize_masked(codes)
ref=[[codes[i,j] for j in range(2)] for i in range(3)]
states=[[opt.init(ref[i][j]) for j in range(2)] for i in range(3)]
expected_count=np.zeros((3,2),dtype=int)
step=jax.jit(C.update_masked)
visits=[([0,2],[[1,1]]),([0],[[1,0]]),([1,2],[[0,1],[1,0]]),
        ([0,1],[[0,0]]),([2,0],[[1,1]]),([1],[[1,1]])]
for k,(ids,mask) in enumerate(visits):
    previous=np.asarray(codes).copy();old={key:np.asarray(value).copy() for key,value in s.items()}
    idx=jnp.asarray(ids);g=jnp.asarray([[.1*(k+1),-.3*(i+1)] for i in ids])
    codes,s=step(codes,s,idx,g,.003,jnp.asarray(mask))
    active=np.zeros((3,2),dtype=bool);active[ids]=np.broadcast_to(mask,(len(ids),2))
    for pos,i in enumerate(ids):
        for j in range(2):
            if not active[i,j]:continue
            u,states[i][j]=opt.update(g[pos,j],states[i][j],ref[i][j])
            ref[i][j]=optax.apply_updates(ref[i][j],u);expected_count[i,j]+=1
    assert np.array_equal(np.asarray(codes)[~active],previous[~active])
    for key in s:assert np.array_equal(np.asarray(s[key])[~active],old[key][~active])
    assert np.isfinite(np.asarray(codes)).all()
assert np.max(abs(np.asarray(codes)-np.asarray(ref)))<1e-14
assert np.array_equal(np.asarray(s['count']),expected_count)
print('Masked per-coordinate Adam parity and inactive-coordinate state preservation passed')
