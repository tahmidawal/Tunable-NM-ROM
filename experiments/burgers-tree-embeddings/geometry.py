"""Retained-node quadtrees and fixed C2 physical windows."""
from common import jax, jnp, np

def children(cell):
    d,i,j=cell
    return [(d+1,2*i+a,2*j+b) for a in range(2) for b in range(2)]

def leaves(active):
    return [c for c in [(0,0,0)]+list(active) if not any(k in active for k in children(c))]

def uniform(depth):
    active=[]; frontier=[(0,0,0)]
    for _ in range(depth):
        frontier=[k for c in frontier for k in children(c)];active+=frontier
    return active

def allocated(xy, indicator, weights, cap, rank=4, maxdepth=3, random_seed=None):
    """Greedy whole-sibling additions. Caller labels initial-data, random, or oracle."""
    active=[]; rng=np.random.default_rng(random_seed) if random_seed is not None else None
    xy=np.asarray(xy); v=np.asarray(indicator); w=np.asarray(weights)
    while 16+(len(active)+4)*rank<=cap:
        candidates=[c for c in leaves(active) if c[0]<maxdepth]
        if not candidates:break
        def score(c):
            d,i,j=c;h=2.**(-d);lo=np.array([i,j])*h
            inside=np.all((xy>=lo)&(xy<lo+h),axis=1)
            return float(np.sum(w[inside]*v[inside]))
        c=candidates[int(rng.integers(len(candidates)))] if rng else max(candidates,key=score)
        active+=children(c)
    return active

def bc(x):return 16*x[...,0]*(1-x[...,0])*x[...,1]*(1-x[...,1])

def window(x,cell,halo=.25):
    d,i,j=cell;h=2.**(-d);c=(jnp.array([i,j])+.5)*h
    xi=(x-c)/h
    t=jnp.clip((jnp.abs(xi)-.5)/halo,0.,1.)
    taper=1-(10*t**3-15*t**4+6*t**5)
    return jnp.prod(taper,axis=-1)*bc(x)

def local_inputs(x,cell):
    d,i,j=cell;h=2.**(-d);c=(jnp.array([i,j])+.5)*h
    return jnp.concatenate([2*(x-c)/h,jnp.full(x.shape[:-1]+(1,),-d,dtype=x.dtype)],axis=-1)

def valid(active):
    s=set(active)
    assert len(s)==len(active)
    for c in active:
        d,i,j=c;parent=(d-1,i//2,j//2)
        assert d>0 and (d==1 or parent in s)
        assert set(children(parent))<=s
    return True
