"""Retained-node layouts. Oracle scores are offline training diagnostics only."""
import numpy as np


def children(cell):
    d,x,y=cell
    return [(d+1,2*x+i,2*y+j) for i in range(2) for j in range(2)]


def allocate(kind,xy,initial,seed=0,target=None,nodes=20,max_depth=3):
    assert kind in ('uniform','prescribed','random','oracle')
    assert nodes>=4 and nodes%4==0 and max_depth>=1
    if kind=='oracle':assert target is not None,'Oracle allocation is offline only'
    rng=np.random.default_rng(seed);retained=children((0,0,0));leaves=list(retained)
    values=np.asarray(initial if kind!='oracle' else target)
    scores=np.mean(values**2,axis=0) if values.ndim==2 else values**2
    while len(retained)<nodes:
        eligible=[c for c in leaves if c[0]<max_depth];assert eligible
        if kind=='uniform':chosen=min(eligible)
        elif kind=='random':chosen=eligible[int(rng.integers(len(eligible)))]
        else:
            def score(c):
                d,x,y=c;h=2.**-d;center=(np.array([x,y])+.5)*h
                mask=np.all(abs(xy-center)<=.5*h,axis=-1)
                return float(np.sum(scores[mask]))
            chosen=max(eligible,key=score)
        new=children(chosen);retained.extend(new);leaves.remove(chosen);leaves.extend(new)
    return np.asarray(retained,dtype=np.int32)


def validate(cells):
    seen=set()
    for d,x,y in np.asarray(cells):
        assert d>=1 and 0<=x<2**d and 0<=y<2**d
        assert (d,x,y) not in seen
        assert d==1 or (d-1,x//2,y//2) in seen
        seen.add((d,x,y))
    return True
