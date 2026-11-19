"""Runtime geometry with compact, depth-grouped local patch contractions.

Fixed slots cover every admissible geometry at the declared node cap. Arithmetic
on padding is recorded. Local patches scatter into the global field; they are
not evaluated over the entire domain. This is an unqualified cost candidate.
"""
from common import *
import discrete_weak as W, packed_fields as PF


def prepare(basis,cells,transforms,rotation,x,y,depth=4,node_cap=None):
    x,y=np.asarray(x),np.asarray(y)
    assert x.ndim==y.ndim==2 and x.shape==y.shape
    assert len(cells)==len(transforms)
    cap=len(cells) if node_cap is None else node_cap;assert len(cells)<=cap
    base=PF.prepare(basis,[],[],rotation,x,y)
    result={k:base[k] for k in ['basis','sine_x','sine_y']}
    rank=np.asarray(rotation).shape[-1]
    result['transform']=jnp.asarray(np.stack([np.asarray(t)@np.asarray(rotation) for t in transforms])
        if len(cells) else np.empty((0,36,rank)))
    assert np.allclose(np.diff(x[0]),x[0,1]-x[0,0],rtol=0,atol=1e-14)
    assert np.allclose(np.diff(y[0]),y[0,1]-y[0,0],rtol=0,atol=1e-14)
    buckets=[]
    for d in range(1,depth+1):
        count=min(4**d,cap)
        if not count:continue
        active=[i for i,c in enumerate(cells) if c[0]==d];assert len(active)<=count
        slots=np.zeros(count,dtype=np.int32);slots[:len(active)]=active
        mask=np.arange(count)<len(active);h=2.**(-d)
        bx=[];by=[];ix=[];iy=[]
        for slot,valid in zip(slots,mask):
            cell=cells[slot] if valid else (d,0,0)
            center=(np.asarray(cell[1:])+.5)*h
            for axis,cc,values,indices in [(x,center[0],bx,ix),(y,center[1],by,iy)]:
                step=axis[0,1]-axis[0,0];origin=axis[0,0]
                shifts=axis[:,0]-origin;pad=int(np.ceil(np.max(np.abs(shifts))/step))
                width=int(np.ceil(1.5*h/step))+2*pad+3
                first=int(np.floor((cc-.75*h-origin)/step))-pad-1
                ind=first+np.arange(width);inside=(ind>=0)&(ind<axis.shape[1])
                coords=origin+ind[None,:]*step+shifts[:,None]
                values.append(W.raw_axis(coords,cc,h)*inside[None,:,None])
                indices.append(np.clip(ind,0,axis.shape[1]-1).astype(np.int32))
        inverse=np.zeros(len(cells),dtype=np.int32)
        if active:inverse[np.asarray(active)]=np.arange(len(active))
        buckets.append(dict(X=jnp.asarray(np.stack(bx,axis=1)),Y=jnp.asarray(np.stack(by,axis=1)),
            ix=jnp.asarray(np.stack(ix)),iy=jnp.asarray(np.stack(iy)),slot_indices=jnp.asarray(slots),
            slot_mask=jnp.asarray(mask,dtype=jnp.float64),inverse=jnp.asarray(inverse)))
    result['buckets']=tuple(buckets)
    return result


def prepare_stencil(basis,cells,transforms,rotation,L,Q,depth=4,node_cap=None):
    axis=np.arange(1,Q)/Q
    x=np.stack([axis,axis+1/L,axis-1/L,axis,axis])
    y=np.stack([axis,axis,axis,axis+1/L,axis-1/L])
    return prepare(basis,cells,transforms,rotation,x,y,depth,node_cap)


def fields(c,cache):
    q=cache['compact'];root=len(q['basis'])
    B=jnp.einsum('k,kij->ij',c[:root],q['basis'])
    u=q['sine_x']@B@jnp.swapaxes(q['sine_y'],-1,-2)
    nodes,_,rank=q['transform'].shape
    if nodes:
        raw=jnp.einsum('pir,pr->pi',q['transform'],c[root:].reshape(nodes,rank)).reshape(nodes,6,6)
        for bucket in q['buckets']:
            coeff=raw[bucket['slot_indices']]*bucket['slot_mask'][:,None,None]
            patch=jnp.einsum('fbia,bac,fbjc->fbij',bucket['X'],coeff,bucket['Y'])
            u=u.at[jnp.arange(len(u))[:,None,None,None],bucket['ix'][None,:,:,None],
                   bucket['iy'][None,:,None,:]].add(patch)
    return u


def center(c,cache):return fields(c,cache)[0]


def tested_advection(L,Q):
    def evaluate(c,cache):
        u,xp,xm,yp,ym=fields(c,cache)
        adv=u*L*(jnp.where(u>0,u-xm,xp-u)+jnp.where(u>0,u-ym,yp-u))
        transformed=W.dst2(adv)/Q
        return transformed[cache['indices'][0],cache['indices'][1]]
    return evaluate


def patch_entries(q):
    return sum(int(b['X'].shape[0]*b['X'].shape[1]*b['X'].shape[2]*b['Y'].shape[2]) for b in q['buckets'])
