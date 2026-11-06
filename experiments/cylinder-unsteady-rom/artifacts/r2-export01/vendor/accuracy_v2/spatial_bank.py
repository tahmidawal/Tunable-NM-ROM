"""Accuracy-first POD-guided neural spatial bank; no FE fallback at evaluation.

The original bank v1 is unchanged. Values are u/U; derivatives are physical.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from geometry import CHANNEL

FEATURES={'fourier_frequencies':[1.,2.,4.,8.],'radial_widths':[.1,.2,.4]}

def inputs(x,xp=np):
    offset=xp.asarray([1.1,.205]);scale=xp.asarray([1.1,.205])
    z=(x-offset)/scale;eye=xp.broadcast_to(xp.diag(1/scale),(len(x),2,2))
    vals=[z];ders=[eye]
    for f in FEATURES['fourier_frequencies']:
        t=xp.pi*f*z
        vals += [xp.sin(t),xp.cos(t)]
        ders += [xp.cos(t)[:,:,None]*xp.pi*f*eye,-xp.sin(t)[:,:,None]*xp.pi*f*eye]
    d=x-xp.asarray(CHANNEL.center)
    for width in FEATURES['radial_widths']:
        g=xp.exp(-xp.sum(d*d,axis=1)/(2*width**2))
        vals.append(g[:,None]);ders.append((-g[:,None]*d/width**2)[:,None,:])
    return xp.concatenate(vals,axis=1),xp.concatenate(ders,axis=1)

def envelope(x,xp=np):
    """Bounded smooth distance factors; products avoid division at boundary zeros."""
    D=.1;d=x-xp.asarray(CHANNEL.center);rad=xp.sqrt(xp.sum(d*d,axis=1))
    distances=xp.stack([x[:,0],x[:,1],CHANNEL.height-x[:,1],rad-CHANNEL.obstacle_radius],axis=1)
    ones=xp.ones(len(x));zeros=xp.zeros(len(x))
    dr=d/xp.maximum(rad[:,None],1e-300)
    dd=xp.stack([xp.stack([ones,zeros],1),xp.stack([zeros,ones],1),xp.stack([zeros,-ones],1),dr],1)
    e=xp.exp(-distances/D);f=-xp.expm1(-distances/D);df=e[:,:,None]*dd/D
    b=xp.prod(f,axis=1);db=xp.zeros_like(x)
    for j in range(4):
        other=xp.ones(len(x))
        for k in range(4):
            if k!=j:other=other*f[:,k]
        db=db+other[:,None]*df[:,j]
    return b,db

def evaluate_layers(layers,x,xp=np):
    h,dh=inputs(x,xp)
    for i,(w,b) in enumerate(layers):
        h=h@w+b;dh=xp.einsum('nhd,hk->nkd',dh,w)
        if i<len(layers)-1:
            h=xp.tanh(h);dh=dh*(1-h*h)[:,:,None]
    rank=h.shape[1]//2;v=h.reshape(len(x),2,rank)
    g=dh.reshape(len(x),2,rank,2).transpose(0,1,3,2)
    b,db=envelope(x,xp)
    return b[:,None,None]*v,b[:,None,None,None]*g+db[:,None,:,None]*v[:,:,None,:]

class SpatialBank:
    geometry=CHANNEL
    def __init__(self,layers,transform,metadata):
        self.layers=[(np.asarray(w,dtype=np.float64),np.asarray(b,dtype=np.float64)) for w,b in layers]
        self.transform=np.asarray(transform,dtype=np.float64);self.metadata=dict(metadata)
        prior=21
        for w,b in self.layers:
            if w.ndim!=2 or w.shape[0]!=prior or b.shape!=(w.shape[1],) or not np.isfinite(w).all() or not np.isfinite(b).all():raise ValueError('invalid neural layer')
            prior=w.shape[1]
        if prior%2 or self.transform.ndim!=2 or self.transform.shape[0]!=prior//2 or not np.isfinite(self.transform).all():raise ValueError('invalid transform')
    @property
    def rank(self):return self.transform.shape[1]
    def evaluate(self,points,prefix=None):
        x=np.asarray(points,dtype=np.float64);r=self.rank if prefix is None else int(prefix)
        if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or not 1<=r<=self.rank:raise ValueError('invalid evaluation')
        v,g=evaluate_layers(self.layers,x);return v@self.transform[:,:r],g@self.transform[:,:r]
    def values_blocked(self,points,block_points=4096):
        shape=(len(points),2,self.rank)
        if type(block_points) is not int or not 1<=block_points<=16384:raise ValueError('bounded block_points required')
        if int(np.prod(shape,dtype=np.int64))*8>2**30:raise MemoryError('array cap')
        out=np.empty(shape,dtype=np.float64)
        for i in range(0,len(points),block_points):
            x=np.asarray(points[i:i+block_points]);h=inputs(x)[0]
            for j,(w,bias) in enumerate(self.layers):
                h=h@w+bias
                if j<len(self.layers)-1:h=np.tanh(h)
            out[i:i+block_points]=(envelope(x)[0][:,None,None]*h.reshape(len(x),2,-1))@self.transform
        return out
    def save(self,path):
        m=dict(self.metadata,schema='ns-obstacle-pod-guided-neural-v2',features=FEATURES,envelope='bounded-exponential-distances-D0.1',geometry=CHANNEL.as_dict(),layers=len(self.layers),value_units='u/U',gradient_coordinates='physical',FE_fallback=False)
        a={'metadata_json':np.array(json.dumps(m,sort_keys=True)),'transform':self.transform}
        for i,(w,b) in enumerate(self.layers):a['W'+str(i)]=w;a['b'+str(i)]=b
        np.savez_compressed(path,**a);return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            m=json.loads(str(a['metadata_json']))
            if m.get('schema')!='ns-obstacle-pod-guided-neural-v2' or m['features']!=FEATURES or m['geometry']!=json.loads(json.dumps(CHANNEL.as_dict())) or m['FE_fallback'] or m['envelope']!='bounded-exponential-distances-D0.1' or m['value_units']!='u/U' or m['gradient_coordinates']!='physical':raise ValueError('unknown spatial bank')
            return cls([(a['W'+str(i)],a['b'+str(i)]) for i in range(m['layers'])],a['transform'],m)

def initialize(rank=4,width=128,depth=3,seed=29102026):
    rng=np.random.default_rng(seed);sizes=[21]+[width]*depth+[2*rank]
    return SpatialBank([(rng.normal(size=(a,b))/np.sqrt(a),np.zeros(b)) for a,b in zip(sizes[:-1],sizes[1:])],np.eye(rank),{'seed':seed,'variant':'POD-guided supervised spatial bank v2','trained':False})
