"""Self-contained frozen-parent plus four residual-head spatial model.

All live values and physical derivatives evaluate the embedded neural parent and
four neural heads; there is no cached TRAIN-array evaluation path.
"""
from pathlib import Path
import io,json,hashlib
import numpy as np
from geometry import CHANNEL
from accuracy_v2.spatial_bank import SpatialBank,FEATURES
from accuracy_v2.fast_spatial_bank import evaluate_layers_numpy

SCHEMA='ns-obstacle-independent-residual-head-bank-v1'

def _parent(raw,expected_sha):
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('embedded parent hash mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        m=json.loads(str(a['metadata_json']))
        if m.get('schema')!='ns-obstacle-pod-guided-neural-v2' or m['features']!=FEATURES or m['geometry']!=json.loads(json.dumps(CHANNEL.as_dict())) or m['FE_fallback'] or m['envelope']!='bounded-exponential-distances-D0.1' or m['value_units']!='u/U' or m['gradient_coordinates']!='physical':raise ValueError('unknown parent contract')
        layers=[(a['W'+str(i)].copy(),a['b'+str(i)].copy()) for i in range(m['layers'])];transform=a['transform'].copy()
    if any(z.dtype!=np.float64 for pair in layers for z in pair) or transform.dtype!=np.float64:raise ValueError('parent must be f64')
    p=SpatialBank(layers,transform,m)
    if p.rank!=4 or not np.array_equal(p.transform,np.eye(4)):raise ValueError('unrotated four-mode parent required')
    for pair in p.layers:
        for z in pair:z.setflags(write=False)
    p.transform.setflags(write=False)
    return p

class ResidualHeadBank:
    geometry=CHANNEL
    rank=4
    def __init__(self,parent_bytes,parent_sha256,heads,metadata):
        self.parent_bytes=bytes(parent_bytes);self.parent_sha256=parent_sha256;self.parent=_parent(self.parent_bytes,parent_sha256)
        if len(heads)!=4:raise ValueError('four heads required')
        self.heads=[]
        for h in heads:
            if len(h)!=3:raise ValueError('two hidden layers required')
            copy=[]
            for (w,b),shape in zip(h,((21,64),(64,64),(64,2))):
                w=np.asarray(w);b=np.asarray(b)
                if w.dtype!=np.float64 or b.dtype!=np.float64 or w.shape!=shape or b.shape!=(shape[1],) or not np.isfinite(w).all() or not np.isfinite(b).all():raise ValueError('finite f64 fixed head architecture required')
                copy.append((w.copy(),b.copy()))
            self.heads.append(tuple(copy))
        self.heads=tuple(self.heads);self.metadata=dict(metadata);self.transform=np.eye(4)
    def evaluate(self,points,prefix=None):
        x=np.asarray(points,dtype=np.float64);r=4 if prefix is None else prefix
        if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or type(r) is not int or not 1<=r<=4:raise ValueError('invalid evaluation')
        pv,pg=evaluate_layers_numpy(self.parent.layers,x);pv=pv@self.parent.transform;pg=pg@self.parent.transform
        values=[];gradients=[]
        for h in self.heads:
            v,g=evaluate_layers_numpy(h,x);values.append(v);gradients.append(g)
        return (pv+np.concatenate(values,axis=-1))[:,:,:r],(pg+np.concatenate(gradients,axis=-1))[:,:,:,:r]
    def values_blocked(self,points,block_points=4096):
        if type(block_points) is not int or not 1<=block_points<=16384:raise ValueError('bounded block size required')
        if len(points)*2*4*8>2**30:raise MemoryError('output cap before allocation')
        out=np.empty((len(points),2,4),dtype=np.float64)
        for i in range(0,len(points),block_points):out[i:i+block_points]=self.evaluate(points[i:i+block_points])[0]
        return out
    def save(self,path):
        m=dict(self.metadata,schema=SCHEMA,parent_sha256=self.parent_sha256,features=FEATURES,envelope='bounded-exponential-distances-D0.1',geometry=CHANNEL.as_dict(),head_architecture=[21,64,64,2],head_count=4,value_units='u/U',gradient_coordinates='physical',FE_fallback=False)
        arrays={'metadata_json':np.array(json.dumps(m,sort_keys=True)),'embedded_parent_npz':np.frombuffer(self.parent_bytes,dtype=np.uint8).copy()}
        for i,h in enumerate(self.heads):
            for j,(w,b) in enumerate(h):arrays[f'head{i}_W{j}']=w;arrays[f'head{i}_b{j}']=b
        np.savez_compressed(path,**arrays);return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            m=json.loads(str(a['metadata_json']))
            if m.get('schema')!=SCHEMA or m['features']!=FEATURES or m['geometry']!=json.loads(json.dumps(CHANNEL.as_dict())) or m['head_architecture']!=[21,64,64,2] or m['head_count']!=4 or m['FE_fallback'] or m['envelope']!='bounded-exponential-distances-D0.1' or m['value_units']!='u/U' or m['gradient_coordinates']!='physical':raise ValueError('unknown residual model contract')
            raw=a['embedded_parent_npz']
            if raw.dtype!=np.uint8 or raw.ndim!=1:raise ValueError('invalid embedded parent bytes')
            heads=[[(a[f'head{i}_W{j}'].copy(),a[f'head{i}_b{j}'].copy()) for j in range(3)] for i in range(4)]
            return cls(raw.tobytes(),m['parent_sha256'],heads,m)
