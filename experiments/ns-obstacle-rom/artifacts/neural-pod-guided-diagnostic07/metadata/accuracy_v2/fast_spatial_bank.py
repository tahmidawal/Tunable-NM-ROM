"""NumPy-only equivalent derivative contraction for a separately reviewed ROM build.

Training's JAX evaluator and the serialized model remain unchanged.
"""
import numpy as np
from accuracy_v2.spatial_bank import SpatialBank,inputs,envelope

def evaluate_layers_numpy(layers,x):
    h,dh=inputs(x,np)
    for i,(w,b) in enumerate(layers):
        h=h@w+b
        dh=np.einsum('nhd,hk->nkd',dh,w,optimize=True)
        if i<len(layers)-1:
            h=np.tanh(h);dh=dh*(1-h*h)[:,:,None]
    rank=h.shape[1]//2;v=h.reshape(len(x),2,rank)
    g=dh.reshape(len(x),2,rank,2).transpose(0,1,3,2)
    b,db=envelope(x,np)
    return b[:,None,None]*v,b[:,None,None,None]*g+db[:,None,:,None]*v[:,:,None,:]

class FastSpatialBank(SpatialBank):
    def evaluate(self,points,prefix=None):
        x=np.asarray(points,dtype=np.float64);r=self.rank if prefix is None else int(prefix)
        if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or not 1<=r<=self.rank:raise ValueError('invalid evaluation')
        v,g=evaluate_layers_numpy(self.layers,x)
        return v@self.transform[:,:r],g@self.transform[:,:r]
    @staticmethod
    def load(path):
        original=SpatialBank.load(path)
        return FastSpatialBank(original.layers,original.transform,original.metadata)
